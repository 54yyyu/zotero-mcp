"""Search-side of the item-keyed index (#826, #494, #495).

A hit's record key is resolved live against Zotero at search time: its owner
is its parent when it has one, else the item itself. Results collapse by
owner, and bibliographic filters are evaluated against the owner's current
Zotero data rather than metadata copied into Chroma. Every case here runs on
both record shapes: legacy (parent-keyed, metadata copied) and new
(attachment-keyed, own state only).
"""

import sqlite3
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

from conftest import DummyContext

from zotero_mcp import client as zclient
from zotero_mcp import semantic_search
from zotero_mcp.local_db import LocalZoteroReader
from zotero_mcp.semantic_filters import live_fields, matches, split_filters

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.delenv("ZOTERO_LOCAL", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_TYPE", raising=False)
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)
    zclient.clear_active_library()
    yield
    zclient.clear_active_library()


def _regular(key, title="A Paper", item_type="journalArticle", tags=(), doi="", deleted=False, **extra):
    data = {
        "key": key,
        "itemType": item_type,
        "title": title,
        "DOI": doi,
        "tags": [{"tag": t} for t in tags],
        "creators": [],
        **extra,
    }
    if deleted:
        data["deleted"] = 1
    return {"key": key, "data": data}


def _attachment(key, parent=None, title="Full Text PDF", filename="paper.pdf", deleted=False):
    data = {
        "key": key,
        "itemType": "attachment",
        "title": title,
        "filename": filename,
        "contentType": "application/pdf",
        "tags": [],
    }
    if parent:
        data["parentItem"] = parent
    if deleted:
        data["deleted"] = 1
    return {"key": key, "data": data}


class FakeZot:
    """pyzotero stand-in: ``items(itemKey=...)`` omits trashed items, as the
    API does; ``item()`` returns them with ``data.deleted`` set."""

    library_type = "users"
    library_id = "0"

    def __init__(self, items):
        self._items = {i["key"]: i for i in items}
        self.batch_calls: list[list[str]] = []
        self.single_calls: list[str] = []

    def items(self, itemKey="", start=0, limit=100, **kw):
        keys = itemKey.split(",")
        if start == 0:
            self.batch_calls.append(keys)
        found = [self._items[k] for k in keys if k in self._items and not self._items[k]["data"].get("deleted")]
        return found[start : start + limit]

    def item(self, key):
        self.single_calls.append(key)
        if key not in self._items:
            raise LookupError(f"404 {key}")
        return self._items[key]


def _hit(record_id, document="a passage about mindfulness", **meta):
    return (record_id, document, meta)


def _where_ok(where, meta):
    """Just enough of Chroma's where semantics for these tests."""
    if not where:
        return True
    for key, cond in where.items():
        if key == "$and":
            if not all(_where_ok(c, meta) for c in cond):
                return False
        elif isinstance(cond, dict) and "$in" in cond:
            if meta.get(key) not in cond["$in"]:
                return False
        elif meta.get(key) != cond:
            return False
    return True


class FakeChroma:
    """Returns the ranked hits that pass ``where``, ``n_results`` at most."""

    embedding_max_tokens = 8000

    def __init__(self, hits):
        self._hits = list(hits)
        self.calls: list[dict] = []

    def search(self, query_texts, n_results, where=None, where_document=None, **kw):
        self.calls.append({"n_results": n_results, "where": where})
        kept = [h for h in self._hits if _where_ok(where, h[2])][:n_results]
        return {
            "ids": [[h[0] for h in kept]],
            "documents": [[h[1] for h in kept]],
            "metadatas": [[h[2] for h in kept]],
            "distances": [[0.1 + 0.01 * i for i in range(len(kept))]],
        }

    def truncate_text(self, text, max_tokens=None):
        return text

    def read_error(self):
        return None


class FakeReader:
    """LocalZoteroReader stand-in for local mode."""

    def __init__(self, items, owners=()):
        self._items = {i["key"]: i for i in items}
        self._owners = list(owners)
        self.full_calls: list[tuple[list[str], object]] = []
        self.owner_calls: list[object] = []

    def get_full_items(self, keys, *, group_id=0, include_trashed=True, library_id=None):
        self.full_calls.append((list(keys), group_id))
        return {k: self._items[k] for k in keys if k in self._items}

    def get_filter_owners(self, *, group_id=0):
        self.owner_calls.append(group_id)
        return self._owners

    def close(self):
        pass


def _search(monkeypatch, tmp_path, hits, zot_items, *, chunking=True, local=False, reader=None):
    zot = FakeZot(zot_items)
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: zot)
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: local)
    if reader is not None:
        monkeypatch.setattr(semantic_search, "LocalZoteroReader", lambda **kw: reader)
    s = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChroma(hits), config_path=str(tmp_path / "config.json"))
    s._chunking_config = {"enabled": chunking}
    s._reranker_config = {"enabled": False}
    s._reranker = None
    return s, zot


def _keys(out):
    assert "error" not in out, out.get("error")
    return [r["item_key"] for r in out["results"]]


# ---------------------------------------------------------------------------
# Resolution and collapse
# ---------------------------------------------------------------------------


def test_two_attachments_of_one_parent_collapse_to_the_parent(monkeypatch, tmp_path):
    hits = [
        _hit("ATT1#2", "mindfulness relapse in the main paper", item_key="ATT1", chunk_index=2, n_chunks=5, page=4),
        _hit("ATT2#0", "supplement text", item_key="ATT2", chunk_index=0, n_chunks=1),
        _hit("ATT1#0", "intro", item_key="ATT1", chunk_index=0, n_chunks=5),
    ]
    zot = [
        _regular("PARENT01", "Main Paper"),
        _attachment("ATT1", "PARENT01", title="Main PDF", filename="main.pdf"),
        _attachment("ATT2", "PARENT01", title="Supplement", filename="supp.pdf"),
    ]
    s, _ = _search(monkeypatch, tmp_path, hits, zot)

    out = s.search("mindfulness relapse", limit=10)

    assert _keys(out) == ["PARENT01"]
    result = out["results"][0]
    assert result["zotero_item"]["key"] == "PARENT01"
    assert result["chunk_index"] == 2 and result["page"] == 4
    assert result["matched_attachment"] == {"key": "ATT1", "title": "Main PDF", "filename": "main.pdf"}


def test_standalone_attachment_hit_renders_as_the_attachment(monkeypatch, tmp_path):
    hits = [_hit("STAND1#0", item_key="STAND1", chunk_index=0, n_chunks=1)]
    s, _ = _search(monkeypatch, tmp_path, hits, [_attachment("STAND1", title="Loose Scan")])

    out = s.search("q", limit=5)

    assert _keys(out) == ["STAND1"]
    assert out["results"][0]["zotero_item"]["data"]["itemType"] == "attachment"
    # It is its own owner, so naming it again as the matched file says nothing.
    assert "matched_attachment" not in out["results"][0]


def test_reparented_attachment_resolves_to_its_new_parent(monkeypatch, tmp_path):
    # The index still holds ATT1 exactly as it was written; only Zotero changed.
    hits = [_hit("ATT1#0", item_key="ATT1", chunk_index=0, n_chunks=1)]
    zot = [_regular("OLDPAR01", "Old"), _regular("NEWPAR01", "New"), _attachment("ATT1", "NEWPAR01")]
    s, _ = _search(monkeypatch, tmp_path, hits, zot)

    out = s.search("q", limit=5)

    assert _keys(out) == ["NEWPAR01"]
    assert out["results"][0]["zotero_item"]["data"]["title"] == "New"


def test_owners_are_fetched_in_batches_not_one_by_one(monkeypatch, tmp_path):
    hits = [_hit(f"ATT{n}#0", item_key=f"ATT{n}") for n in range(3)]
    zot = [_regular(f"PAR{n}") for n in range(3)] + [_attachment(f"ATT{n}", f"PAR{n}") for n in range(3)]
    s, z = _search(monkeypatch, tmp_path, hits, zot)

    out = s.search("q", limit=5)

    assert _keys(out) == ["PAR0", "PAR1", "PAR2"]
    assert len(z.batch_calls) == 2  # hits, then their owners
    assert z.single_calls == []


@pytest.mark.parametrize("local", [False, True])
def test_trashed_owner_is_dropped_and_limit_still_filled(monkeypatch, tmp_path, local):
    hits = [
        _hit("ATTA#0", item_key="ATTA"),
        _hit("ATTX#0", item_key="ATTX"),  # trashed attachment of a live parent
        _hit("ATTB#0", item_key="ATTB"),
        _hit("ATTC#0", item_key="ATTC"),
    ]
    zot = [
        _regular("TRASHED1", deleted=True),
        _attachment("ATTA", "TRASHED1"),
        _regular("LIVEX001"),
        _attachment("ATTX", "LIVEX001", deleted=True),
        _regular("PARB0001"),
        _attachment("ATTB", "PARB0001"),
        _regular("PARC0001"),
        _attachment("ATTC", "PARC0001"),
    ]
    reader = FakeReader(zot) if local else None
    s, _ = _search(monkeypatch, tmp_path, hits, zot, local=local, reader=reader)

    out = s.search("q", limit=2)

    assert _keys(out) == ["PARB0001", "PARC0001"]


def test_article_level_overfetches_so_collapsing_still_fills_limit(monkeypatch, tmp_path):
    # Not chunked: a bibliographic record and its attachment's record are
    # both article-level hits of one owner.
    hits = [
        _hit("PAR1", item_key="PAR1"),
        _hit("ATT1", item_key="ATT1", file_stamp="1:2"),
        _hit("PAR2", item_key="PAR2"),
        _hit("PAR3", item_key="PAR3"),
    ]
    zot = [_regular("PAR1"), _attachment("ATT1", "PAR1"), _regular("PAR2"), _regular("PAR3")]
    s, _ = _search(monkeypatch, tmp_path, hits, zot, chunking=False)

    out = s.search("q", limit=3)

    assert _keys(out) == ["PAR1", "PAR2", "PAR3"]
    assert s.chroma_client.calls[0]["n_results"] > 3


def test_legacy_parent_keyed_passages_still_render_as_the_parent(monkeypatch, tmp_path):
    legacy_meta = {
        "item_key": "LEGACY01",
        "parent_item_key": "LEGACY01",
        "title": "Copied Title",
        "item_type": "journalArticle",
        "tags": "old",
        "chunk_index": 1,
        "n_chunks": 3,
    }
    hits = [_hit("LEGACY01#1", **legacy_meta), _hit("LEGACY01#0", **{**legacy_meta, "chunk_index": 0})]
    s, _ = _search(monkeypatch, tmp_path, hits, [_regular("LEGACY01", "Live Title")])

    out = s.search("q", limit=5)

    assert _keys(out) == ["LEGACY01"]
    result = out["results"][0]
    assert result["chunk_index"] == 1
    assert result["zotero_item"]["data"]["title"] == "Live Title"
    assert "matched_attachment" not in result


def test_local_mode_resolves_through_the_sqlite_reader(monkeypatch, tmp_path):
    hits = [_hit("ATT1#0", item_key="ATT1", group_id=0)]
    zot_items = [_regular("PARENT01"), _attachment("ATT1", "PARENT01")]
    reader = FakeReader(zot_items)
    s, z = _search(monkeypatch, tmp_path, hits, [], local=True, reader=reader)

    out = s.search("q", limit=5)

    assert _keys(out) == ["PARENT01"]
    assert [keys for keys, _gid in reader.full_calls] == [["ATT1"], ["PARENT01"]]
    assert z.batch_calls == [] and z.single_calls == []


# ---------------------------------------------------------------------------
# split_filters and the evaluator
# ---------------------------------------------------------------------------


def test_split_filters_routes_stored_fields_to_chroma_and_bibliographic_ones_live():
    chroma, live = split_filters({"$and": [{"section": "Methods"}, {"item_type": "book"}, {"group_id": 0}]})
    assert chroma == {"$and": [{"section": "Methods"}, {"group_id": 0}]}
    assert live == {"item_type": "book"}


@pytest.mark.parametrize("field", ["date", "creators", "url"])
def test_split_filters_rejects_fields_only_legacy_records_carry(field):
    with pytest.raises(ValueError, match=f"{field}.*Supported fields:.*item_type"):
        split_filters({"$and": [{"item_type": "book"}, {field: "x"}]})


def test_split_filters_single_side():
    assert split_filters({"tags": "x"}) == (None, {"tags": "x"})
    assert split_filters({"page": {"$gte": 3}}) == ({"page": {"$gte": 3}}, None)
    assert split_filters(None) == (None, None)


def test_split_filters_keeps_unknown_fields_on_the_chroma_side():
    assert split_filters({"file_stamp": "1:1"}) == ({"file_stamp": "1:1"}, None)


def test_split_filters_mixed_or_is_evaluated_live_whole():
    where = {"$or": [{"section": "Methods"}, {"tags": "x"}]}
    assert split_filters(where) == (None, where)
    pure_live = {"$or": [{"doi": "a"}, {"tags": "x"}]}
    assert split_filters(pure_live) == (None, pure_live)
    pure_chroma = {"$or": [{"section": "A"}, {"page": 2}]}
    assert split_filters(pure_chroma) == (pure_chroma, None)


def test_split_filters_multi_key_dict_is_an_implicit_and():
    chroma, live = split_filters({"section": "Methods", "doi": "10.1/x"})
    assert chroma == {"section": "Methods"} and live == {"doi": "10.1/x"}


def test_evaluator_operators():
    f = {"item_type": "book", "page": 4, "tags": frozenset({"a", "b"})}
    assert matches({"item_type": "book"}, f)
    assert matches({"item_type": {"$eq": "book"}}, f)
    assert not matches({"item_type": {"$ne": "book"}}, f)
    assert matches({"item_type": {"$in": ["book", "thesis"]}}, f)
    assert not matches({"item_type": {"$nin": ["book"]}}, f)
    assert matches({"page": {"$gt": 3}}, f) and matches({"page": {"$gte": 4}}, f)
    assert matches({"page": {"$lt": 5}}, f) and matches({"page": {"$lte": 4}}, f)
    assert not matches({"page": {"$gt": 4}}, f)
    assert matches({"$and": [{"page": 4}, {"item_type": "book"}]}, f)
    assert matches({"$or": [{"page": 9}, {"item_type": "book"}]}, f)
    assert not matches({"$or": [{"page": 9}, {"item_type": "x"}]}, f)
    # A field the hit does not carry matches nothing, as in Chroma.
    assert not matches({"section": {"$ne": "Methods"}}, f)


def test_evaluator_rejects_several_operators_in_one_condition():
    with pytest.raises(ValueError, match="one operator"):
        matches({"page": {"$gte": 3, "$lte": 5}}, {"page": 4})
    with pytest.raises(ValueError, match="one operator"):
        split_filters({"tags": {"$in": ["a"], "$nin": ["b"]}})


def test_evaluator_rejects_a_non_list_operand_to_in_and_nin():
    with pytest.raises(ValueError, match=r"\$in takes a list"):
        matches({"item_type": {"$in": "book"}}, {"item_type": "book"})
    with pytest.raises(ValueError, match=r"\$nin takes a list"):
        split_filters({"tags": {"$nin": "x"}})


def test_a_bad_live_filter_is_an_error_even_without_hits(monkeypatch, tmp_path):
    s, _ = _search(monkeypatch, tmp_path, [], [])

    out = s.search("q", limit=5, filters={"doi": {"$in": "10.1/x"}})

    assert "takes a list" in out["error"]


def test_evaluator_tags_are_membership():
    f = {"tags": frozenset({"thesis", "ai"})}
    assert matches({"tags": "thesis"}, f)
    assert not matches({"tags": "the"}, f)
    assert matches({"tags": {"$in": ["x", "ai"]}}, f)
    assert not matches({"tags": {"$nin": ["x", "ai"]}}, f)
    assert matches({"tags": {"$ne": "x"}}, f)


def test_live_fields_reads_citation_key_from_field_or_extra():
    item = _regular("K1", extra="tex.foo: 1\nCitation Key: smith2020")
    assert live_fields(item)["citation_key"] == "smith2020"
    item = _regular("K1", citationKey="doe2021")
    assert live_fields(item)["citation_key"] == "doe2021"


# ---------------------------------------------------------------------------
# Filters at search time
# ---------------------------------------------------------------------------


def _filter_corpus():
    hits = [
        _hit("ATTART#0", item_key="ATTART", chunk_index=0, n_chunks=1, section="Methods"),
        _hit("ATTBOOK#0", item_key="ATTBOOK", chunk_index=0, n_chunks=1, section="Methods"),
        # Legacy record: copied metadata says it is a book tagged "old".
        _hit(
            "LEGACY01#0",
            item_key="LEGACY01",
            title="Copied",
            item_type="book",
            tags="old",
            doi="10.1/stale",
            chunk_index=0,
            n_chunks=1,
        ),
    ]
    zot = [
        _regular("ARTICLE1", item_type="journalArticle", tags=["new"], doi="10.1/art"),
        _attachment("ATTART", "ARTICLE1"),
        _regular("BOOK0001", item_type="book", tags=["keep"], doi="10.1/book"),
        _attachment("ATTBOOK", "BOOK0001"),
        _regular("LEGACY01", item_type="journalArticle", tags=["new"], doi="10.1/legacy"),
    ]
    return hits, zot


def test_item_type_filter_matches_new_shape_passages_via_live_data(monkeypatch, tmp_path):
    hits, zot = _filter_corpus()
    s, _ = _search(monkeypatch, tmp_path, hits, zot)

    out = s.search("q", limit=5, filters={"item_type": "book"})

    # The legacy record's copied item_type ("book") is not trusted.
    assert _keys(out) == ["BOOK0001"]
    assert s.chroma_client.calls[0]["where"] is None


def test_tag_filter_reflects_a_changed_tag_without_reindexing(monkeypatch, tmp_path):
    hits, zot = _filter_corpus()
    s, _ = _search(monkeypatch, tmp_path, hits, zot)

    assert _keys(s.search("q", limit=5, filters={"tags": "new"})) == ["ARTICLE1", "LEGACY01"]
    assert _keys(s.search("q", limit=5, filters={"tags": "old"})) == []


def test_doi_filter_uses_live_data(monkeypatch, tmp_path):
    hits, zot = _filter_corpus()
    s, _ = _search(monkeypatch, tmp_path, hits, zot)

    assert _keys(s.search("q", limit=5, filters={"doi": "10.1/legacy"})) == ["LEGACY01"]
    assert _keys(s.search("q", limit=5, filters={"doi": "10.1/stale"})) == []


def test_section_filter_still_goes_to_chroma(monkeypatch, tmp_path):
    hits, zot = _filter_corpus()
    s, _ = _search(monkeypatch, tmp_path, hits, zot)

    out = s.search(
        "q", limit=5, filters={"$and": [{"section": "Methods"}, {"item_type": "journalArticle"}]}, group_id=0
    )

    assert s.chroma_client.calls[0]["where"] == {"$and": [{"section": "Methods"}, {"group_id": 0}]}
    assert out["filters"] == {"$and": [{"section": "Methods"}, {"item_type": "journalArticle"}]}


def test_mixed_or_takes_chunk_fields_from_the_hit(monkeypatch, tmp_path):
    hits, zot = _filter_corpus()
    s, _ = _search(monkeypatch, tmp_path, hits, zot)

    out = s.search("q", limit=5, filters={"$or": [{"section": "Methods"}, {"doi": "10.1/legacy"}]})

    assert _keys(out) == ["ARTICLE1", "BOOK0001", "LEGACY01"]
    assert s.chroma_client.calls[0]["where"] is None


def _owner(key, tags=(), item_type="journalArticle", attachments=()):
    item = _regular(key, item_type=item_type, tags=tags)
    item["attachment_keys"] = list(attachments)
    return item


def test_local_prefilter_passes_item_keys_to_chroma_and_keeps_limit_exact(monkeypatch, tmp_path):
    hits = [_hit(f"ATT{n}#0", item_key=f"ATT{n}", group_id=0, chunk_index=0, n_chunks=1) for n in range(6)]
    zot = []
    owners = []
    for n in range(6):
        tags = ["keep"] if n % 2 else []
        zot += [_regular(f"PAR{n}", tags=tags), _attachment(f"ATT{n}", f"PAR{n}")]
        owners.append(_owner(f"PAR{n}", tags=tags, attachments=[f"ATT{n}"]))
    reader = FakeReader(zot, owners)
    s, _ = _search(monkeypatch, tmp_path, hits, [], local=True, reader=reader)

    out = s.search("q", limit=2, filters={"tags": "keep"}, group_id=0)

    where = s.chroma_client.calls[0]["where"]
    assert where["$and"][0] == {"group_id": 0}
    assert sorted(where["$and"][1]["item_key"]["$in"]) == ["ATT1", "ATT3", "ATT5", "PAR1", "PAR3", "PAR5"]
    assert _keys(out) == ["PAR1", "PAR3"]
    assert len(s.chroma_client.calls) == 1
    assert reader.owner_calls == [0]


def test_local_prefilter_matches_legacy_parent_keyed_records(monkeypatch, tmp_path):
    hits = [_hit("LEGACY01#0", item_key="LEGACY01", title="Copied", tags="", chunk_index=0, n_chunks=1)]
    zot = [_regular("LEGACY01", tags=["keep"])]
    reader = FakeReader(zot, [_owner("LEGACY01", tags=["keep"])])
    s, _ = _search(monkeypatch, tmp_path, hits, [], local=True, reader=reader)

    assert _keys(s.search("q", limit=2, filters={"tags": "keep"})) == ["LEGACY01"]


def test_local_prefilter_above_threshold_falls_back_to_post_filtering(monkeypatch, tmp_path):
    hits = [_hit(f"ATT{n}#0", item_key=f"ATT{n}") for n in range(4)]
    zot, owners = [], []
    for n in range(4):
        zot += [_regular(f"PAR{n}", tags=["keep"]), _attachment(f"ATT{n}", f"PAR{n}")]
        owners.append(_owner(f"PAR{n}", tags=["keep"], attachments=[f"ATT{n}"]))
    reader = FakeReader(zot, owners)
    s, _ = _search(monkeypatch, tmp_path, hits, [], local=True, reader=reader)
    monkeypatch.setattr(semantic_search, "PREFILTER_MAX_KEYS", 3)

    out = s.search("q", limit=2, filters={"tags": "keep"})

    assert s.chroma_client.calls[0]["where"] is None
    assert _keys(out) == ["PAR0", "PAR1"]


def test_has_fulltext_is_not_prefiltered(monkeypatch, tmp_path):
    hits = [_hit("ATT1#0", item_key="ATT1")]
    reader = FakeReader([_regular("PAR1"), _attachment("ATT1", "PAR1")], [_owner("PAR1", attachments=["ATT1"])])
    s, _ = _search(monkeypatch, tmp_path, hits, [], local=True, reader=reader)

    out = s.search("q", limit=2, filters={"has_fulltext": True})

    assert reader.owner_calls == []
    assert _keys(out) == ["PAR1"]


def test_has_fulltext_on_records():
    # Defined from the hit: an attachment record is full text; a legacy
    # record says so in its own metadata; a bibliographic record does not.
    s = semantic_search.ZoteroSemanticSearch.__new__(semantic_search.ZoteroSemanticSearch)
    assert s._hit_has_fulltext({}, _attachment("A1", "P1"))
    assert s._hit_has_fulltext({"title": "x", "has_fulltext": True}, _regular("P1"))
    assert not s._hit_has_fulltext({"title": "x", "has_fulltext": "failed"}, _regular("P1"))
    assert not s._hit_has_fulltext({}, _regular("P1"))


def test_web_post_filter_requeries_up_to_three_rounds(monkeypatch, tmp_path):
    # 100 non-matching hits before the one matching owner: beyond the first
    # two rounds (8, then 32 candidates), within the third (128).
    hits = [_hit(f"ATT{n:03d}#0", item_key=f"ATT{n:03d}") for n in range(100)]
    hits.append(_hit("ATTKEEP#0", item_key="ATTKEEP"))
    zot = [_regular(f"PAR{n:03d}") for n in range(100)] + [
        _attachment(f"ATT{n:03d}", f"PAR{n:03d}") for n in range(100)
    ]
    zot += [_regular("PARKEEP1", tags=["keep"]), _attachment("ATTKEEP", "PARKEEP1")]
    s, _ = _search(monkeypatch, tmp_path, hits, zot)

    out = s.search("q", limit=2, filters={"tags": "keep"})

    sizes = [c["n_results"] for c in s.chroma_client.calls]
    assert len(sizes) == 3
    assert sizes[1] == sizes[0] * 4 and sizes[2] == sizes[1] * 4
    assert _keys(out) == ["PARKEEP1"]
    assert "1" in out["filter_note"]


def test_unfiltered_search_asks_again_when_one_owner_crowds_out_the_rest(monkeypatch, tmp_path):
    """Twelve passages of one paper fill the first fetch; the second paper comes in the next."""
    hits = [_hit(f"ATTA0001#{n}", item_key="ATTA0001") for n in range(12)]
    hits += [_hit("ATTB0001#0", item_key="ATTB0001")]
    zot = [_regular("PARA0001"), _attachment("ATTA0001", "PARA0001"), _regular("PARB0001"), _attachment("ATTB0001", "PARB0001")]
    s, _ = _search(monkeypatch, tmp_path, hits, zot)

    out = s.search("q", limit=2)

    assert _keys(out) == ["PARA0001", "PARB0001"]
    sizes = [c["n_results"] for c in s.chroma_client.calls]
    assert sizes == [8, 32]
    assert out["filter_note"] is None, "nothing was filtered"


def test_unfiltered_search_stops_when_the_index_is_exhausted(monkeypatch, tmp_path):
    hits = [_hit("ATTA0001#0", item_key="ATTA0001")]
    zot = [_regular("PARA0001"), _attachment("ATTA0001", "PARA0001")]
    s, _ = _search(monkeypatch, tmp_path, hits, zot)

    out = s.search("q", limit=2)

    assert _keys(out) == ["PARA0001"]
    assert len(s.chroma_client.calls) == 1


class _RecordingReranker:
    """Plain rerank() contract; records how many documents each call got."""

    def __init__(self):
        self.sizes: list[int] = []

    def rerank(self, query, documents, top_k=10):
        self.sizes.append(len(documents))
        return list(range(len(documents)))[:top_k]


def test_post_filter_reranks_only_bounded_survivors(monkeypatch, tmp_path):
    # 100 non-matching passages, then 60 matching passages of ONE owner: the
    # search stays short of limit=2 owners and runs all three rounds (8, 32,
    # 128 candidates); the last round has 28 survivors.
    hits = [_hit(f"ATT{n:03d}#0", item_key=f"ATT{n:03d}") for n in range(100)]
    hits += [_hit(f"ATTKEEP#{n}", item_key="ATTKEEP", chunk_index=n, n_chunks=60) for n in range(60)]
    zot = [_regular(f"PAR{n:03d}") for n in range(100)] + [
        _attachment(f"ATT{n:03d}", f"PAR{n:03d}") for n in range(100)
    ]
    zot += [_regular("PARKEEP1", tags=["keep"]), _attachment("ATTKEEP", "PARKEEP1")]
    s, _ = _search(monkeypatch, tmp_path, hits, zot)
    s._reranker_config = {"enabled": True, "candidate_multiplier": 3}
    s._reranker = reranker = _RecordingReranker()

    out = s.search("q", limit=2, filters={"tags": "keep"})

    assert len(s.chroma_client.calls) == 3
    bound = 2 * max(4, 3)
    # Rounds 1 and 2 had no survivors, so nothing was sent to re-rank there.
    assert reranker.sizes == [bound]
    assert _keys(out) == ["PARKEEP1"]
    assert out["rerank"]["candidates"] == bound


def test_unfiltered_search_still_reranks_every_candidate(monkeypatch, tmp_path):
    hits, zot = _filter_corpus()
    s, _ = _search(monkeypatch, tmp_path, hits, zot)
    s._reranker_config = {"enabled": True, "candidate_multiplier": 3}
    s._reranker = reranker = _RecordingReranker()

    s.search("q", limit=1)

    assert reranker.sizes == [3]


def test_no_test_can_find_a_real_zotero_database():
    # conftest points HOME and Path.home at a temporary directory and clears
    # ZOTERO_LOCAL / ZOTERO_DB_PATH, so auto-detection finds nothing.
    with pytest.raises(FileNotFoundError):
        LocalZoteroReader(db_path=None)._find_zotero_db()


def test_web_post_filter_stops_once_limit_is_met(monkeypatch, tmp_path):
    hits, zot = _filter_corpus()
    s, _ = _search(monkeypatch, tmp_path, hits, zot)

    out = s.search("q", limit=1, filters={"tags": "new"})

    assert len(s.chroma_client.calls) == 1
    assert _keys(out) == ["ARTICLE1"]
    assert not out.get("filter_note")


# ---------------------------------------------------------------------------
# Rendering and the CLI's key re-parse
# ---------------------------------------------------------------------------


class _CannedSearch:
    def __init__(self, results):
        self._results = results

    def search(self, query, limit=10, filters=None, group_id=None, **kw):
        return {
            "results": self._results,
            "total_found": len(self._results),
            "filter_note": "Only 1 item matched the filters.",
        }


def test_location_names_the_matched_attachment_and_cli_reads_owner_keys(monkeypatch, tmp_path):
    from zotero_mcp import cli_standalone
    from zotero_mcp.tools import search as search_module

    result = {
        "item_key": "PARENT01",
        "similarity_score": 0.9,
        "matched_passage": "quantum",
        "page": 3,
        "matched_attachment": {"key": "ATTKEY01", "title": "Main PDF", "filename": "main.pdf"},
        "zotero_item": _regular("PARENT01", "Owner Paper"),
    }
    config_dir = tmp_path / ".config" / "zotero-mcp"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text("{}")
    monkeypatch.setattr(semantic_search, "create_semantic_search", lambda *a, **kw: _CannedSearch([result]))
    monkeypatch.setattr(search_module, "_maybe_fire_presearch_sync", lambda _s: None)
    monkeypatch.setattr(search_module, "_library_version_check", lambda *a: None)
    monkeypatch.setattr(zclient, "get_active_group_id", lambda: 0)

    text = search_module.semantic_search(query="quantum", ctx=DummyContext())

    assert "Attachment: Main PDF, p. 3" in text
    assert "Only 1 item matched the filters." in text
    assert cli_standalone._keys_from_markdown(text) == ["PARENT01"]


def test_location_falls_back_to_the_attachment_filename(monkeypatch, tmp_path):
    from zotero_mcp.tools import search as search_module

    result = {
        "item_key": "PARENT01",
        "similarity_score": 0.9,
        "matched_passage": "q",
        "matched_attachment": {"key": "ATTKEY01", "title": "", "filename": "supp.pdf"},
        "zotero_item": _regular("PARENT01"),
    }
    config_dir = tmp_path / ".config" / "zotero-mcp"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text("{}")
    monkeypatch.setattr(semantic_search, "create_semantic_search", lambda *a, **kw: _CannedSearch([result]))
    monkeypatch.setattr(search_module, "_maybe_fire_presearch_sync", lambda _s: None)
    monkeypatch.setattr(search_module, "_library_version_check", lambda *a: None)
    monkeypatch.setattr(zclient, "get_active_group_id", lambda: 0)

    assert "Attachment: supp.pdf" in search_module.semantic_search(query="q", ctx=DummyContext())


# ---------------------------------------------------------------------------
# LocalZoteroReader.get_filter_owners over a real zotero.sqlite
# ---------------------------------------------------------------------------


def _owner_db(path):
    import _search_corpus

    conn = sqlite3.connect(path)
    conn.executescript(_search_corpus.SCHEMA)
    conn.executescript(
        "CREATE TABLE itemAttachments (itemID INTEGER PRIMARY KEY, parentItemID INT, linkMode INT,"
        " contentType TEXT, path TEXT);"
        "CREATE TABLE itemAnnotations (itemID INTEGER PRIMARY KEY, parentItemID INT, type INT, text TEXT,"
        " comment TEXT, color TEXT, pageLabel TEXT, sortIndex TEXT, position TEXT);"
    )
    conn.execute("INSERT INTO libraries VALUES (1, 'user', 1, 1)")
    conn.execute("INSERT INTO libraries VALUES (5, 'group', 1, 1)")
    conn.execute("INSERT INTO groups VALUES (777, 5, 'G', '', 1)")
    for type_id, name in [(1, "journalArticle"), (2, "attachment"), (3, "note")]:
        conn.execute("INSERT INTO itemTypes VALUES (?, ?)", (type_id, name))
    for field_id, name in [(1, "title"), (26, "DOI"), (27, "publicationTitle"), (28, "extra")]:
        conn.execute("INSERT INTO fields VALUES (?, ?)", (field_id, name))
    rows = [
        # itemID, key, type, library
        (1, "PAPER001", 1, 1),
        (2, "ATTCHLD1", 2, 1),  # child attachment of PAPER001
        (3, "STANDATT", 2, 1),  # standalone attachment: an owner
        (4, "NOTE0001", 3, 1),  # notes are never owners
        (5, "TRASHED1", 1, 1),
        (6, "ATTTRSH1", 2, 1),  # trashed child of PAPER001
        (7, "GRPPAPR1", 1, 5),
    ]
    for item_id, key, type_id, lib in rows:
        conn.execute("INSERT INTO items VALUES (?, ?, ?, ?, '2024-01-01', '2024-01-01')", (item_id, key, type_id, lib))
    conn.execute("INSERT INTO itemAttachments VALUES (2, 1, 0, 'application/pdf', 'storage:a.pdf')")
    conn.execute("INSERT INTO itemAttachments VALUES (3, NULL, 0, 'application/pdf', 'storage:s.pdf')")
    conn.execute("INSERT INTO itemAttachments VALUES (6, 1, 0, 'application/pdf', 'storage:t.pdf')")
    conn.execute("INSERT INTO itemNotes VALUES (4, NULL, 'n')")
    conn.execute("INSERT INTO deletedItems VALUES (5)")
    conn.execute("INSERT INTO deletedItems VALUES (6)")
    values = [(1, 1, "Paper One"), (1, 26, "10.1/one"), (1, 28, "Citation Key: one2020"), (3, 1, "Scan")]
    for n, (item_id, field_id, value) in enumerate(values, start=1):
        conn.execute("INSERT INTO itemDataValues VALUES (?, ?)", (n, value))
        conn.execute("INSERT INTO itemData VALUES (?, ?, ?)", (item_id, field_id, n))
    conn.execute("INSERT INTO tags VALUES (1, 'keep')")
    conn.execute("INSERT INTO itemTags VALUES (1, 1, 0)")
    conn.commit()
    conn.close()


def test_get_filter_owners_lists_live_owners_with_their_attachments(tmp_path):
    db = tmp_path / "zotero.sqlite"
    _owner_db(db)
    reader = LocalZoteroReader(db_path=str(db))
    try:
        owners = {o["key"]: o for o in reader.get_filter_owners(group_id=0)}
    finally:
        reader.close()

    assert set(owners) == {"PAPER001", "STANDATT"}
    paper = owners["PAPER001"]
    assert paper["attachment_keys"] == ["ATTCHLD1"]
    fields = live_fields(paper)
    assert fields["title"] == "Paper One" and fields["doi"] == "10.1/one"
    assert fields["citation_key"] == "one2020"
    assert fields["tags"] == frozenset({"keep"})
    assert fields["item_type"] == "journalArticle"
    assert live_fields(owners["STANDATT"])["item_type"] == "attachment"


def test_get_filter_owners_across_libraries(tmp_path):
    db = tmp_path / "zotero.sqlite"
    _owner_db(db)
    reader = LocalZoteroReader(db_path=str(db))
    try:
        keys = {o["key"] for o in reader.get_filter_owners(group_id=None)}
    finally:
        reader.close()
    assert keys == {"PAPER001", "STANDATT", "GRPPAPR1"}
