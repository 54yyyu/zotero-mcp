"""Web mode pre-filters a filtered search through the Zotero API (#826).

Without zotero.sqlite a filter on live fields used to be applied after the
query, over up to three rounds of ever more candidates, each costing owner
lookups. Now a filter the web API can express asks it, once per library, for
the keys of the matching owners (``format=keys``), and Chroma only returns
records of those owners or of attachments whose ``parent_hint`` names one.
Hits are still resolved and checked live, so a stale hint cannot return a
wrong result.

Every Zotero and Chroma object here is a fake; HOME and Path.home point into
tmp_path, nothing touches the network or the user's real configuration.
"""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

from test_semantic_owner_resolution import (  # noqa: E402
    FakeChroma,
    FakeReader,
    FakeZot,
    _attachment,
    _hit,
    _isolated,  # noqa: F401  (autouse fixture)
    _keys,
    _owner,
    _regular,
)

from zotero_mcp import semantic_search  # noqa: E402
from zotero_mcp.chroma_client import read_index_layout  # noqa: E402
from zotero_mcp.semantic_filters import web_query_plan  # noqa: E402


def _matches(where, meta):
    """Just enough of Chroma's where semantics: equality, ``$in``, ``$and`` and ``$or``."""
    if not where:
        return True
    for key, cond in where.items():
        if key == "$and":
            ok = all(_matches(c, meta) for c in cond)
        elif key == "$or":
            ok = any(_matches(c, meta) for c in cond)
        elif isinstance(cond, dict) and "$in" in cond:
            ok = meta.get(key) in cond["$in"]
        else:
            ok = meta.get(key) == cond
        if not ok:
            return False
    return True


class HintChroma(FakeChroma):
    """Answers ``where`` with ``$or``, and says what the index layout is."""

    def __init__(self, hits, layout=None):
        super().__init__(hits)
        self.layout = {
            **read_index_layout("/nonexistent", "none"),
            "article_records": 1,
            "passage_records": 1,
            "attachments_without_parent_hint": 0,
            "passages_without_parent_hint": 0,
            "attachments_without_library": 0,
            "library_ids": [0],
            **(layout or {}),
        }

    def index_layout(self):
        return self.layout

    def separate_passages(self):
        return False

    def search(self, query_texts, n_results, where=None, where_document=None, **kw):
        self.calls.append({"n_results": n_results, "where": where})
        kept = [h for h in self._hits if _matches(where, h[2])][:n_results]
        return {
            "ids": [[h[0] for h in kept]],
            "documents": [[h[1] for h in kept]],
            "metadatas": [[h[2] for h in kept]],
            "distances": [[0.1 + 0.01 * i for i in range(len(kept))]],
        }


class ApiZot(FakeZot):
    """pyzotero stand-in that also answers ``top(format="keys", itemType=..., tag=...)``."""

    def __init__(self, items, library_id="0", fail=False, next_link=False):
        super().__init__(items)
        self.library_type = "users" if library_id == "0" else "groups"
        self.library_id = library_id
        self.fail = fail
        self.links = {"next": "https://example.invalid/next"} if next_link else {}
        self.top_calls: list[dict] = []

    def top(self, **kw):
        self.top_calls.append(kw)
        if self.fail:
            raise RuntimeError("503 Service Unavailable")
        assert kw["format"] == "keys"
        types = kw["itemType"].split(" || ") if kw.get("itemType") else None
        tags = kw.get("tag") or []
        tags = [tags] if isinstance(tags, str) else tags
        out = []
        for item in self._items.values():
            data = item["data"]
            if data.get("parentItem") or data.get("deleted") or data["itemType"] in ("note", "annotation"):
                continue
            if types and data["itemType"] not in types:
                continue
            have = {t["tag"] for t in data.get("tags") or []}
            if all(have & set(clause.split(" || ")) for clause in tags):
                out.append(item["key"])
        limit = kw.get("limit", -1)
        if limit not in (-1, None) and len(out) > limit:
            # As the API does: the first page, and a link to the next.
            out = out[:limit]
            self.links = {"next": "https://example.invalid/next"}
        return ("\n".join(out)).encode()


def _search(monkeypatch, tmp_path, hits, zot_items, *, layout=None, **zot_kw):
    zot = ApiZot(zot_items, **zot_kw)
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: zot)
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    chroma = HintChroma(hits, layout)
    s = semantic_search.ZoteroSemanticSearch(chroma_client=chroma, config_path=str(tmp_path / "config.json"))
    s._chunking_config = {"enabled": True}
    s._reranker_config = {"enabled": False}
    s._reranker = None
    return s, zot


def _hint_hit(att, parent, **meta):
    return _hit(f"{att}#0", item_key=att, parent_hint=parent, group_id=0, chunk_index=0, n_chunks=1, **meta)


def _corpus(articles=6, books=3):
    """Articles first in rank order, then books; every item has one attachment."""
    hits, items = [], []
    for n in range(articles):
        hits.append(_hint_hit(f"ATTART{n}", f"ART{n:05d}"))
        items += [_regular(f"ART{n:05d}", item_type="journalArticle"), _attachment(f"ATTART{n}", f"ART{n:05d}")]
    for n in range(books):
        hits.append(_hint_hit(f"ATTBOK{n}", f"BOOK{n:04d}"))
        items += [_regular(f"BOOK{n:04d}", item_type="book"), _attachment(f"ATTBOK{n}", f"BOOK{n:04d}")]
    return hits, items


# --- the filter plan --------------------------------------------------------------


@pytest.mark.parametrize(
    ("where", "plan"),
    [
        ({"item_type": "book"}, {"itemType": "book", "tag": [], "item_keys": None}),
        ({"item_type": {"$eq": "book"}}, {"itemType": "book", "tag": [], "item_keys": None}),
        (
            {"item_type": {"$in": ["thesis", "book"]}},
            {"itemType": "book || thesis", "tag": [], "item_keys": None},
        ),
        ({"tags": "keep"}, {"itemType": None, "tag": ["keep"], "item_keys": None}),
        ({"tags": {"$in": ["a", "b"]}}, {"itemType": None, "tag": ["a || b"], "item_keys": None}),
        (
            {"$and": [{"tags": "a"}, {"tags": "b"}, {"item_type": "book"}]},
            {"itemType": "book", "tag": ["a", "b"], "item_keys": None},
        ),
        (
            {"$and": [{"item_type": "book"}, {"tags": "a"}]},
            {"itemType": "book", "tag": ["a"], "item_keys": None},
        ),
        ({"item_key": "KEY00001"}, {"itemType": None, "tag": [], "item_keys": {"KEY00001"}}),
        ({"tags": "-x"}, {"itemType": None, "tag": ["\\-x"], "item_keys": None}),
    ],
)
def test_the_plan_of_a_filter_the_api_can_express(where, plan):
    got = web_query_plan(where)

    assert {k: got[k] for k in plan} == plan and got["empty"] is False


@pytest.mark.parametrize(
    "where",
    [
        {"$or": [{"item_type": "book"}, {"tags": "a"}]},
        {"item_type": {"$ne": "book"}},
        {"item_type": {"$nin": ["book"]}},
        {"tags": {"$ne": "a"}},
        {"doi": "10.1/x"},
        {"title": "A"},
        {"publication": "J"},
        {"citation_key": "k"},
        {"has_fulltext": True},
        {"$and": [{"item_type": "book"}, {"doi": "10.1/x"}]},
        # A dict of several keys is an error in Chroma, so it is never planned.
        {"item_type": "book", "doi": "10.1/x"},
        {"item_type": "book", "tags": "a"},
        # The tags are one space-joined string: a value with a space, or an empty one, cannot be sent.
        {"tags": "two words"},
        {"tags": ""},
        {"tags": {"$in": ["a", "b c"]}},
        {"$and": [{"item_type": "book"}, {"tags": "x y"}]},
        {"item_type": "a || b"},
        {"item_type": {"$in": []}},
        {"item_type": {"$gt": "a"}},
        None,
    ],
)
def test_a_filter_the_api_cannot_express_has_no_plan(where):
    assert web_query_plan(where) is None


def test_clauses_with_nothing_in_common_match_nothing():
    plan = web_query_plan({"$and": [{"item_type": "book"}, {"item_type": "thesis"}]})

    assert plan["empty"] is True


# --- the search -------------------------------------------------------------------


def test_one_keys_request_per_library_one_query_and_exact_results(monkeypatch, tmp_path):
    hits, items = _corpus(articles=6, books=3)
    s, zot = _search(monkeypatch, tmp_path, hits, items)

    out = s.search("q", limit=2, filters={"item_type": "book"})

    assert _keys(out) == ["BOOK0000", "BOOK0001"]
    assert zot.top_calls == [{"format": "keys", "limit": semantic_search.PREFILTER_MAX_KEYS + 1, "itemType": "book"}]
    assert len(s.chroma_client.calls) == 1, "the first query already holds only matches"
    where = s.chroma_client.calls[0]["where"]
    assert where == {
        "$or": [
            {"item_key": {"$in": ["BOOK0000", "BOOK0001", "BOOK0002"]}},
            {"parent_hint": {"$in": ["BOOK0000", "BOOK0001", "BOOK0002"]}},
        ]
    }
    # The lookups are the batched ones: the hits, then their parents; one request each.
    assert len(zot.batch_calls) == 2 and zot.single_calls == []


def test_the_pre_filter_combines_with_the_library_and_the_chroma_filters(monkeypatch, tmp_path):
    hits, items = _corpus(articles=1, books=2)
    s, _ = _search(monkeypatch, tmp_path, hits, items)

    s.search("q", limit=2, filters={"$and": [{"item_type": "book"}, {"section": "Methods"}]}, group_id=0)

    where = s.chroma_client.calls[0]["where"]
    assert where["$and"][0] == {"section": "Methods"}
    assert where["$and"][1] == {"group_id": 0}
    assert set(where["$and"][2]) == {"$or"}


def test_each_library_in_the_index_gets_one_request(monkeypatch, tmp_path):
    hits = [_hint_hit("ATTPERS1", "BOOKPERS"), _hint_hit("ATTGRP01", "BOOKGRP1")]
    hits[1][2]["group_id"] = 777
    personal = [_regular("BOOKPERS", item_type="book"), _attachment("ATTPERS1", "BOOKPERS")]
    group = [_regular("BOOKGRP1", item_type="book"), _attachment("ATTGRP01", "BOOKGRP1")]
    s, zot = _search(monkeypatch, tmp_path, hits, personal, layout={"library_ids": [0, 777]})
    group_zot = ApiZot(group, library_id="777")
    s._scoped_clients[777] = group_zot

    out = s.search("q", limit=5, filters={"item_type": "book"})

    assert sorted(_keys(out)) == ["BOOKGRP1", "BOOKPERS"]
    assert len(zot.top_calls) == 1 and len(group_zot.top_calls) == 1
    assert len(s.chroma_client.calls) == 1


def test_a_group_search_asks_only_that_library(monkeypatch, tmp_path):
    hits = [_hint_hit("ATTGRP01", "BOOKGRP1")]
    hits[0][2]["group_id"] = 777
    s, zot = _search(monkeypatch, tmp_path, hits, [], layout={"library_ids": [0, 777]})
    group_zot = ApiZot([_regular("BOOKGRP1", item_type="book"), _attachment("ATTGRP01", "BOOKGRP1")], library_id="777")
    s._scoped_clients[777] = group_zot

    out = s.search("q", limit=5, filters={"item_type": "book"}, group_id=777)

    assert _keys(out) == ["BOOKGRP1"]
    assert zot.top_calls == [] and len(group_zot.top_calls) == 1


def test_a_stale_hint_is_dropped_by_the_live_check(monkeypatch, tmp_path):
    """ATTMOVED was a book's attachment when indexed and is an article's now: its hint still names the book."""
    hits = [
        _hint_hit("ATTMOVED", "BOOK0000"),
        _hint_hit("ATTBOK1", "BOOK0001"),
    ]
    items = [
        _regular("BOOK0000", item_type="book"),
        _regular("MOVEDTO1", item_type="journalArticle"),
        _attachment("ATTMOVED", "MOVEDTO1"),
        _regular("BOOK0001", item_type="book"),
        _attachment("ATTBOK1", "BOOK0001"),
    ]
    s, _ = _search(monkeypatch, tmp_path, hits, items)

    out = s.search("q", limit=5, filters={"item_type": "book"})

    # The stale hit passes Chroma's pre-filter, is resolved to its real owner and fails the live check.
    assert _keys(out) == ["BOOK0001"]
    assert len(s.chroma_client.calls) == 1


def test_a_file_moved_to_a_matching_parent_is_missed_until_it_is_rewritten(monkeypatch, tmp_path):
    """The other side of a stale hint: a miss, never a wrong result."""
    hits = [_hint_hit("ATTMOVED", "ARTICLE1")]
    items = [
        _regular("ARTICLE1", item_type="journalArticle"),
        _regular("BOOK0000", item_type="book"),
        _attachment("ATTMOVED", "BOOK0000"),
    ]
    s, _ = _search(monkeypatch, tmp_path, hits, items)

    out = s.search("q", limit=5, filters={"item_type": "book"})

    assert _keys(out) == []


def test_a_legacy_record_is_matched_by_its_item_key(monkeypatch, tmp_path):
    hits = [_hit("BOOK0000#0", item_key="BOOK0000", title="Copied", chunk_index=0, n_chunks=1)]
    s, _ = _search(monkeypatch, tmp_path, hits, [_regular("BOOK0000", item_type="book")])

    assert _keys(s.search("q", limit=2, filters={"item_type": "book"})) == ["BOOK0000"]


def test_tag_clauses_are_sent_as_repeated_parameters(monkeypatch, tmp_path):
    hits, items = _corpus(articles=0, books=2)
    s, zot = _search(monkeypatch, tmp_path, hits, items)

    s.search("q", limit=2, filters={"$and": [{"tags": "a"}, {"tags": {"$in": ["b", "c"]}}]})

    assert zot.top_calls == [{"format": "keys", "limit": semantic_search.PREFILTER_MAX_KEYS + 1, "tag": ["a", "b || c"]}]


def test_an_item_key_filter_needs_no_request(monkeypatch, tmp_path):
    hits, items = _corpus(articles=2, books=2)
    s, zot = _search(monkeypatch, tmp_path, hits, items)

    out = s.search("q", limit=2, filters={"item_key": {"$in": ["BOOK0001", "ART00000"]}})

    assert zot.top_calls == []
    assert sorted(_keys(out)) == ["ART00000", "BOOK0001"]
    assert len(s.chroma_client.calls) == 1


def test_item_key_and_item_type_are_intersected(monkeypatch, tmp_path):
    hits, items = _corpus(articles=2, books=2)
    s, zot = _search(monkeypatch, tmp_path, hits, items)

    out = s.search(
        "q", limit=5, filters={"$and": [{"item_type": "book"}, {"item_key": {"$in": ["BOOK0001", "ART00000"]}}]}
    )

    assert len(zot.top_calls) == 1
    assert _keys(out) == ["BOOK0001"]


def test_a_tag_with_a_space_or_empty_is_filtered_after_the_search_not_sent(monkeypatch, tmp_path):
    hits, items = _corpus(articles=1, books=1)
    items[0]["data"]["tags"] = [{"tag": "to read"}]
    s, zot = _search(monkeypatch, tmp_path, hits, items)

    out = s.search("q", limit=5, filters={"tags": "to read"})
    assert _keys(out) == ["ART00000"] and zot.top_calls == []

    out = s.search("q", limit=5, filters={"tags": ""})
    assert _keys(out) == ["BOOK0000"] and zot.top_calls == []


def test_a_filter_of_several_keys_is_an_error_and_asks_nobody(monkeypatch, tmp_path):
    hits, items = _corpus()
    s, zot = _search(monkeypatch, tmp_path, hits, items)

    out = s.search("q", limit=5, filters={"item_type": "book", "tags": "a"})

    assert out["error"].startswith("Expected where to have exactly one operator")
    assert zot.top_calls == [] and s.chroma_client.calls == []


def test_clauses_that_exclude_each_other_return_nothing_without_a_query(monkeypatch, tmp_path):
    hits, items = _corpus()
    s, zot = _search(monkeypatch, tmp_path, hits, items)

    out = s.search("q", limit=5, filters={"$and": [{"item_type": "book"}, {"item_type": "thesis"}]})

    assert _keys(out) == [] and zot.top_calls == [] and s.chroma_client.calls == []


# --- when it does not apply -------------------------------------------------------


@pytest.mark.parametrize(
    "filters",
    [
        {"item_type": {"$ne": "book"}},
        {"$or": [{"item_type": "book"}, {"tags": "x"}]},
        {"doi": "10.1/x"},
        {"has_fulltext": True},
    ],
)
def test_a_filter_the_api_cannot_express_is_filtered_after_the_query(monkeypatch, tmp_path, filters):
    hits, items = _corpus()
    s, zot = _search(monkeypatch, tmp_path, hits, items)

    s.search("q", limit=2, filters=filters)

    assert zot.top_calls == []
    assert s.chroma_client.calls[0]["where"] is None


@pytest.mark.parametrize(
    "gap",
    [{"attachments_without_parent_hint": 1}, {"passages_without_parent_hint": 4}, {"attachments_without_library": 1}],
)
def test_an_index_with_records_lacking_a_hint_is_filtered_after_the_query(monkeypatch, tmp_path, gap):
    hits, items = _corpus()
    s, zot = _search(monkeypatch, tmp_path, hits, items, layout=gap)

    out = s.search("q", limit=2, filters={"item_type": "book"})

    assert zot.top_calls == []
    assert s.chroma_client.calls[0]["where"] is None
    assert _keys(out) == ["BOOK0000", "BOOK0001"], "still exact, only slower"


def test_a_missing_library_is_a_gap_only_across_libraries(monkeypatch, tmp_path):
    hits, items = _corpus(articles=0, books=2)
    s, zot = _search(monkeypatch, tmp_path, hits, items, layout={"attachments_without_library": 1})

    s.search("q", limit=2, filters={"item_type": "book"}, group_id=0)

    assert len(zot.top_calls) == 1, "a group_id in the query already leaves out the unattributed"


def test_an_index_without_a_layout_is_filtered_after_the_query(monkeypatch, tmp_path):
    hits, items = _corpus()
    s, zot = _search(monkeypatch, tmp_path, hits, items)
    s.chroma_client.index_layout = lambda: None

    s.search("q", limit=2, filters={"item_type": "book"})

    assert zot.top_calls == []


def test_an_api_failure_falls_back_to_filtering_after_the_query(monkeypatch, tmp_path):
    hits, items = _corpus()
    s, zot = _search(monkeypatch, tmp_path, hits, items, fail=True)

    out = s.search("q", limit=2, filters={"item_type": "book"})

    assert len(zot.top_calls) == 1
    assert s.chroma_client.calls[0]["where"] is None
    assert _keys(out) == ["BOOK0000", "BOOK0001"]


def test_one_library_failing_falls_back_for_the_whole_search(monkeypatch, tmp_path):
    hits, items = _corpus()
    s, _ = _search(monkeypatch, tmp_path, hits, items, layout={"library_ids": [0, 777]})
    s._scoped_clients[777] = ApiZot([], library_id="777", fail=True)

    out = s.search("q", limit=2, filters={"item_type": "book"})

    assert s.chroma_client.calls[0]["where"] is None
    assert _keys(out) == ["BOOK0000", "BOOK0001"]


def test_a_library_without_a_client_falls_back(monkeypatch, tmp_path):
    hits, items = _corpus()
    s, _ = _search(monkeypatch, tmp_path, hits, items, layout={"library_ids": [0, 777]})
    s._scoped_clients[777] = None

    s.search("q", limit=2, filters={"item_type": "book"})

    assert s.chroma_client.calls[0]["where"] is None


def test_a_cut_short_answer_falls_back(monkeypatch, tmp_path):
    hits, items = _corpus()
    s, zot = _search(monkeypatch, tmp_path, hits, items, next_link=True)

    out = s.search("q", limit=2, filters={"item_type": "book"})

    assert s.chroma_client.calls[0]["where"] is None
    assert _keys(out) == ["BOOK0000", "BOOK0001"]


def test_more_owners_than_the_cap_falls_back(monkeypatch, tmp_path):
    hits, items = _corpus(articles=0, books=3)
    s, zot = _search(monkeypatch, tmp_path, hits, items)
    monkeypatch.setattr(semantic_search, "PREFILTER_MAX_KEYS", 2)

    out = s.search("q", limit=2, filters={"item_type": "book"})

    assert s.chroma_client.calls[0]["where"] is None
    assert _keys(out) == ["BOOK0000", "BOOK0001"]
    assert zot.top_calls[0]["limit"] == 3, "never more keys downloaded than one past the cap"


class SlowZot(ApiZot):
    def top(self, **kw):
        import time

        time.sleep(0.05)
        return super().top(**kw)


def test_libraries_not_yet_asked_are_skipped_once_one_is_over_the_cap(monkeypatch, tmp_path):
    """A broad filter fails on the big library; the rest of the libraries are not asked."""
    hits, items = _corpus(articles=0, books=3)
    groups = [778, 779, 780, 781, 782, 783]
    s, zot = _search(monkeypatch, tmp_path, hits, items, layout={"library_ids": [*reversed(groups), 0]})
    monkeypatch.setattr(semantic_search, "PREFILTER_MAX_KEYS", 2)
    others = {g: SlowZot([], library_id=str(g)) for g in groups}
    s._scoped_clients.update(others)

    out = s.search("q", limit=2, filters={"item_type": "book"})

    assert len(zot.top_calls) == 1, "the personal library is asked first"
    asked = sum(len(o.top_calls) for o in others.values())
    assert asked <= semantic_search._API_CONCURRENCY - 1, "only the first wave beside it was asked"
    assert s.chroma_client.calls[0]["where"] is None
    assert _keys(out) == ["BOOK0000", "BOOK0001"]


# --- the memo and the re-check ------------------------------------------------------


def test_a_pre_filtered_search_that_asks_again_looks_each_key_up_once(monkeypatch, tmp_path):
    """Twelve passages of one book crowd the first fetch; the second book comes in the next round."""
    hits = [_hit(f"ATTBOK0#{n}", item_key="ATTBOK0", parent_hint="BOOK0000", chunk_index=n, n_chunks=12) for n in range(12)]
    hits.append(_hint_hit("ATTBOK1", "BOOK0001"))
    items = [
        _regular("BOOK0000", item_type="book"), _attachment("ATTBOK0", "BOOK0000"),
        _regular("BOOK0001", item_type="book"), _attachment("ATTBOK1", "BOOK0001"),
    ]
    s, zot = _search(monkeypatch, tmp_path, hits, items)

    out = s.search("q", limit=2, filters={"item_type": "book"})

    assert _keys(out) == ["BOOK0000", "BOOK0001"]
    assert len(s.chroma_client.calls) == 2
    asked = [key for call in zot.batch_calls for key in call]
    assert len(asked) == len(set(asked)), "a key was fetched again in a later round"
    assert len(zot.top_calls) == 1


def test_the_local_pre_filter_is_checked_live_too(monkeypatch, tmp_path):
    """zotero.sqlite's owner list and the resolved item can disagree (a stale snapshot): the live data wins."""
    hits = [_hit("ATT1#0", item_key="ATT1", chunk_index=0, n_chunks=1)]
    items = [_regular("PAR1", tags=[]), _attachment("ATT1", "PAR1")]
    reader = FakeReader(items, [_owner("PAR1", tags=["keep"], attachments=["ATT1"])])
    zot = ApiZot([])
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: zot)
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    monkeypatch.setattr(semantic_search, "LocalZoteroReader", lambda **kw: reader)
    s = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChroma(hits), config_path=str(tmp_path / "config.json"))
    s._chunking_config = {"enabled": True}
    s._reranker_config = {"enabled": False}
    s._reranker = None

    out = s.search("q", limit=2, filters={"tags": "keep"})

    assert s.chroma_client.calls[0]["where"] == {"item_key": {"$in": ["ATT1", "PAR1"]}}
    assert _keys(out) == [], "the resolved owner has no such tag"
