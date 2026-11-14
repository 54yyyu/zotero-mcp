"""The current code on indexes written by the real old releases (#826).

``tests/fixtures/legacy_indexes/pre017.zip`` was written by zotero-mcp-server 0.16.0 (one
collection, passages only) and ``v0171.zip`` by 0.17.1 (articles in ``zotero_library``, passages in
``zotero_library_passages``), both with chromadb 1.5.9 over the small fake Zotero in
``fake_zotero.py``. ``make_legacy_indexes.py`` rebuilds them; see the README there.

Each test unzips one into ``tmp_path/home/.config/zotero-mcp`` (HOME and ``Path.home`` are already
pointed there by conftest), rebuilds the very same fake Zotero beside it, and runs the current
code in local mode. The embedding is the fixtures' offline one, and anything that is not a query
embedding fails the test: opening an old index must not cost an embedding.
"""

import importlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

FIXTURES = Path(__file__).parent / "fixtures" / "legacy_indexes"
sys.path.insert(0, str(FIXTURES))

import fake_zotero  # noqa: E402
from fake_embedding import FakeDefaultEmbedding  # noqa: E402

from zotero_mcp import extract, local_db, semantic_search  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient  # noqa: E402

ARTICLES = "zotero_library"
PASSAGES = "zotero_library_passages"

#: Passages the old releases wrote per item (chunk_size 400, overlap 50); 0.16 and 0.17.1 agree.
PASSAGES_PER_ITEM = {"ITEMONE1": 4, "ITEMTWO2": 6, "ITEMTHR3": 1, "ITEMFOU4": 19}

QUERIES = {
    "ITEMFOU4": "glacier retreat alpine",
    "ITEMONE1": "soil moisture radar backscatter",
    "ITEMTWO2": "bayesian species abundance forest plots",
    "ITEMTHR3": "lichen taxonomy",
}


def _api_item(key, item_type, tags=(), **fields):
    data = {"key": key, "itemType": item_type, "tags": [{"tag": t} for t in tags], "creators": [], **fields}
    return {"key": key, "data": data}


class OfflineZotero:
    """The Zotero API for the fake library, offline: the user's running Zotero is never asked.

    Search resolves hits from zotero.sqlite in local mode, so ``calls`` should stay empty.
    """

    library_type = "users"
    library_id = "0"
    backoff_until = None

    def __init__(self):
        self.calls: list[str] = []
        self._by_key = {}
        for key, _id, title, _abstract, _creators, tags, doi, _journal in fake_zotero.ITEMS:
            self._by_key[key] = _api_item(key, "journalArticle", tags=tags, title=title, DOI=doi)
        for att_key, _id, parent, name, _text in fake_zotero.ATTACHMENTS:
            self._by_key[att_key] = _api_item(
                att_key, "attachment", title=name, parentItem=parent, filename=name, contentType="application/pdf"
            )

    def item_versions(self, since=None, **kwargs):
        """Every item at the version the fake library holds it at: nothing has changed."""
        return {key: 5 for key in fake_zotero.ITEM_KEYS + fake_zotero.ATTACHMENT_KEYS}

    def last_modified_version(self, **kwargs):
        return 5

    def item(self, key, **kwargs):
        self.calls.append(f"item {key}")
        return self._by_key[key]

    def items(self, itemKey="", start=0, limit=100, **kwargs):
        self.calls.append(f"items {itemKey}")
        return [self._by_key[k] for k in itemKey.split(",") if k in self._by_key][start : start + limit]

    def __getattr__(self, name):
        raise AssertionError(f"unexpected Zotero API call: {name}")


class StrictEmbedding(FakeDefaultEmbedding):
    """The fixtures' embedding, but only for the texts the test allows (the query)."""

    allowed: set[str] | None = set()  # None: refuse nothing, only record
    seen: list[str] = []

    def __call__(self, input):
        for text in input:
            type(self).seen.append(text)
            if type(self).allowed is not None and text not in type(self).allowed:
                raise AssertionError(f"an embedding was asked for: {text[:60]!r}")
        return super().__call__(input)


class Library:
    """One unzipped legacy index, its fake Zotero, and the current code opened on them."""

    def __init__(self, kind, tmp_path, monkeypatch):
        self.kind = kind
        self.home = tmp_path / "home"
        self.config_dir = self.home / ".config" / "zotero-mcp"
        self.config_path = self.config_dir / "config.json"
        self.persist_dir = self.config_dir / "chroma_db"
        with zipfile.ZipFile(FIXTURES / f"{kind}.zip") as zf:
            zf.extractall(self.config_dir)
        self.db = fake_zotero.build(tmp_path / "zotero")
        self.zotero = OfflineZotero()
        monkeypatch.setenv("ZOTERO_DB_PATH", str(self.db))
        monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
        # migrate-index opens zotero.sqlite. The module as ``from .utils import`` finds it, not as the
        # package attribute names it: other test files re-import zotero_mcp and leave the two apart.
        monkeypatch.setattr(importlib.import_module("zotero_mcp.utils"), "is_local_mode", lambda: True)
        monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: self.zotero)
        StrictEmbedding.allowed = set()
        StrictEmbedding.seen = []
        monkeypatch.setattr(ChromaClient, "_create_embedding_function", lambda self: StrictEmbedding())
        self.reads: list[str] = []
        self._spy_on_reads(monkeypatch)

    def _spy_on_reads(self, monkeypatch):
        """Record every attempt to read a file's text, then let it through."""

        def spy(owner, name):
            original = getattr(owner, name)

            def wrapper(*args, **kwargs):
                self.reads.append(name)
                return original(*args, **kwargs)

            monkeypatch.setattr(owner, name, wrapper)

        for name in ("extract_file", "extract_pdf", "extract_html", "extract_text_file"):
            spy(extract, name)
        for name in ("_extract_doc_from_file", "_read_zotero_ft_cache", "_extract_target", "_extract_fulltext_for_item"):
            spy(local_db.LocalZoteroReader, name)

    def search_object(self):
        return semantic_search.create_semantic_search(str(self.config_path))

    def query(self, text):
        """Allow exactly this text to be embedded (the query)."""
        StrictEmbedding.allowed = {text}

    # --- what the index holds ---------------------------------------------------------------

    def collections(self):
        import chromadb
        from chromadb.config import Settings

        # The settings ChromaClient opens with: ChromaDB shares one system per path.
        client = chromadb.PersistentClient(
            str(self.persist_dir), settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        return {c.name: c for c in client.list_collections()}

    def records(self, name):
        """``{record id: metadata}`` of one collection, read straight from ChromaDB; None if absent."""
        col = self.collections().get(name)
        if col is None:
            return None
        got = col.get(include=["metadatas"])
        return dict(zip(got["ids"], got["metadatas"], strict=True))

    def vectors(self, name):
        col = self.collections()[name]
        got = col.get(include=["embeddings"])
        return {i: list(map(float, e)) for i, e in zip(got["ids"], got["embeddings"], strict=True)}

    def snapshot(self):
        """Every record of the index with its metadata and vector, for before/after comparisons."""
        out = {}
        for name in self.collections():
            vectors = self.vectors(name)
            out[name] = {k: (m, vectors[k]) for k, m in self.records(name).items()}
        return out


@pytest.fixture(params=["pre017", "v0171"])
def library(request, tmp_path, monkeypatch):
    return Library(request.param, tmp_path, monkeypatch)


def _search(library, search, query, **kwargs):
    library.query(query)
    return search.search(query, **kwargs)


# --- what the old releases wrote -------------------------------------------------------------


def test_the_fixture_is_the_layout_the_old_release_wrote(library):
    articles, passages = library.records(ARTICLES), library.records(PASSAGES)
    expected_ids = {f"{key}#{n}" for key, count in PASSAGES_PER_ITEM.items() for n in range(count)}
    if library.kind == "pre017":
        # 0.16 with chunking on: passages only, in the one collection; a metadata-only item is one passage.
        assert passages is None
        assert set(articles) == expected_ids
        passage_records = articles
    else:
        assert set(articles) == set(PASSAGES_PER_ITEM)  # one article record per item, a bare KEY
        assert set(passages) == expected_ids
        passage_records = passages
    for record_id, meta in passage_records.items():
        key, number = record_id.split("#")
        assert meta["parent_item_key"] == key and meta["item_key"] == key
        assert meta["title"] and meta["chunk_index"] == int(number)  # metadata is copied
        assert "parent_hint" not in meta and "file_stamp" not in meta
    assert passage_records["ITEMTWO2#0"]["attachment_keys"] == "ATTTWOA2,ATTTWOB2"
    assert passage_records["ITEMTHR3#0"]["attachment_keys"] == ""


def test_the_fixture_vectors_come_from_the_offline_embedding(library):
    col = library.collections()[PASSAGES if library.kind == "v0171" else ARTICLES]
    got = col.get(include=["documents", "embeddings"])
    assert len(got["ids"]) == 30
    for document, vector in zip(got["documents"], got["embeddings"], strict=True):
        assert len(vector) == fake_zotero.DIM
        assert list(map(float, vector)) == pytest.approx(fake_zotero.embed(document), abs=1e-6)


def test_the_old_config_has_no_schema_stamp(library):
    config = json.loads(library.config_path.read_text())["semantic_search"]
    assert "index_schema_version" not in config
    assert config["chunking"]["enabled"] is True


# --- status ------------------------------------------------------------------------------------


def test_the_layout_reads_as_an_old_index(library):
    from zotero_mcp.chroma_client import read_index_layout

    layout = read_index_layout(str(library.persist_dir), ARTICLES)
    assert layout["attachment_records"] == 0 and layout["passages_without_parent_hint"] == 0
    assert layout["legacy_passages"] == 30 and layout["legacy_passage_items"] == 4
    if library.kind == "pre017":
        assert layout["article_records"] == 0 and layout["passage_records"] == 30
        assert layout["unmigrated_passages"] == 30 and layout["passages_in_article_collection"] == 30
        assert layout["passage_collection_exists"] is False
    else:
        assert layout["article_records"] == 4 and layout["legacy_records"] == 4
        assert layout["passage_records"] == 30 and layout["unmigrated_passages"] == 0
        assert layout["passage_collection_exists"] is True


def test_db_status_says_what_an_old_index_needs(library):
    info = library.search_object().get_database_status()["collection_info"]
    assert info["passage_records"] == 30 and info["passage_items"] == 4
    if library.kind == "pre017":
        assert info["article_records"] == 0 and info["unmigrated_passages"] == 30
        assert info["migration"]["items_needing_articles"] == 4
    else:
        assert info["article_records"] == 4 and info["unmigrated_passages"] == 0
        assert "migration" not in info
    assert StrictEmbedding.seen == []


# --- search ------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(QUERIES))
def test_search_returns_the_parent_item(library, key):
    out = _search(library, library.search_object(), QUERIES[key], limit=4)
    assert "error" not in out, out
    results = out["results"]
    assert results[0]["item_key"] == key
    assert results[0]["zotero_item"]["data"]["title"] == next(i[2] for i in fake_zotero.ITEMS if i[0] == key)
    assert len({r["item_key"] for r in results}) == len(results)  # one result per item
    assert set(StrictEmbedding.seen) == {QUERIES[key]}  # nothing but the query was embedded
    assert library.zotero.calls == []  # owners came from zotero.sqlite, not the API


def test_a_passage_hit_keeps_where_it_was_found(library):
    out = _search(library, library.search_object(), "mass balance stakes Hintereis", limit=3, level="passages")
    top = out["results"][0]
    assert top["item_key"] == "ITEMFOU4"
    assert top["n_chunks"] == 19 and top["page"] in (1, 2, 3)
    assert "Mass balance" in top["matched_passage"]


def test_article_level_search(library):
    out = _search(library, library.search_object(), "glacier retreat alpine", limit=4, level="articles")
    if library.kind == "pre017":
        assert out["results"] == [] and "migrate-index" in out["error"]
    else:
        assert "error" not in out, out
        assert out["results"][0]["item_key"] == "ITEMFOU4" and out["level"] == "articles"


def test_filters_read_zotero_now_not_the_copied_metadata(library):
    search = library.search_object()

    def keys(filters, **kwargs):
        out = _search(library, search, "glacier ecology lichen soil", limit=10, filters=filters, **kwargs)
        assert "error" not in out, out
        return {r["item_key"] for r in out["results"]}

    # Every old record copied doi "" and tags ""; Zotero has them.
    assert keys({"doi": "10.1000/ice.4"}) == {"ITEMFOU4"}
    assert keys({"tags": "taxonomy"}) == {"ITEMTHR3"}
    assert keys({"title": "Bayesian methods in community ecology"}) == {"ITEMTWO2"}
    assert keys({"item_type": "journalArticle"}) == set(fake_zotero.ITEM_KEYS)
    assert keys({"title": {"$in": ["A note on lichen taxonomy", "Glacier retreat in the Alps since 1850"]}}) == {
        "ITEMTHR3",
        "ITEMFOU4",
    }
    assert keys({"doi": "10.1000/nope"}) == set()
    assert keys(None, group_id=0) == set(fake_zotero.ITEM_KEYS)
    assert keys(None, group_id=7) == set()


# --- updating an untouched library ---------------------------------------------------------------


def _embedded_nothing(library):
    assert StrictEmbedding.seen == []
    assert library.reads == []


#: The runs a user meets: the startup/pre-search update, ``update-db``, and ``update-db --fulltext``.
RUNS = {
    "automatic": {"automatic": True},
    "automatic_fulltext": {"automatic": True, "extract_fulltext": True},
    "explicit": {},
    "explicit_fulltext": {"extract_fulltext": True},
}

AUTOMATIC = ("automatic", "automatic_fulltext")


@pytest.mark.parametrize("run", AUTOMATIC)
@pytest.mark.parametrize("kind", ["pre017", "v0171"])
def test_an_automatic_update_leaves_an_old_index_alone(kind, run, tmp_path, monkeypatch):
    """The startup update neither migrates, re-keys nor embeds: not one record is written, moved or deleted."""
    library = Library(kind, tmp_path, monkeypatch)
    search = library.search_object()
    before = library.snapshot()
    stats = search.update_database(**RUNS[run])
    assert stats.get("error") is None and stats["errors"] == 0, stats
    assert stats["added_items"] == stats["updated_items"] == stats["deleted_items"] == 0
    assert StrictEmbedding.seen == []
    assert library.reads == []
    assert library.snapshot() == before


# --- migrate-index re-keys the passages to their attachments ----------------------------------------

#: legacy passage id -> its id once re-keyed: the file 0.17 chose, by item.
REKEYED = {
    f"{key}#{n}": f"{fake_zotero.LARGEST[key]}#{n}"
    for key, count in PASSAGES_PER_ITEM.items()
    if key in fake_zotero.LARGEST
    for n in range(count)
}
OWN_PASSAGE_KEYS = {
    "item_key", "group_id", "chunk_index", "n_chunks", "char_start", "char_end", "page", "section", "parent_hint",
}


def _stamped(articles):
    """0.17 article records after the re-key: unchanged, vectors included, but for ``rekeyed_to`` naming the file."""
    return {
        key: ({**meta, **({"rekeyed_to": fake_zotero.LARGEST[key]} if key in fake_zotero.LARGEST else {})}, vector)
        for key, (meta, vector) in articles.items()
    }


def _migrate(library, **kwargs):
    from zotero_mcp.index_migration import migrate_index

    return migrate_index(str(library.config_path), progress=lambda _m: None, **kwargs)


def test_a_dry_run_reports_the_rekey_and_changes_nothing(library):
    before = library.snapshot()
    dry = _migrate(library, dry_run=True)
    assert library.snapshot() == before
    assert dry.needed and dry.legacy_passages == 30 and dry.legacy_items == 4
    assert dry.rekeyed_items == 3 and dry.rekeyed_passages == len(REKEYED)
    assert dry.dropped_items == 1 and dry.dropped_passages == 1  # the file-less item's one passage
    assert dry.skipped_items == 0
    assert StrictEmbedding.seen == [] and library.reads == []


def test_migrate_index_moves_every_passage_to_its_attachment(library):
    before = library.snapshot()
    old_passages = before[ARTICLES if library.kind == "pre017" else PASSAGES]
    report = _migrate(library)
    after = library.snapshot()

    assert StrictEmbedding.seen == [] and library.reads == []  # nothing embedded, no file read
    assert report.rekeyed_items == 3 and report.rekeyed_passages == len(REKEYED) and report.skipped_items == 0
    assert report.renamed is (library.kind == "pre017")

    passages = after[PASSAGES]
    assert set(passages) == set(REKEYED.values())
    for old_id, new_id in REKEYED.items():
        old_meta, old_vector = old_passages[old_id]
        meta, vector = passages[new_id]
        assert vector == old_vector  # the stored vector, untouched
        assert set(meta) <= OWN_PASSAGE_KEYS  # the copied metadata is gone
        assert meta["item_key"] == new_id.split("#")[0] and meta["chunk_index"] == int(new_id.split("#")[1])
        assert meta["parent_hint"] == old_id.split("#")[0]
        assert meta["n_chunks"] == old_meta["n_chunks"]
    # The two-file item went to the larger file, which is what 0.17 indexed.
    assert {k.split("#")[0] for k in passages} == {"ATTONEA1", "ATTTWOB2", "ATTFOUR4"}

    if library.kind == "v0171":
        # Every article record is where it was, vector and fields untouched, the file-less item's
        # included; the re-key only notes which file each item's passages went to.
        assert after[ARTICLES] == _stamped(before[ARTICLES])
        assert set(after[ARTICLES]) == set(fake_zotero.ITEM_KEYS)
    else:
        assert after[ARTICLES] == {}  # still nothing: its articles come with the next update
        assert report.items_needing_articles == 4


def test_a_file_less_items_passage_is_dropped_and_its_record_kept(library):
    """ITEMTHR3's only passage repeated its metadata text: 0.17.1 has the article record for that."""
    before = library.snapshot()
    _migrate(library)
    after = library.snapshot()
    assert not any(k.startswith("ITEMTHR3") for k in after[PASSAGES])
    if library.kind == "v0171":
        assert after[ARTICLES]["ITEMTHR3"] == before[ARTICLES]["ITEMTHR3"]


def test_migrate_index_twice_is_a_no_op(library):
    _migrate(library)
    after = library.snapshot()
    again = _migrate(library)
    assert not again.needed and again.rekeyed_items == 0 and again.rekeyed_passages == 0
    assert library.snapshot() == after
    assert StrictEmbedding.seen == [] and library.reads == []


def test_search_still_finds_the_parents_after_the_rekey(library):
    _migrate(library)
    search = library.search_object()
    for key, query in QUERIES.items():
        if key == "ITEMTHR3":
            continue  # no file, so no passage left: found by its article record, where there is one
        out = _search(library, search, query, limit=4, level="passages")
        assert "error" not in out, out
        assert out["results"][0]["item_key"] == key
    if library.kind == "v0171":
        out = _search(library, search, QUERIES["ITEMTHR3"], limit=4, level="articles")
        assert out["results"][0]["item_key"] == "ITEMTHR3"
    # A file's passage is found through the item that owns the file.
    out = _search(library, search, "mass balance stakes Hintereis", limit=3, level="passages")
    assert out["results"][0]["item_key"] == "ITEMFOU4"
    assert out["results"][0]["matched_attachment"]["key"] == "ATTFOUR4"
    assert set(StrictEmbedding.seen) >= {"mass balance stakes Hintereis"}
    assert library.zotero.calls == []


# --- updates after the migration ---------------------------------------------------------------------


def _new_records(before, after):
    return {c: set(after[c]) - set(before.get(c, {})) for c in after}


def test_updating_a_migrated_0171_index_embeds_nothing(tmp_path, monkeypatch):
    """Its items are untouched legacy items: their article records stand, whatever the passages' keys."""
    library = Library("v0171", tmp_path, monkeypatch)
    _migrate(library)
    search = library.search_object()
    after = library.snapshot()
    for run in sorted(RUNS):
        stats = search.update_database(**RUNS[run])
        assert stats["errors"] == 0, (run, stats)
    assert StrictEmbedding.seen == [] and library.reads == []
    assert library.snapshot() == after


def test_an_automatic_update_of_a_migrated_0160_index_embeds_only_the_missing_articles(tmp_path, monkeypatch):
    """Its items have passages but no article record: one cheap record each, as 0.17.1's gap fill wrote.

    No passage and no attachment record is written, and no file is read.
    """
    library = Library("pre017", tmp_path, monkeypatch)
    _migrate(library)
    StrictEmbedding.allowed = None  # record, do not refuse
    search = library.search_object()
    before = library.snapshot()
    stats = search.update_database(**RUNS["automatic_fulltext"])
    assert stats["errors"] == 0, stats
    after = library.snapshot()
    assert _new_records(before, after) == {ARTICLES: set(fake_zotero.ITEM_KEYS), PASSAGES: set()}
    assert len(StrictEmbedding.seen) == 4 and library.reads == []
    assert after[PASSAGES] == before[PASSAGES]
    StrictEmbedding.seen.clear()
    search.update_database(**RUNS["automatic_fulltext"])
    assert StrictEmbedding.seen == [] and library.snapshot() == after


#: Every file of the chosen kind is indexed (#494): the smaller PDF that 0.16
#: never indexed is new content, with its record and its passages.
EXTRA_FILES = {"ATTTWOA2": {"ATTTWOA2#0", "ATTTWOA2#1"}}


@pytest.mark.parametrize("fresh_migration", [True, False], ids=["after-migrate-index", "straight-from-0160"])
def test_an_explicit_update_writes_only_the_missing_records_not_the_passages(fresh_migration, tmp_path, monkeypatch):
    """4 article records, the 3 attachment records of the files 0.16 indexed, and the
    new smaller PDF with its passages: 10 embeddings, no old passage again.

    Straight from the 0.16 index, the update splits and re-keys first, which embeds nothing.
    """
    library = Library("pre017", tmp_path, monkeypatch)
    if fresh_migration:
        _migrate(library)
    StrictEmbedding.allowed = None
    search = library.search_object()
    before = library.snapshot()
    stats = search.update_database(**RUNS["explicit_fulltext"])
    assert stats["errors"] == 0, stats
    after = library.snapshot()
    new = _new_records(before, after)
    extra_passages = set().union(*EXTRA_FILES.values())
    assert new[ARTICLES] == set(fake_zotero.ITEM_KEYS) | set(fake_zotero.LARGEST.values()) | set(EXTRA_FILES)
    if fresh_migration:
        assert new[PASSAGES] == extra_passages
        assert all(after[PASSAGES][k] == v for k, v in before[PASSAGES].items())
    else:
        # moved by the re-key, vectors untouched
        assert set(after[PASSAGES]) == set(REKEYED.values()) | extra_passages
        assert all(after[PASSAGES][new_id][1] == before[ARTICLES][old_id][1] for old_id, new_id in REKEYED.items())
    assert len(StrictEmbedding.seen) == 7 + len(EXTRA_FILES) + len(extra_passages)  # never an old passage
    attachments = {k: after[ARTICLES][k][0] for k in [*fake_zotero.LARGEST.values(), *EXTRA_FILES]}
    assert all(
        set(m) == {"item_key", "group_id", "date_modified", "file_stamp", "parent_hint", "file_hash"}
        for m in attachments.values()
    )
    assert {m["parent_hint"] for m in attachments.values()} == set(fake_zotero.LARGEST)
    # Each file's content hash (#495): distinct files, so none was copied from another.
    assert len({m["file_hash"] for m in attachments.values()}) == len(attachments)
    # Up to date from now on, for every kind of run.
    StrictEmbedding.seen.clear()
    library.reads.clear()
    for run in sorted(RUNS):
        search.update_database(**RUNS[run])
    assert StrictEmbedding.seen == [] and library.reads == []
    assert library.snapshot() == after


def test_an_explicit_update_of_a_0171_index_re_keys_it_without_embedding(tmp_path, monkeypatch):
    library = Library("v0171", tmp_path, monkeypatch)
    before = library.snapshot()
    search = library.search_object()
    for run in ("explicit", "explicit_fulltext"):
        stats = search.update_database(**RUNS[run])
        assert stats["errors"] == 0, stats
    after = library.snapshot()
    assert StrictEmbedding.seen == [] and library.reads == []
    assert set(after[PASSAGES]) == set(REKEYED.values())
    assert after[ARTICLES] == _stamped(before[ARTICLES])
    assert all(after[PASSAGES][new][1] == before[PASSAGES][old][1] for old, new in REKEYED.items())
