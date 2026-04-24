"""Tests for web-API fulltext ingest and since-based incremental sync.

These tests cover the web-API branch introduced by the PR that adds fulltext
fetching via pyzotero's `fulltext_item` endpoint and version-based incremental
updates for users who don't have `ZOTERO_LOCAL=true` set (e.g. Claude Code on
a headless Linux server talking to Zotero cloud).
"""

import json
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search


class FakeChromaClient:
    """Minimal ChromaClient stand-in that records operations for assertions."""

    def __init__(self, preloaded_ids=None):
        self.embedding_max_tokens = 8000
        self._ids = set(preloaded_ids or [])
        self.added = []  # list of (docs, metas, ids)
        self.deleted = []  # list of ids deleted
        self.reset_calls = 0

    def truncate_text(self, text, max_tokens=None):
        return text[:4000]

    def get_existing_ids(self, ids):
        return {i for i in ids if i in self._ids}

    def get_all_ids(self):
        return set(self._ids)

    def get_document_metadata(self, doc_id):
        return None

    def upsert_documents(self, documents, metadatas, ids):
        self.added.append((list(documents), list(metadatas), list(ids)))
        for i in ids:
            self._ids.add(i)

    def add_documents(self, documents, metadatas, ids):
        self.upsert_documents(documents, metadatas, ids)

    def delete_documents(self, ids):
        self.deleted.extend(list(ids))
        for i in ids:
            self._ids.discard(i)

    def delete_documents_by_parent(self, parent_item_key):
        prefix = f"{parent_item_key}__"
        victims = [i for i in self._ids if i == parent_item_key or i.startswith(prefix)]
        if victims:
            self.delete_documents(victims)
        return len(victims)

    def reset_collection(self):
        self.reset_calls += 1
        self._ids = set()


class FakeZoteroClient:
    """pyzotero client double with scripted responses."""

    def __init__(self):
        self.items_by_key = {}
        self.fulltext_by_key = {}   # key -> dict (content, indexedChars, ...)
        self.children_by_parent = {}  # parent_key -> list[item]
        self.versions_state = {}    # key -> library_version (current state)
        self.version_history = []   # (since_version, changed_dict) pairs
        self.current_library_version = 0
        # Pagination helper
        self.items_order = []
        # Recording calls for assertions
        self.calls = []

    # ---- setup helpers for tests ----

    def load_scenario(self, items, fulltext=None, children=None, library_version=0):
        """items: list of pyzotero-shaped dicts with top-level 'key' and 'data'."""
        for it in items:
            self.items_by_key[it["key"]] = it
            self.items_order.append(it["key"])
            self.versions_state[it["key"]] = library_version
        for k, v in (fulltext or {}).items():
            self.fulltext_by_key[k] = v
        for k, v in (children or {}).items():
            self.children_by_parent[k] = v
        self.current_library_version = library_version

    # ---- pyzotero surface used by the code under test ----

    def items(self, start=0, limit=100, **kwargs):
        self.calls.append(("items", start, limit))
        chunk = self.items_order[start:start + limit]
        return [self.items_by_key[k] for k in chunk]

    def item(self, key):
        self.calls.append(("item", key))
        if key not in self.items_by_key:
            raise LookupError(f"item {key} not found")
        return self.items_by_key[key]

    def children(self, key):
        self.calls.append(("children", key))
        return list(self.children_by_parent.get(key, []))

    def fulltext_item(self, key):
        self.calls.append(("fulltext_item", key))
        if key not in self.fulltext_by_key:
            raise RuntimeError(f"404 fulltext not found for {key}")
        return self.fulltext_by_key[key]

    def item_versions(self, since=None, **kwargs):
        self.calls.append(("item_versions", since))
        if since is None:
            return dict(self.versions_state)
        return {k: v for k, v in self.versions_state.items() if v > since}

    def last_modified_version(self, **kwargs):
        self.calls.append(("last_modified_version",))
        return self.current_library_version


def _paper(key, title="Paper", item_type="conferencePaper", version=1):
    return {
        "key": key,
        "version": version,
        "data": {
            "key": key,
            "itemType": item_type,
            "title": title,
            "abstractNote": f"abstract of {title}",
            "creators": [{"creatorType": "author", "firstName": "A", "lastName": "Author"}],
            "dateAdded": "2024-01-01T00:00:00Z",
            "dateModified": "2024-01-01T00:00:00Z",
        },
    }


def _build_search(monkeypatch, zot: FakeZoteroClient, chroma: FakeChromaClient,
                  config_path: str | None = None) -> "semantic_search.ZoteroSemanticSearch":
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: zot)
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    return semantic_search.ZoteroSemanticSearch(
        chroma_client=chroma,
        config_path=config_path,
    )


# --------- Unit tests: fulltext fetch helper ----------

def test_fetch_fulltext_via_web_api_parent_hit(monkeypatch):
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("AAA")], fulltext={"AAA": {"content": "Body text.", "indexedChars": 10, "totalChars": 10}})
    search = _build_search(monkeypatch, zot, FakeChromaClient())
    text, source = search._fetch_fulltext_via_web_api("AAA")
    assert text == "Body text."
    assert source == "web-api:parent"


def test_fetch_fulltext_via_web_api_attachment_fallback(monkeypatch):
    """When parent has no fulltext (404), walk children and try PDF attachments."""
    parent = _paper("PAR", title="Paper")
    child = {
        "key": "CHILD1",
        "version": 1,
        "data": {"key": "CHILD1", "itemType": "attachment", "contentType": "application/pdf"},
    }
    zot = FakeZoteroClient()
    zot.load_scenario(
        [parent],
        # parent lookup will fail (key not in fulltext_by_key -> raise)
        fulltext={"CHILD1": {"content": "Attachment body."}},
        children={"PAR": [child]},
    )
    search = _build_search(monkeypatch, zot, FakeChromaClient())
    text, source = search._fetch_fulltext_via_web_api("PAR")
    assert text == "Attachment body."
    assert source == "web-api:attachment:CHILD1"
    # Verify we actually tried the parent first, then walked children
    call_names = [c[0] for c in zot.calls]
    assert "fulltext_item" in call_names and "children" in call_names


def test_fetch_fulltext_via_web_api_returns_empty_when_nothing_available(monkeypatch):
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("X")])
    # No fulltext, no children — every lookup fails
    search = _build_search(monkeypatch, zot, FakeChromaClient())
    text, source = search._fetch_fulltext_via_web_api("X")
    assert text == ""
    assert source == ""


def test_fetch_fulltext_skips_non_pdf_children(monkeypatch):
    parent = _paper("PAR")
    child_pdf = {"key": "PDF", "version": 1,
                 "data": {"key": "PDF", "itemType": "attachment", "contentType": "application/pdf"}}
    child_html = {"key": "HTM", "version": 1,
                  "data": {"key": "HTM", "itemType": "attachment", "contentType": "text/html"}}
    zot = FakeZoteroClient()
    zot.load_scenario(
        [parent],
        fulltext={"PDF": {"content": "PDF text."}, "HTM": {"content": "HTML text."}},
        children={"PAR": [child_html, child_pdf]},  # HTML listed first
    )
    search = _build_search(monkeypatch, zot, FakeChromaClient())
    text, source = search._fetch_fulltext_via_web_api("PAR")
    # Should skip the HTML attachment and return the PDF's content
    assert text == "PDF text."
    assert source == "web-api:attachment:PDF"


# --------- Integration tests: _get_items_from_api ----------

def test_get_items_from_api_without_fulltext_leaves_data_untouched(monkeypatch):
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("A"), _paper("B")], fulltext={"A": {"content": "Abody"}, "B": {"content": "Bbody"}})
    search = _build_search(monkeypatch, zot, FakeChromaClient())
    items = search._get_items_from_api(include_fulltext=False)
    assert len(items) == 2
    for it in items:
        assert "fulltext" not in it["data"]


def test_get_items_from_api_with_fulltext_populates_data(monkeypatch):
    zot = FakeZoteroClient()
    zot.load_scenario(
        [_paper("A"), _paper("B")],
        fulltext={"A": {"content": "Abody"}, "B": {"content": "Bbody"}},
    )
    search = _build_search(monkeypatch, zot, FakeChromaClient())
    items = search._get_items_from_api(include_fulltext=True)
    by_key = {it["key"]: it for it in items}
    assert by_key["A"]["data"]["fulltext"] == "Abody"
    assert by_key["A"]["data"]["fulltextSource"] == "web-api:parent"
    assert by_key["B"]["data"]["fulltext"] == "Bbody"


def test_get_items_from_api_with_fulltext_marks_misses_as_attempted(monkeypatch):
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("A")], fulltext={})  # no fulltext for A
    search = _build_search(monkeypatch, zot, FakeChromaClient())
    items = search._get_items_from_api(include_fulltext=True)
    assert items[0]["data"].get("fulltext", "") == ""
    assert items[0]["data"]["fulltext_attempted"] is True


# --------- Integration tests: incremental fetch ----------

def test_get_changed_items_from_api_returns_only_changed_keys(monkeypatch):
    zot = FakeZoteroClient()
    zot.load_scenario(
        [_paper("OLD"), _paper("NEW")],
        library_version=5,
    )
    # Mark OLD as unchanged since v3, NEW as changed at v5
    zot.versions_state = {"OLD": 3, "NEW": 5}
    search = _build_search(monkeypatch, zot, FakeChromaClient())
    changed, current_keys = search._get_changed_items_from_api(since_version=3, include_fulltext=False)
    assert [c["key"] for c in changed] == ["NEW"]
    assert current_keys == {"OLD", "NEW"}


def test_get_changed_items_filters_out_attachments_and_notes(monkeypatch):
    zot = FakeZoteroClient()
    note = _paper("N1", item_type="note")
    ann = _paper("A1", item_type="annotation")
    paper = _paper("P1", item_type="conferencePaper")
    zot.load_scenario([note, ann, paper], library_version=10)
    zot.versions_state = {"N1": 10, "A1": 10, "P1": 10}
    search = _build_search(monkeypatch, zot, FakeChromaClient())
    changed, _ = search._get_changed_items_from_api(since_version=0, include_fulltext=False)
    assert [c["key"] for c in changed] == ["P1"]


def test_get_items_from_api_filters_out_annotations(monkeypatch):
    """Regression guard: the bootstrap path used to exclude only
    attachment+note, letting annotation items (PDF highlights / user
    comments) sneak in as top-level chunks. Confirm they're gone."""
    zot = FakeZoteroClient()
    ann = _paper("A1", item_type="annotation")
    attach = _paper("X1", item_type="attachment")
    note = _paper("N1", item_type="note")
    paper = _paper("P1", item_type="journalArticle")
    zot.load_scenario([ann, attach, note, paper], library_version=3)
    search = _build_search(monkeypatch, zot, FakeChromaClient())
    items = search._get_items_from_api(include_fulltext=False)
    assert [it["key"] for it in items] == ["P1"]


def test_fetch_items_by_keys_filters_out_annotations(monkeypatch):
    """Gap-fill's batched fetch must also drop annotation items so
    previous runs' filter bug can't replicate through the resume path."""
    zot = FakeZoteroClient()
    ann = _paper("A1", item_type="annotation")
    paper = _paper("P1", item_type="journalArticle")
    zot.load_scenario([ann, paper], library_version=5)
    # Replace items() so it returns both when called (simulating batched
    # itemKey fetch)
    original_items = zot.items
    zot.items = lambda **kw: [zot.items_by_key["A1"], zot.items_by_key["P1"]]
    zot.add_parameters = lambda **kw: None
    search = _build_search(monkeypatch, zot, FakeChromaClient())
    result = search._fetch_items_by_keys(["A1", "P1"])
    assert [it["key"] for it in result] == ["P1"]


def test_update_database_cleans_up_stale_annotation_chunks(monkeypatch, tmp_path):
    """A user who ingested with the buggy filter ends up with orphan
    annotation chunks in ChromaDB. update_database must purge them on
    every run (it's idempotent / cheap) so the collection self-heals."""
    config_path = _write_config(tmp_path, extra={"last_sync_version": 10})
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("REAL", item_type="journalArticle")], library_version=10)
    zot.versions_state = {"REAL": 10}

    class CleanableChroma(FakeChromaClient):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            # Seed with a mix: 2 annotation chunks and 1 real paper chunk
            self._chunks = {
                "STALE_ANN_1__0": {"item_type": "annotation", "parent_item_key": "STALE_ANN_1"},
                "STALE_ANN_2__0": {"item_type": "annotation", "parent_item_key": "STALE_ANN_2"},
                "REAL__0": {"item_type": "journalArticle", "parent_item_key": "REAL"},
            }
            self._ids = set(self._chunks.keys())
            self.deleted_by_type = []

        def delete_documents_by_item_type(self, item_type):
            victims = [i for i, m in self._chunks.items() if m.get("item_type") == item_type]
            for v in victims:
                self._ids.discard(v)
                del self._chunks[v]
            if victims:
                self.deleted_by_type.append((item_type, len(victims)))
            return len(victims)

        def get_all_ids(self):
            return set(self._ids)

    chroma = CleanableChroma()
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)
    stats = search.update_database()

    assert ("annotation", 2) in chroma.deleted_by_type
    assert stats["cleaned_annotation_chunks"] == 2


# --------- Integration tests: update_database orchestration ----------

def _write_config(tmp_path, extra: dict | None = None):
    cfg = {
        "semantic_search": {
            "embedding_model": "default",
            "update_config": {"auto_update": False, "update_frequency": "manual"},
            "extraction": {"pdf_max_pages": 10},
            "include_fulltext": True,
        }
    }
    if extra:
        cfg["semantic_search"].update(extra)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    return str(config_path)


def test_update_database_bootstrap_scans_full_library(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path)
    zot = FakeZoteroClient()
    zot.load_scenario(
        [_paper("P1"), _paper("P2")],
        fulltext={"P1": {"content": "Paper one body"}, "P2": {"content": "Paper two body"}},
        library_version=7,
    )
    chroma = FakeChromaClient()
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    stats = search.update_database()

    assert stats["total_items"] == 2
    assert stats["processed_items"] == 2
    # last_sync_version should be persisted to config
    saved = json.loads(open(config_path).read())
    assert saved["semantic_search"]["last_sync_version"] == 7


def test_update_database_incremental_only_fetches_changed(monkeypatch, tmp_path):
    """Second run with last_sync_version set should only reindex changed items."""
    config_path = _write_config(tmp_path, extra={"last_sync_version": 5})
    zot = FakeZoteroClient()
    zot.load_scenario(
        [_paper("OLD"), _paper("NEW")],
        fulltext={"NEW": {"content": "New body"}},
        library_version=8,
    )
    zot.versions_state = {"OLD": 3, "NEW": 8}
    # Preload with new-format chunk ids so the legacy-format detector stays silent
    chroma = FakeChromaClient(preloaded_ids=["OLD__0"])
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    stats = search.update_database()

    # Only NEW should be processed, OLD untouched
    assert stats["processed_items"] == 1
    added_ids = [i for batch in chroma.added for i in batch[2]]
    # Each item now emits one-or-more chunk ids
    assert all(i.startswith("NEW__") for i in added_ids)
    saved = json.loads(open(config_path).read())
    assert saved["semantic_search"]["last_sync_version"] == 8


def test_update_database_incremental_deletes_removed_items(monkeypatch, tmp_path):
    """Items present in ChromaDB but removed from the library should be deleted."""
    config_path = _write_config(tmp_path, extra={"last_sync_version": 5})
    zot = FakeZoteroClient()
    # Only NEW is still in the library; DELETED_ME was removed
    zot.load_scenario([_paper("NEW")], fulltext={"NEW": {"content": "body"}}, library_version=9)
    zot.versions_state = {"NEW": 9}
    # Chunked-format preloads — NEW has 1 chunk, DELETED_ME had 2
    chroma = FakeChromaClient(preloaded_ids=["NEW__0", "DELETED_ME__0", "DELETED_ME__1"])
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    stats = search.update_database()

    # The deletion pass collapses chunks back to parent keys before diffing
    assert "DELETED_ME__0" in chroma.deleted
    assert "DELETED_ME__1" in chroma.deleted
    assert stats["deleted_items"] == 1  # one parent removed


def test_update_database_incremental_noop_when_version_unchanged(monkeypatch, tmp_path):
    """If last_modified_version == last_sync_version, skip ingest entirely."""
    config_path = _write_config(tmp_path, extra={"last_sync_version": 42})
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("X")], library_version=42)
    zot.versions_state = {"X": 42}
    chroma = FakeChromaClient(preloaded_ids=["X__0"])
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    stats = search.update_database()

    assert stats["processed_items"] == 0
    assert stats["added_items"] == 0
    assert stats["deleted_items"] == 0
    # No ingest calls (no items() fetch, no item_versions(since=...) etc.)
    # Must have called last_modified_version to compare
    assert ("last_modified_version",) in zot.calls


def test_update_database_disables_fulltext_when_config_off(monkeypatch, tmp_path):
    """Explicit include_fulltext=False should skip the web-API fulltext fetch."""
    config_path = _write_config(tmp_path, extra={"include_fulltext": False})
    zot = FakeZoteroClient()
    zot.load_scenario(
        [_paper("A")],
        fulltext={"A": {"content": "this should NOT appear"}},
        library_version=3,
    )
    chroma = FakeChromaClient()
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    search.update_database()

    # fulltext_item should never be called
    assert not any(c[0] == "fulltext_item" for c in zot.calls)


def test_update_database_gap_fills_missing_items(monkeypatch, tmp_path):
    """Resume case: ChromaDB is partially populated (e.g. killed mid-rebuild)
    but last_sync_version equals library version. The old noop fast-path
    would return "nothing to reindex"; the new diff-driven path detects
    items in the library that are missing from ChromaDB and fetches them."""
    config_path = _write_config(tmp_path, extra={"last_sync_version": 100})
    zot = FakeZoteroClient()
    # Three items in library, all unchanged since version 100
    zot.load_scenario(
        [_paper("A"), _paper("B"), _paper("C")],
        fulltext={"A": {"content": "Abody"}, "B": {"content": "Bbody"}, "C": {"content": "Cbody"}},
        library_version=100,
    )
    zot.versions_state = {"A": 100, "B": 100, "C": 100}
    # Only A was indexed before the interruption
    chroma = FakeChromaClient(preloaded_ids=["A__0", "A__1"])
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    stats = search.update_database()

    # B and C should have been gap-filled
    assert stats["gap_filled_items"] == 2
    # All gap items should be in ChromaDB now
    added_ids = [i for batch in chroma.added for i in batch[2]]
    added_parents = {i.split("__", 1)[0] for i in added_ids}
    assert added_parents == {"B", "C"}
    # A was left untouched
    assert "A" not in added_parents
    # Watermark advances to current library version
    saved = json.loads(open(config_path).read())
    assert saved["semantic_search"]["last_sync_version"] == 100


def test_update_database_gap_fill_combined_with_since_changes(monkeypatch, tmp_path):
    """Gap-fill + since-V changes should both land in the same run."""
    config_path = _write_config(tmp_path, extra={"last_sync_version": 50})
    zot = FakeZoteroClient()
    zot.load_scenario(
        [_paper("GAP"), _paper("CHANGED")],
        fulltext={"GAP": {"content": "Gbody"}, "CHANGED": {"content": "Cbody"}},
        library_version=80,
    )
    # CHANGED version 75 > 50, GAP version 40 < 50 but absent from ChromaDB
    zot.versions_state = {"GAP": 40, "CHANGED": 75}
    chroma = FakeChromaClient(preloaded_ids=[])  # completely empty
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    stats = search.update_database()

    added_parents = {i.split("__", 1)[0]
                     for batch in chroma.added for i in batch[2]}
    assert added_parents == {"GAP", "CHANGED"}
    assert stats["gap_filled_items"] == 1


def test_update_database_true_noop_when_fully_synced(monkeypatch, tmp_path):
    """Library unchanged AND ChromaDB has every library item => truly nothing to do."""
    config_path = _write_config(tmp_path, extra={"last_sync_version": 99})
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("X")], library_version=99)
    zot.versions_state = {"X": 99}
    chroma = FakeChromaClient(preloaded_ids=["X__0", "X__1"])
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    stats = search.update_database()

    assert stats["processed_items"] == 0
    assert stats["gap_filled_items"] == 0
    assert stats["deleted_items"] == 0
    # No items should have been re-upserted
    assert chroma.added == []


def test_fetch_items_by_keys_batches_requests(monkeypatch):
    """Verifies itemKey= is used and the batch size of 50 is respected."""
    zot = FakeZoteroClient()
    # Preload 120 items
    items = [_paper(f"K{i:03d}") for i in range(120)]
    zot.load_scenario(items, library_version=1)

    # Replace items() to record how itemKey param was passed
    captured_keys: list[list[str]] = []
    original_items = zot.items

    def recording_items(start=0, limit=100, **kwargs):
        # pyzotero's add_parameters stores url_params; fake it here
        captured_keys.append([k for k in getattr(zot, "_fake_itemkey", "").split(",") if k])
        return original_items(start=0, limit=len(captured_keys[-1]) or limit)

    def fake_add_parameters(**params):
        zot._fake_itemkey = params.get("itemKey", "")

    zot.add_parameters = fake_add_parameters
    zot.items = recording_items

    search = _build_search(monkeypatch, zot, FakeChromaClient())
    keys_to_fetch = [f"K{i:03d}" for i in range(120)]
    result = search._fetch_items_by_keys(keys_to_fetch)

    # 120 keys -> 3 batches of 50 (50 + 50 + 20)
    assert len(captured_keys) == 3
    assert [len(b) for b in captured_keys] == [50, 50, 20]
    assert len(result) > 0


def test_fetch_items_by_keys_falls_back_to_per_key_on_batch_error(monkeypatch):
    """If the batched itemKey request raises, each key is re-tried individually
    so one malformed key can't stall the whole backfill."""
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("X"), _paper("Y")], library_version=1)

    def boom_items(**kwargs):
        raise RuntimeError("simulated batch failure")

    zot.items = boom_items
    zot.add_parameters = lambda **kw: None

    search = _build_search(monkeypatch, zot, FakeChromaClient())
    result = search._fetch_items_by_keys(["X", "Y"])
    # Fallback uses zot.item(key) per key, which FakeZoteroClient supports
    returned_keys = {it["key"] for it in result}
    assert returned_keys == {"X", "Y"}


def test_update_database_force_rebuild_triggers_reset_and_full_scan(monkeypatch, tmp_path):
    """force_full_rebuild should reset the collection and do a full scan."""
    config_path = _write_config(tmp_path, extra={"last_sync_version": 100})
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("A"), _paper("B")], library_version=120)
    zot.versions_state = {"A": 120, "B": 120}
    chroma = FakeChromaClient(preloaded_ids=["STALE"])
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    search.update_database(force_full_rebuild=True)

    assert chroma.reset_calls == 1
    # No since-based fetch: full scan used items() not item_versions(since=...)
    assert not any(c[0] == "item_versions" and c[1] is not None for c in zot.calls)


def test_update_database_force_rebuild_updates_last_sync_version(monkeypatch, tmp_path):
    """After force_full_rebuild, last_sync_version must advance to the library's
    current version. Otherwise the next incremental run would use a stale
    watermark and permanently lose items not modified since that older version.
    """
    config_path = _write_config(tmp_path, extra={"last_sync_version": 50})
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("A")], library_version=200)
    zot.versions_state = {"A": 200}
    chroma = FakeChromaClient()
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    search.update_database(force_full_rebuild=True)

    saved = json.loads(open(config_path).read())
    assert saved["semantic_search"]["last_sync_version"] == 200


# --------- Config loaders ----------

def test_load_include_fulltext_defaults_true(monkeypatch, tmp_path):
    # No config file exists
    search = _build_search(monkeypatch, FakeZoteroClient(), FakeChromaClient(),
                           config_path=str(tmp_path / "missing.json"))
    assert search._load_include_fulltext_setting() is True


def test_load_include_fulltext_respects_opt_out(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, extra={"include_fulltext": False})
    search = _build_search(monkeypatch, FakeZoteroClient(), FakeChromaClient(),
                           config_path=config_path)
    assert search._load_include_fulltext_setting() is False


def test_load_last_sync_version_defaults_zero(monkeypatch, tmp_path):
    search = _build_search(monkeypatch, FakeZoteroClient(), FakeChromaClient(),
                           config_path=str(tmp_path / "missing.json"))
    assert search._load_last_sync_version() == 0


def test_load_last_sync_version_reads_int(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, extra={"last_sync_version": 123})
    search = _build_search(monkeypatch, FakeZoteroClient(), FakeChromaClient(),
                           config_path=config_path)
    assert search._load_last_sync_version() == 123
