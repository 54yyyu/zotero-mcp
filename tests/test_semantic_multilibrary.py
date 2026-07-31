"""Tests for group_id into ChromaDB metadata + DB-side filtering (#163, phase 1).

Covers: metadata tagging (local scan + web-API scan), feed exclusion,
metadata-only migration backfill (evidence-based attribution), and DB-side
`where` filtering in `search()`.

Related coverage elsewhere: per-library sync_versions (#393) in
test_sync_watermark_per_library.py; the library-scoped deletion pass (#404)
in test_library_scoped_deletion.py. Still deferred upstream: cross-library
result enrichment (fetching a group hit's full item via a client scoped to
that group).
"""

import json
import logging
import sqlite3
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import client as zclient
from zotero_mcp import semantic_search

GROUP_ID = 6015547


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    """Every test in this file must be deterministic regardless of the host
    shell's Zotero env vars, and must not leak the process-wide
    active-library override across tests."""
    monkeypatch.delenv("ZOTERO_LOCAL", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_TYPE", raising=False)
    zclient.clear_active_library()
    yield
    zclient.clear_active_library()


class _StubZot:
    """Minimal pyzotero-shaped client."""

    def __init__(self, items_by_key=None):
        self._items = items_by_key or {}

    def item(self, key):
        if key not in self._items:
            raise LookupError(f"item {key} not found")
        return self._items[key]


class _FakeChromaClient:
    """ChromaClient stand-in tracking metadata per doc id."""

    def __init__(self, docs: dict | None = None):
        self.embedding_max_tokens = 8000
        self._docs: dict[str, dict] = {k: dict(v) for k, v in (docs or {}).items()}
        self.deleted: list[str] = []
        self.reset_calls = 0
        self.update_calls: list[tuple[list[str], list[dict]]] = []
        self.last_search_where = "UNSET"

    def truncate_text(self, text, max_tokens=None):
        return text

    def get_existing_ids(self, ids):
        return {i for i in ids if i in self._docs}

    def get_all_ids(self, where=None):
        if where and "group_id" in where:
            return {
                i for i, m in self._docs.items()
                if m.get("group_id") == where["group_id"]
            }
        return set(self._docs)

    def get_document_metadata(self, doc_id):
        return self._docs.get(doc_id)

    def upsert_documents(self, documents, metadatas, ids):
        for i, m in zip(ids, metadatas):
            self._docs[i] = dict(m)

    def delete_documents(self, ids):
        for i in ids:
            self._docs.pop(i, None)
        self.deleted.extend(ids)

    def reset_collection(self):
        self.reset_calls += 1
        self._docs = {}

    def iter_metadatas(self, batch_size=500):
        ids = list(self._docs.keys())
        if ids:
            yield ids, [self._docs[i] for i in ids]

    def update_metadatas(self, ids, metadatas):
        self.update_calls.append((list(ids), [dict(m) for m in metadatas]))
        for i, m in zip(ids, metadatas):
            self._docs.setdefault(i, {}).update(m)

    def search(self, query_texts, n_results, where=None, where_document=None):
        self.last_search_where = where
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


def _build_search(monkeypatch, chroma, *, config_path=None, get_zotero_client_fn=None, is_local=False):
    if get_zotero_client_fn is None:
        get_zotero_client_fn = lambda: _StubZot()  # noqa: E731
    monkeypatch.setattr(semantic_search, "get_zotero_client", get_zotero_client_fn)
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: is_local)
    return semantic_search.ZoteroSemanticSearch(chroma_client=chroma, config_path=config_path)


# ---------------------------------------------------------------------------
# _create_metadata tagging
# ---------------------------------------------------------------------------

def test_create_metadata_includes_group_id_from_data(monkeypatch):
    search = _build_search(monkeypatch, _FakeChromaClient())
    item = {"key": "K1", "data": {"title": "T", "group_id": GROUP_ID}}
    meta = search._create_metadata(item)
    assert meta["group_id"] == GROUP_ID


def test_create_metadata_defaults_group_id_to_personal_when_missing(monkeypatch):
    search = _build_search(monkeypatch, _FakeChromaClient())
    item = {"key": "K1", "data": {"title": "T"}}
    meta = search._create_metadata(item)
    assert meta["group_id"] == 0


# ---------------------------------------------------------------------------
# _tag_group_id (web-API scan attribution)
# ---------------------------------------------------------------------------

def test_tag_group_id_defaults_to_personal(monkeypatch):
    search = _build_search(monkeypatch, _FakeChromaClient())
    items = [{"key": "A", "data": {}}]
    search._tag_group_id(items)
    assert items[0]["data"]["group_id"] == 0


def test_tag_group_id_uses_active_group_override(monkeypatch):
    search = _build_search(monkeypatch, _FakeChromaClient())
    zclient.set_active_library(library_id=str(GROUP_ID), library_type="group")
    items = [{"key": "A", "data": {}}]
    search._tag_group_id(items)
    assert items[0]["data"]["group_id"] == GROUP_ID


def test_get_items_from_api_tags_active_group(monkeypatch):
    class _ItemsZot(_StubZot):
        def items(self, start=0, limit=100, **kw):
            if start > 0:
                return []
            return [{"key": "A", "data": {"itemType": "journalArticle", "title": "T"}}]

    search = _build_search(
        monkeypatch, _FakeChromaClient(),
        get_zotero_client_fn=lambda: _ItemsZot(),
    )
    search.zotero_client = _ItemsZot()
    zclient.set_active_library(library_id=str(GROUP_ID), library_type="group")

    items = search._get_items_from_api()

    assert len(items) == 1
    assert items[0]["data"]["group_id"] == GROUP_ID


def test_get_changed_items_from_api_tags_active_group(monkeypatch):
    class _ChangedZot(_StubZot):
        def item_versions(self, since=None, **kw):
            return {"A": 9}

        def item(self, key):
            return {"key": key, "data": {"itemType": "journalArticle", "title": "T"}}

    search = _build_search(monkeypatch, _FakeChromaClient())
    search.zotero_client = _ChangedZot()
    zclient.set_active_library(library_id=str(GROUP_ID), library_type="group")

    changed_items, current_keys = search._get_changed_items_from_api(since_version=5)

    assert len(changed_items) == 1
    assert changed_items[0]["data"]["group_id"] == GROUP_ID
    assert current_keys == {"A"}


# ---------------------------------------------------------------------------
# Local-scan attribution + feed exclusion (LocalZoteroReader.get_key_group_map)
# ---------------------------------------------------------------------------

class _MapReader:
    """LocalZoteroReader double returning a fixed key→group map."""

    def __init__(self, groups, excluded=()):
        self._groups = dict(groups)
        self._excluded = set(excluded)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_key_group_map(self):
        return (self._groups, self._excluded)


def _build_multilib_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE libraries (libraryID INTEGER PRIMARY KEY, type TEXT, editable INT, filesEditable INT);
        CREATE TABLE groups (groupID INTEGER PRIMARY KEY, libraryID INT UNIQUE, name TEXT, description TEXT, version INT);
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT, itemTypeID INT, libraryID INT, dateAdded TEXT, dateModified TEXT);
        CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
        CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
        CREATE TABLE itemData (itemID INT, fieldID INT, valueID INT);
        CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE itemNotes (itemID INT, parentItemID INT, note TEXT);
        CREATE TABLE itemCreators (itemID INT, creatorID INT);
        CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT);
        CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY);
        """
    )
    conn.execute("INSERT INTO libraries VALUES (1, 'user', 1, 1)")
    conn.execute("INSERT INTO libraries VALUES (5, 'group', 1, 1)")
    conn.execute("INSERT INTO libraries VALUES (10, 'feed', 0, 0)")
    conn.execute(f"INSERT INTO groups VALUES ({GROUP_ID}, 5, 'AI in entrepreneurship', '', 1)")
    conn.execute("INSERT INTO itemTypes VALUES (1, 'journalArticle')")

    def add_item(item_id, key, lib_id, title):
        conn.execute(
            "INSERT INTO items (itemID, key, itemTypeID, libraryID, dateAdded, dateModified) "
            "VALUES (?, ?, 1, ?, '2026-01-01 00:00:00', '2026-01-01 00:00:00')",
            (item_id, key, lib_id),
        )
        conn.execute("INSERT INTO itemDataValues (valueID, value) VALUES (?, ?)", (item_id, title))
        conn.execute("INSERT INTO itemData (itemID, fieldID, valueID) VALUES (?, 1, ?)", (item_id, item_id))

    add_item(1, "PERSONAL1", 1, "Personal Paper")
    add_item(2, "GROUPITEM1", 5, "Group Paper")
    add_item(3, "FEEDITEM1", 10, "Feed Paper")
    add_item(4, "TRASHED1", 5, "Trashed Group Paper")
    conn.execute("INSERT INTO deletedItems VALUES (4)")
    conn.commit()
    conn.close()


def test_local_db_scan_tags_group_id_and_excludes_feeds(monkeypatch, tmp_path):
    db_path = tmp_path / "zotero.sqlite"
    _build_multilib_db(db_path)
    search = _build_search(monkeypatch, _FakeChromaClient(), is_local=True)
    search.db_path = str(db_path)

    items = search._get_items_from_local_db(extract_fulltext=False)

    by_key = {it["key"]: it for it in items}
    assert set(by_key) == {"PERSONAL1", "GROUPITEM1"}, "feed item must be excluded"
    assert by_key["PERSONAL1"]["data"]["group_id"] == 0
    assert by_key["GROUPITEM1"]["data"]["group_id"] == GROUP_ID


# ---------------------------------------------------------------------------
# Migration backfill (metadata-only, idempotent, evidence-based)
#
# Attribution must be evidence-based because the deletion pass is scoped by
# group_id: a guessed tag is a future deletion warrant. Evidence is, in
# order: the local database's key→group map (local mode), then membership
# in the active library's item_versions(). No evidence → left untagged,
# which structurally excludes the doc from deletion.
# ---------------------------------------------------------------------------

def test_backfill_web_mode_tags_only_keys_evidenced_in_active_library(monkeypatch):
    """The old behavior tagged every unknown doc with the active library's
    id — which mis-attributes every other library's docs in a pre-#163
    multi-library collection, and mis-attribution authorizes deletion once
    the deletion pass is group_id-scoped."""
    chroma = _FakeChromaClient({
        "KEY1": {"item_key": "KEY1", "title": "T"},
        "OTHERLIB": {"item_key": "OTHERLIB", "title": "T2"},
    })

    class _VersionsZot(_StubZot):
        def item_versions(self, **kw):
            return {"KEY1": 7}

    search = _build_search(
        monkeypatch, chroma, is_local=False, get_zotero_client_fn=lambda: _VersionsZot()
    )

    stats = search._backfill_group_ids()
    assert stats == {"scanned": 2, "migrated": 1, "unattributed": 1}
    assert chroma._docs["KEY1"]["group_id"] == 0
    assert "group_id" not in chroma._docs["OTHERLIB"]
    # Document text/embedding untouched — only update_metadatas was called.
    assert chroma.update_calls == [(["KEY1"], [{"item_key": "KEY1", "title": "T", "group_id": 0}])]

    # Idempotent: nothing newly migrated on a second run; the unevidenced
    # doc is still counted, never guessed at.
    stats2 = search._backfill_group_ids()
    assert stats2 == {"scanned": 2, "migrated": 0, "unattributed": 1}


def test_backfill_group_ids_local_mode_uses_key_group_map(monkeypatch):
    chroma = _FakeChromaClient({
        "GROUPKEY": {"item_key": "GROUPKEY", "title": "T"},
        "USERKEY": {"item_key": "USERKEY", "title": "T2"},
    })
    search = _build_search(monkeypatch, chroma, is_local=True)

    monkeypatch.setattr(
        semantic_search, "LocalZoteroReader",
        lambda **kw: _MapReader({"GROUPKEY": GROUP_ID, "USERKEY": 0}),
    )

    stats = search._backfill_group_ids()
    assert stats["migrated"] == 2
    assert chroma._docs["GROUPKEY"]["group_id"] == GROUP_ID
    assert chroma._docs["USERKEY"]["group_id"] == 0


def test_backfill_local_mode_unknown_key_uses_item_versions_membership(monkeypatch):
    """A key missing from the local map may be WAL-fresh (invisible to the
    immutable sqlite read, #292) — the live local API is the tie-breaker.
    A key with no evidence anywhere stays untagged."""
    chroma = _FakeChromaClient({
        "MAPPED": {"item_key": "MAPPED"},
        "WALFRESH": {"item_key": "WALFRESH"},
        "GONE": {"item_key": "GONE"},
    })

    class _VersionsZot(_StubZot):
        def item_versions(self, **kw):
            return {"MAPPED": 3, "WALFRESH": 9}

    search = _build_search(
        monkeypatch, chroma, is_local=True, get_zotero_client_fn=lambda: _VersionsZot()
    )
    monkeypatch.setattr(
        semantic_search, "LocalZoteroReader", lambda **kw: _MapReader({"MAPPED": GROUP_ID})
    )

    stats = search._backfill_group_ids()

    assert stats == {"scanned": 3, "migrated": 2, "unattributed": 1}
    assert chroma._docs["MAPPED"]["group_id"] == GROUP_ID
    assert chroma._docs["WALFRESH"]["group_id"] == 0  # active library = personal
    assert "group_id" not in chroma._docs["GONE"]


def test_backfill_does_not_fetch_item_versions_when_map_covers_everything(monkeypatch):
    """The API call is evidence of last resort, fetched lazily — a run whose
    local map attributes every doc must not touch the API at all."""
    chroma = _FakeChromaClient({"GROUPKEY": {"item_key": "GROUPKEY"}})

    class _ExplodingZot(_StubZot):
        def item_versions(self, **kw):
            raise AssertionError("item_versions must not be called")

    search = _build_search(
        monkeypatch, chroma, is_local=True, get_zotero_client_fn=lambda: _ExplodingZot()
    )
    monkeypatch.setattr(
        semantic_search, "LocalZoteroReader", lambda **kw: _MapReader({"GROUPKEY": GROUP_ID})
    )

    stats = search._backfill_group_ids()
    assert stats == {"scanned": 1, "migrated": 1, "unattributed": 0}


def test_backfill_empty_item_versions_is_treated_as_a_fault(monkeypatch):
    """An HTTP-200-but-empty item_versions() body is the same response shape
    the deletion pass refuses to act on. Accepting it here as negative
    evidence would mark the one-time migration complete with every doc
    unattributed — permanently, since the schema-version gate then closes."""
    chroma = _FakeChromaClient({"KEY1": {"item_key": "KEY1"}})

    class _EmptyZot(_StubZot):
        def item_versions(self, **kw):
            return {}

    search = _build_search(
        monkeypatch, chroma, is_local=False, get_zotero_client_fn=lambda: _EmptyZot()
    )

    with pytest.raises(Exception, match="no items"):
        search._backfill_group_ids()
    assert "group_id" not in chroma._docs["KEY1"]


def test_backfill_item_versions_failure_propagates_and_tags_nothing(monkeypatch):
    """No evidence available → raise (update_database catches and retries
    next run). Nothing may be tagged by guesswork on the way out."""
    chroma = _FakeChromaClient({"KEY1": {"item_key": "KEY1"}})

    class _DownZot(_StubZot):
        def item_versions(self, **kw):
            raise RuntimeError("api down")

    search = _build_search(
        monkeypatch, chroma, is_local=False, get_zotero_client_fn=lambda: _DownZot()
    )

    with pytest.raises(Exception, match="api down"):
        search._backfill_group_ids()
    assert "group_id" not in chroma._docs["KEY1"]


def test_backfill_group_ids_skips_already_tagged_docs(monkeypatch):
    chroma = _FakeChromaClient({"KEY1": {"item_key": "KEY1", "group_id": GROUP_ID}})
    search = _build_search(monkeypatch, chroma, is_local=False)

    stats = search._backfill_group_ids()
    assert stats == {"scanned": 1, "migrated": 0, "unattributed": 0}
    assert chroma.update_calls == []
    assert chroma._docs["KEY1"]["group_id"] == GROUP_ID  # untouched


def test_key_group_map_includes_trashed_items_with_their_library(tmp_path):
    """Trashed items keep their true library's gid in the map, so the
    backfill attributes trash honestly and each library's own scoped
    deletion pass cleans it — instead of the old blanket fallback tagging
    it with whatever library happened to be active."""
    from zotero_mcp.local_db import LocalZoteroReader

    db_path = tmp_path / "zotero.sqlite"
    _build_multilib_db(db_path)

    with LocalZoteroReader(db_path=str(db_path)) as reader:
        groups, _excluded = reader.get_key_group_map()

    assert groups["TRASHED1"] == GROUP_ID
    assert groups["PERSONAL1"] == 0


# ---------------------------------------------------------------------------
# update_database integration: backfill failure message + unattributed count
# ---------------------------------------------------------------------------

class _UpdateZot(_StubZot):
    """items()/versions-capable stub for full update_database runs."""

    def __init__(self, keys=()):
        super().__init__()
        self._keys = list(keys)

    def items(self, start=0, limit=100, **kw):
        if start:
            return []
        return [
            {"key": k, "version": 1, "data": {
                "key": k, "itemType": "journalArticle", "title": k,
                "dateAdded": "2024-01-01T00:00:00Z", "dateModified": "2024-01-01T00:00:00Z",
            }} for k in self._keys
        ]

    def item_versions(self, since=None, **kw):
        return {k: 5 for k in self._keys}

    def last_modified_version(self, **kw):
        return 5

    def children(self, *a, **kw):
        return []


def _write_update_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "semantic_search": {
            "embedding_model": "default",
            "update_config": {"auto_update": False, "update_frequency": "manual"},
            "include_fulltext": False,
        }
    }))
    return str(config_path)


def test_backfill_failure_log_does_not_recommend_force_rebuild(monkeypatch, tmp_path, caplog):
    """--force-rebuild permanently drops any doc population the current
    ingest paths cannot recreate and re-embeds the whole collection;
    recommending it as the repair for a failed metadata backfill turns a
    metadata bug into data loss."""
    config_path = _write_update_config(tmp_path)
    chroma = _FakeChromaClient()
    search = _build_search(monkeypatch, chroma, config_path=config_path,
                           get_zotero_client_fn=lambda: _UpdateZot())

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(search, "_backfill_group_ids", _boom)

    with caplog.at_level(logging.ERROR, logger="zotero_mcp.semantic_search"):
        search.update_database()

    failures = [r.message for r in caplog.records if "backfill failed" in r.message]
    assert failures, "expected the backfill failure to be logged"
    assert all("force-rebuild" not in m and "force_rebuild" not in m for m in failures)
    # A failed backfill must not close the one-time-migration gate.
    saved = json.loads(open(config_path).read())["semantic_search"]
    assert "index_schema_version" not in saved


def test_unattributed_count_is_persisted_and_rewarned_on_later_updates(monkeypatch, tmp_path, caplog):
    """Docs with no attribution evidence are excluded from library-filtered
    search and from deletion cleanup. That state must stay visible on every
    update, not be a one-time log line."""
    config_path = _write_update_config(tmp_path)
    chroma = _FakeChromaClient({"GHOST": {"item_key": "GHOST", "title": "gone"}})
    search = _build_search(monkeypatch, chroma, config_path=config_path,
                           get_zotero_client_fn=lambda: _UpdateZot(["KEY1"]))

    search.update_database()

    saved = json.loads(open(config_path).read())["semantic_search"]
    assert saved["backfill_unattributed"] == 1
    assert saved["index_schema_version"] == 3

    with caplog.at_level(logging.WARNING, logger="zotero_mcp.semantic_search"):
        search.update_database()
    warned = [r.message for r in caplog.records if "attribut" in r.message]
    assert warned, "later updates must keep warning about unattributed docs"


# ---------------------------------------------------------------------------
# search(): DB-side `where` filtering
# ---------------------------------------------------------------------------

def test_search_no_group_id_means_no_library_filter(monkeypatch):
    chroma = _FakeChromaClient()
    search = _build_search(monkeypatch, chroma)
    search.search(query="q")
    assert chroma.last_search_where is None


def test_search_group_id_only_filter(monkeypatch):
    chroma = _FakeChromaClient()
    search = _build_search(monkeypatch, chroma)
    search.search(query="q", group_id=GROUP_ID)
    assert chroma.last_search_where == {"group_id": GROUP_ID}


def test_search_group_id_personal_filter_is_not_falsy_none(monkeypatch):
    """group_id=0 (personal library) must still apply a filter — 0 is a
    valid group_id, not 'no filter'."""
    chroma = _FakeChromaClient()
    search = _build_search(monkeypatch, chroma)
    search.search(query="q", group_id=0)
    assert chroma.last_search_where == {"group_id": 0}


def test_search_merges_group_id_with_user_filters(monkeypatch):
    chroma = _FakeChromaClient()
    search = _build_search(monkeypatch, chroma)
    search.search(query="q", filters={"item_type": "note"}, group_id=GROUP_ID)
    assert chroma.last_search_where == {"$and": [{"item_type": "note"}, {"group_id": GROUP_ID}]}
