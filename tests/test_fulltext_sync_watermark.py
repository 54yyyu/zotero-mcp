"""Tests for issue #292: `update-db --fulltext` must not advance the sync
watermark past items the local sqlite snapshot never saw.

The local-extraction path reads zotero.sqlite with `immutable=1`, which
cannot see rows that are still in Zotero's WAL. The API-derived
`last_sync_version` watermark, however, *does* cover those rows, so
promoting it unconditionally makes later incremental updates skip the
missed items forever.
"""

import json
import sqlite3

from zotero_mcp.local_db import LocalZoteroReader
from zotero_mcp.semantic_search import ZoteroSemanticSearch


def make_zotero_db(path, keys):
    """Create a minimal zotero.sqlite with the given item keys."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT)"
    )
    conn.execute("INSERT INTO itemTypes VALUES (1, 'journalArticle')")
    conn.execute(
        """CREATE TABLE items (
            itemID INTEGER PRIMARY KEY, itemTypeID INT, dateAdded TEXT,
            dateModified TEXT, clientDateModified TEXT, libraryID INT,
            key TEXT UNIQUE, version INT, synced INT
        )"""
    )
    for i, key in enumerate(keys, start=1):
        conn.execute(
            "INSERT INTO items VALUES (?, 1, '2026-01-01 00:00:00', "
            "'2026-01-01 00:00:00', '2026-01-01 00:00:00', 1, ?, 1, 0)",
            (i, key),
        )
    # Empty side tables referenced by get_items_with_text / get_item_count
    conn.execute("CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY)")
    conn.execute(
        "CREATE TABLE itemData (itemID INT, fieldID INT, valueID INT)"
    )
    conn.execute(
        "CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT)"
    )
    conn.execute(
        "CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT)"
    )
    conn.execute(
        "CREATE TABLE itemNotes (itemID INT, parentItemID INT, note TEXT)"
    )
    conn.execute("CREATE TABLE itemCreators (itemID INT, creatorID INT)")
    conn.execute(
        "CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, "
        "firstName TEXT, lastName TEXT)"
    )
    conn.commit()
    conn.close()


class FakeVersionsZotero:
    """pyzotero stub exposing item_versions / last_modified_version."""

    def __init__(self, versions, error=None, library_version=12):
        self._versions = versions
        self._error = error
        self._library_version = library_version

    def item_versions(self, **kwargs):
        if self._error:
            raise self._error
        return dict(self._versions)

    def last_modified_version(self):
        return self._library_version


class FakeChroma:
    """ChromaClient stub for update_database runs with no items."""

    def reset_collection(self):
        raise AssertionError("reset_collection should not be called")

    def get_all_ids(self):
        return set()


def make_search(db_path, zotero, config_path=None):
    """Build a ZoteroSemanticSearch without touching Chroma or pyzotero."""
    s = object.__new__(ZoteroSemanticSearch)
    s.zotero_client = zotero
    s.db_path = str(db_path)
    s.config_path = str(config_path) if config_path else None
    s.chroma_client = FakeChroma()
    s.update_config = {"auto_update": False, "update_frequency": "manual"}
    return s


def test_get_all_item_keys_returns_every_key(tmp_path):
    db = tmp_path / "zotero.sqlite"
    make_zotero_db(db, ["AAAA1111", "BBBB2222", "CCCC3333"])
    with LocalZoteroReader(db_path=str(db)) as reader:
        assert reader.get_all_item_keys() == {"AAAA1111", "BBBB2222", "CCCC3333"}


def test_watermark_promoted_when_snapshot_complete(tmp_path):
    db = tmp_path / "zotero.sqlite"
    make_zotero_db(db, ["AAAA1111", "BBBB2222"])
    s = make_search(db, FakeVersionsZotero({"AAAA1111": 5, "BBBB2222": 7}))
    assert s._verify_local_snapshot_version(42) == 42


def test_watermark_held_back_when_api_sees_unseen_item(tmp_path):
    """An item visible via the API but absent from the sqlite snapshot
    (e.g. still in the WAL) must block watermark promotion."""
    db = tmp_path / "zotero.sqlite"
    make_zotero_db(db, ["AAAA1111"])
    s = make_search(
        db, FakeVersionsZotero({"AAAA1111": 5, "WALHIDDEN1": 12})
    )
    assert s._verify_local_snapshot_version(42) is None


def test_watermark_held_back_on_api_error(tmp_path):
    db = tmp_path / "zotero.sqlite"
    make_zotero_db(db, ["AAAA1111"])
    s = make_search(
        db, FakeVersionsZotero({}, error=RuntimeError("api down"))
    )
    assert s._verify_local_snapshot_version(42) is None


def test_watermark_uses_scan_time_snapshot_keys(tmp_path):
    """A checkpoint can land *during* the (minutes-long) extraction scan,
    making a fresh sqlite read look complete even though the scan itself
    missed the item. Verification must compare against the keys captured
    at scan time, not a fresh snapshot."""
    db = tmp_path / "zotero.sqlite"
    # Disk state AFTER the mid-scan checkpoint: WALHIDDEN1 is now visible
    make_zotero_db(db, ["AAAA1111", "WALHIDDEN1"])
    s = make_search(
        db, FakeVersionsZotero({"AAAA1111": 5, "WALHIDDEN1": 12})
    )
    # ...but the scan only ever saw AAAA1111
    s._last_scan_snapshot_keys = {"AAAA1111"}
    assert s._verify_local_snapshot_version(42) is None


def test_get_items_from_local_db_captures_snapshot_keys(tmp_path):
    """The metadata scan must record the key set its own connection saw,
    so verification later compares against the same snapshot."""
    db = tmp_path / "zotero.sqlite"
    make_zotero_db(db, ["AAAA1111", "BBBB2222"])
    s = make_search(db, FakeVersionsZotero({}))
    s._get_items_from_local_db(extract_fulltext=False)
    assert s._last_scan_snapshot_keys == {"AAAA1111", "BBBB2222"}


def _write_config(path, last_sync_version):
    path.write_text(
        json.dumps(
            {
                "semantic_search": {
                    "update_config": {
                        "auto_update": False,
                        "update_frequency": "manual",
                    },
                    "last_sync_version": last_sync_version,
                }
            }
        )
    )


def test_update_database_holds_watermark_when_snapshot_stale(
    tmp_path, monkeypatch
):
    """End-to-end through update_database: a WAL-hidden item must keep
    last_sync_version at its previous value."""
    db = tmp_path / "zotero.sqlite"
    make_zotero_db(db, ["AAAA1111"])
    config = tmp_path / "config.json"
    _write_config(config, last_sync_version=10)

    s = make_search(
        db,
        FakeVersionsZotero(
            {"AAAA1111": 5, "WALHIDDEN1": 12}, library_version=12
        ),
        config_path=config,
    )
    monkeypatch.setattr(s, "_get_items_from_source", lambda **kw: [])

    s.update_database(extract_fulltext=True, include_fulltext=False)

    saved = json.loads(config.read_text())
    assert saved["semantic_search"]["last_sync_version"] == 10


def test_update_database_promotes_watermark_when_snapshot_complete(
    tmp_path, monkeypatch
):
    db = tmp_path / "zotero.sqlite"
    make_zotero_db(db, ["AAAA1111"])
    config = tmp_path / "config.json"
    _write_config(config, last_sync_version=10)

    s = make_search(
        db,
        FakeVersionsZotero({"AAAA1111": 5}, library_version=12),
        config_path=config,
    )
    monkeypatch.setattr(s, "_get_items_from_source", lambda **kw: [])

    s.update_database(extract_fulltext=True, include_fulltext=False)

    saved = json.loads(config.read_text())
    assert saved["semantic_search"]["last_sync_version"] == 12
