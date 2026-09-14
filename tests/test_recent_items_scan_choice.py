"""The recent-items ranking query picks a table scan or the index per scope.

Profiled by @mronkko: for a scope that is most of `items`, walking the
(libraryID, key) index and fetching each row by rowid was 3x slower than a
full scan. For a small scope the index wins. Either plan must return exactly
the same items.
"""

import sqlite3

import pytest

import _search_corpus as corpus
from zotero_mcp.local_db import LocalZoteroReader

# Tables the full-record hydration reads that the search corpus leaves out.
_HYDRATION_SCHEMA = """
ALTER TABLE items ADD COLUMN version INTEGER NOT NULL DEFAULT 0;
ALTER TABLE itemNotes ADD COLUMN title TEXT;
CREATE TABLE itemAttachments (
    itemID INTEGER PRIMARY KEY, parentItemID INTEGER, linkMode INTEGER,
    contentType TEXT, path TEXT
);
CREATE TABLE itemAnnotations (
    itemID INTEGER PRIMARY KEY, parentItemID INTEGER, type INTEGER, text TEXT,
    comment TEXT, color TEXT, pageLabel TEXT, sortIndex TEXT, position TEXT
);
CREATE TABLE relationPredicates (predicateID INTEGER PRIMARY KEY, predicate TEXT);
CREATE TABLE itemRelations (itemID INTEGER, predicateID INTEGER, object TEXT);
"""


@pytest.fixture
def reader(tmp_path):
    db = tmp_path / "zotero.sqlite"
    corpus.build_sqlite(db)
    conn = sqlite3.connect(db)
    conn.executescript(_HYDRATION_SCHEMA)
    conn.close()
    r = LocalZoteroReader(db_path=str(db))
    yield r
    r.close()


@pytest.mark.parametrize("group_id", [0, corpus.GROUP_ID])
@pytest.mark.parametrize("sort", ["dateAdded", "dateModified", "title"])
@pytest.mark.parametrize("direction", ["desc", "asc"])
def test_scan_and_index_return_the_same_items(reader, monkeypatch, group_id, sort, direction):
    results = {}
    for choice in (True, False):
        monkeypatch.setattr(
            LocalZoteroReader, "_scope_prefers_scan", lambda self, conn, ids, c=choice: c
        )
        items = reader.get_recent_items(limit=5, sort=sort, direction=direction, group_id=group_id)
        results[choice] = [i["key"] for i in items]
    assert results[True] == results[False]


def test_large_scope_scans_small_scope_uses_the_index(reader):
    conn = reader._get_connection()
    personal = reader._resolve_scope_library_ids(0)
    group = reader._resolve_scope_library_ids(corpus.GROUP_ID)
    assert reader._scope_prefers_scan(conn, personal) is True
    assert reader._scope_prefers_scan(conn, group) is False


def test_share_is_counted_once_per_connection(reader, monkeypatch):
    conn = reader._get_connection()
    ids = reader._resolve_scope_library_ids(0)
    reader._scope_prefers_scan(conn, ids)
    monkeypatch.setattr(reader, "_scan_choice", {tuple(sorted(ids)): "cached"})
    assert reader._scope_prefers_scan(conn, ids) == "cached"
    reader.close()
    assert reader._scan_choice == {}
