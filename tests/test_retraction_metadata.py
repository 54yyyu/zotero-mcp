"""Local metadata exposes Zotero's cached retraction state (#438).

The schema and flags follow Zotero's userdata.sql and Retractions.isRetracted:
https://github.com/zotero/zotero/blob/main/chrome/content/zotero/xpcom/retractions.js
No network or user library is needed.
"""

import json
import sqlite3

import _search_corpus as corpus
import pytest

from zotero_mcp.client import format_item_metadata
from zotero_mcp.library import SqliteBackend
from zotero_mcp.local_db import LocalZoteroReader

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
CREATE TABLE retractedItems (
    itemID INTEGER PRIMARY KEY, data TEXT, flag INT DEFAULT 0,
    FOREIGN KEY (itemID) REFERENCES items(itemID) ON DELETE CASCADE
);
"""


@pytest.fixture
def reader(tmp_path):
    db = tmp_path / "zotero.sqlite"
    corpus.build_sqlite(db)
    with sqlite3.connect(db) as conn:
        conn.executescript(_HYDRATION_SCHEMA)
        # The same key may exist in two libraries; retraction identity is itemID.
        for item_id, key, library_id, flag in (
            (101, "NORMAL01", 1, 0),
            (102, "HIDDEN01", 1, 1),
            (103, "NOWARN01", 1, 2),
            (104, "UNFLAG01", 1, None),
            (105, "NORMAL01", 5, None),
        ):
            conn.execute(
                "INSERT INTO items (itemID, key, itemTypeID, libraryID) VALUES (?, ?, 1, ?)",
                (item_id, key, library_id),
            )
            if flag is not None:
                conn.execute("INSERT INTO retractedItems (itemID, flag) VALUES (?, ?)", (item_id, flag))
    with LocalZoteroReader(db_path=str(db)) as result:
        yield result


@pytest.mark.parametrize(
    "key, expected", [("NORMAL01", True), ("HIDDEN01", False), ("NOWARN01", True), ("UNFLAG01", False)]
)
def test_sqlite_metadata_matches_zotero_retraction_flags(reader, key, expected):
    item = SqliteBackend(reader, group_id=0).get_item(key)
    assert item["data"]["retracted"] is expected
    assert ("**Retraction:**" in format_item_metadata(item)) is expected


def test_retraction_lookup_uses_item_identity_within_library(reader):
    personal = SqliteBackend(reader, group_id=0).get_item("NORMAL01")
    group = SqliteBackend(reader, group_id=corpus.GROUP_ID).get_item("NORMAL01")
    assert personal["data"]["retracted"] is True
    assert group["data"]["retracted"] is False


def test_missing_retraction_table_leaves_status_unknown(reader):
    db = reader.db_path
    reader.close()
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE retractedItems")
    item = SqliteBackend(reader, group_id=0).get_item("NORMAL01")
    assert "retracted" not in item["data"]
    assert "**Retraction:**" not in format_item_metadata(item)


def test_retraction_warning_does_not_depend_on_abstract_or_trash(reader):
    item = SqliteBackend(reader, group_id=0).get_item("NORMAL01")
    item["data"]["deleted"] = 1
    for include_abstract in (True, False):
        text = format_item_metadata(item, include_abstract=include_abstract)
        assert "**Retraction:**" in text
        assert "Zotero" in text
        assert "Trash" in text


def test_api_record_without_cached_status_has_no_invented_warning():
    item = {"key": "REMOTE01", "data": {"key": "REMOTE01", "itemType": "journalArticle", "title": "Remote"}}
    assert "**Retraction:**" not in format_item_metadata(item)
    assert "retracted" not in item["data"]


def test_metadata_tool_returns_local_status_in_json_and_markdown(reader, monkeypatch, dummy_ctx):
    from zotero_mcp import library
    from zotero_mcp.tools.retrieval import get_item_metadata

    backend = SqliteBackend(reader, group_id=0)
    monkeypatch.setattr(library, "get_library_backend", lambda: backend)
    result = get_item_metadata("NORMAL01", format="json", ctx=dummy_ctx)
    assert json.loads(result)["data"]["retracted"] is True
    result = get_item_metadata("NORMAL01", ctx=dummy_ctx)
    assert "**Retraction:**" in result


def test_batch_metadata_reads_retraction_flags_in_one_query(reader):
    conn = reader._get_connection()
    queries = []
    conn.set_trace_callback(queries.append)
    try:
        items = SqliteBackend(reader, group_id=0).get_items(["NORMAL01", "HIDDEN01", "NOWARN01", "UNFLAG01"])
    finally:
        conn.set_trace_callback(None)
    assert {key: item["data"]["retracted"] for key, item in items.items()} == {
        "NORMAL01": True,
        "HIDDEN01": False,
        "NOWARN01": True,
        "UNFLAG01": False,
    }
    reads = [query for query in queries if "FROM retractedItems" in query]
    assert len(reads) == 1
