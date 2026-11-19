"""zotero_get_recent lists top-level items on both backends: regular items and
standalone notes and attachments, never children.

The tool description has always said "Includes standalone notes and
attachments". The SQLite backend returned neither; the API backend listed
/items, where the child PDFs and notes of the last few papers crowd them out.
"""

import pytest

from _full_db import FullDb
from zotero_mcp.library import ApiBackend, SqliteBackend
from zotero_mcp.local_db import LocalZoteroReader


@pytest.fixture
def reader(tmp_path):
    db = FullDb(tmp_path / "zotero.sqlite")
    db.collection("COLL0001")
    paper = db.item("PAPER001", fields={"title": "Paper"}, added="2024-01-01 00:00:00",
                    collections=("COLL0001",))
    pdf = db.item("CHILDPDF", "attachment", parent=paper, added="2024-01-02 00:00:00")
    db.item("CHILDNOT", "note", parent=paper, note="child", added="2024-01-03 00:00:00")
    db.item("ANNOT001", "annotation", parent=pdf, added="2024-01-04 00:00:00")
    db.item("LONENOTE", "note", note="<p>standalone</p>", added="2024-01-05 00:00:00",
            collections=("COLL0001",))
    db.item("LONEPDF1", "attachment", added="2024-01-06 00:00:00")
    db.item("TRASHNOT", "note", note="gone", added="2024-01-07 00:00:00", deleted=True)
    with LocalZoteroReader(db_path=db.done()) as r:
        yield r


def test_sqlite_lists_top_level_items_newest_first(reader):
    recent = SqliteBackend(reader, 0).recent_items(limit=10)
    assert [item["key"] for item in recent] == ["LONEPDF1", "LONENOTE", "PAPER001"]
    assert recent[1]["data"]["note"] == "<p>standalone</p>"


def test_sqlite_collection_scope_keeps_the_standalone_note(reader):
    recent = SqliteBackend(reader, 0).recent_items(limit=10, collection_key="COLL0001")
    assert [item["key"] for item in recent] == ["LONENOTE", "PAPER001"]


@pytest.mark.parametrize("choice", [True, False])
def test_sqlite_scan_and_index_plans_agree(reader, monkeypatch, choice):
    monkeypatch.setattr(LocalZoteroReader, "_scope_prefers_scan", lambda *a: choice)
    recent = reader.get_recent_items(limit=10)
    assert [item["key"] for item in recent] == ["LONEPDF1", "LONENOTE", "PAPER001"]


class _Zot:
    def __init__(self):
        self.calls = []

    def top(self, **params):
        self.calls.append(("top", params))
        return [{"key": "PAPER001"}]

    def collection_items_top(self, key, **params):
        self.calls.append(("collection_items_top", key))
        return [{"key": "PAPER001"}]


def test_api_backend_asks_for_top_level_items():
    zot = _Zot()
    backend = ApiBackend(zot)
    assert backend.recent_items(limit=5) == [{"key": "PAPER001"}]
    assert backend.recent_items(limit=5, collection_key="COLL0001") == [{"key": "PAPER001"}]
    assert [call[0] for call in zot.calls] == ["top", "collection_items_top"]
    assert zot.calls[0][1]["sort"] == "dateAdded" and zot.calls[0][1]["direction"] == "desc"
