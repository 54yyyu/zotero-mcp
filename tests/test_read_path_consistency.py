"""Small read-path agreements: one annotation-type map, one way to name an
item, one timestamp format, and a bounded citation-key scan on the API."""

import threading
from unittest.mock import Mock

import pytest

from _full_db import FullDb
from zotero_mcp import client as _client
from zotero_mcp import library
from zotero_mcp.library import ApiBackend, SqliteBackend
from zotero_mcp.local_db import LocalZoteroReader
from zotero_mcp.tools import retrieval


@pytest.fixture
def reader(tmp_path):
    db = FullDb(tmp_path / "zotero.sqlite")
    db.collection("COLL0001")
    case = db.item("CASE0001", "case", collections=("COLL0001",),
                   fields={"caseName": "Marbury v. Madison",
                           "dateDecided": "1803-02-24 February 24, 1803"},
                   added="2024-03-04 05:06:07")
    pdf = db.item("PDF00001", "attachment", parent=case)
    text_note = db.item("ANNOTXT1", "annotation", parent=pdf)
    db.conn.execute(
        "UPDATE itemAnnotations SET type = 6, text = NULL, comment = 'typed on the page' "
        "WHERE itemID = ?", (text_note,),
    )
    db.conn.execute("INSERT INTO relationPredicates VALUES (1, 'dc:relation')")
    paper = db.item("PAPER001", fields={"title": "Related paper"})
    db.conn.execute(
        "INSERT INTO itemRelations VALUES (?, 1, 'http://zotero.org/users/1/items/CASE0001')",
        (paper,),
    )
    db.base_field("case", "title", "caseName")
    db.base_field("case", "date", "dateDecided")
    with LocalZoteroReader(db_path=db.done()) as r:
        yield r


def test_text_annotations_are_named_in_annotation_search(reader):
    """Type 6 was missing from a second copy of the type map: "unknown"."""
    (hit,) = reader.search_annotations_local("typed on the page")
    assert hit["annotation_type"] == "text"


def test_search_projection_timestamps_match_the_full_record(reader):
    projected = reader.get_items_by_keys(["CASE0001"])["CASE0001"]["data"]
    full = reader.get_full_items(["CASE0001"])["CASE0001"]["data"]
    assert projected["dateAdded"] == full["dateAdded"] == "2024-03-04T05:06:07Z"


@pytest.fixture
def sqlite_tools(reader, monkeypatch):
    monkeypatch.setattr(
        library, "get_library_backend", lambda *a, **k: SqliteBackend(reader, 0)
    )


def test_listings_name_a_case_by_its_case_name(sqlite_tools):
    """Each of these read data["title"], which a case does not have."""
    keys_only = retrieval.get_collection_items("COLL0001", detail="keys_only", ctx=Mock())
    assert "`CASE0001` | Marbury v. Madison (February 24, 1803) [PDF]" in keys_only

    children = retrieval.get_item_children("CASE0001", ctx=Mock())
    assert children.startswith("# Child Items for: Marbury v. Madison")

    grouped = retrieval.get_item_children(["CASE0001", "PAPER001"], ctx=Mock())
    assert "## Marbury v. Madison (`CASE0001`)" in grouped

    related = retrieval.get_item_related("PAPER001", ctx=Mock())
    assert "`CASE0001` — **Marbury v. Madison** (February 24, 1803)" in related


def test_formatting_an_attachment_summary_does_not_take_the_api_lock(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_LOCK_TIMEOUT", "0.05")
    held, release = threading.Event(), threading.Event()

    def hold():
        with _client._zotero_api_lock:
            held.set()
            release.wait(5)

    thread = threading.Thread(target=hold)
    thread.start()
    held.wait(5)
    try:
        assert retrieval._build_attachment_extra({"has_pdf": True}) == {"Attachments": "PDF"}
    finally:
        release.set()
        thread.join()


class _EndlessZot:
    """A library that never contains the key and never runs out of pages."""

    def __init__(self, pages_before_hit=None):
        self.pages = 0
        self.pages_before_hit = pages_before_hit

    def add_parameters(self, **params):
        pass

    def items(self, **params):
        return []

    def top(self, start=0, limit=100, **params):
        self.pages += 1
        batch = [{"key": f"K{start + n}", "data": {"extra": ""}} for n in range(limit)]
        if self.pages == self.pages_before_hit:
            batch[-1] = {"key": "FOUND001", "data": {"citationKey": "smith2020"}}
        return batch


def test_api_citation_key_scan_stops_at_its_budget(monkeypatch):
    zot = _EndlessZot()
    # Already spent: a coarse clock (Windows) can read the same instant twice.
    monkeypatch.setattr(ApiBackend, "_CITEKEY_SCAN_BUDGET", -1.0)
    with pytest.raises(TimeoutError, match="first 100 items"):
        ApiBackend(zot).find_by_citation_key("smith2020")
    assert zot.pages == 1


def test_api_citation_key_scan_still_finds_a_key_on_a_later_page():
    zot = _EndlessZot(pages_before_hit=3)
    assert ApiBackend(zot).find_by_citation_key("smith2020")["key"] == "FOUND001"
    assert zot.pages == 3
