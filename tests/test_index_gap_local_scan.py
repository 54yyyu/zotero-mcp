"""The local --fulltext scan keeps up-to-date items that lack a level (#769)."""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search
from zotero_mcp.extract import DEFAULT_ATTACHMENT_PRIORITY

DATE = "2026-07-02 01:01:48"
ATTACHMENTS = [("ATTKEY01", "storage:paper.pdf", "application/pdf")]


class FakeItem:
    def __init__(self):
        self.item_id = 1
        self.key = "ITEMKEY1"
        self.item_type = "journalArticle"
        self.title = "Calibration"
        self.abstract = ""
        self.extra = ""
        self.doi = None
        self.notes = None
        self.creators = None
        self.date_added = DATE
        self.date_modified = DATE
        self.fulltext = None
        self.fulltext_source = None


class FakeReader:
    attachment_priority = DEFAULT_ATTACHMENT_PRIORITY

    def __init__(self):
        self.extract_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_all_item_keys(self):
        return {"ITEMKEY1"}

    def get_key_group_map(self):
        return ({"ITEMKEY1": 0}, set())

    def get_items_with_text(self, limit=None, include_fulltext=False, key_filter=None, collection_keys=None):
        return [FakeItem()]

    def get_fulltext_meta_for_item(self, item_id):
        return [list(row) for row in ATTACHMENTS]

    def extract_fulltext_for_item(self, item_id):
        self.extract_calls += 1
        return ("extracted text", "pdf")


class FakeChroma:
    embedding_max_tokens = 8000

    def __init__(self, stored):
        self.stored = {"group_id": 0, "date_modified": DATE, "attachment_keys": "ATTKEY01", **stored}

    def get_document_metadata(self, key):
        return self.stored


def _scan(monkeypatch, stored, gaps):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    reader = FakeReader()
    monkeypatch.setattr(semantic_search, "LocalZoteroReader", lambda *a, **kw: reader)
    client = FakeChroma(stored)
    search = semantic_search.ZoteroSemanticSearch(chroma_client=client)
    search._scan_gaps = gaps
    items = search._get_items_from_local_db(extract_fulltext=True, chroma_client=client)
    return items, reader


def test_up_to_date_item_without_a_gap_is_skipped(monkeypatch):
    items, reader = _scan(monkeypatch, {"has_fulltext": True}, {})

    assert items == []
    assert reader.extract_calls == 0


def test_up_to_date_item_with_a_gap_is_kept_and_marked(monkeypatch):
    items, reader = _scan(monkeypatch, {"has_fulltext": True}, {"ITEMKEY1": {"passages"}})

    assert [item["key"] for item in items] == ["ITEMKEY1"]
    assert items[0]["_index_levels"] == {"passages"}
    assert items[0]["data"]["fulltext"] == "extracted text"
    assert reader.extract_calls == 1


def test_item_whose_extraction_failed_is_not_parsed_again_for_a_gap(monkeypatch):
    """A failed PDF can take minutes, and nothing about it has changed."""
    items, reader = _scan(monkeypatch, {"has_fulltext": "failed"}, {"ITEMKEY1": {"passages"}})

    assert [item["key"] for item in items] == ["ITEMKEY1"]
    assert items[0]["data"]["fulltext_attempted"] is True
    assert reader.extract_calls == 0
