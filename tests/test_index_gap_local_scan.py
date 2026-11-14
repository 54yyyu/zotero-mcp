"""The local --fulltext scan keeps up-to-date items that lack a level (#769)."""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from _scan_fakes import ScanReader, target

from zotero_mcp import semantic_search

DATE = "2026-07-02 01:01:48"


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


class FakeReader(ScanReader):
    def __init__(self):
        super().__init__(targets={1: [target("ATTKEY01")]})

    def get_all_item_keys(self):
        return {"ITEMKEY1"}

    def get_key_group_map(self):
        return ({"ITEMKEY1": 0}, set())

    def get_items_with_text(self, limit=None, include_fulltext=False, key_filter=None, collection_keys=None):
        return [FakeItem()]


class FakeChroma:
    embedding_max_tokens = 8000

    def __init__(self, stored):
        self.stored = {"group_id": 0, "date_modified": DATE, "attachment_keys": "ATTKEY01", "title": "T", **stored}

    def get_document_metadata(self, key):
        return self.stored


def _scan(monkeypatch, stored, gaps, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    reader = FakeReader()
    monkeypatch.setattr(semantic_search, "LocalZoteroReader", lambda *a, **kw: reader)
    client = FakeChroma(stored)
    search = semantic_search.ZoteroSemanticSearch(
        chroma_client=client, config_path=str(tmp_path / "config.json")
    )
    search._scan_gaps = gaps
    items = search._get_items_from_local_db(extract_fulltext=True, chroma_client=client)
    return items, reader


# The stored record is legacy (it carries ``title``, #826) and unchanged.


def test_up_to_date_item_without_a_gap_is_skipped(monkeypatch, tmp_path):
    items, reader = _scan(monkeypatch, {"has_fulltext": True}, {}, tmp_path)

    assert items == []
    assert reader.extract_calls == 0


def test_up_to_date_legacy_item_missing_passages_is_rewritten_by_attachment(monkeypatch, tmp_path):
    """Its passages are cut from its files now, under their own keys."""
    items, reader = _scan(monkeypatch, {"has_fulltext": True}, {"ITEMKEY1": {"passages"}}, tmp_path)

    assert [(item["key"], item["unit"]) for item in items] == [("ITEMKEY1", "item"), ("ATTKEY01", "attachment")]
    assert items[0]["legacy_rewrite"] is True
    assert items[1]["text"] == "extracted text"
    assert "_index_levels" not in items[1]
    assert reader.extract_calls == 1


def test_item_whose_extraction_failed_is_not_parsed_again_for_a_gap(monkeypatch, tmp_path):
    """A failed PDF can take minutes, and nothing about it has changed.

    An item indexed without text has no passages to lack (#826).
    """
    items, reader = _scan(monkeypatch, {"has_fulltext": "failed"}, {"ITEMKEY1": {"passages"}}, tmp_path)

    assert items == []
    assert reader.extract_calls == 0
