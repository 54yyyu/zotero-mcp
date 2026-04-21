"""Tests for metadata output formats."""

import json

from zotero_mcp import server


class DummyContext:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


class FakeZoteroForMetadata:
    def __init__(self, item):
        self._item = item

    def item(self, key):
        if key != self._item["key"]:
            raise KeyError(key)
        return self._item


def test_get_item_metadata_json_returns_complete_raw_item(monkeypatch):
    item = {
        "key": "ITEM0001",
        "version": 7,
        "links": {"self": {"href": "https://api.zotero.org/users/1/items/ITEM0001"}},
        "meta": {"numChildren": 2},
        "data": {
            "key": "ITEM0001",
            "itemType": "journalArticle",
            "title": "Genome paper",
            "date": "2025-01-01",
            "creators": [{"creatorType": "author", "firstName": "Lin", "lastName": "Wang"}],
            "abstractNote": "Abstract text",
            "DOI": "10.1234/example",
            "url": "https://example.com/paper",
            "extra": "Citation Key: Wang2025",
            "relations": {"dc:relation": ["http://zotero.org/users/1/items/ABCD1234"]},
            "collections": ["COLL0001"],
            "tags": [{"tag": "seed"}],
            "language": "en",
        },
    }
    fake = FakeZoteroForMetadata(item)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)

    result = server.get_item_metadata("ITEM0001", format="json", ctx=DummyContext())
    parsed = json.loads(result)

    assert parsed["key"] == "ITEM0001"
    assert parsed["version"] == 7
    assert parsed["meta"]["numChildren"] == 2
    assert parsed["data"]["language"] == "en"
    assert parsed["data"]["relations"]["dc:relation"] == [
        "http://zotero.org/users/1/items/ABCD1234"
    ]


def test_get_item_metadata_markdown_remains_readable_summary(monkeypatch):
    item = {
        "key": "ITEM0002",
        "data": {
            "key": "ITEM0002",
            "itemType": "journalArticle",
            "title": "Readable metadata",
            "date": "2024",
            "publicationTitle": "Nature Plants",
            "volume": "10",
            "issue": "2",
            "pages": "12-20",
            "creators": [{"creatorType": "author", "firstName": "Li", "lastName": "Chen"}],
            "DOI": "10.9999/readable",
            "tags": [{"tag": "plants"}],
            "abstractNote": "Summary abstract",
        },
        "meta": {"numChildren": 1},
    }
    fake = FakeZoteroForMetadata(item)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)

    result = server.get_item_metadata("ITEM0002", format="markdown", ctx=DummyContext())

    assert "# Readable metadata" in result
    assert "**Journal:** Nature Plants, Volume 10, Issue 2, Pages 12-20" in result
    assert "**DOI:** 10.9999/readable" in result
    assert "## Abstract" in result
