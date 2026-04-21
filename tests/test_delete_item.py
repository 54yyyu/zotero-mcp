"""Tests for zotero_delete_item — Trash wrapper for any item type (#227).

zotero_delete_note handles notes; the Zotero Web API supports trashing any
item type via PATCH {"deleted": 1}. This test file covers the generic
delete_item tool that wraps that mechanism for books, journalArticles,
webpages, attachments, and so on.
"""

import pytest

from zotero_mcp import server
from conftest import DummyContext


class _FakePatchResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


class _FakeHttpxClient:
    def __init__(self, status_code=204, text=""):
        self._status_code = status_code
        self._text = text
        self.calls = []

    def patch(self, url, headers, content):
        self.calls.append({"url": url, "headers": headers, "content": content})
        return _FakePatchResponse(self._status_code, self._text)


class _FakeZoteroForDelete:
    def __init__(self, items, patch_status=204):
        self._items = items
        self.endpoint = "https://api.zotero.org"
        self.library_type = "users"
        self.library_id = "12345"
        self.client = _FakeHttpxClient(status_code=patch_status)

    def item(self, key):
        if key not in self._items:
            raise KeyError(key)
        return self._items[key]


def _book_item(key="BOOK0001", version=42):
    return {
        "key": key,
        "version": version,
        "data": {
            "key": key, "version": version, "itemType": "book",
            "title": "Some Book",
        },
    }


def _note_item(key="NOTE0001", version=3):
    return {
        "key": key,
        "version": version,
        "data": {
            "key": key, "version": version, "itemType": "note",
            "note": "<p>text</p>",
        },
    }


def _article_item(key="ART00001", version=7):
    return {
        "key": key,
        "version": version,
        "data": {
            "key": key, "version": version, "itemType": "journalArticle",
            "title": "A Paper",
        },
    }


class TestDeleteItemSuccess:
    def test_trashes_book(self, monkeypatch):
        fake = _FakeZoteroForDelete({"BOOK0001": _book_item()})
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.delete_item(item_key="BOOK0001", ctx=DummyContext())

        assert "Successfully trashed" in result
        assert "book" in result
        assert len(fake.client.calls) == 1
        call = fake.client.calls[0]
        assert "BOOK0001" in call["url"]
        assert call["headers"]["If-Unmodified-Since-Version"] == "42"
        assert '"deleted": 1' in call["content"]

    def test_trashes_journal_article(self, monkeypatch):
        fake = _FakeZoteroForDelete({"ART00001": _article_item()})
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.delete_item(item_key="ART00001", ctx=DummyContext())

        assert "Successfully trashed" in result
        assert "journalArticle" in result


class TestDeleteItemNotesSafety:
    def test_refuses_note_by_default(self, monkeypatch):
        """Notes are redirected to zotero_delete_note for explicitness."""
        fake = _FakeZoteroForDelete({"NOTE0001": _note_item()})
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.delete_item(item_key="NOTE0001", ctx=DummyContext())

        assert "is a note" in result
        assert "zotero_delete_note" in result
        assert fake.client.calls == []

    def test_allow_note_override(self, monkeypatch):
        """Explicit opt-in permits trashing a note through delete_item."""
        fake = _FakeZoteroForDelete({"NOTE0001": _note_item()})
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.delete_item(
            item_key="NOTE0001", allow_note=True, ctx=DummyContext()
        )

        assert "Successfully trashed" in result
        assert len(fake.client.calls) == 1


class TestDeleteItemErrors:
    def test_missing_item_key(self, monkeypatch):
        fake = _FakeZoteroForDelete({})
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.delete_item(item_key="ZZZZZZZZ", ctx=DummyContext())

        assert "No item found" in result
        assert fake.client.calls == []

    def test_http_failure_reports(self, monkeypatch):
        fake = _FakeZoteroForDelete({"BOOK0001": _book_item()}, patch_status=412)
        fake.client._text = "Precondition failed"
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.delete_item(item_key="BOOK0001", ctx=DummyContext())

        assert "Failed to trash" in result
        assert "412" in result

    def test_local_only_mode_rejected(self, monkeypatch):
        def _raise(ctx):
            raise ValueError(
                "Cannot perform write operations in local-only mode. "
                "Add ZOTERO_API_KEY and ZOTERO_LIBRARY_ID to enable hybrid mode."
            )
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", _raise)

        result = server.delete_item(item_key="BOOK0001", ctx=DummyContext())

        assert "local-only" in result.lower() or "Cannot perform write" in result


class TestDeleteItemPatchShape:
    """The PATCH url/headers/content must match Zotero's write-API contract."""

    def test_url_targets_correct_library(self, monkeypatch):
        fake = _FakeZoteroForDelete({"BOOK0001": _book_item()})
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        server.delete_item(item_key="BOOK0001", ctx=DummyContext())

        url = fake.client.calls[0]["url"]
        assert "/users/12345/items/BOOK0001" in url

    def test_version_header_matches_fetched_version(self, monkeypatch):
        fake = _FakeZoteroForDelete({"BOOK0001": _book_item(version=99)})
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        server.delete_item(item_key="BOOK0001", ctx=DummyContext())

        assert fake.client.calls[0]["headers"]["If-Unmodified-Since-Version"] == "99"
