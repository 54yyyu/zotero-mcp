"""Tests for zotero_add_by_isbn (#226).

Covers ISBN normalization (10→13 conversion, checksum validation), the
Open Library → Google Books lookup cascade, and the resulting Zotero book
item shape.
"""

import json

import pytest
import requests

from zotero_mcp import server
from zotero_mcp.tools import write as _write
from zotero_mcp.tools._helpers import (
    _isbn10_to_isbn13,
    _isbn13_checksum_valid,
    _normalize_isbn,
)
from conftest import DummyContext, FakeZotero


# ---------------------------------------------------------------------------
# ISBN normalization
# ---------------------------------------------------------------------------

class TestNormalizeIsbn:
    def test_valid_isbn13(self):
        # The Pragmatic Programmer 2e
        assert _normalize_isbn("9780135957059") == "9780135957059"

    def test_valid_isbn13_with_hyphens(self):
        assert _normalize_isbn("978-0-13-595705-9") == "9780135957059"

    def test_isbn10_converted_to_isbn13(self):
        # Gödel, Escher, Bach ISBN-10 -> ISBN-13
        out = _normalize_isbn("0465026567")
        assert out is not None
        assert out.startswith("978")
        assert _isbn13_checksum_valid(out)

    def test_isbn10_x_checksum(self):
        # ISBN-10 ending in X (checksum 10)
        assert _normalize_isbn("080442957X") is not None

    def test_invalid_checksum_rejected(self):
        assert _normalize_isbn("9780135957050") is None  # wrong check digit
        assert _normalize_isbn("0465026560") is None

    def test_short_string_rejected(self):
        assert _normalize_isbn("12345") is None

    def test_non_isbn_prefix_stripped(self):
        assert _normalize_isbn("ISBN: 978-0-13-595705-9") == "9780135957059"
        assert _normalize_isbn("isbn:9780135957059") == "9780135957059"

    def test_empty_and_none(self):
        assert _normalize_isbn("") is None
        assert _normalize_isbn(None) is None

    def test_isbn10_to_isbn13_known_case(self):
        assert _isbn10_to_isbn13("0465026567") == "9780465026562"


# ---------------------------------------------------------------------------
# Lookup helpers: Open Library and Google Books
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def _fake_get_factory(responses):
    """Build a fake requests.get that returns responses keyed by URL substring."""
    def _fake_get(url, **kwargs):
        for substring, resp in responses.items():
            if substring in url:
                return resp
        return _FakeResponse(404)
    return _fake_get


OL_PAYLOAD = {
    "ISBN:9780199735815": {
        "title": "The Oxford Handbook of Philosophy of Mind",
        "authors": [{"name": "Brian McLaughlin"}, {"name": "Ansgar Beckermann"}],
        "publishers": [{"name": "Oxford University Press"}],
        "publish_places": [{"name": "Oxford"}],
        "publish_date": "2009",
        "number_of_pages": 800,
        "url": "https://openlibrary.org/books/OL24312345M",
    }
}

GB_PAYLOAD = {
    "items": [{
        "volumeInfo": {
            "title": "Some Rare Book",
            "subtitle": "A Subtitle",
            "authors": ["Jane Doe"],
            "publisher": "Academic Press",
            "publishedDate": "2020",
            "pageCount": 300,
            "infoLink": "https://books.google.com/books?id=abc",
        }
    }]
}


class TestOpenLibraryLookup:
    def test_hit_returns_normalized_dict(self, monkeypatch):
        monkeypatch.setattr(
            _write, "requests",
            type("R", (), {
                "get": _fake_get_factory({
                    "openlibrary.org": _FakeResponse(200, OL_PAYLOAD),
                }),
                "RequestException": requests.RequestException,
            })
        )
        meta = _write._lookup_isbn_openlibrary("9780199735815", DummyContext())
        assert meta is not None
        assert meta["source"] == "Open Library"
        assert "Oxford Handbook" in meta["title"]
        assert len(meta["creators"]) == 2
        assert meta["creators"][0]["lastName"] == "McLaughlin"
        assert meta["publisher"] == "Oxford University Press"
        assert meta["place"] == "Oxford"

    def test_miss_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            _write, "requests",
            type("R", (), {
                "get": _fake_get_factory({
                    "openlibrary.org": _FakeResponse(200, {}),
                }),
                "RequestException": requests.RequestException,
            })
        )
        assert _write._lookup_isbn_openlibrary("9999999999999", DummyContext()) is None


class TestGoogleBooksLookup:
    def test_hit_returns_normalized_dict(self, monkeypatch):
        monkeypatch.setattr(
            _write, "requests",
            type("R", (), {
                "get": _fake_get_factory({
                    "googleapis.com": _FakeResponse(200, GB_PAYLOAD),
                }),
                "RequestException": requests.RequestException,
            })
        )
        meta = _write._lookup_isbn_google_books("9780000000000", DummyContext())
        assert meta is not None
        assert meta["source"] == "Google Books"
        assert "Some Rare Book: A Subtitle" == meta["title"]
        assert meta["publisher"] == "Academic Press"
        assert meta["place"] == ""  # Google Books doesn't expose place

    def test_no_items_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            _write, "requests",
            type("R", (), {
                "get": _fake_get_factory({
                    "googleapis.com": _FakeResponse(200, {"items": []}),
                }),
                "RequestException": requests.RequestException,
            })
        )
        assert _write._lookup_isbn_google_books("9999999999999", DummyContext()) is None


# ---------------------------------------------------------------------------
# End-to-end add_by_isbn
# ---------------------------------------------------------------------------

class TestAddByIsbnEndToEnd:
    def test_open_library_hit_creates_book(self, monkeypatch):
        fake = FakeZotero()
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client",
            lambda ctx: (fake, fake),
        )
        monkeypatch.setattr(
            _write, "requests",
            type("R", (), {
                "get": _fake_get_factory({
                    "openlibrary.org": _FakeResponse(200, OL_PAYLOAD),
                }),
                "RequestException": requests.RequestException,
            })
        )

        result = server.add_by_isbn(
            isbn="978-0-19-973581-5",
            ctx=DummyContext(),
        )

        assert "Successfully added" in result
        assert "Oxford Handbook" in result
        assert "9780199735815" in result
        assert "Open Library" in result
        # Item was passed to create_items with book shape
        assert len(fake.created) == 1
        created = fake.created[0]
        assert created["itemType"] == "book"
        assert created["ISBN"] == "9780199735815"
        assert created["publisher"] == "Oxford University Press"
        assert created["place"] == "Oxford"

    def test_falls_back_to_google_books_on_open_library_miss(self, monkeypatch):
        fake = FakeZotero()
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client",
            lambda ctx: (fake, fake),
        )
        monkeypatch.setattr(
            _write, "requests",
            type("R", (), {
                "get": _fake_get_factory({
                    "openlibrary.org": _FakeResponse(200, {}),  # OL miss
                    "googleapis.com": _FakeResponse(200, GB_PAYLOAD),  # GB hit
                }),
                "RequestException": requests.RequestException,
            })
        )

        result = server.add_by_isbn(
            isbn="9780135957059",
            ctx=DummyContext(),
        )

        assert "Successfully added" in result
        assert "Google Books" in result

    def test_both_sources_miss_returns_clear_error(self, monkeypatch):
        fake = FakeZotero()
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client",
            lambda ctx: (fake, fake),
        )
        monkeypatch.setattr(
            _write, "requests",
            type("R", (), {
                "get": _fake_get_factory({
                    "openlibrary.org": _FakeResponse(200, {}),
                    "googleapis.com": _FakeResponse(200, {"items": []}),
                }),
                "RequestException": requests.RequestException,
            })
        )

        result = server.add_by_isbn(
            isbn="9780135957059",
            ctx=DummyContext(),
        )

        assert "not found" in result.lower()
        assert len(fake.created) == 0

    def test_invalid_isbn_rejected(self, monkeypatch):
        fake = FakeZotero()
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client",
            lambda ctx: (fake, fake),
        )

        result = server.add_by_isbn(
            isbn="not-an-isbn",
            ctx=DummyContext(),
        )

        assert "not appear to be a valid ISBN" in result
        assert len(fake.created) == 0

    def test_tags_and_collections_applied(self, monkeypatch):
        fake = FakeZotero()
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client",
            lambda ctx: (fake, fake),
        )
        monkeypatch.setattr(
            _write, "requests",
            type("R", (), {
                "get": _fake_get_factory({
                    "openlibrary.org": _FakeResponse(200, OL_PAYLOAD),
                }),
                "RequestException": requests.RequestException,
            })
        )

        server.add_by_isbn(
            isbn="9780199735815",
            tags=["philosophy", "anthology"],
            collections=["COLL0001"],
            ctx=DummyContext(),
        )

        created = fake.created[0]
        assert {t["tag"] for t in created["tags"]} == {"philosophy", "anthology"}
        assert created["collections"] == ["COLL0001"]
