"""Tests that the `place` (publication place) field is exposed on all read paths.

Regression for issue #238 — place was silently dropped by the markdown
formatter (except for an inline publisher-concatenation on book items),
the bibtex generator, and the connector fetch metadata dict. Reverification
workflows could not determine whether users had populated place.
"""

import json

from zotero_mcp.client import format_item_metadata, generate_bibtex
from zotero_mcp.tools import connectors as _conn

from conftest import DummyContext


def _book(**fields):
    data = {
        "key": "BOOK1234",
        "itemType": "book",
        "title": "A Book",
        "creators": [{"creatorType": "author", "lastName": "Doe", "firstName": "J."}],
        "date": "2024",
    }
    data.update(fields)
    return {"data": data}


def _book_section(**fields):
    data = {
        "key": "SECT1234",
        "itemType": "bookSection",
        "title": "A Chapter",
        "creators": [{"creatorType": "author", "lastName": "Doe", "firstName": "J."}],
        "date": "2024",
        "bookTitle": "Parent Book",
    }
    data.update(fields)
    return {"data": data}


def _thesis(**fields):
    data = {
        "key": "THES1234",
        "itemType": "thesis",
        "title": "A Thesis",
        "creators": [{"creatorType": "author", "lastName": "Doe", "firstName": "J."}],
        "date": "2024",
    }
    data.update(fields)
    return {"data": data}


class TestPlaceInMarkdown:
    def test_place_rendered_for_book(self):
        item = _book(publisher="Oxford University Press", place="Oxford")
        output = format_item_metadata(item, include_abstract=False)
        assert "**Place:** Oxford" in output
        assert "**Publisher:** Oxford University Press" in output

    def test_place_rendered_for_book_section(self):
        item = _book_section(publisher="MIT Press", place="Cambridge, MA")
        output = format_item_metadata(item, include_abstract=False)
        assert "**Place:** Cambridge, MA" in output

    def test_place_rendered_for_thesis(self):
        item = _thesis(publisher="Columbia University", place="New York")
        output = format_item_metadata(item, include_abstract=False)
        assert "**Place:** New York" in output

    def test_place_without_publisher(self):
        """Place should be emitted even when publisher is absent."""
        item = _book(place="London")
        output = format_item_metadata(item, include_abstract=False)
        assert "**Place:** London" in output

    def test_place_omitted_when_empty(self):
        item = _book(publisher="Test Press", place="")
        output = format_item_metadata(item, include_abstract=False)
        assert "**Place:**" not in output
        assert "**Publisher:** Test Press" in output

    def test_place_omitted_when_missing(self):
        item = _book(publisher="Test Press")
        output = format_item_metadata(item, include_abstract=False)
        assert "**Place:**" not in output


class TestPlaceInBibtex:
    def test_address_rendered_for_book(self):
        item = _book(publisher="Oxford University Press", place="Oxford")
        output = generate_bibtex(item)
        assert "address = {Oxford}" in output

    def test_address_rendered_for_book_section(self):
        item = _book_section(publisher="MIT Press", place="Cambridge, MA")
        output = generate_bibtex(item)
        assert "address = {Cambridge, MA}" in output

    def test_address_omitted_when_empty(self):
        item = _book(publisher="Test Press", place="")
        output = generate_bibtex(item)
        assert "address" not in output

    def test_address_omitted_when_missing(self):
        item = _book(publisher="Test Press")
        output = generate_bibtex(item)
        assert "address" not in output


class TestConnectorFetchMetadata:
    """The fetch wrapper is a thin builder over pyzotero item data; monkeypatch
    the Zotero client and the fulltext helper so the test stays unit-scoped."""

    def test_place_in_fetch_metadata(self, monkeypatch):
        fake_data = {
            "itemType": "book",
            "title": "A Book",
            "date": "2024",
            "DOI": "",
            "ISBN": "978-0-00-000000-0",
            "ISSN": "",
            "publisher": "Oxford University Press",
            "place": "Oxford",
            "creators": [],
            "tags": [],
        }

        class _FakeZot:
            def item(self, key):
                return {"key": key, "data": fake_data}

        monkeypatch.setattr(_conn._client, "get_zotero_client", lambda: _FakeZot())
        monkeypatch.setattr(
            _conn,
            "get_item_fulltext",
            lambda item_key, ctx: "# A Book\n\n## Full Text\n\nbody",
        )

        result = _conn.connector_fetch(id="ABCD1234", ctx=DummyContext())
        payload = json.loads(result)

        assert payload["metadata"]["place"] == "Oxford"
        assert payload["metadata"]["publisher"] == "Oxford University Press"
        assert payload["metadata"]["isbn"] == "978-0-00-000000-0"
