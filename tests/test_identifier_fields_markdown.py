"""Tests that identifier fields (DOI, ISBN, ISSN) render in markdown output.

Regression for issue #236 — ISBN was emitted by bibtex but dropped by the
markdown formatter, causing agents to conclude ISBN was missing and attempt
to write stale values.
"""

from zotero_mcp.client import format_item_metadata


def _book(**fields):
    data = {
        "key": "BOOK1234",
        "itemType": "book",
        "title": "A Book",
        "creators": [{"creatorType": "author", "lastName": "Doe", "firstName": "J."}],
        "date": "2024",
        "publisher": "Test Press",
    }
    data.update(fields)
    return {"data": data}


def _article(**fields):
    data = {
        "key": "ART12345",
        "itemType": "journalArticle",
        "title": "An Article",
        "creators": [{"creatorType": "author", "lastName": "Doe", "firstName": "J."}],
        "date": "2024",
        "publicationTitle": "Test Journal",
    }
    data.update(fields)
    return {"data": data}


class TestIsbnInMarkdown:
    def test_isbn_rendered(self):
        item = _book(ISBN="9780199735815")
        output = format_item_metadata(item, include_abstract=False)
        assert "**ISBN:** 9780199735815" in output

    def test_isbn_omitted_when_empty(self):
        item = _book(ISBN="")
        output = format_item_metadata(item, include_abstract=False)
        assert "ISBN" not in output

    def test_isbn_omitted_when_missing(self):
        item = _book()
        output = format_item_metadata(item, include_abstract=False)
        assert "ISBN" not in output


class TestIssnInMarkdown:
    def test_issn_rendered(self):
        item = _article(ISSN="0028-0836")
        output = format_item_metadata(item, include_abstract=False)
        assert "**ISSN:** 0028-0836" in output

    def test_issn_omitted_when_empty(self):
        item = _article(ISSN="")
        output = format_item_metadata(item, include_abstract=False)
        assert "ISSN" not in output


class TestDoiStillRendered:
    """Regression guard — #236 fix must not break existing DOI rendering."""

    def test_doi_rendered(self):
        item = _article(DOI="10.1234/example")
        output = format_item_metadata(item, include_abstract=False)
        assert "**DOI:** 10.1234/example" in output
