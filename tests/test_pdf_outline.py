"""Tests for Feature 9: PDF Outline Extraction (zotero_get_pdf_outline)."""

import pytest

from conftest import DummyContext, FakeZotero
from zotero_mcp import server
from zotero_mcp.tools import write as write_tools


# ---------------------------------------------------------------------------
# Helpers: canned outcomes from the isolated TOC reader
# ---------------------------------------------------------------------------

def _patch_toc(monkeypatch, toc=None, status="ok", detail=""):
    """Stub the out-of-process TOC reader with a canned outcome.

    ``get_toc()`` runs in a child interpreter since #372, so faking ``fitz``
    in ``sys.modules`` no longer reaches it — the outcome is stubbed instead.
    """
    outcome = write_tools.TocOutcome(status, list(toc or []), detail)
    monkeypatch.setattr(write_tools, "_extract_pdf_toc", lambda *_a, **_k: outcome)
    # Keep the download hermetic: never probe a real local Zotero server.
    monkeypatch.setattr("zotero_mcp.client.get_local_zotero_client", lambda: None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _pdf_child(key="ATTACH01", filename="paper.pdf"):
    """Return an attachment dict that looks like a PDF child item."""
    return {
        "key": key,
        "data": {
            "itemType": "attachment",
            "contentType": "application/pdf",
            "filename": filename,
            "title": filename,
            "parentItem": "PARENT01",
        },
    }


def _note_child(key="NOTE01"):
    """Return a note child item (should be ignored by get_pdf_outline)."""
    return {
        "key": key,
        "data": {
            "itemType": "note",
            "note": "<p>Some note text</p>",
            "parentItem": "PARENT01",
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetPdfOutlineHappyPath:
    """Happy-path: item with a PDF child, TOC returned as markdown."""

    def test_simple_toc_formatted_as_markdown(self, monkeypatch, dummy_ctx, fake_zot):
        toc = [
            [1, "Introduction", 1],
            [1, "Methods", 5],
            [1, "Results", 10],
            [1, "Conclusion", 15],
        ]
        fake_zot._children["PARENT01"] = [_pdf_child()]

        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)
        _patch_toc(monkeypatch, toc)
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

        result = server.get_pdf_outline(item_key="PARENT01", ctx=dummy_ctx)

        assert "- Introduction (p. 1)" in result
        assert "- Methods (p. 5)" in result
        assert "- Results (p. 10)" in result
        assert "- Conclusion (p. 15)" in result


class TestNestedToc:
    """Nested TOC entries (levels 1, 2, 3) produce proper indentation."""

    def test_indentation_by_level(self, monkeypatch, dummy_ctx, fake_zot):
        toc = [
            [1, "Chapter 1", 1],
            [2, "Section 1.1", 3],
            [3, "Subsection 1.1.1", 5],
            [2, "Section 1.2", 8],
            [1, "Chapter 2", 12],
        ]
        fake_zot._children["ITEM01"] = [_pdf_child()]

        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)
        _patch_toc(monkeypatch, toc)
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

        result = server.get_pdf_outline(item_key="ITEM01", ctx=dummy_ctx)

        # Level 1: no indent
        assert "- Chapter 1 (p. 1)" in result
        # Level 2: 2 spaces
        assert "  - Section 1.1 (p. 3)" in result
        # Level 3: 4 spaces
        assert "    - Subsection 1.1.1 (p. 5)" in result
        assert "  - Section 1.2 (p. 8)" in result
        assert "- Chapter 2 (p. 12)" in result


class TestEmptyToc:
    """Empty TOC returns a descriptive message."""

    def test_empty_toc_message(self, monkeypatch, dummy_ctx, fake_zot):
        fake_zot._children["ITEM01"] = [_pdf_child()]

        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)
        _patch_toc(monkeypatch, toc=[])
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

        result = server.get_pdf_outline(item_key="ITEM01", ctx=dummy_ctx)

        assert "does not contain a table of contents" in result.lower() or \
               "does not contain a table of contents/outline" in result.lower()


class TestNoPdfAttachment:
    """No PDF attachment found -> error message."""

    def test_no_children_at_all(self, monkeypatch, dummy_ctx, fake_zot):
        fake_zot._children["ITEM01"] = []

        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)

        result = server.get_pdf_outline(item_key="ITEM01", ctx=dummy_ctx)

        assert "no pdf" in result.lower() or "pdf attachment" in result.lower()

    def test_only_note_children(self, monkeypatch, dummy_ctx, fake_zot):
        fake_zot._children["ITEM01"] = [_note_child()]

        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)

        result = server.get_pdf_outline(item_key="ITEM01", ctx=dummy_ctx)

        assert "no pdf" in result.lower() or "pdf attachment" in result.lower()


class TestMultipleChildrenOnlyPdfUsed:
    """Multiple children (notes + PDF) -- only the PDF is used."""

    def test_picks_pdf_among_mixed_children(self, monkeypatch, dummy_ctx, fake_zot):
        toc = [
            [1, "Abstract", 1],
            [1, "Body", 2],
        ]
        fake_zot._children["ITEM01"] = [
            _note_child(key="NOTE01"),
            _pdf_child(key="PDF01", filename="main.pdf"),
            _note_child(key="NOTE02"),
        ]

        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)
        _patch_toc(monkeypatch, toc)
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

        result = server.get_pdf_outline(item_key="ITEM01", ctx=dummy_ctx)

        assert "- Abstract (p. 1)" in result
        assert "- Body (p. 2)" in result

    def test_uses_first_pdf_when_multiple_pdfs(self, monkeypatch, dummy_ctx, fake_zot):
        """If there are multiple PDF attachments, the first one is used."""
        toc = [
            [1, "First PDF Outline", 1],
        ]
        fake_zot._children["ITEM01"] = [
            _pdf_child(key="PDF01", filename="first.pdf"),
            _pdf_child(key="PDF02", filename="second.pdf"),
        ]

        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)
        _patch_toc(monkeypatch, toc)
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

        result = server.get_pdf_outline(item_key="ITEM01", ctx=dummy_ctx)

        assert "- First PDF Outline (p. 1)" in result
