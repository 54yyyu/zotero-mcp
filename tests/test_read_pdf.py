"""Tests for zotero_read_pdf_pages tool."""

import sys
import types

import pytest
from conftest import DummyContext, FakeZotero

from zotero_mcp import server

# ---------------------------------------------------------------------------
# Helpers: fake fitz module and document
# ---------------------------------------------------------------------------


class FakePage:
    def __init__(self, text):
        self._text = text

    def get_text(self):
        return self._text


class FakeDocument:
    def __init__(self, pages, total=None):
        self._pages = pages
        self._total = total if total is not None else len(pages)

    def __len__(self):
        return self._total

    def __getitem__(self, index):
        return self._pages[index]

    def close(self):
        pass


def _make_fake_fitz(pages, total=None):
    fake_fitz = types.ModuleType("fitz")
    fake_fitz.open = lambda *args, **kwargs: FakeDocument(pages, total)  # noqa: ARG005
    return fake_fitz


def _patch_fitz(monkeypatch, pages, total=None):
    fake_fitz = _make_fake_fitz(pages, total)
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_ctx():
    return DummyContext()


@pytest.fixture
def fake_zot():
    return FakeZotero()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHappyPath:
    """Single page and page range reads."""

    def test_single_page(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_fitz(monkeypatch, [FakePage("Page 1 content.")] * 10, total=10)
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "Test Paper"),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=3, ctx=dummy_ctx)

        assert "## Page 3" in result
        assert "Page 1 content." in result

    def test_page_range(self, monkeypatch, dummy_ctx, fake_zot):
        pages = [
            FakePage("Content of page 1."),
            FakePage("Content of page 2."),
            FakePage("Content of page 3."),
            FakePage("Content of page 4."),
            FakePage("Content of page 5."),
        ]
        _patch_fitz(monkeypatch, pages)
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "Test Paper"),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=2, end_page=4, ctx=dummy_ctx)

        assert "## Page 2" in result
        assert "Content of page 2." in result
        assert "## Page 3" in result
        assert "Content of page 3." in result
        assert "## Page 4" in result
        assert "Content of page 4." in result
        assert "## Page 1" not in result
        assert "## Page 5" not in result

    def test_header_contains_metadata(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_fitz(monkeypatch, [FakePage("hello")])
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "My Paper Title"),
        )

        result = server.read_pdf_pages(item_key="KEY123", start_page=1, ctx=dummy_ctx)

        assert "# PDF Pages 1-1 of My Paper Title" in result
        assert "**Item Key:** KEY123" in result
        assert "**Total pages in PDF:** 1" in result


class TestErrors:
    """Input validation and error cases."""

    def test_empty_item_key(self, dummy_ctx):
        result = server.read_pdf_pages(item_key="", start_page=1, ctx=dummy_ctx)
        assert "item_key cannot be empty" in result

    def test_whitespace_item_key(self, dummy_ctx):
        result = server.read_pdf_pages(item_key="   ", start_page=1, ctx=dummy_ctx)
        assert "item_key cannot be empty" in result

    def test_end_page_less_than_start_page(self, dummy_ctx):
        result = server.read_pdf_pages(item_key="ITEM01", start_page=5, end_page=3, ctx=dummy_ctx)
        assert "end_page must be greater than or equal to start_page" in result

    def test_no_pdf_attachment(self, monkeypatch, dummy_ctx, fake_zot):
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: None,
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=1, ctx=dummy_ctx)

        assert "No PDF attachment found" in result

    def test_start_page_out_of_range(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_fitz(monkeypatch, [FakePage("p1")], total=1)
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "Paper"),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=5, ctx=dummy_ctx)

        assert "out of range" in result
        assert "1-1" in result

    def test_end_page_out_of_range(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_fitz(monkeypatch, [FakePage("p1")] * 3, total=3)
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "Paper"),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=1, end_page=10, ctx=dummy_ctx)

        assert "out of range" in result
        assert "1-3" in result

    def test_too_many_pages(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_fitz(monkeypatch, [FakePage("p")] * 100, total=100)
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "Paper"),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=1, end_page=55, ctx=dummy_ctx)

        assert "max 50" in result

    def test_missing_fitz_module(self, monkeypatch, dummy_ctx, fake_zot):
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "Paper"),
        )
        monkeypatch.setitem(sys.modules, "fitz", None)

        result = server.read_pdf_pages(item_key="ITEM01", start_page=1, ctx=dummy_ctx)

        assert "PyMuPDF" in result


class TestEdgeCases:
    """Edge case behaviors."""

    def test_end_page_equals_start_page(self, monkeypatch, dummy_ctx, fake_zot):
        """When end_page == start_page, should behave like single page."""
        _patch_fitz(monkeypatch, [FakePage("p1"), FakePage("p2"), FakePage("p3")])
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "Test Paper"),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=2, end_page=2, ctx=dummy_ctx)

        assert "## Page 2" in result
        assert "## Page 1" not in result
        assert "## Page 3" not in result

    def test_reads_last_page(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_fitz(monkeypatch, [FakePage("first"), FakePage("last")])
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "Paper"),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=2, ctx=dummy_ctx)

        assert "## Page 2" in result
        assert "last" in result

    def test_empty_page_text(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_fitz(monkeypatch, [FakePage(""), FakePage("has text"), FakePage("")])
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "Paper"),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=1, end_page=3, ctx=dummy_ctx)

        assert "[No extractable text on this page]" in result
        assert "has text" in result
