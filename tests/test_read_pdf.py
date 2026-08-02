"""Tests for zotero_read_pdf_pages tool."""

import tempfile

import pytest
from conftest import DummyContext, FakeZotero

from zotero_mcp import server
from zotero_mcp.extract import PAGE_SEPARATOR, ExtractedDoc
from zotero_mcp.tools import read_pdf as read_pdf_tools

# ---------------------------------------------------------------------------
# Helpers: fake the extraction seam
# ---------------------------------------------------------------------------


def _patch_extract(monkeypatch, page_texts, total=None, needs_ocr=()):
    """Stand in for ``extract_pdf``/``pdf_page_count`` with known page text.

    Mirrors the real contract the tool depends on: out-of-range indices are
    dropped, and ``page_numbers`` reports the absolute source page for each
    returned page.
    """
    total_pages = total if total is not None else len(page_texts)

    def _fake_page_count(_path):
        return total_pages

    def _fake_extract_pdf(_path, *, pages=None, max_pages=None):
        wanted = [
            p for p in (range(total_pages) if pages is None else pages)
            if 0 <= p < total_pages
        ]
        texts = [page_texts[p % len(page_texts)] for p in wanted]
        return ExtractedDoc(
            text=PAGE_SEPARATOR.join(texts),
            pages=tuple(texts),
            page_numbers=tuple(wanted),
            page_count=total_pages,
            source="pdf",
            needs_ocr=tuple(needs_ocr),
        )

    monkeypatch.setattr("zotero_mcp.tools.read_pdf.pdf_page_count", _fake_page_count)
    monkeypatch.setattr("zotero_mcp.tools.read_pdf.extract_pdf", _fake_extract_pdf)


def _patch_extract_failure(monkeypatch, exc):
    """Make the seam raise, as it does for a corrupt or non-PDF file."""
    def _boom(_path):
        raise exc

    monkeypatch.setattr("zotero_mcp.tools.read_pdf.pdf_page_count", _boom)


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
        _patch_extract(monkeypatch, ["Page 1 content."] * 10, total=10)
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "Test Paper", True),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=3, ctx=dummy_ctx)

        assert "## Page 3" in result
        assert "Page 1 content." in result

    def test_page_range(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_extract(monkeypatch, [
            "Content of page 1.",
            "Content of page 2.",
            "Content of page 3.",
            "Content of page 4.",
            "Content of page 5.",
        ])
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "Test Paper", True),
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
        _patch_extract(monkeypatch, ["hello"])
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "My Paper Title", True),
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
        _patch_extract(monkeypatch, ["p1"], total=1)
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "Paper", True),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=5, ctx=dummy_ctx)

        assert "out of range" in result
        assert "1-1" in result

    def test_end_page_out_of_range(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_extract(monkeypatch, ["p1"] * 3, total=3)
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "Paper", True),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=1, end_page=10, ctx=dummy_ctx)

        assert "out of range" in result
        assert "1-3" in result

    def test_too_many_pages(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_extract(monkeypatch, ["p"] * 100, total=100)
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "Paper", True),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=1, end_page=55, ctx=dummy_ctx)

        assert "max 50" in result

    def test_unreadable_pdf_reports_the_reason(self, monkeypatch, dummy_ctx, fake_zot):
        """A corrupt or non-PDF file surfaces the parser's message rather
        than an empty page range."""
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "Paper", True),
        )
        _patch_extract_failure(monkeypatch, ValueError("Not a PDF: file is empty"))

        result = server.read_pdf_pages(item_key="ITEM01", start_page=1, ctx=dummy_ctx)

        assert "Could not read PDF" in result
        assert "Not a PDF" in result


class TestEdgeCases:
    """Edge case behaviors."""

    def test_end_page_equals_start_page(self, monkeypatch, dummy_ctx, fake_zot):
        """When end_page == start_page, should behave like single page."""
        _patch_extract(monkeypatch, ["p1", "p2", "p3"])
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "Test Paper", True),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=2, end_page=2, ctx=dummy_ctx)

        assert "## Page 2" in result
        assert "## Page 1" not in result
        assert "## Page 3" not in result

    def test_reads_last_page(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_extract(monkeypatch, ["first", "last"])
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "Paper", True),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=2, ctx=dummy_ctx)

        assert "## Page 2" in result
        assert "last" in result

    def test_empty_page_text(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_extract(monkeypatch, ["", "has text", ""])
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/tmp/test.pdf", "Paper", True),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=1, end_page=3, ctx=dummy_ctx)

        assert "[No extractable text on this page]" in result
        assert "has text" in result


class TestCleanupPathSafety:
    """`_cleanup_path` deletes a directory, so its guards have to be tight.

    The original guard was `parent.startswith(tempfile.gettempdir())`. On
    Linux that is `/tmp`, so `_cleanup_path("/tmp/test.pdf")` resolved its
    parent to `/tmp` and called `shutil.rmtree("/tmp")` — wiping the system
    temp directory, including pytest's own temp root, which surfaced as
    unrelated tests erroring with FileNotFoundError. macOS never showed it
    because `gettempdir()` there lives under `/var/folders`.
    """

    def test_never_removes_the_temp_root_itself(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        canary = tmp_path / "canary.txt"
        canary.write_text("do not delete me")

        read_pdf_tools._cleanup_path(str(tmp_path / "test.pdf"))

        assert tmp_path.exists()
        assert canary.exists()

    def test_ignores_directories_we_did_not_create(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        storage = tmp_path / "storage" / "ABCD1234"
        storage.mkdir(parents=True)
        pdf = storage / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        read_pdf_tools._cleanup_path(str(pdf))

        assert pdf.exists(), "a file in the user's Zotero storage must survive"

    def test_removes_our_own_download_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        owned = tmp_path / "zotero_pdf_abc123"
        owned.mkdir()
        pdf = owned / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        read_pdf_tools._cleanup_path(str(pdf))

        assert not owned.exists()

    def test_library_file_is_not_released_by_the_tool(self, monkeypatch, dummy_ctx, fake_zot):
        """A local-storage hit reports is_temp=False and must survive the read."""
        _patch_extract(monkeypatch, ["Body text."], total=1)
        removed = []
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._cleanup_path", lambda p: removed.append(p)
        )
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c: ("/home/me/Zotero/storage/ABCD/paper.pdf", "Paper", False),
        )

        server.read_pdf_pages(item_key="ITEM01", start_page=1, ctx=dummy_ctx)

        assert removed == []
