from pathlib import Path

import pytest

from zotero_mcp.local_db import ZoteroItem, LocalZoteroReader


class FakeLocalZoteroReader(LocalZoteroReader):
    """Subclass that skips DB init and allows injecting fake attachment text."""

    def __init__(self, fake_text: str = "", fake_pdf_path: Path | None = None):
        # Skip parent __init__ entirely — no DB needed
        self.db_path = "/dev/null"
        self._connection = None
        self.pdf_max_pages = 10
        self._fake_text = fake_text
        self._fake_pdf_path = fake_pdf_path

    def _iter_parent_attachments(self, parent_item_id: int):
        """Yield a single fake PDF attachment."""
        yield "FAKEKEY", "storage:fake.pdf", "application/pdf"

    def _resolve_attachment_path(self, attachment_key: str, zotero_path: str):
        """Return the injected fake path."""
        return self._fake_pdf_path

    def _extract_text_from_file(self, file_path):
        """Return the injected fake text instead of reading a real file."""
        return self._fake_text


def test_extract_fulltext_preserves_long_text(tmp_path):
    """Extracted text longer than 10,000 chars should NOT be truncated."""
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.touch()
    long_text = "x" * 25000
    reader = FakeLocalZoteroReader(fake_text=long_text, fake_pdf_path=fake_pdf)
    result = reader._extract_fulltext_for_item(1)
    assert result is not None
    text, source = result
    assert len(text) == 25000, f"Text was truncated to {len(text)} chars"
    assert source == "pdf"


def test_extract_fulltext_empty_returns_none(tmp_path):
    """Empty extracted text should return None."""
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.touch()
    reader = FakeLocalZoteroReader(fake_text="", fake_pdf_path=fake_pdf)
    result = reader._extract_fulltext_for_item(1)
    assert result is None


def test_get_searchable_text_preserves_long_fulltext():
    """get_searchable_text should not aggressively truncate fulltext."""
    long_fulltext = "y" * 20000
    item = ZoteroItem(item_id=1, key="TEST", item_type_id=1, fulltext=long_fulltext)
    text = item.get_searchable_text()
    # The full 20,000 chars should appear in the output (not truncated to 5,000)
    assert "y" * 20000 in text


def test_get_searchable_text_truncates_at_limit():
    """Fulltext beyond 50,000 chars should be truncated with ellipsis."""
    huge_fulltext = "z" * 60000
    item = ZoteroItem(item_id=1, key="TEST", item_type_id=1, fulltext=huge_fulltext)
    text = item.get_searchable_text()
    # Should contain exactly 50,000 z's plus "..." — not all 60,000
    assert "z" * 50000 in text
    assert "z" * 50001 not in text
    assert "..." in text
