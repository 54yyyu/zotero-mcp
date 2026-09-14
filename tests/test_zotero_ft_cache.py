"""Regression tests for #291: read .zotero-ft-cache; survive filename drift.

Local fulltext extraction used to depend entirely on ``itemAttachments.path``
matching a file on disk. If Zotero (or a sync tool, or a non-ASCII rename)
changed the on-disk filename, ``_resolve_attachment_path`` returned ``None``,
``_extract_fulltext_for_item`` returned ``None``, and the indexer marked the
item ``has_fulltext='failed'`` — which incremental updates then skip until
``--force-rebuild``.

Two fixes here:

- ``.zotero-ft-cache``: Zotero writes a plain-text already-extracted
  full-text cache next to each indexed PDF. Reading it skips our own
  extraction entirely and is immune to filename drift.
- Storage-dir scan fallback: when the recorded path doesn't resolve, look
  in the attachment's storage folder for a file whose extension matches
  the recorded content type.
"""

from pathlib import Path

import pytest

from conftest import extracted_doc, skip_on_ci
from zotero_mcp.extract import DEFAULT_ATTACHMENT_PRIORITY
from zotero_mcp.local_db import LocalZoteroReader


class _Reader(LocalZoteroReader):
    """LocalZoteroReader stub: no DB, attachments / storage dir injected."""

    def __init__(self, attachments, storage_dir: Path):
        self.db_path = "/dev/null"
        self._connection = None
        self.pdf_max_pages = 10
        self.attachment_priority = DEFAULT_ATTACHMENT_PRIORITY
        self._attachments = attachments
        self._storage_dir = storage_dir

    def _iter_parent_attachments(self, _parent_item_id: int):
        yield from self._attachments

    def _get_storage_dir(self) -> Path:
        return self._storage_dir


# ---------------------------------------------------------------------------
# .zotero-ft-cache as the fallback path
# ---------------------------------------------------------------------------


@skip_on_ci
def test_ft_cache_covers_an_attachment_whose_file_cannot_be_resolved(tmp_path):
    """The cache is keyed by attachment key, not filename, so it still answers
    when the recorded filename has drifted and no file is found on disk."""
    storage = tmp_path / "storage"
    attachment_dir = storage / "ABCDEFGH"
    attachment_dir.mkdir(parents=True)
    (attachment_dir / ".zotero-ft-cache").write_text(
        "Full body text extracted by Zotero, including the conclusion."
    )
    # Recorded sqlite path points at a filename that no longer exists, and the
    # storage scan finds no PDF beside it either.
    reader = _Reader(
        attachments=[
            ("ABCDEFGH", "storage:original-filename-renamed.pdf", "application/pdf")
        ],
        storage_dir=storage,
    )

    def _boom(_path):
        raise AssertionError("nothing resolvable to extract — must not be called")

    reader._extract_doc_from_file = _boom  # type: ignore[assignment]

    result = reader._extract_fulltext_for_item(item_id=1)
    assert result is not None
    text, source = result
    assert "Full body text" in text
    assert source == "zotero-cache"


@skip_on_ci
def test_our_extraction_wins_over_the_ft_cache(tmp_path):
    """The cache is flat pdftotext with no headings and usually no page
    separators, so a file we can parse ourselves takes precedence."""
    storage = tmp_path / "storage"
    attachment_dir = storage / "BOTHAVAIL"
    attachment_dir.mkdir(parents=True)
    (attachment_dir / ".zotero-ft-cache").write_text("flat cached text")
    pdf = attachment_dir / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 stand-in")

    reader = _Reader(
        attachments=[("BOTHAVAIL", "storage:paper.pdf", "application/pdf")],
        storage_dir=storage,
    )
    reader._resolve_attachment_path = lambda _k, _p: pdf  # type: ignore[assignment]
    reader._extract_doc_from_file = (  # type: ignore[assignment]
        lambda _path: extracted_doc("## Heading\n\nparsed markdown")
    )

    text, source = reader._extract_fulltext_for_item(item_id=1)
    assert source == "pdf"
    assert "parsed markdown" in text
    assert "flat cached text" not in text


@skip_on_ci
def test_ft_cache_rescues_a_file_that_fails_to_parse(tmp_path):
    """A resolvable but unparseable file must not shadow a usable cache."""
    storage = tmp_path / "storage"
    attachment_dir = storage / "CORRUPTP"
    attachment_dir.mkdir(parents=True)
    (attachment_dir / ".zotero-ft-cache").write_text("cached text for a corrupt PDF")
    pdf = attachment_dir / "paper.pdf"
    pdf.write_bytes(b"not actually a pdf")

    reader = _Reader(
        attachments=[("CORRUPTP", "storage:paper.pdf", "application/pdf")],
        storage_dir=storage,
    )
    reader._resolve_attachment_path = lambda _k, _p: pdf  # type: ignore[assignment]
    # What extract_file does for an unreadable file: log and yield nothing.
    reader._extract_doc_from_file = lambda _path: None  # type: ignore[assignment]

    text, source = reader._extract_fulltext_for_item(item_id=1)
    assert source == "zotero-cache"
    assert "cached text for a corrupt PDF" in text


@skip_on_ci
def test_empty_zotero_ft_cache_is_skipped(tmp_path):
    """An empty .zotero-ft-cache file shouldn't masquerade as fulltext."""
    storage = tmp_path / "storage"
    attachment_dir = storage / "EMPTYCAC"
    attachment_dir.mkdir(parents=True)
    (attachment_dir / ".zotero-ft-cache").write_text("")
    reader = _Reader(
        attachments=[("EMPTYCAC", "storage:p.pdf", "application/pdf")],
        storage_dir=storage,
    )
    # No PDF on disk and the cache is empty — must return None, not fake hits.
    assert reader._extract_fulltext_for_item(item_id=1) is None


# ---------------------------------------------------------------------------
# Storage-dir scan fallback
# ---------------------------------------------------------------------------


@skip_on_ci
def test_storage_scan_recovers_from_renamed_pdf(tmp_path):
    """If the recorded filename doesn't resolve but the storage folder
    contains a PDF, use it instead of failing the extraction (#291)."""
    storage = tmp_path / "storage"
    attachment_dir = storage / "RENAMED1"
    attachment_dir.mkdir(parents=True)
    # The actual file has a different name than what sqlite recorded.
    on_disk = attachment_dir / "actual-filename.pdf"
    on_disk.write_bytes(b"%PDF-1.4 fake content")

    reader = _Reader(
        attachments=[
            ("RENAMED1", "storage:recorded-name-but-not-on-disk.pdf", "application/pdf")
        ],
        storage_dir=storage,
    )

    captured: dict[str, Path] = {}

    def _fake_extract(path):
        captured["path"] = path
        return extracted_doc("extracted via scan fallback")

    reader._extract_doc_from_file = _fake_extract  # type: ignore[assignment]

    text, source = reader._extract_fulltext_for_item(item_id=1)
    assert text == "extracted via scan fallback"
    assert source == "pdf"
    assert captured["path"] == on_disk


@skip_on_ci
def test_storage_scan_picks_largest_when_multiple_pdfs(tmp_path):
    storage = tmp_path / "storage"
    attachment_dir = storage / "MULTIPDF"
    attachment_dir.mkdir(parents=True)
    small = attachment_dir / "thumbnail.pdf"
    small.write_bytes(b"%PDF-tiny")
    big = attachment_dir / "body.pdf"
    big.write_bytes(b"%PDF-" + b"x" * 500)

    reader = _Reader(
        attachments=[("MULTIPDF", "storage:missing.pdf", "application/pdf")],
        storage_dir=storage,
    )
    chosen = reader._scan_storage_for_attachment("MULTIPDF", "application/pdf")
    assert chosen == big


@skip_on_ci
def test_storage_scan_no_match_returns_none(tmp_path):
    storage = tmp_path / "storage"
    attachment_dir = storage / "WRONGSUF"
    attachment_dir.mkdir(parents=True)
    (attachment_dir / "metadata.json").write_text("{}")  # not a PDF
    reader = _Reader(
        attachments=[("WRONGSUF", "storage:missing.pdf", "application/pdf")],
        storage_dir=storage,
    )
    assert reader._scan_storage_for_attachment("WRONGSUF", "application/pdf") is None


@skip_on_ci
def test_extract_returns_none_when_neither_cache_nor_scan_helps(tmp_path):
    """Pure absence of both: cache missing AND scan finds nothing → None."""
    storage = tmp_path / "storage"
    storage.mkdir()
    reader = _Reader(
        attachments=[("ABSENT00", "storage:nope.pdf", "application/pdf")],
        storage_dir=storage,
    )
    assert reader._extract_fulltext_for_item(item_id=1) is None
