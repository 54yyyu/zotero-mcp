"""Attachment text extraction — the single place a file becomes text.

Every fulltext path in zotero-mcp funnels through here: the local-storage
reader (``local_db``), the Web API download path (``client``), the
semantic indexer, and ``zotero_read_pdf_pages``. Keeping one seam means
page provenance, OCR routing and engine choice are decided once instead of
drifting across three call sites.

PDFs are parsed by `pdf-inspector <https://github.com/firecrawl/pdf-inspector>`_,
a Rust parser with prebuilt wheels and no Python dependencies. It runs
in-process in roughly a tenth of a second for a journal article, emits
Markdown with real heading structure, and reports which pages carry no text
layer (:attr:`ExtractedDoc.needs_ocr`) — the routing signal an OCR backend
will consume later.

``ExtractedDoc.text`` joins pages with :data:`PAGE_SEPARATOR` so that
character offsets can always be mapped back to a page number. Do not
assemble page-joined text anywhere else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Form feed, inserted between pages of :attr:`ExtractedDoc.text`. Chunk
#: provenance in ``semantic_search`` counts these to recover a page number,
#: so the separator must never appear for any other reason.
PAGE_SEPARATOR = "\f"

_HTML_SUFFIXES = frozenset({".html", ".htm", ".xhtml"})

# Extensions / MIME types we can read as plain text. Used by
# :func:`is_extractable` to gate non-PDF/HTML attachments into the fulltext
# extractor. Binary formats (.docx, .pptx, .epub, video, etc.) are
# deliberately excluded — decoding those as text yields garbage, and we
# don't want it in the semantic index.
_TEXTUAL_SUFFIXES = frozenset({
    ".txt", ".vtt", ".srt", ".sbv", ".md", ".markdown", ".rst",
    ".csv", ".tsv", ".json", ".xml", ".log", ".text",
})
_TEXTUAL_CONTENT_TYPES = frozenset({
    "text/plain", "text/vtt", "text/markdown", "text/csv",
    "text/tab-separated-values", "text/srt", "application/json",
    "application/xml", "text/xml",
})

_MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
_MARKDOWN_CONTENT_TYPES = frozenset({"text/markdown", "text/x-markdown"})

#: Categories an attachment can be sorted into for ``attachment_priority``.
#: ``"other"`` is the catch-all rather than a category anything is labelled
#: with — see :func:`pick_by_priority`.
ATTACHMENT_CATEGORIES = ("pdf", "html", "markdown", "text", "other")

#: Priority applied when nothing is configured: PDF, then HTML snapshot, then
#: whatever else is readable. Listing only these three keeps the historical
#: behaviour exactly, because ``"other"`` sweeps up markdown and plain text
#: together and the larger file wins.
DEFAULT_ATTACHMENT_PRIORITY = ("pdf", "html", "other")


@dataclass(frozen=True)
class ExtractedDoc:
    """Text extracted from one attachment file.

    ``pages`` is per-page for PDFs and a single entry for everything else,
    so ``page_count`` is 1 for non-paginated sources rather than 0.
    """

    #: Pages joined by :data:`PAGE_SEPARATOR`.
    text: str
    #: Per-page content, in the order requested.
    pages: tuple[str, ...]
    #: 0-indexed source page number for each entry in :attr:`pages`. Empty
    #: for non-paginated sources.
    page_numbers: tuple[int, ...]
    #: Pages in the source document, before any limit was applied.
    page_count: int
    #: Source kind: ``"pdf"``, ``"html"`` or ``"text"``.
    source: str
    #: 0-indexed pages with no usable text layer, per pdf-inspector's
    #: classifier. Always empty for non-PDF sources. An OCR backend routes
    #: on this; nothing reads it yet.
    needs_ocr: tuple[int, ...] = ()
    #: True when a ``max_pages`` limit dropped pages from the tail.
    truncated: bool = False

    def __bool__(self) -> bool:
        """An extraction that produced no text is falsy."""
        return bool(self.text.strip())


def _doc_from_pages(
    pages: list[str],
    *,
    page_count: int,
    source: str,
    page_numbers: tuple[int, ...] = (),
    needs_ocr: tuple[int, ...] = (),
    truncated: bool = False,
) -> ExtractedDoc:
    return ExtractedDoc(
        text=PAGE_SEPARATOR.join(pages),
        pages=tuple(pages),
        page_numbers=page_numbers,
        page_count=page_count,
        source=source,
        needs_ocr=needs_ocr,
        truncated=truncated,
    )


def pdf_page_count(file_path: str | Path) -> int:
    """Return the number of pages in a PDF.

    Cheaper than a full extraction (pdf-inspector skips building the page
    content), so callers that need to validate a page range should use this
    rather than extracting and measuring.

    Raises:
        ImportError: pdf-inspector is not installed.
        ValueError: the file is missing, empty, or not a PDF.
    """
    return _pdf_inspector().classify_pdf(str(file_path)).page_count


def extract_pdf(
    file_path: str | Path,
    *,
    pages: list[int] | None = None,
    max_pages: int | None = None,
) -> ExtractedDoc:
    """Extract Markdown from a PDF.

    Args:
        file_path: Path to the PDF.
        pages: Explicit 0-indexed pages to extract, in the order wanted.
            Out-of-range entries are dropped rather than returned as blank
            pages. Mutually exclusive with ``max_pages``.
        max_pages: Extract only the first N pages. ``None`` or a
            non-positive value means the whole document.

    Raises:
        ImportError: pdf-inspector is not installed.
        ValueError: the file is missing, empty, or not a PDF.
    """
    if pages is not None and max_pages is not None:
        raise TypeError("pass either pages or max_pages, not both")

    pdf_inspector = _pdf_inspector()
    path = str(file_path)

    # Both limited modes need the true page count first: pdf-inspector
    # returns an empty page rather than an error for an out-of-range index,
    # which would otherwise pad the output and desynchronize page numbering.
    truncated = False
    if pages is not None:
        total = pdf_page_count(path)
        wanted = [p for p in pages if 0 <= p < total]
        if not wanted:
            return _doc_from_pages([], page_count=total, source="pdf")
    elif max_pages is not None and max_pages > 0:
        total = pdf_page_count(path)
        wanted = list(range(min(max_pages, total)))
        truncated = len(wanted) < total
    else:
        wanted = None
        total = None

    result = pdf_inspector.extract_pages_markdown(path, pages=wanted)
    if total is None:
        total = len(result.pages)

    return _doc_from_pages(
        [page.markdown or "" for page in result.pages],
        page_count=total,
        source="pdf",
        # Read page indices off the page objects rather than
        # ``pages_needing_ocr``: these stay absolute when a subset was
        # requested, so they remain comparable to ``page_numbers``.
        page_numbers=tuple(page.page for page in result.pages),
        needs_ocr=tuple(page.page for page in result.pages if page.needs_ocr),
        truncated=truncated,
    )


def extract_html(file_path: str | Path) -> ExtractedDoc:
    """Convert an HTML snapshot to Markdown."""
    from markdownify import markdownify

    html = Path(file_path).read_bytes().decode("utf-8", errors="replace")
    return _doc_from_pages(
        [markdownify(html, heading_style="ATX").strip()],
        page_count=1,
        source="html",
    )


def extract_text_file(file_path: str | Path) -> ExtractedDoc:
    """Read an already-textual attachment (.txt, .md, .vtt, ...)."""
    text = Path(file_path).read_bytes().decode("utf-8", errors="replace")
    return _doc_from_pages([text], page_count=1, source="text")


def is_extractable(file_path: str | Path, ctype: str | None = None) -> bool:
    """Return True when :func:`extract_file` can get useful text out of a file.

    PDF and HTML have dedicated extractors and are reported separately by
    callers that bucket attachments by kind, so this covers only the plain
    text case: accept iff the MIME type or extension is on the textual
    allowlist, never an arbitrary binary.
    """
    normalized = (ctype or "").lower()
    if normalized.startswith("text/") or normalized in _TEXTUAL_CONTENT_TYPES:
        return True
    return Path(file_path).suffix.lower() in _TEXTUAL_SUFFIXES


def categorize_attachment(file_path: str | Path, ctype: str | None = None) -> str | None:
    """Sort an attachment into an :data:`ATTACHMENT_CATEGORIES` bucket.

    Returns ``None`` for anything we cannot read at all (a .docx, a video),
    so callers can drop it before priority is applied. Never returns
    ``"other"``: that is a catch-all in the priority list, not a label.
    Markdown is reported separately from plain text even though both are
    read the same way, because preferring a hand-converted Markdown copy
    over the original PDF is the whole point of configuring this (#378).
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    normalized = (ctype or "").lower().split(";")[0].strip()

    if normalized == "application/pdf" or suffix == ".pdf":
        return "pdf"
    if normalized.startswith("text/html") or suffix in _HTML_SUFFIXES:
        return "html"
    if normalized in _MARKDOWN_CONTENT_TYPES or suffix in _MARKDOWN_SUFFIXES:
        return "markdown"
    if is_extractable(path, ctype):
        return "text"
    return None


def normalize_attachment_priority(priority) -> tuple[str, ...]:
    """Validate a configured priority list, falling back where it is unusable.

    Unknown names are dropped with a warning rather than raising: a typo in
    a config file should cost the user that one entry, not every fulltext
    lookup. A list that ends up empty falls back to the default entirely.
    """
    if not priority:
        return DEFAULT_ATTACHMENT_PRIORITY
    if isinstance(priority, str):
        priority = [priority]

    cleaned: list[str] = []
    for entry in priority:
        name = str(entry).strip().lower()
        if name not in ATTACHMENT_CATEGORIES:
            logger.warning(
                "Ignoring unknown attachment_priority entry %r (expected one of %s)",
                entry, ", ".join(ATTACHMENT_CATEGORIES),
            )
            continue
        if name not in cleaned:
            cleaned.append(name)

    if not cleaned:
        logger.warning(
            "attachment_priority had no usable entries; using the default %s",
            list(DEFAULT_ATTACHMENT_PRIORITY),
        )
        return DEFAULT_ATTACHMENT_PRIORITY
    return tuple(cleaned)


def pick_by_priority(candidates, priority=DEFAULT_ATTACHMENT_PRIORITY):
    """Choose one attachment from ``candidates`` by category priority.

    ``candidates`` is an iterable of ``(category, size, value)``. The first
    priority entry with any match wins, and within that bucket the largest
    ``size`` wins — for PDFs that reliably picks the body over a stub or a
    thumbnail. Returns the chosen ``value``, or ``None`` if nothing matched.

    ``"other"`` matches every category *not named elsewhere* in the list. So
    the default ``("pdf", "html", "other")`` sweeps markdown and plain text
    into one final bucket, while ``("markdown", "pdf", "html", "other")``
    pulls markdown out in front and leaves the rest to the sweep. A category
    that is neither named nor covered by an ``"other"`` entry is never
    chosen, which is how a caller opts out of a format entirely.
    """
    candidates = list(candidates)
    if not candidates:
        return None
    named = {entry for entry in priority if entry != "other"}
    for entry in priority:
        if entry == "other":
            bucket = [c for c in candidates if c[0] not in named]
        else:
            bucket = [c for c in candidates if c[0] == entry]
        if bucket:
            return max(bucket, key=lambda c: c[1])[2]
    return None


def extract_file(
    file_path: str | Path,
    ctype: str | None = None,
    *,
    max_pages: int | None = None,
) -> ExtractedDoc | None:
    """Extract text from an attachment, dispatching on extension.

    The tolerant entry point: unlike the per-format functions it logs and
    returns ``None`` instead of raising, because its callers are walking a
    library where any single unreadable attachment should be skipped rather
    than abort the batch.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return extract_pdf(path, max_pages=max_pages)
        if suffix in _HTML_SUFFIXES:
            return extract_html(path)
        return extract_text_file(path)
    except Exception as exc:
        logger.warning("Extraction failed for %s: %s", path.name, exc)
        return None


def _pdf_inspector():
    """Import pdf-inspector, with an actionable message when it's absent.

    It is a core dependency with prebuilt wheels for every platform we
    support, so a failure here means a broken install rather than a missing
    extra.
    """
    try:
        import pdf_inspector
    except ImportError as exc:  # pragma: no cover - depends on install state
        from .utils import install_hint

        raise ImportError(
            f"pdf-inspector is required for PDF text extraction. {install_hint()}"
        ) from exc
    return pdf_inspector
