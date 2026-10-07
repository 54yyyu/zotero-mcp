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
import os
import re
import sys
import threading
from collections import OrderedDict
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


# Parse memo for interactive reads. pdf-inspector's Markdown pass costs about
# the same for one page as for the whole document (3.7-6 s on some 28-page
# papers, and it holds the GIL), so a model reading a paper in chunks would
# pay it on every call. Callers that opt in with ``reuse=True`` share one
# whole-document parse per file, keyed on path, mtime and size so an edited
# file is re-read. Bounded by entry count and by total characters, so a few
# thousand-page books cannot pin memory.
_MEMO_MAX_ENTRIES = 3
_MEMO_MAX_CHARS = 8_000_000

_ParseKey = tuple[str, int, int]
_parse_memo: "OrderedDict[_ParseKey, ExtractedDoc]" = OrderedDict()
_memo_lock = threading.Lock()
_parse_lock = threading.Lock()


def _parse_key(path: str) -> _ParseKey | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (path, st.st_mtime_ns, st.st_size)


def _memo_get(key: _ParseKey | None) -> ExtractedDoc | None:
    if key is None:
        return None
    with _memo_lock:
        doc = _parse_memo.get(key)
        if doc is not None:
            _parse_memo.move_to_end(key)
        return doc


def _memo_put(key: _ParseKey, doc: ExtractedDoc) -> None:
    if len(doc.text) > _MEMO_MAX_CHARS:
        return
    with _memo_lock:
        _parse_memo[key] = doc
        _parse_memo.move_to_end(key)
        while len(_parse_memo) > _MEMO_MAX_ENTRIES or (
            sum(len(d.text) for d in _parse_memo.values()) > _MEMO_MAX_CHARS
            and len(_parse_memo) > 1
        ):
            _parse_memo.popitem(last=False)


def _extract_pdf_reused(
    path: str, pages: list[int] | None, max_pages: int | None,
) -> ExtractedDoc | None:
    """Serve a page subset from the memoized whole-document parse.

    Returns ``None`` when the whole document cannot be parsed page by page
    (a parse error, or the whole-document text-layer fallback), so the caller
    runs its ordinary subset path and keeps its error and fallback behaviour.
    """
    key = _parse_key(path)
    whole = _memo_get(key)
    if whole is None and key is not None:
        with _parse_lock:
            whole = _memo_get(key)
            if whole is None:
                try:
                    whole = extract_pdf(path)
                except Exception:
                    return None
                # Kept even when it is not page by page (a scanner's text
                # layer): the next read then skips straight to the subset
                # path instead of parsing the whole file a second time.
                _memo_put(key, whole)
    if whole is None or whole.page_numbers != tuple(range(whole.page_count)):
        return None

    total = whole.page_count
    truncated = False
    if pages is not None:
        wanted = [p for p in pages if 0 <= p < total]
    elif max_pages is not None and max_pages > 0:
        wanted = list(range(min(max_pages, total)))
        truncated = len(wanted) < total
    else:
        return whole
    return _doc_from_pages(
        [whole.pages[p] for p in wanted],
        page_count=total,
        source="pdf",
        page_numbers=tuple(wanted),
        needs_ocr=tuple(p for p in wanted if p in whole.needs_ocr),
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
    memoized = _memo_get(_parse_key(str(file_path)))
    if memoized is not None:
        return memoized.page_count
    return _pdf_inspector().classify_pdf(str(file_path)).page_count


def extract_pdf(
    file_path: str | Path,
    *,
    pages: list[int] | None = None,
    max_pages: int | None = None,
    reuse: bool = False,
) -> ExtractedDoc:
    """Extract Markdown from a PDF.

    Args:
        file_path: Path to the PDF.
        pages: Explicit 0-indexed pages to extract, in the order wanted.
            Out-of-range entries are dropped rather than returned as blank
            pages. Mutually exclusive with ``max_pages``.
        max_pages: Extract only the first N pages. ``None`` or a
            non-positive value means the whole document.
        reuse: Parse the whole document once and serve this and later
            requests for the same unchanged file from a small in-process
            memo. For interactive reads that revisit a file (page-range
            reads, ``zotero_get_item_fulltext``); leave it off for bulk
            indexing, which reads each file once and relies on the cap to
            skip pages.

    Raises:
        ImportError: pdf-inspector is not installed.
        ValueError: the file is missing, empty, or not a PDF.
    """
    if pages is not None and max_pages is not None:
        raise TypeError("pass either pages or max_pages, not both")

    if reuse and (pages is not None or max_pages is not None):
        reused = _extract_pdf_reused(str(file_path), pages, max_pages)
        if reused is not None:
            return reused

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

    # The text-layer fallback is whole-document, so it can't stand in for an
    # explicit subset of pages; a max_pages head is capped proportionally.
    can_fall_back = pages is None or len(set(wanted)) == total

    try:
        result = pdf_inspector.extract_pages_markdown(path, pages=wanted)
    except Exception as exc:
        # Some producers (Ghostscript / PDFCreator) emit content streams the
        # markdown pass rejects while the plain-text pass reads them (#611).
        fallback = can_fall_back and _text_layer_fallback(
            pdf_inspector, path, wanted=wanted, total=total, truncated=truncated,
        )
        if not fallback:
            raise
        logger.info(
            "pdf-inspector markdown failed for %s (%s); used the text layer",
            path, exc,
        )
        return fallback

    if total is None:
        total = len(result.pages)

    # Scanner OCR layers (invisible text over a page image, as written by
    # Xerox/ABBYY devices) come back as empty markdown with needs_ocr on every
    # page, although the plain-text pass returns that layer (#611).
    if (
        can_fall_back
        and result.pages
        and not any((page.markdown or "").strip() for page in result.pages)
    ):
        fallback = _text_layer_fallback(
            pdf_inspector, path, wanted=wanted, total=total, truncated=truncated,
        )
        if fallback is not None:
            return fallback

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


def _text_layer_fallback(
    pdf_inspector, path: str, *, wanted: list[int] | None,
    total: int | None, truncated: bool,
) -> ExtractedDoc | None:
    """Recover a PDF's text layer when the markdown pass yields nothing.

    pdf-inspector's ``extract_text`` is whole-document only (its per-page and
    positional APIs skip invisible text), so the result is one page attributed
    to the first page requested. Under a ``max_pages`` cap the text is cut to
    the same share of the document, so the indexer's limit still holds.
    Returns ``None`` when there is no text layer either.
    """
    try:
        text = pdf_inspector.extract_text(path) or ""
    except Exception:
        return None
    if not text.strip():
        return None
    # The separator must only ever mark page boundaries.
    text = text.replace(PAGE_SEPARATOR, "\n")
    if total is None:
        try:
            total = pdf_page_count(path)
        except Exception:
            total = 1
    if truncated and wanted and total:
        text = text[: len(text) * len(wanted) // total]
    first = wanted[0] if wanted else 0
    return _doc_from_pages(
        [text], page_count=total, source="pdf",
        page_numbers=(first,), truncated=truncated,
    )


def _read_text(file_path: str | Path) -> str:
    """Decode an attachment to text with uniform newlines.

    Newlines are normalized to ``\\n`` so every extraction path emits the
    same thing: pdf-inspector and markdownify already do, and without this a
    CRLF attachment would be the odd one out — putting ``\\r\\n`` into
    embeddings and quoted snippets on Windows but not elsewhere, for the
    same document.
    """
    raw = Path(file_path).read_bytes().decode("utf-8", errors="replace")
    return raw.replace("\r\n", "\n").replace("\r", "\n")


def _inline_image_placeholder(alt: str) -> str:
    """What an embedded ``data:`` image becomes in the Markdown.

    Snapshots saved by the Zotero Connector inline every image as a
    ``data:`` URI, and markdownify copies each one into ``![alt](data:...)``
    verbatim. On one real snapshot that was 6.1M characters of base64 around
    33K characters of text, which no agent can read and no embedding model
    can use.

    A short marker keeps the fact that a figure was there (so an agent can
    say so, or open the PDF) for a few tokens. Brackets are removed from the
    alt text so the marker stays balanced when the image sits inside a link.
    """
    label = " ".join(alt.replace("[", " ").replace("]", " ").split())
    return f"[image: {label}]" if label else "[image]"


def _is_data_uri(value) -> bool:
    return isinstance(value, str) and value.lstrip()[:5].lower() == "data:"


def _html_converter():
    """A markdownify converter that drops embedded ``data:`` URIs.

    markdownify copies a URL into the Markdown from three elements: an
    image's ``src``, a video's ``poster`` or ``src`` (or its first
    ``<source>``), and a link's ``href``. Each is checked here.
    """
    from markdownify import MarkdownConverter

    class _SnapshotConverter(MarkdownConverter):
        def convert_img(self, el, text, parent_tags):
            if _is_data_uri(el.attrs.get("src")):
                return _inline_image_placeholder(el.attrs.get("alt") or "")
            return super().convert_img(el, text, parent_tags)

        def convert_video(self, el, text, parent_tags):
            for attr in ("poster", "src"):
                if _is_data_uri(el.attrs.get(attr)):
                    del el.attrs[attr]
            for source in el.find_all("source"):
                if _is_data_uri(source.attrs.get("src")):
                    source.decompose()
            return super().convert_video(el, text, parent_tags)

        def convert_a(self, el, text, parent_tags):
            if _is_data_uri(el.attrs.get("href")):
                return text
            return super().convert_a(el, text, parent_tags)

    return _SnapshotConverter(heading_style="ATX")


#: A run of this many empty (or ``---``) table cells is layout scaffolding.
_TABLE_RUN = 4
_EMPTY_CELL_RUN = re.compile(r"\|(?:[ \t]*\|){%d,}" % _TABLE_RUN)
_SEPARATOR_RUN = re.compile(r"\|(?: ?-{3,} ?\|){%d,}" % _TABLE_RUN)
#: A separator inside a line with other cells never occurs in a real table row.
_NESTED_SEPARATOR = re.compile(r"\|(?: ?-{3,} ?\|){2,}")


def _collapse_table_scaffolding(markdown: str) -> str:
    """Shorten the runs of empty table cells markdownify emits for layout tables.

    ``html.parser`` does not close an unterminated ``<td>``/``<tr>``, so a
    snapshot whose table omits those end tags (legal HTML) nests every row
    inside the previous one, and markdownify then prints, for each nested
    row, a blank header line and a ``| --- |`` line as wide as the whole
    table. On a 74-row statistics table that was 200K of ``|  |  |`` and
    ``| --- | --- |`` around 17K of data, with a single line of 125K
    characters.

    Only that scaffolding is shortened, so real tables keep their shape: a
    line of nothing but empty cells, a separator line under such a blank
    header, and a line that carries a separator in the middle of its cells.
    A header with text keeps its separator, and a data row keeps its empty
    cells. Every cell with text is kept.
    """
    out = []
    for line in markdown.split("\n"):
        if line.startswith("|"):
            stripped = line.rstrip()
            if _EMPTY_CELL_RUN.fullmatch(stripped):
                line = "|  |"
            elif _SEPARATOR_RUN.fullmatch(stripped):
                if out and out[-1] == "|  |":
                    line = "| --- |"
            elif _NESTED_SEPARATOR.search(line):
                line = _SEPARATOR_RUN.sub("| --- |", _EMPTY_CELL_RUN.sub("|  |", line))
            if line in ("|  |", "| --- |") and out and out[-1] == line:
                continue
        out.append(line)
    return "\n".join(out)
#: Recursion limit for a retry after a snapshot overflowed the default one.
#: markdownify recurses once or twice per nesting level, and html.parser does
#: not close unterminated tags, so a long page of ``<li>`` or ``<td>`` without
#: end tags nests hundreds of levels deep. 5000 covers DOM depths up to about
#: 2000; a page nested deeper than that is generated markup rather than text
#: and keeps failing over to the caller's fallback instead of ballooning (one
#: 3,830-level snapshot converts to 29M characters in 18 s and 1.5 GB).
_DEEP_RECURSION_LIMIT = 5000
#: Stack for the retry thread. CPython before 3.11 spends C stack on every
#: Python call and worker threads can have as little as 512 KB, so the retry
#: gets its own rather than risking a crash at the raised limit.
_DEEP_STACK_BYTES = 64 * 1024 * 1024
_deep_lock = threading.Lock()


def _convert_html(text: str) -> str:
    """Markdown for ``text``, retrying once with more stack for deep pages."""
    try:
        return _html_converter().convert(text)
    except RecursionError as first:
        result: dict[str, object] = {}

        def convert() -> None:
            try:
                result["text"] = _html_converter().convert(text)
            except BaseException as exc:  # handed back to the caller's thread
                result["error"] = exc

        # The recursion limit and the default thread stack size are both
        # process-wide, so one deep conversion at a time, restored afterwards.
        with _deep_lock:
            limit = sys.getrecursionlimit()
            stack = threading.stack_size()
            try:
                sys.setrecursionlimit(max(limit, _DEEP_RECURSION_LIMIT))
                try:
                    threading.stack_size(_DEEP_STACK_BYTES)
                    worker = threading.Thread(target=convert, daemon=True)
                    worker.start()
                except (RuntimeError, ValueError):
                    raise first from None
                finally:
                    threading.stack_size(stack)
                worker.join()
            finally:
                sys.setrecursionlimit(limit)
        if "error" in result:
            raise result["error"]  # type: ignore[misc]
        return result["text"]  # type: ignore[return-value]


def extract_html(file_path: str | Path) -> ExtractedDoc:
    """Convert an HTML snapshot to Markdown, without embedded image data."""
    markdown = _convert_html(_read_text(file_path)).strip()
    return _doc_from_pages(
        [_collapse_table_scaffolding(markdown)],
        page_count=1,
        source="html",
    )


def extract_text_file(file_path: str | Path) -> ExtractedDoc:
    """Read an already-textual attachment (.txt, .md, .vtt, ...)."""
    return _doc_from_pages([_read_text(file_path)], page_count=1, source="text")


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
    reuse: bool = False,
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
            return extract_pdf(path, max_pages=max_pages, reuse=reuse)
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
