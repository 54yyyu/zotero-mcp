"""
PDF utility functions for Zotero annotation creation.

This module provides text search capabilities for PDFs to extract position data
needed for creating Zotero highlight annotations. It handles common PDF text
extraction issues like:
- Hyphenation at line breaks
- Special characters (em-dashes, curly quotes, ligatures)
- Missing word spacing in extracted text
- Page number mismatches

Search Strategy (in order), each on the page given and then its neighbours:
1. For long text (>100 chars): Anchor-based matching (find start/end, highlight between)
2. Exact match in the page's normalized text index
3. Fuzzy matching against that index
"""

from __future__ import annotations

import json
import math
import os
import re
from contextlib import contextmanager
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from zotero_mcp.utils import install_hint

if TYPE_CHECKING:
    from typing import Any

# =============================================================================
# Configuration Constants
# =============================================================================

# Anchor-based matching settings
ANCHOR_MIN_TEXT_LENGTH = 100  # Use anchor matching for text longer than this
ANCHOR_TARGET_LENGTH = 40     # Target length for start/end anchors
ANCHOR_WORD_BOUNDARY_TOLERANCE = 15  # How far to extend to find word boundary
ANCHOR_MATCH_THRESHOLD = 0.75  # Minimum similarity for anchor fuzzy matching
ACROSS_PAGE_MIN_RATIO = 0.8    # How closely the next page must read like the rest of a quote

# Fuzzy matching thresholds (by text length)
FUZZY_THRESHOLD_SHORT = 0.85   # For text < 50 chars
FUZZY_THRESHOLD_MEDIUM = 0.75  # For text 50-150 chars
FUZZY_THRESHOLD_LONG = 0.65    # For text > 150 chars

# Search behavior
DEFAULT_NEIGHBOR_PAGES = 2  # How many pages to search on either side

# Performance optimization
SLIDING_WINDOW_STEP_THRESHOLD = 10000  # Use stepping for texts longer than this


# =============================================================================
# Text Normalization
# =============================================================================

# Character replacement maps for normalization
DASH_REPLACEMENTS = {
    '\u2014': '-',  # em-dash
    '\u2013': '-',  # en-dash
    '\u2012': '-',  # figure dash
    '\u2011': '-',  # non-breaking hyphen
    '\u2010': '-',  # hyphen
    '\u2212': '-',  # minus sign, which PDFs and typed text swap with the hyphen-minus
}

QUOTE_REPLACEMENTS = {
    '\u2018': "'",  # left single quote
    '\u2019': "'",  # right single quote
    '\u201c': '"',  # left double quote
    '\u201d': '"',  # right double quote
}

LIGATURE_REPLACEMENTS = {
    '\ufb01': 'fi',   # fi ligature
    '\ufb02': 'fl',   # fl ligature
    '\ufb00': 'ff',   # ff ligature
    '\ufb03': 'ffi',  # ffi ligature
    '\ufb04': 'ffl',  # ffl ligature
}


def normalize_text(text: str) -> str:
    """
    Normalize text for matching, handling common PDF extraction issues.

    Transformations applied:
    - Remove hyphenation at line breaks ("regard-\\nless" -> "regardless")
    - Remove soft hyphens
    - Normalize dashes (em-dash, en-dash, minus sign, etc.) to simple hyphen
    - Normalize curly quotes to straight quotes
    - Expand common ligatures (fi, fl, ff, etc.)
    - Collapse whitespace to single spaces

    Args:
        text: Raw text to normalize

    Returns:
        Normalized text suitable for comparison
    """
    # Remove hyphenation at line breaks
    text = re.sub(r'[\u00ad\u2010\u2011-]\s*\n\s*', '', text)
    text = text.replace('\u00ad', '')

    # Apply character replacements
    for old, new in DASH_REPLACEMENTS.items():
        text = text.replace(old, new)
    for old, new in QUOTE_REPLACEMENTS.items():
        text = text.replace(old, new)
    for old, new in LIGATURE_REPLACEMENTS.items():
        text = text.replace(old, new)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_for_matching(text: str) -> str:
    """
    Aggressively normalize text for matching.

    This removes ALL spaces and lowercases the text to handle PDFs where
    words are stored without proper spacing between spans. Hyphens and dashes
    are removed too. A hyphen at the end of a line is either typesetting
    ("regard-" / "less") or part of the word ("state-of-the-" / "art"), and
    nothing on the page says which, so a quote typed either way has to match:
    "regardless" finds the first and "state-of-the-art" the second.

    Args:
        text: Text to normalize

    Returns:
        Text with all spaces and hyphens removed, lowercased
    """
    text = normalize_text(text)
    text = re.sub(r'[\s-]+', '', text)
    return text.lower()


# =============================================================================
# Page Text Extraction
# =============================================================================

def _extract_page_spans(page) -> list[dict[str, Any]]:
    """
    Extract all text spans from a PDF page with their bounding boxes.

    Args:
        page: PyMuPDF page object

    Each span also carries its characters with their own boxes, and the line
    it sits on. A match rarely starts or ends on a span boundary -- a span is
    usually most of a line -- so without the characters a highlight covers
    every word the matched text shares a span with.

    Returns:
        List of dicts with 'text', 'bbox', 'chars' and 'line' keys
    """
    blocks = page.get_text("rawdict", flags=11)["blocks"]
    spans = []

    for block_no, block in enumerate(blocks):
        if "lines" not in block:
            continue
        for line_no, line in enumerate(block["lines"]):
            for span in line["spans"]:
                chars = span.get("chars") or []
                spans.append({
                    "text": "".join(c["c"] for c in chars) if chars else span.get("text", ""),
                    "bbox": span["bbox"],
                    "chars": chars,
                    "line": (block_no, line_no),
                })

    return spans


def _build_normalized_text_index(spans: list[dict]) -> tuple[str, list[tuple[int, int, int]]]:
    """
    Build a normalized cumulative text string and index mapping.

    This concatenates all span text (normalized) and tracks where each span's
    text appears in the cumulative string, enabling position-to-span lookups.

    Args:
        spans: List of span dicts with 'text' keys

    Returns:
        Tuple of:
        - Cumulative normalized text string
        - List of (norm_start, norm_end, span_index) tuples
    """
    parts = []
    positions = []
    end = 0

    for i, span in enumerate(spans):
        normalized = normalize_for_matching(span["text"])
        parts.append(normalized)
        positions.append((end, end + len(normalized), i))
        end += len(normalized)

    return "".join(parts), positions


def _get_spans_in_range(
    start_pos: int,
    end_pos: int,
    span_positions: list[tuple[int, int, int]],
    spans: list[dict],
) -> tuple[list, list[str]]:
    """
    Get all spans that overlap with a position range in normalized text.

    Args:
        start_pos: Start position in normalized text
        end_pos: End position in normalized text
        span_positions: Index from _build_normalized_text_index
        spans: Original span list

    Spans cut by either end of the range are clipped to the characters inside
    it, and pieces on the same line are joined into one box, so the highlight
    covers the matched text and nothing else.

    Returns:
        Tuple of (list of bboxes, list of original text strings)
    """
    pieces = []  # (line, bbox, text)

    for norm_start, norm_end, span_idx in span_positions:
        if not (norm_start < end_pos and norm_end > start_pos):
            continue
        span = spans[span_idx]
        bbox, text = span["bbox"], span["text"]
        if norm_start < start_pos or norm_end > end_pos:
            clipped = _clip_span(span, start_pos - norm_start, end_pos - norm_start)
            if clipped is not None:
                bbox, text = clipped
        pieces.append((span.get("line"), tuple(bbox), text))

    bboxes: list = []
    texts: list[str] = []
    previous_line = object()
    for line, bbox, text in pieces:
        if line is not None and line == previous_line:
            x0, y0, x1, y1 = bboxes[-1]
            bboxes[-1] = (min(x0, bbox[0]), min(y0, bbox[1]), max(x1, bbox[2]), max(y1, bbox[3]))
            texts[-1] += text
        else:
            bboxes.append(bbox)
            texts.append(text)
        previous_line = line

    return bboxes, texts


def _clip_span(span: dict, lo: int, hi: int) -> tuple[tuple, str] | None:
    """
    Box and text of the characters of a span inside [lo, hi).

    ``lo`` and ``hi`` are offsets into the span's *normalized* text. Each
    character is normalized on its own to find where it lands there; that
    reproduces the span-level normalization exactly, because every rule in
    normalize_for_matching either maps one character (dashes, quotes,
    ligatures, case) or deletes it (whitespace, hyphens).

    A deleted character has no width there. It is kept when it sits after the
    first character of the range and before the range's end, so the hyphen
    that closes a line stays inside the box of the word it breaks.

    Returns None when the span has no character data, so the caller keeps the
    whole-span box rather than dropping the piece.
    """
    chars = span.get("chars") or []
    offset = 0
    first = last = None
    box = None
    for idx, char in enumerate(chars):
        width = len(normalize_for_matching(char["c"]))
        inside = offset < hi and offset + width > lo if width else (
            lo < offset < hi and not char["c"].isspace())
        if inside:
            x0, y0, x1, y1 = char["bbox"]
            box = (x0, y0, x1, y1) if box is None else (
                min(box[0], x0), min(box[1], y0), max(box[2], x1), max(box[3], y1)
            )
            first = idx if first is None else first
            last = idx
        offset += width
    if box is None:
        return None
    return box, "".join(c["c"] for c in chars[first:last + 1])


# =============================================================================
# Coordinate Conversion
# =============================================================================

def _pdf_to_page_transform(page) -> tuple[float, float, float, float, float, float]:
    """
    Native PDF user space -> PyMuPDF page space, as (a, b, c, d, e, f).

    PyMuPDF reports text in the unrotated page with the CropBox's top-left
    corner as origin. On an unrotated page that mapping is
    page.transformation_matrix. On a page with /Rotate, PyMuPDF builds that
    matrix as (1, 0, 0, -1, 0, cropbox height) and so drops a page box origin
    that is not (0, 0): rects would land off by the CropBox / MediaBox offset.
    The mapping below is the same on every page, rotated or not.
    """
    if not getattr(page, "rotation", 0) % 360:
        return tuple(page.transformation_matrix)
    cropbox, mediabox = page.cropbox, page.mediabox
    return (1.0, 0.0, 0.0, -1.0, -cropbox.x0, mediabox.y1 - cropbox.y0)


def _page_to_pdf_transform(page) -> tuple[float, float, float, float, float, float]:
    """
    Inverse of _pdf_to_page_transform as (a, b, c, d, e, f).

    Maps PyMuPDF page space (top-left origin, normalized to the CropBox so
    page.rect is (0, 0, w, h)) back to native PDF user space (MediaBox
    lower-left origin), which is where Zotero positions annotations. Besides
    flipping the y-axis, this restores any non-zero page box origin.
    """
    a, b, c, d, e, f = _pdf_to_page_transform(page)
    det = a * d - b * c
    return (
        d / det,
        -b / det,
        -c / det,
        a / det,
        (c * f - d * e) / det,
        (b * e - a * f) / det,
    )


def _convert_rects_to_zotero(
    bboxes: list[tuple[float, float, float, float]],
    page,
) -> list[list[float]]:
    """
    Convert PyMuPDF bounding boxes to Zotero's coordinate system.

    PyMuPDF uses top-left origin (y increases downward) relative to the
    CropBox. Zotero uses native PDF user space: bottom-left origin
    (y increases upward) relative to the MediaBox.

    Args:
        bboxes: List of (x0, y0, x1, y1) tuples from PyMuPDF
        page: The fitz page the bboxes belong to

    Returns:
        List of [x0, y1, x1, y2] rects in Zotero coordinates
    """
    a, b, c, d, e, f = _page_to_pdf_transform(page)

    rects = []

    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        corner_xs = (a * x0 + c * y0 + e, a * x1 + c * y1 + e)
        corner_ys = (b * x0 + d * y0 + f, b * x1 + d * y1 + f)

        rect = [
            min(corner_xs),
            min(corner_ys),
            max(corner_xs),
            max(corner_ys),
        ]
        rects.append(rect)

    return rects


def _rect_distance(a: list[float], b: list[float]) -> float:
    """
    Shortest distance between two rects.

    Args:
        a: [x1, y1, x2, y2] rect in Zotero coordinates
        b: [x1, y1, x2, y2] rect in Zotero coordinates

    Returns:
        Shortest distance between the rects, 0 when they overlap
    """
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return math.hypot(dx, dy)


def _is_counted_char(c: str) -> bool:
    """
    Check whether Zotero counts a character when numbering a page's characters.

    Zotero's PDF text extraction leaves out spaces and control characters.

    Args:
        c: A single character

    Returns:
        False for a space or a control character, True otherwise
    """
    code = ord(c)
    return c != " " and not (code <= 0x1F or 0x7F <= code <= 0x9F)


def _extract_page_char_rects(page) -> list[list[float]]:
    """
    Extract the rects of the characters Zotero counts on a page, in content order.

    Leaves out spaces, control characters, characters with a zero font size and
    characters outside the page, like Zotero's PDF text extraction. A character's
    position in this list is its character offset in the sort index.

    Args:
        page: The fitz page

    Returns:
        List of [x1, y1, x2, y2] rects in Zotero coordinates, or an empty list
        if the page's text cannot be extracted
    """
    try:
        # The default flags (199) clip text to the page. Leave out
        # TEXT_PRESERVE_IMAGES (4): image data has no characters and is slow.
        raw = page.get_text("rawdict", flags=195)
    except Exception:
        return []

    bboxes = [
        char["bbox"]
        for block in raw.get("blocks", [])
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span.get("size")
        for char in span.get("chars", [])
        if _is_counted_char(char["c"])
    ]
    return _convert_rects_to_zotero(bboxes, page)


#: Pages a PageCache keeps. A page's characters with their boxes run to a
#: couple of megabytes, and a batch works through a document page by page.
_PAGE_CACHE_SIZE = 12


class PageCache:
    """Extracted text of the pages of ONE document, kept across lookups.

    Placing a highlight reads its page twice (the text index to find the
    quote, the character boxes for the sort index) and its neighbours on a
    miss. A batch of annotations on the same pages repeated all of that per
    annotation. Pass one cache to every call made on the same open document.
    """

    def __init__(self) -> None:
        self._text: dict[int, tuple] = {}
        self._chars: dict[int, list] = {}

    @staticmethod
    def _put(store: dict, key: int, value):
        while len(store) >= _PAGE_CACHE_SIZE:
            del store[next(iter(store))]
        store[key] = value
        return value

    def text_index(self, page, page_index: int) -> tuple:
        """(spans, cumulative, span_positions) for a page."""
        found = self._text.get(page_index)
        if found is None:
            spans = _extract_page_spans(page)
            cumulative, span_positions = _build_normalized_text_index(spans)
            found = self._put(self._text, page_index, (spans, cumulative, span_positions))
        return found

    def char_rects(self, page, page_index: int) -> list:
        found = self._chars.get(page_index)
        if found is None:
            found = self._put(self._chars, page_index, _extract_page_char_rects(page))
        return found


def _build_sort_index(page, page_index: int, rects: list[list[float]],
                      cache: PageCache | None = None) -> str:
    """
    Build Zotero annotation sort index string.

    Format: PPPPP|OOOOOO|TTTTT (page|character offset|top)

    The offset is the position of the page character closest to the
    annotation's topmost rect. The top is the page height minus the top edge
    of that rect. Zotero orders a page's annotations by these two fields.

    The offset comes from PyMuPDF's text extraction, while Zotero uses pdf.js.
    It can differ from Zotero's own offset by a few characters.

    Args:
        page: The fitz page the rects belong to
        page_index: 0-indexed page number
        rects: Annotation rects in Zotero coordinates
        cache: The document's PageCache, when the caller keeps one

    Returns:
        Sort index string
    """
    offset = 0
    top = 0.0
    if rects:
        rect = max(rects, key=lambda r: r[3])
        chars = cache.char_rects(page, page_index) if cache else _extract_page_char_rects(page)
        if chars:
            offset = min(range(len(chars)), key=lambda i: _rect_distance(chars[i], rect))
        # page.rect is the page as displayed, so a quarter turn swaps its sides.
        page_height = page.rect.width if getattr(page, "rotation", 0) % 180 else page.rect.height
        top = max(page_height - rect[3], 0.0)
    return f"{page_index:05d}|{offset:06d}|{math.floor(top):05d}"


def _build_search_result(
    page_index: int,
    bboxes: list,
    texts: list[str],
    page,
    cache: PageCache | None = None,
) -> dict:
    """
    Build a successful search result dict.

    Args:
        page_index: 0-indexed page number
        bboxes: List of bounding boxes
        texts: List of matched text strings
        page: The fitz page for coordinate conversion
        cache: The document's PageCache, when the caller keeps one

    Returns:
        Dict with pageIndex, rects, sort_index, matched_text
    """
    rects = _convert_rects_to_zotero(bboxes, page)
    sort_index = _build_sort_index(page, page_index, rects, cache)

    return {
        "pageIndex": page_index,
        "rects": rects,
        "sort_index": sort_index,
        "matched_text": " ".join(texts),
    }


# =============================================================================
# Search Strategies
# =============================================================================

def _sliding_window_match(text: str, pattern: str) -> tuple[int, int, float] | None:
    """
    Best fuzzy match for pattern in text, using a sliding window.

    Always returns the best window found; callers apply their own threshold,
    because a below-threshold best match still feeds "did you mean" feedback.

    A window is fully scored only when ``quick_ratio()``, an upper bound on
    ``ratio()``, could beat the best so far. Window starts are sampled an
    eighth of the pattern apart and the neighbourhood of the best one is then
    scanned at every offset: a window 1.2x the pattern still holds nearly all
    of a match that begins between two samples, and scoring every offset of a
    page cost seconds per missed quote.

    The window is wider than the pattern, so the span returned is where the
    pattern's first and last matched runs fall inside it, not the window's
    start.

    Args:
        text: Normalized text to search in
        pattern: Normalized pattern to find

    Returns:
        (start, end, score), or None when the pattern cannot fit in the text
    """
    pattern_len = len(pattern)
    if pattern_len == 0 or len(text) < pattern_len:
        return None

    window_size = int(pattern_len * 1.2)
    matcher = SequenceMatcher(None, pattern)
    best = [0.0, 0]  # ratio, window start

    def scan(starts) -> None:
        for i in starts:
            matcher.set_seq2(text[i:i + window_size])
            if matcher.quick_ratio() <= best[0]:
                continue
            ratio = matcher.ratio()
            if ratio > best[0]:
                best[:] = [ratio, i]

    last = len(text) - pattern_len + 1
    step = max(1, pattern_len // 8)
    if len(text) >= SLIDING_WINDOW_STEP_THRESHOLD:
        step = max(step, len(text) // 5000)
    scan(range(0, last, step))
    if step > 1 and best[0] > 0:
        scan(range(max(0, best[1] - step), min(last, best[1] + step)))

    ratio, i = best
    matcher.set_seq2(text[i:i + window_size])
    # Runs of three or more: a lone matching letter says nothing about where
    # the pattern sits.
    runs = [b for b in matcher.get_matching_blocks() if b.size >= 3]
    if not runs:
        return i, min(i + pattern_len, len(text)), ratio
    first, final = runs[0], runs[-1]
    # A few unmatched characters before the first run or after the last are a
    # typo at the quote's edge, and the span is widened to take them in. More
    # than that is wording the page does not have, and nothing is added for it.
    slack = max(3, pattern_len // 10)
    unmatched_head = first.a if first.a <= slack else 0
    unmatched_tail = pattern_len - (final.a + final.size)
    if unmatched_tail > slack:
        unmatched_tail = 0
    start = i + max(0, first.b - unmatched_head)
    end = i + min(window_size, final.b + final.size + unmatched_tail)
    return start, min(end, len(text)), ratio


def _get_dynamic_threshold(text_length: int) -> float:
    """
    Get fuzzy matching threshold based on text length.

    Longer passages need lower thresholds because there's more opportunity
    for small variations to accumulate.
    """
    if text_length < 50:
        return FUZZY_THRESHOLD_SHORT
    elif text_length < 150:
        return FUZZY_THRESHOLD_MEDIUM
    else:
        return FUZZY_THRESHOLD_LONG


def _extract_anchor(text: str, from_start: bool) -> str:
    """
    Extract an anchor phrase from the start or end of text.

    Tries to break at word boundaries for better matching.

    Args:
        text: Full text to extract from
        from_start: If True, extract from start; if False, from end

    Returns:
        Anchor string, or empty string if text is too short
    """
    text = text.strip()

    if len(text) < ANCHOR_TARGET_LENGTH * 2:
        return ""

    if from_start:
        anchor = text[:ANCHOR_TARGET_LENGTH]
        # Extend to word boundary
        next_space = text.find(" ", ANCHOR_TARGET_LENGTH)
        if 0 < next_space < ANCHOR_TARGET_LENGTH + ANCHOR_WORD_BOUNDARY_TOLERANCE:
            anchor = text[:next_space]
    else:
        anchor = text[-ANCHOR_TARGET_LENGTH:]
        # Find word boundary
        remaining = text[:-ANCHOR_TARGET_LENGTH]
        last_space = remaining.rfind(" ")
        if last_space != -1 and len(remaining) - last_space < ANCHOR_WORD_BOUNDARY_TOLERANCE:
            anchor = text[last_space + 1:]

    return anchor.strip()


def _leading_overlap(needle: str, text: str) -> int:
    """
    How much of ``needle``'s start ``text`` also starts with, typos allowed.

    Used for a quote that runs off the end of a page: ``text`` is the page
    from where the quote begins, and whatever follows the quote there (a
    footnote, the page number) is not part of the overlap. Matching runs
    count while they stay in step with the needle and no more than a few
    characters apart.
    """
    matcher = SequenceMatcher(None, needle, text[:len(needle)], autojunk=False)
    covered = 0
    for a, b, size in matcher.get_matching_blocks():
        if size < 4:
            continue
        if abs(a - b) > 3 or a - covered > 8:
            break
        covered = a + size
    return covered


def _anchor_based_search(page, page_index: int, search_text: str, text_index,
                         cache: PageCache | None = None, following=None) -> dict | None:
    """
    Search for long text using anchor-based matching.

    Instead of matching the entire passage, finds the START (~40 chars) and
    END (~40 chars) of the passage, then highlights everything between them.
    This is robust against variations in the middle of the text.

    Both ends have to be found. When the end is not on this page, the quote
    may continue on the next one: if the rest of it is there, the result
    carries ``nextPageRects`` for that part, the way Zotero stores a highlight
    across a page break. Otherwise there is no answer here and the caller's
    other strategies judge the whole quote. Guessing the end from the quote's
    length, as this used to, highlighted the right number of characters after
    any passage that merely began like the quote.

    Args:
        page: PyMuPDF page object
        page_index: 0-indexed page number
        search_text: Full text to highlight
        text_index: (spans, cumulative, span_positions) for the page
        cache: The document's PageCache, when the caller keeps one
        following: Callable returning ``(page, text_index)`` for the next
            page, or None when there is none

    Returns:
        Search result dict if found, None otherwise
    """
    start_anchor = _extract_anchor(search_text, from_start=True)
    end_anchor = _extract_anchor(search_text, from_start=False)
    if not start_anchor or not end_anchor:
        return None

    spans, cumulative, span_positions = text_index
    if not spans or not cumulative:
        return None

    # Find start anchor
    normalized_start = normalize_for_matching(start_anchor)
    start_pos = cumulative.find(normalized_start)
    if start_pos == -1:
        match = _sliding_window_match(cumulative, normalized_start)
        if match and match[2] >= ANCHOR_MATCH_THRESHOLD:
            start_pos = match[0]
        else:
            return None

    # Find end anchor (search after start)
    normalized_end = normalize_for_matching(end_anchor)
    search_offset = start_pos + len(normalized_start) // 2
    end_pos = cumulative.find(normalized_end, search_offset)
    if end_pos != -1:
        end_pos += len(normalized_end)
    else:
        match = _sliding_window_match(cumulative[search_offset:], normalized_end)
        if not match or match[2] < ANCHOR_MATCH_THRESHOLD:
            return _across_page_break(page, page_index, search_text, text_index, start_pos,
                                      normalized_end, cache, following)
        end_pos = search_offset + match[1]

    bboxes, texts = _get_spans_in_range(start_pos, end_pos, span_positions, spans)
    if not bboxes:
        return None
    return _build_search_result(page_index, bboxes, texts, page, cache)


def _across_page_break(page, page_index: int, search_text: str, text_index, start_pos: int,
                       normalized_end: str, cache: PageCache | None, following) -> dict | None:
    """
    A long quote that starts at ``start_pos`` on this page and ends on the next.

    The part on this page is what the page holds of the quote's start; the
    rest must then be on the next page, ending at the quote's end anchor and
    reading like the rest of the quote. None when either does not hold.
    """
    upcoming = following() if following else None
    if upcoming is None:
        return None
    next_page, (next_spans, next_cumulative, next_positions) = upcoming
    spans, cumulative, span_positions = text_index

    needle = normalize_for_matching(search_text)
    here = _leading_overlap(needle, cumulative[start_pos:])
    rest = needle[here:]
    if here < ANCHOR_TARGET_LENGTH // 2 or len(rest) < 6 or not next_cumulative:
        return None

    # The end anchor, or all of the rest when less than an anchor is left.
    key = normalized_end if len(rest) >= len(normalized_end) else rest
    end_pos = next_cumulative.find(key)
    if end_pos != -1:
        end_pos += len(key)
    elif key is rest:
        return None
    else:
        match = _sliding_window_match(next_cumulative, key)
        if not match or match[2] < ANCHOR_MATCH_THRESHOLD:
            return None
        end_pos = match[1]
    begin = max(0, end_pos - len(rest))
    if SequenceMatcher(None, rest, next_cumulative[begin:end_pos], autojunk=False).ratio() < ACROSS_PAGE_MIN_RATIO:
        return None

    bboxes, texts = _get_spans_in_range(start_pos, start_pos + here, span_positions, spans)
    next_bboxes, next_texts = _get_spans_in_range(begin, end_pos, next_positions, next_spans)
    if not bboxes or not next_bboxes:
        return None
    result = _build_search_result(page_index, bboxes, texts + next_texts, page, cache)
    result["nextPageRects"] = _convert_rects_to_zotero(next_bboxes, next_page)
    return result


def _exact_index_match(search_text: str, text_index) -> tuple[list, list[str]] | None:
    """Boxes and texts of ``search_text`` where the page's text index holds it exactly."""
    spans, cumulative, span_positions = text_index
    needle = normalize_for_matching(search_text)
    if not spans or not needle:
        return None
    start = cumulative.find(needle)
    if start == -1:
        return None
    bboxes, texts = _get_spans_in_range(start, start + len(needle), span_positions, spans)
    return (bboxes, texts) if bboxes else None


def _fuzzy_search_page(search_text: str, text_index) -> dict | None:
    """
    The closest loose match for a text on a PDF page.

    For text the exact strategies did not find: a typo, another spelling,
    characters the PDF stores differently.

    Args:
        search_text: Text to search for
        text_index: (spans, cumulative, span_positions) for the page

    Returns:
        Dict with 'rects', 'matched_text', 'score' if found, None otherwise.
        'rects' is empty when the best match is below the length-dependent
        threshold, so the match can still be offered as a suggestion.
    """
    spans, cumulative, span_positions = text_index
    normalized_search = normalize_for_matching(search_text)
    if not spans or not normalized_search or not cumulative:
        return None

    match = _sliding_window_match(cumulative, normalized_search)
    if match is None:
        return None
    match_start, match_end, match_score = match
    bboxes, texts = _get_spans_in_range(match_start, match_end, span_positions, spans)
    if not bboxes:
        return None
    return {
        "rects": bboxes if match_score >= _get_dynamic_threshold(len(search_text)) else [],
        "bboxes": bboxes,
        "matched_text": " ".join(texts),
        "score": match_score,
    }


def _exact_search_page(page, page_index: int, search_text: str, cache: PageCache,
                       following=None) -> dict | None:
    """
    Find text on one page where it is written as given.

    A long passage is found by both of its ends (see _anchor_based_search).
    Anything else, and a long passage that did not answer that way, is looked
    up in the page's text index, which ignores case and spacing and reads
    across line-break hyphens, ligatures, and dash and quote variants. Of
    several occurrences on the page the first is taken.

    PyMuPDF's own ``page.search_for`` is not used. It answered before the
    index did, and on PyMuPDF 2.x it joins words hyphenated across lines and
    then reports every hit after such a word a few characters to the left of
    where it is, so most pages of a typeset paper gave highlights that began
    and ended inside neighbouring words.
    """
    text_index = cache.text_index(page, page_index)

    if len(search_text) > ANCHOR_MIN_TEXT_LENGTH:
        result = _anchor_based_search(page, page_index, search_text, text_index, cache, following)
        if result:
            return result

    found = _exact_index_match(search_text, text_index)
    if found:
        return _build_search_result(page_index, found[0], found[1], page, cache)
    return None


def _fuzzy_search_single_page(page, page_index: int, search_text: str, best_debug: dict,
                              cache: PageCache) -> dict | None:
    """
    The loose match for a text on one page, when it is close enough.

    Every page's best match is noted in ``best_debug`` (mutated), close
    enough or not, for the "nearest matches" a miss reports.
    """
    fuzzy_result = _fuzzy_search_page(search_text, cache.text_index(page, page_index))
    if not fuzzy_result:
        return None
    score = fuzzy_result["score"]
    if score > best_debug["score"]:
        best_debug["match"] = fuzzy_result["matched_text"]
        best_debug["score"] = score
        best_debug["page"] = page_index
    best_debug.setdefault("candidates", []).append({
        "page_index": page_index, "score": score,
        "matched_text": fuzzy_result["matched_text"], "bboxes": fuzzy_result["bboxes"],
    })
    if not fuzzy_result["rects"]:
        return None
    return _build_search_result(page_index, fuzzy_result["rects"], [fuzzy_result["matched_text"]], page, cache)


# =============================================================================
# Public API
# =============================================================================

@contextmanager
def open_pdf(pdf):
    """
    An open PyMuPDF document for a path, or an already-open one passed through.

    Lets a caller open a PDF once and hand the document to several helpers
    here, instead of each reopening and re-parsing the file. Only a document
    this function opened is closed on exit.
    """
    if not isinstance(pdf, (str, os.PathLike)):
        yield pdf
        return
    try:
        from zotero_mcp._pymupdf import fitz
    except ImportError as exc:
        raise ImportError(f"PDF features require PyMuPDF. {install_hint('pdf')}") from exc
    document = fitz.open(pdf)
    try:
        yield document
    finally:
        document.close()


def page_range_error(document, page_num: int) -> str | None:
    """The error for a 1-indexed page outside the document, or None."""
    total = len(document)
    if 1 <= page_num <= total:
        return None
    return f"Page {page_num} out of range (PDF has {total} pages)"


def page_label(page, page_num: int) -> str:
    """A page's printed label ("iv", "12"), falling back to its 1-indexed number."""
    try:
        label = page.get_label()
        if label:
            return label
    except Exception:
        pass
    return str(page_num)


def find_text_position(pdf, page_num: int, search_text: str, cache: PageCache | None = None) -> dict:
    """
    Search for text in a PDF and return position data for Zotero annotation.

    Searches the specified page first, then DEFAULT_NEIGHBOR_PAGES pages on
    either side. Every one of those pages is tried for the text as written
    before any is tried for a loose match, so a near miss on the page given
    never wins over the text itself one page on.

    Args:
        pdf: Path to the PDF file, or an open PyMuPDF document
        page_num: 1-indexed page number to search on
        search_text: Text to find
        cache: A PageCache for this document, to share between calls

    Returns:
        On success:
            {
                "pageIndex": int,  # 0-indexed page where found
                "rects": [[x1, y1, x2, y2], ...],  # Bounding boxes
                "nextPageRects": [...],  # only for a quote that runs on to the next page
                "sort_index": str,  # For annotation ordering
                "matched_text": str,  # Actual matched text
            }

        On failure:
            {
                "error": str,
                "best_match": str | None,  # Best partial match found
                "best_score": float,  # Similarity score
                "page_found": int | None,  # Page with best match
                "pages_searched": [int, ...],  # Pages that were searched
            }
    """
    cache = cache if cache is not None else PageCache()
    with open_pdf(pdf) as doc:
        error = page_range_error(doc, page_num)
        if error:
            return {"error": error, "best_match": None, "best_score": 0, "pages_searched": []}

        total_pages = len(doc)
        target_index = page_num - 1
        pages_to_search = [target_index]
        for offset in range(1, DEFAULT_NEIGHBOR_PAGES + 1):
            if target_index - offset >= 0:
                pages_to_search.append(target_index - offset)
            if target_index + offset < total_pages:
                pages_to_search.append(target_index + offset)

        for page_index in pages_to_search:
            def following(index=page_index + 1):
                if index >= total_pages:
                    return None
                return doc[index], cache.text_index(doc[index], index)

            result = _exact_search_page(doc[page_index], page_index, search_text, cache, following)
            if result:
                return result

        best_debug = {"match": None, "score": 0.0, "page": None, "candidates": []}
        for page_index in pages_to_search:
            result = _fuzzy_search_single_page(doc[page_index], page_index, search_text, best_debug, cache)
            if result:
                return result

        return {
            "error": f"Could not find text on page {page_num} or neighboring pages",
            "best_match": best_debug["match"],
            "best_score": best_debug["score"],
            "page_found": best_debug["page"] + 1 if best_debug["page"] is not None else None,
            "pages_searched": [p + 1 for p in pages_to_search],
            "candidates": _nearest_candidates(doc, best_debug["candidates"]),
        }


#: A miss lists at most this many near matches, none scoring below the floor.
MAX_CANDIDATES = 3
CANDIDATE_MIN_SCORE = 0.4


def _nearest_candidates(doc, found: list[dict]) -> list[dict]:
    """The closest fuzzy matches of a failed search, best first, ready to show.

    Each is ``{"page", "best_match", "score", "rect", "rects"}``; ``rect`` and
    ``rects`` are ``[x, y, w, h]`` boxes normalized to the page as displayed,
    the form ``--rect`` and ``annotations preview`` use, so the candidate can
    be looked at before the text is corrected.
    """
    ranked = sorted((c for c in found if c["score"] >= CANDIDATE_MIN_SCORE),
                    key=lambda c: -c["score"])[:MAX_CANDIDATES]
    candidates = []
    for candidate in ranked:
        page = doc[candidate["page_index"]]
        rects = zotero_rects_to_shown(page, _convert_rects_to_zotero(candidate["bboxes"], page))
        if not rects:
            continue
        candidates.append({
            "page": candidate["page_index"] + 1,
            "best_match": normalize_text(candidate["matched_text"])[:200],
            "score": round(candidate["score"], 3),
            "rect": union_box(rects),
            "rects": rects,
        })
    return candidates


def get_page_label(pdf, page_num: int) -> str:
    """
    Get the page label for a given page number.

    Some PDFs have custom page labels (e.g., "i", "ii", "1", "2").

    Args:
        pdf: Path to the PDF file, or an open PyMuPDF document
        page_num: 1-indexed page number

    Returns:
        Page label if available, otherwise the page number as string
    """
    try:
        with open_pdf(pdf) as doc:
            if page_range_error(doc, page_num):
                return str(page_num)
            return page_label(doc[page_num - 1], page_num)
    except ImportError:
        return str(page_num)


#: Share of a rect's height left out at the top and at the bottom when its
#: text is read back (see text_in_rects).
_TEXT_BAND_INSET = 0.3


def text_in_rects(pdf, page_index: int, rects: list[list[float]]) -> str:
    """
    Readable text inside annotation rects, as a person would see it highlighted.

    The matcher works on text extracted without inter-word spaces, so its
    ``matched_text`` runs words together ("theTransformeristhe..."). This reads
    the page again under each final rect, with normal spacing and ligatures
    expanded, which is what a preview should show.

    Only the middle of each rect's height is read. The boxes PyMuPDF reports
    for neighbouring lines overlap in many fonts, and a rect read at its full
    height then returns pieces of the lines above and below as well.

    Args:
        pdf: Path to the PDF file, or an open PyMuPDF document
        page_index: 0-indexed page number
        rects: Rects in Zotero (PDF user space) coordinates

    Returns:
        The text of all rects joined by spaces
    """
    from zotero_mcp._pymupdf import fitz
    with open_pdf(pdf) as doc:
        page = doc[page_index]
        to_page = fitz.Matrix(*_pdf_to_page_transform(page))
        parts = []
        for x0, y0, x1, y1 in rects:
            box = fitz.Rect(x0, y0, x1, y1) * to_page
            box.normalize()
            inset = box.height * _TEXT_BAND_INSET
            box.y0 += inset
            box.y1 -= inset
            parts.append(page.get_textbox(box))
        return normalize_text(" ".join(parts))


def build_annotation_position(page_index: int, rects: list[list[float]],
                              next_page_rects: list[list[float]] | None = None) -> str:
    """
    Build the annotationPosition JSON string for Zotero.

    Args:
        page_index: 0-indexed page number
        rects: List of [x1, y1, x2, y2] bounding boxes
        next_page_rects: Boxes on the following page, for a highlight that
            runs across a page break (Zotero's ``nextPageRects``)

    Returns:
        JSON string for Zotero's annotationPosition field
    """
    position = {"pageIndex": page_index, "rects": rects}
    if next_page_rects:
        position["nextPageRects"] = next_page_rects
    return json.dumps(position)


def build_area_position_data(
    pdf,
    page_num: int,
    x: float,
    y: float,
    width: float,
    height: float,
    cache: PageCache | None = None,
) -> dict:
    """
    Build Zotero position data for an area/image annotation on a PDF page.

    Args:
        pdf: Path to the PDF file, or an open PyMuPDF document
        page_num: 1-indexed page number
        x: Normalized left coordinate (0..1)
        y: Normalized top coordinate (0..1)
        width: Normalized width (0..1)
        height: Normalized height (0..1)

    Returns:
        On success: {"pageIndex": int, "rects": [[x1, y1, x2, y2]], "sort_index": str}
        On failure: {"error": str}
    """
    with open_pdf(pdf) as doc:
        error = page_range_error(doc, page_num)
        if error:
            return {"error": error}

        target_index = page_num - 1
        page = doc[target_index]
        rects = _shown_rect_to_zotero(page, x, y, x + width, y + height)

        return {
            "pageIndex": target_index,
            "rects": rects,
            "sort_index": _build_sort_index(page, target_index, rects, cache),
        }


#: Side of a sticky note in PDF points; Zotero's reader uses the same size.
NOTE_SIZE = 22


def build_note_position_data(pdf, page_num: int, x: float, y: float,
                             cache: PageCache | None = None) -> dict:
    """
    Build Zotero position data for a sticky note centered on a page point.

    Args:
        pdf: Path to the PDF file, or an open PyMuPDF document
        page_num: 1-indexed page number
        x, y: Normalized point (0..1) on the page as displayed

    Returns:
        On success: {"pageIndex": int, "rects": [[x1, y1, x2, y2]], "sort_index": str}
        On failure: {"error": str}
    """
    with open_pdf(pdf) as doc:
        error = page_range_error(doc, page_num)
        if error:
            return {"error": error}

        target_index = page_num - 1
        page = doc[target_index]
        ((px, py, _, _),) = _shown_rect_to_zotero(page, x, y, x, y)
        half = NOTE_SIZE / 2
        rect = [round(v, 4) for v in (px - half, py - half, px + half, py + half)]

        return {
            "pageIndex": target_index,
            "rects": [rect],
            "sort_index": _build_sort_index(page, target_index, [rect], cache),
        }


def _shown_rect_to_zotero(page, x0: float, y0: float, x1: float, y1: float):
    """Normalized corners on the page as displayed -> _convert_rects_to_zotero output.

    The page as displayed is page.rect, which reflects /Rotate, but
    _convert_rects_to_zotero expects unrotated page space, the frame PyMuPDF
    text coordinates use.
    """
    x0, x1 = x0 * page.rect.width, x1 * page.rect.width
    y0, y1 = y0 * page.rect.height, y1 * page.rect.height
    if getattr(page, "rotation", 0):
        a, b, c, d, e, f = page.derotation_matrix
        xs = (a * x0 + c * y0 + e, a * x1 + c * y1 + e)
        ys = (b * x0 + d * y0 + f, b * x1 + d * y1 + f)
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    bbox = [tuple(round(v, 4) for v in (x0, y0, x1, y1))]
    return _convert_rects_to_zotero(bbox, page)


def zotero_rects_to_shown(page, rects: list[list[float]]) -> list[list[float]]:
    """Stored Zotero rects -> normalized ``[x, y, w, h]`` boxes on the page as displayed.

    The inverse of ``_shown_rect_to_zotero``: native PDF user space (bottom-left
    origin, MediaBox) back to the top-left origin frame ``create`` and ``batch``
    take, undoing any /Rotate on the way. A box listed this way can be passed
    straight back to ``--rect``.
    """
    to_page = _pdf_to_page_transform(page)
    rotate = tuple(page.rotation_matrix) if getattr(page, "rotation", 0) % 360 else None
    width, height = page.rect.width, page.rect.height
    boxes = []
    for x0, y0, x1, y1 in rects:
        corners = []
        for x, y in ((x0, y0), (x1, y1)):
            a, b, c, d, e, f = to_page
            x, y = a * x + c * y + e, b * x + d * y + f
            if rotate:
                a, b, c, d, e, f = rotate
                x, y = a * x + c * y + e, b * x + d * y + f
            corners.append((x, y))
        left, right = sorted(p[0] for p in corners)
        top, bottom = sorted(p[1] for p in corners)
        boxes.append([round(v, 4) for v in (
            left / width, top / height, (right - left) / width, (bottom - top) / height)])
    return boxes


def union_box(boxes: list[list[float]]) -> list[float]:
    """The ``[x, y, w, h]`` box that covers all of the given ``[x, y, w, h]`` boxes."""
    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    right = max(b[0] + b[2] for b in boxes)
    bottom = max(b[1] + b[3] for b in boxes)
    return [round(v, 4) for v in (left, top, right - left, bottom - top)]


_DOI_IN_TEXT_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+")


def doi_from_pdf(pdf) -> tuple[str | None, str]:
    """The DOI of the article a PDF holds, and where it was found (or why not).

    Looks at the document's metadata (including XMP) and at the whole first
    page, not just its first lines: a PDF cut from a journal spread carries the
    neighbouring article's DOI too, so the answer is only given when it is
    unambiguous. Metadata naming exactly one DOI wins unless page 1 names
    others and not that one. Otherwise page 1 must name exactly one DOI.
    Anything else returns ``(None, "ambiguous: ...")`` rather than a guess.
    """
    from zotero_mcp.identifiers import doi_match_key, normalize_doi

    def dois_in(text: str) -> dict[str, str]:
        # A DOI wrapped at one of its own hyphens is joined back up first.
        found: dict[str, str] = {}
        for match in _DOI_IN_TEXT_RE.finditer(re.sub(r"-[ \t]*\n\s*", "-", text or "")):
            doi = normalize_doi(match.group(0))
            if doi:
                found.setdefault(doi_match_key(doi), doi)
        return found

    with open_pdf(pdf) as doc:
        meta = " ".join(str(v) for v in (doc.metadata or {}).values() if v)
        try:
            meta += " " + (doc.get_xml_metadata() or "")
        except Exception:
            pass
        first_page = doc[0].get_text() if len(doc) else ""

    from_meta = dois_in(meta)
    on_page = dois_in(first_page)
    named = ", ".join(list(on_page.values())[:3])
    if len(from_meta) == 1:
        (key, doi), = from_meta.items()
        if not on_page or key in on_page:
            return doi, "metadata"
        return None, f"ambiguous: metadata names {doi}, page 1 names {named}"
    if len(on_page) == 1:
        return next(iter(on_page.values())), "page 1"
    if not on_page:
        return None, "no DOI in the metadata or on page 1"
    return None, f"ambiguous: {named}"
