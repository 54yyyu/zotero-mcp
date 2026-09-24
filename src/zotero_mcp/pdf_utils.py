"""
PDF utility functions for Zotero annotation creation.

This module provides text search capabilities for PDFs to extract position data
needed for creating Zotero highlight annotations. It handles common PDF text
extraction issues like:
- Hyphenation at line breaks
- Special characters (em-dashes, curly quotes, ligatures)
- Missing word spacing in extracted text
- Page number mismatches

Search Strategy (in order):
1. For long text (>100 chars): Anchor-based matching (find start/end, highlight between)
2. Exact match using PyMuPDF's search
3. Fuzzy matching with normalized text comparison
"""

from __future__ import annotations

import json
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
    - Normalize dashes (em-dash, en-dash, etc.) to simple hyphen
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
    Aggressively normalize text for fuzzy matching.

    This removes ALL spaces and lowercases the text to handle PDFs where
    words are stored without proper spacing between spans.

    Args:
        text: Text to normalize

    Returns:
        Text with all spaces removed, lowercased
    """
    text = normalize_text(text)
    text = re.sub(r'\s+', '', text)
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
    cumulative = ""
    positions = []

    for i, span in enumerate(spans):
        start = len(cumulative)
        normalized = normalize_for_matching(span["text"])
        cumulative += normalized
        end = len(cumulative)
        positions.append((start, end, i))

    return cumulative, positions


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
    ligatures, case) or deletes whitespace.

    Returns None when the span has no character data, so the caller keeps the
    whole-span box rather than dropping the piece.
    """
    chars = span.get("chars") or []
    offset = 0
    first = last = None
    box = None
    for idx, char in enumerate(chars):
        width = len(normalize_for_matching(char["c"]))
        if width and offset < hi and offset + width > lo:
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

def _page_to_pdf_transform(page) -> tuple[float, float, float, float, float, float]:
    """
    Inverse of page.transformation_matrix as (a, b, c, d, e, f).

    Maps PyMuPDF page space (top-left origin, normalized to the CropBox so
    page.rect is (0, 0, w, h)) back to native PDF user space (MediaBox
    lower-left origin), which is where Zotero positions annotations. Besides
    flipping the y-axis, this restores any non-zero page box origin and
    accounts for page rotation.
    """
    a, b, c, d, e, f = page.transformation_matrix
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
) -> tuple[list[list[float]], float, float]:
    """
    Convert PyMuPDF bounding boxes to Zotero's coordinate system.

    PyMuPDF uses top-left origin (y increases downward) relative to the
    CropBox. Zotero uses native PDF user space: bottom-left origin
    (y increases upward) relative to the MediaBox.

    Args:
        bboxes: List of (x0, y0, x1, y1) tuples from PyMuPDF
        page: The fitz page the bboxes belong to

    Returns:
        Tuple of:
        - List of [x0, y1, x1, y2] rects in Zotero coordinates
        - Minimum y position (for sort index)
        - Minimum x position (for sort index)
    """
    a, b, c, d, e, f = _page_to_pdf_transform(page)

    rects = []
    min_y = float("inf")
    min_x = float("inf")

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
        min_y = min(min_y, rect[1])
        min_x = min(min_x, rect[0])

    return rects, min_y, min_x


def _build_sort_index(page_index: int, min_y: float, min_x: float) -> str:
    """
    Build Zotero annotation sort index string.

    Format: PPPPP|YYYYYY|XXXXX (page|y-position|x-position)

    Args:
        page_index: 0-indexed page number
        min_y: Minimum y position
        min_x: Minimum x position

    Returns:
        Sort index string
    """
    return f"{page_index:05d}|{int(min_y):06d}|{int(min_x):05d}"


def _build_search_result(
    page_index: int,
    bboxes: list,
    texts: list[str],
    page,
) -> dict:
    """
    Build a successful search result dict.

    Args:
        page_index: 0-indexed page number
        bboxes: List of bounding boxes
        texts: List of matched text strings
        page: The fitz page for coordinate conversion

    Returns:
        Dict with pageIndex, rects, sort_index, matched_text
    """
    rects, min_y, min_x = _convert_rects_to_zotero(bboxes, page)
    sort_index = _build_sort_index(page_index, min_y, min_x)

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
    ``ratio()``, could beat the best so far. That skips most windows without
    changing which one wins.

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
    best = [0.0, 0, 0]  # ratio, start, end

    def scan(starts) -> None:
        for i in starts:
            matcher.set_seq2(text[i:i + window_size])
            if matcher.quick_ratio() <= best[0]:
                continue
            ratio = matcher.ratio()
            if ratio > best[0]:
                best[:] = [ratio, i, min(i + pattern_len, len(text))]

    # Use stepping for very long texts, then refine around the best step.
    step = max(1, len(text) // 5000) if len(text) >= SLIDING_WINDOW_STEP_THRESHOLD else 1
    scan(range(0, len(text) - pattern_len + 1, step))
    if step > 1 and best[0] > 0:
        scan(range(max(0, best[1] - step), min(len(text) - pattern_len + 1, best[1] + step)))

    return best[1], best[2], best[0]


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


def _anchor_based_search(page, page_index: int, search_text: str, text_index) -> dict | None:
    """
    Search for long text using anchor-based matching.

    Instead of matching the entire passage, finds the START (~40 chars) and
    END (~40 chars) of the passage, then highlights everything between them.
    This is robust against variations in the middle of the text.

    Args:
        page: PyMuPDF page object
        page_index: 0-indexed page number
        search_text: Full text to highlight
        text_index: (spans, cumulative, span_positions) for the page

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
    if end_pos == -1:
        match = _sliding_window_match(cumulative[search_offset:], normalized_end)
        if match and match[2] >= ANCHOR_MATCH_THRESHOLD:
            end_pos = search_offset + match[0] + len(normalized_end)
        else:
            # Estimate based on text length
            estimated_len = int(len(normalize_for_matching(search_text)) * 1.1)
            end_pos = min(start_pos + estimated_len, len(cumulative))
    else:
        end_pos = end_pos + len(normalized_end)

    bboxes, texts = _get_spans_in_range(start_pos, end_pos, span_positions, spans)
    if not bboxes:
        return None
    return _build_search_result(page_index, bboxes, texts, page)


def _fuzzy_search_page(search_text: str, text_index) -> dict | None:
    """
    Perform fuzzy text search on a PDF page.

    Handles cases where exact matching fails due to hyphenation,
    whitespace differences, or character variations.

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

    threshold = _get_dynamic_threshold(len(search_text))

    # Try exact match first
    match_start = cumulative.find(normalized_search)
    if match_start != -1:
        bboxes, texts = _get_spans_in_range(
            match_start, match_start + len(normalized_search), span_positions, spans
        )
        if bboxes:
            return {"rects": bboxes, "matched_text": " ".join(texts), "score": 1.0}

    match = _sliding_window_match(cumulative, normalized_search)
    if match is None:
        return None
    match_start, match_end, match_score = match
    bboxes, texts = _get_spans_in_range(match_start, match_end, span_positions, spans)
    if not bboxes:
        return None
    return {
        "rects": bboxes if match_score >= threshold else [],
        "matched_text": " ".join(texts),
        "score": match_score,
    }


def _search_single_page(page, page_index: int, search_text: str, best_debug: dict) -> dict | None:
    """
    Search for text on a single PDF page using multiple strategies.

    Strategy order:
    1. Anchor-based matching (for long text)
    2. Exact search via PyMuPDF
    3. Fuzzy matching

    The page's text index is built at most once, on first use, and shared by
    the anchor and fuzzy strategies.

    Args:
        page: PyMuPDF page object
        page_index: 0-indexed page number
        search_text: Text to search for
        best_debug: Dict to track best match for debug info (mutated)

    Returns:
        Search result dict if found, None otherwise
    """
    cached_index: list = []

    def text_index():
        if not cached_index:
            spans = _extract_page_spans(page)
            cumulative, span_positions = _build_normalized_text_index(spans)
            cached_index.append((spans, cumulative, span_positions))
        return cached_index[0]

    # Strategy 1: Anchor-based matching for long passages
    if len(search_text) > ANCHOR_MIN_TEXT_LENGTH:
        result = _anchor_based_search(page, page_index, search_text, text_index())
        if result:
            return result

    # Strategy 2: Exact search
    text_instances = page.search_for(search_text)
    if not text_instances:
        # Try with normalized whitespace
        text_instances = page.search_for(" ".join(search_text.split()))
    if text_instances:
        rects, min_y, min_x = _convert_rects_to_zotero(list(text_instances), page)
        return {
            "pageIndex": page_index,
            "rects": rects,
            "sort_index": _build_sort_index(page_index, min_y, min_x),
            "matched_text": search_text,
        }

    # Strategy 3: Fuzzy matching
    fuzzy_result = _fuzzy_search_page(search_text, text_index())
    if fuzzy_result:
        score = fuzzy_result.get("score", 0)
        if score > best_debug["score"]:
            best_debug["match"] = fuzzy_result.get("matched_text")
            best_debug["score"] = score
            best_debug["page"] = page_index
        if fuzzy_result.get("rects"):
            rects, min_y, min_x = _convert_rects_to_zotero(fuzzy_result["rects"], page)
            return {
                "pageIndex": page_index,
                "rects": rects,
                "sort_index": _build_sort_index(page_index, min_y, min_x),
                "matched_text": fuzzy_result["matched_text"],
            }

    return None


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
        import fitz
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


def find_text_position(pdf, page_num: int, search_text: str) -> dict:
    """
    Search for text in a PDF and return position data for Zotero annotation.

    Searches the specified page first, then DEFAULT_NEIGHBOR_PAGES pages on
    either side. Uses anchor-based, exact and fuzzy matching to handle
    PDF text extraction issues.

    Args:
        pdf: Path to the PDF file, or an open PyMuPDF document
        page_num: 1-indexed page number to search on
        search_text: Text to find

    Returns:
        On success:
            {
                "pageIndex": int,  # 0-indexed page where found
                "rects": [[x1, y1, x2, y2], ...],  # Bounding boxes
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

        best_debug = {"match": None, "score": 0.0, "page": None}
        for page_index in pages_to_search:
            result = _search_single_page(doc[page_index], page_index, search_text, best_debug)
            if result:
                return result

        return {
            "error": f"Could not find text on page {page_num} or neighboring pages",
            "best_match": best_debug["match"],
            "best_score": best_debug["score"],
            "page_found": best_debug["page"] + 1 if best_debug["page"] is not None else None,
            "pages_searched": [p + 1 for p in pages_to_search],
        }


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


def text_in_rects(pdf, page_index: int, rects: list[list[float]]) -> str:
    """
    Readable text inside annotation rects, as a person would see it highlighted.

    The matcher works on text extracted without inter-word spaces, so its
    ``matched_text`` runs words together ("theTransformeristhe..."). This reads
    the page again under each final rect, with normal spacing and ligatures
    expanded, which is what a preview should show.

    Args:
        pdf: Path to the PDF file, or an open PyMuPDF document
        page_index: 0-indexed page number
        rects: Rects in Zotero (PDF user space) coordinates

    Returns:
        The text of all rects joined by spaces
    """
    import fitz

    with open_pdf(pdf) as doc:
        page = doc[page_index]
        to_page = page.transformation_matrix
        parts = []
        for x0, y0, x1, y1 in rects:
            box = fitz.Rect(x0, y0, x1, y1) * to_page
            box.normalize()
            parts.append(page.get_textbox(box))
        return normalize_text(" ".join(parts))


def build_annotation_position(page_index: int, rects: list[list[float]]) -> str:
    """
    Build the annotationPosition JSON string for Zotero.

    Args:
        page_index: 0-indexed page number
        rects: List of [x1, y1, x2, y2] bounding boxes

    Returns:
        JSON string for Zotero's annotationPosition field
    """
    return json.dumps({
        "pageIndex": page_index,
        "rects": rects,
    })

def build_area_position_data(
    pdf,
    page_num: int,
    x: float,
    y: float,
    width: float,
    height: float,
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
        rects, min_y, min_x = _shown_rect_to_zotero(page, x, y, x + width, y + height)

        return {
            "pageIndex": target_index,
            "rects": rects,
            "sort_index": _build_sort_index(target_index, min_y, min_x),
        }


#: Side of a sticky note in PDF points; Zotero's reader uses the same size.
NOTE_SIZE = 22


def build_note_position_data(pdf, page_num: int, x: float, y: float) -> dict:
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
        ((px, py, _, _),), _, _ = _shown_rect_to_zotero(page, x, y, x, y)
        half = NOTE_SIZE / 2
        rect = [round(v, 4) for v in (px - half, py - half, px + half, py + half)]

        return {
            "pageIndex": target_index,
            "rects": [rect],
            "sort_index": _build_sort_index(target_index, max(rect[1], 0), max(rect[0], 0)),
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
