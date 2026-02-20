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
import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

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
    text = re.sub(r'-\s*\n\s*', '', text)
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

    Returns:
        List of dicts with 'text' and 'bbox' keys
    """
    blocks = page.get_text("dict", flags=11)["blocks"]
    spans = []

    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                spans.append({
                    "text": span["text"],
                    "bbox": span["bbox"],
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

    Returns:
        Tuple of (list of bboxes, list of original text strings)
    """
    bboxes = []
    texts = []

    for norm_start, norm_end, span_idx in span_positions:
        if norm_start < end_pos and norm_end > start_pos:
            bboxes.append(spans[span_idx]["bbox"])
            texts.append(spans[span_idx]["text"])

    return bboxes, texts


# =============================================================================
# Coordinate Conversion
# =============================================================================

def _convert_rects_to_zotero(
    bboxes: list[tuple[float, float, float, float]],
    page_height: float,
) -> tuple[list[list[float]], float, float]:
    """
    Convert PyMuPDF bounding boxes to Zotero's coordinate system.

    PyMuPDF uses top-left origin (y increases downward).
    Zotero/PDF uses bottom-left origin (y increases upward).

    Args:
        bboxes: List of (x0, y0, x1, y1) tuples from PyMuPDF
        page_height: Height of the page

    Returns:
        Tuple of:
        - List of [x0, y1, x1, y2] rects in Zotero coordinates
        - Minimum y position (for sort index)
        - Minimum x position (for sort index)
    """
    rects = []
    min_y = float("inf")
    min_x = float("inf")

    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        # Transform Y coordinates
        pdf_y1 = page_height - y1  # Bottom in PDF coords
        pdf_y2 = page_height - y0  # Top in PDF coords

        rects.append([x0, pdf_y1, x1, pdf_y2])
        min_y = min(min_y, pdf_y1)
        min_x = min(min_x, x0)

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
    page_height: float,
) -> dict:
    """
    Build a successful search result dict.

    Args:
        page_index: 0-indexed page number
        bboxes: List of bounding boxes
        texts: List of matched text strings
        page_height: Page height for coordinate conversion

    Returns:
        Dict with pageIndex, rects, sort_index, matched_text
    """
    rects, min_y, min_x = _convert_rects_to_zotero(bboxes, page_height)
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

def _sliding_window_match(
    text: str,
    pattern: str,
    threshold: float,
    return_best: bool = False,
) -> tuple[int, int, float] | None:
    """
    Find the best fuzzy match for pattern in text using sliding window.

    Uses difflib.SequenceMatcher for similarity comparison.

    Args:
        text: Text to search in (should be normalized)
        pattern: Pattern to find (should be normalized)
        threshold: Minimum similarity ratio (0.0 to 1.0)
        return_best: If True, return best match even if below threshold

    Returns:
        Tuple of (start, end, score) or None if no match found
    """
    pattern_len = len(pattern)
    if pattern_len == 0 or len(text) < pattern_len:
        return None

    best_ratio = 0.0
    best_start = 0
    best_end = 0

    window_size = int(pattern_len * 1.2)
    text_lower = text.lower()
    pattern_lower = pattern.lower()

    # Use stepping for very long texts
    step = 1
    if len(text) >= SLIDING_WINDOW_STEP_THRESHOLD:
        step = max(1, len(text) // 5000)

    # First pass: find approximate location
    for i in range(0, len(text) - pattern_len + 1, step):
        window = text_lower[i:i + window_size]
        ratio = SequenceMatcher(None, pattern_lower, window).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i
            best_end = min(i + pattern_len, len(text))

    # Refine if we used stepping
    if step > 1 and best_ratio > 0:
        refine_start = max(0, best_start - step)
        refine_end = min(len(text) - pattern_len + 1, best_start + step)

        for i in range(refine_start, refine_end):
            window = text_lower[i:i + window_size]
            ratio = SequenceMatcher(None, pattern_lower, window).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i
                best_end = min(i + pattern_len, len(text))

    if best_ratio >= threshold or return_best:
        return (best_start, best_end, best_ratio)

    return None


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


def _anchor_based_search(page, page_index: int, search_text: str) -> dict | None:
    """
    Search for long text using anchor-based matching.

    Instead of matching the entire passage, finds the START (~40 chars) and
    END (~40 chars) of the passage, then highlights everything between them.
    This is robust against variations in the middle of the text.

    Args:
        page: PyMuPDF page object
        page_index: 0-indexed page number
        search_text: Full text to highlight

    Returns:
        Search result dict if found, None otherwise
    """
    page_height = page.rect.height

    # Extract anchors
    start_anchor = _extract_anchor(search_text, from_start=True)
    end_anchor = _extract_anchor(search_text, from_start=False)

    if not start_anchor or not end_anchor:
        return None

    # Build text index
    spans = _extract_page_spans(page)
    if not spans:
        return None

    cumulative, span_positions = _build_normalized_text_index(spans)
    if not cumulative:
        return None

    # Find start anchor
    normalized_start = normalize_for_matching(start_anchor)
    start_pos = cumulative.find(normalized_start)

    if start_pos == -1:
        match = _sliding_window_match(
            cumulative, normalized_start, ANCHOR_MATCH_THRESHOLD, return_best=True
        )
        if match and match[2] >= ANCHOR_MATCH_THRESHOLD:
            start_pos = match[0]
        else:
            return None

    # Find end anchor (search after start)
    normalized_end = normalize_for_matching(end_anchor)
    search_offset = start_pos + len(normalized_start) // 2
    end_pos = cumulative.find(normalized_end, search_offset)

    if end_pos == -1:
        remaining = cumulative[search_offset:]
        match = _sliding_window_match(
            remaining, normalized_end, ANCHOR_MATCH_THRESHOLD, return_best=True
        )
        if match and match[2] >= ANCHOR_MATCH_THRESHOLD:
            end_pos = search_offset + match[0] + len(normalized_end)
        else:
            # Estimate based on text length
            estimated_len = int(len(normalize_for_matching(search_text)) * 1.1)
            end_pos = min(start_pos + estimated_len, len(cumulative))
    else:
        end_pos = end_pos + len(normalized_end)

    # Get matching spans
    bboxes, texts = _get_spans_in_range(start_pos, end_pos, span_positions, spans)
    if not bboxes:
        return None

    return _build_search_result(page_index, bboxes, texts, page_height)


def _fuzzy_search_page(
    page,
    search_text: str,
    threshold: float | None = None,
) -> dict | None:
    """
    Perform fuzzy text search on a PDF page.

    Handles cases where exact matching fails due to hyphenation,
    whitespace differences, or character variations.

    Args:
        page: PyMuPDF page object
        search_text: Text to search for
        threshold: Minimum similarity (0.0-1.0), or None for dynamic

    Returns:
        Dict with 'rects', 'matched_text', 'score' if found, None otherwise
    """
    spans = _extract_page_spans(page)
    if not spans:
        return None

    cumulative, span_positions = _build_normalized_text_index(spans)
    normalized_search = normalize_for_matching(search_text)

    if not normalized_search or not cumulative:
        return None

    if threshold is None:
        threshold = _get_dynamic_threshold(len(search_text))

    # Try exact match first
    match_start = cumulative.find(normalized_search)

    if match_start != -1:
        match_end = match_start + len(normalized_search)
        bboxes, texts = _get_spans_in_range(match_start, match_end, span_positions, spans)

        if bboxes:
            return {
                "rects": bboxes,
                "matched_text": " ".join(texts),
                "score": 1.0,
            }

    # Try sliding window fuzzy match
    match_result = _sliding_window_match(
        cumulative, normalized_search, threshold, return_best=True
    )

    if match_result is None:
        return None

    match_start, match_end, match_score = match_result
    bboxes, texts = _get_spans_in_range(match_start, match_end, span_positions, spans)

    if not bboxes:
        return None

    # Return result (may be below threshold for debug purposes)
    return {
        "rects": bboxes if match_score >= threshold else [],
        "matched_text": " ".join(texts),
        "score": match_score,
    }


def _search_single_page(
    page,
    page_index: int,
    search_text: str,
    fuzzy: bool,
    best_debug: dict,
) -> dict | None:
    """
    Search for text on a single PDF page using multiple strategies.

    Strategy order:
    1. Anchor-based matching (for long text)
    2. Exact search via PyMuPDF
    3. Fuzzy matching

    Args:
        page: PyMuPDF page object
        page_index: 0-indexed page number
        search_text: Text to search for
        fuzzy: Whether to use fuzzy matching as fallback
        best_debug: Dict to track best match for debug info (mutated)

    Returns:
        Search result dict if found, None otherwise
    """
    page_height = page.rect.height

    # Strategy 1: Anchor-based matching for long passages
    if len(search_text) > ANCHOR_MIN_TEXT_LENGTH:
        result = _anchor_based_search(page, page_index, search_text)
        if result:
            return result

    # Strategy 2: Exact search
    text_instances = page.search_for(search_text)

    if not text_instances:
        # Try with normalized whitespace
        normalized = " ".join(search_text.split())
        text_instances = page.search_for(normalized)

    if text_instances:
        rects, min_y, min_x = _convert_rects_to_zotero(
            [r for r in text_instances], page_height
        )
        return {
            "pageIndex": page_index,
            "rects": rects,
            "sort_index": _build_sort_index(page_index, min_y, min_x),
            "matched_text": search_text,
        }

    # Strategy 3: Fuzzy matching
    if fuzzy:
        fuzzy_result = _fuzzy_search_page(page, search_text)

        if fuzzy_result:
            # Update debug info
            score = fuzzy_result.get("score", 0)
            if score > best_debug["score"]:
                best_debug["match"] = fuzzy_result.get("matched_text")
                best_debug["score"] = score
                best_debug["page"] = page_index

            # Return if we have valid rects
            if fuzzy_result.get("rects"):
                bboxes = fuzzy_result["rects"]
                rects, min_y, min_x = _convert_rects_to_zotero(bboxes, page_height)

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

def find_text_position(
    pdf_path: str,
    page_num: int,
    search_text: str,
    fuzzy: bool = True,
    search_neighbors: int = DEFAULT_NEIGHBOR_PAGES,
) -> dict:
    """
    Search for text in a PDF and return position data for Zotero annotation.

    Searches the specified page first, then neighboring pages if not found.
    Uses multiple matching strategies (anchor-based, exact, fuzzy) to handle
    various PDF text extraction issues.

    Args:
        pdf_path: Path to the PDF file
        page_num: 1-indexed page number to search on
        search_text: Text to find
        fuzzy: If True, use fuzzy matching as fallback
        search_neighbors: Number of pages to search on either side

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
    try:
        import fitz
    except ImportError:
        raise ImportError(
            "pymupdf is required for PDF text search. "
            "Install it with: pip install pymupdf"
        )

    doc = fitz.open(pdf_path)

    try:
        target_index = page_num - 1
        total_pages = len(doc)

        if target_index < 0 or target_index >= total_pages:
            return {
                "error": f"Page {page_num} out of range (PDF has {total_pages} pages)",
                "best_match": None,
                "best_score": 0,
                "pages_searched": [],
            }

        # Build page search order: target first, then neighbors
        pages_to_search = [target_index]
        for offset in range(1, search_neighbors + 1):
            if target_index - offset >= 0:
                pages_to_search.append(target_index - offset)
            if target_index + offset < total_pages:
                pages_to_search.append(target_index + offset)

        best_debug = {"match": None, "score": 0.0, "page": None}

        for page_index in pages_to_search:
            page = doc[page_index]
            result = _search_single_page(page, page_index, search_text, fuzzy, best_debug)

            if result:
                return result

        # No match found
        return {
            "error": f"Could not find text on page {page_num} or neighboring pages",
            "best_match": best_debug["match"],
            "best_score": best_debug["score"],
            "page_found": best_debug["page"] + 1 if best_debug["page"] is not None else None,
            "pages_searched": [p + 1 for p in pages_to_search],
        }

    finally:
        doc.close()


def get_page_label(pdf_path: str, page_num: int) -> str:
    """
    Get the page label for a given page number.

    Some PDFs have custom page labels (e.g., "i", "ii", "1", "2").

    Args:
        pdf_path: Path to the PDF file
        page_num: 1-indexed page number

    Returns:
        Page label if available, otherwise the page number as string
    """
    try:
        import fitz
    except ImportError:
        return str(page_num)

    doc = fitz.open(pdf_path)

    try:
        page_index = page_num - 1

        if page_index < 0 or page_index >= len(doc):
            return str(page_num)

        page = doc[page_index]

        if hasattr(page, "get_label"):
            label = page.get_label()
            if label:
                return label

        return str(page_num)

    finally:
        doc.close()


def verify_pdf_attachment(pdf_path: str) -> bool:
    """
    Verify that a file is a valid PDF.

    Args:
        pdf_path: Path to the file to check

    Returns:
        True if valid PDF, False otherwise
    """
    try:
        import fitz

        doc = fitz.open(pdf_path)
        is_pdf = doc.is_pdf
        doc.close()
        return is_pdf
    except Exception:
        return False


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
