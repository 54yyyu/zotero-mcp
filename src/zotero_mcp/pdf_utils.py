"""
PDF utility functions for annotation creation.

This module provides functions to search for text in PDFs and extract
position data needed for creating Zotero annotations.
"""

import json
import re


def normalize_text(text: str) -> str:
    """
    Normalize text for fuzzy matching.

    Handles common PDF text extraction issues:
    - Hyphenation at line breaks (e.g., "regard-\\nless" -> "regardless")
    - Multiple whitespace -> single space
    - Various dash types, curly quotes, special characters

    Args:
        text: Raw text to normalize

    Returns:
        Normalized text for comparison
    """
    # Remove hyphenation at line breaks (hyphen followed by newline and optional spaces)
    text = re.sub(r'-\s*\n\s*', '', text)
    # Also handle soft hyphens and other dash variants at line breaks
    text = re.sub(r'[\u00ad\u2010\u2011-]\s*\n\s*', '', text)
    # Normalize various dash types to simple hyphen
    text = text.replace('\u2014', '-')  # em-dash
    text = text.replace('\u2013', '-')  # en-dash
    text = text.replace('\u2012', '-')  # figure dash
    text = text.replace('\u2011', '-')  # non-breaking hyphen
    text = text.replace('\u2010', '-')  # hyphen
    # Normalize curly quotes to straight quotes
    text = text.replace('\u2018', "'")  # left single quote
    text = text.replace('\u2019', "'")  # right single quote
    text = text.replace('\u201c', '"')  # left double quote
    text = text.replace('\u201d', '"')  # right double quote
    # Handle common ligatures
    text = text.replace('\ufb01', 'fi')  # fi ligature
    text = text.replace('\ufb02', 'fl')  # fl ligature
    text = text.replace('\ufb00', 'ff')  # ff ligature
    text = text.replace('\ufb03', 'ffi')  # ffi ligature
    text = text.replace('\ufb04', 'ffl')  # ffl ligature
    # Collapse all whitespace to single spaces
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def normalize_for_matching(text: str) -> str:
    """
    Aggressively normalize text for fuzzy matching comparison.

    This removes ALL spaces to handle PDFs where words are stored
    without proper spacing between spans.

    Args:
        text: Text to normalize

    Returns:
        Text with all spaces removed, lowercased
    """
    # First apply standard normalization
    text = normalize_text(text)
    # Remove all spaces for comparison
    text = re.sub(r'\s+', '', text)
    # Lowercase for case-insensitive matching
    text = text.lower()
    return text


def find_text_position(
    pdf_path: str,
    page_num: int,
    search_text: str,
    fuzzy: bool = True,
    search_neighbors: int = 2,
) -> dict:
    """
    Search for text in a PDF and return position data for Zotero annotation.

    Args:
        pdf_path: Path to the PDF file
        page_num: 1-indexed page number to search on
        search_text: Text to find (exact or fuzzy match)
        fuzzy: If True, use fuzzy matching to handle hyphenation and line breaks
        search_neighbors: Number of neighboring pages to search if not found on target page

    Returns:
        Dict with annotation position data on success:
        - pageIndex: 0-indexed page number
        - rects: list of [x1, y1, x2, y2] bounding boxes in PDF coordinates
        - sort_index: string for annotationSortIndex (PPPPP|YYYYYY|XXXXX)
        - matched_text: the actual text that was matched (useful for fuzzy matches)

        Dict with debug info on failure:
        - error: error message
        - best_match: best matching text found (if any)
        - best_score: similarity score of best match
        - page_searched: list of pages that were searched
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError(
            "pymupdf is required for PDF text search. "
            "Install it with: pip install pymupdf"
        )

    doc = fitz.open(pdf_path)

    try:
        # Convert to 0-indexed
        target_page_index = page_num - 1
        total_pages = len(doc)

        if target_page_index < 0 or target_page_index >= total_pages:
            return {
                "error": f"Page {page_num} out of range (PDF has {total_pages} pages)",
                "best_match": None,
                "best_score": 0,
                "pages_searched": [],
            }

        # Build list of pages to search: target first, then neighbors
        pages_to_search = [target_page_index]
        for offset in range(1, search_neighbors + 1):
            if target_page_index - offset >= 0:
                pages_to_search.append(target_page_index - offset)
            if target_page_index + offset < total_pages:
                pages_to_search.append(target_page_index + offset)

        # Track best match across all pages for debug info
        best_debug = {"match": None, "score": 0, "page": None}

        for page_index in pages_to_search:
            page = doc[page_index]
            result = _search_single_page(page, page_index, search_text, fuzzy, best_debug)
            if result and "error" not in result:
                return result

        # No match found - return debug info
        return {
            "error": f"Could not find text on page {page_num} or neighboring pages",
            "best_match": best_debug["match"],
            "best_score": best_debug["score"],
            "page_found": best_debug["page"] + 1 if best_debug["page"] is not None else None,
            "pages_searched": [p + 1 for p in pages_to_search],
        }

    finally:
        doc.close()


def _search_single_page(
    page, page_index: int, search_text: str, fuzzy: bool, best_debug: dict
) -> dict | None:
    """
    Search for text on a single PDF page.

    Args:
        page: PyMuPDF page object
        page_index: 0-indexed page number
        search_text: Text to search for
        fuzzy: Whether to use fuzzy matching
        best_debug: Dict to track best match for debug info (mutated)

    Returns:
        Dict with position data if found, None otherwise
    """
    page_height = page.rect.height

    # Try exact search first
    text_instances = page.search_for(search_text)

    if not text_instances:
        # Try with normalized whitespace
        normalized_text = " ".join(search_text.split())
        text_instances = page.search_for(normalized_text)

    # If still no match and fuzzy is enabled, try fuzzy matching
    matched_text = search_text
    if not text_instances and fuzzy:
        fuzzy_result = _fuzzy_search_page(page, search_text)
        if fuzzy_result:
            if fuzzy_result.get("score", 1.0) >= 0:  # Accept any valid match
                text_instances = fuzzy_result["rects"]
                matched_text = fuzzy_result["matched_text"]
            # Update debug info with best match
            if fuzzy_result.get("score", 0) > best_debug["score"]:
                best_debug["match"] = fuzzy_result.get("matched_text")
                best_debug["score"] = fuzzy_result.get("score", 0)
                best_debug["page"] = page_index

    if not text_instances:
        return None

    # Convert fitz.Rect objects to [x1, y1, x2, y2] lists
    # PyMuPDF uses top-left origin (y increases downward)
    # Zotero/PDF uses bottom-left origin (y increases upward)
    rects = []
    min_y_pdf = float("inf")
    min_x = float("inf")

    for rect in text_instances:
        # Handle both fitz.Rect objects and raw tuples/lists
        if hasattr(rect, 'x0'):
            x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
        else:
            x0, y0, x1, y1 = rect[0], rect[1], rect[2], rect[3]

        # Transform Y coordinates from top-left to bottom-left origin
        pdf_y1 = page_height - y1
        pdf_y2 = page_height - y0

        rects.append([x0, pdf_y1, x1, pdf_y2])
        min_y_pdf = min(min_y_pdf, pdf_y1)
        min_x = min(min_x, x0)

    # Build sort index: PPPPP|YYYYYY|XXXXX
    sort_index = f"{page_index:05d}|{int(min_y_pdf):06d}|{int(min_x):05d}"

    return {
        "pageIndex": page_index,
        "rects": rects,
        "sort_index": sort_index,
        "matched_text": matched_text,
    }


def _get_dynamic_threshold(text_length: int) -> float:
    """
    Calculate dynamic fuzzy matching threshold based on text length.

    Longer passages need lower thresholds because there's more opportunity
    for small variations to accumulate.

    Args:
        text_length: Number of characters in the search text

    Returns:
        Threshold between 0.65 and 0.85
    """
    if text_length < 50:
        return 0.85  # Short text: be strict
    elif text_length < 150:
        return 0.75  # Medium text: moderate
    else:
        return 0.65  # Long passages: be lenient


def _fuzzy_search_page(page, search_text: str, threshold: float | None = None) -> dict | None:
    """
    Perform fuzzy text search on a PDF page.

    This handles cases where the search text doesn't exactly match the PDF
    due to hyphenation at line breaks, different whitespace, etc.

    Args:
        page: PyMuPDF page object
        search_text: Text to search for
        threshold: Minimum similarity ratio (0.0 to 1.0) for a match.
                   If None, uses dynamic threshold based on text length.

    Returns:
        Dict with 'rects', 'matched_text', and 'score' if found, None otherwise.
        Even when no match above threshold, may return best match info for debugging.
    """
    # Get all text blocks with position info
    blocks = page.get_text("dict", flags=11)["blocks"]

    # Extract text spans with their positions
    spans = []
    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                spans.append({
                    "text": span["text"],
                    "bbox": span["bbox"],  # (x0, y0, x1, y1)
                })

    if not spans:
        return None

    # Build cumulative normalized text and track span boundaries
    cumulative_normalized = ""
    span_norm_positions = []  # (norm_start, norm_end, span_idx)

    for i, span in enumerate(spans):
        norm_start = len(cumulative_normalized)
        span_normalized = normalize_for_matching(span["text"])
        cumulative_normalized += span_normalized
        norm_end = len(cumulative_normalized)
        span_norm_positions.append((norm_start, norm_end, i))

    # Normalize search text the same way
    normalized_search = normalize_for_matching(search_text)

    if not normalized_search or not cumulative_normalized:
        return None

    # Use dynamic threshold if not specified
    if threshold is None:
        threshold = _get_dynamic_threshold(len(search_text))

    # Try exact match first
    match_start = cumulative_normalized.find(normalized_search)
    match_score = 1.0  # Exact match

    if match_start == -1:
        # For long passages, try matching just the first sentence
        first_sentence_match = None
        if len(search_text) > 100:
            first_sentence_match = _try_first_sentence_match(
                cumulative_normalized, search_text, span_norm_positions, spans
            )
            if first_sentence_match:
                return first_sentence_match

        # Try sliding window fuzzy match
        match_result = _sliding_window_match(
            cumulative_normalized, normalized_search, threshold, return_best=True
        )
        if match_result is None:
            return None

        match_start, match_end, match_score = match_result

        # If score is below threshold, return debug info but mark as not matched
        if match_score < threshold:
            # Find spans for debug info
            debug_rects, debug_texts = _get_matching_spans(
                match_start, match_end, span_norm_positions, spans
            )
            return {
                "rects": [],  # Empty rects = no valid match
                "matched_text": " ".join(debug_texts) if debug_texts else None,
                "score": match_score,
            }
    else:
        match_end = match_start + len(normalized_search)

    # Find which spans overlap with our match
    matching_rects, matched_span_texts = _get_matching_spans(
        match_start, match_end, span_norm_positions, spans
    )

    if not matching_rects:
        return None

    matched_text = " ".join(matched_span_texts)

    return {
        "rects": matching_rects,
        "matched_text": matched_text,
        "score": match_score,
    }


def _get_matching_spans(
    match_start: int, match_end: int, span_norm_positions: list, spans: list
) -> tuple[list, list]:
    """Get spans that overlap with a match region."""
    matching_rects = []
    matched_span_texts = []

    for norm_start, norm_end, span_idx in span_norm_positions:
        if norm_start < match_end and norm_end > match_start:
            matching_rects.append(spans[span_idx]["bbox"])
            matched_span_texts.append(spans[span_idx]["text"])

    return matching_rects, matched_span_texts


def _try_first_sentence_match(
    cumulative_normalized: str,
    search_text: str,
    span_norm_positions: list,
    spans: list
) -> dict | None:
    """
    For long passages, try matching just the first sentence and extend from there.

    This handles cases where the full passage may have too many variations,
    but the first sentence matches well.
    """
    # Extract first sentence (up to first period, question mark, or ~100 chars)
    first_sentence = search_text
    for delimiter in ['. ', '? ', '! ']:
        idx = search_text.find(delimiter)
        if idx > 0 and idx < 150:
            first_sentence = search_text[:idx + 1]
            break

    if len(first_sentence) >= len(search_text):
        return None  # No sentence break found

    normalized_first = normalize_for_matching(first_sentence)

    # Try exact match of first sentence
    match_start = cumulative_normalized.find(normalized_first)

    if match_start == -1:
        # Try fuzzy match with higher threshold for short text
        result = _sliding_window_match(
            cumulative_normalized, normalized_first, 0.80, return_best=True
        )
        if result is None or result[2] < 0.80:
            return None
        match_start = result[0]

    # Found first sentence - now extend to cover full search text length
    # Estimate end position based on original search length ratio
    ratio = len(search_text) / len(first_sentence)
    estimated_end = match_start + int(len(normalized_first) * ratio * 1.1)
    match_end = min(estimated_end, len(cumulative_normalized))

    # Get matching spans
    matching_rects, matched_span_texts = _get_matching_spans(
        match_start, match_end, span_norm_positions, spans
    )

    if not matching_rects:
        return None

    return {
        "rects": matching_rects,
        "matched_text": " ".join(matched_span_texts),
        "score": 0.90,  # High score since we matched first sentence exactly
    }


def _sliding_window_match(
    text: str, pattern: str, threshold: float, return_best: bool = False
) -> tuple | None:
    """
    Find the best fuzzy match for pattern in text using sliding window.

    Args:
        text: Text to search in
        pattern: Pattern to find
        threshold: Minimum similarity ratio
        return_best: If True, return best match even if below threshold (for debug)

    Returns:
        Tuple of (start, end, score) or None if no match found.
        If return_best=True, returns best match regardless of threshold.
    """
    from difflib import SequenceMatcher

    pattern_len = len(pattern)
    if pattern_len == 0 or len(text) < pattern_len:
        return None

    best_ratio = 0
    best_start = 0
    best_end = 0

    # Use a window slightly larger than the pattern to account for variations
    window_size = int(pattern_len * 1.2)

    text_lower = text.lower()
    pattern_lower = pattern.lower()

    # Optimization: skip positions in larger steps for very long texts
    step = 1 if len(text) < 10000 else max(1, len(text) // 5000)

    for i in range(0, len(text) - pattern_len + 1, step):
        window = text_lower[i:i + window_size]
        ratio = SequenceMatcher(None, pattern_lower, window).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i
            best_end = min(i + pattern_len, len(text))

    # If we used stepping, refine around best position
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


def _map_normalized_to_original(original: str, normalized: str, normalized_pos: int) -> int:
    """
    Map a position in normalized text back to approximate position in original.

    This is a best-effort mapping since normalization is lossy.
    """
    if normalized_pos <= 0:
        return 0
    if normalized_pos >= len(normalized):
        return len(original)

    # Calculate ratio and apply to original length
    ratio = normalized_pos / len(normalized)
    return int(ratio * len(original))


def get_page_label(pdf_path: str, page_num: int) -> str:
    """
    Get the page label for a given page number.

    Some PDFs have custom page labels (e.g., "i", "ii", "1", "2").
    This function returns the label if available, otherwise the page number.

    Args:
        pdf_path: Path to the PDF file
        page_num: 1-indexed page number

    Returns:
        Page label as string
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

        # Try to get page label from PDF metadata
        # PyMuPDF provides this via page.get_label() in newer versions
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
        JSON string for annotationPosition field
    """
    position_data = {
        "pageIndex": page_index,
        "rects": rects,
    }
    return json.dumps(position_data)
