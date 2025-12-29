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
    - Various dash types

    Args:
        text: Raw text to normalize

    Returns:
        Normalized text for comparison
    """
    # Remove hyphenation at line breaks (hyphen followed by newline and optional spaces)
    text = re.sub(r'-\s*\n\s*', '', text)
    # Also handle soft hyphens and other dash variants at line breaks
    text = re.sub(r'[\u00ad\u2010\u2011-]\s*\n\s*', '', text)
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


def find_text_position(pdf_path: str, page_num: int, search_text: str, fuzzy: bool = True) -> dict | None:
    """
    Search for text in a PDF and return position data for Zotero annotation.

    Args:
        pdf_path: Path to the PDF file
        page_num: 1-indexed page number to search on
        search_text: Text to find (exact or fuzzy match)
        fuzzy: If True, use fuzzy matching to handle hyphenation and line breaks

    Returns:
        Dict with annotation position data:
        - pageIndex: 0-indexed page number
        - rects: list of [x1, y1, x2, y2] bounding boxes in PDF coordinates
        - sort_index: string for annotationSortIndex (PPPPP|YYYYYY|XXXXX)
        - matched_text: the actual text that was matched (useful for fuzzy matches)

        Returns None if text not found.
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
        page_index = page_num - 1

        if page_index < 0 or page_index >= len(doc):
            return None

        page = doc[page_index]
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
                text_instances = fuzzy_result["rects"]
                matched_text = fuzzy_result["matched_text"]

        if not text_instances:
            return None

        # Convert fitz.Rect objects to [x1, y1, x2, y2] lists
        # PyMuPDF uses top-left origin (y increases downward)
        # Zotero/PDF uses bottom-left origin (y increases upward)
        # Transform: new_y = page_height - old_y
        rects = []
        min_y_pdf = float("inf")  # In PDF coordinates (from bottom)
        min_x = float("inf")

        for rect in text_instances:
            # Handle both fitz.Rect objects and raw tuples/lists
            if hasattr(rect, 'x0'):
                x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
            else:
                x0, y0, x1, y1 = rect[0], rect[1], rect[2], rect[3]

            # Transform Y coordinates from top-left to bottom-left origin
            pdf_y1 = page_height - y1  # Bottom of highlight in PDF coords
            pdf_y2 = page_height - y0  # Top of highlight in PDF coords

            rects.append([x0, pdf_y1, x1, pdf_y2])
            min_y_pdf = min(min_y_pdf, pdf_y1)
            min_x = min(min_x, x0)

        # Build sort index: PPPPP|YYYYYY|XXXXX
        # Page (5 digits), Y position (6 digits), X position (5 digits)
        # Use PDF coordinates for sort index (y from bottom)
        sort_index = f"{page_index:05d}|{int(min_y_pdf):06d}|{int(min_x):05d}"

        return {
            "pageIndex": page_index,
            "rects": rects,
            "sort_index": sort_index,
            "matched_text": matched_text,
        }

    finally:
        doc.close()


def _fuzzy_search_page(page, search_text: str, threshold: float = 0.8) -> dict | None:
    """
    Perform fuzzy text search on a PDF page.

    This handles cases where the search text doesn't exactly match the PDF
    due to hyphenation at line breaks, different whitespace, etc.

    Args:
        page: PyMuPDF page object
        search_text: Text to search for
        threshold: Minimum similarity ratio (0.0 to 1.0) for a match

    Returns:
        Dict with 'rects' (list of fitz.Rect) and 'matched_text' if found, None otherwise
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
    # This lets us map positions in normalized text back to spans
    cumulative_normalized = ""
    span_norm_positions = []  # (norm_start, norm_end, span_idx)

    for i, span in enumerate(spans):
        norm_start = len(cumulative_normalized)
        # Normalize each span's text (remove spaces, lowercase)
        span_normalized = normalize_for_matching(span["text"])
        cumulative_normalized += span_normalized
        norm_end = len(cumulative_normalized)
        span_norm_positions.append((norm_start, norm_end, i))

    # Normalize search text the same way
    normalized_search = normalize_for_matching(search_text)

    if not normalized_search or not cumulative_normalized:
        return None

    # Try to find the normalized search text
    match_start = cumulative_normalized.find(normalized_search)

    if match_start == -1:
        # Try sliding window fuzzy match
        match_result = _sliding_window_match(cumulative_normalized, normalized_search, threshold)
        if match_result is None:
            return None
        match_start, match_end, _ = match_result
    else:
        match_end = match_start + len(normalized_search)

    # Find which spans overlap with our match in normalized space
    matching_rects = []
    matched_span_texts = []

    for norm_start, norm_end, span_idx in span_norm_positions:
        # Check if this span overlaps with the match
        if norm_start < match_end and norm_end > match_start:
            matching_rects.append(spans[span_idx]["bbox"])
            matched_span_texts.append(spans[span_idx]["text"])

    if not matching_rects:
        return None

    # Reconstruct matched text from spans (with original spacing)
    matched_text = " ".join(matched_span_texts)

    return {
        "rects": matching_rects,
        "matched_text": matched_text,
    }


def _sliding_window_match(text: str, pattern: str, threshold: float) -> tuple | None:
    """
    Find the best fuzzy match for pattern in text using sliding window.

    Args:
        text: Text to search in
        pattern: Pattern to find
        threshold: Minimum similarity ratio

    Returns:
        Tuple of (start, end, matched_text) or None if no match above threshold
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

    for i in range(len(text) - pattern_len + 1):
        window = text_lower[i:i + window_size]
        ratio = SequenceMatcher(None, pattern_lower, window).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i
            # Estimate actual match length
            best_end = min(i + pattern_len + 20, len(text))

    if best_ratio >= threshold:
        # Refine the end position
        matched_text = text[best_start:best_end].strip()
        return (best_start, best_end, matched_text)

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
