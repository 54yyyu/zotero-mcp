"""
PDF utility functions for annotation creation.

This module provides functions to search for text in PDFs and extract
position data needed for creating Zotero annotations.
"""

import json


def find_text_position(pdf_path: str, page_num: int, search_text: str) -> dict | None:
    """
    Search for text in a PDF and return position data for Zotero annotation.

    Args:
        pdf_path: Path to the PDF file
        page_num: 1-indexed page number to search on
        search_text: Exact text to find

    Returns:
        Dict with annotation position data:
        - pageIndex: 0-indexed page number
        - rects: list of [x1, y1, x2, y2] bounding boxes in PDF coordinates
        - sort_index: string for annotationSortIndex (PPPPP|YYYYYY|XXXXX)

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

        # Search for the text on the page
        text_instances = page.search_for(search_text)

        if not text_instances:
            # Try with normalized whitespace
            normalized_text = " ".join(search_text.split())
            text_instances = page.search_for(normalized_text)

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
            # Transform Y coordinates from top-left to bottom-left origin
            # rect.y0 is top of text (smaller in fitz), rect.y1 is bottom (larger in fitz)
            # In PDF coords: y1 (bottom of rect) = page_height - rect.y1
            #                y2 (top of rect) = page_height - rect.y0
            pdf_y1 = page_height - rect.y1  # Bottom of highlight in PDF coords
            pdf_y2 = page_height - rect.y0  # Top of highlight in PDF coords

            rects.append([rect.x0, pdf_y1, rect.x1, pdf_y2])
            min_y_pdf = min(min_y_pdf, pdf_y1)
            min_x = min(min_x, rect.x0)

        # Build sort index: PPPPP|YYYYYY|XXXXX
        # Page (5 digits), Y position (6 digits), X position (5 digits)
        # Use PDF coordinates for sort index (y from bottom)
        sort_index = f"{page_index:05d}|{int(min_y_pdf):06d}|{int(min_x):05d}"

        return {
            "pageIndex": page_index,
            "rects": rects,
            "sort_index": sort_index,
        }

    finally:
        doc.close()


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
