"""Regression tests for PDF annotation sort indexes (real PyMuPDF).

Zotero orders annotations by their sort index, PPPPP|OOOOOO|TTTTT
(page|character offset|top). The offset counts the page's characters in
content order. On a two-column page, an annotation low in the left column
sorts before one at the top of the right column, because the left column's
text comes first.

These tests build such a page and check the sort indexes that
find_text_position and build_note_position_data produce.
"""

import os
import tempfile

import pytest

from zotero_mcp.pdf_utils import build_note_position_data, find_text_position

LEFT_TOP = "first sentence of the left column"
LEFT_BOTTOM = "last sentence of the left column"
RIGHT_TOP = "first sentence of the right column"

# Reading order is LEFT_TOP, LEFT_BOTTOM, RIGHT_TOP.
RIGHT_COLUMN_X = 330
SENTENCES = [
    (LEFT_TOP, (72, 100)),
    (LEFT_BOTTOM, (72, 700)),
    (RIGHT_TOP, (RIGHT_COLUMN_X, 100)),
]


def _make_pdf(tmpdir, name="two_columns.pdf"):
    """Create a one-page, two-column PDF containing SENTENCES, return its path."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for text, point in SENTENCES:
        page.insert_text(fitz.Point(*point), text, fontsize=11)

    path = os.path.join(tmpdir, name)
    doc.save(path)
    doc.close()
    return path


def _sort_index(path, text):
    result = find_text_position(path, 1, text)
    assert "error" not in result
    return result["sort_index"]


def test_sort_index_follows_reading_order():
    """Text low in the left column sorts before text at the top of the right
    column, even though the right column's text is higher on the page.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _make_pdf(tmpdir)

        indexes = [_sort_index(path, text) for text, _ in SENTENCES]

        assert indexes == sorted(indexes)


def test_sort_index_offset_and_top():
    """The offset counts the page's characters before the text, leaving out
    spaces, and the top is the text's distance from the top of the page.
    """
    import fitz

    with tempfile.TemporaryDirectory() as tmpdir:
        path = _make_pdf(tmpdir)

        doc = fitz.open(path)
        pymupdf_rect = doc[0].search_for(RIGHT_TOP)[0]
        doc.close()

        page_part, offset_part, top_part = _sort_index(path, RIGHT_TOP).split("|")
        assert page_part == "00000"
        assert int(offset_part) == len((LEFT_TOP + LEFT_BOTTOM).replace(" ", ""))
        assert int(top_part) == pytest.approx(pymupdf_rect.y0, abs=1.0)


def test_note_sort_index_uses_nearest_text():
    """A note in the right column's margin takes the offset of the nearest
    text, so it sorts after the whole left column. It still sorts before that
    text, because the note sits higher on the page.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _make_pdf(tmpdir)

        # Just above the start of RIGHT_TOP.
        result = build_note_position_data(path, 1, RIGHT_COLUMN_X / 612, 80 / 792)

        assert "error" not in result
        note_index = result["sort_index"]
        assert _sort_index(path, LEFT_BOTTOM) < note_index < _sort_index(path, RIGHT_TOP)
