"""Regression tests for annotation coordinate conversion (real PyMuPDF).

Zotero stores annotation rects in the PDF's native user space (MediaBox
lower-left origin), while PyMuPDF search results live in a CropBox-normalized
space with a (0, 0) origin. These tests assert that find_text_position and
build_area_position_data map coordinates back to user space via the inverse
page transformation matrix, covering PDFs whose page box origin is not (0, 0).
"""

import os
import tempfile

import pytest

from zotero_mcp.pdf_utils import build_area_position_data, find_text_position

SENTENCE = "judgmental anchoring has durable effects lasting up to one week"

# Page boxes from the reproduction case (Furnham & Boo 2011):
# MediaBox [41.9, 46.5, 637.2, 840.2], CropBox [41.9, 0, 637.2, 793.7].
OFFSET_MEDIABOX = (41.9, 46.5, 637.2, 840.2)


def _make_pdf(tmpdir, mediabox=None, name="test.pdf"):
    """Create a one-page PDF containing SENTENCE, return (path, reference_rects).

    reference_rects are the search_for results mapped through the inverse
    transformation matrix, i.e. the expected user-space rects.
    """
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    if mediabox is not None:
        page.set_mediabox(fitz.Rect(*mediabox))
    page.insert_text(fitz.Point(72, 100), SENTENCE, fontsize=11)

    inv = ~page.transformation_matrix
    reference_rects = [
        [round(v, 4) for v in (r * inv)]
        for r in page.search_for(SENTENCE)
    ]
    assert reference_rects, "test PDF must contain the search text"

    path = os.path.join(tmpdir, name)
    doc.save(path)
    doc.close()
    return path, reference_rects


def _assert_rects_close(rects, reference_rects, tolerance=1.0):
    assert len(rects) == len(reference_rects)
    for rect, reference in zip(rects, reference_rects):
        assert rect == pytest.approx(reference, abs=tolerance)


@pytest.mark.parametrize("mediabox", [None, OFFSET_MEDIABOX])
def test_find_text_position_returns_user_space_rects(mediabox):
    with tempfile.TemporaryDirectory() as tmpdir:
        path, reference_rects = _make_pdf(tmpdir, mediabox=mediabox)

        result = find_text_position(path, 1, SENTENCE)

        assert "error" not in result
        assert result["pageIndex"] == 0
        _assert_rects_close(result["rects"], reference_rects)


def test_find_text_position_offset_mediabox_includes_origin():
    """The stored rect must be translated by the MediaBox origin (41.9, 46.5)
    relative to what the CropBox-normalized conversion would produce."""
    import fitz

    with tempfile.TemporaryDirectory() as tmpdir:
        path, _ = _make_pdf(tmpdir, mediabox=OFFSET_MEDIABOX)

        doc = fitz.open(path)
        page = doc[0]
        pymupdf_rect = page.search_for(SENTENCE)[0]
        crop_height = page.rect.height
        doc.close()

        result = find_text_position(path, 1, SENTENCE)
        rect = result["rects"][0]

        assert rect[0] == pytest.approx(pymupdf_rect.x0 + 41.9, abs=1.0)
        assert rect[2] == pytest.approx(pymupdf_rect.x1 + 41.9, abs=1.0)
        assert rect[1] == pytest.approx(
            (crop_height - pymupdf_rect.y1) + 46.5, abs=1.0
        )
        assert rect[3] == pytest.approx(
            (crop_height - pymupdf_rect.y0) + 46.5, abs=1.0
        )

        # Sort index is derived from the corrected coordinates.
        page_part, y_part, x_part = result["sort_index"].split("|")
        assert page_part == "00000"
        assert int(y_part) == int(rect[1])
        assert int(x_part) == int(rect[0])


def test_build_area_position_offset_mediabox():
    with tempfile.TemporaryDirectory() as tmpdir:
        path, _ = _make_pdf(tmpdir, mediabox=OFFSET_MEDIABOX)

        result = build_area_position_data(path, 1, 0.1, 0.2, 0.3, 0.4)

        assert "error" not in result
        (rect,) = result["rects"]

        # Fractions are relative to the visible page (CropBox, 595.3 x 793.7);
        # the stored rect adds the MediaBox origin: x + 41.9, y -> 840.2 - y.
        crop_w, crop_h = 637.2 - 41.9, 793.7
        assert rect[0] == pytest.approx(0.1 * crop_w + 41.9, abs=0.01)
        assert rect[2] == pytest.approx(0.4 * crop_w + 41.9, abs=0.01)
        assert rect[3] == pytest.approx(840.2 - 0.2 * crop_h, abs=0.01)
        assert rect[1] == pytest.approx(840.2 - 0.6 * crop_h, abs=0.01)
