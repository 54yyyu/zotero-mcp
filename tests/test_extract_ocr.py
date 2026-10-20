"""Real-parser regressions for scanned PDFs with an invisible OCR layer (#724)."""

import re
import sys

import pytest

from zotero_mcp.extract import PAGE_SEPARATOR, extract_pdf


@pytest.fixture
def scanned_pdf(tmp_path):
    pymupdf = pytest.importorskip("pymupdf", reason="Build the synthetic PDF")
    path = tmp_path / "ocr.pdf"
    with pymupdf.open() as pdf:
        for index in range(2):
            page = pdf.new_page()
            image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
            image.clear_with(245)
            page.insert_image(page.rect, stream=image.tobytes("png"))
            page.insert_textbox(
                pymupdf.Rect(60, 100, 520, 500),
                f"OCR_PAGE_{index}: " + " ".join(f"word{n}" for n in range(50)),
                fontsize=12,
                render_mode=3,
            )
            page.insert_text((60, 800), f"VISIBLE STAMP {index}", fontsize=10)
        page = pdf.new_page()
        page.insert_image(page.rect, stream=image.tobytes("png"))
        pdf.save(path)
    return path


@pytest.mark.parametrize(
    ("options", "page_numbers", "needs_ocr", "truncated"),
    [
        ({}, (0, 1, 2), (2,), False),
        ({"pages": [1]}, (1,), (), False),
        ({"pages": [2, 1, 0]}, (2, 1, 0), (2,), False),
        ({"max_pages": 1}, (0,), (), True),
    ],
)
def test_recovers_ocr_text_without_losing_page_provenance(scanned_pdf, options, page_numbers, needs_ocr, truncated):
    doc = extract_pdf(scanned_pdf, **options)
    assert doc.page_numbers == page_numbers
    assert doc.page_count == 3
    assert doc.needs_ocr == needs_ocr
    assert doc.truncated == truncated
    for number, text in zip(doc.page_numbers, doc.pages):
        if number < 2:
            assert f"OCR_PAGE_{number}" in text
            assert f"VISIBLE STAMP {number}" in text
            assert f"OCR_PAGE_{1 - number}" not in text
        else:
            assert "OCR_PAGE_" not in text
    assert doc.text == PAGE_SEPARATOR.join(doc.pages)
    assert doc.text.count(PAGE_SEPARATOR) == len(doc.pages) - 1


def test_missing_optional_pymupdf_keeps_markdown_result(scanned_pdf, monkeypatch):
    monkeypatch.setitem(sys.modules, "pymupdf", None)
    monkeypatch.setitem(sys.modules, "fitz", None)
    doc = extract_pdf(scanned_pdf, pages=[0])
    assert "VISIBLE STAMP 0" in doc.text
    assert "OCR_PAGE_0" not in doc.text


@pytest.mark.parametrize("with_image", [False, True])
def test_born_digital_text_keeps_markdown_structure(tmp_path, with_image):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "digital.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((60, 100), "Ordinary heading", fontsize=24)
        page.insert_text((60, 150), "Born digital body", fontsize=12)
        if with_image:
            image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
            image.clear_with(245)
            page.insert_image(pymupdf.Rect(60, 200, 160, 300), stream=image.tobytes("png"))
        pdf.save(path)
    import pdf_inspector

    expected = pdf_inspector.extract_pages_markdown(str(path)).pages[0].markdown
    assert extract_pdf(path).pages == (expected,)


def test_cached_page_reads_keep_recovered_text_on_the_correct_page(scanned_pdf):
    second = extract_pdf(scanned_pdf, pages=[1], reuse=True)
    first = extract_pdf(scanned_pdf, pages=[0], reuse=True)
    assert second.page_numbers == (1,)
    assert "OCR_PAGE_1" in second.text and "OCR_PAGE_0" not in second.text
    assert first.page_numbers == (0,)
    assert "OCR_PAGE_0" in first.text and "OCR_PAGE_1" not in first.text


def test_optional_recovery_error_preserves_successful_primary_parse(scanned_pdf, monkeypatch):
    import pymupdf

    def unavailable(*args, **kwargs):
        raise RuntimeError("Optional parser cannot open this file")

    monkeypatch.setattr(pymupdf, "open", unavailable)
    doc = extract_pdf(scanned_pdf, pages=[1])
    assert "VISIBLE STAMP 1" in doc.text
    assert doc.page_numbers == (1,)


def test_fragmented_ocr_spans_are_not_mistaken_for_visible_stamp(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "fragmented.pdf"
    body = "102020261020202610202026"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
        image.clear_with(245)
        page.insert_image(page.rect, stream=image.tobytes("png"))
        for index, char in enumerate(body):
            page.insert_text((60 + 12 * index, 100), char, fontsize=12, render_mode=3)
        page.insert_text((60, 800), "Downloaded at 10/20/2026", fontsize=10)
        pdf.save(path)
    assert body in re.sub(r"\s+", "", extract_pdf(path).text)
