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


def _review_scan_page(pdf, pymupdf):
    page = pdf.new_page()
    image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
    image.clear_with(250)
    page.insert_image(page.rect, stream=image.tobytes("png"))
    return page


@pytest.mark.parametrize("variant", ["misread", "heading", "case"])
def test_two_text_layers_do_not_duplicate_the_page(tmp_path, variant):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "two-layers.pdf"
    body = " ".join(["the quiet market sold silver lamps and simple maps"] * 12)
    with pymupdf.open() as pdf:
        page = _review_scan_page(pdf, pymupdf)
        if variant == "heading":
            for box, text, size in [
                ((60, 80, 540, 300), body, 11),
                ((60, 320, 540, 360), "Second section", 22),
                ((60, 370, 540, 700), body, 11),
            ]:
                for mode in (0, 3):
                    page.insert_textbox(pymupdf.Rect(box), text, fontsize=size, render_mode=mode)
        else:
            page.insert_textbox(pymupdf.Rect(60, 80, 540, 760), body, fontsize=11)
            hidden = body.upper() if variant == "case" else body.replace("m", "rn").replace("l", "1")
            page.insert_textbox(pymupdf.Rect(60, 80, 540, 760), hidden, fontsize=11, render_mode=3)
        pdf.save(path)
    doc = extract_pdf(path, pages=[0])
    # At most one transcription plus the recovery hint, never both full layers.
    assert len(re.findall(r"\w{2,}", doc.text)) <= (240 if variant == "heading" else 140)
    if variant in ("heading", "case"):
        assert "*[Plain text" not in doc.text


def _review_chart_pdf(path, kind, pymupdf):
    doc = pymupdf.open()
    page = doc.new_page()
    image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), False)
    image.clear_with(200)
    page.insert_image(
        page.rect if kind == "glyph_scan" else pymupdf.Rect(72, 80, 400, 300),
        stream=image.tobytes("png"),
        keep_proportion=False,
    )
    page.insert_textbox(
        pymupdf.Rect(90, 90, 500, 280),
        "Readable hidden transcription of scan " * 15
        if kind == "glyph_scan"
        else "Annual Rainfall in Millimetres by Region",
        fontsize=11,
        render_mode=3,
    )
    page.insert_text((90, 280), "1990 2000 2010 Year", fontsize=11, render_mode=3)
    body = [
        "The committee reviewed the proposal and found the budget sound.",
        "Several members asked for a clearer timeline before the vote.",
        "A revised plan will be circulated to all members next month.",
    ] * (1 if kind == "glyph_scan" else 4)
    if kind == "figure":
        page.insert_text((72, 330), "Results and Discussion", fontsize=20)
        for i, line in enumerate(body):
            page.insert_text((72, 360 + 16 * i), line, fontsize=11)
    else:
        chars = sorted(set("".join(body)))
        code = {c: i + 1 for i, c in enumerate(chars)}
        font = doc.get_new_xref()
        glyph_names = " ".join(f"/C{ord(c)}" for c in chars)
        doc.update_object(
            font,
            f"<</Type/Font/Subtype/Type1/BaseFont/Times-Roman/FirstChar 1/LastChar {len(chars)}"
            f"/Encoding<</Type/Encoding/Differences[1 {glyph_names}]>>>>",
        )
        page.clean_contents()
        xref = page.xref
        for key in ("Resources", "Font"):
            xtype, value = doc.xref_get_key(xref, key)
            if xtype != "xref":
                break
            xref = int(value.split()[0])
        doc.xref_set_key(
            xref,
            "FX" if key == "Font" and xtype == "xref" else ("Font/FX" if key == "Font" else "Resources/Font/FX"),
            f"{font} 0 R",
        )
        ops = "".join(
            f"BT /FX 11 Tf 72 {462 - 16 * i} Td <{''.join(f'{code[c]:02x}' for c in line)}> Tj ET\n"
            for i, line in enumerate(body)
        )
        content = page.get_contents()[0]
        doc.update_stream(content, doc.xref_stream(content) + b"\n" + ops.encode())
    doc.new_page().insert_text((72, 100), "An ordinary second page.", fontsize=11)
    doc.save(path)
    doc.close()


@pytest.mark.parametrize("kind", ["glyph", "figure"])
def test_chart_ocr_does_not_replace_the_body(tmp_path, kind):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / f"{kind}.pdf"
    _review_chart_pdf(path, kind, pymupdf)
    doc = extract_pdf(path, pages=[0])
    assert "*[Plain text" not in doc.text
    assert not re.search(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", doc.text)
    if kind == "glyph":
        assert doc.needs_ocr == (0,)
    else:
        assert "# Results and Discussion" in doc.text
        assert "The committee reviewed" in doc.text


def test_failed_optional_page_does_not_stop_later_pages(scanned_pdf, monkeypatch, caplog):
    pymupdf = pytest.importorskip("pymupdf")
    original = pymupdf.Page.get_texttrace

    def fail_first(self, *args, **kwargs):
        if self.number == 0:
            raise RuntimeError("synthetic first-page failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_texttrace", fail_first)
    doc = extract_pdf(scanned_pdf, pages=[0, 1])
    assert "OCR_PAGE_0" not in doc.pages[0]
    assert "OCR_PAGE_1" in doc.pages[1]
    assert "recovery" in caplog.text.lower()


def test_undecodable_visible_remainder_keeps_ocr_route(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "unreadable-remainder.pdf"
    _review_chart_pdf(path, "glyph_scan", pymupdf)
    doc = extract_pdf(path, pages=[0])
    assert doc.needs_ocr == (0,)
    assert "Plain text from the PDF's OCR layer" not in doc.text
    assert "\ufffd" not in doc.text


def _cost_pdf(tmp_path, pymupdf, mixed):
    path = tmp_path / "cost.pdf"
    with pymupdf.open() as pdf:
        for short in [True, False] if mixed else [False, False]:
            page = _review_scan_page(pdf, pymupdf)
            page.insert_textbox(
                pymupdf.Rect(60, 100, 520, 700),
                " ".join(f"word{n}" for n in range(80)),
                fontsize=12,
                render_mode=3 if short else 0,
            )
            if short:
                page.insert_text((60, 800), "DOWNLOAD STAMP", fontsize=10)
        pdf.save(path)
    return path


def test_rich_document_never_opens_optional_parser(tmp_path, monkeypatch):
    import pymupdf

    path = _cost_pdf(tmp_path, pymupdf, mixed=False)
    import pdf_inspector

    expected = tuple(p.markdown for p in pdf_inspector.extract_pages_markdown(str(path)).pages)
    assert all(len(text.split()) >= 60 for text in expected)
    opened = []
    original = pymupdf.open

    def counted(*args, **kwargs):
        opened.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(pymupdf, "open", counted)
    assert extract_pdf(path).pages == expected
    assert opened == []


def test_mixed_document_inspects_images_only_on_short_pages(tmp_path, monkeypatch):
    import pymupdf

    path = _cost_pdf(tmp_path, pymupdf, mixed=True)
    visited = []
    original = pymupdf.Page.get_image_info

    def counted(page, *args, **kwargs):
        visited.append(page.number)
        return original(page, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_image_info", counted)
    doc = extract_pdf(path)
    assert "word79" in doc.pages[0] and "DOWNLOAD STAMP" in doc.pages[0]
    assert "word79" in doc.pages[1] and "*[Plain text" not in doc.pages[1]
    assert visited == [0]


@pytest.mark.parametrize("variant", ["heading", "misread"])
def test_short_duplicate_layers_never_emit_both_copies(tmp_path, variant):
    import pdf_inspector
    import pymupdf

    path = tmp_path / f"short-{variant}.pdf"
    body = "The quiet market sold silver lamps and simple maps."
    with pymupdf.open() as pdf:
        page = _review_scan_page(pdf, pymupdf)
        for mode in (0, 3):
            text = body.replace("m", "rn").replace("l", "1") if mode == 3 and variant == "misread" else body
            page.insert_text((60, 80), text, fontsize=11, render_mode=mode)
            page.insert_text((60, 130), "Second section", fontsize=22, render_mode=mode)
            page.insert_text((60, 180), text, fontsize=11, render_mode=mode)
        pdf.save(path)
    expected = pdf_inspector.extract_pages_markdown(str(path)).pages[0].markdown
    assert len(expected.split()) < 60
    doc = extract_pdf(path)
    assert len(doc.text.split()) <= len(expected.split()) + 12
    if variant == "heading":
        assert doc.pages == (expected,)
