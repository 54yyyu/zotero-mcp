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


# Based on the synthetic repros shared by @kremserw in PR #729.
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


def test_hidden_text_without_a_scan_image_keeps_primary_markdown(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "no-image.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((60, 100), "Ordinary heading", fontsize=24)
        page.insert_text((60, 160), "Visible body", fontsize=12)
        page.insert_text((60, 200), "HIDDEN OCR WORDS", fontsize=12, render_mode=3)
        pdf.save(path)
    doc = extract_pdf(path)
    assert "# Ordinary heading" in doc.text
    assert "HIDDEN OCR WORDS" not in doc.text
    assert "*[Plain text" not in doc.text


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


def test_recovered_hidden_only_scan_has_hint_and_clears_ocr_route(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "hidden-only.pdf"
    with pymupdf.open() as pdf:
        page = _review_scan_page(pdf, pymupdf)
        page.insert_text((60, 100), "Readable hidden transcription", fontsize=12, render_mode=3)
        pdf.save(path)
    doc = extract_pdf(path, pages=[0])
    assert "Plain text from the PDF's OCR layer" in doc.text
    assert "Readable hidden transcription" in doc.text
    assert doc.needs_ocr == ()


def test_positioned_words_inside_one_trace_span_keep_word_boundaries(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "positioned.pdf"
    with pymupdf.open() as pdf:
        page = _review_scan_page(pdf, pymupdf)
        page.insert_text((60, 100), "seed", render_mode=3)
        content = page.get_contents()[-1]
        pdf.update_stream(
            content,
            b"BT /helv 12 Tf 3 Tr 1 0 0 1 60 742 Tm "
            b"(first) Tj 35 0 Td (second) Tj 50 0 Td (third) Tj 0 -20 Td (fourth) Tj ET",
        )
        page.insert_text((60, 800), "STAMP", fontsize=10)
        pdf.save(path)
    doc = extract_pdf(path)
    assert re.findall(r"first|second|third|fourth", doc.text) == ["first", "second", "third", "fourth"]
    assert "firstsecond" not in doc.text
    assert "thirdfourth" not in doc.text
    assert "STAMP" in doc.text


@pytest.mark.parametrize("control", ["\x00", "\x0c", "\x85"])
def test_trace_controls_never_become_page_boundaries(tmp_path, monkeypatch, control):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "controls.pdf"
    with pymupdf.open() as pdf:
        page = _review_scan_page(pdf, pymupdf)
        page.insert_text((60, 100), "First hidden words then more", render_mode=3)
        pdf.save(path)
    original = pymupdf.Page.get_texttrace

    def with_control(self, *args, **kwargs):
        spans = original(self, *args, **kwargs)
        chars = list(spans[0]["chars"])
        # Emulate a corrupt decoded glyph at the real parser's trace boundary.
        chars[5] = (ord(control), *chars[5][1:])
        spans[0]["chars"] = tuple(chars)
        return spans

    monkeypatch.setattr(pymupdf.Page, "get_texttrace", with_control)
    doc = extract_pdf(path)
    assert "Plain text from the PDF's OCR layer" in doc.text
    assert not re.search(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", doc.text)
    assert doc.page_numbers == (0,)


def test_stamp_between_hidden_lines_is_preserved(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "stamp-between-lines.pdf"
    with pymupdf.open() as pdf:
        page = _review_scan_page(pdf, pymupdf)
        page.insert_text((60, 100), "seed", render_mode=3)
        pdf.update_stream(
            page.get_contents()[-1],
            b"BT /helv 12 Tf 3 Tr 1 0 0 1 60 742 Tm "
            b"(The first line of hidden recovered text) Tj 0 -100 Td "
            b"(The second line of hidden recovered text) Tj ET",
        )
        page.insert_text((60, 150), "VISIBLE STAMP")
        pdf.save(path)
    doc = extract_pdf(path)
    assert "The first line" in doc.text and "The second line" in doc.text
    assert "VISIBLE STAMP" in doc.text


def test_positioned_words_with_a_normal_space_are_separated(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "normal-space.pdf"
    with pymupdf.open() as pdf:
        page = _review_scan_page(pdf, pymupdf)
        page.insert_text((60, 100), "seed", render_mode=3)
        pdf.update_stream(
            page.get_contents()[-1],
            b"BT /helv 12 Tf 3 Tr 1 0 0 1 60 742 Tm (alpha) Tj 33 0 Td (bravo) Tj ET",
        )
        pdf.save(path)
    assert "alpha bravo" in extract_pdf(path).text


def test_undecodable_visible_remainder_keeps_ocr_route(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "unreadable-remainder.pdf"
    _review_chart_pdf(path, "glyph_scan", pymupdf)
    doc = extract_pdf(path, pages=[0])
    assert doc.needs_ocr == (0,)
    assert "Plain text from the PDF's OCR layer" not in doc.text
    assert "\ufffd" not in doc.text


@pytest.mark.parametrize("stamp_size", [10, 12])
def test_separate_stamp_on_an_ocr_baseline_is_preserved(tmp_path, stamp_size):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "same-baseline-stamp.pdf"
    with pymupdf.open() as pdf:
        page = _review_scan_page(pdf, pymupdf)
        page.insert_text((60, 100), "The first line of hidden recovered text", fontsize=12, render_mode=3)
        page.insert_text((60, 200), "The second line of hidden recovered text", fontsize=12, render_mode=3)
        page.insert_text((60, 100), "The first line of visible recopied text", fontsize=12)
        page.insert_text((400, 100), "VISIBLE STAMP", fontsize=stamp_size)
        pdf.save(path)
    doc = extract_pdf(path)
    assert "The first line of hidden recovered text" in doc.text
    assert doc.text.count("VISIBLE STAMP") == 1
