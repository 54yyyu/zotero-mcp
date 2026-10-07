"""Tests for the extraction seam (``zotero_mcp.extract``).

Everything that turns an attachment into text goes through this module, so
the invariants downstream code relies on are pinned here: the page separator
that chunk provenance counts, the page cap the indexer depends on, and the
tolerant-vs-raising split between ``extract_file`` and the per-format
functions.
"""

from pathlib import Path

import pytest

from zotero_mcp.extract import (
    PAGE_SEPARATOR,
    ExtractedDoc,
    extract_file,
    extract_html,
    extract_pdf,
    extract_text_file,
    is_extractable,
    pdf_page_count,
)


def _write_pdf(path: Path, pages: list[str]) -> Path:
    """Write a minimal multi-page PDF with one text string per page."""
    fitz = pytest.importorskip("fitz", reason="PyMuPDF builds the fixture PDFs")
    doc = fitz.open()
    for body in pages:
        page = doc.new_page()
        page.insert_text((72, 144), body, fontsize=24)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def three_page_pdf(tmp_path):
    return _write_pdf(
        tmp_path / "sample.pdf",
        ["Alpha page one", "Bravo page two", "Charlie page three"],
    )


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


class TestExtractPdf:
    def test_extracts_every_page_by_default(self, three_page_pdf):
        doc = extract_pdf(three_page_pdf)
        assert doc.page_count == 3
        assert len(doc.pages) == 3
        assert doc.source == "pdf"
        assert not doc.truncated
        assert "Alpha" in doc.pages[0]
        assert "Charlie" in doc.pages[2]

    def test_text_joins_pages_with_the_page_separator(self, three_page_pdf):
        """Chunk provenance counts these separators to recover a page number,
        so the count must be exactly one fewer than the page count."""
        doc = extract_pdf(three_page_pdf)
        assert doc.text.count(PAGE_SEPARATOR) == doc.page_count - 1
        assert doc.text == PAGE_SEPARATOR.join(doc.pages)

    def test_max_pages_truncates_from_the_tail(self, three_page_pdf):
        doc = extract_pdf(three_page_pdf, max_pages=2)
        assert len(doc.pages) == 2
        assert doc.truncated
        # page_count stays the size of the document, not of the excerpt.
        assert doc.page_count == 3
        assert "Charlie" not in doc.text

    def test_max_pages_beyond_the_document_is_not_truncation(self, three_page_pdf):
        doc = extract_pdf(three_page_pdf, max_pages=99)
        assert len(doc.pages) == 3
        assert not doc.truncated

    def test_non_positive_max_pages_means_no_limit(self, three_page_pdf):
        assert len(extract_pdf(three_page_pdf, max_pages=0).pages) == 3

    def test_explicit_pages_are_returned_in_order(self, three_page_pdf):
        doc = extract_pdf(three_page_pdf, pages=[2, 0])
        assert doc.page_numbers == (2, 0)
        assert "Charlie" in doc.pages[0]
        assert "Alpha" in doc.pages[1]

    def test_out_of_range_pages_are_dropped_not_padded(self, three_page_pdf):
        """pdf-inspector returns a blank page for an out-of-range index rather
        than erroring; silently keeping it would desynchronize page numbering."""
        doc = extract_pdf(three_page_pdf, pages=[0, 99])
        assert doc.page_numbers == (0,)
        assert doc.page_count == 3

    def test_all_pages_out_of_range_yields_an_empty_doc(self, three_page_pdf):
        doc = extract_pdf(three_page_pdf, pages=[50, 99])
        assert doc.pages == ()
        assert not doc

    def test_pages_and_max_pages_together_is_a_type_error(self, three_page_pdf):
        with pytest.raises(TypeError):
            extract_pdf(three_page_pdf, pages=[0], max_pages=1)

    def test_born_digital_pdf_needs_no_ocr(self, three_page_pdf):
        assert extract_pdf(three_page_pdf).needs_ocr == ()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError):
            extract_pdf(tmp_path / "nope.pdf")

    def test_non_pdf_raises(self, tmp_path):
        decoy = tmp_path / "notreally.pdf"
        decoy.write_text("this is plain text, not a PDF")
        with pytest.raises(ValueError):
            extract_pdf(decoy)


class TestPdfPageCount:
    def test_counts_pages(self, three_page_pdf):
        assert pdf_page_count(three_page_pdf) == 3

    def test_raises_on_a_missing_file(self, tmp_path):
        with pytest.raises(ValueError):
            pdf_page_count(tmp_path / "absent.pdf")


# ---------------------------------------------------------------------------
# HTML and plain text
# ---------------------------------------------------------------------------


class TestExtractHtml:
    def test_converts_structure_to_markdown(self, tmp_path):
        snapshot = tmp_path / "page.html"
        snapshot.write_text("<h1>Title</h1><p>Body <em>text</em>.</p>")
        doc = extract_html(snapshot)
        assert doc.source == "html"
        assert doc.page_count == 1
        assert "# Title" in doc.text
        assert "Body" in doc.text

    def test_undecodable_bytes_do_not_raise(self, tmp_path):
        snapshot = tmp_path / "latin.html"
        snapshot.write_bytes(b"<p>caf\xe9</p>")
        assert "caf" in extract_html(snapshot).text

    def test_embedded_data_images_are_dropped(self, tmp_path):
        # Connector snapshots inline images as data: URIs; copying them into
        # the Markdown turned a 33K-character article into 6.1M characters.
        payload = "iVBORw0KGgo" + "A" * 200_000
        snapshot = tmp_path / "page.html"
        snapshot.write_text(
            f'<p>Before</p><img alt="Logo" src="data:image/png;base64,{payload}">'
            '<p>After</p>'
        )
        text = extract_html(snapshot).text
        assert "Before" in text and "After" in text
        assert "base64" not in text
        assert len(text) < 200

    def test_svg_data_uri_with_parentheses_is_dropped(self, tmp_path):
        # A utf8 SVG URI can contain ")", which would end a Markdown image
        # early; dropping at the element level avoids parsing it at all.
        snapshot = tmp_path / "page.html"
        snapshot.write_text(
            "<p>Text</p><img src=\"data:image/svg+xml;utf8,<svg>"
            "<path d='M0 0 (1)'/></svg>\">"
        )
        text = extract_html(snapshot).text
        assert "Text" in text
        assert "svg" not in text

    def test_data_images_inside_links_leave_the_link_text(self, tmp_path):
        snapshot = tmp_path / "page.html"
        snapshot.write_text(
            '<a href="https://example.org/">'
            '<img alt="Home" src="data:image/png;base64,AAAA">Home page</a>'
        )
        text = extract_html(snapshot).text
        assert "https://example.org/" in text
        assert "Home page" in text
        assert "AAAA" not in text

    def test_data_images_leave_a_marker_with_alt_text(self, tmp_path):
        snapshot = tmp_path / "page.html"
        snapshot.write_text(
            '<img alt="  Figure 2:\n adoption [2024] " src="data:image/png;base64,AAAA">'
            '<img src="data:image/png;base64,BBBB">'
            '<img alt="   " src="data:image/gif;base64,CCCC">'
        )
        text = extract_html(snapshot).text
        assert "[image: Figure 2: adoption 2024]" in text
        assert text.count("[image]") == 2

    def test_data_video_posters_and_sources_are_dropped(self, tmp_path):
        # markdownify renders <video> as an image of its poster, so a data:
        # poster leaked base64 the same way an <img> did.
        snapshot = tmp_path / "page.html"
        snapshot.write_text(
            '<video poster="data:image/png;base64,PPPP">No video support.</video>'
            '<video poster="data:image/png;base64,QQQQ" src="https://example.org/a.mp4">'
            "Clip</video>"
            '<video><source src="data:video/mp4;base64,SSSS">'
            '<source src="https://example.org/b.mp4">Fallback</video>'
        )
        text = extract_html(snapshot).text
        for payload in ("PPPP", "QQQQ", "SSSS", "data:"):
            assert payload not in text
        assert "No video support." in text
        assert "https://example.org/a.mp4" in text
        assert "https://example.org/b.mp4" in text

    def test_data_links_keep_only_their_text(self, tmp_path):
        snapshot = tmp_path / "page.html"
        snapshot.write_text(
            '<p><a href="data:application/pdf;base64,LLLL">Download</a> and '
            '<a href="https://example.org/">site</a></p>'
        )
        text = extract_html(snapshot).text
        assert "LLLL" not in text and "data:" not in text
        assert "Download" in text
        assert "[site](https://example.org/)" in text

    def test_remote_images_are_kept(self, tmp_path):
        snapshot = tmp_path / "page.html"
        snapshot.write_text('<img alt="Figure 1" src="https://example.org/f1.png">')
        assert "![Figure 1](https://example.org/f1.png)" in extract_html(snapshot).text


class TestExtractTextFile:
    def test_reads_content_verbatim(self, tmp_path):
        note = tmp_path / "notes.md"
        # write_bytes, not write_text: the latter applies the platform's
        # newline translation, so this would assert against \r\n on Windows
        # and stop testing what it means to.
        note.write_bytes(b"# Heading\n\nsome body")
        doc = extract_text_file(note)
        assert doc.text == "# Heading\n\nsome body"
        assert doc.source == "text"
        assert doc.page_count == 1

    @pytest.mark.parametrize("raw", [b"a\r\nb", b"a\rb"])
    def test_newlines_are_normalized(self, raw, tmp_path):
        """A CRLF attachment must not put \\r\\n into embeddings when the
        same document read as a PDF would yield \\n."""
        note = tmp_path / "crlf.txt"
        note.write_bytes(raw)
        assert extract_text_file(note).text == "a\nb"


# ---------------------------------------------------------------------------
# Dispatch and the extractability gate
# ---------------------------------------------------------------------------


class TestExtractFile:
    def test_dispatches_on_extension(self, tmp_path, three_page_pdf):
        html = tmp_path / "s.html"
        html.write_text("<p>hi</p>")
        txt = tmp_path / "s.txt"
        txt.write_text("hi")

        assert extract_file(three_page_pdf).source == "pdf"
        assert extract_file(html).source == "html"
        assert extract_file(txt).source == "text"

    def test_forwards_max_pages_to_the_pdf_path(self, three_page_pdf):
        assert len(extract_file(three_page_pdf, max_pages=1).pages) == 1

    def test_returns_none_instead_of_raising(self, tmp_path):
        """Callers walk whole libraries; one unreadable attachment must not
        abort the batch."""
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"not a pdf at all")
        assert extract_file(broken) is None

    def test_logs_when_extraction_fails(self, tmp_path, caplog):
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"")
        with caplog.at_level("WARNING", logger="zotero_mcp.extract"):
            assert extract_file(broken) is None
        assert "broken.pdf" in caplog.text


class TestIsExtractable:
    @pytest.mark.parametrize(
        "name,ctype",
        [
            ("a.txt", "text/plain"),
            ("a.vtt", "text/vtt"),
            ("captions.srt", None),
            ("a.txt", "text/x-asm"),
            ("notes.md", None),
        ],
    )
    def test_accepts_textual(self, name, ctype):
        assert is_extractable(Path(name), ctype)

    @pytest.mark.parametrize(
        "name,ctype",
        [
            ("paper.docx", "application/vnd.openxmlformats-officedocument"
                           ".wordprocessingml.document"),
            ("talk.mp4", "video/mp4"),
            ("a.pdf", "application/pdf"),
        ],
    )
    def test_rejects_binary_and_pdf(self, name, ctype):
        assert not is_extractable(Path(name), ctype)


class TestExtractedDoc:
    def test_is_falsy_when_it_holds_only_whitespace(self):
        blank = ExtractedDoc(
            text="   \n ", pages=("   \n ",), page_numbers=(0,),
            page_count=1, source="pdf",
        )
        assert not blank

    def test_is_truthy_with_content(self):
        filled = ExtractedDoc(
            text="body", pages=("body",), page_numbers=(0,),
            page_count=1, source="pdf",
        )
        assert filled


# ---------------------------------------------------------------------------
# Text-layer fallback for scanner-OCR PDFs (#611)
# ---------------------------------------------------------------------------


class _FakePage:
    def __init__(self, page, markdown, needs_ocr=False):
        self.page = page
        self.markdown = markdown
        self.needs_ocr = needs_ocr


class _FakeInspector:
    """Stand-in for ``pdf_inspector`` with scriptable markdown output.

    ``markdown`` maps a 0-indexed page to its markdown; a page mapped to
    ``""`` is reported with ``needs_ocr=True``, the way pdf-inspector
    reports a page whose only text is an invisible scanner-OCR layer.
    """

    def __init__(self, markdown, text="", raise_markdown=None):
        self.markdown = markdown
        self.text = text
        self.raise_markdown = raise_markdown
        self.text_calls = 0

    def classify_pdf(self, path):
        class _C:
            page_count = len(self.markdown)
        return _C()

    def extract_pages_markdown(self, path, pages=None):
        if self.raise_markdown is not None:
            raise self.raise_markdown
        wanted = range(len(self.markdown)) if pages is None else pages

        class _R:
            pass
        result = _R()
        result.pages = [
            _FakePage(p, self.markdown[p], needs_ocr=not self.markdown[p])
            for p in wanted
        ]
        return result

    def extract_text(self, path):
        self.text_calls += 1
        return self.text


@pytest.fixture
def fake_inspector(monkeypatch):
    def install(**kwargs):
        fake = _FakeInspector(**kwargs)
        monkeypatch.setattr("zotero_mcp.extract._pdf_inspector", lambda: fake)
        return fake
    return install


class TestOcrLayerFallback:
    def test_all_pages_flagged_needs_ocr_fall_back_to_the_text_layer(
        self, fake_inspector
    ):
        fake_inspector(markdown=["", "", ""], text="Scanned decision text\n")
        doc = extract_pdf("scan.pdf")
        assert "Scanned decision text" in doc.text
        assert doc.page_count == 3
        assert doc.needs_ocr == ()
        assert doc

    def test_markdown_parse_error_falls_back_to_the_text_layer(
        self, fake_inspector
    ):
        fake_inspector(
            markdown=["", ""],
            text="Ghostscript output\n",
            raise_markdown=ValueError("invalid content stream"),
        )
        doc = extract_pdf("gs.pdf")
        assert "Ghostscript output" in doc.text
        assert doc.page_count == 2

    def test_parse_error_with_no_text_layer_still_raises(self, fake_inspector):
        fake_inspector(
            markdown=["x"], text="", raise_markdown=ValueError("broken"),
        )
        with pytest.raises(ValueError, match="broken"):
            extract_pdf("broken.pdf")

    def test_normal_markdown_does_not_call_the_fallback(self, fake_inspector):
        fake = fake_inspector(markdown=["# One", "Two"], text="unused")
        doc = extract_pdf("ok.pdf")
        assert doc.pages == ("# One", "Two")
        assert fake.text_calls == 0

    def test_one_text_page_among_empty_ones_does_not_fall_back(
        self, fake_inspector
    ):
        # A genuinely mixed document (one born-digital page, the rest
        # image-only) keeps its per-page markdown and OCR routing.
        fake = fake_inspector(markdown=["Cover", "", ""], text="unused")
        doc = extract_pdf("mixed.pdf")
        assert fake.text_calls == 0
        assert doc.needs_ocr == (1, 2)

    def test_max_pages_keeps_page_numbering_and_truncation(
        self, fake_inspector
    ):
        fake_inspector(markdown=["", "", "", ""], text="a" * 400)
        doc = extract_pdf("scan.pdf", max_pages=2)
        assert doc.page_numbers == (0,)
        assert doc.page_count == 4
        assert doc.truncated
        # The text layer cannot be split by page, so the cap is applied
        # proportionally rather than indexing the whole document.
        assert 0 < len(doc.text) <= 200

    def test_partial_page_selection_does_not_return_other_pages_text(
        self, fake_inspector
    ):
        # The whole-document text layer cannot be attributed to page 2
        # alone, so an explicit subset keeps the old (empty) result.
        fake_inspector(markdown=["", "", ""], text="whole document")
        doc = extract_pdf("scan.pdf", pages=[2])
        assert doc.page_numbers == (2,)
        assert "whole document" not in doc.text
        assert doc.needs_ocr == (2,)

    def test_selection_covering_every_page_falls_back(self, fake_inspector):
        fake_inspector(markdown=["", ""], text="all of it")
        doc = extract_pdf("scan.pdf", pages=[0, 1])
        assert "all of it" in doc.text


class _CountingInspector(_FakeInspector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.markdown_calls = []

    def extract_pages_markdown(self, path, pages=None):
        self.markdown_calls.append(pages)
        return super().extract_pages_markdown(path, pages=pages)


@pytest.fixture
def counting_inspector(monkeypatch):
    from zotero_mcp import extract

    extract._parse_memo.clear()
    fake = _CountingInspector(markdown=["p0", "p1", "p2", "p3"])
    monkeypatch.setattr("zotero_mcp.extract._pdf_inspector", lambda: fake)
    yield fake
    extract._parse_memo.clear()


class TestParseReuse:
    """pdf-inspector's markdown pass costs the same for one page as for the
    whole file (3.7-6 s on some 28-page papers), so reading a paper in chunks
    must not re-run it for every range."""

    def test_page_ranges_of_one_file_parse_it_once(self, counting_inspector, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 one")
        first = extract_pdf(pdf, pages=[0, 1], reuse=True)
        second = extract_pdf(pdf, pages=[2, 3], reuse=True)
        head = extract_pdf(pdf, max_pages=3, reuse=True)
        assert len(counting_inspector.markdown_calls) == 1
        assert first.pages == ("p0", "p1") and first.page_numbers == (0, 1)
        assert second.pages == ("p2", "p3") and second.page_numbers == (2, 3)
        assert head.pages == ("p0", "p1", "p2") and head.truncated
        assert head.page_count == 4

    def test_without_reuse_every_call_parses(self, counting_inspector, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 one")
        extract_pdf(pdf, pages=[0])
        extract_pdf(pdf, pages=[1])
        assert counting_inspector.markdown_calls == [[0], [1]]

    def test_a_changed_file_is_parsed_again(self, counting_inspector, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 one")
        extract_pdf(pdf, pages=[0], reuse=True)
        pdf.write_bytes(b"%PDF-1.4 one, edited")
        extract_pdf(pdf, pages=[0], reuse=True)
        assert len(counting_inspector.markdown_calls) == 2

    def test_out_of_range_pages_are_dropped_like_the_uncached_path(
        self, counting_inspector, tmp_path
    ):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 one")
        doc = extract_pdf(pdf, pages=[3, 9], reuse=True)
        assert doc.page_numbers == (3,)
        assert not extract_pdf(pdf, pages=[9], reuse=True).pages

    def test_memo_is_bounded(self, counting_inspector, tmp_path, monkeypatch):
        from zotero_mcp import extract

        monkeypatch.setattr(extract, "_MEMO_MAX_ENTRIES", 2)
        for i in range(4):
            pdf = tmp_path / f"p{i}.pdf"
            pdf.write_bytes(b"%PDF-1.4 " + bytes([65 + i]))
            extract_pdf(pdf, pages=[0], reuse=True)
        assert len(extract._parse_memo) == 2

    def test_memo_skips_documents_over_the_character_budget(
        self, counting_inspector, tmp_path, monkeypatch
    ):
        from zotero_mcp import extract

        monkeypatch.setattr(extract, "_MEMO_MAX_CHARS", 3)
        pdf = tmp_path / "big.pdf"
        pdf.write_bytes(b"%PDF-1.4 big")
        extract_pdf(pdf, pages=[0], reuse=True)
        assert not extract._parse_memo

    def test_ocr_flags_stay_absolute_in_a_reused_slice(self, monkeypatch, tmp_path):
        from zotero_mcp import extract

        extract._parse_memo.clear()
        fake = _CountingInspector(markdown=["a", "", "c"])
        monkeypatch.setattr("zotero_mcp.extract._pdf_inspector", lambda: fake)
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF-1.4 s")
        doc = extract_pdf(pdf, pages=[1, 2], reuse=True)
        assert doc.page_numbers == (1, 2) and doc.needs_ocr == (1,)
        extract._parse_memo.clear()

    def test_page_count_reads_the_memo(self, counting_inspector, tmp_path, monkeypatch):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 one")
        extract_pdf(pdf, pages=[0], reuse=True)
        monkeypatch.setattr(counting_inspector, "classify_pdf", lambda p: 1 / 0)
        assert pdf_page_count(pdf) == 4

    def test_scanner_text_layer_is_not_reparsed_whole_on_every_read(
        self, monkeypatch, tmp_path
    ):
        """A PDF whose whole-document parse falls back to the text layer cannot
        be sliced by page; later reads must not repeat that whole parse."""
        from zotero_mcp import extract

        extract._parse_memo.clear()
        fake = _CountingInspector(markdown=["", "", ""], text="scanned text")
        monkeypatch.setattr("zotero_mcp.extract._pdf_inspector", lambda: fake)
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF-1.4 s")
        extract_pdf(pdf, pages=[0], reuse=True)
        assert fake.markdown_calls == [None, [0]]  # whole parse, then the subset
        extract_pdf(pdf, pages=[1], reuse=True)
        assert fake.markdown_calls == [None, [0], [1]]
        extract._parse_memo.clear()


class TestReaderOptsIntoReuse:
    def test_reader_passes_reuse_only_when_asked(self, monkeypatch):
        from zotero_mcp import local_db

        seen = []
        monkeypatch.setattr(
            local_db, "extract_file", lambda p, **kw: seen.append(kw) or None
        )
        indexer = local_db.LocalZoteroReader(db_path="x.sqlite")
        interactive = local_db.LocalZoteroReader(db_path="x.sqlite", reuse_pdf_parse=True)
        indexer._extract_doc_from_file(Path("a.pdf"))
        interactive._extract_doc_from_file(Path("a.pdf"))
        assert [kw["reuse"] for kw in seen] == [False, True]
