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


class TestExtractTextFile:
    def test_reads_content_verbatim(self, tmp_path):
        note = tmp_path / "notes.md"
        note.write_text("# Heading\n\nsome body")
        doc = extract_text_file(note)
        assert doc.text == "# Heading\n\nsome body"
        assert doc.source == "text"
        assert doc.page_count == 1


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
