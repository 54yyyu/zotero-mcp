"""PDFs with a damaged cross-reference table are read through PyMuPDF's repaired copy (#563).

pdf-inspector reads the table as written and raises "Invalid PDF structure"
when it is wrong; PyMuPDF rebuilds it. The files here are real PDFs broken
the way producers break them.
"""

import re
import sys

import pytest

from zotero_mcp import extract
from zotero_mcp.extract import extract_pdf, pdf_page_count

pymupdf = pytest.importorskip("pymupdf")


def _good_pdf() -> bytes:
    with pymupdf.open() as pdf:
        for number in range(1, 4):
            page = pdf.new_page()
            page.insert_text((72, 100), f"Body text of page {number} for the damaged file.", fontsize=11)
        return pdf.tobytes()


def _wrong_startxref(data: bytes) -> bytes:
    return re.sub(rb"startxref\s+\d+", b"startxref\n12345", data)


def _no_xref(data: bytes) -> bytes:
    return data[:data.rfind(b"xref")]


def _junk_before_header(data: bytes) -> bytes:
    return b"\x00" * 2000 + data


@pytest.fixture(autouse=True)
def _fresh_memos():
    extract._parse_memo.clear()
    extract._repair_memo.clear()
    yield
    extract._parse_memo.clear()
    extract._repair_memo.clear()


@pytest.mark.parametrize("damage", [_wrong_startxref, _no_xref, _junk_before_header])
class TestDamagedFiles:
    def _path(self, tmp_path, damage):
        import pdf_inspector

        path = tmp_path / "damaged.pdf"
        path.write_bytes(damage(_good_pdf()))
        with pytest.raises(ValueError):          # the premise: pdf-inspector alone refuses it
            pdf_inspector.classify_pdf(str(path))
        return path

    def test_page_count(self, tmp_path, damage):
        assert pdf_page_count(self._path(tmp_path, damage)) == 3

    def test_every_page_keeps_its_own_text(self, tmp_path, damage):
        doc = extract_pdf(self._path(tmp_path, damage))
        assert doc.page_count == 3 and doc.page_numbers == (0, 1, 2)
        for number, text in enumerate(doc.pages, start=1):
            assert f"page {number} for the damaged file" in text

    def test_a_page_range(self, tmp_path, damage):
        doc = extract_pdf(self._path(tmp_path, damage), pages=[1], reuse=True)
        assert doc.page_numbers == (1,) and "page 2 for the damaged file" in doc.text

    def test_without_pymupdf_the_original_error_stands(self, tmp_path, damage, monkeypatch):
        path = self._path(tmp_path, damage)
        monkeypatch.setitem(sys.modules, "pymupdf", None)
        monkeypatch.setitem(sys.modules, "fitz", None)
        with pytest.raises(ValueError):
            pdf_page_count(path)


def test_the_file_is_re_saved_once_for_a_read(tmp_path, monkeypatch):
    path = tmp_path / "damaged.pdf"
    path.write_bytes(_wrong_startxref(_good_pdf()))
    saves = []
    original = pymupdf.Document.tobytes
    monkeypatch.setattr(pymupdf.Document, "tobytes",
                        lambda self, *a, **k: (saves.append(1), original(self, *a, **k))[1])
    assert pdf_page_count(path) == 3
    assert extract_pdf(path, pages=[0]).page_numbers == (0,)
    assert len(saves) == 1


@pytest.mark.parametrize("content", [b"", b"this is not a PDF at all", b"<html><body>a page</body></html>"])
def test_a_file_that_is_no_pdf_still_fails(tmp_path, content):
    path = tmp_path / "not.pdf"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        pdf_page_count(path)
    with pytest.raises(ValueError):
        extract_pdf(path)


def test_a_sound_pdf_is_never_re_saved(tmp_path, monkeypatch):
    path = tmp_path / "good.pdf"
    path.write_bytes(_good_pdf())
    monkeypatch.setattr(extract, "_repaired_pdf", lambda _path: pytest.fail("no repair for a sound file"))
    assert pdf_page_count(path) == 3
    assert extract_pdf(path).page_count == 3
