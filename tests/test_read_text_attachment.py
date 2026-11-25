"""`read` on a Markdown or plain-text attachment addressed by its own key: blocks of lines stand in for pages."""

import sqlite3
from types import SimpleNamespace

import pytest
from conftest import DummyContext

import zotero_mcp.local_db as local_db
from zotero_mcp.tools import read_pdf as read_pdf_mod
from zotero_mcp.tools.read_pdf import PdfReadError, find_in_pdf, format_find, read_pdf_text

N = read_pdf_mod._TEXT_BLOCK_LINES


def _attachment(content_type, **data):
    return {"data": {"itemType": "attachment", "contentType": content_type, **data}}


ITEMS = {
    "MD1": _attachment("text/markdown", filename="paper.md", title="Full Text Markdown"),
    "LINKED": _attachment("", path="C:/papers/paper.md", title="Linked Markdown"),
    "PDF1": _attachment("application/pdf", filename="paper.pdf", title="Full Text PDF"),
    "PARENT": {"data": {"itemType": "journalArticle", "title": "Paper"}},
}


class _Library:
    """Counts reader opens and item lookups; resolves every attachment to ``files[key]``."""

    def __init__(self, files):
        self.files, self.opens, self.lookups, self.error = files, 0, 0, None

    def get_item(self, key):
        self.lookups += 1
        return ITEMS.get(key)

    def reader(self, db_path=None):
        library = self

        class Reader:
            def __enter__(self):
                library.opens += 1
                if library.error:
                    raise library.error
                return self

            def __exit__(self, *exc):
                return False

            def resolve_attachment_file(self, key):
                return library.files.get(key)

            def get_attachment_by_key(self, key):
                if key in ITEMS and ITEMS[key]["data"]["itemType"] == "attachment":
                    return {"content_type": ITEMS[key]["data"]["contentType"], "title": ITEMS[key]["data"]["title"]}
                return None

        return Reader()


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A 120-line Markdown attachment MD1 (CRLF, as Windows writes it) and a PDF attachment PDF1, in local mode."""
    path = tmp_path / "paper.md"
    path.write_bytes(("\r\n".join(f"line {i}" + (" low dose ≤7.5 mg" if i == 75 else "")
                                  for i in range(1, 121)) + "\r\n").encode("utf-8"))  # final newline: no extra line
    fake = _Library({"MD1": path, "LINKED": path, "PDF1": tmp_path / "paper.pdf"})
    monkeypatch.setattr(read_pdf_mod._library, "get_library_backend", lambda: fake)
    monkeypatch.setattr(local_db, "LocalZoteroReader", fake.reader)
    monkeypatch.setattr(read_pdf_mod, "load_config",
                        lambda: SimpleNamespace(resolve_zotero_db_path=lambda: "zotero.sqlite"))
    monkeypatch.setattr(read_pdf_mod._utils, "is_local_mode", lambda: True)
    return fake


def test_blocks_of_lines(library):
    blocks, lines = read_pdf_mod._line_blocks(str(library.files["MD1"]))
    assert len(blocks) == 3 and lines == 120 and blocks[1].split("\n")[0] == f"line {N + 1}" and "\r" not in blocks[0]
    assert read_pdf_mod._block_lines(2, lines) == f"lines {N + 1}-{2 * N}"
    assert read_pdf_mod._block_lines(3, lines) == f"lines {2 * N + 1}-120"


def test_no_phantom_block_at_a_multiple_of_the_block_size(tmp_path):
    path = tmp_path / "f.md"
    path.write_text("".join(f"x{i}\n" for i in range(N)), encoding="utf-8")
    assert read_pdf_mod._line_blocks(str(path))[0] == ["\n".join(f"x{i}" for i in range(N))]


def test_empty_file_is_an_error(tmp_path):
    path = tmp_path / "e.md"
    path.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(PdfReadError) as err:
        read_pdf_mod._line_blocks(str(path))
    assert err.value.code == "empty_text_attachment"


def test_find_reports_line_ranges(library):
    result = find_in_pdf("MD1", "low dose ≤7.5", ctx=DummyContext())
    assert [m["page"] for m in result["matches"]] == [2] and result["total_pages"] == 3
    out = format_find(result)
    assert f"on 1 of 3 blocks of {N} lines" in out and f"lines {N + 1}-{2 * N} (1 hit)" in out
    assert "zotero-cli read MD1 --start-page 2" in out


def test_find_within_a_block_range(library):
    assert find_in_pdf("MD1", "low dose", 3, 3, ctx=DummyContext())["matches"] == []
    with pytest.raises(PdfReadError) as err:
        find_in_pdf("MD1", "low dose", 4, ctx=DummyContext())
    assert err.value.code == "page_out_of_range"


def test_read_blocks(library):
    out = read_pdf_text("MD1", 2, 9, ctx=DummyContext(), surface="cli")
    assert out.startswith(f"# Full Text Markdown, lines {N + 1}-120 of 120")
    assert f"## Block 3 (lines {2 * N + 1}-120)" in out
    assert f"## Block 2 (lines {N + 1}-{2 * N})" in out and "line 75 low dose ≤7.5 mg" in out
    assert "## Block 3" in out and "## Block 4" not in out


@pytest.mark.parametrize("start,end,code", [(4, None, "page_out_of_range"), (0, None, "page_out_of_range"),
                                            (3, 2, "invalid_page_range")])
def test_read_range_errors(library, start, end, code):
    with pytest.raises(PdfReadError) as err:
        read_pdf_text("MD1", start, end, ctx=DummyContext())
    assert err.value.code == code


def test_image_needs_a_pdf(library):
    with pytest.raises(PdfReadError) as err:
        read_pdf_mod.render_pdf_pages("MD1", 1, ctx=DummyContext())
    assert err.value.code == "not_a_pdf"


def test_a_pdf_read_costs_one_lookup_and_one_reader_open(library, monkeypatch):
    """A PDF key is resolved exactly as before the change: the text check adds no database open."""
    monkeypatch.setattr(read_pdf_mod, "extract_pdf", lambda path: SimpleNamespace(
        page_numbers=[0], pages=["low dose"], page_count=1))
    result = find_in_pdf("PDF1", "low dose", ctx=DummyContext())
    assert result["title"] == "Full Text PDF" and "block_lines" not in result
    assert (library.lookups, library.opens) == (1, 1)


def test_a_text_read_looks_the_item_up_once(library):
    read_pdf_text("MD1", 1, ctx=DummyContext())
    assert (library.lookups, library.opens) == (1, 1)


@pytest.mark.parametrize("key", ["PDF1", "PARENT", "NONE"])
def test_only_a_text_attachment_opens_the_reader(library, key):
    assert read_pdf_mod._text_attachment(key, ITEMS.get(key)) is None and library.opens == 0


def test_a_linked_markdown_file_is_recognised(library):
    assert read_pdf_mod._text_attachment("LINKED", ITEMS["LINKED"]) == (str(library.files["LINKED"]), "Linked Markdown")


def test_not_in_local_mode_nothing_changes(library, monkeypatch):
    monkeypatch.setattr(read_pdf_mod._utils, "is_local_mode", lambda: False)
    assert read_pdf_mod._text_attachment("MD1", ITEMS["MD1"]) is None and library.opens == 0


def test_a_missing_file_says_so(library):
    del library.files["MD1"]
    with pytest.raises(PdfReadError) as err:
        find_in_pdf("MD1", "x", ctx=DummyContext())
    assert err.value.code == "attachment_file_missing"


def test_a_database_error_surfaces(library):
    library.error = sqlite3.OperationalError("database is locked")
    with pytest.raises(sqlite3.OperationalError):
        find_in_pdf("MD1", "x", ctx=DummyContext())
