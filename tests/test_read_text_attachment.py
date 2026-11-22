"""`read` on a Markdown or plain-text attachment addressed by its own key: blocks of lines stand in for pages."""

from types import SimpleNamespace

import pytest
from conftest import DummyContext

from zotero_mcp.tools import read_pdf as read_pdf_mod
from zotero_mcp.tools.read_pdf import PdfReadError, find_in_pdf, format_find, read_pdf_text

N = read_pdf_mod._TEXT_BLOCK_LINES


@pytest.fixture
def text_file(tmp_path, monkeypatch):
    """A 120-line Markdown attachment (CRLF, as Windows writes it) that the key MD1 resolves to."""
    path = tmp_path / "paper.md"
    path.write_bytes(("\r\n".join(f"line {i}" + (" low dose ≤7.5 mg" if i == 75 else "")
                                  for i in range(1, 121)) + "\r\n").encode("utf-8"))  # final newline: no extra line
    monkeypatch.setattr(read_pdf_mod, "_text_attachment",
                        lambda key: (str(path), "Full Text Markdown") if key == "MD1" else None)
    return path


def test_blocks_of_lines(text_file):
    blocks, lines = read_pdf_mod._line_blocks(str(text_file))
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


def test_find_reports_line_ranges(text_file):
    result = find_in_pdf("MD1", "low dose ≤7.5", ctx=DummyContext())
    assert [m["page"] for m in result["matches"]] == [2] and result["total_pages"] == 3
    out = format_find(result)
    assert f"on 1 of 3 blocks of {N} lines" in out and f"lines {N + 1}-{2 * N} (1 hit)" in out
    assert "zotero-cli read MD1 --start-page 2" in out


def test_find_within_a_block_range(text_file):
    assert find_in_pdf("MD1", "low dose", 3, 3, ctx=DummyContext())["matches"] == []
    with pytest.raises(PdfReadError) as err:
        find_in_pdf("MD1", "low dose", 4, ctx=DummyContext())
    assert err.value.code == "page_out_of_range"


def test_read_blocks(text_file):
    out = read_pdf_text("MD1", 2, 9, ctx=DummyContext(), surface="cli")
    assert out.startswith(f"# Full Text Markdown, lines {N + 1}-120 of 120")
    assert f"## Block 3 (lines {2 * N + 1}-120)" in out
    assert f"## Block 2 (lines {N + 1}-{2 * N})" in out and "line 75 low dose ≤7.5 mg" in out
    assert "## Block 3" in out and "## Block 4" not in out


@pytest.mark.parametrize("start,end,code", [(4, None, "page_out_of_range"), (0, None, "page_out_of_range"),
                                            (3, 2, "invalid_page_range")])
def test_read_range_errors(text_file, start, end, code):
    with pytest.raises(PdfReadError) as err:
        read_pdf_text("MD1", start, end, ctx=DummyContext())
    assert err.value.code == code


def test_image_needs_a_pdf(text_file):
    with pytest.raises(PdfReadError) as err:
        read_pdf_mod.render_pdf_pages("MD1", 1, ctx=DummyContext())
    assert err.value.code == "not_a_pdf"


def test_parent_key_still_means_the_pdf(text_file, monkeypatch):
    monkeypatch.setattr(read_pdf_mod, "_get_pdf_path", lambda key, ctx: None)
    with pytest.raises(PdfReadError) as err:
        find_in_pdf("PARENT", "x", ctx=DummyContext())
    assert err.value.code == "no_pdf_attachment"


class _Reader:
    rows = {"MD1": {"content_type": "text/markdown", "zotero_path": "storage:paper.md", "title": "Full Text Markdown"},
            "PDF1": {"content_type": "application/pdf", "zotero_path": "storage:paper.pdf", "title": "Full Text PDF"}}

    def __init__(self, db_path=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_attachment_by_key(self, key):
        return self.rows.get(key)

    def resolve_attachment_file(self, key):
        return f"/storage/{key}/file"


def test_text_attachment_lookup(monkeypatch):
    import zotero_mcp.local_db as local_db

    monkeypatch.setattr(local_db, "LocalZoteroReader", _Reader)
    monkeypatch.setattr(read_pdf_mod, "load_config",
                        lambda: SimpleNamespace(resolve_zotero_db_path=lambda: "zotero.sqlite"))
    monkeypatch.setattr(read_pdf_mod._utils, "is_local_mode", lambda: True)
    assert read_pdf_mod._text_attachment("MD1") == ("/storage/MD1/file", "Full Text Markdown")
    assert read_pdf_mod._text_attachment("PDF1") is None
    assert read_pdf_mod._text_attachment("NONE") is None
    monkeypatch.setattr(read_pdf_mod._utils, "is_local_mode", lambda: False)
    assert read_pdf_mod._text_attachment("MD1") is None
