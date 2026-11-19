"""EPUB highlights: the CFI points at the quote, in the chapter asked for.

Each CFI is resolved here the way Zotero's reader does it (epub.js): an even
step picks an element child, an odd step picks one of the parent's text
nodes counted one by one, and the offset is a position in that node's raw
characters. A test passes when the resolved range reads as the quote.
"""

import json
import re
import zipfile
from unittest.mock import MagicMock
from xml.dom import minidom

import pytest

from zotero_mcp import epub_utils
from zotero_mcp.epub_utils import find_text_in_epub
from zotero_mcp.tools import annotations

_CONTAINER = (
    '<?xml version="1.0"?><container version="1.0" '
    'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
    '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
    '</rootfiles></container>'
)


def _chapter(body: str) -> str:
    return ('<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml">'
            f'<head><title>t</title><meta charset="utf-8"/></head><body>{body}</body></html>')


def make_epub(path, bodies) -> str:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", _CONTAINER)
        manifest = "".join(f'<item id="c{i}" href="c{i}.xhtml" media-type="application/xhtml+xml"/>'
                           for i in range(len(bodies)))
        spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(bodies)))
        archive.writestr("OEBPS/content.opf",
                         '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                         f'<manifest>{manifest}</manifest><spine>{spine}</spine></package>')
        for i, body in enumerate(bodies):
            archive.writestr(f"OEBPS/c{i}.xhtml", _chapter(body))
    return str(path)


def _resolve(document, steps: str):
    """(text node, offset) for a CFI path such as ``/4/2/3:11``, as epub.js walks it."""
    path, offset = steps.split(":")
    node = document.documentElement
    for number in (int(n) for n in path.strip("/").split("/")):
        if number % 2 == 0:
            node = [c for c in node.childNodes if c.nodeType == c.ELEMENT_NODE][number // 2 - 1]
        else:
            node = [c for c in node.childNodes if c.nodeType == c.TEXT_NODE][(number - 1) // 2]
    assert node.nodeType == node.TEXT_NODE
    return node, int(offset)


def _text_nodes(node):
    for child in node.childNodes:
        if child.nodeType == child.TEXT_NODE:
            yield child
        else:
            yield from _text_nodes(child)


def resolved_text(body: str, cfi: str) -> str:
    """The text a CFI range selects in a chapter body."""
    match = re.fullmatch(r"epubcfi\(/6/\d+!([^,]*),([^,]+),([^,]+)\)", cfi)
    assert match, cfi
    common, start, end = match.groups()
    document = minidom.parseString(_chapter(body).replace("&nbsp;", " "))
    start_node, start_offset = _resolve(document, common + start)
    end_node, end_offset = _resolve(document, common + end)
    assert start_offset <= len(start_node.data) and end_offset <= len(end_node.data)
    if start_node is end_node:
        return start_node.data[start_offset:end_offset]
    nodes = list(_text_nodes(document.documentElement))
    first = next(i for i, n in enumerate(nodes) if n is start_node)
    last = next(i for i, n in enumerate(nodes) if n is end_node)
    middle = "".join(n.data for n in nodes[first + 1:last])
    return start_node.data[start_offset:] + middle + end_node.data[:end_offset]


@pytest.mark.parametrize("body, quote", [
    # one text node, no surprises
    ("<p>Hello brave new world again</p>", "new world"),
    # the node starts with a space: offsets used to be one short (" world agai")
    ("<p>Hello <b>bold</b> world again</p>", "world again"),
    # a whitespace-only node between two inline elements used to go uncounted,
    # so the CFI named that one-space node
    ("<p>One <i>two</i> <b>three</b> four five</p>", "four five"),
    # indented source: the node starts with a line break and spaces
    ("<p>\n    Hello brave new world\n  </p>", "brave new"),
    # pretty-printed: a whitespace-only node comes before the first element
    ("<p>\n  <span>A</span> more text here.\n</p>", "more text"),
    # entities are single characters in the DOM
    ("<p>Tom &amp; Jerry &lt;3 cheese&nbsp;now</p>", "cheese"),
    # a quote that starts in one element and ends in the text after it
    ("<p>Hello <b>bold</b> world again</p>", "bold world"),
    # runs of spaces inside the node
    ("<p>one   two \n three   four</p>", "three four"),
])
def test_cfi_resolves_to_the_quote(tmp_path, body, quote):
    path = make_epub(tmp_path / "book.epub", ["<p>front matter</p>", body])
    found = find_text_in_epub(path, 2, quote)
    assert "error" not in found, found
    assert found["chapter_found"] == 2
    assert " ".join(resolved_text(body, found["cfi"]).split()) == quote


class TestChapter:
    BODIES = ["<p>Contents: The Long Night</p>", "<p>filler</p>",
              "<h1>The Long Night</h1><p>It was dark.</p>", "<p>The Long Night again</p>"]

    def test_the_chapter_asked_for_is_searched_first(self, tmp_path):
        path = make_epub(tmp_path / "book.epub", self.BODIES)
        found = find_text_in_epub(path, 3, "The Long Night")
        assert found["chapter_found"] == 3 and found["spine_index"] == 2
        assert found["cfi"].startswith("epubcfi(/6/6!")

    def test_a_wrong_chapter_falls_back_to_the_nearest_one_with_the_text(self, tmp_path):
        path = make_epub(tmp_path / "book.epub", self.BODIES)
        assert find_text_in_epub(path, 2, "It was dark")["chapter_found"] == 3
        # chapters 1 and 3 are equally near; the earlier one wins
        assert find_text_in_epub(path, 2, "The Long Night")["chapter_found"] == 1

    def test_a_miss_is_an_error(self, tmp_path):
        path = make_epub(tmp_path / "book.epub", self.BODIES)
        assert find_text_in_epub(path, 1, "not in the book")["error"] == "Text not found in EPUB"

    def test_each_chapter_is_parsed_once_for_a_batch(self, tmp_path, monkeypatch):
        path = make_epub(tmp_path / "book.epub", [f"<p>chapter {i} text</p>" for i in range(1, 9)])
        feeds = []
        original = epub_utils.CFITextParser.feed
        monkeypatch.setattr(epub_utils.CFITextParser, "feed",
                            lambda self, data: (feeds.append(1), original(self, data))[1])
        cache = {}
        assert find_text_in_epub(path, 5, "chapter 5 text", cache=cache)["chapter_found"] == 5
        assert len(feeds) == 1                      # the chapter asked for, nothing else
        assert find_text_in_epub(path, 5, "chapter 5", cache=cache)["chapter_found"] == 5
        assert len(feeds) == 1                      # and not again
        find_text_in_epub(path, 1, "nowhere", cache=cache)
        assert len(feeds) == 8                      # a miss reads every chapter, once


def test_sort_index_is_the_spine_position_and_the_offset_in_that_chapter(tmp_path, monkeypatch):
    """Zotero's reader writes ``<0-based section>|<characters of the section before the range>``."""
    body = "<p>First paragraph.</p>\n<p>Second one has the needle in it.</p>"
    path = make_epub(tmp_path / "book.epub", ["<p>a long front matter chapter</p>", body])
    created = []

    def create_items(payloads, **_kwargs):
        created.extend(payloads)
        return {"success": {str(i): f"KEY{i}" for i in range(len(payloads))}, "failed": {}}

    monkeypatch.setattr(annotations, "_fetch_attachment_file",
                        lambda key, tmp, **kw: (path, "book.epub", "epub", None))
    monkeypatch.setattr(annotations, "_get_note_write_client",
                        lambda _op: (MagicMock(create_items=create_items), None))
    (result,) = annotations.create_annotations(
        "ATTACH01", [{"page": 2, "text": "the needle"}], ctx=MagicMock(), allow_epub=True)
    assert result["ok"], result
    before = "First paragraph.\nSecond one has "
    assert created[0]["annotationSortIndex"] == f"00001|{len(before):08d}"
    cfi = json.loads(created[0]["annotationPosition"])["value"]
    assert resolved_text(body, cfi) == "the needle"
