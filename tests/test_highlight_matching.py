"""Where a highlight lands: hyphenated compounds, long quotes, page breaks, neighbours.

Every PDF here is synthetic. A highlight is judged by the characters whose
centres lie inside its rects, read independently of ``text_in_rects``.
"""

import json
import textwrap
from unittest.mock import MagicMock

import pytest

from zotero_mcp import pdf_utils
from zotero_mcp.pdf_utils import find_text_position, text_in_rects
from zotero_mcp.tools import annotations

fitz = pytest.importorskip("fitz")


def make_pdf(path, pages, *, fontsize=10, leading=14, fontname="helv") -> str:
    """One page per entry; an entry is a list of lines, or a paragraph to wrap."""
    doc = fitz.open()
    for lines in pages:
        if isinstance(lines, str):
            lines = textwrap.wrap(lines, 85)
        page = doc.new_page(width=612, height=792)
        for index, line in enumerate(lines):
            page.insert_text((72, 100 + leading * index), line, fontsize=fontsize, fontname=fontname)
    doc.save(str(path))
    doc.close()
    return str(path)


def covered(path, page_index, rects) -> str:
    """Text of the characters whose centre is inside the Zotero rects, a rect per piece."""
    with fitz.open(path) as doc:
        page = doc[page_index]
        to_page = fitz.Matrix(*pdf_utils._pdf_to_page_transform(page))
        pieces = []
        for rect in rects:
            box = fitz.Rect(*rect) * to_page
            box.normalize()
            text = ""
            for block in page.get_text("rawdict", flags=3)["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        for char in span["chars"]:
                            x0, y0, x1, y1 = char["bbox"]
                            if box.x0 - 0.5 <= (x0 + x1) / 2 <= box.x1 + 0.5 and box.y0 <= (y0 + y1) / 2 <= box.y1:
                                text += char["c"]
            pieces.append(text.strip())
        return " | ".join(pieces)


# ---------------------------------------------------------------------------
# A hyphen at the end of a line
# ---------------------------------------------------------------------------

@pytest.fixture
def hyphen_pdf(tmp_path):
    return make_pdf(tmp_path / "hyphens.pdf", [[
        "We compare against several strong baselines and report state-of-the-",
        "art results on both benchmarks, using a well-",
        "known optimizer and a learning rate of 3e-",
        "4 throughout. The encoder-",
        "decoder model is described in Section 3, regard-",
        "less of the sample size.",
    ]])


@pytest.mark.parametrize("quote, covers", [
    # the hyphen belongs to the word; it used to be dropped from the index, so
    # the quote as printed only matched loosely and the highlight drifted
    ("report state-of-the-art results on both", "report state-of-the- | art results on both"),
    ("using a well-known optimizer", "using a well- | known optimizer"),
    ("The encoder-decoder model is described", "The encoder- | decoder model is described"),
    ("learning rate of 3e-4 throughout", "learning rate of 3e- | 4 throughout"),
    # the hyphen is typesetting: both spellings find the word
    ("in Section 3, regardless of the sample", "in Section 3, regard- | less of the sample"),
    ("in Section 3, regard-less of the sample", "in Section 3, regard- | less of the sample"),
    # plain words further down a page that has such hyphens: PyMuPDF 2.x's own
    # search reported them four characters to the left ("gard- | less of the s")
    ("less of the sample size.", "less of the sample size."),
    ("model is described in Section 3", "model is described in Section 3"),
    # typed without its hyphens
    ("report state of the art results", "report state-of-the- | art results"),
    # a long quote, whose last words hold such a hyphen: it ran one letter too far
    ("We compare against several strong baselines and report state-of-the-art results on both "
     "benchmarks, using a well-known optimizer",
     "We compare against several strong baselines and report state-of-the- | "
     "art results on both benchmarks, using a well- | known optimizer"),
])
def test_a_quote_across_a_line_end_hyphen_is_covered_exactly(hyphen_pdf, quote, covers):
    result = find_text_position(hyphen_pdf, 1, quote)
    assert "error" not in result, result
    assert covered(hyphen_pdf, 0, result["rects"]) == covers


# ---------------------------------------------------------------------------
# Long quotes (the start-and-end anchor path)
# ---------------------------------------------------------------------------

PAGE_1 = ("Attention mechanisms have become an integral part of compelling sequence modeling and transduction "
          "models in various tasks, allowing modeling of dependencies without regard to their distance in the input "
          "or output sequences. In all but a few cases, however, such attention mechanisms are used in conjunction "
          "with a recurrent network. In this work we propose the Transformer, a model architecture eschewing "
          "recurrence and instead relying entirely on an attention mechanism to draw global dependencies between "
          "input and output.")
PAGE_2 = ("The Transformer allows for significantly more parallelization and can reach a new state of the art in "
          "translation quality after being trained for as little as twelve hours on eight GPUs.")
ACROSS = ("a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to "
          "draw global dependencies between input and output. The Transformer allows for significantly more "
          "parallelization")


@pytest.fixture
def long_pdf(tmp_path):
    return make_pdf(tmp_path / "long.pdf", [PAGE_1, PAGE_2, "An unrelated third page."], fontname="tiro")


def letters(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())


class TestLongQuotes:
    def test_a_quote_within_the_page_is_covered_from_its_first_to_its_last_word(self, long_pdf):
        quote = ("In all but a few cases, however, such attention mechanisms are used in conjunction "
                 "with a recurrent network. In this work we propose the Transformer")
        result = find_text_position(long_pdf, 1, quote)
        assert "nextPageRects" not in result
        assert letters(covered(long_pdf, 0, result["rects"])) == letters(quote)

    def test_an_elided_middle_still_matches_between_its_two_ends(self, long_pdf):
        quote = ("Attention mechanisms have become an integral part of compelling sequence modeling ... "
                 "In this work we propose the Transformer, a model architecture eschewing recurrence")
        result = find_text_position(long_pdf, 1, quote)
        got = letters(covered(long_pdf, 0, result["rects"]))
        assert got.startswith("attentionmechanismshave") and got.endswith("eschewingrecurrence")

    def test_an_invented_ending_is_not_placed_by_guessing_its_length(self, long_pdf):
        """The start matched, the end was nowhere, and the highlight used to run
        1.1x the quote's length from the start, to the middle of a word."""
        quote = ("In all but a few cases, however, such attention mechanisms are used in conjunction with a "
                 "recurrent network, which the authors then go on to criticise at some length for its cost, "
                 "its sequential nature and the difficulty of training it on long documents in practice")
        result = find_text_position(long_pdf, 1, quote)
        assert "error" in result
        assert result["candidates"] and result["candidates"][0]["page"] == 1

    @pytest.mark.parametrize("hint", [1, 2])
    def test_a_quote_across_a_page_break_covers_both_pages(self, long_pdf, hint):
        result = find_text_position(long_pdf, hint, ACROSS)
        assert result["pageIndex"] == 0
        first = "a model architecture eschewing recurrence and instead relying entirely on an attention " \
                "mechanism to draw global dependencies between input and output."
        assert letters(covered(long_pdf, 0, result["rects"])) == letters(first)
        assert covered(long_pdf, 1, result["nextPageRects"]) == \
            "The Transformer allows for significantly more parallelization"

    def test_the_rest_must_really_be_on_the_next_page(self, long_pdf):
        quote = ACROSS.replace("The Transformer allows for significantly more parallelization",
                               "Nothing like this sentence is printed anywhere on the following page")
        result = find_text_position(long_pdf, 1, quote)
        assert "nextPageRects" not in result

    def test_typos_in_both_ends_keep_the_last_word_whole(self, long_pdf):
        quote = ("In all but a few cases, however, such attenton mechanisms are used in conjunction with a "
                 "recurrent network. In this work we propose the Transfomer, a model")
        result = find_text_position(long_pdf, 1, quote)
        assert covered(long_pdf, 0, result["rects"]).endswith("the Transformer, a model")

    def test_a_highlight_across_a_page_break_is_stored_as_one_annotation(self, long_pdf, monkeypatch):
        created = []

        def create_items(payloads, **_kwargs):
            created.extend(payloads)
            return {"success": {str(i): f"KEY{i}" for i in range(len(payloads))}, "failed": {}}

        monkeypatch.setattr(annotations, "_fetch_attachment_file",
                            lambda key, tmp, **kw: (long_pdf, "long.pdf", "pdf", None))
        monkeypatch.setattr(annotations, "_get_note_write_client",
                            lambda _op: (MagicMock(create_items=create_items), None))
        (result,) = annotations.create_annotations("ATTACH01", [{"page": 1, "text": ACROSS}], ctx=MagicMock())
        assert result["ok"] and result["page_found"] == 1 and result["continues_on_page"] == 2
        position = json.loads(created[0]["annotationPosition"])
        assert position["pageIndex"] == 0
        assert covered(long_pdf, 1, position["nextPageRects"]) == \
            "The Transformer allows for significantly more parallelization"

    def test_a_dry_run_shows_both_parts(self, long_pdf, monkeypatch):
        monkeypatch.setattr(annotations, "_fetch_attachment_file",
                            lambda key, tmp, **kw: (long_pdf, "long.pdf", "pdf", None))
        (result,) = annotations.create_annotations(
            "ATTACH01", [{"page": 1, "text": ACROSS}], ctx=MagicMock(), dry_run=True)
        assert letters(result["matched_text"]) == letters(ACROSS)


# ---------------------------------------------------------------------------
# The page given and its neighbours
# ---------------------------------------------------------------------------

def test_the_text_itself_on_the_next_page_beats_a_near_miss_on_the_page_given(tmp_path):
    path = make_pdf(tmp_path / "neighbours.pdf", [
        ["The model reaches 91.2 percent accuracy on the held-out test set."],
        ["The model reaches 97.2 percent accuracy on the held-out test set."],
    ])
    quote = "The model reaches 97.2 percent accuracy on the held-out test set."
    result = find_text_position(path, 1, quote)
    assert result["pageIndex"] == 1
    assert covered(path, 1, result["rects"]) == quote


# ---------------------------------------------------------------------------
# Reading a highlight back
# ---------------------------------------------------------------------------

def test_text_in_rects_leaves_out_the_lines_above_and_below(tmp_path):
    """11pt Helvetica set 12pt apart: PyMuPDF's line boxes are 15pt high and overlap."""
    lines = ["First line of the paragraph with plenty of words in it.",
             "The second line is the one that gets highlighted here.",
             "Third line of the paragraph, which must stay out of it."]
    path = make_pdf(tmp_path / "tight.pdf", [lines], fontsize=11, leading=12)
    result = find_text_position(path, 1, "second line is the one that gets highlighted")
    assert text_in_rects(path, 0, result["rects"]) == "second line is the one that gets highlighted"


# ---------------------------------------------------------------------------
# A batch reads each page once
# ---------------------------------------------------------------------------

def test_a_batch_on_one_page_extracts_the_page_once(long_pdf, monkeypatch):
    monkeypatch.setattr(annotations, "_fetch_attachment_file",
                        lambda key, tmp, **kw: (long_pdf, "long.pdf", "pdf", None))
    counts = {"spans": 0, "chars": 0}
    spans, chars = pdf_utils._extract_page_spans, pdf_utils._extract_page_char_rects
    monkeypatch.setattr(pdf_utils, "_extract_page_spans",
                        lambda page: (counts.__setitem__("spans", counts["spans"] + 1), spans(page))[1])
    monkeypatch.setattr(pdf_utils, "_extract_page_char_rects",
                        lambda page: (counts.__setitem__("chars", counts["chars"] + 1), chars(page))[1])
    specs = [
        {"page": 1, "text": "integral part of compelling sequence modeling"},
        {"page": 1, "text": "attenton mechanisms are used in conjunction"},      # loose match
        {"page": 1, "text": "In all but a few cases, however, such attention mechanisms are used in conjunction "
                            "with a recurrent network. In this work we propose the Transformer"},
        {"page": 1, "rect": [0.1, 0.1, 0.3, 0.2]},
        {"page": 1, "note": [0.5, 0.5], "comment": "a note"},
    ]
    results = annotations.create_annotations("ATTACH01", specs, ctx=MagicMock(), dry_run=True)
    assert [r["ok"] for r in results] == [True] * 5, results
    # page 1 for every spec, and its two neighbours for the loose match
    assert counts == {"spans": 3, "chars": 1}
