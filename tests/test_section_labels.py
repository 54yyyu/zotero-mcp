"""Section labels on indexed passages: ``_section_offsets`` / ``_section_at``.

The parser only has line breaks to go on (extracted PDF text loses styling),
so these cases pin the shapes it must accept and the ones it must not: a
recognised heading starts a label, an unrecognised heading-shaped line ends
one, and prose that merely uses a section word never becomes a heading.
"""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp.semantic_search import _section_at, _section_offsets


def _label_of(text: str, needle: str) -> str | None:
    """The label in force where *needle* first occurs in *text*."""
    return _section_at(_section_offsets(text), text.index(needle))


ARTICLE = """Title of the paper
A. Author, B. Author

Abstract
We study how passages are labelled.

1. Introduction
Prior work has labelled nothing.

2 Methods
2.1 Participants
Two hundred students took part.

Results:
The effect was large.

IV. Discussion
We discuss the large effect.

References
Author, A. (2020). A paper.
"""


def test_text_without_headings_has_no_labels():
    text = "Just a paragraph of prose.\nAnother line of prose without headings."
    assert _section_offsets(text) == []
    assert _section_at([], 10) is None


def test_empty_text():
    assert _section_offsets("") == []
    assert _section_offsets(None) == []  # type: ignore[arg-type]


def test_front_matter_is_unlabelled():
    assert _label_of(ARTICLE, "Title of the paper") is None
    assert _label_of(ARTICLE, "A. Author, B. Author") is None


@pytest.mark.parametrize(
    "needle, label",
    [
        ("We study how passages", "Abstract"),
        ("Prior work has labelled", "Introduction"),  # "1. Introduction"
        ("Two hundred students", "Methods"),  # "2.1 Participants" -> Methods
        ("The effect was large", "Results"),  # trailing colon accepted
        ("We discuss the large", "Discussion"),  # roman numeral "IV."
        ("Author, A. (2020)", "References"),
    ],
)
def test_numbered_and_plain_headings_are_recognised(needle, label):
    assert _label_of(ARTICLE, needle) == label


def test_offsets_are_in_document_order():
    marks = _section_offsets(ARTICLE)
    offsets = [offset for offset, _ in marks]
    assert offsets == sorted(offsets)


def test_section_word_inside_prose_is_not_a_heading():
    text = "Introduction\nThe results of this study, in discussion with others, are new.\n"
    # "results" and "discussion" occur mid-sentence: still Introduction.
    assert _label_of(text, "are new") == "Introduction"


def test_heading_with_trailing_text_is_not_a_heading():
    text = "Methods\nWe did things.\nResults from earlier work were mixed\nmore prose\n"
    # "Results from earlier work..." is a sentence, not a heading line.
    assert _label_of(text, "more prose") == "Methods"


def test_chapter_heading_ends_the_previous_section():
    book = (
        "Chapter 1\nConclusion\nThe first chapter concludes.\n\n"
        "Chapter 2 Piaget and Vygotsky\nA new chapter about development.\n"
    )
    assert _label_of(book, "The first chapter concludes") == "Conclusion"
    # Without the reset this inherited "Conclusion" from chapter 1.
    assert _label_of(book, "A new chapter about development") is None


@pytest.mark.parametrize("heading", ["CHAPTER THREE", "Part II", "Book 4: Later Work", "PART ONE"])
def test_chapter_part_book_variants_reset(heading):
    text = f"Discussion\nEarlier discussion.\n{heading}\nFresh material.\n"
    assert _label_of(text, "Earlier discussion") == "Discussion"
    assert _label_of(text, "Fresh material") is None


def test_unrecognised_all_caps_heading_ends_a_section():
    text = "References\nA. Author (2001).\nTHE SECOND ESSAY\nBody of the essay.\n"
    assert _label_of(text, "Body of the essay") is None


def test_recognised_all_caps_heading_keeps_its_label():
    # "METHODS" matches both the heading list and the all-caps reset shape;
    # the recognised label must win.
    text = "Introduction\nWhy.\nMETHODS\nHow we did it.\n"
    assert _label_of(text, "How we did it") == "Methods"


def test_label_resumes_after_a_reset():
    text = "CHAPTER 5\nUntitled stretch.\nResults\nNumbers here.\n"
    assert _label_of(text, "Untitled stretch") is None
    assert _label_of(text, "Numbers here") == "Results"


def test_title_case_subheadings_do_not_reset():
    # Journal subheadings are title case; they must not clear the label.
    text = "Methods\nSampling Strategy\nWe sampled schools.\n"
    assert _label_of(text, "We sampled schools") == "Methods"
