"""The span a fuzzy match returns is the quote, not the window it was found in."""

import pytest

from zotero_mcp.pdf_utils import _sliding_window_match

_BODY = "however, such attention mechanisms are used in conjunction with a recurrent network and nothing else"
_TEXT = ("lorem ipsum dolor sit amet consectetur adipiscing elit " * 20
         + _BODY + " "
         + "sed do eiusmod tempor incididunt ut labore et dolore " * 20)


@pytest.mark.parametrize("quote, found", [
    # one typo
    ("such attention mechanisms are used in conjuncton with a recurrent network",
     "such attention mechanisms are used in conjunction with a recurrent network"),
    # another spelling
    ("such attention mechanisms are utilised in conjunction with a recurrent network",
     "such attention mechanisms are used in conjunction with a recurrent network"),
])
def test_span_starts_and_ends_on_the_quoted_words(quote, found):
    start, end, score = _sliding_window_match(_TEXT, quote)
    assert score > 0.85
    assert _TEXT[start:end] == found


def test_a_miss_still_reports_its_best_window():
    start, end, score = _sliding_window_match(_TEXT, "zebras jumping quickly over lazy dogs at noon")
    assert 0 < score < 0.6
    assert 0 <= start < end <= len(_TEXT)
