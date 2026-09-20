"""Tests for the match-key primitives in ``identifiers.py``.

Upstream #496: four places in the codebase each answered "are these the
same work?" differently. ``doi_match_key`` and ``normalize_title_for_matching``
are the first two primitives of the single stdlib-only home for that
normalisation; later tasks add ISBN/arXiv movers and ``metadata_match_keys``
on top.
"""

import pytest

from zotero_mcp.identifiers import doi_match_key, normalize_title_for_matching


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://doi.org/10.1000/ABC", "10.1000/abc"),
        ("doi:10.1000/abc.", "10.1000/abc"),
        ("10.1000/ABC", "10.1000/abc"),
        ("n/a", None),
        ("", None),
    ],
)
def test_doi_match_key(raw, expected):
    assert doi_match_key(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Self-Attention in <i>Drosophila</i>", "self attention in drosophila"),
        ("Self–Attention in Drosophila", "self attention in drosophila"),
        ("The self attention in drosophila!", "self attention in drosophila"),
        ("Über die &amp; Grenzen", "uber die grenzen"),
        ("DREAM<sub>(D)</sub>: x", "dream d x"),
        ("&lt;i&gt; literal", "i literal"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_title_for_matching(raw, expected):
    assert normalize_title_for_matching(raw) == expected
