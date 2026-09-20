"""Tests for the match-key primitives in ``identifiers.py``.

Upstream #496: four places in the codebase each answered "are these the
same work?" differently. ``doi_match_key`` and ``normalize_title_for_matching``
are the first two primitives of the single stdlib-only home for that
normalisation; this file also covers the ISBN and arXiv movers
(``isbn_match_keys``, ``arxiv_identity``); a later task adds
``metadata_match_keys`` on top.
"""

import pytest

from zotero_mcp.identifiers import (
    arxiv_identity,
    doi_match_key,
    isbn_match_keys,
    normalize_title_for_matching,
)


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


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0-306-40615-2", {"9780306406157"}),
        ("0306406153", set()),  # deliberately bad checksum
    ],
)
def test_isbn_match_keys(raw, expected):
    assert isbn_match_keys(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2101.00001v2", "2101.00001"),
        ("10.48550/arXiv.2101.00001", "2101.00001"),
        ("hep-ph/9901234v3", "hep-ph/9901234"),
    ],
)
def test_arxiv_identity(raw, expected):
    assert arxiv_identity(raw) == expected
