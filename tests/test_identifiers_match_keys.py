"""Tests for the match-key primitives in ``identifiers.py``.

Upstream #496: four places in the codebase each answered "are these the
same work?" differently. ``doi_match_key`` and ``normalize_title_for_matching``
are the first two primitives of the single stdlib-only home for that
normalisation; this file also covers the ISBN and arXiv movers
(``isbn_match_keys``, ``arxiv_identity``) and, below, ``metadata_match_keys``
-- the adapter that lets the four detection sites call one function
regardless of whether they hold a pyzotero API item, a bare data dict, or a
``local_db.ZoteroItem``.
"""

from types import SimpleNamespace

import pytest

from zotero_mcp.identifiers import (
    arxiv_identity,
    doi_match_key,
    isbn_match_keys,
    metadata_match_keys,
    normalize_title_for_matching,
)
from zotero_mcp.local_db import ZoteroItem


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
        # The "a" of a compound term is not an article: stripping it would
        # fold these to "b testing" and match them with "B Testing".
        ("A/B testing", "a b testing"),
        ("A-B testing", "a b testing"),
        ("B Testing", "b testing"),
        # A real article is dropped, whatever follows it.
        ("An Introduction to Mediation", "introduction to mediation"),
        ("The correction for attenuation", "correction for attenuation"),
        ("A Right to Strike?", "right to strike"),
        ("The R book", "r book"),
        ("The M-form Society", "m form society"),
        ("A 2 × 2 taxonomy", "2 2 taxonomy"),
        ("“The Firm”", "firm"),
        ("<i>The</i> Firm", "firm"),
        ("The Élan of Firms", "elan of firms"),
        # Only a leading article, and only one.
        ("Theory of the Firm", "theory of the firm"),
        ("The The", "the"),
    ],
)
def test_leading_article_dropped_only_when_whitespace_follows(raw, expected):
    assert normalize_title_for_matching(raw) == expected


@pytest.mark.parametrize("ch", [chr(i) for i in range(128)])
def test_ascii_fast_path_folds_every_ascii_character_like_the_unicode_path(ch):
    """ASCII titles skip the per-character category loop; the key must not
    depend on which path produced it."""
    from zotero_mcp.identifiers import _fold_ascii, _fold_unicode

    s = f"x{ch}y"
    assert _fold_ascii(s) == _fold_unicode(s)


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


# ---------------------------------------------------------------------------
# metadata_match_keys: one identity set regardless of item shape (T1.4)
#
# ``doi_match_key`` requires a 4-9 digit registrant code (``DOI_RE``), so
# the fixtures use "10.1000/x" rather than a shorter placeholder.
_DOI = "10.1000/x"
_TITLE = "Foo Bar"
_TITLE_KEY = "foo bar"
_EXTRA = "arXiv: 2101.00001"
_EXPECTED = frozenset({("doi", _DOI), ("title", _TITLE_KEY), ("arxiv", "2101.00001")})


def test_metadata_match_keys_api_item_dict():
    api_item = {"data": {"DOI": _DOI, "title": _TITLE, "extra": _EXTRA}}
    assert metadata_match_keys(api_item) == _EXPECTED


def test_metadata_match_keys_bare_data_dict():
    bare_data = {"DOI": _DOI, "title": _TITLE, "extra": _EXTRA}
    assert metadata_match_keys(bare_data) == _EXPECTED


def test_metadata_match_keys_reader_item():
    """The real ``local_db.ZoteroItem``, not a hand-rolled stand-in.

    ``item_id``/``key``/``item_type_id`` are required, non-identifying
    dataclass fields; ``tests/test_local_db.py`` constructs the class the
    same way (``ZoteroItem(item_id=1, key="TEST", item_type_id=1, ...)``).
    """
    reader_item = ZoteroItem(
        item_id=1,
        key="ABCD1234",
        item_type_id=1,
        doi=_DOI,
        title=_TITLE,
        extra=_EXTRA,
    )
    assert metadata_match_keys(reader_item) == _EXPECTED


@pytest.mark.parametrize(
    "kinds,expected",
    [
        (None, _EXPECTED),
        (("doi", "title"), {("doi", _DOI), ("title", _TITLE_KEY)}),
        ({"title"}, {("title", _TITLE_KEY)}),
        (("arxiv",), {("arxiv", "2101.00001")}),
        ((), frozenset()),
    ],
)
def test_metadata_match_keys_kinds_limits_what_is_computed(kinds, expected):
    item = {"data": {"DOI": _DOI, "title": _TITLE, "extra": _EXTRA}}
    assert metadata_match_keys(item, kinds=kinds) == expected


def test_metadata_match_keys_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="issn"):
        metadata_match_keys({"title": _TITLE}, kinds=("title", "issn"))


def test_metadata_match_keys_no_identifying_fields_is_empty_not_a_crash():
    assert metadata_match_keys({}) == frozenset()
    assert metadata_match_keys({"data": {}}) == frozenset()
    # ZoteroItem has no url/ISBN/archiveID attributes at all, so this also
    # proves _fields_of's getattr(..., default=None) tolerates a shape
    # that is simply missing most of the fields it looks for.
    assert metadata_match_keys(ZoteroItem(item_id=1, key="X", item_type_id=1)) == frozenset()


def test_metadata_match_keys_title_that_normalizes_to_empty_emits_no_title_key():
    # normalize_title_for_matching("!!!") == "" (all-punctuation title);
    # the falsy-string guard in metadata_match_keys must not emit
    # ("title", "").
    assert metadata_match_keys({"title": "!!!"}) == frozenset()


def test_metadata_match_keys_garbage_doi_emits_no_doi_key():
    assert metadata_match_keys({"DOI": "n/a"}) == frozenset()


def test_metadata_match_keys_isbn_field_with_two_isbns_emits_both():
    keys = metadata_match_keys({"ISBN": "0-306-40615-2, 0136091814"})
    assert keys == {("isbn", "9780306406157"), ("isbn", "9780136091813")}


@pytest.mark.parametrize(
    "field,value",
    [
        ("url", "https://arxiv.org/abs/2101.00001"),
        ("archiveID", "arXiv:2101.00001"),
        ("extra", "arXiv: 2101.00001"),
    ],
)
def test_metadata_match_keys_arxiv_reachable_from_url_archiveid_extra(field, value):
    assert metadata_match_keys({field: value}) == {("arxiv", "2101.00001")}


def test_metadata_match_keys_arxiv_reachable_from_doi_field():
    """arXiv's DataCite DOI form yields both an arxiv key and a doi key.

    The DOI field is consulted for both ``doi_match_key`` and
    ``arxiv_identity``, so an item added by DOI is still recognized by a
    later add of the same paper's arXiv ID -- see ``arxiv_identity``'s own
    docstring for why that matters.
    """
    keys = metadata_match_keys({"DOI": "10.48550/arXiv.2101.00001"})
    assert keys == {("doi", "10.48550/arxiv.2101.00001"), ("arxiv", "2101.00001")}


def test_metadata_match_keys_duck_types_snake_case_archive_id_attribute():
    """_fields_of tolerates shapes beyond ZoteroItem's own attribute set.

    ZoteroItem carries no archiveID, so this pins the snake_case fallback
    (``archive_id`` before ``archiveID``) with a minimal object of its
    own -- it is not a substitute for the required-by-brief parity tests
    above, which construct the real ZoteroItem.
    """
    reader_like = SimpleNamespace(archive_id="arXiv:2101.00001")
    assert metadata_match_keys(reader_like) == {("arxiv", "2101.00001")}
