"""Tests for the public identifier helpers.

``normalize_doi`` moved out of ``tools/_helpers`` so downstream consumers
can import canonical DOI handling without the tool layer. The behaviour is
unchanged; ``tests/test_shared_helpers.py`` still exercises it through the
private alias, which is what proves the move is transparent.
"""

import pytest

from zotero_mcp.identifiers import normalize_doi


class TestNormalizeDoi:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("10.1038/nphys1170", "10.1038/nphys1170"),
            ("  10.1038/nphys1170  ", "10.1038/nphys1170"),
            ("doi:10.1038/nphys1170", "10.1038/nphys1170"),
            ("DOI: 10.1038/nphys1170", "10.1038/nphys1170"),
            ("https://doi.org/10.1038/nphys1170", "10.1038/nphys1170"),
            ("http://dx.doi.org/10.1038/nphys1170", "10.1038/nphys1170"),
            ("https://DOI.ORG/10.1038/nphys1170", "10.1038/nphys1170"),
        ],
    )
    def test_accepted_forms(self, raw, expected):
        assert normalize_doi(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("10.1038/nphys1170.", "10.1038/nphys1170"),
            ("10.1038/nphys1170,", "10.1038/nphys1170"),
            ("10.1038/nphys1170)", "10.1038/nphys1170"),
            ("10.1038/nphys1170];", "10.1038/nphys1170"),
        ],
    )
    def test_trailing_punctuation_from_prose(self, raw, expected):
        assert normalize_doi(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # TAO (Terrestrial, Atmospheric and Oceanic Sciences) puts a
            # parenthesised topical suffix on its DOIs, and it is part of the
            # DOI: CrossRef resolves the closing bracket and 404s without it.
            (
                "10.3319/TAO.2009.05.25.02(IWNOP)",
                "10.3319/TAO.2009.05.25.02(IWNOP)",
            ),
            (
                "doi:10.3319/TAO.2009.05.25.02(IWNOP)",
                "10.3319/TAO.2009.05.25.02(IWNOP)",
            ),
            (
                "https://doi.org/10.3319/TAO.2009.05.25.02(IWNOP)",
                "10.3319/TAO.2009.05.25.02(IWNOP)",
            ),
            # Prose punctuation still goes, and only the punctuation.
            (
                "10.3319/TAO.2009.05.25.02(IWNOP).",
                "10.3319/TAO.2009.05.25.02(IWNOP)",
            ),
            ("10.1234/a[b]", "10.1234/a[b]"),
        ],
    )
    def test_balanced_brackets_are_part_of_the_doi(self, raw, expected):
        """A closing bracket that has an opener inside the DOI is not prose
        punctuation. Stripping it unconditionally turns a resolvable DOI into
        a 404 (upstream #469)."""
        assert normalize_doi(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            "",
            "   ",
            "not-a-doi",
            "10.1/x",                              # registrant prefix too short
            "11.1038/nphys1170",                   # wrong directory indicator
            "https://example.com/10.1038/nphys1170",  # not a doi.org URL
        ],
    )
    def test_rejected(self, raw):
        assert normalize_doi(raw) is None

    def test_case_is_preserved(self):
        """DOIs resolve case-insensitively, but Scite echoes back what it
        was given, so normalisation must not lower-case."""
        assert normalize_doi("10.1038/NPhys1170") == "10.1038/NPhys1170"

    def test_non_string_input_is_coerced(self):
        assert normalize_doi(12345) is None


    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # A browser's address bar percent-encodes the brackets and angle
            # brackets of Wiley's SICI DOIs. The URL names the same DOI as
            # the bare form, so it must normalise to the bare form.
            (
                "https://doi.org/10.1002/%28SICI%291097-4636%28199709%2936%3A3%3C287%3A%3AAID-JBM2%3E3.0.CO%3B2-E",
                "10.1002/(SICI)1097-4636(199709)36:3<287::AID-JBM2>3.0.CO;2-E",
            ),
            ("https://doi.org/10.1000%2Fabc", "10.1000/abc"),
            # "%" inside a DOI is itself encoded in a URL as %25.
            ("https://doi.org/10.1000/50%25off", "10.1000/50%off"),
        ],
    )
    def test_percent_encoded_doi_url_is_decoded(self, raw, expected):
        assert normalize_doi(raw) == expected

    def test_bare_doi_is_not_percent_decoded(self):
        """Only a URL is percent-encoded; a bare DOI's '%' is literal."""
        assert normalize_doi("10.1000/50%25off") == "10.1000/50%25off"

def test_private_alias_is_the_same_object():
    """`tools/_helpers._normalize_doi` is kept as a compatibility alias;
    existing callers and tests must not observe the move."""
    from zotero_mcp.tools._helpers import _normalize_doi

    assert _normalize_doi is normalize_doi
