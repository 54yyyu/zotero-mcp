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


def test_private_alias_is_the_same_object():
    """`tools/_helpers._normalize_doi` is kept as a compatibility alias;
    existing callers and tests must not observe the move."""
    from zotero_mcp.tools._helpers import _normalize_doi

    assert _normalize_doi is normalize_doi
