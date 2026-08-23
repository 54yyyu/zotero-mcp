"""Type-specific base fields resolve for title *and* date (#452).

Zotero stores several types' title and date under type-specific keys mapped to
a base field: a case's title is `caseName` and its date `dateDecided`, a
statute's are `nameOfAct` and `dateEnacted`, a patent's date is `issueDate`.
Every formatter read `data["title"]` and `data["date"]` directly, so the data
was present and correct in Zotero throughout and only the presentation lost it.

The reporter's summary of why this is worse than a cosmetic bug: the search
layer resolves base fields correctly, so an affected item is *findable but
unidentifiable*, and the only recovery is a second `format="json"` call at
roughly 450 tokens against the ~15 it would have cost to print the name
already in hand.

The item throughout is theirs: QI92BRVI, a UK Upper Tribunal decision.
"""

import pytest

from zotero_mcp.client import format_item_metadata, generate_bibtex
from zotero_mcp.utils import format_item_result, item_display_date, item_display_title


def _case():
    return {"key": "QI92BRVI", "data": {
        "key": "QI92BRVI", "itemType": "case",
        "caseName": "DSG Retail Limited v The Information Commissioner",
        "court": "Upper Tribunal", "dateDecided": "8 October 2024",
        "docketNumber": "[2024] UKUT 287 (AAC)", "creators": [], "tags": [],
    }, "meta": {"parsedDate": "2024-10-08"}}


def _statute():
    return {"key": "ST123456", "data": {
        "key": "ST123456", "itemType": "statute",
        "nameOfAct": "Data Protection Act 2018", "dateEnacted": "2018-05-23",
        "publicLawNumber": "c. 12", "creators": [], "tags": [],
    }, "meta": {}}


def _email():
    return {"key": "EM123456", "data": {
        "key": "EM123456", "itemType": "email",
        "subject": "Re: the tribunal decision", "date": "2024-10-09",
        "creators": [], "tags": [],
    }, "meta": {}}


def _patent():
    return {"key": "PT123456", "data": {
        "key": "PT123456", "itemType": "patent",
        "title": "A Widget", "issueDate": "2019-03-05",
        "patentNumber": "US10,000,000", "creators": [], "tags": [],
    }, "meta": {}}


def _article():
    return {"key": "AAAA1111", "data": {
        "key": "AAAA1111", "itemType": "journalArticle",
        "title": "Attention Is All You Need", "date": "2017-06-12",
        "publicationTitle": "NeurIPS", "creators": [], "tags": [],
    }, "meta": {}}


class TestResolvers:
    @pytest.mark.parametrize("item,expected", [
        (_case(), "DSG Retail Limited v The Information Commissioner"),
        (_statute(), "Data Protection Act 2018"),
        (_email(), "Re: the tribunal decision"),
        (_article(), "Attention Is All You Need"),
    ])
    def test_title_resolves(self, item, expected):
        assert item_display_title(item["data"]) == expected

    @pytest.mark.parametrize("item,expected", [
        (_case(), "8 October 2024"),
        (_statute(), "2018-05-23"),
        (_patent(), "2019-03-05"),
        (_email(), "2024-10-09"),
        (_article(), "2017-06-12"),
    ])
    def test_date_resolves(self, item, expected):
        assert item_display_date(item["data"]) == expected

    def test_missing_date_is_empty_not_a_placeholder(self):
        """Callers pick their own placeholder; the resolver reports absence."""
        assert item_display_date({"itemType": "case"}) == ""

    def test_unknown_item_type_falls_back_to_the_plain_fields(self):
        data = {"itemType": "notARealType", "title": "T", "date": "2020"}
        assert item_display_title(data) == "T"
        assert item_display_date(data) == "2020"


class TestSearchListings:
    """`format_item_result` — the path the issue was reported against."""

    @pytest.mark.parametrize("item,title,date", [
        (_case(), "DSG Retail Limited v The Information Commissioner", "8 October 2024"),
        (_statute(), "Data Protection Act 2018", "2018-05-23"),
        (_patent(), "A Widget", "2019-03-05"),
    ])
    def test_neither_untitled_nor_no_date(self, item, title, date):
        rendered = "\n".join(format_item_result(item, index=2))
        assert title in rendered
        assert f"**Date:** {date}" in rendered
        assert "Untitled" not in rendered
        assert "No date" not in rendered

    def test_a_genuinely_dateless_item_still_says_no_date(self):
        item = {"key": "X", "data": {"key": "X", "itemType": "case",
                                     "caseName": "A v B", "creators": [], "tags": []}}
        assert "**Date:** No date" in "\n".join(format_item_result(item))

    def test_ordinary_articles_are_unaffected(self):
        rendered = "\n".join(format_item_result(_article(), index=1))
        assert "Attention Is All You Need" in rendered
        assert "**Date:** 2017-06-12" in rendered


class TestItemMetadata:
    def test_case_renders_title_and_date(self):
        rendered = format_item_metadata(_case())
        assert rendered.startswith("# DSG Retail Limited v The Information Commissioner")
        assert "**Date:** 8 October 2024" in rendered

    def test_type_specific_fields_are_no_longer_dropped(self):
        """The formatter branched on journalArticle and bookSection by name and
        rendered nothing type-specific for anything else, so a case's whole
        field set vanished and its record looked empty."""
        rendered = format_item_metadata(_case())
        assert "Upper Tribunal" in rendered
        assert "[2024] UKUT 287 (AAC)" in rendered

    def test_the_resolved_title_field_is_not_also_repeated_raw(self):
        """`caseName` is already the heading; printing it again under its own
        name would read as two different pieces of information."""
        rendered = format_item_metadata(_case())
        assert "**Case Name:**" not in rendered
        assert "**Date Decided:**" not in rendered

    def test_statute_number_field_appears(self):
        assert "c. 12" in format_item_metadata(_statute())

    def test_ordinary_article_output_is_unchanged(self):
        rendered = format_item_metadata(_article())
        assert "**Journal:** NeurIPS" in rendered
        # No spray of empty type-specific lines.
        assert "**Publication Title:**" not in rendered


class TestBibtex:
    def test_case_is_no_longer_an_empty_misc_entry(self):
        """The reporter's reproduction 3: `@misc{nodate_QI92BRVI\\n}`."""
        bib = generate_bibtex(_case())
        assert "DSG Retail Limited v The Information Commissioner" in bib
        assert "nodate" not in bib
        assert "year = {2024}" in bib

    def test_statute_exports_name_and_year(self):
        bib = generate_bibtex(_statute())
        assert "Data Protection Act 2018" in bib
        assert "year = {2018}" in bib

    def test_year_appears_exactly_once(self):
        """A display date is not a BibTeX year, and emitting the field twice
        makes the entry invalid."""
        for item in (_case(), _statute(), _article()):
            assert generate_bibtex(item).count("year = {") == 1

    def test_year_comes_from_a_display_date_not_its_first_four_characters(self):
        """`"8 October 2024"[:4]` is `"8 Oc"`."""
        assert "year = {2024}" in generate_bibtex(_case())

    def test_ordinary_article_bibtex_is_unchanged(self):
        bib = generate_bibtex(_article())
        assert "@article{" in bib
        assert "title = {Attention Is All You Need}" in bib
        assert "year = {2017}" in bib
