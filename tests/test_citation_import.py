"""Unit tests for zotero_mcp.citation_import (parse + converters)."""

import json
from pathlib import Path

import pytest

from zotero_mcp.citation_import import (
    CSL_TYPE_MAP,
    _csl_text,
    _format_bibtex_date,
    _format_csl_date,
    _parse_bibtex_author_list,
    bibtex_entry_to_zotero,
    coerce_csl_json_input,
    csl_json_to_zotero,
    merge_tags,
    parse_bibtex,
)
from zotero_mcp.schema import valid_fields

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def schema_template(item_type: str) -> dict:
    """A template with every field Zotero says the type has.

    ``make_template`` below is hand-written and covers only the fields its
    own tests assert on, which is exactly how a dropped field can pass
    unnoticed: ``_set_if_in_template`` writes a field only when the template
    already carries it. The fixture tests build from the bundled Zotero
    schema instead, so a value the converter maps to a real field cannot go
    missing just because this file forgot to list it.
    """
    return {
        "itemType": item_type,
        **{f: "" for f in valid_fields(item_type)},
        "creators": [],
        "tags": [],
        "collections": [],
        "relations": {},
    }

# ---------------------------------------------------------------------------
# Template fixture — mirrors the subset of Zotero's item_template() we need
# ---------------------------------------------------------------------------

def make_template(item_type: str) -> dict:
    base = {
        "itemType": item_type,
        "title": "",
        "creators": [],
        "tags": [],
        "collections": [],
        "relations": {},
        "date": "",
        "abstractNote": "",
        "url": "",
        "DOI": "",
        "extra": "",
        "shortTitle": "",
        "language": "",
    }
    article_fields = {
        "publicationTitle": "",
        "volume": "",
        "issue": "",
        "pages": "",
        "ISSN": "",
        "publisher": "",
        "series": "",
        "seriesText": "",
        "journalAbbreviation": "",
    }
    if item_type in ("journalArticle", "preprint",
                     "magazineArticle", "newspaperArticle"):
        base.update(article_fields)
    if item_type == "conferencePaper":
        base.update(article_fields)
        base.update({"proceedingsTitle": "", "conferenceName": "",
                     "place": "", "ISBN": ""})
    if item_type == "bookSection":
        base.update({
            "bookTitle": "", "publisher": "", "place": "", "ISBN": "",
            "pages": "", "edition": "", "volume": "", "ISSN": "",
            "series": "", "seriesNumber": "", "numberOfVolumes": "",
        })
    if item_type == "book":
        base.update({
            "publisher": "", "place": "", "ISBN": "", "numPages": "",
            "edition": "", "volume": "", "ISSN": "", "series": "",
            "seriesNumber": "", "numberOfVolumes": "",
        })
    if item_type == "thesis":
        base.update({
            "thesisType": "", "university": "", "place": "", "numPages": "",
        })
    if item_type == "report":
        base.update({
            "reportNumber": "", "reportType": "", "institution": "",
            "place": "", "pages": "", "seriesTitle": "",
        })
    if item_type == "webpage":
        base.update({"websiteTitle": "", "websiteType": "", "accessDate": ""})
    if item_type == "patent":
        base.update({"patentNumber": "", "place": "", "country": "",
                     "issuingAuthority": "", "pages": ""})
    if item_type == "document":
        base.update({"publisher": ""})
    return base


# ---------------------------------------------------------------------------
# parse_bibtex
# ---------------------------------------------------------------------------

class TestParseBibtex:
    def test_parses_single_article(self):
        bib = """
        @article{smith2020,
          title = {Hello World},
          author = {Smith, John},
          journal = {Nature},
          year = {2020},
        }
        """
        entries = parse_bibtex(bib)
        assert len(entries) == 1
        e = entries[0]
        assert e["entry_type"] == "article"
        assert e["citekey"] == "smith2020"
        assert e["fields"]["title"] == "Hello World"
        assert e["fields"]["author"] == "Smith, John"

    def test_parses_multiple_entries(self):
        bib = """
        @article{a, title={A}, author={X, Y}, year={2020}}
        @book{b, title={B}, author={P, Q}, year={2021}, publisher={Pub}}
        """
        entries = parse_bibtex(bib)
        assert len(entries) == 2
        assert entries[0]["entry_type"] == "article"
        assert entries[1]["entry_type"] == "book"

    def test_empty_input_returns_empty_list(self):
        assert parse_bibtex("") == []
        assert parse_bibtex("   ") == []

    def test_unicode_conversion(self):
        """LaTeX accents should be converted to unicode."""
        bib = r"@article{a, title={Caf{\'e}}, author={Doe, J}, year=2020}"
        entries = parse_bibtex(bib)
        assert entries[0]["fields"]["title"] == "Café"


# ---------------------------------------------------------------------------
# Author parsing
# ---------------------------------------------------------------------------

class TestAuthorParsing:
    def test_last_first_format(self):
        c = _parse_bibtex_author_list("Smith, John")
        assert c == [{"creatorType": "author", "firstName": "John", "lastName": "Smith"}]

    def test_first_last_format(self):
        c = _parse_bibtex_author_list("John Smith")
        assert c == [{"creatorType": "author", "firstName": "John", "lastName": "Smith"}]

    def test_multiple_authors_split_on_and(self):
        c = _parse_bibtex_author_list("Smith, John and Jane Doe")
        assert len(c) == 2
        assert c[0]["lastName"] == "Smith"
        assert c[1]["lastName"] == "Doe"

    def test_corporate_author_with_llc(self):
        c = _parse_bibtex_author_list("{Acme Consortium, LLC}")
        assert c == [{"creatorType": "author", "name": "Acme Consortium, LLC"}]

    def test_corporate_author_with_inc_suffix(self):
        c = _parse_bibtex_author_list("Google Inc")
        assert c == [{"creatorType": "author", "name": "Google Inc"}]

    def test_brace_protected_author_with_and(self):
        """`{Smith and Jones, Ltd}` should NOT be split on ' and '."""
        c = _parse_bibtex_author_list("{Smith and Jones, Ltd}")
        assert len(c) == 1
        assert c[0].get("name") == "Smith and Jones, Ltd"

    def test_empty_input(self):
        assert _parse_bibtex_author_list("") == []

    def test_single_word_author(self):
        c = _parse_bibtex_author_list("Cher")
        assert c == [{"creatorType": "author", "name": "Cher"}]


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

class TestDateParsing:
    def test_year_only(self):
        assert _format_bibtex_date("2020", "", "", "") == "2020"

    def test_year_month_name(self):
        assert _format_bibtex_date("2020", "March", "", "") == "2020-03"

    def test_year_month_number(self):
        assert _format_bibtex_date("2020", "3", "", "") == "2020-03"

    def test_year_month_day(self):
        assert _format_bibtex_date("2020", "mar", "5", "") == "2020-03-05"

    def test_iso_date_overrides(self):
        assert _format_bibtex_date("2020", "mar", "5", "1999-12-31") == "1999-12-31"

    def test_empty_year(self):
        assert _format_bibtex_date("", "mar", "", "") == ""

    def test_csl_date_parts(self):
        assert _format_csl_date({"date-parts": [[2020, 3, 15]]}) == "2020-03-15"

    def test_csl_date_literal(self):
        assert _format_csl_date({"literal": "circa 1920"}) == "circa 1920"

    def test_csl_date_raw(self):
        assert _format_csl_date({"raw": "2019-05"}) == "2019-05"

    def test_csl_date_year_only(self):
        assert _format_csl_date({"date-parts": [[2020]]}) == "2020"


# ---------------------------------------------------------------------------
# bibtex_entry_to_zotero
# ---------------------------------------------------------------------------

class TestBibtexToZotero:
    def test_article_basic(self):
        bib = """
        @article{s20,
          title={A Paper},
          author={Smith, John and Doe, Jane},
          journal={Nature},
          year={2020},
          volume={42},
          number={7},
          pages={1--10},
          doi={10.1/x},
        }
        """
        item = bibtex_entry_to_zotero(parse_bibtex(bib)[0], make_template)
        assert item["itemType"] == "journalArticle"
        assert item["title"] == "A Paper"
        assert item["publicationTitle"] == "Nature"
        assert item["volume"] == "42"
        assert item["issue"] == "7"
        assert item["pages"] == "1-10"  # normalized from --
        assert item["DOI"] == "10.1/x"
        assert len(item["creators"]) == 2
        assert "Citation Key: s20" in item["extra"]

    def test_inproceedings_uses_proceedings_title(self):
        bib = """
        @inproceedings{d19,
          title={ML Paper},
          author={Doe, J},
          booktitle={Proc. ICML},
          year={2019},
        }
        """
        item = bibtex_entry_to_zotero(parse_bibtex(bib)[0], make_template)
        assert item["itemType"] == "conferencePaper"
        assert item["proceedingsTitle"] == "Proc. ICML"

    def test_inbook_uses_book_title(self):
        bib = """
        @incollection{k, title={Chapter}, author={Kim, K},
          booktitle={Big Book}, year={2020}, publisher={Pub}}
        """
        item = bibtex_entry_to_zotero(parse_bibtex(bib)[0], make_template)
        assert item["itemType"] == "bookSection"
        assert item["bookTitle"] == "Big Book"

    def test_phdthesis_sets_thesis_type(self):
        bib = "@phdthesis{t, title={Diss}, author={A, B}, school={MIT}, year={2020}}"
        item = bibtex_entry_to_zotero(parse_bibtex(bib)[0], make_template)
        assert item["itemType"] == "thesis"
        assert item["thesisType"] == "PhD thesis"
        # "school" should populate university
        assert item["university"] == "MIT"

    def test_keywords_become_tags(self):
        bib = "@article{x, title={T}, author={A, B}, year={2020}, keywords={alpha, beta, gamma}}"
        item = bibtex_entry_to_zotero(parse_bibtex(bib)[0], make_template)
        tag_names = [t["tag"] for t in item["tags"]]
        assert tag_names == ["alpha", "beta", "gamma"]

    def test_keywords_semicolon_separated(self):
        bib = "@article{x, title={T}, author={A, B}, year={2020}, keywords={alpha; beta}}"
        item = bibtex_entry_to_zotero(parse_bibtex(bib)[0], make_template)
        tag_names = [t["tag"] for t in item["tags"]]
        assert tag_names == ["alpha", "beta"]

    def test_unknown_field_goes_to_extra(self):
        bib = "@article{x, title={T}, author={A, B}, year={2020}, funding={NSF-123}}"
        item = bibtex_entry_to_zotero(parse_bibtex(bib)[0], make_template)
        assert "funding: NSF-123" in item["extra"]

    def test_note_appended_to_extra(self):
        bib = "@article{x, title={T}, author={A, B}, year={2020}, note={See also...}}"
        item = bibtex_entry_to_zotero(parse_bibtex(bib)[0], make_template)
        assert "See also..." in item["extra"]

    def test_misc_maps_to_document(self):
        bib = "@misc{x, title={T}, author={A, B}, year={2020}}"
        item = bibtex_entry_to_zotero(parse_bibtex(bib)[0], make_template)
        assert item["itemType"] == "document"

    def test_arxiv_eprint(self):
        bib = "@article{x, title={T}, author={A, B}, year={2020}, eprint={2010.12345}, eprinttype={arxiv}}"
        item = bibtex_entry_to_zotero(parse_bibtex(bib)[0], make_template)
        assert "arXiv: 2010.12345" in item["extra"]
        assert item["url"] == "https://arxiv.org/abs/2010.12345"


# ---------------------------------------------------------------------------
# csl_json_to_zotero
# ---------------------------------------------------------------------------

class TestCslJsonToZotero:
    def test_article_journal(self):
        csl = {
            "type": "article-journal",
            "id": "X2020",
            "title": "Hello",
            "author": [{"given": "John", "family": "Smith"}],
            "issued": {"date-parts": [[2020, 3, 15]]},
            "container-title": "Nature",
            "volume": "42",
            "issue": "7",
            "page": "1-10",
            "DOI": "10.1/x",
        }
        item = csl_json_to_zotero(csl, make_template)
        assert item["itemType"] == "journalArticle"
        assert item["title"] == "Hello"
        assert item["publicationTitle"] == "Nature"
        assert item["date"] == "2020-03-15"
        assert item["DOI"] == "10.1/x"
        assert item["creators"][0]["firstName"] == "John"
        assert item["creators"][0]["lastName"] == "Smith"
        assert "Citation Key: X2020" in item["extra"]

    def test_chapter_uses_book_title(self):
        csl = {
            "type": "chapter", "title": "C",
            "author": [{"family": "K"}],
            "container-title": "Big Book",
            "publisher": "Pub",
            "issued": {"date-parts": [[2020]]},
        }
        item = csl_json_to_zotero(csl, make_template)
        assert item["itemType"] == "bookSection"
        assert item["bookTitle"] == "Big Book"

    def test_paper_conference(self):
        csl = {
            "type": "paper-conference", "title": "P",
            "author": [{"family": "A"}],
            "container-title": "ICML 2020",
            "issued": {"date-parts": [[2020]]},
        }
        item = csl_json_to_zotero(csl, make_template)
        assert item["itemType"] == "conferencePaper"
        assert item["proceedingsTitle"] == "ICML 2020"

    def test_literal_author(self):
        csl = {
            "type": "article-journal", "title": "t",
            "author": [{"literal": "Acme Corp."}],
        }
        item = csl_json_to_zotero(csl, make_template)
        assert item["creators"][0]["name"] == "Acme Corp."

    def test_editor_as_creator(self):
        csl = {
            "type": "book", "title": "t",
            "editor": [{"given": "E", "family": "Edit"}],
        }
        item = csl_json_to_zotero(csl, make_template)
        assert item["creators"][0]["creatorType"] == "editor"

    def test_keyword_list(self):
        csl = {"type": "article-journal", "title": "t",
               "keyword": ["alpha", "beta"]}
        item = csl_json_to_zotero(csl, make_template)
        assert [t["tag"] for t in item["tags"]] == ["alpha", "beta"]

    def test_unknown_type_falls_back_to_document(self):
        csl = {"type": "song-and-dance", "title": "t"}
        item = csl_json_to_zotero(csl, make_template)
        assert item["itemType"] == "document"

    def test_unmapped_field_to_extra(self):
        csl = {"type": "article-journal", "title": "t",
               "custom-field": "custom-value"}
        item = csl_json_to_zotero(csl, make_template)
        assert "custom-field: custom-value" in item["extra"]

    def test_report_number(self):
        csl = {"type": "report", "title": "t", "number": "R-42"}
        item = csl_json_to_zotero(csl, make_template)
        assert item["reportNumber"] == "R-42"


# ---------------------------------------------------------------------------
# Crossref CSL JSON — recorded provider responses
#
# Crossref's transform endpoint is the most common source of CSL JSON in
# practice, and it differs from the spec in two ways that used to break the
# converter outright: it keeps Crossref's own `type` vocabulary, and it sends
# ISSN/ISBN as arrays. Every CSL fixture above is hand-written in spec form,
# so none of them ever exercised either shape. See tests/fixtures/README.md.
# ---------------------------------------------------------------------------

class TestCrossrefCslTransform:
    def test_journal_article_keeps_every_mapped_field(self):
        csl = load_fixture("crossref_csl_journal_article.json")
        # The two shapes under test, asserted so a refreshed fixture that lost
        # them fails here rather than silently making the test vacuous.
        assert csl["type"] == "journal-article"
        assert csl["ISSN"] == ["0883-9026"]

        item = csl_json_to_zotero(csl, schema_template)

        assert item["itemType"] == "journalArticle"
        assert item["publicationTitle"] == "Journal of Business Venturing"
        assert item["volume"] == "35"
        assert item["issue"] == "1"
        assert item["pages"] == "105970"
        assert item["ISSN"] == "0883-9026"
        assert item["DOI"] == "10.1016/j.jbusvent.2019.105970"
        assert item["date"] == "2020-01"
        assert item["creators"][0] == {
            "creatorType": "author",
            "firstName": "Evan J.",
            "lastName": "Douglas",
        }

    def test_book_chapter_keeps_every_mapped_field(self):
        csl = load_fixture("crossref_csl_book_chapter.json")
        assert csl["type"] == "book-chapter"
        assert csl["ISBN"] == ["9781607320395"]

        item = csl_json_to_zotero(csl, schema_template)

        assert item["itemType"] == "bookSection"
        assert item["bookTitle"] == "The Archaeology of Class War"
        assert item["pages"] == "161-185"
        assert item["ISBN"] == "9781607320395"
        assert item["publisher"] == "University of Colorado Press"
        assert item["DOI"] == "10.5876/9781607320395.c008"
        assert [(c["creatorType"], c["lastName"]) for c in item["creators"]] == [
            ("author", "Chicone"), ("editor", "Larkin"), ("editor", "McGuire"),
        ]

    @pytest.mark.parametrize("fixture", [
        "crossref_csl_journal_article.json",
        "crossref_csl_book_chapter.json",
    ])
    def test_converts_without_raising(self, fixture):
        # Before the array coercion this raised AttributeError on .strip(),
        # so no field assertion above was reachable at all.
        assert csl_json_to_zotero(load_fixture(fixture), schema_template)


class TestCrossrefTypeVocabulary:
    """Crossref spells its types differently from the CSL spec.

    An unmapped type is not merely mislabelled: it resolves to `document`,
    whose template has no publicationTitle/volume/issue/pages, and
    `_set_if_in_template` then drops those values without a word.
    """

    @pytest.mark.parametrize("csl_type,expected", [
        ("journal-article", "journalArticle"),
        ("proceedings-article", "conferencePaper"),
        ("book-chapter", "bookSection"),
        ("book-part", "bookSection"),
        ("book-section", "bookSection"),
        ("reference-entry", "dictionaryEntry"),
        ("monograph", "book"),
        ("edited-book", "book"),
        ("reference-book", "book"),
        ("report-component", "report"),
        ("dissertation", "thesis"),
        ("posted-content", "preprint"),
        # Spelled the same in both vocabularies.
        ("book", "book"),
        ("report", "report"),
    ])
    def test_crossref_type_maps(self, csl_type, expected):
        item = csl_json_to_zotero({"type": csl_type, "title": "t"}, schema_template)
        assert item["itemType"] == expected

    def test_spec_types_still_win_their_own_spelling(self):
        # The aliases are additions, not replacements.
        assert csl_json_to_zotero(
            {"type": "article-journal", "title": "t"}, schema_template
        )["itemType"] == "journalArticle"
        assert csl_json_to_zotero(
            {"type": "chapter", "title": "t"}, schema_template
        )["itemType"] == "bookSection"

    def test_journal_article_container_reaches_publication_title(self):
        item = csl_json_to_zotero({
            "type": "journal-article", "title": "t",
            "container-title": "Nature", "volume": "42",
            "issue": "7", "page": "1-10",
        }, schema_template)
        assert item["publicationTitle"] == "Nature"
        assert (item["volume"], item["issue"], item["pages"]) == ("42", "7", "1-10")


class TestCslTextCoercion:
    def test_array_takes_first_element(self):
        assert _csl_text(["0883-9026", "1873-2003"]) == "0883-9026"

    def test_array_skips_empty_entries(self):
        assert _csl_text(["", "   ", "real"]) == "real"

    def test_empty_array_is_empty_string(self):
        assert _csl_text([]) == ""

    def test_none_is_empty_string(self):
        assert _csl_text(None) == ""

    def test_string_is_trimmed(self):
        assert _csl_text("  padded  ") == "padded"

    def test_number_becomes_string(self):
        assert _csl_text(42) == "42"

    def test_array_of_numbers(self):
        assert _csl_text([42]) == "42"

    def test_title_array_from_plain_works_endpoint(self):
        # /works/{doi} — as opposed to its CSL transform — arrays title and
        # container-title too.
        item = csl_json_to_zotero({
            "type": "journal-article",
            "title": ["A Paper"],
            "container-title": ["Journal of Things"],
        }, schema_template)
        assert item["title"] == "A Paper"
        assert item["publicationTitle"] == "Journal of Things"

    def test_numeric_volume_and_issue(self):
        item = csl_json_to_zotero(
            {"type": "article-journal", "title": "t", "volume": 35, "issue": 1},
            schema_template,
        )
        assert item["volume"] == "35"
        assert item["issue"] == "1"


# ---------------------------------------------------------------------------
# coerce_csl_json_input
# ---------------------------------------------------------------------------

class TestCoerceCslInput:
    def test_accepts_json_string(self):
        out = coerce_csl_json_input('[{"type":"article-journal","title":"t"}]')
        assert out == [{"type": "article-journal", "title": "t"}]

    def test_accepts_single_object_string(self):
        out = coerce_csl_json_input('{"type":"book","title":"t"}')
        assert out == [{"type": "book", "title": "t"}]

    def test_accepts_dict(self):
        out = coerce_csl_json_input({"type": "book"})
        assert out == [{"type": "book"}]

    def test_accepts_list(self):
        out = coerce_csl_json_input([{"type": "book"}, {"type": "article-journal"}])
        assert len(out) == 2

    def test_empty_string_returns_empty(self):
        assert coerce_csl_json_input("") == []
        assert coerce_csl_json_input("   ") == []

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            coerce_csl_json_input("{not valid")

    def test_wrong_type_raises(self):
        with pytest.raises(ValueError):
            coerce_csl_json_input(42)


# ---------------------------------------------------------------------------
# merge_tags
# ---------------------------------------------------------------------------

class TestMergeTags:
    def test_merges_and_preserves_order(self):
        assert merge_tags(["a", "b"], ["c"]) == ["a", "b", "c"]

    def test_deduplicates_case_insensitive(self):
        assert merge_tags(["Alpha"], ["alpha", "beta"]) == ["Alpha", "beta"]

    def test_strips_whitespace(self):
        assert merge_tags(["  a  ", ""], [" b "]) == ["a", "b"]

    def test_empty_inputs(self):
        assert merge_tags([], []) == []
        assert merge_tags(None, None) == []


# ---------------------------------------------------------------------------
# Container types (#465 follow-up)
# ---------------------------------------------------------------------------

class TestCrossrefContainerTypes:
    """Publishers register ordinary articles under CrossRef's container
    types. A "journal-issue" that carries a title, volume, issue and page
    range is an article, and the "document" fallthrough has nowhere to put
    any of those fields.

    Found in a real library: DOI 10.13165/vpa-20-19-2-04 is an article its
    publisher registered as `journal-issue`, filed as a `document` with the
    journal name, volume, issue and pages all missing.
    """

    def test_journal_issue_is_an_article(self):
        assert CSL_TYPE_MAP["journal-issue"] == "journalArticle"

    def test_journal_volume_is_an_article(self):
        assert CSL_TYPE_MAP["journal-volume"] == "journalArticle"

    def test_native_zotero_types_are_used(self):
        assert CSL_TYPE_MAP["dataset"] == "dataset"
        assert CSL_TYPE_MAP["standard"] == "standard"

    def test_book_container_spellings_agree(self):
        for name in ("book-set", "book-series", "proceedings",
                     "proceedings-series"):
            assert CSL_TYPE_MAP[name] == "book"

    def test_book_track_is_a_section(self):
        assert CSL_TYPE_MAP["book-track"] == "bookSection"

    def test_journal_issue_conversion_keeps_the_article_fields(self):
        """The end-to-end point: the fields survive the conversion."""
        csl = {
            "type": "journal-issue",
            "title": ["Determinants of Public Trust"],
            "container-title": ["Public Policy and Administration"],
            "volume": "19",
            "issue": "2",
            "page": "205-218",
            "ISSN": ["2029-2872"],
            "DOI": "10.13165/vpa-20-19-2-04",
        }
        item = csl_json_to_zotero(csl, make_template)
        assert item["itemType"] == "journalArticle"
        assert item["publicationTitle"] == "Public Policy and Administration"
        assert item["volume"] == "19"
        assert item["issue"] == "2"
        assert item["pages"] == "205-218"


# ---------------------------------------------------------------------------
# Callers who pre-flattened ISSN to work around Defect 1
# ---------------------------------------------------------------------------

class TestPreFlattenedIssnCaller:
    """A downstream caller hit Defect 1 independently and shipped a transform
    that flattens ``ISSN`` from Crossref's array to a plain string before
    calling. Their workaround must become a no-op here, not a conflict — and
    it must not be able to hide a regression in the array handling.

    The reason this needs pinning: on the unfixed code, flattening ISSN alone
    made the *call* succeed while the item came out as a ``document`` with
    publicationTitle, volume, issue and pages silently dropped, because
    Defect 2 typed it. A caller's workaround turned a loud crash into quiet
    data loss. Both spellings must now produce the identical, correct item.
    """

    def _entry(self, issn):
        return {
            "type": "journal-article",
            "title": "Eight Simple Guidelines",
            "container-title": "Organizational Research Methods",
            "DOI": "10.1177/1094428121991907",
            "volume": "25",
            "issue": "1",
            "page": "48-87",
            "ISSN": issn,
            "issued": {"date-parts": [[2022, 1]]},
        }

    def test_array_and_flattened_issn_agree(self):
        as_sent = csl_json_to_zotero(self._entry(["1094-4281", "1552-7425"]),
                                     make_template)
        pre_flattened = csl_json_to_zotero(self._entry("1094-4281"),
                                           make_template)
        assert as_sent == pre_flattened

    def test_both_keep_the_article_type_and_fields(self):
        """The regression the caller's workaround was masking."""
        for issn in (["1094-4281", "1552-7425"], "1094-4281"):
            item = csl_json_to_zotero(self._entry(issn), make_template)
            assert item["itemType"] == "journalArticle", (
                "a flattened ISSN must not route the entry to 'document'"
            )
            assert item["publicationTitle"] == "Organizational Research Methods"
            assert item["volume"] == "25"
            assert item["pages"] == "48-87"
            assert item["ISSN"] == "1094-4281"

    def test_absent_id_is_not_required(self):
        """Crossref's transform endpoint emits no ``id`` at all. Conversion
        must not depend on one being injected."""
        entry = self._entry(["1094-4281"])
        assert "id" not in entry
        item = csl_json_to_zotero(entry, make_template)
        assert item["itemType"] == "journalArticle"
        assert item["title"] == "Eight Simple Guidelines"
