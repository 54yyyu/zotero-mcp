"""Unit tests for zotero_mcp.citation_import (parse + converters)."""

import pytest

from zotero_mcp.citation_import import (
    _format_bibtex_date,
    _format_csl_date,
    _parse_bibtex_author_list,
    bibtex_entry_to_zotero,
    coerce_csl_json_input,
    csl_json_to_zotero,
    merge_tags,
    parse_bibtex,
)

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
