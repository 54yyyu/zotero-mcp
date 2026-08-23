"""Reading bibliographic metadata out of a publisher page's <head>.

Journal platforms publish the article's citation on the landing page as
Highwire ``citation_*`` tags; Zotero's browser connector reads exactly these.
The fixture below is the real head of an OJS/PKP article page (the one in the
bug report), trimmed to its meta tags.
"""

from zotero_mcp.html_metadata import (
    _split_name,
    extract_embedded_metadata,
    parse_meta_tags,
)

OJS_HEAD = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="citation_journal_title" content="Public Policy and Administration"/>
<meta name="citation_journal_abbrev" content="PPA"/>
<meta name="citation_issn" content="2029-2872"/>
<meta name="citation_author" content="Mohamad Thahir Haning"/>
<meta name="citation_author_institution" content="Hasanuddin University"/>
<meta name="citation_author" content="Hasniati Hamzah"/>
<meta name="citation_author" content="Mashuri H Tahili"/>
<meta name="citation_title" content="DETERMINANTS OF PUBLIC TRUST"/>
<meta name="citation_language" content="en"/>
<meta name="citation_date" content="2020/07/07"/>
<meta name="citation_volume" content="19"/>
<meta name="citation_issue" content="2"/>
<meta name="citation_firstpage" content="205"/>
<meta name="citation_lastpage" content="218"/>
<meta name="citation_doi" content="10.13165/VPA-20-19-2-04"/>
<meta name="citation_pdf_url" content="https://ojs.example/download/5155/4799"/>
</head><body><p>ignored</p></body></html>
"""


class TestParseMetaTags:
    def test_repeated_names_are_all_kept(self):
        tags = parse_meta_tags(OJS_HEAD)
        assert len(tags["citation_author"]) == 3

    def test_stops_at_body(self):
        html = (
            '<html><head><meta name="citation_title" content="Real"></head>'
            '<body><meta name="citation_title" content="Injected"></body></html>'
        )
        assert parse_meta_tags(html)["citation_title"] == ["Real"]

    def test_malformed_markup_does_not_raise(self):
        html = '<html><head><meta name="citation_title" content="T"><<<>'
        assert parse_meta_tags(html)["citation_title"] == ["T"]

    def test_empty_document(self):
        assert parse_meta_tags("") == {}


class TestSplitName:
    def test_comma_form_is_last_first(self):
        assert _split_name("Haning, Mohamad Thahir") == ("Mohamad Thahir", "Haning")

    def test_plain_form_takes_last_token_as_surname(self):
        assert _split_name("Mohamad Thahir Haning") == ("Mohamad Thahir", "Haning")

    def test_single_token_is_all_surname(self):
        assert _split_name("Plato") == ("", "Plato")

    def test_blank(self):
        assert _split_name("   ") == ("", "")


class TestExtractEmbeddedMetadata:
    def test_reads_the_full_citation(self):
        m = extract_embedded_metadata(OJS_HEAD)
        assert m.title == "DETERMINANTS OF PUBLIC TRUST"
        assert m.publication == "Public Policy and Administration"
        assert m.volume == "19"
        assert m.issue == "2"
        assert m.pages == "205-218"
        assert m.doi == "10.13165/VPA-20-19-2-04"
        assert m.issn == "2029-2872"
        assert m.language == "en"
        assert m.authors == [
            ("Mohamad Thahir", "Haning"),
            ("Hasniati", "Hamzah"),
            ("Mashuri H", "Tahili"),
        ]

    def test_slash_dates_are_normalized(self):
        assert extract_embedded_metadata(OJS_HEAD).date == "2020-07-07"

    def test_recognized_as_an_article(self):
        m = extract_embedded_metadata(OJS_HEAD)
        assert m.is_usable()
        assert m.looks_like_article()
        assert not m.looks_like_chapter()

    def test_single_page_is_not_rendered_as_a_range(self):
        html = ('<head><meta name="citation_firstpage" content="7">'
                '<meta name="citation_lastpage" content="7"></head>')
        assert extract_embedded_metadata(html).pages == "7"

    def test_one_ended_page_range(self):
        html = '<head><meta name="citation_firstpage" content="7"></head>'
        assert extract_embedded_metadata(html).pages == "7"

    def test_doi_url_form_is_reduced_to_the_doi(self):
        html = ('<head><meta name="citation_doi" '
                'content="https://doi.org/10.1000/abc.123"></head>')
        assert extract_embedded_metadata(html).doi == "10.1000/abc.123"

    def test_dublin_core_fallback(self):
        html = ('<head><meta name="DC.Title" content="A DC Paper">'
                '<meta name="DC.Creator" content="Smith, Jane">'
                '<meta name="DC.Publisher" content="Some Press"></head>')
        m = extract_embedded_metadata(html)
        assert m.title == "A DC Paper"
        assert m.publisher == "Some Press"
        assert m.authors == [("Jane", "Smith")]

    def test_highwire_wins_over_dublin_core(self):
        html = ('<head><meta name="citation_title" content="Highwire">'
                '<meta name="DC.Title" content="Dublin"></head>')
        assert extract_embedded_metadata(html).title == "Highwire"

    def test_open_graph_is_a_last_resort_title(self):
        html = '<head><meta property="og:title" content="OG Title"></head>'
        assert extract_embedded_metadata(html).title == "OG Title"

    def test_book_chapter_shape(self):
        html = ('<head><meta name="citation_inbook_title" content="A Handbook">'
                '<meta name="citation_isbn" content="9780521195621">'
                '<meta name="citation_title" content="Chapter Seven"></head>')
        m = extract_embedded_metadata(html)
        assert m.looks_like_chapter()
        assert not m.looks_like_article()

    def test_duplicate_authors_are_collapsed(self):
        html = ('<head><meta name="citation_author" content="Smith, Jane">'
                '<meta name="DC.Creator" content="Jane Smith"></head>')
        assert extract_embedded_metadata(html).authors == [("Jane", "Smith")]

    def test_semicolon_packed_authors(self):
        html = ('<head><meta name="citation_authors" '
                'content="Smith, Jane; Doe, John"></head>')
        assert extract_embedded_metadata(html).authors == [
            ("Jane", "Smith"), ("John", "Doe"),
        ]

    def test_page_with_no_metadata_is_not_usable(self):
        html = "<head><title>Just a page</title></head>"
        assert not extract_embedded_metadata(html).is_usable()
