"""Tests for the merged zotero_add_item tool.

zotero_add_item replaced the six zotero_add_by_* tools. It owns two
things the per-source implementations never had: source-shape detection
and the dispatch that hands the shared arguments to the right adder.
Both are tested here; the per-source behavior stays in the
test_add_by_*.py / test_add_from_file.py suites.
"""

import json

import pytest
from conftest import DummyContext

from zotero_mcp.tools import _helpers, write

# ---------------------------------------------------------------------------
# detect_source_type
# ---------------------------------------------------------------------------


class TestDetectSourceType:

    @pytest.mark.parametrize("source", [
        "10.1145/3708319",
        "doi:10.1145/3708319",
        "DOI: 10.1145/3708319".replace(" ", ""),
        "https://doi.org/10.1145/3708319",
        "http://dx.doi.org/10.1038/nature12373",
        "  10.1145/3708319  ",
    ])
    def test_doi(self, source):
        assert write.detect_source_type(source) == "doi"

    @pytest.mark.parametrize("source", [
        "https://arxiv.org/abs/2401.00001",
        "https://arxiv.org/pdf/2401.00001.pdf",
        "arXiv:2401.00001",
        "2401.00001",
        "hep-ph/9901234",
        "https://example.com/some/page",
        "http://example.com",
        "example.com/page",
        "www.example.com",
    ])
    def test_url(self, source):
        assert write.detect_source_type(source) == "url"

    @pytest.mark.parametrize("source", [
        "9780132350884",
        "978-0-13-235088-4",
        "0132350882",
        "isbn:9780132350884",
    ])
    def test_isbn(self, source):
        assert write.detect_source_type(source) == "isbn"

    def test_isbn_bearing_url_stays_a_url(self):
        """A URL is a URL under auto-detection.

        _normalize_isbn also accepts URL forms, but treating any http(s)
        source with a checksum-passing digit run as a book would misroute
        ordinary web pages. The isbn implementation still understands the
        URL form, so source_type='isbn' is the override.
        """
        assert write.detect_source_type(
            "https://isbndb.com/book/9780132350884"
        ) == "url"

    @pytest.mark.parametrize("source", [
        "@article{smith2020, title={A}}",
        "  @Book{x, title={B}}",
        "% a leading comment\n@article{smith2020, title={A}}",
        "@article(smith2020, title={A})",
    ])
    def test_inline_bibtex(self, source):
        assert write.detect_source_type(source) == "bibtex"

    @pytest.mark.parametrize("source", [
        '{"id": "smith2020", "type": "article-journal"}',
        '[{"id": "smith2020"}]',
        '  [{"id": "a"}, {"id": "b"}]',
    ])
    def test_inline_csl_json(self, source):
        assert write.detect_source_type(source) == "csl_json"

    @pytest.mark.parametrize("source", [
        "10.1145/3708319, 10.1038/nature12373",
        "10.1145/3708319,10.1038/nature12373",
        "10.1145/3708319\n10.1038/nature12373",
        "10.1145/3708319\n10.1038/nature12373\n10.1000/xyz123",
        "  10.1145/3708319 , 10.1038/nature12373  ",
        '["10.1145/3708319", "10.1038/nature12373"]',
    ])
    def test_multi_doi_batch(self, source):
        assert write.detect_source_type(source) == "doi"

    def test_single_doi_is_not_batch_shaped(self):
        # A lone DOI still goes through the ordinary single-DOI branch.
        assert write.detect_source_type("10.1145/3708319") == "doi"

    def test_csl_json_list_of_objects_stays_csl_json(self):
        # A JSON array of objects (real CSL JSON) must not be mistaken for
        # a multi-DOI batch just because it starts with '['.
        assert write.detect_source_type(
            '[{"id": "smith2020"}, {"id": "jones2021"}]'
        ) == "csl_json"

    def test_mixed_doi_and_non_doi_list_is_not_a_batch(self):
        # If any comma-separated token fails to normalize as a DOI, this
        # isn't an unambiguous multi-DOI batch — fall through (here, to the
        # "not a path either" ValueError) rather than guessing.
        with pytest.raises(ValueError):
            write.detect_source_type("10.1145/3708319, not-a-doi")


# ---------------------------------------------------------------------------
# Splitting a batch argument into tokens
# ---------------------------------------------------------------------------


class TestSplitMultiValue:
    """Commas are legal inside URLs (query strings routinely carry them) and
    inside DOIs, so splitting on one unconditionally turns a single working
    identifier into a truncated one plus a junk sibling. A line is therefore
    kept whole only when its comma-tokens don't all validate *and* the line
    itself does — splitting a line that isn't a valid identifier anyway can
    only help, and is what keeps one bad token from costing its neighbours."""

    def test_url_with_comma_in_query_string_stays_one_token(self):
        url = "https://example.com/page?ids=1,2"
        assert write._split_multi_value(url, "url", write._looks_like_url) == [url]

    def test_doi_containing_a_comma_stays_one_token(self):
        # _normalize_doi's own `10.\d{4,9}/\S+` accepts a comma in the suffix.
        doi = "10.1000/foo,bar"
        assert write._split_multi_value(
            doi, "doi", _helpers._normalize_doi
        ) == [doi]

    def test_comma_separated_dois_still_split(self):
        assert write._split_multi_value(
            "10.1145/3708319,10.1038/nature12373", "doi", _helpers._normalize_doi
        ) == ["10.1145/3708319", "10.1038/nature12373"]

    def test_newline_always_separates_even_when_a_line_is_invalid(self):
        # No identifier can contain a raw newline, so that split is never
        # ambiguous — unlike the comma, it needs no all-valid gate.
        assert write._split_multi_value(
            "https://example.com/a\nnot a url at all", "url", write._looks_like_url
        ) == ["https://example.com/a", "not a url at all"]

    def test_comma_inside_one_line_of_a_newline_batch(self):
        """The mixed case: newlines separate, but the comma inside the first
        line belongs to that URL's query string."""
        assert write._split_multi_value(
            "https://a.com/x?ids=1,2\nhttps://b.com", "url", write._looks_like_url
        ) == ["https://a.com/x?ids=1,2", "https://b.com"]

    def test_explicit_json_array_is_taken_verbatim(self):
        # The caller structured this deliberately; no gate applies.
        assert write._split_multi_value(
            '["https://example.com/a?x=1,2", "https://example.com/b"]',
            "url", write._looks_like_url,
        ) == ["https://example.com/a?x=1,2", "https://example.com/b"]

    def test_explicit_list_is_taken_verbatim(self):
        assert write._split_multi_value(
            ["https://example.com/a?x=1,2"], "url", write._looks_like_url
        ) == ["https://example.com/a?x=1,2"]

    def test_isbn_list_splits_on_validated_tokens(self):
        assert write._split_multi_value(
            "978-0-19-973581-5, 9780135957059", "isbn", _helpers._normalize_isbn
        ) == ["978-0-19-973581-5", "9780135957059"]

    def test_one_invalid_token_still_splits_when_the_line_is_not_an_identifier(self):
        """The batch promise is that one bad token never fails its
        neighbours. Two ISBNs where the second fails its checksum is not a
        valid ISBN as a whole either, so splitting is the only reading that
        can add the good one."""
        assert write._split_multi_value(
            "9780199735815, 9780132350885", "isbn", _helpers._normalize_isbn
        ) == ["9780199735815", "9780132350885"]

    def test_one_bad_url_still_splits(self):
        assert write._split_multi_value(
            "https://a.com, not a url", "url", write._looks_like_url
        ) == ["https://a.com", "not a url"]

    def test_one_bad_doi_still_splits(self):
        assert write._split_multi_value(
            "10.1145/3708319, not-a-doi", "doi", _helpers._normalize_doi
        ) == ["10.1145/3708319", "not-a-doi"]


class TestLooksLikeUrl:
    """detect_source_type's URL heuristic, extracted so the batch-split gate
    and detection can't drift apart."""

    @pytest.mark.parametrize("s", [
        "https://example.com/page",
        "http://example.com",
        "example.com/page",
        "www.example.com",
        "sub.host.co.uk/x",
    ])
    def test_accepts(self, s):
        assert write._looks_like_url(s)

    @pytest.mark.parametrize("s", [
        "2", "not a url at all", "notes.txt", "draft.tex", "",
    ])
    def test_rejects(self, s):
        assert not write._looks_like_url(s)

    @pytest.mark.parametrize("source,expected", [
        ("/Users/me/refs.bib", "bibtex"),
        ("/Users/me/refs.bibtex", "bibtex"),
        ("refs.bib", "bibtex"),
        ("/Users/me/refs.json", "csl_json"),
        ("/Users/me/refs.csljson", "csl_json"),
        ("/Users/me/paper.pdf", "file"),
        ("~/Documents/book.epub", "file"),
        ("./paper.pdf", "file"),
        ("papers/2024/paper.pdf", "file"),
        (r"C:\Users\me\paper.pdf", "file"),
        ("/Users/me/no-extension-file", "file"),
    ])
    def test_paths(self, source, expected):
        assert write.detect_source_type(source) == expected

    # -- ambiguous / ordering cases ----------------------------------------

    def test_doi_inside_url_beats_generic_url(self):
        assert write.detect_source_type(
            "https://doi.org/10.1234/abc.pdf"
        ) == "doi"

    def test_url_ending_in_json_is_a_url_not_csl_json(self):
        assert write.detect_source_type(
            "https://example.com/records/data.json"
        ) == "url"

    def test_bare_doi_is_not_read_as_a_path(self):
        # A DOI contains a '/', so ordering matters.
        assert write.detect_source_type("10.1234/some/deep/path") == "doi"

    def test_isbn_checksum_must_pass(self):
        # One digit off: not an ISBN, and not anything else either.
        with pytest.raises(ValueError):
            write.detect_source_type("9780132350885")

    def test_arbitrary_number_is_not_an_isbn(self):
        with pytest.raises(ValueError):
            write.detect_source_type("1234567890123")

    @pytest.mark.parametrize("source", ["notes.txt", "draft.tex", "readme.rst"])
    def test_bare_filename_is_not_read_as_a_host(self, source):
        """Scheme-less 'name.suffix' is only a URL when the suffix reads like
        a TLD — otherwise a stray filename would become a webpage item."""
        with pytest.raises(ValueError):
            write.detect_source_type(source)

    @pytest.mark.parametrize("source", ["", "   ", None])
    def test_empty_source_rejected(self, source):
        with pytest.raises(ValueError):
            write.detect_source_type(source)

    def test_unrecognized_source_explains_the_override(self):
        with pytest.raises(ValueError) as exc:
            write.detect_source_type("please add my paper")
        assert "source_type" in str(exc.value)


# ---------------------------------------------------------------------------
# add_item dispatch
# ---------------------------------------------------------------------------


@pytest.fixture
def calls(monkeypatch):
    """Record which adder add_item routed to, and with what kwargs."""
    recorded = {}

    def _record(name):
        def _fn(**kwargs):
            recorded["name"] = name
            recorded["kwargs"] = kwargs
            return f"called {name}"
        return _fn

    for name in ("add_by_doi", "add_by_url", "add_by_isbn",
                 "add_by_bibtex", "add_by_csl_json", "add_from_file"):
        monkeypatch.setattr(write, name, _record(name))
    return recorded


class TestAddItemDispatch:

    def test_auto_doi(self, calls):
        write.add_item(source="10.1145/3708319", ctx=DummyContext())
        assert calls["name"] == "add_by_doi"
        assert calls["kwargs"]["doi"] == "10.1145/3708319"

    def test_auto_url(self, calls):
        write.add_item(source="https://example.com/p", ctx=DummyContext())
        assert calls["name"] == "add_by_url"
        assert calls["kwargs"]["url"] == "https://example.com/p"

    def test_auto_isbn(self, calls):
        write.add_item(source="978-0-13-235088-4", ctx=DummyContext())
        assert calls["name"] == "add_by_isbn"
        assert calls["kwargs"]["isbn"] == "978-0-13-235088-4"

    def test_auto_file(self, calls):
        write.add_item(source="/Users/me/paper.pdf", ctx=DummyContext())
        assert calls["name"] == "add_from_file"
        assert calls["kwargs"]["file_path"] == "/Users/me/paper.pdf"

    def test_auto_inline_bibtex(self, calls):
        write.add_item(source="@article{a, title={T}}", ctx=DummyContext())
        assert calls["name"] == "add_by_bibtex"
        assert calls["kwargs"]["bibtex"] == "@article{a, title={T}}"
        assert "file_path" not in calls["kwargs"]

    def test_auto_bibtex_file_path(self, calls):
        write.add_item(source="/Users/me/refs.bib", ctx=DummyContext())
        assert calls["name"] == "add_by_bibtex"
        assert calls["kwargs"]["file_path"] == "/Users/me/refs.bib"

    def test_auto_inline_csl_json(self, calls):
        write.add_item(source='[{"id": "a"}]', ctx=DummyContext())
        assert calls["name"] == "add_by_csl_json"
        assert calls["kwargs"]["csl_json"] == '[{"id": "a"}]'

    def test_auto_csl_json_file_path(self, calls):
        write.add_item(source="/Users/me/refs.json", ctx=DummyContext())
        assert calls["name"] == "add_by_csl_json"
        assert calls["kwargs"]["file_path"] == "/Users/me/refs.json"

    def test_structured_csl_json_object_is_serialized(self, calls):
        entry = {"id": "smith2020", "type": "article-journal"}
        write.add_item(source=entry, ctx=DummyContext())
        assert calls["name"] == "add_by_csl_json"
        assert json.loads(calls["kwargs"]["csl_json"]) == entry

    def test_explicit_source_type_overrides_detection(self, calls):
        # Shaped like a DOI, but the caller insists it is a URL.
        write.add_item(
            source="https://doi.org/10.1234/x", source_type="url",
            ctx=DummyContext(),
        )
        assert calls["name"] == "add_by_url"

    def test_explicit_bibtex_type_treats_loose_text_as_inline(self, calls):
        write.add_item(
            source="this is not bibtex", source_type="bibtex",
            ctx=DummyContext(),
        )
        assert calls["name"] == "add_by_bibtex"
        assert calls["kwargs"]["bibtex"] == "this is not bibtex"

    def test_unknown_source_type_rejected(self, calls):
        result = write.add_item(
            source="10.1145/3708319", source_type="magic", ctx=DummyContext()
        )
        assert result.startswith("Error")
        assert "source_type" in result
        assert calls == {}

    def test_undetectable_source_returns_error(self, calls):
        result = write.add_item(source="please add my paper", ctx=DummyContext())
        assert result.startswith("Error")
        assert calls == {}

    def test_empty_source_returns_error(self, calls):
        assert write.add_item(source="", ctx=DummyContext()).startswith("Error")
        assert calls == {}


class TestAddItemSharedArguments:

    def test_shared_args_reach_the_adder(self, calls):
        write.add_item(
            source="10.1145/3708319",
            collections=["9SU943GB"],
            tags=["MCP"],
            attach_mode="required",
            if_exists="file",
            create_missing_collections=True,
            ctx=DummyContext(),
        )
        kwargs = calls["kwargs"]
        assert kwargs["collections"] == ["9SU943GB"]
        assert kwargs["tags"] == ["MCP"]
        assert kwargs["attach_mode"] == "required"
        assert kwargs["if_exists"] == "file"
        assert kwargs["create_missing_collections"] is True

    def test_isbn_route_gets_no_attach_mode(self, calls):
        # add_by_isbn has no attach_mode parameter — passing one would raise.
        write.add_item(
            source="9780132350884", attach_mode="required", ctx=DummyContext()
        )
        assert calls["name"] == "add_by_isbn"
        assert "attach_mode" not in calls["kwargs"]

    def test_title_reaches_the_file_route(self, calls):
        write.add_item(
            source="/Users/me/paper.pdf", title="Manual Title",
            ctx=DummyContext(),
        )
        assert calls["kwargs"]["title"] == "Manual Title"

    def test_title_is_not_forwarded_to_other_routes(self, calls):
        write.add_item(
            source="10.1145/3708319", title="Manual Title", ctx=DummyContext()
        )
        assert calls["name"] == "add_by_doi"
        assert "title" not in calls["kwargs"]
