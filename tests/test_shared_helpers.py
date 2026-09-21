"""Tests for shared helper functions in server.py and utils.py."""

import pytest
from unittest.mock import patch, MagicMock

from zotero_mcp import server
from zotero_mcp import utils as _utils
from zotero_mcp.tools import _helpers
from zotero_mcp.utils import clean_html
from conftest import DummyContext, FakeZotero


# ---------------------------------------------------------------------------
# _normalize_str_list_input
# ---------------------------------------------------------------------------

class TestNormalizeStrListInput:
    def test_none_returns_empty(self):
        assert server._normalize_str_list_input(None) == []

    def test_empty_string_returns_empty(self):
        assert server._normalize_str_list_input("") == []
        assert server._normalize_str_list_input("   ") == []

    def test_list_passthrough(self):
        assert server._normalize_str_list_input(["a", "b"]) == ["a", "b"]

    def test_list_strips_whitespace(self):
        assert server._normalize_str_list_input(["  a ", " b "]) == ["a", "b"]

    def test_list_filters_empty(self):
        assert server._normalize_str_list_input(["a", "", "  ", "b"]) == ["a", "b"]

    def test_json_list_string(self):
        assert server._normalize_str_list_input('["tag1", "tag2"]') == ["tag1", "tag2"]

    def test_json_single_string(self):
        assert server._normalize_str_list_input('"hello"') == ["hello"]

    def test_comma_separated(self):
        assert server._normalize_str_list_input("a, b, c") == ["a", "b", "c"]

    def test_single_value(self):
        assert server._normalize_str_list_input("single") == ["single"]

    def test_json_dict_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            server._normalize_str_list_input('{"not": "a-list"}')

    def test_non_string_non_list_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            server._normalize_str_list_input(42)

    def test_field_name_in_error(self):
        with pytest.raises(ValueError, match="tags"):
            server._normalize_str_list_input(42, field_name="tags")


# ---------------------------------------------------------------------------
# clean_html (with collapse_whitespace=True, replaces _strip_xml_tags)
# ---------------------------------------------------------------------------

class TestStripXmlTags:
    def test_jats_tags(self):
        assert clean_html("<jats:p>Hello <jats:italic>world</jats:italic></jats:p>", collapse_whitespace=True) == "Hello world"

    def test_html_tags(self):
        assert clean_html("<p>Hello <b>world</b></p>", collapse_whitespace=True) == "Hello world"

    def test_none_returns_empty(self):
        assert clean_html(None, collapse_whitespace=True) == ""
        assert clean_html("", collapse_whitespace=True) == ""

    def test_plain_text_unchanged(self):
        assert clean_html("No tags here", collapse_whitespace=True) == "No tags here"

    def test_whitespace_normalized(self):
        assert clean_html("a   b\n\nc", collapse_whitespace=True) == "a b c"


# ---------------------------------------------------------------------------
# _normalize_doi
# ---------------------------------------------------------------------------

class TestNormalizeDoi:
    def test_bare_doi(self):
        assert server._normalize_doi("10.1038/nphys1170") == "10.1038/nphys1170"

    def test_doi_prefix(self):
        assert server._normalize_doi("doi:10.1038/nphys1170") == "10.1038/nphys1170"

    def test_doi_url_https(self):
        assert server._normalize_doi("https://doi.org/10.1038/nphys1170") == "10.1038/nphys1170"

    def test_doi_url_http_dx(self):
        assert server._normalize_doi("http://dx.doi.org/10.1038/nphys1170") == "10.1038/nphys1170"

    def test_trailing_punctuation_stripped(self):
        assert server._normalize_doi("10.1038/nphys1170.") == "10.1038/nphys1170"
        assert server._normalize_doi("10.1038/nphys1170)") == "10.1038/nphys1170"

    def test_invalid_doi_returns_none(self):
        assert server._normalize_doi("not-a-doi") is None
        assert server._normalize_doi("") is None
        assert server._normalize_doi(None) is None

    def test_url_without_doi_returns_none(self):
        assert server._normalize_doi("https://example.com/foo") is None


# ---------------------------------------------------------------------------
# _normalize_arxiv_id
# ---------------------------------------------------------------------------

class TestNormalizeArxivId:
    def test_new_format(self):
        assert server._normalize_arxiv_id("2401.00001") == "2401.00001"

    def test_versioned(self):
        assert server._normalize_arxiv_id("2401.00001v2") == "2401.00001v2"

    def test_old_format(self):
        assert server._normalize_arxiv_id("hep-ph/9901234") == "hep-ph/9901234"

    def test_old_format_dotted_archive(self):
        """Archive names carrying a period are the legacy norm, not an edge.

        math.*, cs.*, q-bio.* and friends spell their subcategory after a dot
        (arXiv's own taxonomy keys), so ``math.GT/0309136`` is as valid as
        ``hep-ph/9901234``. A dotted name that fails to normalize is not
        merely rejected here: it also drops the item out of arXiv dedup
        (``_arxiv_identity``), so re-adding it duplicates the library entry.
        """
        assert server._normalize_arxiv_id("math.GT/0309136") == "math.GT/0309136"
        assert server._normalize_arxiv_id("cs.LG/0105021") == "cs.LG/0105021"
        assert server._normalize_arxiv_id("q-bio.BM/0301001") == "q-bio.BM/0301001"

    def test_old_format_dotted_archive_with_dash_subcategory(self):
        # cond-mat's own subcategories combine both separators.
        assert server._normalize_arxiv_id("cond-mat.mtrl-sci/9901001") == (
            "cond-mat.mtrl-sci/9901001"
        )

    def test_old_format_dotted_versioned(self):
        assert server._normalize_arxiv_id("math.GT/0309136v1") == "math.GT/0309136v1"

    def test_arxiv_prefix_on_dotted_id(self):
        assert server._normalize_arxiv_id("arXiv:math.GT/0309136") == "math.GT/0309136"

    def test_abs_url_dotted_id(self):
        assert server._normalize_arxiv_id("https://arxiv.org/abs/math.GT/0309136") == (
            "math.GT/0309136"
        )

    def test_legacy_archive_name_cannot_start_a_match_without_slash(self):
        assert server._normalize_arxiv_id("math.GT.0309136") is None
        assert server._normalize_arxiv_id("math.GT/03091") is None

    def test_arxiv_prefix(self):
        assert server._normalize_arxiv_id("arXiv:2401.00001") == "2401.00001"

    def test_abs_url(self):
        assert server._normalize_arxiv_id("https://arxiv.org/abs/2401.00001") == "2401.00001"

    def test_pdf_url(self):
        assert server._normalize_arxiv_id("https://arxiv.org/pdf/2401.00001.pdf") == "2401.00001"

    def test_invalid_returns_none(self):
        assert server._normalize_arxiv_id("not-an-id") is None
        assert server._normalize_arxiv_id("") is None
        assert server._normalize_arxiv_id(None) is None


# ---------------------------------------------------------------------------
# _arxiv_identity
# ---------------------------------------------------------------------------

class TestArxivIdentity:
    """The version-independent identity used for deduplication.

    Distinct from _normalize_arxiv_id, which keeps the version because its
    callers use the result to fetch a specific version from arXiv.
    """

    def test_strips_the_version(self):
        assert _helpers._arxiv_identity("2401.00001v2") == "2401.00001"
        assert _helpers._arxiv_identity("https://arxiv.org/abs/2401.00001v11") == "2401.00001"
        assert _helpers._arxiv_identity("hep-ph/9901234v2") == "hep-ph/9901234"

    def test_accepts_the_datacite_doi(self):
        assert _helpers._arxiv_identity("10.48550/arXiv.2401.00001") == "2401.00001"
        assert _helpers._arxiv_identity(
            "https://doi.org/10.48550/arXiv.2401.00001v3"
        ) == "2401.00001"

    def test_every_spelling_of_one_paper_agrees(self):
        spellings = [
            "2401.00001",
            "2401.00001v1",
            "arXiv:2401.00001",
            "http://arxiv.org/abs/2401.00001",
            "https://arxiv.org/abs/2401.00001v2",
            "https://arxiv.org/pdf/2401.00001v3.pdf",
            "10.48550/arXiv.2401.00001",
        ]
        assert {_helpers._arxiv_identity(s) for s in spellings} == {"2401.00001"}

    def test_distinct_papers_stay_distinct(self):
        assert _helpers._arxiv_identity("2401.00001") != _helpers._arxiv_identity("2401.00002")

    def test_non_arxiv_returns_none(self):
        assert _helpers._arxiv_identity("10.1038/nature12373") is None
        assert _helpers._arxiv_identity("https://example.com/foo") is None
        assert _helpers._arxiv_identity("not-an-id") is None
        assert _helpers._arxiv_identity("") is None
        assert _helpers._arxiv_identity(None) is None


# ---------------------------------------------------------------------------
# _title_search_query
# ---------------------------------------------------------------------------

class TestTitleSearchQuery:
    """Normalization of a fetched title into a quick-search query.

    Zotero's quick search ANDs whitespace-separated tokens, so anything the
    source added that the stored title lacks — JATS tags, XML entities — is
    a token that matches nothing and zeroes the whole lookup.
    """

    def test_strips_jats_markup(self):
        assert _helpers._title_search_query(
            "Growth of <i>Escherichia coli</i> at <sub>4</sub>C"
        ) == "Growth of Escherichia coli at 4 C"

    def test_markup_inside_a_word_splits_it(self):
        """Every token must occur in the stored title, marked up or not.

        Measured on the Web API: an item stored as 'DREAM<sub>(D)</sub>: …'
        was not found by 'DREAM(D): …', because deleting the tags glues a
        token the stored spelling does not contain. A space keeps each piece
        a substring of both spellings.
        """
        query = _helpers._title_search_query("DREAM<sub>(D)</sub>: adaptive MCMC")
        for stored in ("DREAM<sub>(D)</sub>: adaptive MCMC", "DREAM(D): adaptive MCMC"):
            assert all(t.lower() in stored.lower() for t in query.split()), (
                stored, query)

    def test_resolves_xml_entities(self):
        assert _helpers._title_search_query("Ethics &amp; Society") == "Ethics & Society"
        assert _helpers._title_search_query("A &lt; B") == "A < B"

    def test_escaped_tags_survive_as_literal_text(self):
        """'&lt;i&gt;' is text in a title, not markup — stripping order matters."""
        assert _helpers._title_search_query("The &lt;i&gt; Element") == "The <i> Element"

    def test_strips_the_markup_the_crossref_mapping_keeps(self):
        """The DOI path's title has been through the CrossRef repairs already.

        strip_unsupported_markup keeps the markup Zotero renders, and spells
        small caps as a styled span, so those tags still reach the query.
        """
        mapped = _utils.repair_crossref_string(_utils.strip_unsupported_markup(
            "<scp>DNA</scp> repair in <i>E. coli</i> &amp; CO<sub>2</sub>"
        ))
        assert "<i>" in mapped and "<span" in mapped
        assert _helpers._title_search_query(mapped) == "DNA repair in E. coli & CO 2"

    def test_a_decoded_angle_bracket_is_not_a_tag(self):
        """By the time the DOI path's title gets here, '&lt;' is already '<'.

        A '<' not followed by a letter is text, so the words between it and
        a later '>' stay in the query rather than being read as one tag.
        """
        mapped = _utils.repair_crossref_string(
            "Effects of &lt;10 Hz stimulation on theta &gt; baseline"
        )
        assert _helpers._title_search_query(mapped) == (
            "Effects of <10 Hz stimulation on theta > baseline"
        )

    def test_collapses_arxiv_wrap_whitespace(self):
        assert _helpers._title_search_query(
            "Chain-of-Thought Prompting Elicits Reasoning   in Large Language Models"
        ) == "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
        assert _helpers._title_search_query("  padded  title \n here ") == "padded title here"

    def test_a_clean_title_is_returned_unchanged(self):
        assert _helpers._title_search_query("RL's Razor") == "RL's Razor"

    def test_nothing_usable_returns_none(self):
        assert _helpers._title_search_query("") is None
        assert _helpers._title_search_query(None) is None
        assert _helpers._title_search_query("   ") is None
        assert _helpers._title_search_query("<i></i>") is None


# ---------------------------------------------------------------------------
# _resolve_collection_names
# ---------------------------------------------------------------------------

class TestResolveCollectionNames:
    def test_resolve_single_name(self):
        zot = FakeZotero()
        zot._collections = [
            {"key": "COL001", "data": {"name": "PhD Research"}},
            {"key": "COL002", "data": {"name": "Other"}},
        ]
        result = server._resolve_collection_names(zot, ["PhD Research"])
        assert result == ["COL001"]

    def test_case_insensitive(self):
        zot = FakeZotero()
        zot._collections = [{"key": "COL001", "data": {"name": "PhD Research"}}]
        result = server._resolve_collection_names(zot, ["phd research"])
        assert result == ["COL001"]

    def test_multiple_names(self):
        zot = FakeZotero()
        zot._collections = [
            {"key": "COL001", "data": {"name": "A"}},
            {"key": "COL002", "data": {"name": "B"}},
        ]
        result = server._resolve_collection_names(zot, ["A", "B"])
        assert result == ["COL001", "COL002"]

    def test_no_match_raises(self):
        zot = FakeZotero()
        zot._collections = [{"key": "COL001", "data": {"name": "Other"}}]
        with pytest.raises(ValueError, match="No collection found"):
            server._resolve_collection_names(zot, ["Nonexistent"])

    def test_duplicate_names_returns_all(self):
        zot = FakeZotero()
        zot._collections = [
            {"key": "COL001", "data": {"name": "Research"}},
            {"key": "COL002", "data": {"name": "Research"}},
        ]
        ctx = DummyContext()
        result = server._resolve_collection_names(zot, ["Research"], ctx=ctx)
        assert set(result) == {"COL001", "COL002"}

    def test_empty_list_returns_empty(self):
        zot = FakeZotero()
        assert server._resolve_collection_names(zot, []) == []


# ---------------------------------------------------------------------------
# _get_write_client
# ---------------------------------------------------------------------------

class TestGetWriteClient:
    def test_web_mode_returns_same_client(self, monkeypatch):
        fake = FakeZotero()
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)
        read_zot, write_zot = server._get_write_client(DummyContext())
        assert read_zot is write_zot
        assert read_zot is fake

    def test_hybrid_mode_different_clients(self, monkeypatch):
        local = FakeZotero()
        web = FakeZotero()
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: local)
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: True)
        monkeypatch.setattr("zotero_mcp.client.get_web_zotero_client", lambda: web)
        monkeypatch.setattr("zotero_mcp.client.get_active_library", lambda: {})
        read_zot, write_zot = server._get_write_client(DummyContext())
        assert read_zot is local
        assert write_zot is web

    def test_local_only_raises(self, monkeypatch):
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: FakeZotero())
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: True)
        monkeypatch.setattr("zotero_mcp.client.get_web_zotero_client", lambda: None)
        with pytest.raises(ValueError, match="Cannot perform write"):
            server._get_write_client(DummyContext())

    def test_library_override_propagated(self, monkeypatch):
        local = FakeZotero()
        web = FakeZotero()
        web.library_id = "personal"
        web.library_type = "user"
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: local)
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: True)
        monkeypatch.setattr("zotero_mcp.client.get_web_zotero_client", lambda: web)
        monkeypatch.setattr("zotero_mcp.client.get_active_library", lambda: {
            "library_id": "group123", "library_type": "group"
        })
        _, write_zot = server._get_write_client(DummyContext())
        assert write_zot.library_id == "group123"
        assert write_zot.library_type == "groups"

    def test_cleared_override_no_change(self, monkeypatch):
        local = FakeZotero()
        web = FakeZotero()
        web.library_id = "personal"
        web.library_type = "user"
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: local)
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: True)
        monkeypatch.setattr("zotero_mcp.client.get_web_zotero_client", lambda: web)
        monkeypatch.setattr("zotero_mcp.client.get_active_library", lambda: {})
        _, write_zot = server._get_write_client(DummyContext())
        assert write_zot.library_id == "personal"
        assert write_zot.library_type == "user"


# ---------------------------------------------------------------------------
# _handle_write_response
# ---------------------------------------------------------------------------

class TestHandleWriteResponse:
    def test_httpx_200(self):
        from conftest import _FakeResponse
        assert server._handle_write_response(_FakeResponse(200)) is True

    def test_httpx_204(self):
        from conftest import _FakeResponse
        assert server._handle_write_response(_FakeResponse(204)) is True

    def test_httpx_412_fails(self):
        from conftest import _FakeResponse
        assert server._handle_write_response(_FakeResponse(412)) is False

    def test_dict_with_success(self):
        assert server._handle_write_response({"success": {"0": "KEY"}}) is True

    def test_dict_with_empty_success(self):
        assert server._handle_write_response({"success": {}, "failed": {"0": "err"}}) is False

    def test_bool_true(self):
        assert server._handle_write_response(True) is True

    def test_bool_false(self):
        assert server._handle_write_response(False) is False

    def test_logs_error_on_failure(self):
        from conftest import _FakeResponse
        ctx = DummyContext()
        ctx.errors = []
        ctx.error = lambda msg: ctx.errors.append(msg)
        server._handle_write_response(_FakeResponse(412, "Precondition Failed"), ctx=ctx)
        assert len(ctx.errors) == 1
        assert "412" in ctx.errors[0]


# ---------------------------------------------------------------------------
# CROSSREF_TYPE_MAP
# ---------------------------------------------------------------------------

class TestCrossrefTypeMap:
    def test_journal_article(self):
        assert server.CROSSREF_TYPE_MAP["journal-article"] == "journalArticle"

    def test_preprint(self):
        assert server.CROSSREF_TYPE_MAP["posted-content"] == "preprint"

    def test_edited_book(self):
        assert server.CROSSREF_TYPE_MAP["edited-book"] == "book"

    def test_standard_uses_the_native_zotero_type(self):
        """Zotero has a `standard` item type; routing CrossRef's `standard`
        to `document` discarded the fields that type exists to hold."""
        assert server.CROSSREF_TYPE_MAP["standard"] == "standard"

    def test_dataset_uses_the_native_zotero_type(self):
        assert server.CROSSREF_TYPE_MAP["dataset"] == "dataset"

    def test_unknown_type_fallback(self):
        assert server.CROSSREF_TYPE_MAP.get("unknown-type", "document") == "document"


# ---------------------------------------------------------------------------
# _strip_unwritable_fields — pyzotero check_items whitelist omission workaround
# ---------------------------------------------------------------------------

class TestStripUnwritableFields:
    """pyzotero's check_items() rejects keys outside a hardcoded whitelist that
    omits `lastRead` (set by Zotero's PDF reader). Without stripping, any
    fetch→mutate→update on an opened attachment raises
    InvalidItemFieldsError before the request leaves the client."""

    def test_strips_last_read_from_attachment(self):
        from zotero_mcp.tools import _helpers
        item = {
            "key": "ATT1",
            "version": 3,
            "data": {
                "key": "ATT1",
                "itemType": "attachment",
                "linkMode": "imported_file",
                "filename": "paper.pdf",
                "lastRead": 1780565972,
                "tags": [],
            },
        }
        returned = _helpers._strip_unwritable_fields(item)
        assert returned is item
        assert "lastRead" not in item["data"]
        assert item["data"]["filename"] == "paper.pdf"

    def test_noop_when_field_absent(self):
        from zotero_mcp.tools import _helpers
        item = {"data": {"itemType": "journalArticle", "title": "x"}}
        _helpers._strip_unwritable_fields(item)
        assert item["data"] == {"itemType": "journalArticle", "title": "x"}

    def test_safe_when_data_missing(self):
        from zotero_mcp.tools import _helpers
        # Should not raise on a malformed item with no data dict.
        _helpers._strip_unwritable_fields({"key": "ABC"})


class TestCapitalizeName:
    """Port of Zotero.Utilities.capitalizeName, including its restraint.

    The cases below are the ones documented in Zotero's own source, so a
    divergence here is a divergence from what the browser connector would
    have produced for the same record.
    """

    @pytest.mark.parametrize("shouted,expected", [
        ("O'NEAL", "O'Neal"),
        ("o'neal", "O'Neal"),
        # Mixed case is assumed deliberate and never second-guessed.
        ("O'neal", "O'neal"),
        ("John MacGregor O'NEILL", "John MacGregor O'Neill"),
        ("martha McMiddlename WASHINGTON", "Martha McMiddlename Washington"),
        ("R SOLE", "R Sole"),
    ])
    def test_matches_zoteros_documented_cases(self, shouted, expected):
        assert _utils.capitalize_name(shouted) == expected

    def test_empty_and_non_string_pass_through(self):
        assert _utils.capitalize_name("") == ""
        assert _utils.capitalize_name(None) is None

    def test_runs_of_spaces_are_preserved(self):
        """The original splits on a single space, not on whitespace runs.
        Collapsing them is the classic way to break this port."""
        assert _utils.capitalize_name("A  B") == "A  B"

    def test_hyphenated_and_accented_names(self):
        assert _utils.capitalize_name("JEAN-LUC") == "Jean-Luc"
        assert _utils.capitalize_name("SOLÉ") == "Solé"


class TestStripUnsupportedMarkup:
    def test_inline_emphasis_survives(self):
        assert _utils.strip_unsupported_markup(
            "Growth of <i>E. coli</i>"
        ) == "Growth of <i>E. coli</i>"

    def test_mathml_tags_are_dropped_but_their_text_is_kept(self):
        assert _utils.strip_unsupported_markup(
            "<mml:math><mml:mi>T</mml:mi></mml:math>"
        ) == "T"

    def test_cdata_is_unwrapped(self):
        assert _utils.strip_unsupported_markup(
            "<jats:p><![CDATA[Raw & wild]]></jats:p>"
        ) == "Raw & wild"

    def test_small_caps_becomes_a_styled_span(self):
        assert _utils.strip_unsupported_markup("<scp>Abc</scp>") == (
            '<span style="font-variant:small-caps;">Abc</span>'
        )

    def test_attributes_are_stripped_from_kept_tags(self):
        assert _utils.strip_unsupported_markup(
            '<i class="species">E. coli</i>'
        ) == "<i>E. coli</i>"

    def test_empty_input(self):
        assert _utils.strip_unsupported_markup("") == ""


class TestRepairCrossrefString:
    """CrossRef deposits carry two kinds of damage Zotero repairs on the way in."""

    def test_xml_entities_are_decoded(self):
        assert _utils.repair_crossref_string("College A&amp;P Courses") == (
            "College A&P Courses"
        )

    def test_escaped_markup_survives_as_markup(self):
        """Real record 10.35537/10915/59006 deposits its italics escaped."""
        assert _utils.repair_crossref_string(
            "subtribu &lt;i&gt;Oxylobinae&lt;/i&gt; King &amp; Rob"
        ) == "subtribu <i>Oxylobinae</i> King & Rob"

    def test_newlines_are_dropped(self):
        """Real record 10.1021/acssynbio.9b00027 has one mid-title. A raw
        newline in a title also breaks this package's markdown headings."""
        assert _utils.repair_crossref_string("Yeast\nPromoters") == (
            "YeastPromoters"
        )

    def test_mojibake_is_repaired(self):
        """UTF-8 bytes decoded as Latin-1 and re-served as UTF-8: an en dash
        arrives as three characters (10.1057/9780230391116.0016)."""
        mangled = "pages 10\u00e2\u0080\u009320"
        assert _utils.repair_crossref_string(mangled) == "pages 10\u201320"

    def test_undecodable_control_characters_are_stripped_not_raised(self):
        assert "\u009f" not in _utils.repair_crossref_string("a\u009fb\u00ff")

    def test_clean_text_is_untouched(self):
        assert _utils.repair_crossref_string("Ordinary title") == "Ordinary title"

    def test_non_strings_pass_through(self):
        assert _utils.repair_crossref_string(None) is None
        assert _utils.repair_crossref_string(["a"]) == ["a"]
