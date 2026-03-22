"""Tests for shared helper functions in server.py and utils.py."""

import pytest
from unittest.mock import patch, MagicMock

from zotero_mcp import server
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
        assert write_zot.library_type == "group"

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

    def test_standard_is_document(self):
        assert server.CROSSREF_TYPE_MAP["standard"] == "document"

    def test_unknown_type_fallback(self):
        assert server.CROSSREF_TYPE_MAP.get("unknown-type", "document") == "document"
