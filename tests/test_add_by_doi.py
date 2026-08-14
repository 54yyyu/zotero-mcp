"""Tests for the DOI source of zotero_add_item (formerly zotero_add_by_doi)."""

import pytest
from unittest.mock import MagicMock, patch

from zotero_mcp import server
from zotero_mcp.tools import write
from conftest import DummyContext, FakeZotero, fake_crossref_get


# ---------------------------------------------------------------------------
# Sample CrossRef response data
# ---------------------------------------------------------------------------

def _make_crossref_message(**overrides):
    """Build a CrossRef /works response message dict with sensible defaults."""
    msg = {
        "type": "journal-article",
        "title": ["Effects of Climate Change on Coral Reefs"],
        "container-title": ["Nature Ecology & Evolution"],
        "DOI": "10.1234/test.2024.001",
        "URL": "https://doi.org/10.1234/test.2024.001",
        "volume": "8",
        "issue": "3",
        "page": "123-145",
        "ISSN": ["1234-5678"],
        "publisher": "Nature Publishing Group",
        "abstract": "<jats:p>Coral reefs are declining due to <jats:italic>warming</jats:italic> oceans.</jats:p>",
        "published-print": {"date-parts": [[2024, 3, 15]]},
        "author": [
            {"given": "Jane", "family": "Smith", "sequence": "first"},
            {"given": "Bob", "family": "Jones", "sequence": "additional"},
        ],
    }
    msg.update(overrides)
    return msg


def _make_crossref_response(message_overrides=None, status=200):
    """Build a mock requests.Response for CrossRef API."""
    msg = _make_crossref_message(**(message_overrides or {}))
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"status": "ok", "message": msg}
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_ctx():
    return DummyContext()


@pytest.fixture
def fake_zot():
    return FakeZotero()


@pytest.fixture
def patch_write_client(monkeypatch, fake_zot):
    """Patch _get_write_client to return (fake_zot, fake_zot)."""
    monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot))


@pytest.fixture
def patch_crossref_success(monkeypatch):
    """Patch requests.get to return a successful CrossRef response."""
    resp = _make_crossref_response()
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: resp)
    return resp


# ---------------------------------------------------------------------------
# DOI Normalization (unit tests for _normalize_doi)
# ---------------------------------------------------------------------------

class TestNormalizeDoi:
    def test_bare_doi(self):
        assert server._normalize_doi("10.1234/test.001") == "10.1234/test.001"

    def test_doi_prefix(self):
        assert server._normalize_doi("doi:10.1234/test.001") == "10.1234/test.001"

    def test_doi_prefix_uppercase(self):
        assert server._normalize_doi("DOI:10.1234/test.001") == "10.1234/test.001"

    def test_https_doi_url(self):
        assert server._normalize_doi("https://doi.org/10.1234/test.001") == "10.1234/test.001"

    def test_http_dx_doi_url(self):
        assert server._normalize_doi("http://dx.doi.org/10.1234/test.001") == "10.1234/test.001"

    def test_trailing_period(self):
        assert server._normalize_doi("10.1234/test.001.") == "10.1234/test.001"

    def test_trailing_comma(self):
        assert server._normalize_doi("10.1234/test.001,") == "10.1234/test.001"

    def test_trailing_paren(self):
        assert server._normalize_doi("10.1234/test.001)") == "10.1234/test.001"

    def test_trailing_semicolon(self):
        assert server._normalize_doi("10.1234/test.001;") == "10.1234/test.001"

    def test_trailing_bracket(self):
        assert server._normalize_doi("10.1234/test.001]") == "10.1234/test.001"

    def test_url_with_trailing_punctuation(self):
        assert server._normalize_doi("https://doi.org/10.1234/test.001.") == "10.1234/test.001"

    def test_none_returns_none(self):
        assert server._normalize_doi(None) is None

    def test_empty_returns_none(self):
        assert server._normalize_doi("") is None

    def test_invalid_doi_returns_none(self):
        assert server._normalize_doi("not-a-doi") is None

    def test_url_without_doi_returns_none(self):
        assert server._normalize_doi("https://example.com/something") is None

    def test_whitespace_stripped(self):
        assert server._normalize_doi("  10.1234/test.001  ") == "10.1234/test.001"


# ---------------------------------------------------------------------------
# CrossRef Type Mapping
# ---------------------------------------------------------------------------

class TestCrossrefTypeMap:
    def test_journal_article(self):
        assert server.CROSSREF_TYPE_MAP["journal-article"] == "journalArticle"

    def test_posted_content_is_preprint(self):
        assert server.CROSSREF_TYPE_MAP["posted-content"] == "preprint"

    def test_book_chapter(self):
        assert server.CROSSREF_TYPE_MAP["book-chapter"] == "bookSection"

    def test_proceedings_article(self):
        assert server.CROSSREF_TYPE_MAP["proceedings-article"] == "conferencePaper"

    def test_dissertation(self):
        assert server.CROSSREF_TYPE_MAP["dissertation"] == "thesis"

    def test_unknown_type_falls_back_to_document(self):
        # The implementation should use .get(type, "document")
        assert server.CROSSREF_TYPE_MAP.get("unknown-type-xyz", "document") == "document"


# ---------------------------------------------------------------------------
# Happy Path: add_by_doi creates item with correct fields
# ---------------------------------------------------------------------------

class TestAddByDoiHappyPath:
    def test_creates_item_with_mapped_fields(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response()
        )

        result = write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )

        # Should have created exactly one item
        assert len(fake_zot.created) == 1
        item = fake_zot.created[0]

        assert item["itemType"] == "journalArticle"
        assert item["title"] == "Effects of Climate Change on Coral Reefs"
        assert item["DOI"] == "10.1234/test.2024.001"
        assert item["publicationTitle"] == "Nature Ecology & Evolution"
        assert item["volume"] == "8"
        assert item["issue"] == "3"
        assert item["pages"] == "123-145"
        assert item["ISSN"] == "1234-5678"

    def test_creators_mapped_with_given_family(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response()
        )

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )
        item = fake_zot.created[0]

        assert len(item["creators"]) == 2
        assert item["creators"][0]["firstName"] == "Jane"
        assert item["creators"][0]["lastName"] == "Smith"
        assert item["creators"][0]["creatorType"] == "author"
        assert item["creators"][1]["firstName"] == "Bob"
        assert item["creators"][1]["lastName"] == "Jones"

    def test_abstract_xml_stripped(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response()
        )

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )
        item = fake_zot.created[0]

        # JATS tags should be stripped
        assert "<jats:" not in item["abstractNote"]
        assert "Coral reefs are declining" in item["abstractNote"]
        assert "warming" in item["abstractNote"]

    def test_result_contains_confirmation(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response()
        )

        result = write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )
        # Result should be a string containing the title or DOI
        assert isinstance(result, str)
        assert "10.1234/test.2024.001" in result or "Coral Reefs" in result


# ---------------------------------------------------------------------------
# attach_mode semantics — 'none' skips, 'required' fails the entry
# ---------------------------------------------------------------------------

class TestAddByDoiAttachMode:
    """The DOI adder must honor attach_mode via _try_attach_oa_pdf.

    Regression tests for the bug where 'none' was silently ignored by
    add_by_doi (it always tried to download+upload a PDF) and 'required'
    was unimplemented (it silently behaved like 'auto').
    """

    def test_none_mode_creates_item_without_pdf_attempt(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response()
        )

        result = write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            attach_mode="none",
            ctx=dummy_ctx,
        )

        assert len(fake_zot.created) == 1
        assert "skipped (attach_mode=none)" in result

    def test_required_mode_fails_entry_when_no_pdf_found(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        # requests.get always returns the CrossRef shape, so every OA-source
        # lookup inside _try_attach_oa_pdf finds nothing — no PDF available.
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response()
        )

        result = write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            attach_mode="required",
            ctx=dummy_ctx,
        )

        # The item is still created (it existed before the PDF lookup ran);
        # only the reported outcome for this entry is a failure.
        assert len(fake_zot.created) == 1
        assert result.startswith("Error:")
        assert "attach_mode='required'" in result
        assert "KEY0000" in result


# ---------------------------------------------------------------------------
# Field Mapping: container-title array, ISSN array, date extraction
# ---------------------------------------------------------------------------

class TestFieldMapping:
    def test_container_title_extracted_from_array(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        """container-title is an array in CrossRef; should extract [0]."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        msg = {"container-title": ["Journal of Testing", "J Test"]}
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response(msg)
        )

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )
        item = fake_zot.created[0]
        assert item.get("publicationTitle") == "Journal of Testing"

    def test_empty_container_title_array(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        msg = {"container-title": []}
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response(msg)
        )

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )
        item = fake_zot.created[0]
        # Should not crash; publicationTitle should be empty or absent
        assert item.get("publicationTitle", "") == ""

    def test_issn_extracted_from_array(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        msg = {"ISSN": ["1111-2222", "3333-4444"]}
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response(msg)
        )

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )
        item = fake_zot.created[0]
        assert item.get("ISSN") == "1111-2222"

    def test_institutional_author(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        """Institutional authors have 'name' instead of given/family."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        msg = {
            "author": [
                {"name": "World Health Organization"},
                {"given": "Alice", "family": "Chen"},
            ]
        }
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response(msg)
        )

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )
        item = fake_zot.created[0]
        creators = item["creators"]

        assert len(creators) == 2
        # Institutional author: single-field "name" format
        institutional = creators[0]
        assert institutional["name"] == "World Health Organization"
        assert institutional["creatorType"] == "author"
        assert "firstName" not in institutional
        assert "lastName" not in institutional

        # Regular author
        assert creators[1]["firstName"] == "Alice"
        assert creators[1]["lastName"] == "Chen"

    def test_editor_creator_type(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        """CrossRef 'editor' role should map to creatorType 'editor'."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        msg = {
            "author": [],
            "editor": [
                {"given": "Ed", "family": "Itor"},
            ],
        }
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response(msg)
        )

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )
        item = fake_zot.created[0]

        editors = [c for c in item["creators"] if c["creatorType"] == "editor"]
        assert len(editors) == 1
        assert editors[0]["lastName"] == "Itor"


# ---------------------------------------------------------------------------
# Field Validation: only template-valid fields are set
# ---------------------------------------------------------------------------

class TestFieldValidation:
    def test_only_template_fields_set(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        """Fields not in the item_template should NOT appear on the item."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        # CrossRef message with fields that don't exist in journalArticle template
        msg = {
            "subject": ["Biology", "Ecology"],
            "funder": [{"name": "NSF"}],
        }
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response(msg)
        )

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )
        item = fake_zot.created[0]

        # These CrossRef fields should not be set on the Zotero item
        assert "subject" not in item
        assert "funder" not in item

    def test_preprint_type_mapping(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        """posted-content should create a preprint item type."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        msg = {"type": "posted-content"}
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response(msg)
        )

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )
        item = fake_zot.created[0]
        assert item["itemType"] == "preprint"

    def test_unknown_type_falls_back_to_document(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        msg = {"type": "totally-unknown-type"}
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response(msg)
        )

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )
        item = fake_zot.created[0]
        assert item["itemType"] == "document"


# ---------------------------------------------------------------------------
# Tags and Collections
# ---------------------------------------------------------------------------

class TestTagsAndCollections:
    def test_tags_applied(self, monkeypatch, fake_zot, dummy_ctx):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response()
        )

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            tags=["climate", "ecology"],
            ctx=dummy_ctx,
        )
        item = fake_zot.created[0]

        tag_names = [t["tag"] for t in item["tags"]]
        assert "climate" in tag_names
        assert "ecology" in tag_names

    def test_tags_as_string(self, monkeypatch, fake_zot, dummy_ctx):
        """Tags can be passed as a comma-separated string."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response()
        )

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            tags="climate, ecology",
            ctx=dummy_ctx,
        )
        item = fake_zot.created[0]

        tag_names = [t["tag"] for t in item["tags"]]
        assert "climate" in tag_names
        assert "ecology" in tag_names

    def test_collections_applied(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        fake_zot._collections = [
            {"key": "ABCD1234", "data": {"name": "Climate", "parentCollection": False}},
        ]
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response()
        )

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            collections=["ABCD1234"],
            ctx=dummy_ctx,
        )
        item = fake_zot.created[0]

        assert "ABCD1234" in item["collections"]

    def test_collection_name_resolved_to_key(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        """A collection NAME resolves to its key before the item is created."""
        fake_zot._collections = [
            {"key": "ABCD1234", "data": {"name": "Climate", "parentCollection": False}},
        ]
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response()
        )

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            collections=["climate"],
            ctx=dummy_ctx,
        )
        item = fake_zot.created[0]

        assert item["collections"] == ["ABCD1234"]

    def test_unknown_collection_fails_before_create(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        """A bad collection spec must fail the add BEFORE any item is created."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response()
        )

        result = write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            collections=["no-such-collection"],
            ctx=dummy_ctx,
        )

        assert "not found" in result
        assert fake_zot.created == []

    def test_no_tags_or_collections(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        """When tags/collections are None, item should still be created."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response()
        )

        result = write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )
        assert len(fake_zot.created) == 1


# ---------------------------------------------------------------------------
# JATS XML Stripping in Abstract
# ---------------------------------------------------------------------------

class TestAbstractXmlStripping:
    def test_nested_jats_tags_stripped(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        msg = {
            "abstract": (
                "<jats:p>This study examines <jats:italic>Drosophila</jats:italic> "
                "with <jats:bold>novel</jats:bold> methods.</jats:p>"
            )
        }
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response(msg)
        )

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )
        item = fake_zot.created[0]

        assert "<jats:" not in item["abstractNote"]
        assert "Drosophila" in item["abstractNote"]
        assert "novel" in item["abstractNote"]

    def test_html_tags_stripped(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        msg = {"abstract": "<p>Plain <b>HTML</b> abstract.</p>"}
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response(msg)
        )

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )
        item = fake_zot.created[0]

        assert "<p>" not in item["abstractNote"]
        assert "<b>" not in item["abstractNote"]
        assert "Plain HTML abstract." in item["abstractNote"]

    def test_missing_abstract_handled(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        msg = {}  # no abstract key at all
        # Remove abstract from defaults
        resp = _make_crossref_response(msg)
        del resp.json.return_value["message"]["abstract"]
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp)

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )
        item = fake_zot.created[0]

        # Should not crash; abstract should be empty
        assert item.get("abstractNote", "") == ""


# ---------------------------------------------------------------------------
# Error Cases: CrossRef 404 (DOI not found)
# ---------------------------------------------------------------------------

class TestCrossrefNotFound:
    def test_crossref_404_returns_error(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        resp_404 = MagicMock()
        resp_404.status_code = 404
        resp_404.raise_for_status.side_effect = Exception("HTTP 404")
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp_404)

        result = write.add_item(
            source="10.1234/nonexistent",
            source_type="doi",
            ctx=dummy_ctx,
        )

        assert len(fake_zot.created) == 0
        assert isinstance(result, str)
        # Should mention that the DOI was not found
        result_lower = result.lower()
        assert "not found" in result_lower or "404" in result_lower or "error" in result_lower

    def test_invalid_doi_format_returns_error(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )

        result = write.add_item(source="not-a-real-doi", source_type="doi", ctx=dummy_ctx)

        assert len(fake_zot.created) == 0
        assert isinstance(result, str)
        result_lower = result.lower()
        assert "invalid" in result_lower or "error" in result_lower or "not" in result_lower


# ---------------------------------------------------------------------------
# Error Cases: CrossRef Timeout
# ---------------------------------------------------------------------------

class TestCrossrefTimeout:
    def test_crossref_timeout_returns_error(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        import requests as req_lib

        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )

        def timeout_get(*args, **kwargs):
            raise req_lib.exceptions.Timeout("Connection timed out")

        monkeypatch.setattr("requests.get", timeout_get)

        result = write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )

        assert len(fake_zot.created) == 0
        assert isinstance(result, str)
        result_lower = result.lower()
        assert "timeout" in result_lower or "error" in result_lower

    def test_crossref_connection_error(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        import requests as req_lib

        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )

        def conn_error(*args, **kwargs):
            raise req_lib.exceptions.ConnectionError("Network unreachable")

        monkeypatch.setattr("requests.get", conn_error)

        result = write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )

        assert len(fake_zot.created) == 0
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Hybrid Mode / Local-Only Rejection
# ---------------------------------------------------------------------------

class TestHybridMode:
    def test_local_only_mode_returns_error(
        self, monkeypatch, dummy_ctx
    ):
        """In local-only mode (no web credentials), should return an error."""
        def raise_local_only(ctx):
            raise ValueError(
                "Cannot perform write operations in local-only mode. "
                "Add ZOTERO_API_KEY and ZOTERO_LIBRARY_ID to enable hybrid mode."
            )

        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", raise_local_only)

        result = write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )

        assert isinstance(result, str)
        assert "local-only" in result.lower() or "write" in result.lower()

    def test_hybrid_mode_uses_web_for_write(
        self, monkeypatch, dummy_ctx
    ):
        """In hybrid mode, write_zot should be the web client."""
        read_zot = FakeZotero()
        write_zot = FakeZotero()
        write_zot.library_id = "web-99999"

        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (read_zot, write_zot)
        )
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _make_crossref_response()
        )

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )

        # Item should be created via write_zot, not read_zot
        assert len(write_zot.created) == 1
        assert len(read_zot.created) == 0


# ---------------------------------------------------------------------------
# User-Agent Header Sent to CrossRef
# ---------------------------------------------------------------------------

class TestCrossrefUserAgent:
    def test_user_agent_header_sent(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )

        captured_kwargs = {}

        def capture_get(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return _make_crossref_response()

        monkeypatch.setattr("requests.get", capture_get)

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )

        headers = captured_kwargs.get("headers", {})
        assert "User-Agent" in headers
        assert "zotero-mcp" in headers["User-Agent"]

    def test_timeout_passed_to_requests(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )

        captured_kwargs = {}

        def capture_get(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return _make_crossref_response()

        monkeypatch.setattr("requests.get", capture_get)

        write.add_item(
            source="10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )

        # Should set a timeout (15s per the plan)
        assert "timeout" in captured_kwargs
        assert captured_kwargs["timeout"] > 0


# ---------------------------------------------------------------------------
# DOI normalization applied by add_by_doi itself
# ---------------------------------------------------------------------------

class TestDoiNormalizationInAddByDoi:
    def test_doi_url_normalized_before_request(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        """A DOI passed as URL should be normalized before calling CrossRef."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )

        captured_args = []

        def capture_get(url, *args, **kwargs):
            captured_args.append(url)
            return _make_crossref_response()

        monkeypatch.setattr("requests.get", capture_get)

        write.add_item(
            source="https://doi.org/10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )

        # The first request (CrossRef) should use the bare DOI
        assert len(captured_args) >= 1
        assert "10.1234/test.2024.001" in captured_args[0]
        assert "api.crossref.org" in captured_args[0]

    def test_doi_prefix_stripped(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )

        captured_args = []

        def capture_get(url, *args, **kwargs):
            captured_args.append(url)
            return _make_crossref_response()

        monkeypatch.setattr("requests.get", capture_get)

        write.add_item(
            source="doi:10.1234/test.2024.001",
            source_type="doi",
            ctx=dummy_ctx,
        )

        assert len(captured_args) >= 1
        assert "doi%3A" not in captured_args[0].lower()
        assert "doi:10" not in captured_args[0]


# ---------------------------------------------------------------------------
# Multiple DOIs in one call
# ---------------------------------------------------------------------------

class TestMultipleDois:
    def test_creates_multiple_items(self, monkeypatch, fake_zot, dummy_ctx):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )

        messages = {
            "10.1111/a": _make_crossref_message(DOI="10.1111/a", title=["Paper A"]),
            "10.2222/b": _make_crossref_message(DOI="10.2222/b", title=["Paper B"]),
        }
        monkeypatch.setattr("requests.get", fake_crossref_get(messages.get))

        result = write.add_item(
            source="10.1111/a, 10.2222/b", source_type="doi", ctx=dummy_ctx,
        )

        assert len(fake_zot.created) == 2
        assert fake_zot.created[0]["title"] == "Paper A"
        assert fake_zot.created[1]["title"] == "Paper B"
        assert "# Added 2 of 2 DOIs" in result
        assert "Successfully added: **Paper A**" in result
        assert "Successfully added: **Paper B**" in result

    def test_auto_detected_from_newline_separated_list(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        """source_type='auto' should route a newline-separated DOI list to
        the batch path without an explicit source_type='doi' override."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr(
            "requests.get",
            fake_crossref_get(lambda doi: _make_crossref_message(DOI=doi)),
        )

        result = write.add_item(source="10.1111/a\n10.2222/b", ctx=dummy_ctx)

        assert len(fake_zot.created) == 2
        assert "# Added 2 of 2 DOIs" in result

    def test_partial_failure_reports_both(self, monkeypatch, fake_zot, dummy_ctx):
        """One 404 alongside one success: the successful DOI is still
        created and the failure is reported, not silently dropped."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )

        messages = {"10.1111/a": _make_crossref_message(DOI="10.1111/a",
                                                        title=["Paper A"])}
        monkeypatch.setattr("requests.get", fake_crossref_get(messages.get))

        result = write.add_item(
            source="10.1111/a, 10.9999/missing", source_type="doi", ctx=dummy_ctx,
        )

        assert len(fake_zot.created) == 1
        assert fake_zot.created[0]["title"] == "Paper A"
        assert "Successfully added: **Paper A**" in result
        assert "DOI not found on CrossRef: 10.9999/missing" in result


# ---------------------------------------------------------------------------
# CrossRef metadata reaches the OA-PDF cascade
# ---------------------------------------------------------------------------

_ARXIV_PREPRINT_RELATION = {
    "relation": {"has-preprint": [{"id-type": "arxiv", "id": "2307.02743"}]}
}


class TestCrossrefMetadataReachesTheCascade:
    """_try_attach_oa_pdf's four-source cascade includes "arXiv (via
    CrossRef)", which reads the has-preprint relation out of the CrossRef
    message the DOI was just fetched with. If that message isn't forwarded,
    the source is not merely unused — it can never fire, and DOIs whose only
    open copy is the referenced preprint stop attaching.
    """

    @pytest.fixture
    def spy_attach(self, monkeypatch):
        calls = []

        def _spy(write_zot, item_key, doi, ctx, crossref_metadata=None,
                 attach_mode="auto"):
            calls.append({"doi": doi, "crossref_metadata": crossref_metadata})
            return "no OA PDF found"

        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._try_attach_oa_pdf", _spy
        )
        return calls

    def test_single_doi_forwards_the_fetched_message(
        self, monkeypatch, fake_zot, dummy_ctx, spy_attach
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        msg = _make_crossref_message(DOI="10.1111/a", **_ARXIV_PREPRINT_RELATION)
        monkeypatch.setattr("requests.get", fake_crossref_get(lambda d: msg))

        write.add_item(source="10.1111/a", source_type="doi", ctx=dummy_ctx)

        assert len(spy_attach) == 1
        assert spy_attach[0]["crossref_metadata"] == msg

    def test_batch_forwards_each_entrys_own_message(
        self, monkeypatch, fake_zot, dummy_ctx, spy_attach
    ):
        """The batched create pass must pair each created item with its own
        CrossRef message, not the first one or none at all."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        messages = {
            "10.1111/a": _make_crossref_message(
                DOI="10.1111/a", title=["Paper A"], **_ARXIV_PREPRINT_RELATION),
            "10.2222/b": _make_crossref_message(DOI="10.2222/b", title=["Paper B"]),
        }
        monkeypatch.setattr("requests.get", fake_crossref_get(messages.get))

        write.add_item(
            source="10.1111/a, 10.2222/b", source_type="doi", ctx=dummy_ctx,
        )

        by_doi = {c["doi"]: c["crossref_metadata"] for c in spy_attach}
        assert by_doi["10.1111/a"] == messages["10.1111/a"]
        assert by_doi["10.2222/b"] == messages["10.2222/b"]

    def test_arxiv_preprint_is_actually_attached(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        """End to end: Unpaywall/S2/PMC find nothing, so the arXiv leg is the
        only source that can produce a PDF."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        msg = _make_crossref_message(DOI="10.1111/a", **_ARXIV_PREPRINT_RELATION)
        monkeypatch.setattr("requests.get", fake_crossref_get(lambda d: msg))
        for source in ("_try_unpaywall", "_try_semantic_scholar", "_try_pmc"):
            monkeypatch.setattr(
                f"zotero_mcp.tools._helpers.{source}", lambda *a, **kw: None
            )
        downloaded = []
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._download_and_attach_pdf",
            lambda zot, key, url, doi, ctx: downloaded.append(url) or "",
        )

        result = write.add_item(
            source="10.1111/a", source_type="doi", attach_mode="required",
            ctx=dummy_ctx,
        )

        assert downloaded == ["https://arxiv.org/pdf/2307.02743.pdf"]
        assert "attach_mode='required'" not in result

    def test_bibtex_still_attaches_without_any_crossref_message(
        self, monkeypatch, fake_zot, dummy_ctx, spy_attach
    ):
        """_create_and_attach_batch is shared with the bibtex/CSL-JSON
        importers, which have no CrossRef message to forward."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )

        write.add_item(
            source="@article{k, title={T}, doi={10.1111/a}, year={2024}}",
            source_type="bibtex",
            ctx=dummy_ctx,
        )

        assert len(spy_attach) == 1
        assert spy_attach[0]["crossref_metadata"] is None


# ---------------------------------------------------------------------------
# Repeated identifiers within one call
# ---------------------------------------------------------------------------

class TestRepeatedIdentifiers:
    """Dedup against the library runs in phase 1, before anything has been
    created, so it cannot see an item this same call is about to add. Without
    collapsing repeats up front, the same DOI listed twice was created twice
    even under if_exists='skip'."""

    def test_same_doi_three_times_creates_one_item(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr(
            "requests.get",
            fake_crossref_get(lambda doi: _make_crossref_message(DOI=doi)),
        )

        result = write.add_by_doi(
            doi=["10.1111/a", "10.1111/a", "10.1111/a"], ctx=dummy_ctx,
        )

        assert len(fake_zot.created) == 1
        # Still one line per requested token, so the caller can see what
        # happened to each thing they asked for.
        assert result.count("## ") == 3
        assert result.count("Same DOI as entry 1 in this request") == 2
        assert "# Added 1 of 3 DOIs" in result
        assert "2 repeated in this request" in result

    def test_repeats_are_matched_case_insensitively(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        """DOIs are case-insensitive, and CrossRef echoes canonical case."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr(
            "requests.get",
            fake_crossref_get(lambda doi: _make_crossref_message(DOI=doi)),
        )

        write.add_by_doi(doi=["10.1111/aBc", "10.1111/ABC"], ctx=dummy_ctx)

        assert len(fake_zot.created) == 1

    def test_only_one_crossref_fetch_for_a_repeat(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        seen = []
        inner = fake_crossref_get(lambda doi: _make_crossref_message(DOI=doi))

        def _get(url, params=None, **kwargs):
            if "crossref.org" in url:
                seen.append(url)
            return inner(url, params=params, **kwargs)

        monkeypatch.setattr("requests.get", _get)

        write.add_by_doi(doi=["10.1111/a", "10.1111/a"], ctx=dummy_ctx)

        # One DOI of real work, so the single-DOI endpoint is used rather
        # than a batch filter — the repeat costs no CrossRef round-trip.
        assert seen == ["https://api.crossref.org/works/10.1111/a"]

    def test_distinct_dois_are_untouched(self, monkeypatch, fake_zot, dummy_ctx):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr(
            "requests.get",
            fake_crossref_get(lambda doi: _make_crossref_message(DOI=doi)),
        )

        result = write.add_by_doi(doi=["10.1111/a", "10.2222/b"], ctx=dummy_ctx)

        assert len(fake_zot.created) == 2
        assert "repeated in this request" not in result

    def test_invalid_tokens_each_get_their_own_error(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        """Two unnormalizable tokens are not "the same identifier" — there is
        no identifier to compare. Each keeps its own line."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )

        result = write.add_by_doi(doi=["nope", "nope"], ctx=dummy_ctx)

        assert len(fake_zot.created) == 0
        assert result.count("does not appear to be a valid DOI") == 2
        assert "# Added 0 of 2 DOIs" in result


class TestHonestBatchCount:
    def test_header_counts_successes_not_inputs(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        """The header said "Added N" from len(tokens) — the number asked for,
        not the number added — so a batch where everything failed still
        claimed to have added them all."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        messages = {"10.1111/a": _make_crossref_message(DOI="10.1111/a")}
        monkeypatch.setattr("requests.get", fake_crossref_get(messages.get))

        result = write.add_by_doi(
            doi=["10.1111/a", "10.9999/missing"], ctx=dummy_ctx,
        )

        assert len(fake_zot.created) == 1
        assert "# Added 1 of 2 DOIs" in result
        assert "1 failed" in result

    def test_existing_items_count_as_reused_not_added(
        self, monkeypatch, fake_zot, dummy_ctx
    ):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers.find_existing_items",
            lambda zot, **kw: (
                [{"key": "OLD1", "version": 1,
                  "data": {"itemType": "journalArticle", "DOI": "10.1111/a",
                           "title": "Old", "collections": []}}]
                if kw.get("doi") == "10.1111/a" else []
            ),
        )
        monkeypatch.setattr(
            "requests.get",
            fake_crossref_get(lambda doi: _make_crossref_message(DOI=doi)),
        )

        result = write.add_by_doi(
            doi=["10.1111/a", "10.2222/b"], if_exists="skip", ctx=dummy_ctx,
        )

        assert "# Added 1 of 2 DOIs" in result
        assert "1 already in library" in result
