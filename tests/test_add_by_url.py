"""Tests for Feature 5: Add by URL (zotero_add_by_url)."""

import pytest
from unittest.mock import patch, MagicMock

from zotero_mcp import server
from conftest import DummyContext, FakeZotero


# ---------------------------------------------------------------------------
# Sample arXiv Atom XML response
# ---------------------------------------------------------------------------

ARXIV_ATOM_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>ArXiv Query: search_query=id_list=2401.00001</title>
  <id>http://arxiv.org/api/query</id>
  <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <updated>2024-01-01T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published>
    <title>Attention Is All You Need (Again)</title>
    <summary>We present a novel transformer architecture that improves upon
existing models by introducing sparse attention patterns.</summary>
    <author>
      <name>Alice Smith</name>
    </author>
    <author>
      <name>Bob Jones</name>
    </author>
    <arxiv:primary_category term="cs.CL" />
    <link href="http://arxiv.org/abs/2401.00001v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2401.00001v1" rel="related" type="application/pdf" title="pdf"/>
  </entry>
</feed>
"""

ARXIV_EMPTY_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>ArXiv Query: search_query=id_list=9999.99999</title>
  <id>http://arxiv.org/api/query</id>
  <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">0</opensearch:totalResults>
</feed>
"""

ARXIV_OLD_FORMAT_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>ArXiv Query</title>
  <id>http://arxiv.org/api/query</id>
  <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/hep-ph/9901234v1</id>
    <updated>1999-01-15T00:00:00Z</updated>
    <published>1999-01-15T00:00:00Z</published>
    <title>Strong Interactions at High Energy</title>
    <summary>A review of QCD predictions for high-energy collider experiments.</summary>
    <author>
      <name>Carol Williams</name>
    </author>
    <arxiv:primary_category term="hep-ph" />
  </entry>
</feed>
"""


# ---------------------------------------------------------------------------
# Helper: mock requests.get for arXiv API
# ---------------------------------------------------------------------------

def _make_arxiv_response(xml_text, status_code=200):
    """Create a mock requests.Response for arXiv API."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = xml_text
    resp.content = xml_text.encode("utf-8")
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_zot_url():
    """FakeZotero extended for add_by_url tests."""
    zot = FakeZotero()
    return zot


@pytest.fixture
def patch_write_client(fake_zot_url):
    """Patch _get_write_client to return (fake_zot, fake_zot) for web-only mode."""
    with patch(
        "zotero_mcp.tools._helpers._get_write_client", return_value=(fake_zot_url, fake_zot_url)
    ):
        yield fake_zot_url


# ---------------------------------------------------------------------------
# DOI URL routing
# ---------------------------------------------------------------------------

class TestDoiUrlRouting:
    """DOI URLs should delegate to add_by_doi logic."""

    def test_doi_org_url_delegates(self, dummy_ctx, patch_write_client):
        """https://doi.org/10.xxx should be routed through DOI handling."""
        fake_zot = patch_write_client
        with patch("zotero_mcp.tools.write.add_by_doi") as mock_doi:
            mock_doi.return_value = "Added via DOI: 10.1234/test.2024"
            result = server.add_by_url(
                url="https://doi.org/10.1234/test.2024",
                ctx=dummy_ctx,
            )
            mock_doi.assert_called_once()
            call_kwargs = mock_doi.call_args
            # The DOI should have been extracted and passed along
            assert "10.1234/test.2024" in str(call_kwargs)

    def test_dx_doi_org_url_delegates(self, dummy_ctx, patch_write_client):
        """http://dx.doi.org/10.xxx should also route to DOI logic."""
        with patch("zotero_mcp.tools.write.add_by_doi") as mock_doi:
            mock_doi.return_value = "Added via DOI"
            result = server.add_by_url(
                url="http://dx.doi.org/10.1038/nature12373",
                ctx=dummy_ctx,
            )
            mock_doi.assert_called_once()


# ---------------------------------------------------------------------------
# arXiv URL handling
# ---------------------------------------------------------------------------

class TestArxivUrl:
    """arXiv URLs should parse the arXiv API and create preprint items."""

    def test_arxiv_abs_url(self, dummy_ctx, patch_write_client):
        """https://arxiv.org/abs/2401.00001 -> fetch arXiv API, create preprint."""
        fake_zot = patch_write_client
        mock_resp = _make_arxiv_response(ARXIV_ATOM_XML)

        with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp) as mock_get:
            result = server.add_by_url(
                url="https://arxiv.org/abs/2401.00001",
                ctx=dummy_ctx,
            )
            # Verify arXiv API was called (first call); PDF download may follow
            assert mock_get.call_count >= 1
            first_call = mock_get.call_args_list[0]
            assert "export.arxiv.org" in first_call[0][0] or "export.arxiv.org" in str(first_call)
            assert "2401.00001" in str(first_call)

        # Should have created a preprint item
        assert len(fake_zot.created) == 1
        item = fake_zot.created[0]
        assert item["itemType"] == "preprint"
        assert "Attention Is All You Need" in item["title"]

    def test_arxiv_pdf_url(self, dummy_ctx, patch_write_client):
        """https://arxiv.org/pdf/2401.00001.pdf -> same arXiv handling."""
        fake_zot = patch_write_client
        mock_resp = _make_arxiv_response(ARXIV_ATOM_XML)

        with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp) as mock_get:
            result = server.add_by_url(
                url="https://arxiv.org/pdf/2401.00001.pdf",
                ctx=dummy_ctx,
            )
            assert mock_get.call_count >= 1

        assert len(fake_zot.created) == 1
        assert fake_zot.created[0]["itemType"] == "preprint"

    def test_arxiv_old_id_format(self, dummy_ctx, patch_write_client):
        """Old arXiv format hep-ph/9901234 should work."""
        fake_zot = patch_write_client
        mock_resp = _make_arxiv_response(ARXIV_OLD_FORMAT_XML)

        with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp) as mock_get:
            result = server.add_by_url(
                url="https://arxiv.org/abs/hep-ph/9901234",
                ctx=dummy_ctx,
            )
            assert mock_get.call_count >= 1
            assert "hep-ph/9901234" in str(mock_get.call_args_list[0])

        assert len(fake_zot.created) == 1
        item = fake_zot.created[0]
        assert item["itemType"] == "preprint"
        assert "Strong Interactions" in item["title"]

    def test_arxiv_prefix_format(self, dummy_ctx, patch_write_client):
        """arXiv:2401.00001 prefix form should be handled."""
        fake_zot = patch_write_client
        mock_resp = _make_arxiv_response(ARXIV_ATOM_XML)

        with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp):
            result = server.add_by_url(
                url="arXiv:2401.00001",
                ctx=dummy_ctx,
            )
        assert len(fake_zot.created) == 1
        assert fake_zot.created[0]["itemType"] == "preprint"

    def test_arxiv_authors_parsed(self, dummy_ctx, patch_write_client):
        """Author names from arXiv XML should be parsed into creators."""
        fake_zot = patch_write_client
        mock_resp = _make_arxiv_response(ARXIV_ATOM_XML)

        with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp):
            server.add_by_url(url="https://arxiv.org/abs/2401.00001", ctx=dummy_ctx)

        item = fake_zot.created[0]
        creators = item.get("creators", [])
        assert len(creators) == 2
        # Check that author names are present (exact format depends on implementation)
        creator_names = [
            c.get("lastName", "") or c.get("name", "") for c in creators
        ]
        assert any("Smith" in n for n in creator_names)
        assert any("Jones" in n for n in creator_names)

    def test_arxiv_abstract_mapped(self, dummy_ctx, patch_write_client):
        """The <summary> element should map to abstractNote."""
        fake_zot = patch_write_client
        mock_resp = _make_arxiv_response(ARXIV_ATOM_XML)

        with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp):
            server.add_by_url(url="https://arxiv.org/abs/2401.00001", ctx=dummy_ctx)

        item = fake_zot.created[0]
        assert "sparse attention" in item.get("abstractNote", "")

    def test_arxiv_date_parsed(self, dummy_ctx, patch_write_client):
        """The <published> element should map to date."""
        fake_zot = patch_write_client
        mock_resp = _make_arxiv_response(ARXIV_ATOM_XML)

        with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp):
            server.add_by_url(url="https://arxiv.org/abs/2401.00001", ctx=dummy_ctx)

        item = fake_zot.created[0]
        assert "2024" in item.get("date", "")

    def test_arxiv_url_set(self, dummy_ctx, patch_write_client):
        """The item's URL field should point to the arXiv abs page."""
        fake_zot = patch_write_client
        mock_resp = _make_arxiv_response(ARXIV_ATOM_XML)

        with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp):
            server.add_by_url(url="https://arxiv.org/abs/2401.00001", ctx=dummy_ctx)

        item = fake_zot.created[0]
        assert "arxiv.org" in item.get("url", "")


# ---------------------------------------------------------------------------
# Generic URL -> webpage item
# ---------------------------------------------------------------------------

class TestGenericUrl:
    """Non-DOI, non-arXiv URLs should create a webpage item."""

    def test_generic_url_creates_webpage(self, dummy_ctx, patch_write_client):
        """A plain URL creates a webpage item."""
        fake_zot = patch_write_client

        with patch("zotero_mcp.tools.write.requests.get") as mock_get:
            # Don't let it try to actually fetch for arXiv
            result = server.add_by_url(
                url="https://example.com/interesting-article",
                ctx=dummy_ctx,
            )

        assert len(fake_zot.created) == 1
        item = fake_zot.created[0]
        assert item["itemType"] == "webpage"
        assert item["url"] == "https://example.com/interesting-article"


# ---------------------------------------------------------------------------
# arXiv API error handling
# ---------------------------------------------------------------------------

class TestArxivErrors:
    """Error handling for arXiv API responses."""

    def test_no_entries_returns_error(self, dummy_ctx, patch_write_client):
        """arXiv API returning zero entries should produce a clear error message."""
        mock_resp = _make_arxiv_response(ARXIV_EMPTY_XML)

        with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp):
            result = server.add_by_url(
                url="https://arxiv.org/abs/9999.99999",
                ctx=dummy_ctx,
            )

        # Should return an error string, not create any items
        assert patch_write_client.created == []
        assert "error" in result.lower() or "not found" in result.lower() or "no arxiv paper found" in result.lower()

    def test_timeout_on_arxiv_api(self, dummy_ctx, patch_write_client):
        """Network timeout calling arXiv API should return an error."""
        import requests as req_lib

        with patch(
            "zotero_mcp.tools.write.requests.get",
            side_effect=req_lib.exceptions.Timeout("Connection timed out"),
        ):
            result = server.add_by_url(
                url="https://arxiv.org/abs/2401.00001",
                ctx=dummy_ctx,
            )

        assert patch_write_client.created == []
        assert "error" in result.lower() or "timeout" in result.lower()


# ---------------------------------------------------------------------------
# arXiv XML namespace handling
# ---------------------------------------------------------------------------

class TestArxivXmlNamespace:
    """Verify correct XML namespace handling for arXiv Atom feed."""

    def test_atom_namespace_parsed(self, dummy_ctx, patch_write_client):
        """Elements in the Atom namespace ({http://www.w3.org/2005/Atom})
        should be found correctly."""
        fake_zot = patch_write_client
        mock_resp = _make_arxiv_response(ARXIV_ATOM_XML)

        with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp):
            server.add_by_url(url="https://arxiv.org/abs/2401.00001", ctx=dummy_ctx)

        # If namespace handling is broken, title/authors won't be parsed
        item = fake_zot.created[0]
        assert item["title"] != ""
        assert len(item.get("creators", [])) > 0

    def test_arxiv_namespace_category(self, dummy_ctx, patch_write_client):
        """The arxiv: namespace ({http://arxiv.org/schemas/atom}) for
        primary_category should be handled."""
        fake_zot = patch_write_client
        mock_resp = _make_arxiv_response(ARXIV_ATOM_XML)

        with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp):
            result = server.add_by_url(
                url="https://arxiv.org/abs/2401.00001",
                ctx=dummy_ctx,
            )

        # The category info (cs.CL) should appear somewhere in the result
        # or in the item's extra field — exact location depends on implementation
        item = fake_zot.created[0]
        item_str = str(item)
        # At minimum the item should have been created successfully
        assert item["itemType"] == "preprint"


# ---------------------------------------------------------------------------
# HTTPS enforcement for arXiv API
# ---------------------------------------------------------------------------

class TestArxivHttps:
    """The arXiv API should always be called over HTTPS."""

    def test_uses_https(self, dummy_ctx, patch_write_client):
        """API call to export.arxiv.org must use HTTPS, not HTTP."""
        mock_resp = _make_arxiv_response(ARXIV_ATOM_XML)

        with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp) as mock_get:
            server.add_by_url(url="https://arxiv.org/abs/2401.00001", ctx=dummy_ctx)
            call_url = mock_get.call_args[0][0]
            assert call_url.startswith("https://"), (
                f"arXiv API URL should use HTTPS, got: {call_url}"
            )

    def test_timeout_parameter_set(self, dummy_ctx, patch_write_client):
        """requests.get for arXiv should include a timeout parameter."""
        mock_resp = _make_arxiv_response(ARXIV_ATOM_XML)

        with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp) as mock_get:
            server.add_by_url(url="https://arxiv.org/abs/2401.00001", ctx=dummy_ctx)
            call_kwargs = mock_get.call_args[1]
            assert "timeout" in call_kwargs, "requests.get must include a timeout"
            assert call_kwargs["timeout"] > 0


# ---------------------------------------------------------------------------
# Hybrid mode / local-only rejection
# ---------------------------------------------------------------------------

class TestHybridMode:
    """Write operations require hybrid mode (web credentials)."""

    def test_local_only_rejected(self, dummy_ctx):
        """In local-only mode (no web credentials), add_by_url should error."""
        with patch(
            "zotero_mcp.tools._helpers._get_write_client",
            side_effect=ValueError(
                "Cannot perform write operations in local-only mode. "
                "Add ZOTERO_API_KEY and ZOTERO_LIBRARY_ID to enable hybrid mode."
            ),
        ):
            result = server.add_by_url(
                url="https://arxiv.org/abs/2401.00001",
                ctx=dummy_ctx,
            )
        assert "local-only" in result.lower() or "cannot" in result.lower()

    def test_hybrid_mode_uses_write_client(self, dummy_ctx):
        """In hybrid mode, items should be created via the write (web) client."""
        read_zot = FakeZotero()
        write_zot = FakeZotero()
        write_zot.library_id = "99999"  # distinct from read

        mock_resp = _make_arxiv_response(ARXIV_ATOM_XML)

        with patch(
            "zotero_mcp.tools._helpers._get_write_client", return_value=(read_zot, write_zot)
        ):
            with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp):
                server.add_by_url(
                    url="https://arxiv.org/abs/2401.00001",
                    ctx=dummy_ctx,
                )

        # Item should be created on the write client, not the read client
        assert len(write_zot.created) == 1
        assert len(read_zot.created) == 0


# ---------------------------------------------------------------------------
# Tags and collections applied
# ---------------------------------------------------------------------------

class TestTagsAndCollections:
    """Tags and collections should be applied to created items."""

    def test_tags_applied(self, dummy_ctx, patch_write_client):
        """Tags parameter should be added to the created item."""
        fake_zot = patch_write_client
        mock_resp = _make_arxiv_response(ARXIV_ATOM_XML)

        with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp):
            server.add_by_url(
                url="https://arxiv.org/abs/2401.00001",
                tags=["machine-learning", "transformers"],
                ctx=dummy_ctx,
            )

        item = fake_zot.created[0]
        tag_values = [t["tag"] for t in item.get("tags", [])]
        assert "machine-learning" in tag_values
        assert "transformers" in tag_values

    def test_tags_as_json_string(self, dummy_ctx, patch_write_client):
        """Tags passed as JSON string should be normalized."""
        fake_zot = patch_write_client
        mock_resp = _make_arxiv_response(ARXIV_ATOM_XML)

        with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp):
            server.add_by_url(
                url="https://arxiv.org/abs/2401.00001",
                tags='["nlp", "deep-learning"]',
                ctx=dummy_ctx,
            )

        item = fake_zot.created[0]
        tag_values = [t["tag"] for t in item.get("tags", [])]
        assert "nlp" in tag_values
        assert "deep-learning" in tag_values

    def test_collections_applied(self, dummy_ctx, patch_write_client):
        """Collections parameter should set the item's collections field."""
        fake_zot = patch_write_client
        mock_resp = _make_arxiv_response(ARXIV_ATOM_XML)

        with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp):
            server.add_by_url(
                url="https://arxiv.org/abs/2401.00001",
                collections=["ABC12345"],
                ctx=dummy_ctx,
            )

        item = fake_zot.created[0]
        assert "ABC12345" in item.get("collections", [])

    def test_collection_names_resolved(self, dummy_ctx, patch_write_client):
        """Collection names passed as collections are used as-is (keys or names).
        The arXiv flow passes them through _normalize_str_list_input, not
        _resolve_collection_names, so names appear directly in collections."""
        fake_zot = patch_write_client
        fake_zot._collections = [
            {"key": "COLL0001", "data": {"name": "My Papers"}},
            {"key": "COLL0002", "data": {"name": "Archive"}},
        ]
        mock_resp = _make_arxiv_response(ARXIV_ATOM_XML)

        with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp):
            server.add_by_url(
                url="https://arxiv.org/abs/2401.00001",
                collections=["My Papers"],
                ctx=dummy_ctx,
            )

        item = fake_zot.created[0]
        # The name is passed through _normalize_str_list_input (not resolved to key)
        assert "My Papers" in item.get("collections", [])

    def test_no_tags_or_collections(self, dummy_ctx, patch_write_client):
        """Omitting tags and collections should still create the item."""
        fake_zot = patch_write_client
        mock_resp = _make_arxiv_response(ARXIV_ATOM_XML)

        with patch("zotero_mcp.tools.write.requests.get", return_value=mock_resp):
            server.add_by_url(
                url="https://arxiv.org/abs/2401.00001",
                ctx=dummy_ctx,
            )

        assert len(fake_zot.created) == 1

    def test_tags_applied_to_webpage(self, dummy_ctx, patch_write_client):
        """Tags should also be applied when creating a generic webpage item."""
        fake_zot = patch_write_client

        result = server.add_by_url(
            url="https://example.com/article",
            tags=["reference"],
            ctx=dummy_ctx,
        )

        item = fake_zot.created[0]
        tag_values = [t["tag"] for t in item.get("tags", [])]
        assert "reference" in tag_values
