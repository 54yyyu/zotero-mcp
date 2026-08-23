"""``add_by_url`` must read the citation a publisher page publishes about itself.

Reported against a real library: adding
``https://ojs.mruni.eu/ojs/public-policy-and-administration/article/view/5155``
produced a ``webpage`` item whose only populated field was the URL — no
title, no authors, no journal. That page serves a complete set of Highwire
``citation_*`` tags, and OJS/PKP runs a large share of the world's journals,
so the blank item was not an edge case.

The old code never fetched the page at all: the generic-URL branch built a
``webpage`` template, set ``title = url``, and created it.
"""

from unittest.mock import MagicMock, patch

import pytest
from conftest import DummyContext, FakeZotero

from zotero_mcp.tools import write

OJS_PAGE = """<html><head>
<meta name="citation_journal_title" content="Public Policy and Administration"/>
<meta name="citation_issn" content="2029-2872"/>
<meta name="citation_author" content="Mohamad Thahir Haning"/>
<meta name="citation_author" content="Hasniati Hamzah"/>
<meta name="citation_title" content="DETERMINANTS OF PUBLIC TRUST"/>
<meta name="citation_date" content="2020/07/07"/>
<meta name="citation_volume" content="19"/>
<meta name="citation_issue" content="2"/>
<meta name="citation_firstpage" content="205"/>
<meta name="citation_lastpage" content="218"/>
</head><body></body></html>
"""

# Same page, but declaring its DOI — the route that should defer to CrossRef.
OJS_PAGE_WITH_DOI = OJS_PAGE.replace(
    "</head>",
    '<meta name="citation_doi" content="10.13165/VPA-20-19-2-04"/></head>',
)

BARE_PAGE = "<html><head><title>A blog</title></head><body>hi</body></html>"

ARTICLE_URL = "https://ojs.example/ojs/ppa/article/view/5155"


def _html_response(body: str, content_type: str = "text/html; charset=utf-8"):
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"Content-Type": content_type}
    resp.encoding = "utf-8"
    resp.raise_for_status = MagicMock()
    resp.iter_content = MagicMock(return_value=[body.encode("utf-8")])
    resp.close = MagicMock()
    return resp


@pytest.fixture
def fake_zot():
    return FakeZotero()


@pytest.fixture
def patched(fake_zot):
    with patch(
        "zotero_mcp.tools._helpers._get_write_client",
        return_value=(fake_zot, fake_zot),
    ):
        yield fake_zot


class TestEmbeddedMetadataIsUsed:
    def test_article_page_creates_a_journal_article(self, patched):
        with patch("zotero_mcp.tools.write.requests.get",
                   return_value=_html_response(OJS_PAGE)):
            result = write.add_by_url(url=ARTICLE_URL, ctx=DummyContext())

        assert len(patched.created) == 1
        item = patched.created[0]
        assert item["itemType"] == "journalArticle", (
            "a page advertising a journal, volume, issue and page range is "
            "not a webpage"
        )
        assert item["title"] == "DETERMINANTS OF PUBLIC TRUST"
        assert item["publicationTitle"] == "Public Policy and Administration"
        assert item["volume"] == "19"
        assert item["issue"] == "2"
        assert item["pages"] == "205-218"
        assert item["date"] == "2020-07-07"
        assert item["ISSN"] == "2029-2872"
        assert item["url"] == ARTICLE_URL
        assert [c["lastName"] for c in item["creators"]] == ["Haning", "Hamzah"]
        assert "Successfully added" in result

    def test_declared_doi_routes_through_add_by_doi(self, patched):
        """A page that names its own DOI should take the DOI path, which
        already handles de-duplication, CrossRef and OA PDF attachment."""
        with patch("zotero_mcp.tools.write.requests.get",
                   return_value=_html_response(OJS_PAGE_WITH_DOI)), \
             patch("zotero_mcp.tools.write.add_by_doi",
                   return_value="Successfully added: **X**") as mock_doi:
            result = write.add_by_url(url=ARTICLE_URL, ctx=DummyContext())

        mock_doi.assert_called_once()
        assert mock_doi.call_args.kwargs["doi"] == "10.13165/VPA-20-19-2-04"
        assert "Successfully added" in result

    def test_doi_route_failure_falls_back_to_the_page(self, patched):
        """CrossRef not knowing the DOI must not cost us the metadata the
        page already handed us."""
        with patch("zotero_mcp.tools.write.requests.get",
                   return_value=_html_response(OJS_PAGE_WITH_DOI)), \
             patch("zotero_mcp.tools.write.add_by_doi",
                   return_value="DOI not found on CrossRef: 10.13165/x"):
            write.add_by_url(url=ARTICLE_URL, ctx=DummyContext())

        assert len(patched.created) == 1
        assert patched.created[0]["itemType"] == "journalArticle"
        assert patched.created[0]["title"] == "DETERMINANTS OF PUBLIC TRUST"

    def test_tags_and_collections_still_apply(self, patched):
        with patch("zotero_mcp.tools.write.requests.get",
                   return_value=_html_response(OJS_PAGE)):
            write.add_by_url(url=ARTICLE_URL, tags=["screened"],
                             ctx=DummyContext())

        assert patched.created[0]["tags"] == [{"tag": "screened"}]


class TestDegradesToTheOldBehaviour:
    """Embedded metadata is an enrichment. Every failure to read it must
    still produce the item it produced before."""

    def test_page_without_metadata_stays_a_webpage(self, patched):
        with patch("zotero_mcp.tools.write.requests.get",
                   return_value=_html_response(BARE_PAGE)):
            result = write.add_by_url(url="https://example.com/post",
                                      ctx=DummyContext())

        item = patched.created[0]
        assert item["itemType"] == "webpage"
        assert item["url"] == "https://example.com/post"
        # Degraded, and says so.
        assert "no citation metadata" in result

    def test_fetch_failure_stays_a_webpage_and_says_why(self, patched):
        with patch("zotero_mcp.tools.write.requests.get",
                   side_effect=OSError("connection refused")):
            result = write.add_by_url(url="https://example.com/post",
                                      ctx=DummyContext())

        assert patched.created[0]["itemType"] == "webpage"
        assert "Only the URL could be recorded" in result

    def test_tls_failure_is_named(self, patched):
        """A host whose certificate will not verify under OpenSSL — seen on a
        real university OJS install that browsers and curl accept. The user
        needs to know that is why the item is blank."""
        import requests as _requests
        with patch("zotero_mcp.tools.write.requests.get",
                   side_effect=_requests.exceptions.SSLError("bad chain")):
            result = write.add_by_url(url="https://ojs.example/article/view/1",
                                      ctx=DummyContext())

        assert patched.created[0]["itemType"] == "webpage"
        assert "certificate could not be verified" in result

    def test_non_html_response_is_not_parsed(self, patched):
        with patch("zotero_mcp.tools.write.requests.get",
                   return_value=_html_response(OJS_PAGE, "application/pdf")):
            write.add_by_url(url="https://example.com/f.pdf", ctx=DummyContext())

        assert patched.created[0]["itemType"] == "webpage"

    def test_oversized_page_is_truncated_not_refused(self, patched):
        """A huge page must not hold the write open; the head is enough."""
        filler = "<!-- " + ("x" * 200_000) + " -->"
        resp = _html_response(OJS_PAGE)
        resp.iter_content = MagicMock(return_value=[
            OJS_PAGE.encode("utf-8"),
            *[filler.encode("utf-8")] * 20,
        ])
        with patch("zotero_mcp.tools.write.requests.get", return_value=resp):
            write.add_by_url(url=ARTICLE_URL, ctx=DummyContext())

        assert patched.created[0]["itemType"] == "journalArticle"
