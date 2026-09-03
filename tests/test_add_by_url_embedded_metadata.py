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


# ---------------------------------------------------------------------------
# Page metadata fills the gaps CrossRef leaves
# ---------------------------------------------------------------------------

class TestSupplementalMetadata:
    """A publisher can register an article's DOI as a ``journal-issue``,
    whose CrossRef record legitimately carries no title, authors, volume,
    issue or pages — while the article's own landing page advertises all of
    them.

    Live example: 10.13165/vpa-20-19-2-04. CrossRef returns
    ``title: []``, no author, no volume, no issue, no page. Routing that page
    through its DOI without consulting the page again produced a *titleless*
    item — worse than not asking CrossRef at all.
    """

    ISSUE_LEVEL_CROSSREF = {
        "type": "journal-issue",
        "title": [],
        "container-title": ["Public Policy and Administration"],
        "DOI": "10.13165/vpa-20-19-2-04",
        "ISSN": ["1648-2603"],
        "published": {"date-parts": [[2020]]},
    }

    def _crossref(self, message):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"status": "ok", "message": message}
        resp.raise_for_status = MagicMock()
        return resp

    def _meta(self):
        from zotero_mcp.html_metadata import extract_embedded_metadata
        return extract_embedded_metadata(OJS_PAGE)

    def test_page_fills_what_crossref_omits(self, patched):
        from zotero_mcp.tools import write as w
        with patch("zotero_mcp.tools.write.requests.get",
                   return_value=self._crossref(self.ISSUE_LEVEL_CROSSREF)), \
             patch("zotero_mcp.tools._helpers._try_attach_oa_pdf",
                   return_value="skipped"):
            w.add_by_doi(doi="10.13165/vpa-20-19-2-04",
                         supplemental=self._meta(), ctx=DummyContext())

        item = patched.created[0]
        assert item["title"] == "DETERMINANTS OF PUBLIC TRUST", (
            "CrossRef sent title: [] — the page had the title all along"
        )
        assert [c["lastName"] for c in item["creators"]] == ["Haning", "Hamzah"]

    def test_crossref_still_wins_where_it_speaks(self, patched):
        """The page fills gaps; it does not override. CrossRef is the
        authority wherever it says anything at all."""
        from zotero_mcp.tools import write as w
        message = dict(self.ISSUE_LEVEL_CROSSREF)
        message["title"] = ["The CrossRef Title"]
        with patch("zotero_mcp.tools.write.requests.get",
                   return_value=self._crossref(message)), \
             patch("zotero_mcp.tools._helpers._try_attach_oa_pdf",
                   return_value="skipped"):
            w.add_by_doi(doi="10.13165/vpa-20-19-2-04",
                         supplemental=self._meta(), ctx=DummyContext())

        assert patched.created[0]["title"] == "The CrossRef Title"

    def test_no_supplemental_is_unchanged(self, patched):
        """Callers that pass nothing behave exactly as before."""
        from zotero_mcp.tools import write as w
        with patch("zotero_mcp.tools.write.requests.get",
                   return_value=self._crossref(self.ISSUE_LEVEL_CROSSREF)), \
             patch("zotero_mcp.tools._helpers._try_attach_oa_pdf",
                   return_value="skipped"):
            w.add_by_doi(doi="10.13165/vpa-20-19-2-04", ctx=DummyContext())

        assert not patched.created[0].get("title")

    def test_add_by_url_passes_the_page_down(self, patched):
        """The wiring: a DOI-declaring page hands its own tags to the DOI
        route rather than discarding them."""
        with patch("zotero_mcp.tools.write.requests.get",
                   return_value=_html_response(OJS_PAGE_WITH_DOI)), \
             patch("zotero_mcp.tools.write.add_by_doi",
                   return_value="Successfully added: **X**") as mock_doi:
            write.add_by_url(url=ARTICLE_URL, ctx=DummyContext())

        supplied = mock_doi.call_args.kwargs["supplemental"]
        assert supplied.title == "DETERMINANTS OF PUBLIC TRUST"
        assert supplied.volume == "19"


# ---------------------------------------------------------------------------
# The DOI route resolves its own landing page when CrossRef is thin
# ---------------------------------------------------------------------------

class TestDoiRouteSelfSupplements:
    """The url route hands its tags down as ``supplemental``. A caller who
    passes a bare DOI has no page to hand over — and got a titleless item for
    exactly the DOIs where CrossRef says least.

    Reported by the reviewing agent, who put it better than the fix did:
    registry silence is not evidence of absence, and that had been applied to
    one of the two routes.
    """

    THIN = {
        "type": "journal-issue",
        "title": [],
        "container-title": ["Public Policy and Administration"],
        "DOI": "10.13165/vpa-20-19-2-04",
        "URL": "https://doi.org/10.13165/vpa-20-19-2-04",
        "ISSN": ["1648-2603"],
        "published": {"date-parts": [[2020]]},
    }
    FULL = {
        "type": "journal-article",
        "title": ["A Complete CrossRef Record"],
        "container-title": ["Some Journal"],
        "DOI": "10.1234/full",
        "volume": "9", "issue": "2", "page": "1-20",
        "author": [{"given": "Ada", "family": "Lovelace"}],
        "published": {"date-parts": [[2024]]},
    }

    def _dispatch(self, crossref_msg, page_html=OJS_PAGE):
        """requests.get stub: CrossRef for api.crossref.org, HTML otherwise."""
        def _get(url, **kwargs):
            if "api.crossref.org" in url:
                r = MagicMock()
                r.status_code = 200
                r.json.return_value = {"status": "ok", "message": crossref_msg}
                r.raise_for_status = MagicMock()
                return r
            return _html_response(page_html)
        return _get

    def test_thin_record_reads_the_landing_page(self, patched):
        with patch("zotero_mcp.tools.write.requests.get",
                   side_effect=self._dispatch(self.THIN)), \
             patch("zotero_mcp.tools._helpers._try_attach_oa_pdf",
                   return_value="skipped"):
            write.add_by_doi(doi="10.13165/vpa-20-19-2-04", ctx=DummyContext())

        item = patched.created[0]
        assert item["title"] == "DETERMINANTS OF PUBLIC TRUST"
        assert [c["lastName"] for c in item["creators"]] == ["Haning", "Hamzah"]

    def test_complete_record_does_not_fetch_a_page(self, patched):
        """The fetch is gated, not routine. A DOI whose CrossRef record
        stands on its own must not pay for a page load."""
        calls = []
        def _get(url, **kwargs):
            calls.append(url)
            return self._dispatch(self.FULL)(url, **kwargs)

        with patch("zotero_mcp.tools.write.requests.get", side_effect=_get), \
             patch("zotero_mcp.tools._helpers._try_attach_oa_pdf",
                   return_value="skipped"):
            write.add_by_doi(doi="10.1234/full", ctx=DummyContext())

        assert all("api.crossref.org" in u for u in calls), (
            f"expected only the CrossRef call, got {calls}"
        )
        assert patched.created[0]["title"] == "A Complete CrossRef Record"

    def test_unreachable_landing_page_degrades(self, patched):
        """The page is best-effort. A publisher whose host will not answer
        must not fail the import — this is the reported DOI's real situation,
        whose landing page serves a chain OpenSSL will not verify."""
        import requests as _requests
        def _get(url, **kwargs):
            if "api.crossref.org" in url:
                return self._dispatch(self.THIN)(url, **kwargs)
            raise _requests.exceptions.SSLError("bad chain")

        with patch("zotero_mcp.tools.write.requests.get", side_effect=_get), \
             patch("zotero_mcp.tools._helpers._try_attach_oa_pdf",
                   return_value="skipped"):
            out = write.add_by_doi(doi="10.13165/vpa-20-19-2-04",
                                   ctx=DummyContext())

        assert patched.created  # item still created
        assert "Successfully added" in out

    def test_supplied_metadata_is_not_refetched(self, patched):
        """The url route already has the page; the DOI route must not go
        back for it."""
        calls = []
        def _get(url, **kwargs):
            calls.append(url)
            return self._dispatch(self.THIN)(url, **kwargs)

        from zotero_mcp.html_metadata import extract_embedded_metadata
        with patch("zotero_mcp.tools.write.requests.get", side_effect=_get), \
             patch("zotero_mcp.tools._helpers._try_attach_oa_pdf",
                   return_value="skipped"):
            write.add_by_doi(doi="10.13165/vpa-20-19-2-04",
                             supplemental=extract_embedded_metadata(OJS_PAGE),
                             ctx=DummyContext())

        assert all("api.crossref.org" in u for u in calls), (
            f"page was re-fetched despite being supplied: {calls}"
        )
        assert patched.created[0]["title"] == "DETERMINANTS OF PUBLIC TRUST"


class TestThinRecordSurvivesTheBatchPath:
    """The batch path resolves metadata for every DOI in one CrossRef request,
    then builds items in a second pass — a different code path from the
    single-DOI one, and the seam where #477's landing-page fallback could go
    missing without any existing test noticing.

    ``supplemental`` describes one specific page, so a batch cannot be handed
    one; each thin record must therefore read its own landing page.
    """

    DOIS = ["10.13165/vpa-20-19-2-04", "10.13165/vpa-20-19-2-05"]

    @staticmethod
    def _thin(doi):
        return {
            "type": "journal-issue",
            "title": [],
            "container-title": ["Public Policy and Administration"],
            "DOI": doi,
            "URL": f"https://ojs.example/landing/{doi[-2:]}",
            "published": {"date-parts": [[2020]]},
        }

    def _dispatch(self):
        """CrossRef answers the batched filter form and the single form; any
        other host serves the landing page."""
        def _get(url, **kwargs):
            if "api.crossref.org" in url:
                r = MagicMock()
                r.status_code = 200
                r.raise_for_status = MagicMock()
                if url.rstrip("/").endswith("/works"):
                    r.json.return_value = {
                        "status": "ok",
                        "message": {"items": [self._thin(d) for d in self.DOIS]},
                    }
                else:
                    doi = url.rsplit("/works/", 1)[1]
                    r.json.return_value = {"status": "ok", "message": self._thin(doi)}
                return r
            return _html_response(OJS_PAGE)
        return _get

    def test_each_thin_record_in_a_batch_reads_its_own_page(self, patched):
        with patch("zotero_mcp.tools.write.requests.get",
                   side_effect=self._dispatch()), \
             patch("zotero_mcp.tools._helpers._try_attach_oa_pdf",
                   return_value="skipped"):
            out = write.add_by_doi(doi=self.DOIS, attach_mode="none",
                                   ctx=DummyContext())

        assert out.splitlines()[0].startswith("# Added 2 of 2 DOIs"), out
        titles = [i["title"] for i in patched.created]
        assert titles == ["DETERMINANTS OF PUBLIC TRUST"] * 2, titles

    def test_batch_landing_pages_are_read_outside_the_api_lock(self, patched):
        """The fallback is outbound HTTP to a third party. Holding the
        process-wide Zotero lock across it is the exact starvation the lock
        narrowing exists to prevent."""
        from test_lock_scope import _lock_is_free

        free_during_page_fetch = []

        def _get(url, **kwargs):
            if "api.crossref.org" not in url:
                free_during_page_fetch.append(_lock_is_free())
            return self._dispatch()(url, **kwargs)

        with patch("zotero_mcp.tools.write.requests.get", side_effect=_get), \
             patch("zotero_mcp.tools._helpers._try_attach_oa_pdf",
                   return_value="skipped"):
            write.add_by_doi(doi=self.DOIS, attach_mode="none",
                             ctx=DummyContext())

        assert free_during_page_fetch, "no landing page fetched -- proves nothing"
        assert all(free_during_page_fetch), (
            "the Zotero API lock was held across a landing-page fetch"
        )


class TestThinRecordDetection:
    def test_empty_title_list_is_thin(self):
        assert write._crossref_record_is_thin({"title": []}, "journal-article")

    def test_absent_title_is_thin(self):
        assert write._crossref_record_is_thin({}, "journal-article")

    def test_blank_title_string_is_thin(self):
        assert write._crossref_record_is_thin({"title": "  "}, "journal-article")

    def test_container_type_is_thin_even_with_a_title(self):
        assert write._crossref_record_is_thin(
            {"title": ["An Issue"]}, "journal-issue"
        )

    def test_ordinary_titled_article_is_not_thin(self):
        assert not write._crossref_record_is_thin(
            {"title": ["A Paper"]}, "journal-article"
        )

