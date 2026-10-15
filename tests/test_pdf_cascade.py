"""Unit tests for the PDF attachment cascade (_try_unpaywall, _try_openalex,
_try_arxiv_from_crossref, _try_semantic_scholar, _try_pmc, _download_and_attach_pdf,
_try_attach_oa_pdf)."""

import json

import pytest
import requests

from zotero_mcp.server import (
    _download_and_attach_pdf,
    _try_arxiv_from_crossref,
    _try_attach_oa_pdf,
    _try_openalex,
    _try_pmc,
    _try_semantic_scholar,
    _try_unpaywall,
)
from conftest import DummyContext, FakeZotero


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeHTTPResponse:
    """Minimal requests.Response stand-in."""

    def __init__(self, status_code=200, json_data=None, content=b"",
                 headers=None):
        self.status_code = status_code
        self._json = json_data
        self.content = content
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def iter_content(self, chunk_size=8192):
        yield self.content


class _AttachZotero(FakeZotero):
    """FakeZotero extended with attachment_both tracking."""

    def __init__(self):
        super().__init__()
        self.attachments = []

    def attachment_both(self, files, parentid=None, **kwargs):
        self.attachments.append({"files": files, "parentid": parentid})
        # Shape matters: pyzotero reports client-side rejections by putting
        # the payload in "failure" rather than raising (#403), so callers
        # check for a registered key. A fake that returns None reads as a
        # failed upload.
        return {
            "success": [{"key": f"ATCH{len(self.attachments):04d}"}],
            "failure": [],
            "unchanged": [],
        }


def _allow_ssrf_guard(monkeypatch):
    """Bypass the SSRF host check so a test can exercise download logic
    (content-type / size) without depending on real DNS resolution."""
    from zotero_mcp.tools import _helpers
    monkeypatch.setattr(_helpers, "_url_resolves_to_public_host", lambda url: True)


# ---------------------------------------------------------------------------
# _try_unpaywall
# ---------------------------------------------------------------------------

class TestTryUnpaywall:

    def test_unpaywall_best_location(self, monkeypatch, dummy_ctx):
        """best_oa_location has url_for_pdf -> returns that URL."""
        payload = {
            "best_oa_location": {"url_for_pdf": "https://example.com/paper.pdf"},
            "oa_locations": [],
        }

        def fake_get(url, **kwargs):
            return _FakeHTTPResponse(200, json_data=payload)

        monkeypatch.setattr(requests, "get", fake_get)
        result = _try_unpaywall("10.1234/test", dummy_ctx)
        assert result == "https://example.com/paper.pdf"

    def test_unpaywall_iterates_oa_locations(self, monkeypatch, dummy_ctx):
        """best_oa_location has no url_for_pdf, fallback to oa_locations[1]."""
        payload = {
            "best_oa_location": {"url": "https://landing.example.com"},
            "oa_locations": [
                {"url_for_pdf": None},
                {"url_for_pdf": "https://alt.example.com/paper.pdf"},
            ],
        }

        def fake_get(url, **kwargs):
            return _FakeHTTPResponse(200, json_data=payload)

        monkeypatch.setattr(requests, "get", fake_get)
        result = _try_unpaywall("10.1234/test", dummy_ctx)
        assert result == "https://alt.example.com/paper.pdf"

    def test_unpaywall_no_oa(self, monkeypatch, dummy_ctx):
        """is_oa false, no locations -> returns None."""
        payload = {
            "is_oa": False,
            "best_oa_location": None,
            "oa_locations": [],
        }

        def fake_get(url, **kwargs):
            return _FakeHTTPResponse(200, json_data=payload)

        monkeypatch.setattr(requests, "get", fake_get)
        result = _try_unpaywall("10.1234/closed", dummy_ctx)
        assert result is None

    def test_unpaywall_timeout(self, monkeypatch, dummy_ctx):
        """requests.get raises Timeout -> returns None gracefully."""

        def fake_get(url, **kwargs):
            raise requests.exceptions.Timeout("timed out")

        monkeypatch.setattr(requests, "get", fake_get)
        result = _try_unpaywall("10.1234/slow", dummy_ctx)
        assert result is None


# ---------------------------------------------------------------------------
# _try_openalex
# ---------------------------------------------------------------------------

class TestTryOpenAlex:

    def _get(self, monkeypatch, payload, status=200):
        seen = []

        def fake_get(url, **kwargs):
            seen.append((url, kwargs))
            return _FakeHTTPResponse(status, json_data=payload)

        monkeypatch.setattr(requests, "get", fake_get)
        return seen

    def test_collects_pdf_urls_skipping_nulls(self, monkeypatch, dummy_ctx):
        """The shape of 10.1175/jcli-d-12-00821.1: no best_oa_location, a
        publisher location without a pdf_url, and a closed repository copy."""
        self._get(monkeypatch, {
            "best_oa_location": None,
            "locations": [
                {"is_oa": False, "pdf_url": None},
                {"is_oa": False, "pdf_url": "http://repo.example.org/1/paper.pdf"},
            ],
        })
        result = _try_openalex("10.1234/test", dummy_ctx)
        assert result == ["http://repo.example.org/1/paper.pdf"]

    def test_best_then_open_then_closed_without_duplicates(self, monkeypatch, dummy_ctx):
        self._get(monkeypatch, {
            "best_oa_location": {"is_oa": True, "pdf_url": "https://b.example/best.pdf"},
            "locations": [
                {"is_oa": False, "pdf_url": "https://pub.example/paywalled.pdf"},
                {"is_oa": True, "pdf_url": "https://b.example/best.pdf"},
                {"is_oa": True, "pdf_url": "https://repo.example/oa.pdf"},
                {"is_oa": False, "pdf_url": "https://pub.example/paywalled.pdf"},
            ],
        })
        result = _try_openalex("10.1234/test", dummy_ctx)
        assert result == [
            "https://b.example/best.pdf",
            "https://repo.example/oa.pdf",
            "https://pub.example/paywalled.pdf",
        ]

    def test_missing_locations(self, monkeypatch, dummy_ctx):
        self._get(monkeypatch, {"best_oa_location": None})
        assert _try_openalex("10.1234/test", dummy_ctx) is None

    def test_no_record(self, monkeypatch, dummy_ctx):
        self._get(monkeypatch, None, status=404)
        assert _try_openalex("10.1234/test", dummy_ctx) is None

    def test_timeout(self, monkeypatch, dummy_ctx):
        def fake_get(url, **kwargs):
            raise requests.exceptions.Timeout("timed out")

        monkeypatch.setattr(requests, "get", fake_get)
        assert _try_openalex("10.1234/slow", dummy_ctx) is None

    def test_identifies_itself(self, monkeypatch, dummy_ctx):
        """Same mailto as discovery.py, so requests land in the polite pool."""
        seen = self._get(monkeypatch, {"locations": []})
        _try_openalex("10.1234/test", dummy_ctx)
        url, kwargs = seen[0]
        assert url == "https://api.openalex.org/works/https://doi.org/10.1234/test"
        assert kwargs["params"]["mailto"] == "zotero-mcp@users.noreply.github.com"


# ---------------------------------------------------------------------------
# _try_arxiv_from_crossref
# ---------------------------------------------------------------------------

class TestTryArxivFromCrossref:

    def test_arxiv_from_crossref_doi_format(self, dummy_ctx):
        """relation has-preprint with id-type 'doi' containing arXiv DOI."""
        metadata = {
            "relation": {
                "has-preprint": [
                    {"id-type": "doi", "id": "10.48550/arXiv.2307.02743"}
                ]
            }
        }
        result = _try_arxiv_from_crossref(metadata, dummy_ctx)
        assert result == "https://arxiv.org/pdf/2307.02743.pdf"

    def test_arxiv_from_crossref_arxiv_type(self, dummy_ctx):
        """relation has-preprint with id-type 'arxiv' and bare arXiv id."""
        metadata = {
            "relation": {
                "has-preprint": [
                    {"id-type": "arxiv", "id": "2307.02743"}
                ]
            }
        }
        result = _try_arxiv_from_crossref(metadata, dummy_ctx)
        assert result == "https://arxiv.org/pdf/2307.02743.pdf"

    def test_arxiv_from_crossref_no_relation(self, dummy_ctx):
        """No relation field at all -> returns None."""
        metadata = {"title": ["Some Paper"], "DOI": "10.1234/test"}
        result = _try_arxiv_from_crossref(metadata, dummy_ctx)
        assert result is None


# ---------------------------------------------------------------------------
# _try_semantic_scholar
# ---------------------------------------------------------------------------

class TestTrySemanticScholar:

    def test_semantic_scholar_has_pdf(self, monkeypatch, dummy_ctx):
        """S2 returns openAccessPdf with url -> returns that URL."""
        payload = {
            "openAccessPdf": {"url": "https://s2.example.com/paper.pdf"}
        }

        def fake_get(url, **kwargs):
            return _FakeHTTPResponse(200, json_data=payload)

        monkeypatch.setattr(requests, "get", fake_get)
        result = _try_semantic_scholar("10.1234/test", dummy_ctx)
        assert result == "https://s2.example.com/paper.pdf"

    def test_semantic_scholar_no_pdf(self, monkeypatch, dummy_ctx):
        """S2 returns openAccessPdf: null -> returns None."""
        payload = {"openAccessPdf": None}

        def fake_get(url, **kwargs):
            return _FakeHTTPResponse(200, json_data=payload)

        monkeypatch.setattr(requests, "get", fake_get)
        result = _try_semantic_scholar("10.1234/closed", dummy_ctx)
        assert result is None


# ---------------------------------------------------------------------------
# _try_pmc
# ---------------------------------------------------------------------------

class TestTryPmc:

    def test_pmc_found(self, monkeypatch, dummy_ctx):
        """NCBI converter returns pmcid -> returns PMC PDF URL."""
        payload = {"records": [{"pmcid": "PMC1234567"}]}

        def fake_get(url, **kwargs):
            return _FakeHTTPResponse(200, json_data=payload)

        monkeypatch.setattr(requests, "get", fake_get)
        result = _try_pmc("10.1234/test", dummy_ctx)
        assert result == "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/pdf/"

    def test_pmc_no_pmcid(self, monkeypatch, dummy_ctx):
        """NCBI returns record without pmcid -> returns None."""
        payload = {"records": [{"doi": "10.1234/test"}]}

        def fake_get(url, **kwargs):
            return _FakeHTTPResponse(200, json_data=payload)

        monkeypatch.setattr(requests, "get", fake_get)
        result = _try_pmc("10.1234/test", dummy_ctx)
        assert result is None


# ---------------------------------------------------------------------------
# _download_and_attach_pdf
# ---------------------------------------------------------------------------

class TestDownloadAndAttachPdf:

    def test_download_content_type_check(self, monkeypatch, dummy_ctx):
        """Response with text/html content-type -> file NOT attached."""
        zot = _AttachZotero()
        _allow_ssrf_guard(monkeypatch)

        def fake_get(url, **kwargs):
            return _FakeHTTPResponse(
                200, content=b"<html>Not a PDF</html>",
                headers={"Content-Type": "text/html"},
            )

        monkeypatch.setattr(requests, "get", fake_get)
        result = _download_and_attach_pdf(zot, "ITEM1", "https://x.com/f.pdf",
                                          "10.1234/test", dummy_ctx)
        assert result is None
        assert len(zot.attachments) == 0

    def test_download_too_small(self, monkeypatch, dummy_ctx):
        """Response < 1000 bytes -> file NOT attached."""
        zot = _AttachZotero()
        _allow_ssrf_guard(monkeypatch)
        tiny_content = b"%PDF-1.4 tiny"  # well under 1000 bytes

        def fake_get(url, **kwargs):
            return _FakeHTTPResponse(
                200, content=tiny_content,
                headers={"Content-Type": "application/pdf"},
            )

        monkeypatch.setattr(requests, "get", fake_get)
        result = _download_and_attach_pdf(zot, "ITEM1", "https://x.com/f.pdf",
                                          "10.1234/test", dummy_ctx)
        assert result is None
        assert len(zot.attachments) == 0

    # -- SSRF guard ---------------------------------------------------------

    def test_download_rejects_loopback(self, monkeypatch, dummy_ctx):
        """A URL resolving to loopback is rejected before any HTTP request."""
        zot = _AttachZotero()
        called = []
        monkeypatch.setattr(requests, "get",
                            lambda *a, **k: called.append(1))
        result = _download_and_attach_pdf(
            zot, "ITEM1", "http://127.0.0.1:23119/api/users/0", "10.1/x", dummy_ctx)
        assert result is None
        assert called == []          # guard short-circuits before requests.get
        assert len(zot.attachments) == 0

    def test_download_rejects_cloud_metadata(self, monkeypatch, dummy_ctx):
        """The 169.254.169.254 cloud-metadata endpoint is rejected."""
        zot = _AttachZotero()
        called = []
        monkeypatch.setattr(requests, "get", lambda *a, **k: called.append(1))
        result = _download_and_attach_pdf(
            zot, "ITEM1", "http://169.254.169.254/latest/meta-data/", "10.1/x", dummy_ctx)
        assert result is None
        assert called == []

    def test_download_rejects_private_range(self, monkeypatch, dummy_ctx):
        """An RFC1918 host is rejected."""
        zot = _AttachZotero()
        called = []
        monkeypatch.setattr(requests, "get", lambda *a, **k: called.append(1))
        result = _download_and_attach_pdf(
            zot, "ITEM1", "http://10.0.0.5/secret.pdf", "10.1/x", dummy_ctx)
        assert result is None
        assert called == []

    def test_download_rejects_non_http_scheme(self, monkeypatch, dummy_ctx):
        """Non-http(s) schemes (file://, gopher://) are rejected."""
        zot = _AttachZotero()
        called = []
        monkeypatch.setattr(requests, "get", lambda *a, **k: called.append(1))
        for url in ("file:///etc/passwd", "gopher://127.0.0.1/x"):
            assert _download_and_attach_pdf(zot, "ITEM1", url, "10.1/x", dummy_ctx) is None
        assert called == []

    def test_download_rejects_redirect_to_private(self, monkeypatch, dummy_ctx):
        """A public URL that 302-redirects to a private host is rejected."""
        from zotero_mcp.tools import _helpers

        # First hop (public) passes the resolver; the redirect target is a
        # literal loopback IP, which the guard rejects on re-validation.
        monkeypatch.setattr(_helpers, "_url_resolves_to_public_host",
                            lambda url: "evil.example" in url)

        def fake_get(url, **kwargs):
            return _FakeHTTPResponse(
                302, headers={"Location": "http://127.0.0.1:23119/api"})

        monkeypatch.setattr(requests, "get", fake_get)
        result = _download_and_attach_pdf(
            zot := _AttachZotero(), "ITEM1", "https://evil.example/p.pdf",
            "10.1/x", dummy_ctx)
        assert result is None
        assert len(zot.attachments) == 0


# ---------------------------------------------------------------------------
# _try_attach_oa_pdf (full cascade)
# ---------------------------------------------------------------------------

class TestTryAttachOaPdf:

    def test_cascade_order(self, monkeypatch, dummy_ctx):
        """All sources return None except PMC (last) -> cascade reaches it."""
        zot = _AttachZotero()
        doi = "10.1234/cascade"
        call_log = []

        # Make Unpaywall, S2 return nothing; arXiv has no crossref metadata
        def fake_get(url, **kwargs):
            call_log.append(url)
            # Unpaywall
            if "unpaywall.org" in url:
                return _FakeHTTPResponse(200, json_data={
                    "best_oa_location": None, "oa_locations": [],
                })
            # Semantic Scholar
            if "semanticscholar.org" in url:
                return _FakeHTTPResponse(200, json_data={"openAccessPdf": None})
            # NCBI ID converter -> return a PMCID
            if "ncbi.nlm.nih.gov/tools/idconv" in url:
                return _FakeHTTPResponse(200, json_data={
                    "records": [{"pmcid": "PMC9999999"}],
                })
            # Actual PDF download from PMC URL
            if "pmc.ncbi.nlm.nih.gov/articles" in url:
                return _FakeHTTPResponse(
                    200,
                    content=b"%PDF-1.4 " + b"x" * 2000,
                    headers={"Content-Type": "application/pdf"},
                )
            return _FakeHTTPResponse(404)

        monkeypatch.setattr(requests, "get", fake_get)
        result = _try_attach_oa_pdf(zot, "ITEM1", doi, dummy_ctx,
                                    crossref_metadata=None)
        assert "PDF attached" in result
        assert "PubMed Central" in result
        assert len(zot.attachments) == 1

    def test_cascade_all_fail(self, monkeypatch, dummy_ctx):
        """All sources return None -> message includes 'no open-access PDF found'."""
        zot = _AttachZotero()

        def fake_get(url, **kwargs):
            # Every API returns empty/no-match responses
            if "unpaywall.org" in url:
                return _FakeHTTPResponse(200, json_data={
                    "best_oa_location": None, "oa_locations": [],
                })
            if "semanticscholar.org" in url:
                return _FakeHTTPResponse(200, json_data={"openAccessPdf": None})
            if "ncbi.nlm.nih.gov" in url:
                return _FakeHTTPResponse(200, json_data={"records": []})
            return _FakeHTTPResponse(404)

        monkeypatch.setattr(requests, "get", fake_get)
        result = _try_attach_oa_pdf(zot, "ITEM1", "10.1234/nope", dummy_ctx,
                                    crossref_metadata=None)
        assert "no open-access PDF found" in result
        assert len(zot.attachments) == 0


class TestPublisherPagePdfUrl:
    """``citation_pdf_url`` is how the browser connector finds a PDF.

    The aggregators in the cascade answer "is there a copy of this
    somewhere?". A publisher page answers "here is the version of record",
    and it is the only source that knows about a paper the aggregators have
    never indexed. The reader was already extracting the tag
    (``html_metadata.EmbeddedMetadata.pdf_url``); nothing consumed it.
    """

    PDF = b"%PDF-1.4 " + b"x" * 2000

    def _publisher_only(self, publisher_url):
        """Every aggregator draws a blank; only the publisher has the PDF."""
        def fake_get(url, **kwargs):
            if url == publisher_url:
                return _FakeHTTPResponse(
                    200, content=self.PDF,
                    headers={"Content-Type": "application/pdf"},
                )
            if "unpaywall.org" in url:
                return _FakeHTTPResponse(200, json_data={
                    "best_oa_location": None, "oa_locations": []})
            if "semanticscholar.org" in url:
                return _FakeHTTPResponse(200, json_data={"openAccessPdf": None})
            if "ncbi.nlm.nih.gov" in url:
                return _FakeHTTPResponse(200, json_data={"records": []})
            return _FakeHTTPResponse(404)
        return fake_get

    def test_page_pdf_is_attached_when_no_aggregator_has_it(
        self, monkeypatch, dummy_ctx
    ):
        _allow_ssrf_guard(monkeypatch)
        zot = _AttachZotero()
        url = "https://example.com/article/1/pdf"
        monkeypatch.setattr(requests, "get", self._publisher_only(url))

        result = _try_attach_oa_pdf(zot, "ITEM1", "10.1234/x", dummy_ctx,
                                    page_pdf_url=url)

        assert "PDF attached" in result
        assert "publisher" in result
        assert len(zot.attachments) == 1

    def test_unpaywall_wins_when_it_has_a_copy(self, monkeypatch, dummy_ctx):
        """Both have a copy. Unpaywall's is known-open-access; the
        publisher's could be an access-denied stub, and the cascade returns
        on the first success, so a stub would win outright."""
        _allow_ssrf_guard(monkeypatch)
        zot = _AttachZotero()
        page_url = "https://example.com/article/1/pdf"

        def fake_get(url, **kwargs):
            if "unpaywall.org" in url:
                return _FakeHTTPResponse(200, json_data={
                    "best_oa_location": {"url_for_pdf": "https://example.org/x.pdf"},
                    "oa_locations": [],
                })
            if url in (page_url, "https://example.org/x.pdf"):
                return _FakeHTTPResponse(
                    200, content=self.PDF,
                    headers={"Content-Type": "application/pdf"},
                )
            return _FakeHTTPResponse(404)

        monkeypatch.setattr(requests, "get", fake_get)
        result = _try_attach_oa_pdf(zot, "ITEM1", "10.1234/x", dummy_ctx,
                                    page_pdf_url=page_url)
        assert "Unpaywall" in result
        assert "publisher" not in result

    def test_a_paywalled_page_pdf_falls_through(self, monkeypatch, dummy_ctx):
        """The publisher's copy is the one most likely to be paywalled, so
        failing there must cost nothing but the attempt."""
        _allow_ssrf_guard(monkeypatch)
        zot = _AttachZotero()

        def fake_get(url, **kwargs):
            if "example.com" in url:
                # A paywall serves the login page, not the PDF.
                return _FakeHTTPResponse(200, content=b"<html>Sign in</html>",
                                         headers={"Content-Type": "text/html"})
            if "unpaywall.org" in url:
                return _FakeHTTPResponse(200, json_data={
                    "best_oa_location": {"url_for_pdf": "https://example.org/x.pdf"},
                    "oa_locations": [],
                })
            if "example.org" in url:
                return _FakeHTTPResponse(
                    200, content=self.PDF,
                    headers={"Content-Type": "application/pdf"},
                )
            return _FakeHTTPResponse(404)

        monkeypatch.setattr(requests, "get", fake_get)
        result = _try_attach_oa_pdf(
            zot, "ITEM1", "10.1234/x", dummy_ctx,
            page_pdf_url="https://example.com/article/1/pdf",
        )
        assert "PDF attached" in result
        assert "Unpaywall" in result

    def test_no_page_url_leaves_the_cascade_as_it_was(
        self, monkeypatch, dummy_ctx
    ):
        _allow_ssrf_guard(monkeypatch)
        zot = _AttachZotero()
        monkeypatch.setattr(requests, "get",
                            self._publisher_only("https://example.com/unused.pdf"))
        result = _try_attach_oa_pdf(zot, "ITEM1", "10.1234/x", dummy_ctx)
        assert "no open-access PDF found" in result
        assert "publisher" not in result

    def test_the_failure_message_is_derived_from_the_source_table(
        self, monkeypatch, dummy_ctx
    ):
        """The list used to be a hardcoded string beside the source table,
        free to drift away from it -- and it had: the table said "PubMed
        Central" and the sentence said "PMC".

        This pins the table's spelling, which catches that specific drift.
        It cannot prove derivation in general: a corrected hardcoded string
        would satisfy it too, and the one source that would distinguish
        them -- the publisher's page -- can never appear in this sentence,
        because supplying its URL always populates found_urls and takes the
        other branch.
        """
        zot = _AttachZotero()
        monkeypatch.setattr(requests, "get",
                            lambda url, **kw: _FakeHTTPResponse(404))
        result = _try_attach_oa_pdf(zot, "ITEM1", "10.1234/x", dummy_ctx)
        assert "PubMed Central" in result, result
        for expected in ("Unpaywall", "OpenAlex", "arXiv", "Semantic Scholar"):
            assert expected in result, result


class TestOpenAlexInTheCascade:
    """OpenAlex goes after Unpaywall and before the publisher's page."""

    PDF = b"%PDF-1.4 " + b"x" * 2000
    PAGE_URL = "https://journals.example.com/previewpdf/1.pdf"
    REPO_URL = "http://repo.example.org/1/paper.pdf"

    def _fake_get(self, unpaywall_pdf=None, openalex_urls=(), pdf_urls=(), log=None):
        def fake_get(url, **kwargs):
            if log is not None:
                log.append(url)
            if "unpaywall.org" in url:
                best = {"url_for_pdf": unpaywall_pdf} if unpaywall_pdf else None
                return _FakeHTTPResponse(200, json_data={
                    "best_oa_location": best, "oa_locations": []})
            if "openalex.org" in url:
                return _FakeHTTPResponse(200, json_data={
                    "best_oa_location": None,
                    "locations": [{"is_oa": False, "pdf_url": u} for u in openalex_urls],
                })
            if url in pdf_urls:
                return _FakeHTTPResponse(
                    200, content=self.PDF,
                    headers={"Content-Type": "application/pdf"},
                )
            return _FakeHTTPResponse(404)
        return fake_get

    def test_openalex_before_the_publisher_page(self, monkeypatch, dummy_ctx):
        """Both download. The publisher's citation_pdf_url is the one that
        can be a preview, so the repository copy has to be tried first."""
        _allow_ssrf_guard(monkeypatch)
        zot = _AttachZotero()
        monkeypatch.setattr(requests, "get", self._fake_get(
            openalex_urls=[self.REPO_URL],
            pdf_urls=(self.REPO_URL, self.PAGE_URL)))
        result = _try_attach_oa_pdf(zot, "ITEM1", "10.1234/x", dummy_ctx,
                                    page_pdf_url=self.PAGE_URL)
        assert "PDF attached (source: OpenAlex)" in result
        assert len(zot.attachments) == 1

    def test_unpaywall_before_openalex(self, monkeypatch, dummy_ctx):
        _allow_ssrf_guard(monkeypatch)
        zot = _AttachZotero()
        log = []
        monkeypatch.setattr(requests, "get", self._fake_get(
            unpaywall_pdf="https://oa.example.org/x.pdf",
            openalex_urls=[self.REPO_URL],
            pdf_urls=("https://oa.example.org/x.pdf", self.REPO_URL), log=log))
        result = _try_attach_oa_pdf(zot, "ITEM1", "10.1234/x", dummy_ctx)
        assert "Unpaywall" in result
        assert not any("openalex.org" in u for u in log)

    def test_later_openalex_locations_are_tried(self, monkeypatch, dummy_ctx):
        _allow_ssrf_guard(monkeypatch)
        zot = _AttachZotero()
        monkeypatch.setattr(requests, "get", self._fake_get(
            openalex_urls=["https://pub.example/paywalled.pdf", self.REPO_URL],
            pdf_urls=(self.REPO_URL,)))
        result = _try_attach_oa_pdf(zot, "ITEM1", "10.1234/x", dummy_ctx)
        assert "PDF attached (source: OpenAlex)" in result

    def test_a_url_both_sources_name_is_fetched_once(self, monkeypatch, dummy_ctx):
        _allow_ssrf_guard(monkeypatch)
        zot = _AttachZotero()
        log = []
        monkeypatch.setattr(requests, "get", self._fake_get(
            openalex_urls=[self.PAGE_URL], log=log))
        result = _try_attach_oa_pdf(zot, "ITEM1", "10.1234/x", dummy_ctx,
                                    page_pdf_url=self.PAGE_URL)
        assert "PDF attached" not in result
        assert log.count(self.PAGE_URL) == 1


def _pdf_with_pages(n_pages):
    """A minimal, valid PDF with ``n_pages`` blank pages, padded past the
    1000-byte floor in ``_download_and_attach_pdf``."""
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + i} 0 R" for i in range(n_pages))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objs += [b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>"] * n_pages
    out = bytearray(b"%PDF-1.4\n%" + b"x" * 1200 + b"\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    return bytes(out)


class TestPreviewPdfIsNotAttached:
    """A publisher's citation_pdf_url can serve a first-page preview: a real
    PDF of the right content type, above the size floor. AMS does this for
    10.1175/jcli-d-12-00821.1, pages 2137-2143, with a 2-page PDF. The page
    range on the item is the only thing that gives it away."""

    PAGE_URL = "https://journals.example.com/previewpdf/1.pdf"
    PMC_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/pdf/"

    def _fake_get(self, pdfs):
        def fake_get(url, **kwargs):
            if url in pdfs:
                return _FakeHTTPResponse(
                    200, content=pdfs[url],
                    headers={"Content-Type": "application/pdf"},
                )
            if "unpaywall.org" in url:
                return _FakeHTTPResponse(200, json_data={
                    "best_oa_location": None, "oa_locations": []})
            if "openalex.org" in url:
                return _FakeHTTPResponse(200, json_data={"locations": []})
            if "semanticscholar.org" in url:
                return _FakeHTTPResponse(200, json_data={"openAccessPdf": None})
            if "ncbi.nlm.nih.gov/tools/idconv" in url:
                return _FakeHTTPResponse(200, json_data={
                    "records": [{"pmcid": "PMC1234567"}]})
            return _FakeHTTPResponse(404)
        return fake_get

    def test_preview_is_skipped_for_the_next_source(self, monkeypatch):
        _allow_ssrf_guard(monkeypatch)
        zot = _AttachZotero()
        messages = []

        class _Ctx(DummyContext):
            def info(self, msg, *_args, **_kwargs):
                messages.append(msg)

        ctx = _Ctx()
        monkeypatch.setattr(requests, "get", self._fake_get({
            self.PAGE_URL: _pdf_with_pages(2),
            self.PMC_URL: _pdf_with_pages(7),
        }))
        result = _try_attach_oa_pdf(zot, "ITEM1", "10.1234/x", ctx,
                                    page_pdf_url=self.PAGE_URL,
                                    pages="2137-2143")
        assert "PDF attached (source: PubMed Central)" in result
        assert len(zot.attachments) == 1
        assert any("likely a preview" in m for m in messages)

    def test_a_preview_alone_is_never_uploaded(self, monkeypatch, dummy_ctx):
        _allow_ssrf_guard(monkeypatch)
        zot = _AttachZotero()
        monkeypatch.setattr(requests, "get", self._fake_get({
            self.PAGE_URL: _pdf_with_pages(2),
        }))
        result = _try_attach_oa_pdf(zot, "ITEM1", "10.1234/x", dummy_ctx,
                                    page_pdf_url=self.PAGE_URL,
                                    pages="2137-2143")
        assert "PDF attached" not in result
        assert zot.attachments == []

    def test_crossref_page_is_used_when_the_item_has_none(self, monkeypatch, dummy_ctx):
        _allow_ssrf_guard(monkeypatch)
        zot = _AttachZotero()
        monkeypatch.setattr(requests, "get", self._fake_get({
            self.PAGE_URL: _pdf_with_pages(2),
        }))
        result = _try_attach_oa_pdf(zot, "ITEM1", "10.1234/x", dummy_ctx,
                                    crossref_metadata={"page": "2137-2143"},
                                    page_pdf_url=self.PAGE_URL)
        assert "PDF attached" not in result
        assert zot.attachments == []

    @pytest.mark.parametrize("pages", [None, "", "e1234", "2137", "1-5, 9-12", "xii-xx"])
    def test_short_pdf_is_kept_without_a_page_range(self, monkeypatch, dummy_ctx, pages):
        _allow_ssrf_guard(monkeypatch)
        zot = _AttachZotero()
        monkeypatch.setattr(requests, "get", self._fake_get({
            self.PAGE_URL: _pdf_with_pages(2),
        }))
        result = _try_attach_oa_pdf(zot, "ITEM1", "10.1234/x", dummy_ctx,
                                    page_pdf_url=self.PAGE_URL, pages=pages)
        assert "PDF attached (source: the publisher's page)" in result
        assert len(zot.attachments) == 1

    def test_full_pdf_is_kept_with_a_page_range(self, monkeypatch, dummy_ctx):
        _allow_ssrf_guard(monkeypatch)
        zot = _AttachZotero()
        monkeypatch.setattr(requests, "get", self._fake_get({
            self.PAGE_URL: _pdf_with_pages(7),
        }))
        result = _try_attach_oa_pdf(zot, "ITEM1", "10.1234/x", dummy_ctx,
                                    page_pdf_url=self.PAGE_URL,
                                    pages="2137-2143")
        assert "PDF attached (source: the publisher's page)" in result

    def test_a_pdf_that_cannot_be_counted_is_kept(self, monkeypatch, dummy_ctx):
        """Page counting is a heuristic on top of the existing checks, so a
        file the counter cannot parse is attached as before."""
        _allow_ssrf_guard(monkeypatch)
        zot = _AttachZotero()
        monkeypatch.setattr(requests, "get", self._fake_get({
            self.PAGE_URL: b"%PDF-1.4 " + b"x" * 2000,
        }))
        result = _try_attach_oa_pdf(zot, "ITEM1", "10.1234/x", dummy_ctx,
                                    page_pdf_url=self.PAGE_URL,
                                    pages="2137-2143")
        assert "PDF attached" in result


class TestExpectedPageCount:

    @pytest.mark.parametrize("pages,expected", [
        ("2137-2143", 7),
        ("2137\u20132143", 7),
        ("2137--2143", 7),
        (" 2137 - 2143 ", 7),
        ("1189-95", 7),
        ("S12-S20", 9),
        ("S12-20", 9),
    ])
    def test_ranges(self, pages, expected):
        from zotero_mcp.tools._helpers import _expected_page_count
        assert _expected_page_count(pages) == expected

    @pytest.mark.parametrize("pages", [
        None, "", "2137", "e1234", "1-5, 9-12", "xii-xx", "A1-B9", "1-1", "1-99999",
    ])
    def test_not_a_usable_range(self, pages):
        from zotero_mcp.tools._helpers import _expected_page_count
        assert _expected_page_count(pages) is None


class TestAThirdPartyUrlIsNotTrusted:
    """`citation_pdf_url` is the first source in this cascade that comes
    from arbitrary third-party HTML rather than a known API over a known
    host. Two sinks had been relying on their inputs being curated."""

    def test_a_non_http_url_is_never_linked(self, dummy_ctx):
        """`attach_mode="linked_url"` never fetches, so it never reaches the
        SSRF guard. A file:// in a meta tag would be written to the library
        verbatim."""
        zot = _AttachZotero()
        result = _try_attach_oa_pdf(zot, "ITEM1", "10.1234/x", dummy_ctx,
                                    attach_mode="linked_url",
                                    page_pdf_url="file:///etc/passwd")
        assert zot.created == [] or all(
            i.get("url") != "file:///etc/passwd" for i in zot.created
        )
        assert "PDF linked" not in result

    def test_html_served_as_application_pdf_is_not_attached(
        self, monkeypatch, dummy_ctx
    ):
        """Content-Type is the server's claim about the bytes, not the
        bytes. Interstitials and viewer endpoints serve HTML under
        application/pdf."""
        _allow_ssrf_guard(monkeypatch)
        zot = _AttachZotero()

        def fake_get(url, **kwargs):
            return _FakeHTTPResponse(
                200,
                content=b"<html>" + b"<p>Sign in</p>" * 200 + b"</html>",
                headers={"Content-Type": "application/pdf"},
            )

        monkeypatch.setattr(requests, "get", fake_get)
        result = _try_attach_oa_pdf(zot, "ITEM1", "10.1234/x", dummy_ctx,
                                    page_pdf_url="https://example.com/a.pdf")
        assert len(zot.attachments) == 0
        assert "PDF attached" not in result
