"""Unit tests for the PDF attachment cascade (_try_unpaywall, _try_arxiv_from_crossref,
_try_semantic_scholar, _try_pmc, _download_and_attach_pdf, _try_attach_oa_pdf)."""

import json

import pytest
import requests

from zotero_mcp.server import (
    _download_and_attach_pdf,
    _try_arxiv_from_crossref,
    _try_attach_oa_pdf,
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

        def fake_get(url, **kwargs):
            return _FakeHTTPResponse(
                200, content=b"<html>Not a PDF</html>",
                headers={"Content-Type": "text/html"},
            )

        monkeypatch.setattr(requests, "get", fake_get)
        result = _download_and_attach_pdf(zot, "ITEM1", "https://x.com/f.pdf",
                                          "10.1234/test", dummy_ctx)
        assert result is False
        assert len(zot.attachments) == 0

    def test_download_too_small(self, monkeypatch, dummy_ctx):
        """Response < 1000 bytes -> file NOT attached."""
        zot = _AttachZotero()
        tiny_content = b"%PDF-1.4 tiny"  # well under 1000 bytes

        def fake_get(url, **kwargs):
            return _FakeHTTPResponse(
                200, content=tiny_content,
                headers={"Content-Type": "application/pdf"},
            )

        monkeypatch.setattr(requests, "get", fake_get)
        result = _download_and_attach_pdf(zot, "ITEM1", "https://x.com/f.pdf",
                                          "10.1234/test", dummy_ctx)
        assert result is False
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
