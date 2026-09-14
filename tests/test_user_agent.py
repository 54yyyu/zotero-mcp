"""Every outbound client this package wires up must identify itself the same way.

`_fetch_embedded_metadata` once sent
``Mozilla/5.0 (compatible; zotero-mcp/1.0; ...)`` while the three API
clients beside it sent an honest ``zotero-mcp/1.0 (...)``. Four copies of
one string is how that drift went unnoticed, and it cost real pages:
SpringerLink's WAF challenges a Mozilla-prefixed UA whose connection is not
a browser's, so the disguise was served a challenge page where the honest
form was served the article.

The WAF's behaviour is third-party, undertested (one publisher, one IP, one
day) and will drift, so it is evidence for the change rather than something
asserted here. What is asserted is what is ours: nothing claims to be a
browser, and every wired call site sends the one constant.

This file is deliberately not inside a feature-specific test module. The
invariant is cross-cutting, and the next person adding an outbound client
should find it.
"""

from unittest.mock import MagicMock, patch

import pytest
from conftest import DummyContext

from zotero_mcp import utils as _utils
from zotero_mcp.tools import _helpers, write


def _captured(mock_get):
    """The User-Agent from a captured requests.get call, however it was passed."""
    headers = mock_get.call_args.kwargs.get("headers") or {}
    return headers.get("User-Agent")


def _html_response(body="<html><head></head><body></body></html>"):
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"Content-Type": "text/html"}
    resp.encoding = "utf-8"
    resp.raise_for_status = MagicMock()
    resp.iter_content = MagicMock(return_value=[body.encode()])
    resp.close = MagicMock()
    return resp


def _json_response(payload):
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"Content-Type": "application/json"}
    resp.json = MagicMock(return_value=payload)
    resp.raise_for_status = MagicMock()
    return resp


# Each entry: a label, and a callable that performs one outbound request
# with requests.get patched, returning the captured User-Agent.
def _page_fetch():
    with patch("zotero_mcp.tools.write.requests.get",
               return_value=_html_response()) as m:
        write._fetch_embedded_metadata("https://example.com/a", DummyContext())
    return _captured(m)


def _crossref():
    with patch("zotero_mcp.tools.write.requests.get",
               return_value=_json_response({"message": {}})) as m:
        write._crossref_get("https://api.crossref.org/works/10.1/x", {},
                            DummyContext(), 10)
    return _captured(m)


def _openlibrary():
    with patch("zotero_mcp.tools.write.requests.get",
               return_value=_json_response({})) as m:
        write._lookup_isbn_openlibrary("9780262033848", DummyContext())
    return _captured(m)


def _google_books():
    with patch("zotero_mcp.tools.write.requests.get",
               return_value=_json_response({"totalItems": 0})) as m:
        write._lookup_isbn_google_books("9780262033848", DummyContext())
    return _captured(m)


def _publisher_pdf():
    pdf = MagicMock()
    pdf.status_code = 200
    pdf.headers = {"Content-Type": "application/pdf"}
    with patch("zotero_mcp.tools._helpers.requests.get", return_value=pdf) as m:
        _helpers._guarded_pdf_get("https://example.com/a.pdf", DummyContext())
    return _captured(m)


WIRED_CLIENTS = {
    "publisher landing page": _page_fetch,
    "CrossRef": _crossref,
    "OpenLibrary": _openlibrary,
    "Google Books": _google_books,
    "publisher PDF": _publisher_pdf,
}


@pytest.mark.parametrize("name,perform", WIRED_CLIENTS.items(), ids=list(WIRED_CLIENTS))
class TestEveryWiredClientIdentifiesItselfTheSameWay:
    def test_sends_the_shared_constant(self, name, perform):
        assert perform() == _utils.USER_AGENT, (
            f"{name} does not send the shared User-Agent"
        )

    def test_does_not_pose_as_a_browser(self, name, perform):
        ua = perform()
        assert ua is not None and not ua.startswith("Mozilla/"), (
            f"{name} claims to be a browser: {ua!r}"
        )


class TestTheConstantItself:
    def test_names_the_project(self):
        assert "zotero-mcp" in _utils.USER_AGENT

    def test_is_not_a_browser_claim(self):
        assert not _utils.USER_AGENT.startswith("Mozilla/")
