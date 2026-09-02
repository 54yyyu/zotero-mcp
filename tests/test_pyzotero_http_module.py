"""pyzotero 1.15 replaced httpx with httpx2; our local-mode client must follow.

For the local Zotero server we hand pyzotero our own HTTP client, pinned to
HTTP/1.1 (#160). pyzotero turns error responses into its exception classes by
calling ``raise_for_status`` inside ``except <its library>.HTTPError``. A client
from the *other* library returns responses whose errors that clause does not
catch, so with pyzotero 1.15.1 and an ``httpx.Client`` every non-2xx from the
local API escaped as a raw ``httpx.HTTPStatusError`` — a 404 was no longer a
``ResourceNotFoundError``, a 412 no longer a ``PreConditionFailedError`` — and
nothing downstream that catches pyzotero's errors saw them.

These tests are written against whichever library pyzotero imports, so they
hold on both sides of the 1.15 switch.
"""

import sys

import pytest
from pyzotero import zotero_errors as ze
from pyzotero.zotero import Zotero

from zotero_mcp import client as zclient


def _module_pyzotero_uses():
    """Read it off a real pyzotero instance, independently of the helper under test."""
    zot = Zotero(library_id="1", library_type="user", api_key="x" * 24)
    return sys.modules[type(zot.client).__module__.split(".")[0]]


def test_helper_names_the_library_pyzotero_uses():
    assert zclient._pyzotero_http() is _module_pyzotero_uses()


def test_local_client_is_the_kind_pyzotero_handles():
    c = zclient._make_local_http_client()
    try:
        assert isinstance(c, _module_pyzotero_uses().Client)
    finally:
        c.close()


def test_error_responses_still_become_pyzotero_errors():
    """The user-visible regression: a 404 through our client must be a
    ResourceNotFoundError, not the HTTP library's own status error."""
    http = zclient._pyzotero_http()

    def not_found(request):
        return http.Response(404, headers={"Content-Type": "text/plain"},
                             content=b"Not found", request=request)

    zot = Zotero(library_id="1", library_type="user", api_key="x" * 24)
    zot.client = http.Client(transport=http.MockTransport(not_found))

    with pytest.raises(ze.ResourceNotFoundError):
        zot.item("NOPE")
