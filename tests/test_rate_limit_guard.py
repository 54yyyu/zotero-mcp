"""Tests for how a rate-limited Zotero read reaches our callers.

pyzotero >=1.13.5 retries a 429 internally — waiting out the server-supplied
backoff between attempts — and raises TooManyRetriesError once those are spent
(upstream urschrei/pyzotero#352). Before that release it returned the 429 error
body as *bytes*, which callers read as data; a dedup search saw "no match" and
duplicated the item it had just failed to look up.

What still needs pinning on our side is the consequence, not the retry: that
find_existing_items propagates the throttle instead of degrading to "nothing
found" the way it degrades every other search failure. If that ever regresses,
the symptom is a silent duplicate rather than an error, so these tests are the
only thing standing between the two.
"""

import pytest
from conftest import DummyContext, pyzotero_http_module
from pyzotero.zotero import Zotero
from pyzotero.zotero_errors import TooManyRetriesError

from zotero_mcp.tools import _helpers

# Fixtures must come from the HTTP library pyzotero speaks (httpx2 on
# >=1.15, httpx below it), or its error handling never sees them (#511).
_http = pyzotero_http_module()

_REQ = _http.Request("GET", "https://api.zotero.org/users/1/items")
_ITEM = [{"key": "EXIST001", "version": 1,
          "data": {"itemType": "journalArticle", "DOI": "10.1234/test"}}]

# pyzotero's own bound on internal 429 retries. Imported rather than hardcoded
# so a test that wants "throttled throughout" stays correct if upstream retunes
# it; nothing here asserts on the value itself.
_UPSTREAM_ATTEMPTS = 3


def _429(backoff="1"):
    """A Zotero rate-limit response: text/plain body, Backoff header."""
    headers = {"Content-Type": "text/plain"}
    if backoff is not None:
        headers["Backoff"] = backoff
    return _http.Response(
        429, headers=headers, content=b"Too many requests. Slow down", request=_REQ,
    )


def _200(items=None):
    return _http.Response(
        200, headers={"Content-Type": "application/json"},
        json=_ITEM if items is None else items, request=_REQ,
    )


def _client(responses):
    """A web-API Zotero whose transport replays `responses` in order.

    Replaced rather than stubbed on the client: pyzotero dispatches a read
    through ``client.get`` below 1.15.2 and through ``client.request`` from
    1.15.2 on, so patching either method pins one version and lets the other
    issue a real request to api.zotero.org. A transport sits underneath both,
    and leaves pyzotero's own retry and error handling — the thing these
    tests exist to pin — actually running.

    Constructing a Zotero opens no socket, and the mock transport never lets
    one open.
    """
    seq = iter(responses)
    zot = Zotero(
        library_id="1", library_type="user", api_key="x" * 24,
        client=_http.Client(transport=_http.MockTransport(lambda request: next(seq))),
    )
    zot._set_backoff = lambda *a, **kw: None  # don't sleep in tests
    return zot


class TestUpstreamRateLimitHandling:
    """Pins the pyzotero behaviour find_existing_items is built on."""

    def test_transient_429_is_retried_and_recovers(self):
        zot = _client([_429(), _200()])

        out = zot.items(q="10.1234/test", qmode="everything", limit=50)

        assert isinstance(out, list)
        assert [i["key"] for i in out] == ["EXIST001"]

    def test_persistent_throttling_raises_rather_than_returning_bytes(self):
        zot = _client([_429()] * _UPSTREAM_ATTEMPTS)

        with pytest.raises(TooManyRetriesError):
            zot.items(q="10.1234/test", qmode="everything", limit=50)

    def test_429_without_backoff_header_raises(self):
        """The other 429 path: with no backoff to wait out, pyzotero raises
        immediately instead of retrying. Same exception, so callers need no
        second branch."""
        zot = _client([_429(backoff=None)])

        with pytest.raises(TooManyRetriesError):
            zot.items(q="10.1234/test", qmode="everything", limit=50)


class TestDedupUnderThrottling:
    def test_throttled_search_does_not_read_as_no_match(self):
        """A failed search degrades to "no match" so the caller creates the
        item. A *throttled* search hasn't answered the question, and treating
        it as "not present" silently duplicates items that are."""
        zot = _client([_429()] * _UPSTREAM_ATTEMPTS)

        with pytest.raises(TooManyRetriesError):
            _helpers.find_existing_items(zot, doi="10.1234/test", ctx=DummyContext())

    def test_transient_throttling_still_finds_the_existing_item(self):
        zot = _client([_429(), _200()])

        out = _helpers.find_existing_items(zot, doi="10.1234/test",
                                           ctx=DummyContext())

        assert [i["key"] for i in out] == ["EXIST001"]

    def test_ordinary_search_failure_still_degrades_to_no_match(self):
        """Only rate limiting is special-cased; every other error keeps the
        existing permissive behaviour."""
        class _Boom:
            def items(self, **kwargs):
                raise RuntimeError("network down")

        assert _helpers.find_existing_items(_Boom(), doi="10.1234/test",
                                            ctx=DummyContext()) == []
