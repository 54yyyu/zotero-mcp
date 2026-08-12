"""Tests for the HTTP 429 guard on Zotero reads.

pyzotero's error_handler special-cases 429: it records the server-supplied
backoff and returns instead of raising, documented as "leaving the caller to
retry" — but no caller does. The 429 response then reaches retrieve()'s format
dispatch, which keys off the response Content-Type; Zotero sends text/plain for
error bodies, which matches none of the JSON/Atom/BibTeX branches and falls
through to `return retrieved.content`.

So a throttled read returns the error body as *bytes*. Iterating bytes yields
ints, which is how a dedup check ends up doing int.get("data"), and len() reports
the byte count as a plausible-looking result count.
"""

import httpx
import pytest
from conftest import DummyContext

from zotero_mcp.client import _MAX_RATE_LIMIT_RETRIES, ZoteroRateLimitedError, _ZoteroClient
from zotero_mcp.tools import _helpers

_REQ = httpx.Request("GET", "https://api.zotero.org/users/1/items")
_ITEM = [{"key": "EXIST001", "version": 1,
          "data": {"itemType": "journalArticle", "DOI": "10.1234/test"}}]


def _429():
    """A Zotero rate-limit response: text/plain body, Backoff header."""
    return httpx.Response(
        429,
        headers={"Content-Type": "text/plain", "Backoff": "1"},
        content=b"Too many requests. Slow down",
        request=_REQ,
    )


def _200(items=None):
    return httpx.Response(
        200, headers={"Content-Type": "application/json"},
        json=_ITEM if items is None else items, request=_REQ,
    )


def _client(responses):
    """A guarded client whose transport replays `responses` in order."""
    zot = _ZoteroClient(library_id="1", library_type="user", api_key="x" * 24)
    seq = iter(responses)
    zot.client.get = lambda url, params=None, timeout=None, **kw: next(seq)
    zot._set_backoff = lambda *a, **kw: None  # don't sleep in tests
    return zot


class TestRateLimitGuard:
    def test_transient_429_is_retried_and_recovers(self):
        zot = _client([_429(), _200()])

        out = zot.items(q="10.1234/test", qmode="everything", limit=50)

        assert isinstance(out, list)
        assert [i["key"] for i in out] == ["EXIST001"]

    def test_persistent_429_raises_rather_than_returning_bytes(self):
        zot = _client([_429()] * _MAX_RATE_LIMIT_RETRIES)

        with pytest.raises(ZoteroRateLimitedError):
            zot.items(q="10.1234/test", qmode="everything", limit=50)

    def test_retry_is_bounded(self):
        calls = []
        zot = _ZoteroClient(library_id="1", library_type="user", api_key="x" * 24)

        def _get(url, params=None, timeout=None, **kw):
            calls.append(url)
            return _429()

        zot.client.get = _get
        zot._set_backoff = lambda *a, **kw: None

        with pytest.raises(ZoteroRateLimitedError):
            zot.items(q="10.1234/test", qmode="everything", limit=50)
        assert len(calls) == _MAX_RATE_LIMIT_RETRIES

    def test_unguarded_pyzotero_would_have_returned_bytes(self):
        """Pins the upstream behaviour this guard exists for, so the guard
        can't be quietly removed as redundant."""
        from pyzotero.zotero import Zotero

        zot = Zotero(library_id="1", library_type="user", api_key="x" * 24)
        zot.client.get = lambda url, params=None, timeout=None, **kw: _429()
        zot._set_backoff = lambda *a, **kw: None

        out = zot.items(q="10.1234/test", qmode="everything", limit=50)

        assert isinstance(out, bytes)
        assert all(isinstance(x, int) for x in out)
        with pytest.raises(AttributeError, match="'int' object has no attribute 'get'"):
            for item in out:
                item.get("data", {})


class TestDedupUnderThrottling:
    def test_throttled_search_does_not_read_as_no_match(self):
        """A failed search degrades to "no match" so the caller creates the
        item. A *throttled* search hasn't answered the question, and treating
        it as "not present" silently duplicates items that are."""
        zot = _client([_429()] * _MAX_RATE_LIMIT_RETRIES)

        with pytest.raises(ZoteroRateLimitedError):
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
