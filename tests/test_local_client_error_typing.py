"""Regression tests for #512: local-API errors must reach pyzotero's typed
error handling.

pyzotero funnels every response through one place::

    try:
        resp.raise_for_status()
    except httpx2.HTTPError as exc:
        error_handler(self, resp, exc)

Local mode is the only path where *we* build the client rather than letting
pyzotero build its own, so it is the only one that can hand pyzotero responses
from a different HTTP library than the ``httpx2`` (pre-1.15: ``httpx``) it
matches on. Disjoint hierarchies, so nothing is caught: ``error_handler`` never
runs, no server backoff is recorded, no 429 is retried, and every error status
arrives as a raw ``HTTPStatusError`` instead of the pyzotero exception the call
sites are written against.

The damage is worse than the exception type. ``find_existing_items``
deliberately re-raises a throttled search rather than degrading it to "no
match", because "no match" makes ``if_exists='file'`` create a duplicate of an
item that is already there. A leaked ``HTTPStatusError`` misses that clause and
lands in the generic ``except Exception`` below it, so the duplicate appears
silently.

Nothing here pins a *library*: the responses are built from whatever library
the client under test produces, which is exactly the coupling that has to hold.
"""

import sys

import pytest
from conftest import DummyContext, pyzotero_http_module
from pyzotero import zotero
from pyzotero.zotero_errors import (
    PreConditionFailedError,
    ResourceNotFoundError,
    TooManyRetriesError,
    UserNotAuthorisedError,
)

from zotero_mcp import client as zclient
from zotero_mcp.tools import _helpers

_ITEM = [{"key": "EXIST001", "version": 1,
          "data": {"itemType": "journalArticle", "DOI": "10.1234/test"}}]

# A Zotero rate-limit response. Backoff: 0 keeps pyzotero's real wait between
# retries free rather than stubbing out the waiting the fix exists to restore.
_THROTTLED = (429, {"Content-Type": "text/plain", "Backoff": "0"},
              b"Too many requests. Slow down")
_FOUND = (200, {"Content-Type": "application/json"}, _ITEM)

_SEARCH = {"q": "10.1234/test", "qmode": "everything", "limit": 50}


def _local_zotero(responses):
    """A local-mode Zotero built the way the server builds it, answering from
    *responses* instead of reaching Zotero.

    Each entry is ``(status, headers, body)``; the last one repeats once the
    list is spent, so a single entry means "always this". The responses come
    from the client's *own* HTTP library rather than a fixed import, because
    that is precisely what production does — pyzotero sees whatever the client
    we handed it produces, and recognises only its own library's.

    ``_transport`` is private, but assigning it is how both libraries' mock
    transports are reached on an already-constructed client; the #160 test
    reads the same attribute.
    """
    client = zclient._make_local_http_client()
    http = sys.modules[type(client).__module__.partition(".")[0]]
    remaining = list(responses)

    def handler(_request):
        status, headers, body = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        payload = {"json": body} if isinstance(body, list) else {"content": body}
        return http.Response(status, headers=headers, **payload)

    client._transport = http.MockTransport(handler)
    return zotero.Zotero(library_id="0", library_type="user", api_key=None,
                         local=True, client=client)


def test_local_client_is_the_client_pyzotero_would_have_built():
    """The invariant behind everything below: our client has to be an instance
    of the same class pyzotero constructs for itself, or its error handling
    cannot see what that client returns."""
    client = zclient._make_local_http_client()
    try:
        assert type(client) is pyzotero_http_module().Client
    finally:
        client.close()


def test_persistent_throttling_raises_rather_than_leaking_a_raw_http_error():
    zot = _local_zotero([_THROTTLED])

    with pytest.raises(TooManyRetriesError):
        zot.items(**_SEARCH)


def test_transient_throttling_is_retried_and_recovers():
    """Not just the exception type: with the error unrecognised, pyzotero's
    retry loop never sees a 429 either, so a throttled local client hammers
    instead of backing off and retrying."""
    zot = _local_zotero([_THROTTLED, _FOUND])

    assert [i["key"] for i in zot.items(**_SEARCH)] == ["EXIST001"]


@pytest.mark.parametrize("status, expected", [
    (403, UserNotAuthorisedError),
    (404, ResourceNotFoundError),
    (412, PreConditionFailedError),
])
def test_error_statuses_reach_pyzoteros_typed_handlers(status, expected):
    """429 is the damaging one, but every typed handler is bypassed the same
    way — e.g. the PreConditionFailedError retry in _helpers.update_with_retry."""
    zot = _local_zotero([(status, {"Content-Type": "text/plain"}, b"nope")])

    with pytest.raises(expected):
        zot.items(**_SEARCH)


def test_throttled_dedup_search_does_not_read_as_no_match():
    """The user-visible consequence: a throttled search that reads as "nothing
    found" makes if_exists='file' duplicate an item that is present."""
    zot = _local_zotero([_THROTTLED])

    with pytest.raises(TooManyRetriesError):
        _helpers.find_existing_items(zot, doi="10.1234/test", ctx=DummyContext())
