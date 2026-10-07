"""find_related_papers and the Scite tools must not hold the Zotero API lock
across third-party network calls.

Both were wrapped whole in @with_zotero_api_lock, so every OpenAlex or Scite
request ran with the process-wide lock held: with 1.5 s of OpenAlex latency a
concurrent zotero_get_recent waited 6.25 s. The lock now covers only the reads
from the Zotero backend (the #431 pattern).
"""

import threading

from conftest import DummyContext, FakeZotero
from test_discovery import FakeResponse

from zotero_mcp import client as _client
from zotero_mcp.tools import discovery
from zotero_mcp.tools import scite as scite_tools


def _lock_is_free() -> bool:
    """True iff no thread holds the lock. Probes from another thread: the
    lock is an RLock, so a same-thread acquire always succeeds."""
    result = {}

    def probe():
        got = _client._zotero_api_lock.acquire(blocking=False)
        result["got"] = got
        if got:
            _client._zotero_api_lock.release()

    t = threading.Thread(target=probe)
    t.start()
    t.join(timeout=5)
    return result.get("got", False)


class _Zot(FakeZotero):
    def __init__(self, items=None):
        super().__init__()
        self._items = items or []
        self._params = {}

    def add_parameters(self, **kw):
        self._params = kw

    def items(self, **kw):
        return self._items

    def collection_items(self, *a, **kw):
        return self._items


def _item(doi):
    return {"key": "ITEM0001", "data": {"itemType": "journalArticle", "title": "T", "DOI": doi}}


def test_find_related_papers_releases_the_lock_during_openalex_calls(monkeypatch):
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: _Zot())
    seen = []

    def fake_get(url, params=None, timeout=None, **kw):
        seen.append(_lock_is_free())
        if url.endswith("/works/https://doi.org/10.1234/x"):
            return FakeResponse(200, {
                "id": "https://openalex.org/W1", "title": "Src",
                "referenced_works": ["https://openalex.org/W10"],
            })
        return FakeResponse(200, {"results": [{
            "id": "https://openalex.org/W10", "title": "Ref", "doi": "https://doi.org/10.1234/ref",
            "publication_year": 2010, "cited_by_count": 1, "authorships": [],
        }]})

    monkeypatch.setattr(discovery.requests, "get", fake_get)
    out = discovery.find_related_papers("10.1234/x", direction="references", ctx=DummyContext())

    assert "Ref" in out
    assert seen and all(seen), "the Zotero API lock was held during an OpenAlex request"


def test_find_related_papers_still_takes_the_lock_for_zotero_reads(monkeypatch):
    held = []

    class Zot(_Zot):
        def items(self, **kw):
            held.append(not _lock_is_free())
            return []

    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: Zot())

    def fake_get(url, params=None, timeout=None, **kw):
        if url.endswith("/works/https://doi.org/10.1234/x"):
            return FakeResponse(200, {"id": "https://openalex.org/W1", "title": "Src",
                                      "referenced_works": ["https://openalex.org/W10"]})
        return FakeResponse(200, {"results": [{
            "id": "https://openalex.org/W10", "title": "Ref", "doi": "https://doi.org/10.1234/ref",
            "publication_year": 2010, "cited_by_count": 1, "authorships": [],
        }]})

    monkeypatch.setattr(discovery.requests, "get", fake_get)
    discovery.find_related_papers("10.1234/x", direction="references", ctx=DummyContext())

    assert held and all(held), "library membership was read without the Zotero API lock"


def _spy_scite(monkeypatch, seen):
    def free(name, ret):
        def f(*a, **kw):
            seen.append((name, _lock_is_free()))
            return ret
        return f

    monkeypatch.setattr(scite_tools._scite, "get_tally", free("get_tally", {"supporting": 1}))
    monkeypatch.setattr(scite_tools._scite, "get_paper", free("get_paper", {"title": "P"}))
    monkeypatch.setattr(scite_tools._scite, "get_tallies_batch", free("get_tallies_batch", {}))
    monkeypatch.setattr(scite_tools._scite, "get_papers_batch", free("get_papers_batch", {"10.1234/a": {}}))


def test_scite_enrich_item_releases_the_lock_for_scite_calls(monkeypatch):
    seen = []
    _spy_scite(monkeypatch, seen)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: _Zot([_item("10.1234/a")]))

    scite_tools.enrich_item(item_key="ITEM0001", ctx=DummyContext())

    assert {n for n, _ in seen} == {"get_tally", "get_paper"}
    assert all(free for _, free in seen)


def test_scite_enrich_search_releases_the_lock_for_scite_calls(monkeypatch):
    seen = []
    _spy_scite(monkeypatch, seen)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: _Zot([_item("10.1234/a")]))

    scite_tools.enrich_search("anything", ctx=DummyContext())

    assert seen and all(free for _, free in seen)


def test_scite_check_retractions_releases_the_lock_for_scite_calls(monkeypatch):
    seen = []
    _spy_scite(monkeypatch, seen)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: _Zot([_item("10.1234/a")]))

    scite_tools.check_retractions(ctx=DummyContext())

    assert seen and all(free for _, free in seen)


def test_scite_zotero_reads_still_hold_the_lock(monkeypatch):
    held = []

    class Zot(_Zot):
        def items(self, **kw):
            held.append(not _lock_is_free())
            return [_item("10.1234/a")]

        def item(self, key):
            held.append(not _lock_is_free())
            return _item("10.1234/a")

    seen = []
    _spy_scite(monkeypatch, seen)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: Zot())

    scite_tools.enrich_search("q", ctx=DummyContext())
    scite_tools.check_retractions(ctx=DummyContext())
    scite_tools.enrich_item(item_key="ITEM0001", ctx=DummyContext())

    assert len(held) == 3 and all(held)
