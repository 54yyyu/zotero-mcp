"""In local mode a DOI dedup check reads the DOI field from zotero.sqlite.

The API path has to run qmode='everything' because the DOI field is not
searchable over HTTP — and against the local API that is a full-text scan of
the whole library, which pinned Zotero at 100% CPU for minutes on a batch of
11 DOIs. The DOI is a column in zotero.sqlite, so in local mode it is looked
up there and the expensive search never runs.
"""

import pytest
from conftest import FakeZotero

from zotero_mcp.tools import _helpers

DOI = "10.1234/test.2024.001"


def _item(key, doi):
    return {
        "key": key,
        "version": 7,
        "data": {"key": key, "itemType": "journalArticle", "title": key, "DOI": doi},
    }


class _Reader:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def advanced_search_sql(self, conditions, join_mode="all", group_id=0):
        self.calls.append((conditions, group_id))
        return self.hits


class _NoSearchZotero(FakeZotero):
    """Records quick searches. find_existing_items swallows exceptions from
    the search, so raising here would read as "no match", not as a failure."""

    def __init__(self):
        super().__init__()
        self.searches = []

    def items(self, **kwargs):
        self.searches.append(kwargs)
        return []


@pytest.fixture
def local_sqlite(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_BACKEND", "sqlite")

    def install(reader):
        monkeypatch.setattr("zotero_mcp.library._sqlite_reader", lambda: reader)
        return reader

    return install


def test_match_comes_from_sqlite_and_is_fetched_through_the_client(local_sqlite):
    zot = _NoSearchZotero()
    zot._items = [_item("EXIST001", DOI.upper())]
    reader = local_sqlite(_Reader([_item("EXIST001", DOI.upper())]))

    out = _helpers.find_existing_items(zot, doi=DOI, title="Existing")

    assert [i["key"] for i in out] == ["EXIST001"]
    # The returned dict is the client's, so its version is the one writes use.
    assert out[0] is zot._items[0]
    assert reader.calls == [([{"field": "DOI", "operation": "contains", "value": DOI}], 0)]
    assert zot.searches == []


def test_sqlite_miss_is_final(local_sqlite):
    zot = _NoSearchZotero()
    local_sqlite(_Reader([]))
    assert _helpers.find_existing_items(zot, doi=DOI, title="Anything") == []
    assert zot.searches == []


def test_substring_hit_with_a_different_doi_is_rejected(local_sqlite):
    zot = _NoSearchZotero()
    zot._items = [_item("LONGER01", DOI + "5")]
    local_sqlite(_Reader([_item("LONGER01", DOI + "5")]))
    assert _helpers.find_existing_items(zot, doi=DOI) == []
    assert zot.searches == []


def test_a_candidate_the_client_cannot_fetch_is_skipped(local_sqlite):
    """A key the DB snapshot has but the API lost costs that candidate only."""
    class _Lost(_NoSearchZotero):
        def item(self, item_key):
            if item_key == "GONE0001":
                raise RuntimeError("404 Not Found")
            return super().item(item_key)

    zot = _Lost()
    zot._items = [_item("EXIST001", DOI)]
    local_sqlite(_Reader([_item("GONE0001", DOI), _item("EXIST001", DOI)]))
    out = _helpers.find_existing_items(zot, doi=DOI)
    assert [i["key"] for i in out] == ["EXIST001"]


def test_falls_back_to_quick_search_when_sqlite_cannot_answer(local_sqlite):
    zot = FakeZotero()
    zot._items = [_item("EXIST001", DOI)]
    local_sqlite(_Reader(None))
    assert [i["key"] for i in _helpers.find_existing_items(zot, doi=DOI)] == ["EXIST001"]


def test_falls_back_when_there_is_no_database(local_sqlite):
    zot = FakeZotero()
    zot._items = [_item("EXIST001", DOI)]
    local_sqlite(None)
    assert [i["key"] for i in _helpers.find_existing_items(zot, doi=DOI)] == ["EXIST001"]


def test_web_mode_never_touches_sqlite(monkeypatch):
    monkeypatch.delenv("ZOTERO_LOCAL", raising=False)

    def boom():
        raise AssertionError("SQLite must not be read in web mode")

    monkeypatch.setattr("zotero_mcp.library._sqlite_reader", boom)
    zot = FakeZotero()
    zot._items = [_item("EXIST001", DOI)]
    assert [i["key"] for i in _helpers.find_existing_items(zot, doi=DOI)] == ["EXIST001"]
