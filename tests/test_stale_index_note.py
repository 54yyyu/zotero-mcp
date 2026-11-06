"""zotero_semantic_search says, in one line, when the index is behind.

The check compares the library version recorded at the last update with the
library's version now, read with one request while the query is embedded and
reused for a minute. No status call is needed first (#783).
"""

import pytest
from conftest import DummyContext

from zotero_mcp import client as _client
from zotero_mcp.tools import search as search_module


class FakeZotero:
    def __init__(self, version):
        self.version = version
        self.calls = 0

    def last_modified_version(self):
        self.calls += 1
        return self.version


class FakeSearch:
    def __init__(self, zotero, indexed, results=True):
        self.zotero_client = zotero
        self.indexed = indexed
        self.results = results

    def _load_last_sync_version(self):
        return self.indexed

    def search(self, query, limit=10, filters=None, group_id=None):
        hits = [{
            "item_key": "ITEM0001",
            "similarity_score": 0.5,
            "matched_passage": "a passage",
            "zotero_item": {"key": "ITEM0001", "data": {"title": "A Paper", "itemType": "journalArticle"}},
        }] if self.results else []
        return {"results": hits, "total_found": len(hits)}


@pytest.fixture
def run(monkeypatch, tmp_path):
    from zotero_mcp import semantic_search

    config_dir = tmp_path / ".config" / "zotero-mcp"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text("{}")
    monkeypatch.setattr(search_module.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(search_module, "_maybe_fire_presearch_sync", lambda _s: False)
    monkeypatch.setattr(search_module, "_library_versions", {}, raising=False)
    monkeypatch.setattr(_client, "get_active_group_id", lambda: 0)

    def run(search, **kwargs):
        monkeypatch.setattr(semantic_search, "create_semantic_search", lambda *a, **kw: search)
        return search_module.semantic_search(query="q", ctx=DummyContext(), **kwargs)

    return run


def test_a_stale_index_adds_one_line(run):
    out = run(FakeSearch(FakeZotero(120), indexed=100))

    assert out.rstrip().endswith(
        "Note: the library has changed since the index was last updated "
        "(version 120, index at 100); zotero_update_search_database adds the changes."
    )


def test_an_up_to_date_index_adds_nothing(run):
    out = run(FakeSearch(FakeZotero(100), indexed=100))

    assert "Note:" not in out


def test_no_results_still_says_the_index_is_behind(run):
    out = run(FakeSearch(FakeZotero(120), indexed=100, results=False))

    assert out.startswith("No semantically similar items")
    assert "zotero_update_search_database" in out


def test_the_version_is_read_once_a_minute(run):
    zotero = FakeZotero(120)
    run(FakeSearch(zotero, indexed=100))
    run(FakeSearch(zotero, indexed=100))

    assert zotero.calls == 1


@pytest.mark.parametrize("kwargs", [{"search_all_libraries": True}, {"library_id": 4242}])
def test_only_the_active_library_is_checked(run, monkeypatch, kwargs):
    monkeypatch.setattr(search_module._helpers, "global_search_error", lambda: None)
    zotero = FakeZotero(120)

    out = run(FakeSearch(zotero, indexed=100), **kwargs)

    assert zotero.calls == 0 and "Note:" not in out


def test_no_note_while_the_presearch_sync_catches_up(run, monkeypatch):
    monkeypatch.setattr(search_module, "_maybe_fire_presearch_sync", lambda _s: True)
    zotero = FakeZotero(120)

    out = run(FakeSearch(zotero, indexed=100))

    assert zotero.calls == 0 and "Note:" not in out


def test_a_failed_version_read_adds_nothing(run):
    class Down(FakeZotero):
        def last_modified_version(self):
            raise ConnectionError("Zotero is closed")

    out = run(FakeSearch(Down(0), indexed=100))

    assert "Note:" not in out and "A Paper" in out


def test_a_failed_read_is_not_retried_on_every_search(run):
    class Down(FakeZotero):
        def last_modified_version(self):
            self.calls += 1
            raise ConnectionError("Zotero is closed")

    zotero = Down(0)
    run(FakeSearch(zotero, indexed=100))
    run(FakeSearch(zotero, indexed=100))

    assert zotero.calls == 1
