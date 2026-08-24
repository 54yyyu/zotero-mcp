"""Tests for the #167 SQLite/pyzotero backend routing in tools/search.py.

Verifies that zotero_search_items / zotero_advanced_search (a) use the
SQLite backend when ZOTERO_SEARCH_BACKEND=sqlite and a local DB is
available, never touching the pyzotero client; (b) fall back to the
existing pyzotero-based path on any condition the SQL backend doesn't
support; and (c) stay inert (ignore ZOTERO_SEARCH_BACKEND) outside local
mode, since the sqlite backend requires a readable zotero.sqlite.
"""

from conftest import DummyContext
from test_sql_search_backend import GROUP_ID, _build_db

from zotero_mcp import client as _client
from zotero_mcp import server
from zotero_mcp import utils as _utils
from zotero_mcp.local_db import LocalZoteroReader
from zotero_mcp.tools import _helpers
from zotero_mcp.tools import search as search_module


class _RefusingZotero:
    """A pyzotero stand-in that fails the test if the API path is used."""

    def add_parameters(self, **kwargs):
        raise AssertionError("pyzotero path should not have been used")

    def items(self, *args, **kwargs):
        raise AssertionError("pyzotero path should not have been used")

    def collection(self, key):
        raise AssertionError("pyzotero path should not have been used")


class _FallbackZotero:
    """A pyzotero stand-in returning a single fixed item, for fallback checks."""

    def __init__(self, items):
        self._items = items

    def add_parameters(self, **kwargs):
        pass

    def items(self, *args, **kwargs):
        return self._items


def _sqlite_reader(tmp_path):
    db_path = tmp_path / "zotero.sqlite"
    _build_db(db_path)
    return LocalZoteroReader(db_path=str(db_path))


def test_search_items_uses_sqlite_backend_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(_utils, "get_search_backend", lambda: "sqlite")
    monkeypatch.setattr(search_module, "get_local_zotero_reader", lambda: _sqlite_reader(tmp_path))
    monkeypatch.setattr(_client, "get_zotero_client", lambda: _RefusingZotero())
    monkeypatch.setattr(_client, "get_active_group_id", lambda: 0)

    result = server.search_items(query="Quantum", ctx=DummyContext())

    assert "Quantum Networks and Learning" in result


def test_advanced_search_uses_sqlite_backend_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(_utils, "get_search_backend", lambda: "sqlite")
    monkeypatch.setattr(search_module, "get_local_zotero_reader", lambda: _sqlite_reader(tmp_path))
    monkeypatch.setattr(_client, "get_zotero_client", lambda: _RefusingZotero())
    monkeypatch.setattr(_client, "get_active_group_id", lambda: 0)

    result = server.advanced_search(
        conditions=[{"field": "title", "operation": "contains", "value": "Quantum"}],
        ctx=DummyContext(),
    )

    assert "Quantum Networks and Learning" in result


def test_search_items_falls_back_when_tag_filter_unsupported(monkeypatch, tmp_path):
    fake_item = {
        "key": "FALLBACK1",
        "data": {"itemType": "journalArticle", "title": "Fallback Item",
                  "date": "2024", "creators": [], "tags": [{"tag": "physics"}]},
    }
    monkeypatch.setattr(_utils, "get_search_backend", lambda: "sqlite")
    monkeypatch.setattr(search_module, "get_local_zotero_reader", lambda: _sqlite_reader(tmp_path))
    monkeypatch.setattr(_client, "get_zotero_client", lambda: _FallbackZotero([fake_item]))
    monkeypatch.setattr(_client, "get_active_group_id", lambda: 0)
    monkeypatch.setattr(_utils, "_generate_search_variants", lambda q: [q])

    result = server.search_items(query="Fallback", tag=["physics"], ctx=DummyContext())

    assert "Fallback Item" in result


def test_advanced_search_falls_back_on_unsupported_field(monkeypatch, tmp_path):
    fake_item = {
        "key": "FALLBACK2",
        "data": {"itemType": "journalArticle", "title": "Fallback Advanced",
                  "date": "2024", "creators": [], "tags": []},
    }
    monkeypatch.setattr(_utils, "get_search_backend", lambda: "sqlite")
    monkeypatch.setattr(search_module, "get_local_zotero_reader", lambda: _sqlite_reader(tmp_path))
    monkeypatch.setattr(_client, "get_zotero_client", lambda: _FallbackZotero([fake_item]))
    monkeypatch.setattr(_client, "get_active_group_id", lambda: 0)

    result = server.advanced_search(
        conditions=[{"field": "volume", "operation": "is", "value": "3"}],
        ctx=DummyContext(),
    )

    assert "No items found matching the search criteria." in result


def test_search_items_inert_when_not_local_mode(monkeypatch):
    """Even with ZOTERO_SEARCH_BACKEND=sqlite, the real get_local_zotero_reader()
    returns None outside local mode — the tool must fall back automatically,
    without needing its own local-mode check."""
    fake_item = {
        "key": "WEBMODE1",
        "data": {"itemType": "journalArticle", "title": "Web Mode Item",
                  "date": "2024", "creators": [], "tags": []},
    }
    monkeypatch.delenv("ZOTERO_LOCAL", raising=False)
    monkeypatch.setattr(_utils, "get_search_backend", lambda: "sqlite")
    monkeypatch.setattr(_client, "get_zotero_client", lambda: _FallbackZotero([fake_item]))
    monkeypatch.setattr(_client, "get_active_group_id", lambda: 0)
    monkeypatch.setattr(_utils, "_generate_search_variants", lambda q: [q])

    result = server.search_items(query="Web Mode", ctx=DummyContext())

    assert "Web Mode Item" in result


# ---------------------------------------------------------------------------
# Global search across all libraries (#163)
# ---------------------------------------------------------------------------

def _sqlite_mode(monkeypatch, tmp_path):
    """Wire the tools to the SQLite backend over the shared fixture DB, with a
    pyzotero client that fails the test if the API path is ever taken.

    The fixture is built once and each call hands back a fresh reader over
    that same file — a global search opens one for the gate check and one per
    cascade attempt, so rebuilding per call would collide on the schema.
    """
    db_path = tmp_path / "zotero.sqlite"
    _build_db(db_path)
    open_reader = lambda: LocalZoteroReader(db_path=str(db_path))  # noqa: E731

    monkeypatch.setattr(_utils, "get_search_backend", lambda: "sqlite")
    monkeypatch.setattr(search_module, "get_local_zotero_reader", open_reader)
    monkeypatch.setattr(_helpers, "get_local_zotero_reader", open_reader)
    monkeypatch.setattr(_client, "get_zotero_client", lambda: _RefusingZotero())
    monkeypatch.setattr(_client, "get_active_group_id", lambda: 0)


def test_global_search_is_refused_without_the_sqlite_backend(monkeypatch):
    monkeypatch.setattr(_utils, "get_search_backend", lambda: "api")
    monkeypatch.setattr(_client, "get_zotero_client", lambda: _RefusingZotero())

    result = server.search_items(query="Quantum", search_all_libraries=True, ctx=DummyContext())

    assert "ZOTERO_SEARCH_BACKEND=sqlite" in result
    assert "Quantum Networks" not in result


def test_global_search_spans_libraries_and_labels_each_result(monkeypatch, tmp_path):
    _sqlite_mode(monkeypatch, tmp_path)

    result = server.search_items(query="quantum", search_all_libraries=True, ctx=DummyContext())

    assert "Quantum Networks and Learning" in result
    assert "Group Library Paper about quantum" in result
    assert "**Library:** My Library (personal)" in result
    assert f"**Library:** Test Group (groupID={GROUP_ID})" in result


def test_single_library_search_does_not_label_libraries(monkeypatch, tmp_path):
    _sqlite_mode(monkeypatch, tmp_path)

    result = server.search_items(query="quantum", ctx=DummyContext())

    assert "Quantum Networks and Learning" in result
    assert "Group Library Paper" not in result
    assert "**Library:**" not in result


def test_global_search_rejects_a_collection_scope(monkeypatch, tmp_path):
    _sqlite_mode(monkeypatch, tmp_path)

    result = server.search_items(
        query="quantum", collection_key="COLLB001", search_all_libraries=True, ctx=DummyContext()
    )

    assert "collection" in result.lower()
    assert "Error" in result


def test_global_search_serves_a_tag_filter_in_sql(monkeypatch, tmp_path):
    _sqlite_mode(monkeypatch, tmp_path)

    result = server.search_items(
        query="quantum", tag=["physics"], search_all_libraries=True, ctx=DummyContext()
    )

    assert "Quantum Networks and Learning" in result
    assert "Group Library Paper about quantum" in result


def test_global_scope_survives_the_fallback_cascade(monkeypatch, tmp_path):
    """A query that misses first and only hits on a simplification must still
    be global — a retry that reverted to the active library would silently
    answer a narrower question than the caller asked."""
    _sqlite_mode(monkeypatch, tmp_path)

    # "Group 9999" finds nothing; strategy 1 simplifies it to the author-ish
    # first token "Group", which matches the group library's item.
    result = server.search_items(
        query="Group 9999", search_all_libraries=True, ctx=DummyContext()
    )

    assert "Group Library Paper about quantum" in result
    assert f"**Library:** Test Group (groupID={GROUP_ID})" in result


def test_global_advanced_search_spans_libraries(monkeypatch, tmp_path):
    _sqlite_mode(monkeypatch, tmp_path)

    result = server.advanced_search(
        conditions=[{"field": "title", "operation": "contains", "value": "quantum"}],
        search_all_libraries=True, ctx=DummyContext(),
    )

    assert "Quantum Networks and Learning" in result
    assert "Group Library Paper about quantum" in result
    assert f"**Library:** Test Group (groupID={GROUP_ID})" in result


def test_global_advanced_search_rejects_a_collection_condition(monkeypatch, tmp_path):
    _sqlite_mode(monkeypatch, tmp_path)

    result = server.advanced_search(
        conditions=[{"field": "collection", "operation": "is", "value": "COLLB001"}],
        search_all_libraries=True, ctx=DummyContext(),
    )

    assert "Error" in result
    assert "collection" in result.lower()
