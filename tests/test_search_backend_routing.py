"""Tests for the #167 SQLite/pyzotero backend routing in tools/search.py.

Verifies that zotero_search_items / zotero_advanced_search (a) use the
SQLite backend when ZOTERO_SEARCH_BACKEND=sqlite and a local DB is
available, never touching the pyzotero client; (b) fall back to the
existing pyzotero-based path on any condition the SQL backend doesn't
support; and (c) stay inert (ignore ZOTERO_SEARCH_BACKEND) outside local
mode, since the sqlite backend requires a readable zotero.sqlite.
"""

from conftest import DummyContext
from test_sql_search_backend import _build_db

from zotero_mcp import client as _client
from zotero_mcp import server
from zotero_mcp import utils as _utils
from zotero_mcp.local_db import LocalZoteroReader
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
