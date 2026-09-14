"""SQLite is the default read backend in local mode, with the API as fallback.

Measured on a 44k-item library, SQLite answers the read tools' calls one to
three orders of magnitude faster than the API (list_tags 110 ms vs 104 s,
children of 25 items 1.8 ms vs 2.5 s), so it is used whenever the database is
on this machine. The API stays available two ways: ZOTERO_BACKEND=api, and a
per-call fallback for any query SQLite cannot express.
"""

import pytest

from zotero_mcp import library
from zotero_mcp import utils as _utils
from zotero_mcp.library import FallbackBackend, UnsupportedByBackend


@pytest.fixture
def unset(monkeypatch):
    monkeypatch.delenv("ZOTERO_BACKEND", raising=False)
    monkeypatch.delenv("ZOTERO_SEARCH_BACKEND", raising=False)


@pytest.mark.parametrize("local, expected", [(True, "sqlite"), (False, "api")])
def test_unset_follows_local_mode(unset, monkeypatch, local, expected):
    monkeypatch.setattr(_utils, "is_local_mode", lambda: local)
    assert _utils.get_search_backend() == expected


def test_explicit_api_opts_out_in_local_mode(unset, monkeypatch):
    monkeypatch.setattr(_utils, "is_local_mode", lambda: True)
    monkeypatch.setenv("ZOTERO_BACKEND", "api")
    assert _utils.get_search_backend() == "api"


@pytest.mark.parametrize("var", ["ZOTERO_BACKEND", "ZOTERO_SEARCH_BACKEND"])
def test_explicit_sqlite_wins(unset, monkeypatch, var):
    monkeypatch.setattr(_utils, "is_local_mode", lambda: False)
    monkeypatch.setenv(var, "sqlite")
    monkeypatch.setenv("ZOTERO_BACKEND" if var != "ZOTERO_BACKEND" else "ZOTERO_SEARCH_BACKEND", "api")
    assert _utils.get_search_backend() == "sqlite"


class _Primary:
    name = "sqlite"

    def search_items(self, query, **kw):
        raise UnsupportedByBackend("wildcard tag")

    def get_item(self, key):
        return {"key": key, "source": "sqlite"}

    def list_tags(self, **kw):
        raise ValueError("disk error")


class _Api:
    name = "api"

    def __init__(self):
        self.calls = []

    def search_items(self, query, **kw):
        self.calls.append(("search_items", query))
        return [{"key": "APIHIT01"}]


def test_unsupported_query_is_answered_by_the_api():
    api = _Api()
    backend = FallbackBackend(_Primary(), lambda: api)
    assert backend.search_items("x", tag=["foo*"]) == [{"key": "APIHIT01"}]
    assert api.calls == [("search_items", "x")]


def test_supported_query_never_builds_the_api_client():
    built = []
    backend = FallbackBackend(_Primary(), lambda: built.append(1) or _Api())
    assert backend.get_item("K1")["source"] == "sqlite"
    assert built == []
    assert backend.name == "sqlite"


def test_a_real_failure_is_not_hidden_by_the_fallback():
    backend = FallbackBackend(_Primary(), lambda: pytest.fail("must not fall back"))
    with pytest.raises(ValueError):
        backend.list_tags()


def test_get_library_backend_wraps_sqlite_with_the_fallback(monkeypatch):
    monkeypatch.setattr(library, "configured_backend", lambda: "sqlite")
    monkeypatch.setattr(library, "_sqlite_reader", lambda: object())
    backend = library.get_library_backend(zot=object())
    assert isinstance(backend, FallbackBackend)
    assert backend.name == "sqlite"
