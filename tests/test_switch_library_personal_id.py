"""Regression tests for #603: list_libraries -> switch_library round trip.

In local mode `zotero_list_libraries` shows the personal library by its
SQLite libraryID (normally 1), but `zotero_switch_library` only accepted
"0" for library_type="user", so feeding the listed id back failed.
"""

import types

import pytest
from conftest import DummyContext

from zotero_mcp import client as _client
from zotero_mcp.tools import retrieval

_LIBRARIES = [
    {"libraryID": 1, "type": "user", "itemCount": 10},
    {"libraryID": 2, "type": "group", "groupID": 5294983, "groupName": "G",
     "groupDescription": "", "itemCount": 3},
    {"libraryID": 3, "type": "feed", "feedName": "F", "itemCount": 4},
]


class _FakeReader:
    def __init__(self, *args, **kwargs):
        pass

    def get_libraries(self):
        return [dict(lib) for lib in _LIBRARIES]

    def close(self):
        pass


@pytest.fixture
def local_mode(monkeypatch):
    import zotero_mcp.local_db as local_db

    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setattr(local_db, "LocalZoteroReader", _FakeReader)
    monkeypatch.setattr(
        retrieval, "load_config",
        lambda: types.SimpleNamespace(resolve_zotero_db_path=lambda: "/nonexistent.sqlite"),
    )
    monkeypatch.setattr(
        retrieval._library, "get_library_backend",
        lambda: types.SimpleNamespace(name="sqlite"),
    )
    _client.clear_active_library()
    yield
    _client.clear_active_library()


@pytest.mark.parametrize("library_id", ["0", "user", "1"])
def test_local_user_library_ids_accepted(local_mode, library_id):
    assert retrieval.validate_library_switch(library_id, "user") is None


@pytest.mark.parametrize("library_id", ["2", "3", "99"])
def test_local_non_user_library_ids_rejected(local_mode, library_id):
    assert retrieval.validate_library_switch(library_id, "user") is not None


def test_listed_id_round_trips_through_switch(local_mode):
    listing = retrieval.list_libraries(ctx=DummyContext())
    assert "libraryID=1" in listing
    result = retrieval.switch_library("1", "user", ctx=DummyContext())
    assert result.startswith("Successfully switched")
    # Stored in the "0" form the rest of local mode expects.
    assert _client.get_active_library() == {"library_id": "0", "library_type": "user"}


def test_switch_user_keyword_normalizes(local_mode):
    result = retrieval.switch_library("user", "user", ctx=DummyContext())
    assert result.startswith("Successfully switched")
    assert _client.get_active_library()["library_id"] == "0"


class _ProbeClient:
    """Web-mode client stub that records which library the probe hits."""

    def __init__(self, probed):
        self._probed = probed

    def add_parameters(self, **_kwargs):
        pass

    def items(self):
        self._probed.append(_client.get_active_library())
        return []


@pytest.fixture
def web_mode(monkeypatch):
    monkeypatch.delenv("ZOTERO_LOCAL", raising=False)
    monkeypatch.setattr(
        retrieval._library, "get_library_backend",
        lambda: types.SimpleNamespace(name="web"),
    )
    probed = []
    monkeypatch.setattr(retrieval._client, "get_zotero_client", lambda: _ProbeClient(probed))
    _client.clear_active_library()
    yield probed
    _client.clear_active_library()


@pytest.mark.parametrize("library_id", ["user", "0"])
def test_web_user_keyword_maps_to_configured_user_id(web_mode, monkeypatch, library_id):
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "12345")
    monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "user")
    result = retrieval.switch_library(library_id, "user", ctx=DummyContext())
    assert result.startswith("Successfully switched")
    assert web_mode == [{"library_id": "12345", "library_type": "user"}]


@pytest.mark.parametrize("env_type", ["group", "Group", " GROUP "])
@pytest.mark.parametrize("library_id", ["user", "0"])
def test_web_user_keyword_not_mapped_to_group_id(web_mode, monkeypatch, library_id, env_type):
    """With a group configured, ZOTERO_LIBRARY_ID is a groupID: never probe /users/<groupID>."""
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "5294983")
    monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", env_type)
    retrieval.switch_library(library_id, "user", ctx=DummyContext())
    assert web_mode, "probe should have run"
    assert all(p["library_id"] != "5294983" for p in web_mode)
    assert web_mode == [{"library_id": library_id, "library_type": "user"}]
