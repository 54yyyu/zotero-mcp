"""Tests for the single write-client resolution path and its error message."""

import time

import pytest
from conftest import DummyContext

from zotero_mcp import client as _client
from zotero_mcp.tools import _helpers


@pytest.fixture
def local_mode(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)


@pytest.fixture
def web_mode(monkeypatch):
    monkeypatch.delenv("ZOTERO_LOCAL", raising=False)
    monkeypatch.setenv("ZOTERO_API_KEY", "webkey")
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "123")


def _server_supports_local_writes():
    _client._local_probe_cache.update(
        {"server_id": "srv-1", "checked_at": time.monotonic()}
    )


class TestResolutionOrder:
    def test_web_mode_uses_one_client_for_both(self, web_mode, monkeypatch, fake_zot):
        monkeypatch.setattr(_client, "get_zotero_client", lambda: fake_zot)
        read, write, mode = _helpers.resolve_write_client()
        assert mode == "web"
        assert read is write is fake_zot

    def test_local_key_wins_over_web_credentials(
        self, local_mode, monkeypatch, fake_zot, fake_local_zot
    ):
        monkeypatch.setenv("ZOTERO_API_KEY", "webkey")
        monkeypatch.setenv("ZOTERO_LIBRARY_ID", "123")
        monkeypatch.setattr(_client, "get_local_write_client", lambda: fake_local_zot)
        monkeypatch.setattr(_client, "get_web_zotero_client", lambda: fake_zot)
        _read, write, mode = _helpers.resolve_write_client()
        assert mode == "local"
        assert write is fake_local_zot

    def test_local_mode_reads_and_writes_through_the_same_client(
        self, local_mode, monkeypatch, fake_local_zot
    ):
        """Local object versions are scoped to the Zotero database that issued
        them, so a read from one backend can't validate a write to another."""
        monkeypatch.setattr(_client, "get_local_write_client", lambda: fake_local_zot)
        read, write, _mode = _helpers.resolve_write_client()
        assert read is write is fake_local_zot

    def test_falls_back_to_hybrid_without_a_local_key(
        self, local_mode, monkeypatch, fake_zot
    ):
        web = object()
        monkeypatch.setattr(_client, "get_local_write_client", lambda: None)
        monkeypatch.setattr(_client, "get_web_zotero_client", lambda: web)
        monkeypatch.setattr(_client, "get_zotero_client", lambda: fake_zot)
        read, write, mode = _helpers.resolve_write_client()
        assert mode == "hybrid"
        assert read is fake_zot
        assert write is web

    def test_raises_when_nothing_can_write(self, local_mode, monkeypatch):
        monkeypatch.setattr(_client, "get_local_write_client", lambda: None)
        monkeypatch.setattr(_client, "get_web_zotero_client", lambda: None)
        with pytest.raises(ValueError, match="Cannot perform write"):
            _helpers.resolve_write_client()

    def test_group_override_reaches_the_local_write_client(
        self, local_mode, monkeypatch
    ):
        """Uses the real factory: it is the thing that reads the override."""
        monkeypatch.setenv("ZOTERO_LOCAL_API_KEY", "k")
        _client.set_active_library("99", "group")
        try:
            _read, write, _mode = _helpers.resolve_write_client()
            assert write.library_id == "99"
            # Singular in the override, plural in the URL path pyzotero builds.
            assert write.library_type == "groups"
        finally:
            _client.clear_active_library()

    def test_a_user_library_override_cannot_break_the_users_zero_mapping(
        self, local_mode, monkeypatch
    ):
        """zotero_switch_library reports the SQLite libraryID (typically 1) for
        the local user library, but the local API only serves users/0. Applying
        the override on top of the factory's mapping sent every write to a
        library that does not exist."""
        monkeypatch.setenv("ZOTERO_LOCAL_API_KEY", "k")
        _client.set_active_library("1", "user")
        try:
            _read, write, mode = _helpers.resolve_write_client()
            assert mode == "local"
            assert (write.library_type, write.library_id) == ("users", "0")
        finally:
            _client.clear_active_library()


class TestBackwardCompatibleWrapper:
    def test_still_returns_a_pair(self, web_mode, monkeypatch, fake_zot):
        monkeypatch.setattr(_client, "get_zotero_client", lambda: fake_zot)
        assert _helpers._get_write_client(None) == (fake_zot, fake_zot)

    def test_raises_valueerror_for_the_tools_to_catch(self, local_mode, monkeypatch):
        monkeypatch.setattr(_client, "get_local_write_client", lambda: None)
        monkeypatch.setattr(_client, "get_web_zotero_client", lambda: None)
        with pytest.raises(ValueError):
            _helpers._get_write_client(None)


class TestRecoveryAfterARejectedKey:
    def test_next_write_falls_back_to_the_web_api(self, local_mode, monkeypatch, fake_zot):
        """A consumed single-use key must not permanently shadow working web
        credentials — that would be a regression for existing hybrid users."""
        from conftest import FakeLocalZotero, _FakeResponse

        monkeypatch.setenv("ZOTERO_API_KEY", "webkey")
        monkeypatch.setenv("ZOTERO_LIBRARY_ID", "123")
        monkeypatch.setenv("ZOTERO_LOCAL_API_KEY", "consumed")
        monkeypatch.setattr(_client, "get_web_zotero_client", lambda: fake_zot)
        monkeypatch.setattr(_client, "get_zotero_client", lambda: fake_zot)

        assert _helpers.resolve_write_client()[2] == "local"

        # The write comes back 401: the key is dead and gets dropped.
        _helpers.describe_write_failure(_FakeResponse(401), FakeLocalZotero())
        monkeypatch.delenv("ZOTERO_LOCAL_API_KEY")

        assert _helpers.resolve_write_client()[2] == "hybrid"


class TestCallersUseTheResolver:
    def test_batch_update_tags_does_not_build_its_own_read_client(
        self, monkeypatch, fake_zot
    ):
        """A separately-built read client makes `write_zot is not zot` always
        true, so every item pays for a re-fetch it doesn't need."""
        from zotero_mcp import server

        def _forbidden():
            raise AssertionError("should take the read client from the resolver")

        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr(_client, "get_zotero_client", _forbidden)

        fake_zot.add_parameters = lambda **kwargs: None
        result = server.batch_update_tags(
            query="anything", add_tags=["x"], ctx=DummyContext()
        )
        assert "Error" not in result


class TestUnavailableMessage:
    def test_leads_with_authorizing_when_the_server_supports_it(self):
        _server_supports_local_writes()
        msg = _helpers.write_unavailable_message()
        assert "Cannot perform write operations" in msg
        assert msg.index("authorize-local") < msg.index("ZOTERO_API_KEY")

    def test_leads_with_web_credentials_on_an_older_zotero(self):
        msg = _helpers.write_unavailable_message()
        assert "Zotero 10 or newer" in msg
        assert msg.index("ZOTERO_API_KEY") < msg.index("authorize-local")

    def test_names_the_operation(self):
        assert "creating notes" in _helpers.write_unavailable_message("creating notes")

    def test_keeps_the_phrase_the_write_tools_are_tested_against(self):
        assert "local-only mode" in _helpers.write_unavailable_message()
