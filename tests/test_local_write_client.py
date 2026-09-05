"""Tests for local API write credentials, capability probing and the client factory."""

import json

import pytest
from pyzotero.errors import LocalAPIDeniedError, TooManyRequestsError

from zotero_mcp import client as _client


@pytest.fixture
def local_mode(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_TYPE", raising=False)
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)


def _write_config(payload):
    """Write the isolated config file the autouse fixture points us at."""
    _client.ZOTERO_MCP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _client.ZOTERO_MCP_CONFIG_PATH.write_text(json.dumps(payload), encoding="utf-8")


class TestCredentialPrecedence:
    def test_no_credentials_anywhere(self):
        assert _client.get_local_write_credentials() == (None, None, None)

    def test_config_file(self):
        _write_config({"local_api": {"key": "from-config", "server_id": "srv-1"}})
        assert _client.get_local_write_credentials() == ("from-config", "srv-1", "config")

    def test_env_beats_config(self, monkeypatch):
        _write_config({"local_api": {"key": "from-config", "server_id": "srv-1"}})
        monkeypatch.setenv("ZOTERO_LOCAL_API_KEY", "from-env")
        monkeypatch.setenv("ZOTERO_LOCAL_SERVER_ID", "srv-2")
        assert _client.get_local_write_credentials() == ("from-env", "srv-2", "env")

    def test_session_beats_env(self, monkeypatch):
        monkeypatch.setenv("ZOTERO_LOCAL_API_KEY", "from-env")
        _client.store_local_write_credentials("from-session", "srv-3", remember=True)
        key, server_id, source = _client.get_local_write_credentials()
        assert (key, server_id, source) == ("from-session", "srv-3", "session")

    def test_unrelated_config_sections_survive_a_grant(self):
        _write_config({"semantic_search": {"model": "default"}})
        _client.store_local_write_credentials("k", "srv", remember=True)
        stored = json.loads(_client.ZOTERO_MCP_CONFIG_PATH.read_text(encoding="utf-8"))
        assert stored["semantic_search"] == {"model": "default"}
        assert stored["local_api"]["key"] == "k"

    def test_key_is_not_written_to_client_env(self):
        """client_env is rebuilt wholesale by `zotero-mcp setup`, and is the
        section copied out to MCP clients. The key must not live there."""
        _write_config({"client_env": {"ZOTERO_LOCAL": "true"}})
        _client.store_local_write_credentials("secret-key", "srv", remember=True)
        stored = json.loads(_client.ZOTERO_MCP_CONFIG_PATH.read_text(encoding="utf-8"))
        assert "secret-key" not in json.dumps(stored["client_env"])
        assert stored["client_env"] == {"ZOTERO_LOCAL": "true"}

    def test_clear_removes_key_from_disk_and_session(self):
        _client.store_local_write_credentials("k", "srv", remember=True)
        assert _client.clear_local_write_credentials() is True
        assert _client.get_local_write_credentials() == (None, None, None)
        stored = json.loads(_client.ZOTERO_MCP_CONFIG_PATH.read_text(encoding="utf-8"))
        assert "local_api" not in stored

    def test_clear_is_a_no_op_without_stored_credentials(self):
        assert _client.clear_local_write_credentials() is False

    def test_clear_reports_a_session_only_key_honestly(self, monkeypatch):
        """A key granted this run lives in memory, not the config file — saying
        'nothing to remove' would make revoke look broken."""
        monkeypatch.setattr(
            _client, "_local_write_state", {"key": "k", "server_id": "s", "remember": False}
        )
        assert _client.clear_local_write_credentials() is True

    def test_an_unparseable_config_is_never_overwritten(self):
        """It also holds semantic_search and client_env; rewriting a file we
        failed to read would silently discard both."""
        _client.ZOTERO_MCP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        broken = '{"semantic_search": {"model": "x"} THIS IS NOT JSON'
        _client.ZOTERO_MCP_CONFIG_PATH.write_text(broken, encoding="utf-8")

        assert _client.store_local_write_credentials("k", "s", remember=True) is None
        assert _client.ZOTERO_MCP_CONFIG_PATH.read_text(encoding="utf-8") == broken
        # The session copy is still set, so the caller can carry on and warn.
        assert _client.get_local_write_credentials()[0] == "k"

    def test_clearing_leaves_an_unparseable_config_alone_too(self):
        _client.ZOTERO_MCP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        broken = "{not json"
        _client.ZOTERO_MCP_CONFIG_PATH.write_text(broken, encoding="utf-8")
        _client.clear_local_write_credentials()
        assert _client.ZOTERO_MCP_CONFIG_PATH.read_text(encoding="utf-8") == broken

    def test_no_partial_file_is_left_behind_on_write(self):
        _client.store_local_write_credentials("k", "s", remember=True)
        leftovers = list(_client.ZOTERO_MCP_CONFIG_PATH.parent.glob("*.tmp"))
        assert leftovers == []


class TestProbe:
    def _stub_response(self, monkeypatch, headers):
        class _Resp:
            def __init__(self):
                self.headers = headers

        class _Client:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get(self, url):
                self.url = url
                return _Resp()

        monkeypatch.setattr(_client, "_make_local_http_client", lambda *a, **k: _Client())

    def test_reports_the_server_id_header(self, monkeypatch):
        _client._local_probe_cache.clear()
        self._stub_response(monkeypatch, {"zotero-server-id": "abc123"})
        assert _client.probe_local_server_id() == "abc123"

    def test_absent_header_means_no_local_write_support(self, monkeypatch):
        _client._local_probe_cache.clear()
        self._stub_response(monkeypatch, {})
        assert _client.probe_local_server_id() is None

    def test_positive_result_is_cached(self, monkeypatch):
        _client._local_probe_cache.clear()
        calls = []

        class _Client:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get(self, url):
                calls.append(url)
                return type("R", (), {"headers": {"zotero-server-id": "abc"}})()

        monkeypatch.setattr(_client, "_make_local_http_client", lambda *a, **k: _Client())
        assert _client.probe_local_server_id() == "abc"
        assert _client.probe_local_server_id() == "abc"
        assert len(calls) == 1
        # The trailing slash matters: bare /api is a 404.
        assert calls[0].endswith("/api/")


class TestGetLocalWriteClient:
    def test_none_without_a_key(self, local_mode):
        assert _client.get_local_write_client() is None

    def test_none_outside_local_mode(self, monkeypatch):
        monkeypatch.delenv("ZOTERO_LOCAL", raising=False)
        monkeypatch.setenv("ZOTERO_LOCAL_API_KEY", "k")
        assert _client.get_local_write_client() is None

    def test_kill_switch(self, local_mode, monkeypatch):
        monkeypatch.setenv("ZOTERO_LOCAL_API_KEY", "k")
        monkeypatch.setenv("ZOTERO_LOCAL_WRITE", "false")
        assert _client.get_local_write_client() is None

    def test_kill_switch_default_and_auto_are_on(self, local_mode, monkeypatch):
        monkeypatch.setenv("ZOTERO_LOCAL_API_KEY", "k")
        assert _client.get_local_write_client() is not None
        monkeypatch.setenv("ZOTERO_LOCAL_WRITE", "auto")
        assert _client.get_local_write_client() is not None

    def test_carries_the_key_and_server_id(self, local_mode, monkeypatch):
        monkeypatch.setenv("ZOTERO_LOCAL_API_KEY", "k")
        monkeypatch.setenv("ZOTERO_LOCAL_SERVER_ID", "srv-9")
        zot = _client.get_local_write_client()
        assert zot.local is True
        assert zot.local_api_key == "k"
        # Cached so the first write of each call doesn't re-probe.
        assert zot.server_id == "srv-9"
        assert zot.endpoint == "http://localhost:23119/api"

    def test_user_library_is_always_users_zero(self, local_mode, monkeypatch):
        """The local API addresses the user library as users/0, whatever the
        web user id happens to be — hybrid setups do configure a real one."""
        monkeypatch.setenv("ZOTERO_LOCAL_API_KEY", "k")
        monkeypatch.setenv("ZOTERO_LIBRARY_ID", "7654321")
        zot = _client.get_local_write_client()
        assert zot.library_id == "0"
        assert zot.library_type == "users"

    def test_group_library_keeps_its_real_id(self, local_mode, monkeypatch):
        monkeypatch.setenv("ZOTERO_LOCAL_API_KEY", "k")
        _client.set_active_library("555666", "group")
        try:
            zot = _client.get_local_write_client()
            assert zot.library_id == "555666"
            assert zot.library_type == "groups"
        finally:
            _client.clear_active_library()

    def test_cloud_api_key_is_never_sent_to_localhost(self, local_mode, monkeypatch):
        """In hybrid mode the read client is built with the zotero.org key, and
        pyzotero's default headers would turn that into an Authorization
        bearer aimed at a local HTTP server that has no use for it."""
        monkeypatch.setenv("ZOTERO_API_KEY", "WEBKEYSECRET")
        monkeypatch.setenv("ZOTERO_LIBRARY_ID", "7654321")
        for zot in (_client.get_zotero_client(), _client.get_local_zotero_client()):
            if zot is None:
                continue  # no local Zotero running; the probe client is optional
            assert "authorization" not in {k.lower() for k in zot.client.headers}
            # The headers worth restoring are still there.
            assert zot.client.headers.get("zotero-api-version") == "3"

    def test_group_library_without_an_id_returns_none(self, local_mode, monkeypatch):
        """Documented never to raise — the caller reads None as 'fall back'."""
        monkeypatch.setenv("ZOTERO_LOCAL_API_KEY", "k")
        monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "group")
        monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
        assert _client.get_local_write_client() is None

    def test_write_client_gets_a_usable_timeout(self, local_mode, monkeypatch):
        """A write inherits the injected client's timeout — pyzotero passes
        none of its own on _write — so the httpx 5s default would be fatal."""
        monkeypatch.setenv("ZOTERO_LOCAL_API_KEY", "k")
        zot = _client.get_local_write_client()
        assert zot.client.timeout.read == _client._LOCAL_WRITE_TIMEOUT


class TestAuthorize:
    def _stub_zotero(self, monkeypatch, behaviour):
        class _Zot:
            def __init__(self, *a, **k):
                self.server_id = "srv-new"
                self.client = None

            def authorize_local(self, app_name):
                return behaviour(app_name)

        monkeypatch.setattr(_client.zotero, "Zotero", _Zot)

    def test_persists_a_remembered_key(self, local_mode, monkeypatch):
        self._stub_zotero(monkeypatch, lambda n: {"key": "kk", "remember": True})
        result = _client.authorize_local_api("Tester")
        assert result["key"] == "kk"
        assert result["server_id"] == "srv-new"
        assert _client.get_local_write_credentials()[0] == "kk"
        stored = json.loads(_client.ZOTERO_MCP_CONFIG_PATH.read_text(encoding="utf-8"))
        assert stored["local_api"]["app_name"] == "Tester"

    def test_single_use_key_never_reaches_the_disk(self, local_mode, monkeypatch):
        """It is consumed by the first write, and a dead key on disk would
        outrank working web credentials on every later run."""
        self._stub_zotero(monkeypatch, lambda n: {"key": "kk", "remember": False})
        _client.authorize_local_api()
        assert not _client.ZOTERO_MCP_CONFIG_PATH.exists()
        # Still usable for the one write it is good for, and still flagged so
        # the resulting 401 can explain itself.
        assert _client.get_local_write_credentials() == ("kk", "srv-new", "session")
        assert _client.get_write_capabilities(probe=False)["local_key_remember"] is False

    def test_single_use_key_does_not_overwrite_a_stored_one(self, local_mode, monkeypatch):
        _client.store_local_write_credentials("good-key", "srv", remember=True)
        self._stub_zotero(monkeypatch, lambda n: {"key": "throwaway", "remember": False})
        _client.authorize_local_api()
        stored = json.loads(_client.ZOTERO_MCP_CONFIG_PATH.read_text(encoding="utf-8"))
        assert stored["local_api"]["key"] == "good-key"

    def test_no_save_keeps_the_key_out_of_the_config(self, local_mode, monkeypatch):
        self._stub_zotero(monkeypatch, lambda n: {"key": "kk", "remember": True})
        _client.authorize_local_api(persist=False)
        assert not _client.ZOTERO_MCP_CONFIG_PATH.exists()
        assert _client.get_local_write_credentials()[0] == "kk"

    def test_denial_propagates(self, local_mode, monkeypatch):
        def _deny(_name):
            raise LocalAPIDeniedError("denied")

        self._stub_zotero(monkeypatch, _deny)
        with pytest.raises(LocalAPIDeniedError):
            _client.authorize_local_api()
        assert _client.get_local_write_credentials() == (None, None, None)

    def test_rate_limit_propagates(self, local_mode, monkeypatch):
        def _limit(_name):
            raise TooManyRequestsError("slow down")

        self._stub_zotero(monkeypatch, _limit)
        with pytest.raises(TooManyRequestsError):
            _client.authorize_local_api()

    def test_second_concurrent_request_is_refused(self, local_mode, monkeypatch):
        """Two dialogs at once is the failure mode worth preventing."""
        def _reentrant(_name):
            with pytest.raises(_client.ZoteroAuthInProgressError):
                _client.authorize_local_api()
            return {"key": "kk", "remember": True}

        self._stub_zotero(monkeypatch, _reentrant)
        _client.authorize_local_api()

    def test_lock_is_released_after_a_failure(self, local_mode, monkeypatch):
        def _boom(_name):
            raise LocalAPIDeniedError("denied")

        self._stub_zotero(monkeypatch, _boom)
        for _ in range(2):
            with pytest.raises(LocalAPIDeniedError):
                _client.authorize_local_api()


class TestWriteCapabilities:
    def test_local_when_a_key_is_held(self, local_mode, monkeypatch):
        monkeypatch.setenv("ZOTERO_LOCAL_API_KEY", "k")
        caps = _client.get_write_capabilities()
        assert caps["mode"] == "local"
        assert caps["has_local_key"] is True
        assert caps["local_key_source"] == "env"

    def test_hybrid_when_only_web_credentials_exist(self, local_mode, monkeypatch):
        monkeypatch.setenv("ZOTERO_API_KEY", "webkey")
        monkeypatch.setenv("ZOTERO_LIBRARY_ID", "123")
        caps = _client.get_write_capabilities()
        assert caps["mode"] == "hybrid"

    def test_none_when_nothing_is_configured(self, local_mode):
        assert _client.get_write_capabilities()["mode"] == "none"

    def test_web_outside_local_mode(self, monkeypatch):
        monkeypatch.delenv("ZOTERO_LOCAL", raising=False)
        monkeypatch.setenv("ZOTERO_API_KEY", "webkey")
        monkeypatch.setenv("ZOTERO_LIBRARY_ID", "123")
        assert _client.get_write_capabilities()["mode"] == "web"

    def test_reports_single_use_keys(self, local_mode):
        _client.store_local_write_credentials("k", "srv", remember=False)
        assert _client.get_write_capabilities()["local_key_remember"] is False
