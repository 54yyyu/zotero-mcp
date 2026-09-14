"""Tests for the local-write authorization CLI command and MCP tools."""

import argparse
import json
import time

import httpx
import pytest
from conftest import DummyContext
from pyzotero.errors import LocalAPIDeniedError, TooManyRequestsError, UnsupportedParamsError

from zotero_mcp import cli as _cli
from zotero_mcp import client as _client
from zotero_mcp import server


@pytest.fixture
def local_mode(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)


def _server_supports_local_writes():
    _client._local_probe_cache.update(
        {"server_id": "srv-1", "checked_at": time.monotonic()}
    )


def _args(**overrides):
    defaults = {
        "app_name": None, "timeout": 120.0, "status": False,
        "revoke": False, "print_key": False, "no_save": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCliCommand:
    def test_parser_and_dispatch_agree(self, monkeypatch):
        """The parser and the handler have to stay in step — a CLI subcommand
        shipped broken for exactly this reason (#335)."""
        called = {}

        def _capture(parsed):
            called["args"] = parsed
            return 0

        monkeypatch.setattr(_cli, "setup_zotero_environment", lambda: None)
        monkeypatch.setattr(_cli, "cmd_authorize_local", _capture)
        monkeypatch.setattr("sys.argv", ["zotero-mcp", "authorize-local", "--status"])
        with pytest.raises(SystemExit) as exc:
            _cli.main()
        assert exc.value.code == 0
        assert called["args"].status is True
        # Every flag the handler reads must exist on the parsed namespace.
        for flag in ("app_name", "timeout", "status", "revoke", "print_key", "no_save"):
            assert hasattr(called["args"], flag), flag

    def test_status_does_not_open_a_dialog(self, local_mode, monkeypatch, capsys):
        monkeypatch.setattr(
            _client, "authorize_local_api",
            lambda **k: pytest.fail("status must not authorize"),
        )
        assert _cli.cmd_authorize_local(_args(status=True)) == 0
        assert "Write mode:" in capsys.readouterr().out

    def test_revoke_clears_the_stored_key(self, local_mode, capsys):
        _client.store_local_write_credentials("k", "srv", remember=True)
        assert _cli.cmd_authorize_local(_args(revoke=True)) == 0
        assert _client.get_local_write_credentials() == (None, None, None)
        out = capsys.readouterr().out
        assert "removed" in out
        # Zotero keeps its own record; say so rather than implying we cleared it.
        assert "Clear Write Authorizations" in out

    def test_revoke_warns_about_a_key_it_cannot_reach(self, local_mode, monkeypatch, capsys):
        """Revoke can't unset a variable in someone else's environment, and
        staying quiet would make it look like the key was gone."""
        monkeypatch.setenv("ZOTERO_LOCAL_API_KEY", "from-env")
        _cli.cmd_authorize_local(_args(revoke=True))
        out = capsys.readouterr().out
        assert "ZOTERO_LOCAL_API_KEY is still set" in out
        assert _client.get_local_write_credentials()[0] == "from-env"

    def test_refuses_outside_local_mode(self, monkeypatch, capsys):
        monkeypatch.delenv("ZOTERO_LOCAL", raising=False)
        assert _cli.cmd_authorize_local(_args()) == 7
        assert "ZOTERO_LOCAL is not enabled" in capsys.readouterr().out

    def test_reports_an_older_zotero_without_prompting(self, local_mode, capsys):
        assert _cli.cmd_authorize_local(_args()) == 5
        assert "Zotero 10" in capsys.readouterr().out

    def test_grants_and_saves(self, local_mode, monkeypatch, capsys):
        _server_supports_local_writes()
        monkeypatch.setattr(
            _client, "authorize_local_api",
            lambda **k: {"key": "kk", "remember": True, "server_id": "srv-1",
                         "stored_at": "/tmp/config.json"},
        )
        assert _cli.cmd_authorize_local(_args()) == 0
        out = capsys.readouterr().out
        assert "Always Allow" in out  # instructions shown before blocking
        assert "reusable key" in out
        assert "Saved to" in out

    def test_warns_loudly_about_a_single_use_key(self, local_mode, monkeypatch, capsys):
        _server_supports_local_writes()
        monkeypatch.setattr(
            _client, "authorize_local_api",
            lambda **k: {"key": "kk", "remember": False, "stored_at": "/tmp/c.json"},
        )
        assert _cli.cmd_authorize_local(_args()) == 0
        assert "SINGLE-USE" in capsys.readouterr().out

    def test_does_not_print_the_key_by_default(self, local_mode, monkeypatch, capsys):
        _server_supports_local_writes()
        monkeypatch.setattr(
            _client, "authorize_local_api",
            lambda **k: {"key": "super-secret", "remember": True, "stored_at": "/x"},
        )
        _cli.cmd_authorize_local(_args())
        assert "super-secret" not in capsys.readouterr().out

    def test_print_flag_emits_exportable_variables(self, local_mode, monkeypatch, capsys):
        _server_supports_local_writes()
        monkeypatch.setattr(
            _client, "authorize_local_api",
            lambda **k: {"key": "kk", "remember": True, "server_id": "srv-1",
                         "stored_at": "/x"},
        )
        _cli.cmd_authorize_local(_args(print_key=True))
        out = capsys.readouterr().out
        assert "ZOTERO_LOCAL_API_KEY=kk" in out
        assert "ZOTERO_LOCAL_SERVER_ID=srv-1" in out

    @pytest.mark.parametrize(
        ("error", "code"),
        [
            (LocalAPIDeniedError("denied"), 3),
            (TooManyRequestsError("slow"), 4),
            (UnsupportedParamsError("no server id"), 5),
        ],
    )
    def test_distinct_exit_codes_per_failure(self, local_mode, monkeypatch, error, code):
        _server_supports_local_writes()

        def _raise(**_kwargs):
            raise error

        monkeypatch.setattr(_client, "authorize_local_api", _raise)
        assert _cli.cmd_authorize_local(_args()) == code


class TestAuthorizeTool:
    def test_explains_that_local_mode_is_off(self, monkeypatch):
        monkeypatch.delenv("ZOTERO_LOCAL", raising=False)
        result = server.authorize_local_writes(ctx=DummyContext())
        assert "not in local mode" in result

    def test_reports_a_persistent_grant(self, local_mode, monkeypatch):
        monkeypatch.setattr(
            _client, "authorize_local_api", lambda **k: {"key": "kk", "remember": True}
        )
        result = server.authorize_local_writes(ctx=DummyContext())
        assert "granted and saved" in result

    def test_reports_a_single_use_grant(self, local_mode, monkeypatch):
        monkeypatch.setattr(
            _client, "authorize_local_api", lambda **k: {"key": "kk", "remember": False}
        )
        result = server.authorize_local_writes(ctx=DummyContext())
        assert "SINGLE write" in result
        assert "Always Allow" in result

    def test_timeout_is_clamped_under_the_client_deadline(self, local_mode, monkeypatch):
        """Past FastMCP's ~60s timeout the transport dies first and the user
        sees an opaque failure instead of 'you didn't answer the dialog'."""
        seen = {}

        def _record(**kwargs):
            seen.update(kwargs)
            return {"key": "kk", "remember": True}

        monkeypatch.setattr(_client, "authorize_local_api", _record)
        server.authorize_local_writes(timeout=9999, ctx=DummyContext())
        assert seen["timeout"] == 55

    def test_never_raises(self, local_mode, monkeypatch):
        for error in (
            LocalAPIDeniedError("d"),
            TooManyRequestsError("r"),
            UnsupportedParamsError("u"),
            httpx.ReadTimeout("t"),
            RuntimeError("boom"),
            _client.ZoteroAuthInProgressError("busy"),
        ):
            def _raise(_e=error, **_kwargs):
                raise _e

            monkeypatch.setattr(_client, "authorize_local_api", _raise)
            result = server.authorize_local_writes(ctx=DummyContext())
            assert isinstance(result, str) and result

    def test_timeout_says_the_dialog_is_probably_still_open(self, local_mode, monkeypatch):
        def _raise(**_kwargs):
            raise httpx.ReadTimeout("timed out")

        monkeypatch.setattr(_client, "authorize_local_api", _raise)
        assert "still open" in server.authorize_local_writes(ctx=DummyContext())


class TestCapabilitiesTool:
    def test_points_at_authorizing_when_that_is_the_fix(self, local_mode):
        _server_supports_local_writes()
        result = server.write_capabilities(ctx=DummyContext())
        assert "**Mode:** none" in result
        assert "zotero_authorize_local_writes" in result

    def test_points_at_web_credentials_on_an_older_zotero(self, local_mode):
        result = server.write_capabilities(ctx=DummyContext())
        assert "needs Zotero 10 or newer" in result
        assert "ZOTERO_API_KEY" in result

    def test_flags_a_single_use_key(self, local_mode):
        _server_supports_local_writes()
        _client.store_local_write_credentials("k", "srv-1", remember=False)
        result = server.write_capabilities(ctx=DummyContext())
        assert "SINGLE USE" in result

    def test_nudges_hybrid_users_who_could_go_local(self, local_mode, monkeypatch):
        _server_supports_local_writes()
        monkeypatch.setenv("ZOTERO_API_KEY", "webkey")
        monkeypatch.setenv("ZOTERO_LIBRARY_ID", "123")
        result = server.write_capabilities(ctx=DummyContext())
        assert "**Mode:** hybrid" in result
        assert "zotero_authorize_local_writes" in result

    def test_never_leaks_the_key(self, local_mode):
        _client.store_local_write_credentials("super-secret", "srv-1", remember=True)
        assert "super-secret" not in server.write_capabilities(ctx=DummyContext())


class TestSetupDoesNotClobberTheKey:
    def test_standalone_config_rewrite_preserves_local_api(self, tmp_path, monkeypatch):
        """`zotero-mcp setup` rebuilds client_env from scratch; the key lives
        in its own section precisely so that rewrite cannot drop it."""
        from zotero_mcp import setup_helper

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setattr(
            _client, "ZOTERO_MCP_CONFIG_PATH",
            tmp_path / ".config" / "zotero-mcp" / "config.json",
        )
        _client.store_local_write_credentials("keep-me", "srv", remember=True)

        setup_helper._write_standalone_config(
            local=True, api_key="", library_id="", library_type="user",
            semantic_config={}, no_claude=True,
        )

        stored = json.loads(
            (tmp_path / ".config" / "zotero-mcp" / "config.json").read_text(encoding="utf-8")
        )
        assert stored["local_api"]["key"] == "keep-me"
        assert "keep-me" not in json.dumps(stored["client_env"])
