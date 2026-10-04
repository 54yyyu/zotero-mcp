"""HTTP authentication at the real FastMCP boundary and CLI configuration."""

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import uvicorn
from fastmcp import FastMCP
from fastmcp.server.auth import TokenVerifier
from fastmcp.server.middleware import Middleware
from starlette.testclient import TestClient

from zotero_mcp._http_auth import (
    _is_loopback_host,
    configure_http_auth,
    validate_auth_token,
)

TOKEN_ENV = "ZOTERO_MCP_AUTH_TOKEN"
TOKEN = "synthetic-test-secret_0123456789"
AUTH_HEADERS = {"Authorization": f"Bearer {TOKEN}"}
ACCEPT_HEADERS = {"Accept": "application/json, text/event-stream"}
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "auth-regression-test", "version": "1"},
    },
}


@pytest.fixture(autouse=True)
def isolate_auth_environment(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)


class RequestProbe(Middleware):
    def __init__(self):
        self.methods = []

    async def on_request(self, context, call_next):
        self.methods.append(context.method)
        return await call_next(context)


def protected_server(monkeypatch, *, allow_unauthenticated=False):
    """A real MCP server whose only tool has no external dependencies."""
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    server = FastMCP("Authentication regression fixture")
    probe = RequestProbe()
    server.add_middleware(probe)
    calls = []

    @server.tool
    def private_fixture_tool() -> str:
        """Return a synthetic private value."""
        calls.append(True)
        return "private-fixture-result"

    configure_http_auth(server, "0.0.0.0", allow_unauthenticated)
    return server, probe, calls


@pytest.mark.parametrize("host", ["localhost", "LOCALHOST", "LocalHost.", "127.0.0.1", "127.9.8.7", "::1"])
def test_loopback_without_token_is_allowed_with_one_stderr_warning(host, capsys):
    server = FastMCP("Loopback fixture")
    assert _is_loopback_host(host)
    configure_http_auth(server, host)
    assert server.auth is None
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("Warning:") == 1
    assert "tunnel" in captured.err.lower()


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "::", "192.168.1.4", "example.test", "localhost.example.test", "localhost..", "", "127.1"]
)
def test_nonloopback_without_auth_is_rejected_without_dns(host, monkeypatch, capsys):
    def no_dns(*args, **kwargs):
        pytest.fail("The bind guard must not resolve hostnames")

    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    assert not _is_loopback_host(host)
    with pytest.raises(ValueError, match="non-loopback"):
        configure_http_auth(FastMCP("Bind guard fixture"), host)
    assert capsys.readouterr().out == ""


def test_explicit_override_allows_unauthenticated_nonloopback(capsys):
    server = FastMCP("Explicit override fixture")
    configure_http_auth(server, "0.0.0.0", allow_unauthenticated=True)
    assert server.auth is None
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("Warning:") == 1


@pytest.mark.parametrize("value", [None, 123, True, [], {}, "", " ", " secret", "secret\n", "bad:secret", "s\u00e9cret", "ab=cd"])
def test_invalid_token_rejected_without_echoing_value(value):
    with pytest.raises(ValueError, match=TOKEN_ENV) as caught:
        validate_auth_token(value)
    if isinstance(value, str) and value.strip():
        assert value not in str(caught.value)


@pytest.mark.parametrize("value", ["abc", "AZaz09-._~+/", "YWJjZA=="])
def test_rfc6750_bearer_characters_are_supported(value):
    assert validate_auth_token(value) == value


def test_existing_auth_provider_is_preserved_without_shared_secret(capsys):
    class ExistingProvider(TokenVerifier):
        async def verify_token(self, token):
            return None

    provider = ExistingProvider()
    server = FastMCP("Existing authentication fixture", auth=provider)
    configure_http_auth(server, "0.0.0.0", allow_unauthenticated=True)
    assert server.auth is provider
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("value", ["", " ", "bad\nsecret"])
def test_override_cannot_suppress_invalid_configured_token(value, monkeypatch, capsys):
    monkeypatch.setenv(TOKEN_ENV, value)
    with pytest.raises(ValueError, match=TOKEN_ENV):
        configure_http_auth(FastMCP("Invalid token fixture"), "0.0.0.0", True)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("authorization", [None, "Bearer wrong-secret", "Basic wrong-secret", "Bearer"])
@pytest.mark.parametrize(
    ("transport", "method", "path"),
    [
        ("streamable-http", "GET", "/mcp"),
        ("streamable-http", "POST", "/mcp"),
        ("streamable-http", "DELETE", "/mcp"),
        ("sse", "GET", "/sse/"),
        ("sse", "POST", "/messages/"),
    ],
)
def test_real_transport_rejects_missing_or_wrong_credentials(
    transport, method, path, authorization, monkeypatch,
):
    server, probe, calls = protected_server(monkeypatch, allow_unauthenticated=True)
    app = server.http_app(
        transport=transport, path="/sse/" if transport == "sse" else "/mcp", stateless_http=False,
    )
    headers = dict(ACCEPT_HEADERS)
    if authorization is not None:
        headers["Authorization"] = authorization
    with TestClient(app) as client:
        # Neither tool discovery nor invocation may reach the MCP dispatcher.
        for rpc_method in ("tools/list", "tools/call", "server/discover"):
            payload = {"jsonrpc": "2.0", "id": 2, "method": rpc_method}
            if rpc_method == "tools/call":
                payload["params"] = {"name": "private_fixture_tool", "arguments": {}}
            response = client.request(method, path, headers=headers, json=payload)
            assert response.status_code == 401
            assert response.headers["www-authenticate"].lower().startswith("bearer")
            assert "private_fixture" not in response.text
            assert TOKEN not in response.text
    assert probe.methods == []
    assert calls == []


def test_streamable_http_authenticates_real_discovery_and_tool_call(monkeypatch, capsys):
    server, probe, calls = protected_server(monkeypatch)
    app = server.http_app(path="/mcp", json_response=True, stateless_http=False)
    with TestClient(app) as client:
        headers = {**ACCEPT_HEADERS, **AUTH_HEADERS}
        initialized = client.post("/mcp", headers=headers, json=INITIALIZE)
        assert initialized.status_code == 200, initialized.text
        assert "result" in initialized.json()
        session = initialized.headers.get("mcp-session-id")
        if session:
            headers["Mcp-Session-Id"] = session
        headers["MCP-Protocol-Version"] = initialized.json()["result"]["protocolVersion"]
        notification = client.post(
            "/mcp", headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        assert notification.status_code in (200, 202)
        listed = client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert listed.status_code == 200, listed.text
        assert [tool["name"] for tool in listed.json()["result"]["tools"]] == ["private_fixture_tool"]
        payload = {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "private_fixture_tool", "arguments": {}},
        }
        rejected = client.post("/mcp", headers={**headers, "Authorization": "Bearer wrong-secret"}, json=payload)
        assert rejected.status_code == 401
        assert calls == []
        invoked = client.post("/mcp", headers=headers, json=payload)
        assert invoked.status_code == 200, invoked.text
        assert invoked.json()["result"]["content"][0]["text"] == "private-fixture-result"
        assert calls == [True]
    assert "tools/list" in probe.methods
    assert probe.methods.count("tools/call") == 1
    captured = capsys.readouterr()
    assert TOKEN not in captured.out + captured.err
    assert "authentication is disabled" not in captured.err


@contextmanager
def loopback_server(app):
    """Run only the synthetic app, using an already-bound ephemeral socket."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    address = f"http://127.0.0.1:{listener.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_config=None, access_log=False, lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                pytest.fail("Synthetic loopback server did not start")
            time.sleep(0.01)
        yield address
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        assert not thread.is_alive(), "Synthetic loopback server did not stop"


def next_sse_data(lines):
    for line in lines:
        if line.startswith("data: "):
            return line[6:]
    pytest.fail("SSE stream closed before the next event")


def test_sse_authenticates_connection_and_each_message(monkeypatch):
    server, probe, calls = protected_server(monkeypatch)
    app = server.http_app(transport="sse", path="/sse/")
    with loopback_server(app) as address, httpx.Client(base_url=address, timeout=10, trust_env=False) as client:
        with client.stream("GET", "/sse/", headers=AUTH_HEADERS) as stream:
            assert stream.status_code == 200
            lines = stream.iter_lines()
            message_path = next_sse_data(lines)
            assert message_path.startswith("/messages/")
            # An authenticated SSE session alone does not authorize its POSTs.
            for headers in ({}, {"Authorization": "Bearer wrong-secret"}):
                rejected = client.post(message_path, headers=headers, json=INITIALIZE)
                assert rejected.status_code == 401
                assert rejected.headers["www-authenticate"].lower().startswith("bearer")
            assert probe.methods == []
            initialized = client.post(message_path, headers=AUTH_HEADERS, json=INITIALIZE)
            assert initialized.status_code == 202
            assert json.loads(next_sse_data(lines))["id"] == 1
            client.post(
                message_path, headers=AUTH_HEADERS,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            ).raise_for_status()
            client.post(
                message_path, headers=AUTH_HEADERS,
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            ).raise_for_status()
            listed = json.loads(next_sse_data(lines))
            assert [tool["name"] for tool in listed["result"]["tools"]] == ["private_fixture_tool"]
            payload = {
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "private_fixture_tool", "arguments": {}},
            }
            assert client.post(message_path, json=payload).status_code == 401
            assert calls == []
            client.post(message_path, headers=AUTH_HEADERS, json=payload).raise_for_status()
            invoked = json.loads(next_sse_data(lines))
            assert invoked["result"]["content"][0]["text"] == "private-fixture-result"
    assert calls == [True]
    assert probe.methods.count("tools/call") == 1


@pytest.fixture
def cli_fixture(monkeypatch, tmp_path):
    """Exercise real CLI parsing/configuration without starting a listener."""
    from zotero_mcp import cli

    # The CLI adds fallback variables directly, outside monkeypatch.setenv.
    # Keep those additions from leaking into subsequent tests.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for key in list(os.environ):
        if key.startswith("ZOTERO_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("ZOTERO_NO_CLAUDE", "true")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    from zotero_mcp.server import mcp

    monkeypatch.setattr(mcp, "auth", None)
    run = Mock()
    preimport = Mock()
    warmup = Mock()
    monkeypatch.setattr(mcp, "run", run)
    monkeypatch.setattr(cli, "_preimport_semantic_search_on_main_thread", preimport)
    monkeypatch.setattr(cli, "_warmup_reranker_in_background", warmup)
    def no_prompt(*args, **kwargs):
        pytest.fail("serve must not prompt for credentials")

    monkeypatch.setattr("builtins.input", no_prompt)
    monkeypatch.setattr("getpass.getpass", no_prompt)
    config_path = tmp_path / ".config" / "zotero-mcp" / "config.json"
    config_path.parent.mkdir(parents=True)

    def write_config(env):
        config_path.write_text(json.dumps({"client_env": env}), encoding="utf-8")

    def invoke(*args):
        monkeypatch.setattr(sys, "argv", ["zotero-mcp", *args])
        return cli.main()

    return SimpleNamespace(mcp=mcp, run=run, warmup=warmup, preimport=preimport, write_config=write_config, invoke=invoke)


def test_serve_help_exposes_override_without_a_token_argument(cli_fixture, capsys):
    with pytest.raises(SystemExit) as caught:
        cli_fixture.invoke("serve", "--help")
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "--allow-unauthenticated" in help_text
    assert "--token" not in help_text
    cli_fixture.run.assert_not_called()


@pytest.mark.parametrize("transport", ["streamable-http", "sse"])
def test_cli_rejects_unauthenticated_public_bind_before_warmup(transport, cli_fixture, capsys):
    with pytest.raises(SystemExit) as caught:
        cli_fixture.invoke("serve", "--transport", transport, "--host", "0.0.0.0")
    assert caught.value.code == 2
    cli_fixture.run.assert_not_called()
    cli_fixture.preimport.assert_not_called()
    cli_fixture.warmup.assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "non-loopback" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("transport", ["streamable-http", "sse"])
def test_cli_loads_standalone_token_before_auth_configuration(transport, cli_fixture, capsys):
    cli_fixture.write_config({TOKEN_ENV: TOKEN})
    cli_fixture.invoke("serve", "--transport", transport, "--host", "0.0.0.0", "--port", "8765")
    cli_fixture.run.assert_called_once_with(transport=transport, host="0.0.0.0", port=8765)
    assert cli_fixture.mcp.auth is not None
    captured = capsys.readouterr()
    assert TOKEN not in captured.out + captured.err
    assert "authentication is disabled" not in captured.err


@pytest.mark.parametrize("raw_token", [None, 123, True, "", " ", "invalid secret"])
def test_cli_rejects_raw_invalid_config_even_with_override(raw_token, cli_fixture, capsys):
    cli_fixture.write_config({TOKEN_ENV: raw_token})
    with pytest.raises(SystemExit) as caught:
        cli_fixture.invoke("serve", "--transport", "streamable-http", "--allow-unauthenticated")
    assert caught.value.code == 2
    cli_fixture.run.assert_not_called()
    cli_fixture.warmup.assert_not_called()
    captured = capsys.readouterr()
    assert TOKEN_ENV in captured.err
    assert captured.out == ""
    assert "invalid secret" not in captured.err
    assert "Traceback" not in captured.err


def test_environment_token_takes_precedence_over_invalid_config(cli_fixture, monkeypatch):
    cli_fixture.write_config({TOKEN_ENV: None})
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    cli_fixture.invoke("serve", "--transport", "streamable-http", "--host", "0.0.0.0")
    assert asyncio.run(cli_fixture.mcp.auth.verify_token(TOKEN)) is not None
    assert asyncio.run(cli_fixture.mcp.auth.verify_token("None")) is None


def test_empty_environment_token_does_not_fall_back_to_valid_config(cli_fixture, monkeypatch):
    cli_fixture.write_config({TOKEN_ENV: TOKEN})
    monkeypatch.setenv(TOKEN_ENV, "")
    with pytest.raises(SystemExit) as caught:
        cli_fixture.invoke("serve", "--transport", "streamable-http", "--allow-unauthenticated")
    assert caught.value.code == 2
    cli_fixture.run.assert_not_called()


def test_cli_accepts_explicit_unauthenticated_override(cli_fixture, capsys):
    cli_fixture.invoke("serve", "--transport", "streamable-http", "--host", "0.0.0.0", "--allow-unauthenticated")
    cli_fixture.run.assert_called_once_with(transport="streamable-http", host="0.0.0.0", port=8000)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("Warning:") == 1


@pytest.mark.parametrize("args", [(), ("serve",), ("serve", "--transport", "stdio", "--host", "0.0.0.0")])
def test_stdio_does_not_validate_or_enable_http_auth(args, cli_fixture, capsys):
    cli_fixture.write_config({TOKEN_ENV: None})
    cli_fixture.invoke(*args)
    cli_fixture.run.assert_called_once_with(transport="stdio")
    assert cli_fixture.mcp.auth is None
    captured = capsys.readouterr()
    assert TOKEN_ENV not in captured.out + captured.err
    assert "authentication is disabled" not in captured.err


@pytest.mark.parametrize("raw_token", [TOKEN, None, 123])
def test_cli_loads_and_validates_claude_config(raw_token, cli_fixture, tmp_path, monkeypatch):
    from zotero_mcp import setup_helper

    monkeypatch.delenv("ZOTERO_NO_CLAUDE")
    # Cover the actual discovery path without consulting any user config.
    for name in ("APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(name, str(tmp_path / name))
    config_path = setup_helper.claude_config_candidates()[0]
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps({"mcpServers": {"zotero": {"env": {TOKEN_ENV: raw_token}}}}), encoding="utf-8",
    )
    if raw_token == TOKEN:
        cli_fixture.invoke("serve", "--transport", "streamable-http", "--host", "0.0.0.0")
        cli_fixture.run.assert_called_once()
        assert cli_fixture.mcp.auth is not None
    else:
        with pytest.raises(SystemExit) as caught:
            cli_fixture.invoke("serve", "--transport", "streamable-http", "--allow-unauthenticated")
        assert caught.value.code == 2
        cli_fixture.run.assert_not_called()


def test_dotenv_token_keeps_existing_precedence_over_standalone_config(tmp_path):
    """A fresh CLI import must still load .env before applying config values."""
    dotenv_token = "synthetic-dotenv-secret"
    config_path = tmp_path / ".config" / "zotero-mcp" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"client_env": {TOKEN_ENV: TOKEN}}), encoding="utf-8")
    (tmp_path / ".env").write_text(f"{TOKEN_ENV}={dotenv_token}\n", encoding="utf-8")
    script = r'''
import asyncio
import json
import pathlib
import sys
from unittest.mock import patch

sys.path.insert(0, sys.argv[1])
pathlib.Path.home = classmethod(lambda cls: pathlib.Path.cwd())
from fastmcp import FastMCP
from zotero_mcp import cli

def capture_run(self, **kwargs):
    print(json.dumps({
        "transport": kwargs["transport"],
        "dotenv_accepted": asyncio.run(self.auth.verify_token("synthetic-dotenv-secret")) is not None,
        "config_accepted": asyncio.run(self.auth.verify_token("synthetic-test-secret_0123456789")) is not None,
    }))

with patch.object(FastMCP, "run", capture_run), \
     patch.object(cli, "_preimport_semantic_search_on_main_thread"), \
     patch.object(cli, "_warmup_reranker_in_background"):
    sys.argv = ["zotero-mcp", "serve", "--transport", "streamable-http", "--host", "0.0.0.0"]
    cli.main()
'''
    env = {key: value for key, value in os.environ.items() if not key.startswith(("ZOTERO_", "FASTMCP_"))}
    env["ZOTERO_NO_CLAUDE"] = "true"
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[1] / "src")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "transport": "streamable-http", "dotenv_accepted": True, "config_accepted": False,
    }
    assert dotenv_token not in result.stdout + result.stderr
    assert TOKEN not in result.stdout + result.stderr
