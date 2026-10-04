"""HTTP transports must refuse requests addressed to a foreign host (#REBIND).

Bound to loopback without a token, the server used to accept any Host and
Origin. A web page can rebind its own hostname to 127.0.0.1 and then call
every tool, writes included, same-origin from the user's browser. FastMCP's
Host/Origin guard is off unless asked for, so serve() now asks for it.
"""

import inspect

import pytest

from zotero_mcp import cli

INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
               "clientInfo": {"name": "t", "version": "0"}},
}
HEADERS = {"Content-Type": "application/json",
           "Accept": "application/json, text/event-stream"}


@pytest.fixture(autouse=True)
def _no_user_setting(monkeypatch):
    monkeypatch.delenv(cli._HOST_ORIGIN_ENV_VAR, raising=False)


def test_guard_requested_when_fastmcp_supports_it():
    from zotero_mcp.server import mcp

    if "host_origin_protection" not in inspect.signature(mcp.run_http_async).parameters:
        pytest.skip("installed fastmcp predates the Host/Origin guard")
    assert cli._http_guard_kwargs(mcp) == {"host_origin_protection": "auto"}


def test_user_setting_wins(monkeypatch):
    from zotero_mcp.server import mcp

    monkeypatch.setenv(cli._HOST_ORIGIN_ENV_VAR, "false")
    assert cli._http_guard_kwargs(mcp) == {}


def test_old_fastmcp_gets_a_warning(capsys):
    class OldServer:
        async def run_http_async(self, transport="http", host=None, port=None):
            pass

    assert cli._http_guard_kwargs(OldServer()) == {}
    assert "DNS rebinding" in capsys.readouterr().err


@pytest.fixture
def _isolated_cli(monkeypatch, tmp_path):
    """cli.main() loads ~/.config/zotero-mcp into os.environ; keep it from
    reading the real one and from leaking ZOTERO_* into later tests."""
    import os

    saved = dict(os.environ)
    monkeypatch.setenv("HOME", str(tmp_path))
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.mark.parametrize("transport", ["streamable-http", "sse"])
def test_serve_passes_the_guard(monkeypatch, transport, _isolated_cli):
    from zotero_mcp.server import mcp

    calls = []
    monkeypatch.setattr(mcp, "run", lambda **kw: calls.append(kw))
    seen = []
    monkeypatch.setattr(
        cli, "_http_guard_kwargs",
        lambda server, *a, **k: seen.append(a) or {"host_origin_protection": "auto"},
    )
    monkeypatch.setattr(cli, "_preimport_semantic_search_on_main_thread", lambda: None, raising=False)
    monkeypatch.setattr(cli, "_warmup_reranker_in_background", lambda: None)
    monkeypatch.setattr("sys.argv", ["zotero-mcp", "serve", "--transport", transport, "--port", "8799"])
    with pytest.warns(UserWarning) if transport == "sse" else _nullcontext():
        cli.main()
    assert calls and calls[-1].get("host_origin_protection") == "auto"
    assert seen == [(transport, "localhost")]  # transport and bind host reach the guard


def _nullcontext():
    import contextlib

    return contextlib.nullcontext()


def test_guarded_app_rejects_a_rebound_host_and_serves_localhost():
    from starlette.testclient import TestClient

    from zotero_mcp.server import mcp

    if "host_origin_protection" not in inspect.signature(mcp.http_app).parameters:
        pytest.skip("installed fastmcp predates the Host/Origin guard")
    app = mcp.http_app(host_origin_protection="auto")
    with TestClient(app, base_url="http://127.0.0.1:8799") as client:
        rebound = client.post("/mcp", json=INIT, headers={
            **HEADERS, "Host": "evil.example:8799", "Origin": "http://evil.example:8799"})
        foreign_origin = client.post("/mcp", json=INIT, headers={
            **HEADERS, "Origin": "http://evil.example"})
        local = client.post("/mcp", json=INIT, headers=HEADERS)
    assert rebound.status_code == 421
    assert foreign_origin.status_code == 403
    assert local.status_code == 200


# FastMCP's own option does not reach the SSE app (create_sse_app takes no
# host_origin_protection and silently ignores it), so SSE gets the guard as
# a middleware instead.

def _mw_cls():
    return pytest.importorskip("fastmcp.server.http").HostOriginGuardMiddleware


def test_sse_gets_the_guard_as_middleware():
    from zotero_mcp.server import mcp

    kwargs = cli._http_guard_kwargs(mcp, "sse", "localhost")
    (mw,) = kwargs["middleware"]
    assert mw.cls is _mw_cls()
    assert mw.kwargs["mode"] == "auto"
    assert "localhost" in mw.kwargs["allowed_hosts"]
    assert "host_origin_protection" not in kwargs


def test_sse_guard_follows_the_user_setting(monkeypatch):
    from zotero_mcp.server import mcp

    monkeypatch.setenv(cli._HOST_ORIGIN_ENV_VAR, "false")
    assert cli._http_guard_kwargs(mcp, "sse", "localhost") == {}
    monkeypatch.setenv(cli._HOST_ORIGIN_ENV_VAR, "true")
    (mw,) = cli._http_guard_kwargs(mcp, "sse", "localhost")["middleware"]
    assert mw.kwargs["mode"] == "strict"


def test_sse_guard_keeps_user_allowed_hosts(monkeypatch):
    import fastmcp

    from zotero_mcp.server import mcp

    monkeypatch.setattr(fastmcp.settings, "http_allowed_hosts", ["abc.ngrok.app"], raising=False)
    (mw,) = cli._http_guard_kwargs(mcp, "sse", "127.0.0.1")["middleware"]
    assert mw.kwargs["allowed_hosts"] == ["abc.ngrok.app", "127.0.0.1"]


def test_sse_without_middleware_support_warns(capsys):
    class OldServer:
        async def run_http_async(self, transport="http", host=None, port=None):
            pass

    assert cli._http_guard_kwargs(OldServer(), "sse", "localhost") == {}
    assert "DNS rebinding" in capsys.readouterr().err


def test_guarded_sse_app_rejects_a_rebound_host_and_serves_localhost():
    from starlette.testclient import TestClient

    from zotero_mcp.server import mcp

    app = mcp.http_app(transport="sse", **cli._http_guard_kwargs(mcp, "sse", "127.0.0.1"))
    with TestClient(app, base_url="http://127.0.0.1:8799") as client:
        rebound = client.post("/messages/", json=INIT, headers={
            "Host": "evil.example:8799", "Origin": "http://evil.example:8799"})
        foreign_origin = client.post("/messages/", json=INIT, headers={
            "Origin": "http://evil.example"})
        local = client.post("/messages/", json=INIT)
    assert rebound.status_code == 421
    assert foreign_origin.status_code == 403
    # No session id, so the app itself answers, but the guard let it through.
    assert local.status_code not in (421, 403)


@pytest.mark.parametrize("transport", ["streamable-http", "sse"])
def test_operator_is_told_how_to_allow_a_proxy(transport, capsys):
    from zotero_mcp.server import mcp

    cli._http_guard_kwargs(mcp, transport, "localhost")
    err = capsys.readouterr().err
    assert "FASTMCP_HTTP_ALLOWED_HOSTS" in err and "tunnel" in err


def test_no_notice_when_the_user_turned_the_guard_off(monkeypatch, capsys):
    from zotero_mcp.server import mcp

    monkeypatch.setenv(cli._HOST_ORIGIN_ENV_VAR, "false")
    cli._http_guard_kwargs(mcp, "sse", "localhost")
    assert capsys.readouterr().err == ""
