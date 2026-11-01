"""zotero_switch_library on a shared HTTP server stays within the client
session that made it (#739).

The active library was one process-wide dict, so over streamable HTTP or
SSE one client's switch silently re-scoped every other client's reads and
writes. A switch made in an HTTP client session is now kept for that
session; stdio, in-memory and session-less (2026-07-28) requests keep the
process-wide library, as before.
"""

import asyncio
import json

import pytest

pytest.importorskip("fastmcp")
httpx = pytest.importorskip("httpx")

from fastmcp import Client  # noqa: E402
from fastmcp.utilities.tests import asgi_server  # noqa: E402

from zotero_mcp import client as _client  # noqa: E402
from zotero_mcp._app import _ZoteroMCP  # noqa: E402
from zotero_mcp._context import Context  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_library_state(monkeypatch):
    monkeypatch.setattr(_client, "_active_library_override", {})
    monkeypatch.setattr(_client, "_session_library_overrides", {}, raising=False)


def _server():
    # No application lifespan: nothing here may start a semantic update.
    server = _ZoteroMCP("library scope probe")

    @server.tool(name="switch")
    def switch(library_id: str, ctx: Context) -> str:
        _client.set_active_library(library_id, "group")
        return "ok"

    @server.tool(name="active")
    def active() -> str:
        return json.dumps({
            "library": _client.get_active_library(),
            "group": _client.get_active_group_id(),
        })

    return server


def _data(result):
    return json.loads(result.data)


def test_sse_clients_keep_their_own_library():
    async def run():
        async with asgi_server(_server(), transport="sse") as srv:
            async with srv.client() as a, srv.client() as b:
                await a.call_tool("switch", {"library_id": "111"})
                await b.call_tool("switch", {"library_id": "222"})
                return (
                    _data(await a.call_tool("active", {})),
                    _data(await b.call_tool("active", {})),
                )

    a, b = asyncio.run(run())

    assert a["group"] == 111
    assert b["group"] == 222
    assert _client._active_library_override == {}


def test_a_switch_in_one_session_leaves_another_on_the_default():
    async def run():
        async with asgi_server(_server(), transport="sse") as srv:
            async with srv.client() as a, srv.client() as b:
                await a.call_tool("switch", {"library_id": "111"})
                return _data(await b.call_tool("active", {}))

    assert asyncio.run(run()) == {"library": {}, "group": 0}


_ACCEPT = {"accept": "application/json, text/event-stream",
           "content-type": "application/json"}


async def _rpc(http, session, rpc_id, method, params):
    headers = dict(_ACCEPT, **({"mcp-session-id": session} if session else {}))
    body = {"jsonrpc": "2.0", "method": method, "params": params}
    if rpc_id is not None:
        body["id"] = rpc_id
    resp = await http.post("/mcp", json=body, headers=headers)
    resp.raise_for_status()
    if rpc_id is None:
        return resp, None
    text = resp.text
    if "data:" in text:
        text = next(ln[5:] for ln in text.splitlines() if ln.startswith("data:"))
    return resp, json.loads(text)["result"]


async def _handshake_session(http):
    resp, _ = await _rpc(http, None, 1, "initialize", {
        "protocolVersion": "2025-11-25", "capabilities": {},
        "clientInfo": {"name": "probe", "version": "1"},
    })
    session = resp.headers["mcp-session-id"]
    await _rpc(http, session, None, "notifications/initialized", {})
    return session


async def _call(http, session, name, args):
    _, result = await _rpc(http, session, 2, "tools/call", {"name": name, "arguments": args})
    return json.loads(result["content"][0]["text"]) if name == "active" else result


def test_streamable_http_sessions_keep_their_own_library():
    async def run():
        async with asgi_server(_server(), transport="streamable-http") as srv:
            transport = httpx.ASGITransport(app=srv.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as http:
                a = await _handshake_session(http)
                b = await _handshake_session(http)
                await _call(http, a, "switch", {"library_id": "111"})
                return await _call(http, a, "active", {}), await _call(http, b, "active", {})

    a, b = asyncio.run(run())

    assert a["group"] == 111
    assert b["group"] == 0


def test_in_memory_and_stdio_keep_the_process_wide_library():
    async def run():
        server = _server()
        async with Client(server) as a:
            await a.call_tool("switch", {"library_id": "333"})
        async with Client(server) as b:
            return _data(await b.call_tool("active", {}))

    assert asyncio.run(run())["group"] == 333
    assert _client._active_library_override["library_id"] == "333"


def test_sessionless_http_keeps_the_process_wide_library():
    # Protocol 2026-07-28 requests carry no session: the switch is
    # server-wide, as it always was, and the tool says so.
    async def run():
        async with asgi_server(_server(), transport="streamable-http") as srv:
            async with srv.client() as a:
                await a.call_tool("switch", {"library_id": "444"})
            async with srv.client() as b:
                return _data(await b.call_tool("active", {}))

    assert asyncio.run(run())["group"] == 444


def test_session_overrides_are_bounded(monkeypatch):
    monkeypatch.setattr(_client, "_MAX_SESSION_OVERRIDES", 2)
    for session in ("s1", "s2", "s3"):
        with _client.library_session(session):
            _client.set_active_library(session, "group")

    assert list(_client._session_library_overrides) == ["s2", "s3"]
    with _client.library_session("s3"):
        assert _client.get_active_library()["library_id"] == "s3"
        _client.clear_active_library()
        assert _client.get_active_library() == {}
    assert _client._active_library_override == {}


def test_switch_note_warns_on_sessionless_http(monkeypatch):
    from zotero_mcp.tools import retrieval

    monkeypatch.setattr("zotero_mcp._context.on_sessionless_http", lambda: True)
    assert "server-wide" in retrieval._switch_note()
    monkeypatch.setattr("zotero_mcp._context.on_sessionless_http", lambda: False)
    assert retrieval._switch_note() == retrieval.SWITCH_SUCCESS_NOTE
