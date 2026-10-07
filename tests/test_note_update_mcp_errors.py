"""Exercise note updates through MCP, without touching a real Zotero library."""

import asyncio
import copy
import json
import sys

import httpx
import pytest
from conftest import pyzotero_http_module
from fastmcp import Client, FastMCP
from pyzotero.zotero import Zotero
from pyzotero.zotero_errors import ResourceNotFoundError

from zotero_mcp import cli_standalone
from zotero_mcp.tools import annotations

KEY = "NOTE0001"
ORIGINAL = {
    "key": KEY,
    "version": 1,
    "links": {},
    "library": {},
    "meta": {},
    "data": {"key": KEY, "version": 1, "itemType": "note", "note": "<p>old</p>"},
}


class NoteBackend:
    def __init__(self):
        self.record = copy.deepcopy(ORIGINAL)
        self.read_error = None
        self.write_error = None
        self.response = True
        self.writes = 0

    def item(self, key):
        if self.read_error:
            raise self.read_error
        return copy.deepcopy(self.record)

    def update_item(self, item):
        self.writes += 1
        if self.write_error:
            raise self.write_error
        if self.response:
            self.record = copy.deepcopy(item)
        return self.response


@pytest.fixture
def note_server(monkeypatch):
    backend = NoteBackend()
    monkeypatch.setattr(annotations, "_get_note_write_client", lambda _: (backend, None))
    server = FastMCP("Note update protocol regression")
    server.tool(getattr(annotations.manage_note, "fn", annotations.manage_note), name="zotero_manage_note")
    return server, backend


def call(server, **arguments):
    async def run():
        async with Client(server) as client:
            return await client.call_tool_mcp(
                "zotero_manage_note", {"action": "update", "item_key": KEY, **arguments}, timeout=5
            )

    return asyncio.run(run())


def text(result):
    return "\n".join(block.text for block in result.content if block.type == "text")


@pytest.mark.parametrize("status", [403, 412, 500])
def test_http_write_failure_is_mcp_error(note_server, monkeypatch, status):
    """Real PyZotero GET/PATCH over a synthetic transport reproduces HTTP 500."""
    server, _ = note_server
    http = pyzotero_http_module()
    requests = []

    def respond(request):
        requests.append(request.method)
        if request.url.path.endswith("/itemFields"):
            return http.Response(200, json=[])
        if request.method == "GET":
            return http.Response(200, json=ORIGINAL)
        assert request.method == "PATCH"
        assert json.loads(request.content)["note"] == "<p>new</p>"
        return http.Response(status, text="Synthetic write failure")

    with http.Client(transport=http.MockTransport(respond)) as client:
        zot = Zotero(
            "0", "user", local=True, api_key=None, client=client, local_api_key="test-key", server_id="test-server"
        )
        monkeypatch.setattr(annotations, "_get_note_write_client", lambda _: (zot, None))
        result = call(server, note_text="<p>new</p>")

    assert requests[0] == "GET"
    assert requests[-1] == "PATCH"
    assert requests.count("PATCH") == 1
    assert result.isError is True
    assert "Error updating note" in text(result)
    assert str(status) in text(result)
    assert "Successfully" not in text(result)


@pytest.mark.parametrize("stage", ["read", "write"])
@pytest.mark.parametrize("error", [httpx.ConnectError("Connection refused"), httpx.ReadTimeout("Read timed out")])
def test_transport_error_preserves_cause(note_server, stage, error):
    server, backend = note_server
    setattr(backend, f"{stage}_error", error)
    result = call(server, note_text="<p>new</p>")
    assert result.isError is True
    assert str(error) in text(result)
    assert "No item found" not in text(result)
    assert backend.record == ORIGINAL


def test_missing_note_is_mcp_error(note_server):
    server, backend = note_server
    backend.read_error = ResourceNotFoundError("HTTP 404")
    result = call(server, note_text="<p>new</p>")
    assert result.isError is True
    assert f"No item found with key: {KEY}" in text(result)
    assert backend.writes == 0


def test_non_note_is_mcp_error(note_server):
    server, backend = note_server
    backend.record["data"]["itemType"] = "book"
    result = call(server, note_text="<p>new</p>")
    assert result.isError is True
    assert "is not a note" in text(result)
    assert backend.writes == 0


def test_missing_text_is_mcp_error(note_server):
    server, backend = note_server
    result = call(server)
    assert result.isError is True
    assert "requires note_text" in text(result)
    assert backend.writes == 0


def test_unavailable_writer_is_mcp_error(note_server, monkeypatch):
    server, backend = note_server
    monkeypatch.setattr(annotations, "_get_note_write_client", lambda _: (None, "No writable backend"))
    result = call(server, note_text="<p>new</p>")
    assert result.isError is True
    assert "No writable backend" in text(result)
    assert backend.writes == 0


@pytest.mark.parametrize("response", [False, {"success": False}, httpx.Response(500, text="write failed")])
def test_rejected_write_response_is_mcp_error(note_server, response):
    server, backend = note_server
    backend.response = response
    result = call(server, note_text="<p>new</p>")
    assert result.isError is True
    assert "Failed to update note" in text(result)


@pytest.mark.parametrize(
    "body,append,expected",
    [
        ("<p>new</p>", False, "<p>new</p>"),
        ("<p>more</p>", True, "<p>old</p><p>more</p>"),
        ("", False, ""),
    ],
)
def test_success_preserves_update_semantics(note_server, body, append, expected):
    server, backend = note_server
    result = call(server, note_text=body, append=append)
    assert result.isError is False
    assert text(result) == f"Successfully updated note {KEY}"
    assert backend.record["data"]["note"] == expected
    assert backend.writes == 1


def test_failure_does_not_poison_session_or_retry_append(note_server):
    server, backend = note_server

    async def run():
        async with Client(server) as client:
            args = {"action": "update", "item_key": KEY, "note_text": "<p>more</p>", "append": True}
            backend.write_error = httpx.ReadTimeout("Write outcome unknown")
            failed = await client.call_tool_mcp("zotero_manage_note", args, timeout=5)
            assert backend.writes == 1
            backend.write_error = None
            recovered = await client.call_tool_mcp("zotero_manage_note", args, timeout=5)
            return failed, recovered

    failed, recovered = asyncio.run(run())
    assert failed.isError is True
    assert recovered.isError is False
    assert backend.record["data"]["note"] == "<p>old</p><p>more</p>"
    assert backend.writes == 2


@pytest.mark.parametrize("json_mode", [False, True])
def test_cli_update_failure_exits_nonzero(note_server, monkeypatch, capsys, json_mode):
    _, backend = note_server
    backend.write_error = httpx.ConnectError("Connection refused")
    monkeypatch.setattr(cli_standalone, "setup_zotero_environment", lambda: None)
    args = ["zotero-cli", "notes", "update", "--item-key", KEY, "--text", "new"]
    if json_mode:
        args.insert(1, "--json")
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.delenv("ZOTERO_CLI_DEBUG", raising=False)
    with pytest.raises(SystemExit) as exc:
        cli_standalone.main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    if json_mode:
        result = json.loads(captured.out)
        assert result["ok"] is False
        assert "Connection refused" in result["error"]["message"]
    else:
        assert captured.out == ""
        assert "Connection refused" in captured.err
