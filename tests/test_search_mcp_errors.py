"""Metadata search failures are not successful empty or partial MCP results."""

import asyncio
import json
import sys

import httpx
import pytest
from fastmcp import Client, FastMCP

from zotero_mcp import cli_standalone
from zotero_mcp.tools import search


class SearchBackend:
    def __init__(self):
        self.params = {}
        self.calls = []
        self.records = []
        self.fail_at = None
        self.error = httpx.ConnectError("Connection refused (synthetic backend)")

    def client(self):
        self.check("client")
        return self

    def check(self, operation):
        if self.fail_at == operation:
            raise self.error

    def add_parameters(self, **kwargs):
        self.params = kwargs

    def items(self):
        self.calls.append(self.params.copy())
        self.check("items")
        if len(self.calls) == 2:
            self.check("second_items")
        return self.records

    def get_collection(self, key):
        self.check("collection")
        return {"key": key}

    def search_items(self, *args, **kwargs):
        self.check("collection_items")
        return self.records


@pytest.fixture
def search_server(monkeypatch, tmp_path):
    backend = SearchBackend()
    monkeypatch.setattr(search._client, "get_zotero_client", backend.client)
    monkeypatch.setattr(search._client, "get_active_group_id", lambda: 0)
    monkeypatch.setattr(search._library, "get_library_backend", lambda: backend)
    monkeypatch.setattr(search._utils, "get_search_backend", lambda: "api")
    monkeypatch.setattr(search.Path, "home", lambda: tmp_path)

    def no_network(*args, **kwargs):
        raise AssertionError("These regression tests must not access the network")

    # Windows asyncio uses a loopback socket pair even for in-memory MCP.
    # Block HTTP transport instead of breaking that event-loop machinery.
    monkeypatch.setattr(httpx.Client, "send", no_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", no_network)
    server = FastMCP("Search error protocol regression")
    server.tool(getattr(search.search_items, "fn", search.search_items), name="zotero_search_items")
    return server, backend


def call(server, arguments):
    async def run():
        async with Client(server) as client:
            return await client.call_tool_mcp("zotero_search_items", arguments, timeout=5)

    return asyncio.run(run())


def text(result):
    return "\n".join(block.text for block in result.content if block.type == "text")


@pytest.mark.parametrize(
    "operation,extra",
    [
        ("client", {}),
        ("items", {}),
        ("collection", {"collection_key": "ABCDEFGH"}),
        ("collection_items", {"collection_key": "ABCDEFGH"}),
    ],
)
def test_backend_failure_is_mcp_error(search_server, operation, extra):
    server, backend = search_server
    backend.fail_at = operation
    result = call(server, {"query": "Synthetic", **extra})
    assert result.isError is True
    assert "Error searching Zotero" in text(result)
    assert "Connection refused" in text(result)
    assert "No items found" not in text(result)


@pytest.mark.parametrize("error", [httpx.ReadTimeout("synthetic timeout"), PermissionError("synthetic denied")])
def test_timeout_and_authorization_failure_stop_the_cascade(search_server, error):
    server, backend = search_server
    backend.fail_at = "items"
    backend.error = error
    result = call(server, {"query": "Müller 2024 methods"})
    assert result.isError is True
    assert str(error) in text(result)
    assert len(backend.calls) == 1


@pytest.mark.parametrize(
    "records", [[], [{"key": "ABCDEFGH", "data": {"itemType": "book", "title": "Partial result"}}]]
)
def test_later_variant_failure_is_not_empty_or_partial_success(search_server, records):
    server, backend = search_server
    backend.records = records
    backend.fail_at = "second_items"
    result = call(server, {"query": "Müller"})
    assert result.isError is True
    assert "No items found" not in text(result)
    assert "Partial result" not in text(result)
    assert len(backend.calls) == 2
    assert len({c["q"] for c in backend.calls}) == 2


@pytest.mark.parametrize("extra", [{}, {"collection_key": "ABCDEFGH"}])
def test_successful_empty_search_remains_success(search_server, extra):
    server, _ = search_server
    result = call(server, {"query": "Synthetic", **extra})
    assert result.isError is False
    assert "No items found matching query" in text(result)


def test_populated_normalization_search_remains_success(search_server):
    server, backend = search_server
    backend.records = [{"key": "ABCDEFGH", "data": {"itemType": "book", "title": "Synthetic match"}}]
    result = call(server, {"query": "Müller"})
    assert result.isError is False
    assert text(result).count("Synthetic match") == 1


def test_same_mcp_session_recovers_after_backend_recovers(search_server):
    server, backend = search_server

    async def run():
        async with Client(server) as client:
            backend.fail_at = "items"
            failed = await client.call_tool_mcp("zotero_search_items", {"query": "Synthetic"})
            backend.fail_at = None
            recovered = await client.call_tool_mcp("zotero_search_items", {"query": "Synthetic"})
        return failed, recovered

    failed, recovered = asyncio.run(run())
    assert failed.isError is True
    assert recovered.isError is False
    assert "No items found" in text(recovered)


@pytest.mark.parametrize("json_mode", [False, True], ids=["text", "json"])
def test_cli_search_failure_exits_one(search_server, monkeypatch, capsys, json_mode):
    _, backend = search_server
    backend.fail_at = "items"
    monkeypatch.setattr(cli_standalone, "setup_zotero_environment", lambda: None)
    monkeypatch.delenv("ZOTERO_CLI_DEBUG", raising=False)
    argv = ["zotero-cli", "search", "Synthetic"]
    if json_mode:
        argv.insert(1, "--json")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as raised:
        cli_standalone.main()
    assert raised.value.code == 1
    captured = capsys.readouterr()
    if json_mode:
        result = json.loads(captured.out)
        assert result["ok"] is False
        assert "Connection refused" in result["error"]["message"]
        assert "data" not in result
    else:
        assert captured.out == ""
        assert "Connection refused" in captured.err
