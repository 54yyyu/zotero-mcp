"""Recent-item failures must be errors in the MCP tools/call response.

Exercise the real tool through an in-memory MCP client/server session. Only
the Zotero backend is synthetic; a separate server avoids the application's
indexing lifespan and never needs a running Zotero instance or network access.
"""

import asyncio

import httpx
import pytest
from fastmcp import Client, FastMCP
from mcp.types import CallToolResult
from pyzotero.zotero_errors import ResourceNotFoundError

from zotero_mcp.tools import retrieval

COLLECTION_KEY = "ABCDEFGH"
CONNECTION_ERROR = "[WinError 10061] Connection refused"


class RecentBackend:
    def __init__(self):
        self.fail_at = None
        self.records = []
        self.collection_record = {"key": COLLECTION_KEY}
        self.collection_error = None

    def _check_connection(self, operation):
        if self.fail_at == operation:
            raise httpx.ConnectError(CONNECTION_ERROR)

    def client(self):
        self._check_connection("client")
        return self

    def items(self, start=0, limit=100, **kwargs):
        self._check_connection("items")
        return self.records[start : start + limit]

    def collection(self, key):
        self._check_connection("collection")
        if self.collection_error is not None:
            raise self.collection_error
        return self.collection_record

    def collection_items(self, key, start=0, limit=100, **kwargs):
        self._check_connection("collection_items")
        return self.records[start : start + limit]


@pytest.fixture
def recent_server(monkeypatch):
    backend = RecentBackend()
    monkeypatch.setattr(retrieval._client, "get_zotero_client", backend.client)
    server = FastMCP("Recent item protocol regression")
    # FastMCP 2's decorator returns a FunctionTool; 3 leaves the function intact.
    tool_function = getattr(retrieval.get_recent, "fn", retrieval.get_recent)
    server.tool(tool_function, name="zotero_get_recent")
    return server, backend


def _call(server, arguments):
    async def run():
        async with Client(server) as client:
            return await client.call_tool_mcp(
                "zotero_get_recent",
                arguments,
                timeout=5,
            )

    return asyncio.run(run())


def _text(result):
    assert isinstance(result, CallToolResult)
    return "\n".join(block.text for block in result.content if block.type == "text")


@pytest.mark.parametrize(
    "operation,arguments",
    [
        ("client", {}),
        ("items", {}),
        ("collection", {"collection_key": COLLECTION_KEY}),
        ("collection_items", {"collection_key": COLLECTION_KEY}),
    ],
)
def test_connection_failure_is_mcp_error(recent_server, operation, arguments):
    server, backend = recent_server
    backend.fail_at = operation

    result = _call(server, arguments)

    assert result.isError is True
    text = _text(result)
    assert "Error fetching recent items" in text
    assert CONNECTION_ERROR in text
    assert "Collection not found" not in text


@pytest.mark.parametrize(
    "method,arguments",
    [
        ("items", {"limit": 101}),
        ("collection_items", {"limit": 101, "collection_key": COLLECTION_KEY}),
    ],
)
def test_later_page_failure_is_mcp_error(recent_server, monkeypatch, method, arguments):
    server, backend = recent_server
    requested_starts = []

    def paged_items(*args, start=0, limit=100, **kwargs):
        requested_starts.append(start)
        if start >= 100:
            raise httpx.ConnectError(CONNECTION_ERROR)
        return [
            {
                "key": f"ITEM{i:04d}",
                "data": {"itemType": "book", "title": "Partial result"},
            }
            for i in range(100)
        ]

    monkeypatch.setattr(backend, method, paged_items)

    result = _call(server, arguments)

    assert requested_starts == [0, 100]
    assert result.isError is True
    text = _text(result)
    assert CONNECTION_ERROR in text
    assert "Partial result" not in text
    assert "Most Recently Added Items" not in text


def test_backend_timeout_is_mcp_error(recent_server, monkeypatch):
    server, backend = recent_server

    def timed_out_collection(key):
        raise httpx.ReadTimeout("Zotero request timed out")

    monkeypatch.setattr(backend, "collection", timed_out_collection)

    result = _call(server, {"collection_key": COLLECTION_KEY})

    assert result.isError is True
    text = _text(result)
    assert "Zotero request timed out" in text
    assert "Collection not found" not in text


@pytest.mark.parametrize("missing", ["404", "empty_response", "wrong_key"])
def test_missing_collection_is_mcp_error(recent_server, missing):
    server, backend = recent_server
    if missing == "404":
        backend.collection_error = ResourceNotFoundError("HTTP 404 Not Found")
    elif missing == "empty_response":
        backend.collection_record = None
    else:
        backend.collection_record = {"key": "OTHERKEY"}

    result = _call(server, {"collection_key": COLLECTION_KEY})

    assert result.isError is True
    text = _text(result)
    assert f"Collection not found: '{COLLECTION_KEY}'" in text
    assert "zotero_get_collections" in text


@pytest.mark.parametrize("collection_key", [None, COLLECTION_KEY])
def test_empty_results_remain_mcp_success(recent_server, collection_key):
    server, _ = recent_server
    arguments = {"collection_key": collection_key} if collection_key else {}

    result = _call(server, arguments)

    assert result.isError is False
    expected = (
        f"No items found in collection: {collection_key}"
        if collection_key
        else "No items found in your Zotero library."
    )
    assert _text(result) == expected


@pytest.mark.parametrize("collection_key", [None, COLLECTION_KEY])
def test_recent_items_remain_mcp_success(recent_server, collection_key):
    server, backend = recent_server
    backend.records = [
        {
            "key": "ITEM0001",
            "data": {
                "itemType": "journalArticle",
                "title": "Recent paper",
                "dateAdded": "2026-09-01T12:00:00Z",
            },
        }
    ]
    arguments = {"limit": 10}
    if collection_key:
        arguments["collection_key"] = collection_key

    result = _call(server, arguments)

    assert result.isError is False
    text = _text(result)
    scope = f" in Collection {collection_key}" if collection_key else ""
    assert text.startswith(f"# 1 Most Recently Added Items{scope}\n")
    assert "Recent paper" in text
    assert "ITEM0001" in text
    assert "2026-09-01T12:00:00Z" in text


def test_connection_recovers_in_same_mcp_session(recent_server):
    server, backend = recent_server

    async def run():
        async with Client(server) as client:
            backend.fail_at = "items"
            failed = await client.call_tool_mcp("zotero_get_recent", {}, timeout=5)
            backend.fail_at = None
            recovered = await client.call_tool_mcp("zotero_get_recent", {}, timeout=5)
            return failed, recovered

    failed, recovered = asyncio.run(run())

    assert failed.isError is True
    assert CONNECTION_ERROR in _text(failed)
    assert recovered.isError is False
    assert _text(recovered) == "No items found in your Zotero library."
