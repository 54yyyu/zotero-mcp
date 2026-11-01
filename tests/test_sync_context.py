"""Tools are sync; FastMCP's context logging is async. The message must
still reach the client instead of dying as an un-awaited coroutine."""

import asyncio

import pytest

fastmcp = pytest.importorskip("fastmcp")

from zotero_mcp._app import _ZoteroMCP  # noqa: E402
from zotero_mcp._context import Context  # noqa: E402


def test_a_sync_tools_log_message_reaches_the_client():
    server = _ZoteroMCP("probe")

    @server.tool(name="probe")
    def probe(ctx: Context) -> str:
        ctx.info("halfway there")
        return "done"

    received = []

    async def on_log(message):
        received.append(message.data)

    async def run():
        async with fastmcp.Client(server, log_handler=on_log) as client:
            return await client.call_tool("probe", {})

    result = asyncio.run(run())

    assert result.content[0].text == "done"
    assert any("halfway there" in str(entry) for entry in received)


def test_a_stand_in_context_is_passed_through(dummy_ctx):
    server = _ZoteroMCP("probe")
    seen = []

    @server.tool(name="probe")
    def probe(ctx: Context) -> str:
        seen.append(ctx)
        return "done"

    assert probe(ctx=dummy_ctx) == "done"
    assert seen == [dummy_ctx]


def test_string_tools_answer_as_text_without_an_output_schema():
    """A markdown-returning tool advertises no {"result": string} wrapper
    schema (~35 tokens per tool, every request) and its answer still arrives
    as plain text content, sent once rather than again as structured content."""
    server = _ZoteroMCP("probe")

    @server.tool(name="probe")
    def probe(ctx: Context) -> str:
        return "# Results"

    async def run():
        async with fastmcp.Client(server) as client:
            tools = await client.list_tools()
            return tools, await client.call_tool("probe", {})

    tools, result = asyncio.run(run())

    # Renamed from outputSchema in MCP SDK v2; read whichever this one has.
    tool = tools[0]
    schema = tool.output_schema if hasattr(tool, "output_schema") else tool.outputSchema
    assert schema is None
    assert [c.type for c in result.content] == ["text"]
    assert result.content[0].text == "# Results"
    assert result.structured_content is None
