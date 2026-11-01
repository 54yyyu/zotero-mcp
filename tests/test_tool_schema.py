"""The advertised tool schemas are compact, and validation is not narrowed.

``zotero_mcp._tool_schema`` rewrites what a tool's input schema *shows*
(canonical shapes, no null branches, no docstring-derived parameter
descriptions). The functions still accept every lenient form they always
did; these tests hold both halves of that.
"""

import asyncio

import pytest

fastmcp = pytest.importorskip("fastmcp")

from zotero_mcp._app import _ZoteroMCP  # noqa: E402
from zotero_mcp._context import Context  # noqa: E402
from zotero_mcp._tool_schema import compact  # noqa: E402


def _probe_server():
    server = _ZoteroMCP("probe")

    @server.tool(name="probe", description="Probe tool.")
    def probe(
        limit: int | str | None = 10,
        tags: list[str] | str | None = None,
        rect: list[float] | str | None = None,
        key: str | None = None,
        *,
        ctx: Context,
    ) -> str:
        """Probe.

        Args:
            limit: a docstring description that must not be advertised.
        """
        return repr((limit, tags, rect, key))

    return server, probe


def _run(server, args):
    async def go():
        async with fastmcp.Client(server) as client:
            tools = await client.list_tools()
            result = await client.call_tool("probe", args)
            return tools[0], result.content[0].text

    return asyncio.run(go())


def _input_schema(tool):
    return tool.input_schema if hasattr(tool, "input_schema") else tool.inputSchema


class TestAdvertised:
    def test_canonical_shapes_only(self):
        server, _ = _probe_server()
        tool, _ = _run(server, {})
        props = _input_schema(tool)["properties"]
        assert props["limit"] == {"type": "integer", "default": 10}
        assert props["tags"]["type"] == "array"
        assert props["tags"]["items"] == {"type": "string"}
        assert props["key"]["type"] == "string"
        assert "anyOf" not in props["key"]
        # A string here is not a fallback encoding, so both shapes stay.
        assert {b["type"] for b in props["rect"]["anyOf"]} == {"array", "string"}

    def test_docstring_args_are_not_advertised(self):
        server, fn = _probe_server()
        tool, _ = _run(server, {})
        assert all("description" not in p for p in _input_schema(tool)["properties"].values())
        # The module-level function keeps its docstring for developers.
        assert "docstring description" in fn.__doc__

    def test_lenient_inputs_still_validate(self):
        server, _ = _probe_server()
        _, text = _run(server, {"limit": "7", "tags": "solo", "rect": "[0, 0, 1, 1]", "key": None})
        assert text == repr(("7", "solo", "[0, 0, 1, 1]", None))


class TestCompact:
    def test_required_nullable_keeps_null(self):
        # No default: null is a meaningful value (e.g. "clear the parent").
        schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        assert compact("parent_key", schema, default=object()) == schema

    def test_open_object_marker_is_dropped(self):
        schema = {"anyOf": [{"type": "object", "additionalProperties": True}, {"type": "string"}, {"type": "null"}]}
        assert compact("fields", schema, default=None) == {
            "anyOf": [{"type": "object"}, {"type": "string"}]
        }


def test_real_registry_advertises_no_parameter_descriptions():
    from zotero_mcp.server import mcp

    async def go():
        return await mcp.list_tools()

    leaked = [
        (t.name, name)
        for t in asyncio.run(go())
        for name, prop in (t.parameters or {}).get("properties", {}).items()
        if "description" in prop
    ]
    assert not leaked, f"docstring Args leaked into parameter schemas: {leaked}"
