"""Tool descriptions are shipped to every MCP client, and models copy the
tool names they mention. A description that names a tool that no longer
exists (renamed in 0.9.0) sends the model to a dead end. This test reads
the descriptions the client actually receives and checks every
``zotero_*`` / ``scite_*`` token against the names that can be registered.
"""

import asyncio
import re

import pytest

import zotero_mcp.server  # noqa: F401  (registers tools, applies toolsets)
from zotero_mcp._app import mcp
from zotero_mcp.toolsets import optional_tool_names

TOOL_NAME_RE = re.compile(r"\b(zotero_[a-z_]+|scite_[a-z_]+)\b")

# Names that can appear in descriptions without being tools.
ALLOWED_NON_TOOL_NAMES = {
    "zotero_mcp",  # the package
}


@pytest.fixture(scope="module")
def shipped_tools():
    return asyncio.run(mcp.list_tools())


def test_every_tool_name_in_a_description_is_registrable(shipped_tools):
    registered_now = {tool.name for tool in shipped_tools}
    registrable = registered_now | optional_tool_names()
    assert registered_now, "no tools registered; import of zotero_mcp.server failed to register"

    offenders = []
    for tool in shipped_tools:
        for name in TOOL_NAME_RE.findall(tool.description or ""):
            if name in registrable or name in ALLOWED_NON_TOOL_NAMES:
                continue
            offenders.append((tool.name, name))

    assert offenders == [], "descriptions name tools that do not exist: " + ", ".join(
        f"{tool} -> {name}" for tool, name in offenders
    )
