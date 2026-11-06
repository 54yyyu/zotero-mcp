"""Running the tools without the MCP server (``zotero-cli``).

Importing any ``fastmcp`` module loads FastMCP and the MCP SDK, about 0.45 s
before a command does anything, and the tool modules need them only for the
server: the ``@mcp.tool`` registry, the ``Context`` annotation FastMCP injects
by, and ``ToolError``. ``zotero-cli`` calls :func:`enable` before it imports a
tool module, and those modules then get the stand-ins below instead.

The switch is per process and only takes effect before the first tool module
is imported; once one is loaded with the real FastMCP (the server, the test
suite) it changes nothing, so the two can never be mixed.
"""

import sys

ENABLED = False

#: Modules that bind the server or its stand-ins when they are imported.
_BINDING_MODULES = ("zotero_mcp._app", "zotero_mcp._context", "zotero_mcp.tools")


def enable() -> bool:
    """Run this process's tools without FastMCP, if none is loaded yet."""
    global ENABLED
    if not any(name in sys.modules for name in _BINDING_MODULES):
        ENABLED = True
    return ENABLED


class ToolError(Exception):
    """Stands in for ``fastmcp.exceptions.ToolError``."""


class Context:
    """Stands in for ``fastmcp.Context`` as the tools' ``ctx`` annotation."""

    def info(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass


class App:
    """Stands in for the FastMCP app: registering a tool, prompt or
    resource leaves the function as it is, and nothing else is served."""

    def _register(self, name_or_fn=None, *_args, **_kwargs):
        if callable(name_or_fn):
            return name_or_fn
        return lambda fn: fn

    tool = prompt = resource = _register

    def __getattr__(self, name):
        raise RuntimeError(
            f"mcp.{name}: the MCP server is not loaded in this process "
            "(zotero-cli runs the tools without it)."
        )
