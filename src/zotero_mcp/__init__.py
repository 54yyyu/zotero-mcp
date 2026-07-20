"""
Zotero MCP - Model Context Protocol server for Zotero

This module provides tools for AI assistants to interact with Zotero libraries.
"""

from ._version import __version__ as __version__

__all__ = ["__version__", "mcp"]


def __getattr__(name: str):
    if name == "mcp":
        from .server import mcp
        return mcp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
