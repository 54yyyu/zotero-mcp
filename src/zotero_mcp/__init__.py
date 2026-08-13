"""
Zotero MCP - Model Context Protocol server for Zotero

This module provides tools for AI assistants to interact with Zotero libraries.
"""

from typing import TYPE_CHECKING

from ._version import __version__ as __version__

if TYPE_CHECKING:  # pragma: no cover - type checkers only
    from .server import mcp as mcp

__all__ = ["__version__", "mcp"]


def __getattr__(name: str):
    """Import ``mcp`` lazily (PEP 562).

    ``from .server import mcp`` at module scope made *every* import of any
    submodule pay for the whole server: FastMCP, the MCP SDK, pydantic,
    pyzotero, bibtexparser and unidecode all get pulled in before the
    submodule's own code runs. Measured on 0.9.1, ``from
    zotero_mcp.schema import valid_fields`` cost ~1.45 s and ~1480 modules,
    of which ``schema`` itself was 8 microseconds -- it is a stdlib-only
    module whose whole point is offline field resolution.

    That matters for consumers that use this package as a library rather
    than as a server: schema lookups, DOI normalisation and the CLI's
    lighter subcommands should not require the server's dependency tree to
    be importable, let alone imported.

    ``from zotero_mcp import mcp`` keeps working unchanged, and still
    degrades to AttributeError rather than raising when the server's
    optional dependencies are missing -- matching the previous
    ``try/except ImportError`` behaviour.
    """
    if name == "mcp":
        try:
            from .server import mcp
        except ImportError as exc:  # optional server deps absent
            raise AttributeError(
                "zotero_mcp.mcp is unavailable: the MCP server dependencies "
                f"are not installed ({exc})."
            ) from exc
        return mcp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


# These modules are not imported by default but are available
# pdfannots_helper and pdfannots_downloader
