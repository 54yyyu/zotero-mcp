"""
Zotero MCP - Model Context Protocol server for Zotero

This module provides tools for AI assistants to interact with Zotero libraries.
"""

from ._version import __version__ as __version__

try:
    from .server import mcp  # noqa: F401
except ImportError:
    pass

# These modules are not imported by default but are available
# pdfannots_helper and pdfannots_downloader
