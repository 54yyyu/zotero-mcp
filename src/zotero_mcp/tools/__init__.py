"""Tool modules — importing this package registers all tools with the MCP app."""

from zotero_mcp.tools import (  # noqa: F401
    annotations,
    connectors,
    retrieval,
    search,
    write,
)

# Optional: Scite enrichment (requires ``pip install zotero-mcp-server[scite]``)
try:
    from zotero_mcp.tools import scite as scite  # noqa: F401
except ImportError:
    pass

# Optional: EZProxy paper fetching (requires ``pip install zotero-mcp-server[ezproxy]``)
try:
    from zotero_mcp.tools import proxy as proxy  # noqa: F401
except ImportError:
    pass
