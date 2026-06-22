"""Tool modules — importing this package registers all tools with the MCP app."""

from zotero_mcp.tools import (  # noqa: F401
    annotations,
    connectors,
    discovery,
    read_pdf,
    retrieval,
    search,
    synthesis,
    write,
)

# Optional: Scite enrichment (requires ``pip install zotero-mcp-server[scite]``)
try:
    from zotero_mcp.tools import scite as scite  # noqa: F401
except ImportError:
    pass

# Register MCP prompts (research workflows) and resources (library context).
# Importing these is a side effect that binds their @mcp.prompt / @mcp.resource
# handlers, exactly like the tool modules above.
from zotero_mcp import prompts as prompts  # noqa: F401,E402
from zotero_mcp import resources as resources  # noqa: F401,E402
