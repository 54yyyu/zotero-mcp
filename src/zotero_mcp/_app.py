"""FastMCP application instance and server lifecycle."""

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import FastMCP

from zotero_mcp.utils import is_local_mode

# Configure logging from environment variable
# Set ZOTERO_MCP_LOG_LEVEL=DEBUG in Claude Desktop config to enable debug logs
_log_level = os.environ.get("ZOTERO_MCP_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.WARNING),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)


def _sync_semantic_update() -> None:
    """Check for and run semantic search auto-update (called in a worker thread)."""
    from zotero_mcp.semantic_search import create_semantic_search

    config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
    if not config_path.exists():
        return

    # Avoid initializing ChromaDB on every server startup when semantic
    # auto-update is disabled. This also avoids racing a foreground
    # zotero_semantic_search call for the same persisted ChromaDB directory.
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        update_cfg = cfg.get("semantic_search", {}).get("update_config", {})
        if not update_cfg.get("auto_update", False):
            return
    except Exception:
        pass

    search = create_semantic_search(str(config_path))
    if not search.should_update_database():
        return

    sys.stderr.write("Auto-updating semantic search database...\n")
    stats = search.update_database(extract_fulltext=is_local_mode())
    sys.stderr.write(
        f"Database update completed: {stats.get('processed_items', 0)} items processed\n"
    )


@asynccontextmanager
async def server_lifespan(server: FastMCP):
    """Manage server startup and shutdown lifecycle.

    Semantic search initialization (ChromaDB + embedding model) is
    offloaded to a worker thread so it cannot block the event loop.
    The previous synchronous call prevented FastMCP from responding
    to the MCP ``initialize`` request within the 60-second client
    timeout.

    On shutdown the worker thread is left to finish on its own —
    ``asyncio.to_thread`` threads cannot be interrupted, and
    ChromaDB (SQLite WAL) is crash-safe, so an unfinished update
    simply resumes on the next startup.
    """
    sys.stderr.write("Starting Zotero MCP server...\n")

    async def _background_update():
        try:
            await asyncio.to_thread(_sync_semantic_update)
        except Exception as e:
            sys.stderr.write(f"Warning: Could not check semantic search auto-update: {e}\n")

    asyncio.create_task(_background_update())

    yield {}

    sys.stderr.write("Shutting down Zotero MCP server...\n")


# Create an MCP server (fastmcp 2.14+ no longer accepts `dependencies`)
mcp = FastMCP("Zotero", lifespan=server_lifespan)
