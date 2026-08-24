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
    """Check for and run semantic search auto-update (called in a worker thread).

    Every early return below happens *before* ``zotero_mcp.semantic_search`` is
    imported. That module pulls in ChromaDB and numpy, which costs roughly a
    second even when warm, and on Windows the import — running here, in the
    lifespan's worker thread — wedged the process for the length of the first
    tool call (#485). ``update_policy`` answers "is an update due?" from the
    config file alone, with no third-party imports at all, so only a server
    that is actually about to index anything pays for ChromaDB.
    """
    from zotero_mcp.update_policy import should_update

    config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
    if not config_path.exists():
        return

    # Avoid initializing ChromaDB on every server startup when no semantic
    # auto-update is due. This also avoids racing a foreground
    # zotero_semantic_search call for the same persisted ChromaDB directory.
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        update_cfg = cfg.get("semantic_search", {}).get("update_config", {})
    except Exception:
        # An unreadable config cannot say an update is due, and guessing "yes"
        # here is what would drag the heavy import back in on every startup.
        return

    if not should_update(update_cfg):
        return

    from zotero_mcp.semantic_search import create_semantic_search

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

    async def _refresh_schema():
        # TTL-gated conditional GET; degrades to the vendored floor on failure.
        try:
            from zotero_mcp import schema
            if await asyncio.to_thread(schema.refresh) == "offline":
                sys.stderr.write(
                    "Warning: could not refresh the Zotero schema; using the "
                    "cached or vendored copy.\n"
                )
        except Exception as e:
            sys.stderr.write(f"Warning: Zotero schema refresh task failed: {e}\n")

    asyncio.create_task(_background_update())
    asyncio.create_task(_refresh_schema())

    yield {}

    sys.stderr.write("Shutting down Zotero MCP server...\n")


# Create an MCP server (fastmcp 2.14+ no longer accepts `dependencies`)
mcp = FastMCP("Zotero", lifespan=server_lifespan)
