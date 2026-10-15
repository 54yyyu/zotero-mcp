"""FastMCP application instance and server lifecycle."""

import asyncio
import json
import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import FastMCP

from zotero_mcp._context import sync_context
from zotero_mcp._version import __version__
from zotero_mcp.utils import is_local_mode

# Configure logging from environment variable
# Set ZOTERO_MCP_LOG_LEVEL=DEBUG in Claude Desktop config to enable debug logs
_log_level = os.environ.get("ZOTERO_MCP_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.WARNING),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)


def _semantic_update_command() -> list[str] | None:
    """The ``update-db`` command line if a semantic auto-update is due, else None.

    Decided from the config file alone: ``config_light`` imports nothing
    third-party, so a server with no update due never loads ChromaDB (#485).
    The update itself runs in a child process (``_spawn_semantic_update``).
    """
    from zotero_mcp.config_light import should_update

    config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            cfg = json.load(f)
        update_cfg = cfg.get("semantic_search", {}).get("update_config", {})
    except Exception:
        # An unreadable config cannot say an update is due, and guessing "yes"
        # here is what would drag the heavy work back in on every startup.
        return None

    if not should_update(update_cfg):
        return None

    cmd = [sys.executable, "-m", "zotero_mcp.cli", "update-db"]
    if is_local_mode():
        cmd.append("--fulltext")
    return cmd


def _spawn_semantic_update(cmd: list[str]) -> subprocess.Popen:
    """Start the update in its own process, output piped back to us.

    Out of process because in a worker thread it shares this interpreter: a
    native call that holds the GIL (model and index loading, ChromaDB) or, on
    Windows, a DLL load stalls the event loop too, so the server could not
    answer ``initialize`` until the update finished and the client gave up
    (#706, #485).

    stdin is DEVNULL and stdout/stderr are a pipe to us, never the client's
    pipes: a child that outlived the server would otherwise hold them open
    and the client would never see the connection close (#431).
    """
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        # A console child of a windowless server would open a console window.
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _relay_update_output(proc: subprocess.Popen) -> None:
    """Copy the child's output to our stderr (the client's log) until it exits."""
    for line in proc.stdout:
        sys.stderr.write(line)
    returncode = proc.wait()
    if returncode != 0:
        sys.stderr.write(f"Warning: semantic search auto-update exited with code {returncode}\n")


@asynccontextmanager
async def server_lifespan(server: FastMCP):
    """Manage server startup and shutdown lifecycle.

    The semantic auto-update runs as a child ``zotero-mcp update-db`` process
    (see ``_spawn_semantic_update``), with only the relay of its output on a
    worker thread, so nothing it does can hold up the MCP handshake (#706).
    The update lock in ``update_database`` still keeps it from racing a
    manual ``update-db`` or another server.

    On shutdown an unfinished update is terminated: ChromaDB (SQLite WAL)
    is crash-safe, so it simply resumes on the next startup.
    """
    sys.stderr.write("Starting Zotero MCP server...\n")
    update_proc: subprocess.Popen | None = None

    async def _background_update():
        nonlocal update_proc
        try:
            cmd = _semantic_update_command()
            if cmd is None:
                return
            sys.stderr.write("Auto-updating semantic search database...\n")
            update_proc = _spawn_semantic_update(cmd)
            await asyncio.to_thread(_relay_update_output, update_proc)
        except Exception as e:
            sys.stderr.write(f"Warning: Could not run semantic search auto-update: {e}\n")

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
    if update_proc is not None and update_proc.poll() is None:
        update_proc.terminate()


class _ZoteroMCP(FastMCP):
    """FastMCP whose tools get a context they can log to synchronously."""

    def tool(self, name_or_fn=None, **kwargs):
        if callable(name_or_fn):
            return super().tool(sync_context(name_or_fn), **kwargs)
        register = super().tool(name_or_fn, **kwargs)
        return lambda fn: register(sync_context(fn))


# Create an MCP server (fastmcp 2.14+ no longer accepts `dependencies`)
mcp = _ZoteroMCP("Zotero", version=__version__, lifespan=server_lifespan)
