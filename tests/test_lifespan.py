"""Tests for server lifespan — verifies startup is non-blocking."""

import asyncio
import threading
from unittest.mock import patch

import pytest

from zotero_mcp._app import server_lifespan


@pytest.mark.asyncio
async def test_lifespan_yields_before_sync_update_completes():
    """The lifespan must yield (allow request handling) while the
    background semantic update is still running."""
    entered = threading.Event()
    proceed = threading.Event()

    def slow_update():
        entered.set()
        proceed.wait(timeout=5)

    with patch("zotero_mcp._app._sync_semantic_update", slow_update):
        async with server_lifespan(None) as ctx:
            assert ctx == {}
            # Yield to the event loop so the background task can start.
            await asyncio.sleep(0.1)
            assert entered.is_set(), \
                "_sync_semantic_update was never called in the background"
        proceed.set()


@pytest.mark.asyncio
async def test_lifespan_yields_when_update_raises():
    """Exceptions in the background update must not prevent the server
    from starting."""

    def exploding_update():
        raise RuntimeError("ChromaDB exploded")

    with patch("zotero_mcp._app._sync_semantic_update", exploding_update):
        async with server_lifespan(None) as ctx:
            assert ctx == {}
            await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_lifespan_yields_when_config_missing():
    """When no config exists, the background task completes instantly
    and the lifespan still yields normally."""

    def noop_update():
        pass

    with patch("zotero_mcp._app._sync_semantic_update", noop_update):
        async with server_lifespan(None) as ctx:
            assert ctx == {}
