"""Tests for the local-mode startup connection check.

When ZOTERO_LOCAL=true and Zotero is not reachable, server_lifespan() should
write an actionable warning to stderr. When Zotero is reachable (or local mode
is off), no warning should appear.
"""

import asyncio
from unittest.mock import patch

import pytest

from zotero_mcp._app import server_lifespan


async def _run_lifespan():
    """Drive server_lifespan through its startup phase and immediately exit."""
    async with server_lifespan(None):
        # Give background tasks (including asyncio.to_thread calls) time to finish.
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_warning_when_local_and_zotero_unreachable(monkeypatch, capsys):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    with patch("zotero_mcp.client.is_local_zotero_available", return_value=False):
        await _run_lifespan()

    err = capsys.readouterr().err
    assert "localhost:23119" in err
    assert "Allow other applications" in err


@pytest.mark.asyncio
async def test_no_warning_when_local_and_zotero_reachable(monkeypatch, capsys):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    with patch("zotero_mcp.client.is_local_zotero_available", return_value=True):
        await _run_lifespan()

    err = capsys.readouterr().err
    assert "localhost:23119" not in err


@pytest.mark.asyncio
async def test_no_warning_in_web_mode(monkeypatch, capsys):
    monkeypatch.delenv("ZOTERO_LOCAL", raising=False)
    with patch("zotero_mcp.client.is_local_zotero_available", return_value=False):
        await _run_lifespan()

    err = capsys.readouterr().err
    assert "localhost:23119" not in err
