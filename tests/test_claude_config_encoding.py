"""Claude Desktop's config must be read as UTF-8, the encoding Claude Desktop writes.

``update_claude_config()`` and ``load_claude_desktop_env_vars()`` opened
``claude_desktop_config.json`` without an encoding, so Python on Windows (before
3.15) decoded it with the locale code page. If another server's entry held a
non-ASCII character, ``zotero-mcp setup`` either crashed (cp950, cp1252 on
some bytes) or decoded it wrongly and wrote the garbled text back (``José``
became ``JosÃ©`` under cp1252, ``Jos矇`` under cp950), and the CLI silently
lost the env vars it reads from the zotero entry. These tests simulate those
locales on any OS.
"""

import builtins
import json
from pathlib import Path

import pytest

from zotero_mcp import cli, setup_helper

OTHER_SERVER_ARG = "C:/Users/José/王小明的筆記"


def _locale_open(code_page):
    real_open = builtins.open

    def open_(file, mode="r", buffering=-1, encoding=None, *args, **kwargs):
        if "b" not in mode and encoding is None:
            encoding = code_page
        return real_open(file, mode, buffering, encoding, *args, **kwargs)

    return open_


@pytest.fixture(params=["cp950", "cp1252"])
def windows_locale(request, monkeypatch):
    """Make encoding-less text ``open()`` calls behave like a Windows machine."""
    open_ = _locale_open(request.param)
    monkeypatch.setattr(setup_helper, "open", open_, raising=False)
    monkeypatch.setattr(cli, "open", open_, raising=False)
    return request.param


def _write_claude_config(path: Path, servers: dict) -> Path:
    # Claude Desktop (Electron) writes JSON.stringify output as UTF-8, unescaped.
    path.write_text(json.dumps({"mcpServers": servers}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_setup_keeps_other_servers_non_ascii_entries(windows_locale, tmp_path):
    config_path = _write_claude_config(
        tmp_path / "claude_desktop_config.json",
        {"notes": {"command": "notes-mcp", "args": [OTHER_SERVER_ARG]}},
    )

    assert setup_helper.update_claude_config(config_path, "zotero-mcp") == config_path

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["mcpServers"]["notes"]["args"] == [OTHER_SERVER_ARG]
    assert config["mcpServers"]["zotero"]["command"] == "zotero-mcp"


def test_cli_reads_zotero_env_from_a_config_with_non_ascii_entries(windows_locale, tmp_path, monkeypatch):
    config_path = _write_claude_config(
        tmp_path / "claude_desktop_config.json",
        {
            "notes": {"command": "notes-mcp", "args": [OTHER_SERVER_ARG]},
            "zotero": {"command": "zotero-mcp", "env": {"ZOTERO_LOCAL": "false", "ZOTERO_LIBRARY_ID": "12345"}},
        },
    )
    monkeypatch.delenv("ZOTERO_NO_CLAUDE", raising=False)
    monkeypatch.setattr(setup_helper, "find_existing_claude_configs", lambda verbose=False: [config_path])

    assert cli.load_claude_desktop_env_vars() == {"ZOTERO_LOCAL": "false", "ZOTERO_LIBRARY_ID": "12345"}
