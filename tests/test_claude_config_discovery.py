"""Regression tests for issue #392: Claude Desktop config path discovery.

Some Claude Desktop builds keep ``claude_desktop_config.json`` outside the
classic location (the reporter's Windows 11 build uses
``%LOCALAPPDATA%\\Claude-3p\\``), so ``zotero-mcp setup`` wrote to a file the
running app never read and reported success anyway.

These tests monkeypatch ``sys.platform``, the relevant environment variables
and ``Path.home()`` so the real user config is never touched.
"""

import json
from pathlib import Path

import pytest

from zotero_mcp import setup_helper

CONFIG_NAME = "claude_desktop_config.json"


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Point Path.home() at a throwaway directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def _write_config(path: Path, marker: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": {}, "_marker": marker}))
    return path


def _windows_env(monkeypatch, tmp_path):
    monkeypatch.setattr(setup_helper.sys, "platform", "win32")
    roaming = tmp_path / "AppData" / "Roaming"
    local = tmp_path / "AppData" / "Local"
    monkeypatch.setenv("APPDATA", str(roaming))
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    return roaming, local


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------

class TestCandidates:
    def test_windows_candidates_include_claude_3p_in_localappdata(self, monkeypatch, tmp_path):
        roaming, local = _windows_env(monkeypatch, tmp_path)

        candidates = setup_helper.claude_config_candidates()

        assert roaming / "Claude" / CONFIG_NAME in candidates
        assert roaming / "Claude Desktop" / CONFIG_NAME in candidates
        assert local / "Claude-3p" / CONFIG_NAME in candidates

    def test_macos_candidates_keep_classic_paths_and_add_claude_3p(self, monkeypatch, fake_home):
        monkeypatch.setattr(setup_helper.sys, "platform", "darwin")

        candidates = setup_helper.claude_config_candidates()
        app_support = fake_home / "Library" / "Application Support"

        assert candidates[0] == app_support / "Claude" / CONFIG_NAME
        assert candidates[1] == app_support / "Claude Desktop" / CONFIG_NAME
        assert app_support / "Claude-3p" / CONFIG_NAME in candidates

    def test_linux_candidates_respect_xdg_config_home(self, monkeypatch, fake_home, tmp_path):
        monkeypatch.setattr(setup_helper.sys, "platform", "linux")
        xdg = tmp_path / "xdg"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

        candidates = setup_helper.claude_config_candidates()

        assert candidates[0] == xdg / "Claude" / CONFIG_NAME
        assert candidates[1] == xdg / "Claude Desktop" / CONFIG_NAME
        assert xdg / "Claude-3p" / CONFIG_NAME in candidates

    def test_candidates_are_unique(self, monkeypatch, tmp_path):
        """APPDATA == LOCALAPPDATA must not yield duplicates."""
        monkeypatch.setattr(setup_helper.sys, "platform", "win32")
        monkeypatch.setenv("APPDATA", str(tmp_path))
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

        candidates = setup_helper.claude_config_candidates()

        assert len(candidates) == len(set(str(p) for p in candidates))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_finds_claude_3p_config_when_only_that_exists(self, monkeypatch, tmp_path):
        """The exact issue #392 layout: only the Claude-3p config is present."""
        _roaming, local = _windows_env(monkeypatch, tmp_path)
        expected = _write_config(local / "Claude-3p" / CONFIG_NAME, "3p")

        assert setup_helper.find_all_claude_configs() == [expected]
        assert setup_helper.find_claude_config() == expected

    def test_returns_every_existing_config(self, monkeypatch, tmp_path):
        roaming, local = _windows_env(monkeypatch, tmp_path)
        classic = _write_config(roaming / "Claude" / CONFIG_NAME, "classic")
        three_p = _write_config(local / "Claude-3p" / CONFIG_NAME, "3p")

        found = setup_helper.find_all_claude_configs()

        assert set(found) == {classic, three_p}

    def test_falls_back_to_historical_default_when_none_exists(self, monkeypatch, tmp_path):
        roaming, _local = _windows_env(monkeypatch, tmp_path)

        found = setup_helper.find_all_claude_configs()

        assert found == [roaming / "Claude Desktop" / CONFIG_NAME]
        assert setup_helper.find_claude_config() == roaming / "Claude Desktop" / CONFIG_NAME

    def test_macos_classic_path_still_wins(self, monkeypatch, fake_home):
        """Existing macOS behaviour is unchanged."""
        monkeypatch.setattr(setup_helper.sys, "platform", "darwin")
        app_support = fake_home / "Library" / "Application Support"
        classic = _write_config(app_support / "Claude" / CONFIG_NAME, "classic")

        assert setup_helper.find_claude_config() == classic

    def test_find_existing_returns_empty_when_nothing_installed(self, monkeypatch, tmp_path):
        _windows_env(monkeypatch, tmp_path)

        assert setup_helper.find_existing_claude_configs() == []


# ---------------------------------------------------------------------------
# setup writes to every detected config and prints the paths
# ---------------------------------------------------------------------------

class _Args:
    """Stand-in for the argparse namespace main() accepts."""

    no_local = False
    no_claude = False
    api_key = None
    library_id = None
    library_type = "user"
    config_path = None
    skip_semantic_search = True
    semantic_config_only = False
    show_secrets = False


class TestSetupWritesAllConfigs:
    def test_setup_updates_every_detected_config_and_prints_paths(
        self, monkeypatch, tmp_path, capsys
    ):
        _roaming, local = _windows_env(monkeypatch, tmp_path)
        classic = _write_config(_roaming / "Claude" / CONFIG_NAME, "classic")
        three_p = _write_config(local / "Claude-3p" / CONFIG_NAME, "3p")

        monkeypatch.setattr(setup_helper, "find_executable", lambda: "/fake/bin/zotero-mcp")
        monkeypatch.setattr(setup_helper, "load_semantic_search_config", lambda path: None)
        monkeypatch.setattr(setup_helper, "load_top_level_db_path", lambda path: None)

        assert setup_helper.main(_Args()) == 0

        for path in (classic, three_p):
            data = json.loads(path.read_text())
            assert data["mcpServers"]["zotero"]["command"] == "/fake/bin/zotero-mcp"

        out = capsys.readouterr().out
        assert str(classic.resolve()) in out
        assert str(three_p.resolve()) in out

    def test_setup_prints_path_on_fresh_install(self, monkeypatch, tmp_path, capsys):
        roaming, _local = _windows_env(monkeypatch, tmp_path)

        monkeypatch.setattr(setup_helper, "find_executable", lambda: "/fake/bin/zotero-mcp")
        monkeypatch.setattr(setup_helper, "load_semantic_search_config", lambda path: None)
        monkeypatch.setattr(setup_helper, "load_top_level_db_path", lambda path: None)

        assert setup_helper.main(_Args()) == 0

        default_path = roaming / "Claude Desktop" / CONFIG_NAME
        assert default_path.exists()
        assert str(default_path.resolve()) in capsys.readouterr().out

    def test_explicit_config_path_is_respected(self, monkeypatch, tmp_path, capsys):
        _roaming, local = _windows_env(monkeypatch, tmp_path)
        _write_config(local / "Claude-3p" / CONFIG_NAME, "3p")

        explicit = tmp_path / "custom" / CONFIG_NAME
        args = _Args()
        args.config_path = str(explicit)

        monkeypatch.setattr(setup_helper, "find_executable", lambda: "/fake/bin/zotero-mcp")
        monkeypatch.setattr(setup_helper, "load_semantic_search_config", lambda path: None)
        monkeypatch.setattr(setup_helper, "load_top_level_db_path", lambda path: None)

        assert setup_helper.main(args) == 0

        assert explicit.exists()
        # The auto-detected config must be left alone when a path is given.
        assert "zotero" not in json.loads(
            (local / "Claude-3p" / CONFIG_NAME).read_text()
        ).get("mcpServers", {})


# ---------------------------------------------------------------------------
# Reading env vars back out
# ---------------------------------------------------------------------------

class TestLoadEnvVars:
    def test_env_vars_read_from_the_config_that_has_zotero(self, monkeypatch, tmp_path):
        from zotero_mcp import cli

        _roaming, local = _windows_env(monkeypatch, tmp_path)
        monkeypatch.delenv("ZOTERO_NO_CLAUDE", raising=False)
        # Classic config exists but has no zotero server; the 3p one does.
        _write_config(_roaming / "Claude" / CONFIG_NAME, "classic")
        three_p = local / "Claude-3p" / CONFIG_NAME
        three_p.parent.mkdir(parents=True, exist_ok=True)
        three_p.write_text(json.dumps(
            {"mcpServers": {"zotero": {"env": {"ZOTERO_LOCAL": "true"}}}}
        ))

        assert cli.load_claude_desktop_env_vars() == {"ZOTERO_LOCAL": "true"}
