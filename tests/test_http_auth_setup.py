"""Setup must preserve HTTP authentication and keep its token out of output."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from zotero_mcp import setup_helper

TOKEN = "existing-http-auth-token-do-not-print"
API_KEY = "replacement-zotero-api-key"


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def _write_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config), encoding="utf-8")


@pytest.mark.parametrize("local", [True, False])
def test_standalone_preserves_auth_while_updating_connection(isolated_home, local):
    path = isolated_home / ".config" / "zotero-mcp" / "config.json"
    semantic = {"embedding_model": "default"}
    _write_config(path, {
        "client_env": {
            "ZOTERO_MCP_AUTH_TOKEN": TOKEN,
            "ZOTERO_LOCAL": "false" if local else "true",
            "ZOTERO_API_KEY": "old-key",
            "ZOTERO_LIBRARY_ID": "old-library",
            "ZOTERO_LIBRARY_TYPE": "user",
        },
        "semantic_search": semantic,
        "zotero_db_path": "existing.sqlite",
    })

    result = setup_helper._write_standalone_config(
        local, API_KEY, "42", "group", {}, no_claude=True,
    )

    config = json.loads(result.read_text(encoding="utf-8"))
    expected_env = {
        "ZOTERO_LOCAL": "true" if local else "false",
        "ZOTERO_MCP_AUTH_TOKEN": TOKEN,
        "ZOTERO_NO_CLAUDE": "true",
    }
    if not local:
        expected_env.update({
            "ZOTERO_API_KEY": API_KEY,
            "ZOTERO_LIBRARY_ID": "42",
            "ZOTERO_LIBRARY_TYPE": "group",
        })
    assert config["client_env"] == expected_env
    assert config["semantic_search"] == semantic
    assert config["zotero_db_path"] == "existing.sqlite"


@pytest.mark.parametrize("local", [True, False])
def test_claude_preserves_auth_while_updating_connection(isolated_home, local, capsys):
    path = isolated_home / "claude" / "claude_desktop_config.json"
    other_server = {"command": "other-server"}
    _write_config(path, {"mcpServers": {
        "zotero": {
            "command": "old-zotero-command",
            "env": {
                "ZOTERO_MCP_AUTH_TOKEN": TOKEN,
                "ZOTERO_LOCAL": "false" if local else "true",
                "ZOTERO_API_KEY": "old-key",
                "ZOTERO_LIBRARY_ID": "old-library",
            },
        },
        "other": other_server,
    }})

    result = setup_helper.update_claude_config(
        path, "new-zotero-command", local, API_KEY, "42", "group",
        semantic_config={
            "embedding_model": "ollama",
            "embedding_config": {"model_name": "test-embedding-model"},
        },
    )

    config = json.loads(result.read_text(encoding="utf-8"))
    expected_env = {
        "ZOTERO_LOCAL": "true" if local else "false",
        "ZOTERO_MCP_AUTH_TOKEN": TOKEN,
        "ZOTERO_EMBEDDING_MODEL": "ollama",
        "OLLAMA_EMBEDDING_MODEL": "test-embedding-model",
    }
    if not local:
        expected_env.update({
            "ZOTERO_API_KEY": API_KEY,
            "ZOTERO_LIBRARY_ID": "42",
            "ZOTERO_LIBRARY_TYPE": "group",
        })
    assert config["mcpServers"]["zotero"] == {
        "command": "new-zotero-command", "env": expected_env,
    }
    assert config["mcpServers"]["other"] == other_server
    assert TOKEN not in capsys.readouterr().out


def test_setup_does_not_add_an_unconfigured_auth_token(isolated_home, monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_AUTH_TOKEN", "ambient-token")
    standalone = setup_helper._write_standalone_config(True, None, None, "user", {})
    claude = setup_helper.update_claude_config(
        isolated_home / "claude" / "config.json", "zotero-mcp", local=True,
    )

    standalone_config = json.loads(standalone.read_text(encoding="utf-8"))
    claude_config = json.loads(claude.read_text(encoding="utf-8"))
    assert "ZOTERO_MCP_AUTH_TOKEN" not in standalone_config["client_env"]
    assert "ZOTERO_MCP_AUTH_TOKEN" not in claude_config["mcpServers"]["zotero"]["env"]


@pytest.mark.parametrize("show_secrets", [False, True])
@pytest.mark.parametrize("token_value", [TOKEN, {"invalid-token": TOKEN}])
def test_setup_summary_hides_auth_unless_explicitly_requested(
    isolated_home, monkeypatch, capsys, show_secrets, token_value,
):
    path = isolated_home / ".config" / "zotero-mcp" / "config.json"
    _write_config(path, {"client_env": {"ZOTERO_MCP_AUTH_TOKEN": token_value}})
    monkeypatch.setattr(setup_helper, "find_executable", lambda: "test-zotero-mcp")
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("Unexpected setup prompt"))
    args = SimpleNamespace(
        no_local=True, no_claude=True, api_key=API_KEY,
        library_id="42", library_type="group", config_path=None,
        skip_semantic_search=True, semantic_config_only=False,
        show_secrets=show_secrets,
    )

    assert setup_helper.main(args) == 0

    output = capsys.readouterr().out
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["client_env"]["ZOTERO_MCP_AUTH_TOKEN"] == token_value
    summary = next(line for line in output.splitlines() if line.startswith("{\"ZOTERO_LOCAL\""))
    displayed = json.loads(summary)
    if show_secrets:
        assert displayed["ZOTERO_MCP_AUTH_TOKEN"] == token_value
        assert displayed["ZOTERO_API_KEY"] == API_KEY
    else:
        assert TOKEN not in output
        assert API_KEY not in output
        assert displayed["ZOTERO_MCP_AUTH_TOKEN"] == "********"
        assert "--show-secrets" in output
