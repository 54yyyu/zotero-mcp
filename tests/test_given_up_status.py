"""Status lists the items given up on after failed embedding runs (#769).

The MCP tool, ``zotero-mcp db-status`` and ``zotero-cli db status`` each show
how many, a few keys, and that a change in Zotero brings them back. HOME and
Path.home point at tmp_path, and the config file is a temporary one.
"""

import json
import sys
from pathlib import Path

import pytest

from zotero_mcp import index_staleness

KEYS = [f"KEY{n:05d}" for n in range(12)]


@pytest.fixture()
def config(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("ZOTERO_EMBEDDING_MODEL", raising=False)
    monkeypatch.setattr(
        index_staleness,
        "check_active_library",
        lambda *_a, **_k: index_staleness.Staleness("unknown", detail="not checked in this test"),
    )
    path = home / ".config" / "zotero-mcp" / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"semantic_search": {
        "persist_directory": str(home / "chroma"),
        "update_config": {"retry_item_keys_given_up": KEYS},
    }}))
    return path


def _assert_listed(out):
    assert "given up on" in out.lower()
    assert "12 (KEY00000, KEY00001, KEY00002 and 9 more)" in out
    assert "KEY00011" not in out
    assert "tried again when it changes in Zotero" in out


def test_status_tool_lists_items_given_up_on(config):
    pytest.importorskip("chromadb")
    from zotero_mcp.tools import search as search_tools

    class _Ctx:
        def info(self, *_):
            pass

        def error(self, *_):
            pass

    tool = search_tools.get_search_database_status
    fn = tool.fn if hasattr(tool, "fn") else tool

    _assert_listed(fn(ctx=_Ctx()))


@pytest.mark.parametrize("which", ["db-status", "zotero-cli"])
def test_cli_status_lists_items_given_up_on(config, monkeypatch, capsys, which):
    import zotero_mcp.cli as cli
    import zotero_mcp.cli_standalone as standalone
    import zotero_mcp.semantic_search as semantic_search

    class _Search:
        def get_database_status(self):
            return {
                "collection_info": {},
                "update_config": json.loads(config.read_text())["semantic_search"]["update_config"],
                "openai_batch": {},
                "chunking": {},
            }

    monkeypatch.setattr(cli, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(standalone, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(semantic_search, "create_semantic_search", lambda *a, **k: _Search())
    if which == "db-status":
        monkeypatch.setattr(sys, "argv", ["zotero-mcp", "db-status", "--config-path", str(config)])
        cli.main()
    else:
        monkeypatch.setattr(sys, "argv", ["zotero-cli", "db", "status", "--config-path", str(config)])
        standalone.main()

    _assert_listed(capsys.readouterr().out)


def test_status_says_nothing_without_items_given_up_on():
    from zotero_mcp.config_light import describe_given_up

    assert describe_given_up({}) is None
    assert describe_given_up({"retry_item_keys_given_up": []}) is None
