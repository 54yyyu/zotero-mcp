"""The CLI's shared recent-item retrieval must not turn failure into success."""

import json
import sys
from types import SimpleNamespace

import pytest

from zotero_mcp import cli_standalone
from zotero_mcp.tools import retrieval


@pytest.mark.parametrize("json_mode", [False, True], ids=["text", "json"])
def test_recent_connection_failure_exits_nonzero(monkeypatch, capsys, json_mode):
    def unavailable_items(**kwargs):
        raise ConnectionError("Connection refused (test backend)")

    monkeypatch.setattr(retrieval._client, "get_zotero_client", lambda: SimpleNamespace(items=unavailable_items))
    monkeypatch.setattr(cli_standalone, "setup_zotero_environment", lambda: None)
    monkeypatch.delenv("ZOTERO_CLI_DEBUG", raising=False)
    args = ["zotero-cli", "get", "recent"]
    if json_mode:
        args.insert(1, "--json")
    monkeypatch.setattr(sys, "argv", args)

    with pytest.raises(SystemExit) as exc:
        cli_standalone.main()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    if json_mode:
        result = json.loads(captured.out)
        assert result["ok"] is False
        assert "Error fetching recent items" in result["error"]["message"]
        assert "Connection refused" in result["error"]["message"]
        assert "data" not in result
    else:
        assert captured.out == ""
        assert "Error fetching recent items" in captured.err
        assert "Connection refused" in captured.err
