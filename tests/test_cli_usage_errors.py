"""Usage errors are reported like every other failure, and before any work.

A value the caller got wrong (broken JSON in a flag, an empty note) used to
be printed as plain text on stderr even under ``--json``, after the tool
modules had been imported; a caller reading stdout for the envelope got
nothing. ``db status`` reported an unreadable index and exited 0.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from zotero_mcp import cli_standalone
from zotero_mcp.cli_standalone import build_parser

_BAD_JSON = [
    (["search", "--mode", "advanced", "--conditions", "{bad"], "search", "--conditions"),
    (["search", "--mode", "semantic", "--filters", "{bad", "topic"], "search", "--filters"),
    (["edit", "ABCD1234", "--creators", "{bad"], "edit", "--creators"),
    (["batch", "--item-keys", "ABCD1234", "--set", "{bad"], "batch", "--set"),
]


def _run(argv, monkeypatch):
    """Run the command the way main() does, with any real work forbidden."""
    def forbidden(*_a, **_k):
        raise AssertionError("the tools were loaded before the arguments were checked")

    monkeypatch.setattr(cli_standalone, "_import_tools", forbidden)
    monkeypatch.setattr(cli_standalone, "setup_zotero_environment", forbidden)
    args = build_parser().parse_args(argv)
    with pytest.raises(SystemExit) as exc:
        cli_standalone._CMD_MAP[args.command](args)
    return exc.value.code


@pytest.mark.parametrize("argv,command,flag", _BAD_JSON)
def test_bad_json_in_a_flag_is_a_json_error(argv, command, flag, monkeypatch, capsys):
    assert _run(["--json", *argv], monkeypatch) == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is False and payload["command"] == command
    assert payload["error"]["code"] == "bad_json"
    assert f"invalid JSON in {flag}" in payload["error"]["message"]
    assert captured.err == ""


@pytest.mark.parametrize("argv,command,flag", _BAD_JSON)
def test_bad_json_in_a_flag_reads_the_same_without_json(argv, command, flag, monkeypatch, capsys):
    assert _run(argv, monkeypatch) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"Error: invalid JSON in {flag}: ")


def test_advanced_search_without_conditions_says_so(monkeypatch, capsys):
    assert _run(["--json", "search", "--mode", "advanced"], monkeypatch) == 1
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "bad_args" and "--conditions" in error["message"]


def test_an_empty_note_is_a_json_error(monkeypatch, capsys):
    argv = ["--json", "notes", "create", "--item-key", "ABCD1234", "--text", ""]
    assert _run(argv, monkeypatch) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "notes create"
    assert payload["error"]["code"] == "bad_args" and "--text" in payload["error"]["message"]


class TestDbStatusExitCode:
    def _status(self, collection_info, monkeypatch):
        search = MagicMock()
        search.get_database_status.return_value = {
            "collection_info": collection_info, "update_config": {}, "openai_batch": {},
        }
        monkeypatch.setattr(cli_standalone, "setup_zotero_environment", lambda: None)
        args = build_parser().parse_args(["db", "status"])
        with patch("zotero_mcp.semantic_search.create_semantic_search", return_value=search):
            cli_standalone.cmd_db(args)

    def test_a_readable_index_exits_zero(self, monkeypatch, capsys):
        self._status({"name": "zotero", "count": 3}, monkeypatch)
        assert "Document count: 3" in capsys.readouterr().out

    @pytest.mark.parametrize("collection_info", [
        {"name": "zotero", "count": 0, "error": "database disk image is malformed"},
        {"name": "zotero", "count": 0, "error": "boom", "unreadable": True,
         "persist_directory": "/tmp/nowhere", "last_written": 0.0, "size_bytes": 2**30},
    ])
    def test_an_index_that_cannot_be_read_exits_one(self, collection_info, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            self._status(collection_info, monkeypatch)
        assert exc.value.code == 1
        out = capsys.readouterr().out
        # The status is still printed; only the exit code changed.
        assert "=== Semantic Search Database Status ===" in out and "Error: " in out
