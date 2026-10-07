"""`zotero-cli library switch` must not claim the switch carries over (#606).

Each zotero-cli command is a new process, and the switch lives in the
process, so the next command is back on the default library. The tool's own
message ("All tools now operate on this library") is right for the MCP
server and wrong here.
"""

import argparse
import json
from unittest.mock import MagicMock, patch

import pytest

from conftest import DummyContext

from zotero_mcp.cli_standalone import cmd_library
from zotero_mcp.tools import retrieval as real_retrieval

# Built from the real constant, so rewording it in retrieval.py cannot leave
# this test passing on a sentence the tool no longer says.
OK = ("Successfully switched to library **6182688** (type=group). "
      + real_retrieval.SWITCH_SUCCESS_NOTE)


def _run(capsys, message, json_out):
    retrieval = MagicMock()
    retrieval.SWITCH_SUCCESS_NOTE = real_retrieval.SWITCH_SUCCESS_NOTE
    retrieval.switch_library.return_value = message
    args = argparse.Namespace(action="switch", library_id="6182688",
                              library_type="group", json_out=json_out,
                              verbose=False)
    with patch("zotero_mcp.cli_standalone.setup_zotero_environment"), \
         patch("zotero_mcp.cli_standalone._import_tools",
               return_value=(None, retrieval, None, None, MagicMock())):
        cmd_library(args)
    return capsys.readouterr().out


def test_switch_says_it_lasts_one_command_and_names_the_env_vars(capsys):
    out = _run(capsys, OK, json_out=False)
    assert real_retrieval.SWITCH_SUCCESS_NOTE not in out
    assert "only lasts for this zotero-cli command" in out
    assert "ZOTERO_LIBRARY_ID=6182688 ZOTERO_LIBRARY_TYPE=group" in out


def test_switch_json_envelope_carries_the_same_text(capsys):
    env = json.loads(_run(capsys, OK, json_out=True))
    assert env["ok"] is True
    assert "All tools now operate" not in env["data"]["text"]
    assert "ZOTERO_LIBRARY_TYPE=group" in env["data"]["text"]


def test_failed_switch_is_left_as_an_error(capsys):
    with pytest.raises(SystemExit):
        _run(capsys, "Error: Could not access library 6182688 (type=group): x. "
                     "Reverted to default library.", json_out=False)


def test_real_switch_library_success_ends_with_the_shared_sentence(monkeypatch):
    """The CLI swaps this sentence out, so the real tool must keep emitting it."""
    import types

    monkeypatch.setattr(real_retrieval, "validate_library_switch", lambda *a, **k: None)
    monkeypatch.setattr(real_retrieval._client, "set_active_library", lambda *a, **k: None)
    monkeypatch.setattr(
        real_retrieval._library, "get_library_backend", lambda: types.SimpleNamespace(name="sqlite")
    )
    text = real_retrieval.switch_library("6182688", "group", ctx=DummyContext())
    assert text.startswith("Successfully switched")
    assert text.endswith(real_retrieval.SWITCH_SUCCESS_NOTE)


def test_default_reset_message_is_short_and_honest(capsys):
    retrieval = MagicMock()
    args = argparse.Namespace(action="reset", library_id="", library_type="default",
                              json_out=False, verbose=False)
    with patch("zotero_mcp.cli_standalone.setup_zotero_environment"), \
         patch("zotero_mcp.cli_standalone._import_tools",
               return_value=(None, retrieval, None, None, MagicMock())):
        cmd_library(args)
    out = capsys.readouterr().out
    assert "always starts on the default library" in out
    assert "nothing to reset" in out
