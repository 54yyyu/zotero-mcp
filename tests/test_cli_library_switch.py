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

from zotero_mcp.cli_standalone import cmd_library

OK = ("Successfully switched to library **6182688** (type=group). "
      "All tools now operate on this library.")


def _run(capsys, message, json_out):
    retrieval = MagicMock()
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
    assert "All tools now operate on this library" not in out
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
