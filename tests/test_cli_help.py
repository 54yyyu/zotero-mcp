import re
from unittest.mock import patch

import pytest

from zotero_mcp.cli import main


def test_zotero_mcp_help_mentions_batch_indexing(capsys):
    with patch("sys.argv", ["zotero-mcp", "help"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Batch API indexing" in out
    assert "zotero-mcp update-db --batch" in out
    assert "batch-status" in out
    assert "batch-import" in out


def test_zotero_mcp_help_still_lists_the_deprecated_openai_commands(capsys):
    """The openai-batch-* commands shipped in 0.10.0 and stay callable."""
    with patch("sys.argv", ["zotero-mcp", "help"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "openai-batch-status" in out
    assert "openai-batch-import" in out


def test_zotero_mcp_help_update_db_shows_batch_flags(capsys):
    with patch("sys.argv", ["zotero-mcp", "help", "update-db"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--batch" in out
    assert "--no-batch" in out
    assert "--batch-provider" in out
    # Deprecated but still accepted, so scripts written against 0.10.0 work.
    assert "--openai-batch" in out
    assert "--no-openai-batch" in out


# --- `zotero-mcp help` must document the whole command line ------------------


def _full_help(capsys, *argv):
    with patch("sys.argv", ["zotero-mcp", *argv]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0
    return capsys.readouterr().out


def test_every_command_and_option_is_documented_in_help(capsys):
    """Fails when a command or option is added to the parser but not to `help`.

    The reference is generated from the parser, so this guards the generator
    (and any option hidden from it) rather than a hand-written list.
    """
    import argparse

    from zotero_mcp.cli import _build_parser
    from zotero_mcp.cli_help import command_options

    parser = _build_parser()
    out = _full_help(capsys, "help")
    sub_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))

    missing = []
    for name, sub in sub_action.choices.items():
        if not re.search(rf"^  {re.escape(name)} ", out, re.MULTILINE):
            missing.append(name)
        for opt in command_options(sub):
            for flag in opt.option_strings:
                # Whole flag only: --batch must not pass by way of --batch-provider.
                if not re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", out):
                    missing.append(f"{name} {flag}")
            if not (opt.help or "").strip():
                missing.append(f"{name} {opt.option_strings} has no description")
        # A user-visible option must not be hidden from the reference.
        for opt in sub._actions:
            if opt.option_strings and not isinstance(opt, argparse._HelpAction):
                assert opt.help is not argparse.SUPPRESS, (
                    f"{name} {opt.option_strings} is hidden from help"
                )
    assert not missing, f"not documented in `zotero-mcp help`: {missing}"


def test_every_command_is_assigned_to_a_help_group():
    import argparse

    from zotero_mcp.cli import _build_parser
    from zotero_mcp.cli_help import HELP_GROUPS

    sub_action = next(a for a in _build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    grouped = [n for _, names in HELP_GROUPS for n in names]
    assert sorted(set(sub_action.choices) - set(grouped)) == [], "classify new commands in HELP_GROUPS"
    assert sorted(set(grouped) - set(sub_action.choices)) == [], "HELP_GROUPS names a command that does not exist"
    assert len(grouped) == len(set(grouped)), "a command is in two groups"


def test_help_shows_flags_with_their_values_and_descriptions(capsys):
    out = _full_help(capsys, "help")
    assert "--transport {stdio,streamable-http,sse}" in out
    assert "--batch-max-tokens N" in out
    assert "Force complete rebuild of the database" in out


def test_dash_h_and_help_print_the_same_reference(capsys):
    assert _full_help(capsys, "-h") == _full_help(capsys, "help") == _full_help(capsys, "--help")


def test_help_for_one_command_still_works(capsys):
    out = _full_help(capsys, "help", "db-status")
    assert "--config-path" in out
    assert "update-db" not in out
