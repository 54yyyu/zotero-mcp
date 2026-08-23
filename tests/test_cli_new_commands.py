"""Commands `zotero-cli` gained to close the gap with the MCP tool surface.

The CLI is the token-cheap way to reach this library from a shell-capable
agent, but a capability it does not expose is one that route cannot use at
all. These pin that each new command parses, routes, and hands the tool the
arguments the user actually typed.
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from zotero_mcp import cli_standalone
from zotero_mcp.cli_standalone import _CMD_MAP, _split_csv, build_parser


def _args(**kwargs):
    """Namespace with real values -- a MagicMock would make every getattr
    truthy and silently enable flags the test never set."""
    defaults = dict(verbose=False, json_out=False)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestSplitCsv:
    def test_absent_flag_is_none_not_empty(self):
        """None and [] mean different things to the tools: "not specified"
        versus "specified as nothing"."""
        assert _split_csv(None) is None
        assert _split_csv("") is None

    def test_splits_and_trims(self):
        assert _split_csv("a, b ,c") == ["a", "b", "c"]

    def test_empty_segments_are_dropped(self):
        assert _split_csv("a,,b,") == ["a", "b"]


class TestParsing:
    @pytest.mark.parametrize("argv,command", [
        (["read", "K1", "--start-page", "3"], "read"),
        (["attach", "K1", "--file", "/tmp/x.pdf"], "attach"),
        (["delete", "item", "K1"], "delete"),
        (["export", "--item-keys", "K1,K2"], "export"),
        (["related", "10.1/x"], "related"),
        (["coverage"], "coverage"),
        (["synthesize"], "synthesize"),
        (["path", "K1"], "path"),
        (["batch", "--item-keys", "K1", "--add-tags", "x"], "batch"),
    ])
    def test_command_parses_and_is_routable(self, argv, command):
        parsed = build_parser().parse_args(argv)
        assert parsed.command == command
        assert command in _CMD_MAP

    def test_read_requires_a_start_page(self):
        """A page range with no start is a typo, not a default."""
        with pytest.raises(SystemExit):
            build_parser().parse_args(["read", "K1"])

    def test_read_end_page_defaults_to_none(self):
        parsed = build_parser().parse_args(["read", "K1", "--start-page", "3"])
        assert parsed.end_page is None

    def test_every_command_in_the_map_has_a_parser(self):
        """A handler with no parser is unreachable; a parser with no handler
        exits 1 with the top-level help. Neither is discoverable by hand."""
        parser = build_parser()
        subparsers = [
            action for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ][0]
        for name in _CMD_MAP:
            assert name in subparsers.choices, f"{name} has a handler but no parser"

    def test_new_commands_accept_json_on_either_side(self):
        parser = build_parser()
        assert parser.parse_args(["--json", "path", "K1"]).json_out is True
        assert parser.parse_args(["path", "--json", "K1"]).json_out is True


class TestDispatch:
    """Each handler passes through what the user typed, unmangled."""

    def test_read_forwards_the_page_range(self, monkeypatch):
        # The handler imports the tool module inside the function, so the
        # module object has to exist before its attribute can be swapped --
        # patching by dotted string would re-trigger the import.
        from zotero_mcp.tools import read_pdf as read_pdf_mod

        args = _args(item_key="K1", start_page=3, end_page=7)
        called = {}

        def fake(item_key, start_page, end_page=None, *, ctx=None):
            called.update(start_page=start_page, end_page=end_page)
            return "page text"

        monkeypatch.setattr(read_pdf_mod, "read_pdf_pages", fake)
        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"):
            cli_standalone.cmd_read(args)
        assert called == {"start_page": 3, "end_page": 7}

    def test_delete_item_defaults_to_refusing_notes(self):
        args = _args(subcommand="item", item_key="K1", allow_note=False)
        write_mod = MagicMock()
        write_mod.delete_item.return_value = "deleted"
        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"), \
             patch("zotero_mcp.cli_standalone._import_tools",
                   return_value=(MagicMock(), MagicMock(), MagicMock(), write_mod, MagicMock())):
            cli_standalone.cmd_delete(args)
        assert write_mod.delete_item.call_args.kwargs["allow_note"] is False

    def test_export_splits_item_keys(self, monkeypatch):
        from zotero_mcp.tools import synthesis as synthesis_mod

        args = _args(item_keys="K1,K2", collection=None, style="apa", format="bib")
        called = {}

        def fake(item_keys=None, collection_key=None, style="apa",
                 export_format="bib", *, ctx=None):
            called["item_keys"] = item_keys
            return "refs"

        monkeypatch.setattr(synthesis_mod, "export_bibliography", fake)
        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"):
            cli_standalone.cmd_export(args)
        assert called["item_keys"] == ["K1", "K2"]

    def test_batch_rejects_malformed_set_json_with_a_usage_error(self, capsys):
        """A broken --set must not reach the write path as `None` and quietly
        perform a different update than the one asked for."""
        args = _args(item_keys="K1", query=None, tag=None, add_tags=None,
                     remove_tags=None, set="{not json", remove_keys=None, limit=50)
        write_mod = MagicMock()
        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"), \
             patch("zotero_mcp.cli_standalone._import_tools",
                   return_value=(MagicMock(), MagicMock(), MagicMock(), write_mod, MagicMock())):
            with pytest.raises(SystemExit) as exc:
                cli_standalone.cmd_batch(args)
        assert exc.value.code == 1
        write_mod.batch_update.assert_not_called()

    def test_batch_json_error_uses_the_envelope(self, capsys):
        import json
        args = _args(item_keys="K1", query=None, tag=None, add_tags=None,
                     remove_tags=None, set="{not json", remove_keys=None,
                     limit=50, json_out=True)
        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"), \
             patch("zotero_mcp.cli_standalone._import_tools",
                   return_value=(MagicMock(),) * 5):
            with pytest.raises(SystemExit):
                cli_standalone.cmd_batch(args)
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is False
        assert payload["error"]["code"] == "bad_json"

    def test_annotation_update_splits_tag_flags(self):
        args = _args(subcommand="update", annotation_key="A1", text=None,
                     comment="hi", color=None, add_tags="x,y", remove_tags="z")
        annotations = MagicMock()
        annotations.update_annotation.return_value = "updated"
        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"), \
             patch("zotero_mcp.cli_standalone._import_tools",
                   return_value=(MagicMock(), MagicMock(), annotations, MagicMock(), MagicMock())):
            cli_standalone.cmd_annotations(args)
        kwargs = annotations.update_annotation.call_args.kwargs
        assert kwargs["add_tags"] == ["x", "y"]
        assert kwargs["remove_tags"] == ["z"]
        assert kwargs["comment"] == "hi"
