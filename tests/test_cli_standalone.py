"""Tests for the standalone CLI module (zotero-cli entry point)."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from zotero_mcp.cli_standalone import (
    CLIContext,
    build_parser,
    cmd_edit,
    cmd_notes,
    cmd_search,
    main,
)
from zotero_mcp._context import Context


# ---------------------------------------------------------------------------
# CLIContext
# ---------------------------------------------------------------------------

class TestCLIContext:
    def test_info_silent_by_default(self, capsys):
        ctx = CLIContext(verbose=False)
        ctx.info("hello")
        assert capsys.readouterr().err == ""

    def test_info_prints_when_verbose(self, capsys):
        ctx = CLIContext(verbose=True)
        ctx.info("hello")
        assert "hello" in capsys.readouterr().err

    def test_warning_always_prints(self, capsys):
        ctx = CLIContext(verbose=False)
        ctx.warning("watch out")
        assert "watch out" in capsys.readouterr().err

    def test_error_always_prints(self, capsys):
        ctx = CLIContext(verbose=False)
        ctx.error("boom")
        assert "boom" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _context.py stub / passthrough
# ---------------------------------------------------------------------------

class TestContextModule:
    def test_context_is_importable(self):
        assert Context is not None

    def test_context_has_required_methods(self):
        # Check the class interface rather than instantiating: when fastmcp is
        # installed the real Context requires constructor args we don't have.
        for method in ("info", "warning", "error"):
            assert callable(getattr(Context, method, None)), f"Context.{method} missing"


# ---------------------------------------------------------------------------
# Parser structure
# ---------------------------------------------------------------------------

class TestParser:
    def setup_method(self):
        self.parser = build_parser()

    def test_no_command_sets_none(self):
        args = self.parser.parse_args([])
        assert args.command is None

    def test_search_alias_s(self):
        args = self.parser.parse_args(["s", "Einstein"])
        assert args.command in ("search", "s")
        assert args.query == "Einstein"

    def test_get_alias_g(self):
        args = self.parser.parse_args(["g", "metadata", "ABC123"])
        assert args.command in ("get", "g")
        assert args.subcommand == "metadata"
        assert args.item_key == "ABC123"

    def test_search_default_mode_is_items(self):
        args = self.parser.parse_args(["search", "test"])
        assert args.mode == "items"

    def test_search_mode_semantic(self):
        args = self.parser.parse_args(["search", "--mode", "semantic", "neural networks"])
        assert args.mode == "semantic"
        assert args.query == "neural networks"

    def test_search_limit(self):
        args = self.parser.parse_args(["search", "--limit", "25", "query"])
        assert args.limit == 25

    def test_edit_item_key_positional(self):
        args = self.parser.parse_args(["edit", "KEY123", "--title", "New Title"])
        assert args.item_key == "KEY123"
        assert args.title == "New Title"

    def test_db_update_flags(self):
        args = self.parser.parse_args(["db", "update", "--force-rebuild", "--limit", "10"])
        assert args.subcommand == "update"
        assert args.force_rebuild is True
        assert args.limit == 10

    def test_add_doi_positional(self):
        args = self.parser.parse_args(["add", "doi", "10.1234/test"])
        assert args.subcommand == "doi"
        assert args.doi == "10.1234/test"

    def test_notes_create_required_item_key(self):
        # --item-key is required for notes create
        with pytest.raises(SystemExit):
            self.parser.parse_args(["notes", "create", "--text", "hello"])

    def test_verbose_flag(self):
        args = self.parser.parse_args(["-v", "search", "test"])
        assert args.verbose is True

    def test_annotations_alias_ann(self):
        args = self.parser.parse_args(["ann", "list"])
        assert args.command in ("annotations", "ann")
        assert args.subcommand == "list"

    def test_collections_alias_coll(self):
        args = self.parser.parse_args(["coll", "search", "my collection"])
        assert args.command in ("collections", "coll")
        assert args.subcommand == "search"


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------

class TestMain:
    def test_no_command_prints_help_and_exits_0(self, capsys):
        with patch("sys.argv", ["zotero-cli"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 0

    def test_unknown_command_exits_nonzero(self):
        with patch("sys.argv", ["zotero-cli", "nonexistent-command-xyz"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code != 0

    def test_keyboard_interrupt_exits_130(self):
        def _raise(*a, **kw):
            raise KeyboardInterrupt

        with patch("sys.argv", ["zotero-cli", "config"]):
            with patch.dict("zotero_mcp.cli_standalone._CMD_MAP", {"config": _raise}):
                with pytest.raises(SystemExit) as exc:
                    main()
        assert exc.value.code == 130

    def test_exception_exits_1_by_default(self):
        def _raise(*a, **kw):
            raise RuntimeError("something broke")

        with patch("sys.argv", ["zotero-cli", "config"]):
            with patch.dict("zotero_mcp.cli_standalone._CMD_MAP", {"config": _raise}):
                with pytest.raises(SystemExit) as exc:
                    main()
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# cmd_search
# ---------------------------------------------------------------------------

class TestCmdSearch:
    def _args(self, **kwargs):
        defaults = dict(
            verbose=False, mode="items", query="test", qmode="titleCreatorYear",
            limit=10, collection=None, conditions=None, join_mode="all",
            sort_by=None, sort_direction="asc", filters=None,
        )
        defaults.update(kwargs)
        return MagicMock(**defaults)

    def test_search_items_called(self, capsys):
        args = self._args(query="Einstein 1905")
        mock_search = MagicMock()
        mock_search.search_items.return_value = "# Results"

        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"):
            with patch("zotero_mcp.cli_standalone._import_tools",
                       return_value=(mock_search, MagicMock(), MagicMock(), MagicMock(), MagicMock())):
                cmd_search(args)

        mock_search.search_items.assert_called_once()
        call_kwargs = mock_search.search_items.call_args.kwargs
        assert call_kwargs["query"] == "Einstein 1905"
        assert "# Results" in capsys.readouterr().out

    def test_search_tag_mode(self):
        args = self._args(mode="tag", query="important,reviewed")
        mock_search = MagicMock()
        mock_search.search_by_tag.return_value = "tagged results"

        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"):
            with patch("zotero_mcp.cli_standalone._import_tools",
                       return_value=(mock_search, MagicMock(), MagicMock(), MagicMock(), MagicMock())):
                cmd_search(args)

        mock_search.search_by_tag.assert_called_once()
        call_kwargs = mock_search.search_by_tag.call_args.kwargs
        assert call_kwargs["tag"] == ["important", "reviewed"]

    def test_search_advanced_invalid_json_exits(self):
        args = self._args(mode="advanced", conditions="not-json")
        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"):
            with patch("zotero_mcp.cli_standalone._import_tools",
                       return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())):
                with pytest.raises(SystemExit) as exc:
                    cmd_search(args)
        assert exc.value.code == 1

    def test_search_semantic_invalid_filters_exits(self):
        args = self._args(mode="semantic", filters="bad-json")
        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"):
            with patch("zotero_mcp.cli_standalone._import_tools",
                       return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())):
                with pytest.raises(SystemExit) as exc:
                    cmd_search(args)
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# cmd_notes
# ---------------------------------------------------------------------------

class TestCmdNotes:
    def test_create_empty_text_exits(self, monkeypatch):
        args = MagicMock(subcommand="create", item_key="KEY1", text="",
                         title="Note", tags=None, verbose=False)
        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"):
            with patch("zotero_mcp.cli_standalone._import_tools",
                       return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())):
                with pytest.raises(SystemExit) as exc:
                    cmd_notes(args)
        assert exc.value.code == 1

    def test_create_reads_stdin_when_dash(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", MagicMock(read=lambda: "note from stdin"))
        args = MagicMock(subcommand="create", item_key="KEY1", text="-",
                         title="T", tags=None, verbose=False)
        mock_annotations = MagicMock()
        mock_annotations.create_note.return_value = "created"

        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"):
            with patch("zotero_mcp.cli_standalone._import_tools",
                       return_value=(MagicMock(), MagicMock(), mock_annotations, MagicMock(), MagicMock())):
                cmd_notes(args)

        call_kwargs = mock_annotations.create_note.call_args.kwargs
        assert call_kwargs["note_text"] == "note from stdin"

    def test_create_splits_tags(self, capsys):
        args = MagicMock(subcommand="create", item_key="KEY1", text="hello",
                         title="T", tags="a,b,c", verbose=False)
        mock_annotations = MagicMock()
        mock_annotations.create_note.return_value = "ok"

        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"):
            with patch("zotero_mcp.cli_standalone._import_tools",
                       return_value=(MagicMock(), MagicMock(), mock_annotations, MagicMock(), MagicMock())):
                cmd_notes(args)

        call_kwargs = mock_annotations.create_note.call_args.kwargs
        assert call_kwargs["tags"] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# cmd_edit
# ---------------------------------------------------------------------------

class TestCmdEdit:
    def _args(self, item_key="KEY1", creators=None, **kwargs):
        defaults = dict(
            verbose=False, title=None, creators=creators, date=None,
            publication_title=None, abstract=None, tags=None, add_tags=None,
            remove_tags=None, collections=None, collection_names=None,
            doi=None, url=None, extra=None, volume=None, issue=None,
            pages=None, publisher=None, issn=None, language=None,
            short_title=None, edition=None, isbn=None, book_title=None,
        )
        defaults.update(kwargs)
        return MagicMock(item_key=item_key, **defaults)

    def test_edit_calls_update_item(self):
        args = self._args(item_key="ABC", title="New Title")
        mock_write = MagicMock()
        mock_write.update_item.return_value = "updated"

        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"):
            with patch("zotero_mcp.cli_standalone._import_tools",
                       return_value=(MagicMock(), MagicMock(), MagicMock(), mock_write, MagicMock())):
                cmd_edit(args)

        mock_write.update_item.assert_called_once()
        assert mock_write.update_item.call_args.kwargs["item_key"] == "ABC"

    def test_edit_invalid_creators_json_exits(self):
        args = self._args(creators="not-valid-json")
        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"):
            with patch("zotero_mcp.cli_standalone._import_tools",
                       return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())):
                with pytest.raises(SystemExit) as exc:
                    cmd_edit(args)
        assert exc.value.code == 1

    def test_edit_valid_creators_json(self):
        creators_json = '[{"firstName": "Albert", "lastName": "Einstein", "creatorType": "author"}]'
        args = self._args(creators=creators_json)
        mock_write = MagicMock()
        mock_write.update_item.return_value = "ok"

        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"):
            with patch("zotero_mcp.cli_standalone._import_tools",
                       return_value=(MagicMock(), MagicMock(), MagicMock(), mock_write, MagicMock())):
                cmd_edit(args)

        call_kwargs = mock_write.update_item.call_args.kwargs
        assert isinstance(call_kwargs["creators"], list)
        assert call_kwargs["creators"][0]["lastName"] == "Einstein"

    def test_edit_splits_comma_tags(self):
        args = self._args(add_tags="tag1,tag2,tag3")
        mock_write = MagicMock()
        mock_write.update_item.return_value = "ok"

        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"):
            with patch("zotero_mcp.cli_standalone._import_tools",
                       return_value=(MagicMock(), MagicMock(), MagicMock(), mock_write, MagicMock())):
                cmd_edit(args)

        call_kwargs = mock_write.update_item.call_args.kwargs
        assert call_kwargs["add_tags"] == ["tag1", "tag2", "tag3"]
