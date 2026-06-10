"""Tests for the standalone CLI module (zotero-cli entry point)."""

import sys
from unittest.mock import MagicMock, create_autospec, patch

import pytest

import zotero_mcp.tools.write as write_tools
from zotero_mcp.cli_standalone import (
    CLIContext,
    build_parser,
    cmd_add,
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

    def test_add_file_flags(self):
        args = self.parser.parse_args([
            "add", "file", "--filepath", "/tmp/x.pdf",
            "--title", "Override", "--item-type", "book",
        ])
        assert args.subcommand == "file"
        assert args.filepath == "/tmp/x.pdf"
        assert args.title == "Override"
        assert args.item_type == "book"

    def test_add_file_defaults(self):
        args = self.parser.parse_args(["add", "file", "--filepath", "/tmp/x.pdf"])
        assert args.title is None
        assert args.item_type == "document"

    def test_add_file_parent_key_removed(self):
        # --parent-key never mapped to a real add_from_file parameter and made
        # every `add file` invocation raise TypeError; it must stay removed.
        with pytest.raises(SystemExit):
            self.parser.parse_args([
                "add", "file", "--filepath", "/tmp/x.pdf", "--parent-key", "ABC12345",
            ])

    def test_add_doi_if_exists_defaults_to_file(self):
        args = self.parser.parse_args(["add", "doi", "10.1234/x"])
        assert args.if_exists == "file"
        assert args.create_collections is False
        assert args.collection is None

    def test_add_doi_if_exists_choices(self):
        args = self.parser.parse_args(
            ["add", "doi", "10.1234/x", "--if-exists", "duplicate"]
        )
        assert args.if_exists == "duplicate"
        with pytest.raises(SystemExit):
            self.parser.parse_args(["add", "doi", "10.1234/x", "--if-exists", "bogus"])

    def test_add_doi_repeatable_collection_flag(self):
        args = self.parser.parse_args([
            "add", "doi", "10.1234/x",
            "-c", "Reading List", "-c", "_project/topic",
            "--collections", "KEY00001",
            "--create-collections",
        ])
        assert args.collection == ["Reading List", "_project/topic"]
        assert args.collections == "KEY00001"
        assert args.create_collections is True

    def test_add_url_and_file_share_common_flags(self):
        for argv in (
            ["add", "url", "https://example.com", "-c", "X", "--if-exists", "skip"],
            ["add", "file", "--filepath", "/tmp/x.pdf", "-c", "X", "--if-exists", "skip"],
        ):
            args = self.parser.parse_args(argv)
            assert args.collection == ["X"]
            assert args.if_exists == "skip"

    def test_add_isbn_subcommand(self):
        args = self.parser.parse_args(
            ["add", "isbn", "9780262046305", "-c", "Books"]
        )
        assert args.subcommand == "isbn"
        assert args.isbn == "9780262046305"
        assert args.collection == ["Books"]
        assert args.if_exists == "file"

    def test_add_bibtex_subcommand(self):
        args = self.parser.parse_args(
            ["add", "bibtex", "--file", "/tmp/refs.bib", "-c", "Topic"]
        )
        assert args.subcommand == "bibtex"
        assert args.file == "/tmp/refs.bib"
        assert args.bibtex is None

    def test_add_csl_json_subcommand(self):
        args = self.parser.parse_args(
            ["add", "csl-json", "--json", "-", "--if-exists", "duplicate"]
        )
        assert args.subcommand == "csl-json"
        assert args.json == "-"
        assert args.if_exists == "duplicate"


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


# ---------------------------------------------------------------------------
# cmd_add
# ---------------------------------------------------------------------------

class TestCmdAdd:
    """cmd_add must call the write tools with kwargs their real signatures accept.

    Plain MagicMocks hide kwarg mismatches (they accept anything), which is how
    `add file` shipped passing a nonexistent parent_key= and raised TypeError on
    every invocation. Autospec the real functions so signature drift fails here.
    """

    def _args(self, **kwargs):
        defaults = dict(
            verbose=False, collections=None, collection=None, tags=None,
            if_exists="file", create_collections=False,
        )
        defaults.update(kwargs)
        return MagicMock(**defaults)

    def _run(self, args):
        mock_write = MagicMock()
        for fn in ("add_by_doi", "add_by_url", "add_from_file",
                   "add_by_isbn", "add_by_bibtex", "add_by_csl_json"):
            setattr(mock_write, fn,
                    create_autospec(getattr(write_tools, fn), return_value="ok"))
        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"):
            with patch("zotero_mcp.cli_standalone._import_tools",
                       return_value=(MagicMock(), MagicMock(), MagicMock(),
                                     mock_write, MagicMock())):
                cmd_add(args)
        return mock_write

    def test_add_file_matches_real_signature(self, capsys):
        # Regression: would raise TypeError while parent_key= was passed.
        args = self._args(subcommand="file", filepath="/tmp/paper.pdf",
                          title=None, item_type="document")
        mock_write = self._run(args)

        mock_write.add_from_file.assert_called_once()
        call_kwargs = mock_write.add_from_file.call_args.kwargs
        assert call_kwargs["file_path"] == "/tmp/paper.pdf"
        assert call_kwargs["title"] is None
        assert call_kwargs["item_type"] == "document"
        assert "ok" in capsys.readouterr().out

    def test_add_file_forwards_title_and_item_type(self):
        args = self._args(subcommand="file", filepath="/tmp/book.epub",
                          title="My Book", item_type="book",
                          collections="ABCD1234,EFGH5678", tags="a,b")
        mock_write = self._run(args)

        call_kwargs = mock_write.add_from_file.call_args.kwargs
        assert call_kwargs["title"] == "My Book"
        assert call_kwargs["item_type"] == "book"
        assert call_kwargs["collections"] == ["ABCD1234", "EFGH5678"]
        assert call_kwargs["tags"] == ["a", "b"]

    def test_add_doi_matches_real_signature(self):
        args = self._args(subcommand="doi", doi="10.1234/test",
                          attach_mode="auto", collections="ABCD1234")
        mock_write = self._run(args)

        mock_write.add_by_doi.assert_called_once()
        call_kwargs = mock_write.add_by_doi.call_args.kwargs
        assert call_kwargs["doi"] == "10.1234/test"
        assert call_kwargs["collections"] == ["ABCD1234"]

    def test_add_url_matches_real_signature(self):
        args = self._args(subcommand="url",
                          url="https://arxiv.org/abs/2301.00001",
                          attach_mode="auto")
        mock_write = self._run(args)

        mock_write.add_by_url.assert_called_once()
        assert mock_write.add_by_url.call_args.kwargs["url"] == (
            "https://arxiv.org/abs/2301.00001"
        )

    def test_add_doi_forwards_if_exists_and_create_flags(self):
        args = self._args(subcommand="doi", doi="10.1234/test",
                          attach_mode="auto", if_exists="skip",
                          create_collections=True)
        mock_write = self._run(args)

        call_kwargs = mock_write.add_by_doi.call_args.kwargs
        assert call_kwargs["if_exists"] == "skip"
        assert call_kwargs["create_missing_collections"] is True

    def test_add_doi_default_if_exists_is_file(self):
        """The CLI defaults to convergent behavior; MCP keeps 'duplicate'."""
        args = self._args(subcommand="doi", doi="10.1234/test",
                          attach_mode="auto")
        mock_write = self._run(args)

        assert mock_write.add_by_doi.call_args.kwargs["if_exists"] == "file"

    def test_repeatable_collection_flag_merges_without_splitting(self):
        # -c values are single specs (never comma-split — names may contain
        # commas); --collections is comma-split; both merge in order.
        args = self._args(subcommand="doi", doi="10.1234/test",
                          attach_mode="auto",
                          collections="KEY00001,Reading List",
                          collection=["_project/a, b topic", "Other"])
        mock_write = self._run(args)

        assert mock_write.add_by_doi.call_args.kwargs["collections"] == [
            "KEY00001", "Reading List", "_project/a, b topic", "Other",
        ]

    def test_add_isbn_matches_real_signature(self):
        args = self._args(subcommand="isbn", isbn="9780262046305",
                          collections="Books")
        mock_write = self._run(args)

        mock_write.add_by_isbn.assert_called_once()
        call_kwargs = mock_write.add_by_isbn.call_args.kwargs
        assert call_kwargs["isbn"] == "9780262046305"
        assert call_kwargs["collections"] == ["Books"]
        assert call_kwargs["if_exists"] == "file"

    def test_add_bibtex_inline_matches_real_signature(self):
        args = self._args(subcommand="bibtex", bibtex="@article{x, title={T}}",
                          file=None, attach_mode="auto")
        mock_write = self._run(args)

        mock_write.add_by_bibtex.assert_called_once()
        call_kwargs = mock_write.add_by_bibtex.call_args.kwargs
        assert call_kwargs["bibtex"] == "@article{x, title={T}}"
        assert call_kwargs["file_path"] is None

    def test_add_bibtex_reads_stdin_when_dash(self, monkeypatch):
        monkeypatch.setattr("sys.stdin",
                            MagicMock(read=lambda: "@book{y, title={Y}}"))
        args = self._args(subcommand="bibtex", bibtex="-", file=None,
                          attach_mode="auto")
        mock_write = self._run(args)

        assert mock_write.add_by_bibtex.call_args.kwargs["bibtex"] == (
            "@book{y, title={Y}}"
        )

    def test_add_csl_json_file_matches_real_signature(self):
        args = self._args(subcommand="csl-json", json=None,
                          file="/tmp/refs.json", attach_mode="auto")
        mock_write = self._run(args)

        mock_write.add_by_csl_json.assert_called_once()
        call_kwargs = mock_write.add_by_csl_json.call_args.kwargs
        assert call_kwargs["file_path"] == "/tmp/refs.json"
        assert call_kwargs["csl_json"] is None
