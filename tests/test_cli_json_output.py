"""`zotero-cli --json` emits one stable envelope per invocation.

The CLI's default output is markdown written for a person. Pipelines, cron
jobs and shell-capable agents had to regex prose to recover an item key.
`--json` gives them a contract instead; these tests pin it, because a
downstream script breaks silently when the shape drifts.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from zotero_mcp import cli_json
from zotero_mcp.cli_standalone import (
    _fetch_projected,
    _keys_from_markdown,
    build_parser,
    cmd_get,
    cmd_search,
)


def _raw(key, item_type="journalArticle", title="A Paper", **fields):
    data = {
        "key": key, "itemType": item_type, "title": title, "date": "2024-05-01",
        "creators": [{"creatorType": "author", "firstName": "Ada", "lastName": "Lovelace"}],
        "tags": [{"tag": "to-read"}], "collections": ["COLL0001"],
        "DOI": "10.1234/x", "abstractNote": "An abstract.",
    }
    data.update(fields)
    return {"key": key, "data": data, "meta": {}}


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------

class TestEnvelope:
    def test_success_shape(self):
        env = cli_json.envelope("search", {"count": 0})
        assert env["ok"] is True
        assert env["command"] == "search"
        assert env["schema"] == cli_json.SCHEMA_VERSION
        assert env["data"] == {"count": 0}
        assert "error" not in env

    def test_failure_shape(self):
        env = cli_json.envelope("search", ok=False, error="boom", code="nope")
        assert env["ok"] is False
        assert env["error"] == {"message": "boom", "code": "nope"}
        assert "data" not in env

    def test_errors_go_to_stdout_not_stderr(self, capsys):
        """A caller reading one stream must see both outcomes -- otherwise a
        failed run looks like an empty one."""
        cli_json.emit_error("search", "boom", "nope")
        captured = capsys.readouterr()
        assert captured.err == ""
        assert json.loads(captured.out)["ok"] is False

    def test_output_is_a_single_line(self, capsys):
        """One object per invocation keeps `... | jq` and read-a-line callers
        working without a streaming parser."""
        cli_json.emit("search", {"items": [{"key": "A"}]})
        out = capsys.readouterr().out
        assert out.count("\n") == 1
        json.loads(out)


# ---------------------------------------------------------------------------
# Item projection
# ---------------------------------------------------------------------------

class TestProjection:
    def test_keys_only_is_minimal_but_identifying(self):
        got = cli_json.project_item(_raw("ITEM0001"), "keys_only")
        assert set(got) == {"key", "itemType", "title", "date"}

    def test_summary_adds_the_fields_a_pipeline_filters_on(self):
        got = cli_json.project_item(_raw("ITEM0001"), "summary")
        assert got["doi"] == "10.1234/x"
        assert got["tags"] == ["to-read"]
        assert got["collections"] == ["COLL0001"]
        assert got["creators"][0]["last"] == "Lovelace"
        assert "raw" not in got

    def test_full_carries_the_complete_record(self):
        got = cli_json.project_item(_raw("ITEM0001"), "full")
        assert got["abstract"] == "An abstract."
        assert got["raw"]["itemType"] == "journalArticle"

    def test_every_level_identifies_the_item(self):
        for detail in ("keys_only", "summary", "full"):
            got = cli_json.project_item(_raw("ITEM0001"), detail)
            assert got["key"] == "ITEM0001"
            assert got["itemType"] == "journalArticle"

    def test_trashed_items_are_flagged(self):
        """A trashed item is indistinguishable from a live one otherwise, and
        acting on one unknowingly is the failure this prevents."""
        got = cli_json.project_item(_raw("ITEM0001", deleted=1), "keys_only")
        assert got["deleted"] is True

    def test_live_items_carry_no_deleted_key(self):
        assert "deleted" not in cli_json.project_item(_raw("ITEM0001"), "keys_only")

    def test_title_resolves_type_specific_base_fields(self):
        """Same resolution as the markdown path, so the two never disagree."""
        statute = _raw("ITEM0002", item_type="statute", title=None)
        statute["data"].pop("title")
        statute["data"]["nameOfAct"] = "The Real Act"
        assert cli_json.project_item(statute)["title"] == "The Real Act"

    def test_note_projection_offers_text_and_html(self):
        note = {"key": "NOTE0001", "data": {
            "key": "NOTE0001", "itemType": "note",
            "note": "<p>First line</p><p>Second</p>", "tags": [],
        }}
        got = cli_json.project_note(note)
        assert got["title"] == "First line"
        assert "First line" in got["text"] and "<p>" not in got["text"]
        assert got["html"] == "<p>First line</p><p>Second</p>"

    def test_tags_normalise_across_endpoint_shapes(self):
        assert cli_json.project_tag("plain") == {"tag": "plain"}
        assert cli_json.project_tag({"tag": "dict"})["tag"] == "dict"


# ---------------------------------------------------------------------------
# Key extraction -- the seam that keeps JSON and markdown agreeing
# ---------------------------------------------------------------------------

class TestKeyExtraction:
    def test_reads_keys_from_the_shared_item_formatter(self):
        from zotero_mcp.utils import format_item_result

        md = "\n".join(format_item_result(_raw("ABCD1234"), index=1))
        assert _keys_from_markdown(md) == ["ABCD1234"]

    def test_reads_keys_from_a_keys_only_listing(self):
        md = "- `AAAA1111` | One (2024)\n- `BBBB2222` | Two (2023) [PDF]"
        assert _keys_from_markdown(md) == ["AAAA1111", "BBBB2222"]

    def test_order_is_preserved_and_duplicates_collapse(self):
        md = ("**Item Key:** BBBB2222\n**Item Key:** AAAA1111\n"
              "**Item Key:** BBBB2222\n")
        assert _keys_from_markdown(md) == ["BBBB2222", "AAAA1111"]

    def test_prose_mentioning_no_keys_yields_none(self):
        assert _keys_from_markdown("No items found matching the criteria.") == []
        assert _keys_from_markdown("") == []


class TestFetchProjected:
    def test_result_order_follows_the_requested_order(self):
        """Rank order carries the answer for a search; the API returns
        whatever order it likes."""
        zot = MagicMock()
        zot.items.return_value = [_raw("BBBB2222"), _raw("AAAA1111")]
        got = _fetch_projected(zot, ["AAAA1111", "BBBB2222"], "keys_only")
        assert [i["key"] for i in got] == ["AAAA1111", "BBBB2222"]

    def test_keys_the_fetch_cannot_resolve_are_dropped_not_faked(self):
        zot = MagicMock()
        zot.items.return_value = [_raw("AAAA1111")]
        got = _fetch_projected(zot, ["AAAA1111", "GONE0000"], "keys_only")
        assert [i["key"] for i in got] == ["AAAA1111"]

    def test_no_keys_makes_no_request(self):
        zot = MagicMock()
        assert _fetch_projected(zot, []) == []
        zot.items.assert_not_called()

    def test_more_than_fifty_keys_are_chunked(self):
        """itemKey takes at most 50 per request."""
        keys = [f"K{i:07d}" for i in range(120)]
        zot = MagicMock()
        zot.items.side_effect = lambda itemKey, start=0, limit=100: [
            _raw(k) for k in itemKey.split(",")
        ]
        got = _fetch_projected(zot, keys, "keys_only")
        assert len(got) == 120
        assert zot.items.call_count == 3

    def test_local_api_children_do_not_crowd_out_the_parents(self):
        """The local API answers an itemKey filter with the requested items
        *plus* their children, and lists the children first. Sizing the request
        to the number of keys asked for therefore truncated the response before
        any parent appeared: `found` came back holding only child keys, and a
        search markdown mode answers with eight items rendered as
        `count: 0` (#499).

        The fake honours `start`/`limit` so the pre-fix call -- one request
        capped at len(keys) -- is expressible against it, and fails.
        """
        keys = ["AAAA1111", "BBBB2222"]
        expanded = [
            _raw("CHILD001", item_type="attachment"),
            _raw("CHILD002", item_type="attachment"),
            _raw("AAAA1111"),
            _raw("CHILD003", item_type="note"),
            _raw("BBBB2222"),
        ]
        zot = MagicMock()
        zot.items.side_effect = (
            lambda itemKey, start=0, limit=100: expanded[start:start + limit]
        )
        got = _fetch_projected(zot, keys, "keys_only")
        assert [i["key"] for i in got] == keys

    def test_a_web_api_selection_still_costs_one_request(self):
        """The web API filters itemKey correctly and returns at most one record
        per key, so the first page comes back short of a full page and paging
        stops there. Fixing the local API's behaviour must not buy a second
        round trip for everyone else."""
        keys = ["AAAA1111", "BBBB2222"]
        zot = MagicMock()
        zot.items.side_effect = lambda itemKey, start=0, limit=100: (
            [_raw(k) for k in itemKey.split(",")] if start == 0 else []
        )
        got = _fetch_projected(zot, keys, "keys_only")
        assert [i["key"] for i in got] == keys
        assert zot.items.call_count == 1


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _args(**kwargs):
    """A namespace with real values -- MagicMock would answer every getattr
    with a truthy mock and silently enable flags the test never set."""
    import argparse
    defaults = dict(verbose=False, json_out=True)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestSearchCommand:
    def test_json_search_returns_projected_items(self, capsys):
        args = _args(mode="items", query="lovelace", qmode="titleCreatorYear",
                     limit=10, collection=None, detail="summary")
        search_mod = MagicMock()
        search_mod.search_items.return_value = "## 1. A Paper\n**Item Key:** ABCD1234\n"
        client = MagicMock()
        client.get_zotero_client.return_value.items.return_value = [_raw("ABCD1234")]

        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"), \
             patch("zotero_mcp.cli_standalone._import_tools",
                   return_value=(search_mod, MagicMock(), MagicMock(), MagicMock(), client)):
            cmd_search(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True
        assert payload["command"] == "search"
        assert payload["data"]["query"] == "lovelace"
        assert payload["data"]["count"] == 1
        assert payload["data"]["items"][0]["key"] == "ABCD1234"

    def test_markdown_mode_is_untouched(self, capsys):
        args = _args(mode="items", query="x", qmode="titleCreatorYear", limit=10,
                     collection=None, json_out=False)
        search_mod = MagicMock()
        search_mod.search_items.return_value = "# Results\n"

        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"), \
             patch("zotero_mcp.cli_standalone._import_tools",
                   return_value=(search_mod, MagicMock(), MagicMock(), MagicMock(), MagicMock())):
            cmd_search(args)

        out = capsys.readouterr().out
        assert out.strip() == "# Results"
        assert "{" not in out

    def test_an_empty_search_is_a_success_with_no_items(self, capsys):
        """Not an error: "nothing matched" is a legitimate answer, and a
        caller must be able to tell it from a failed call."""
        args = _args(mode="items", query="zzz", qmode="titleCreatorYear",
                     limit=10, collection=None, detail="summary")
        search_mod = MagicMock()
        search_mod.search_items.return_value = "No items found."

        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"), \
             patch("zotero_mcp.cli_standalone._import_tools",
                   return_value=(search_mod, MagicMock(), MagicMock(), MagicMock(), MagicMock())):
            cmd_search(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True
        assert payload["data"]["items"] == []


class TestGetCommand:
    def test_metadata_returns_the_raw_record(self, capsys):
        args = _args(subcommand="metadata", item_key="ABCD1234",
                     no_abstract=False, output_format="markdown")
        retrieval = MagicMock()
        retrieval.get_item_metadata.return_value = json.dumps(_raw("ABCD1234"))

        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"), \
             patch("zotero_mcp.cli_standalone._import_tools",
                   return_value=(MagicMock(), retrieval, MagicMock(), MagicMock(), MagicMock())):
            cmd_get(args)

        # --json upgrades the default markdown format to json for this command.
        assert retrieval.get_item_metadata.call_args.kwargs["format"] == "json"
        payload = json.loads(capsys.readouterr().out)
        assert payload["data"]["data"]["itemType"] == "journalArticle"

    def test_an_explicit_format_is_not_overridden(self, capsys):
        args = _args(subcommand="metadata", item_key="ABCD1234",
                     no_abstract=False, output_format="bibtex")
        retrieval = MagicMock()
        retrieval.get_item_metadata.return_value = "@article{x,}"

        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"), \
             patch("zotero_mcp.cli_standalone._import_tools",
                   return_value=(MagicMock(), retrieval, MagicMock(), MagicMock(), MagicMock())):
            cmd_get(args)

        assert retrieval.get_item_metadata.call_args.kwargs["format"] == "bibtex"
        assert json.loads(capsys.readouterr().out)["data"]["text"] == "@article{x,}"

    def test_fulltext_reports_its_size(self, capsys):
        args = _args(subcommand="fulltext", item_key="ABCD1234")
        retrieval = MagicMock()
        retrieval.get_item_fulltext.return_value = "hello world"

        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"), \
             patch("zotero_mcp.cli_standalone._import_tools",
                   return_value=(MagicMock(), retrieval, MagicMock(), MagicMock(), MagicMock())):
            cmd_get(args)

        data = json.loads(capsys.readouterr().out)["data"]
        assert data["chars"] == 11
        assert data["text"] == "hello world"

    def test_unknown_subcommand_is_a_json_error(self, capsys):
        args = _args(subcommand="nonsense")
        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"), \
             patch("zotero_mcp.cli_standalone._import_tools",
                   return_value=(MagicMock(),) * 5):
            with pytest.raises(SystemExit) as exc:
                cmd_get(args)

        assert exc.value.code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is False
        assert payload["error"]["code"] == "unknown_subcommand"


# ---------------------------------------------------------------------------
# Flag placement
# ---------------------------------------------------------------------------

class TestFlagPlacement:
    @pytest.mark.parametrize("argv", [
        ["--json", "search", "x"],
        ["search", "--json", "x"],
        ["--json", "search", "--json", "x"],
    ])
    def test_json_is_accepted_on_either_side_of_the_subcommand(self, argv):
        """Both placements read naturally, and a subparser default must not
        undo a flag given before the subcommand name."""
        assert build_parser().parse_args(argv).json_out is True

    def test_absent_flag_means_markdown(self):
        assert build_parser().parse_args(["search", "x"]).json_out is False

    def test_verbose_also_works_on_either_side(self):
        parser = build_parser()
        assert parser.parse_args(["-v", "search", "x"]).verbose is True
        assert parser.parse_args(["search", "-v", "x"]).verbose is True
        assert parser.parse_args(["search", "x"]).verbose is False
