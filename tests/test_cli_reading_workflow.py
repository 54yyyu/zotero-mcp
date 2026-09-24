"""Reading and annotating a paper end to end from `zotero-cli`.

Dogfooding the CLI on a real paper (read it, then highlight and box its key
passages) turned up a cluster of problems no single-command test caught:

* a failed write came back as `ok: true`, so the agent believed it had worked;
* area boxes, tags and layout detection existed only as MCP tools;
* 48 highlights meant 48 processes, most of the time spent starting Python;
* long or fuzzy-matched highlights covered whole spans, spilling onto
  neighbouring words;
* tables typeset with horizontal rules only, and figures made of two panels,
  were missed or half-boxed by layout detection.

These pin the fixes.
"""

import argparse
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from zotero_mcp import cli_standalone
from zotero_mcp.cli_json import CliError
from zotero_mcp.cli_standalone import (
    ZOTERO_COLORS,
    _cli_vocabulary,
    _out,
    _parse_pages,
    _parse_rect,
    _read_annotation_specs,
    _reports_failure,
    _resolve_color,
    build_parser,
)

fitz = pytest.importorskip("fitz")


def _args(**kwargs):
    defaults = dict(verbose=False, json_out=False)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _with_tools(annotations):
    """Patch the handler's environment setup and tool import."""
    return (
        patch("zotero_mcp.cli_standalone.setup_zotero_environment"),
        patch("zotero_mcp.cli_standalone._import_tools",
              return_value=(MagicMock(), MagicMock(), annotations, MagicMock(), MagicMock())),
    )


# ---------------------------------------------------------------------------
# Failure reporting
# ---------------------------------------------------------------------------

class TestFailureReporting:
    @pytest.mark.parametrize("text", [
        "Error: Cannot perform write operations in local-only mode",
        "Error creating area annotation: boom",
        "Failed to create annotation: {}",
        "Could not find text on page 9",
        "Cannot write: read-only",
        "Invalid library_type 'bogus'. Must be 'user', 'group', or 'feed'.",
        "Group 'missing' not found. Available groups: 123",
        "Feed with libraryID 'missing' not found. Available feeds: 10",
        "Personal library id '99' is not addressable in local mode.",
        "\n**Error:** something",
    ])
    def test_failure_prose_is_recognised(self, text):
        assert _reports_failure(text)

    @pytest.mark.parametrize("text", [
        "Successfully created highlight annotation\n\n**Annotation Key:** ABCD1234",
        "No items found for 'x'",
        "Successfully trashed annotation K (recoverable). Error count: 0",
        "Errors are rare in this paper",
    ])
    def test_success_prose_is_not(self, text):
        assert not _reports_failure(text)

    def test_json_mode_turns_failure_prose_into_an_error_envelope(self, capsys):
        """The dogfood bug: this exact message arrived as ok: true."""
        with pytest.raises(SystemExit) as exc:
            _out(_args(json_out=True), "annotations create",
                 text="Error: Cannot perform write operations in local-only mode")
        assert exc.value.code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is False
        assert payload["error"]["code"] == "tool_error"
        assert "local-only mode" in payload["error"]["message"]

    def test_invalid_library_type_is_an_error_envelope(self, capsys):
        text = "Invalid library_type 'bogus'. Must be 'user', 'group', or 'feed'."

        with pytest.raises(SystemExit) as exc:
            _out(_args(json_out=True), "switch-library", text=text)

        assert exc.value.code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is False
        assert payload["error"]["code"] == "tool_error"
        assert "library_type" in payload["error"]["message"]

    def test_markdown_mode_sends_failures_to_stderr_with_exit_1(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _out(_args(), "annotations create", text="Failed to create annotation: {}")
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Failed to create annotation" in captured.err

    def test_success_is_unchanged(self, capsys):
        _out(_args(json_out=True), "annotations create", text="Successfully created")
        assert json.loads(capsys.readouterr().out)["ok"] is True

    def test_structured_data_is_never_second_guessed(self, capsys):
        """A command that built real data owns its outcome."""
        _out(_args(json_out=True), "read", data={"text": "Error: in the paper"}, text="Error: x")
        assert json.loads(capsys.readouterr().out)["ok"] is True


class TestCliVocabulary:
    def test_mcp_tool_names_become_cli_commands(self):
        text = "Consider using zotero_semantic_search to find specific content."
        assert _cli_vocabulary(text) == (
            "Consider using `zotero-cli search --mode semantic` to find specific content."
        )

    def test_text_without_tool_names_is_untouched(self):
        assert _cli_vocabulary("plain words") == "plain words"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

class TestArgumentHelpers:
    @pytest.mark.parametrize("value", ["0.1,0.2,0.3,0.4", "[0.1, 0.2, 0.3, 0.4]", [0.1, 0.2, 0.3, 0.4]])
    def test_rect_forms(self, value):
        assert _parse_rect(value) == [0.1, 0.2, 0.3, 0.4]

    @pytest.mark.parametrize("value", ["0.1,0.2,0.3", "a,b,c,d", "", [1, 2]])
    def test_bad_rects_are_usage_errors(self, value):
        with pytest.raises(CliError) as exc:
            _parse_rect(value)
        assert exc.value.code == "bad_rect"

    def test_absent_rect_is_none(self):
        assert _parse_rect(None) is None

    @pytest.mark.parametrize("value,expected", [
        ("all", None), (None, None), ("3", [3]), ("3-5", [3, 4, 5]), ("1,4,6-7", [1, 4, 6, 7]),
    ])
    def test_pages(self, value, expected):
        assert _parse_pages(value) == expected

    @pytest.mark.parametrize("value", ["0", "x", "3-", ""])
    def test_bad_pages(self, value):
        with pytest.raises(CliError):
            _parse_pages(value)

    def test_color_names_resolve_to_zotero_palette(self):
        assert _resolve_color("Blue") == ZOTERO_COLORS["blue"] == "#2ea8e5"
        assert _resolve_color("#123456") == "#123456"
        assert len(ZOTERO_COLORS) == 8

    def test_create_accepts_rect_without_text(self):
        parsed = build_parser().parse_args(
            ["annotations", "create", "--attachment-key", "A1", "--page", "3",
             "--rect", "0.1,0.2,0.3,0.4", "--tags", "x,y", "--color", "green"])
        assert parsed.text is None and parsed.rect == "0.1,0.2,0.3,0.4"

    def test_batch_and_layout_parse(self):
        parser = build_parser()
        batch = parser.parse_args(["annotations", "batch", "--attachment-key", "A1", "--dry-run"])
        assert batch.subcommand == "batch" and batch.file == "-" and batch.dry_run
        layout = parser.parse_args(["layout", "A1", "--pages", "3-9"])
        assert layout.command == "layout" and layout.pages == "3-9"


# ---------------------------------------------------------------------------
# annotations create / batch / layout handlers
# ---------------------------------------------------------------------------

class TestCreateForwarding:
    def test_rect_tags_and_color_name_reach_the_tool(self, capsys):
        annotations = MagicMock()
        annotations.create_annotation.return_value = "Successfully created area annotation"
        args = _args(subcommand="create", attachment_key="A1", page=3, text=None,
                     rect="0.1,0.2,0.3,0.4", comment="c", color="purple", tags="a,b")
        env, tools = _with_tools(annotations)
        with env, tools:
            cli_standalone.cmd_annotations(args)
        kwargs = annotations.create_annotation.call_args.kwargs
        assert kwargs["rect"] == [0.1, 0.2, 0.3, 0.4]
        assert kwargs["text"] is None
        assert kwargs["color"] == "#a28ae5"
        assert kwargs["tags"] == ["a", "b"]


class TestNoteForwarding:
    def test_note_point_reaches_the_tool(self, capsys):
        annotations = MagicMock()
        annotations.create_annotation.return_value = "Successfully created sticky note"
        args = _args(subcommand="create", attachment_key="A1", page=1, text=None, rect=None,
                     note="0.1,0.2", comment="c", color="orange", tags=None)
        env, tools = _with_tools(annotations)
        with env, tools:
            cli_standalone.cmd_annotations(args)
        assert annotations.create_annotation.call_args.kwargs["note"] == [0.1, 0.2]

    def test_note_must_be_two_numbers(self):
        with pytest.raises(CliError, match="--note must be two numbers"):
            _parse_rect("0.1,0.2,0.3", size=2, flag="--note")


class TestSpecReader:
    def test_json_lines(self, tmp_path):
        path = tmp_path / "specs.jsonl"
        path.write_text('{"page": 1, "text": "a"}\n\n{"page": 2, "rect": [0, 0, 1, 1]}\n')
        assert [s["page"] for s in _read_annotation_specs(str(path))] == [1, 2]

    def test_json_array(self, tmp_path):
        path = tmp_path / "specs.json"
        path.write_text('[{"page": 1, "text": "a"}]')
        assert _read_annotation_specs(str(path)) == [{"page": 1, "text": "a"}]

    def test_bad_line_names_its_number(self, tmp_path):
        path = tmp_path / "specs.jsonl"
        path.write_text('{"page": 1, "text": "a"}\n{nope}\n')
        with pytest.raises(CliError) as exc:
            _read_annotation_specs(str(path))
        assert exc.value.code == "bad_json" and "Line 2" in str(exc.value)

    def test_empty_and_non_object(self, tmp_path):
        empty = tmp_path / "e.jsonl"
        empty.write_text("  \n")
        with pytest.raises(CliError):
            _read_annotation_specs(str(empty))
        scalars = tmp_path / "s.json"
        scalars.write_text("[1, 2]")
        with pytest.raises(CliError):
            _read_annotation_specs(str(scalars))


class TestBatch:
    def _run(self, tmp_path, capsys, specs, outcomes, json_out=True, dry_run=False):
        path = tmp_path / "specs.jsonl"
        path.write_text("\n".join(json.dumps(s) for s in specs))
        annotations = MagicMock()
        annotations.create_annotations.side_effect = outcomes
        args = _args(subcommand="batch", attachment_key="ATT00001", file=str(path),
                     dry_run=dry_run, json_out=json_out)
        env, tools = _with_tools(annotations)
        code = 0
        with env, tools:
            try:
                cli_standalone.cmd_annotations(args)
            except SystemExit as exc:
                code = exc.code
        return annotations, capsys.readouterr().out, code

    def test_specs_are_grouped_by_attachment_and_reported_in_order(self, tmp_path, capsys):
        def outcomes(key, specs, **kwargs):
            if key == "ATT00001":
                return [
                    {"index": 1, "page": 1, "type": "highlight", "ok": True, "annotation_key": "AAAA1111"},
                    {"index": 2, "page": 9, "type": "highlight", "ok": False,
                     "error": "Error: Could not find text on page 9"},
                ]
            return [{"index": 1, "page": 3, "type": "area", "ok": True, "annotation_key": "BBBB2222"}]

        specs = [
            {"page": 1, "text": "one", "color": "red", "tags": ["t"]},
            {"page": 3, "rect": "0.1,0.1,0.5,0.2", "attachment_key": "OTHER001"},
            {"page": 9, "text": "missing"},
        ]
        annotations, out, code = self._run(tmp_path, capsys, specs, outcomes)

        assert code == 1  # one failure
        data = json.loads(out)["data"]
        assert (data["succeeded"], data["failed"], data["dry_run"]) == (2, 1, False)
        assert [(r["index"], r["ok"]) for r in data["results"]] == [(1, True), (2, True), (3, False)]
        assert data["results"][1]["annotation_key"] == "BBBB2222"
        assert "Could not find" in data["results"][2]["error"]

        calls = {c.args[0]: c.args[1] for c in annotations.create_annotations.call_args_list}
        assert [s["page"] for s in calls["ATT00001"]] == [1, 9]
        assert calls["ATT00001"][0]["color"] == "#ff6666"
        assert calls["OTHER001"][0]["rect"] == [0.1, 0.1, 0.5, 0.2]
        assert annotations.create_annotations.call_count == 2

    def test_dry_run_is_passed_through(self, tmp_path, capsys):
        outcome = [{"index": 1, "page": 2, "type": "highlight", "ok": True,
                    "page_found": 3, "matched_text": "the words"}]
        annotations, out, code = self._run(
            tmp_path, capsys, [{"page": 2, "text": "the words"}],
            lambda key, specs, **kwargs: outcome, json_out=False, dry_run=True)
        assert annotations.create_annotations.call_args.kwargs["dry_run"] is True
        assert code == 0
        assert "(found on p3): the words" in out and "Located 1/1" in out

    def test_all_succeeded_exits_zero_with_a_summary(self, tmp_path, capsys):
        outcome = [{"index": 1, "page": 1, "type": "highlight", "ok": True, "annotation_key": "DDDD4444"}]
        _annotations, out, code = self._run(
            tmp_path, capsys, [{"page": 1, "text": "x"}],
            lambda key, specs, **kwargs: outcome, json_out=False)
        assert code == 0
        assert "Created 1/1" in out and "DDDD4444" in out


class TestLayoutCommand:
    def test_json_lists_regions_with_paste_ready_rects(self, capsys):
        annotations = MagicMock()
        annotations.detect_layouts.return_value = ([
            {"page": 6, "pageLabel": "6", "warnings": [], "regions": [{
                "region_id": 1, "source": "table", "bbox": [0.1937, 0.143, 0.6125, 0.0927],
                "caption_label": "Table 1", "caption_text": "Table 1: ...", "confidence": "high",
            }]},
            {"page": 7, "pageLabel": "7", "warnings": [], "regions": []},
        ], "paper.pdf", None)
        env, tools = _with_tools(annotations)
        with env, tools:
            cli_standalone.cmd_layout(_args(attachment_key="A1", pages="6-7", json_out=True))
        assert annotations.detect_layouts.call_args.args == ("A1", [6, 7])
        data = json.loads(capsys.readouterr().out)["data"]
        assert data["pages_scanned"] == [6, 7]
        assert data["regions"][0]["rect_arg"] == "0.1937,0.1430,0.6125,0.0927"
        assert data["regions"][0]["caption_label"] == "Table 1"

    def test_errors_use_the_failure_envelope(self, capsys):
        annotations = MagicMock()
        annotations.detect_layouts.return_value = ([], "", "Error: Item A1 is not an attachment")
        env, tools = _with_tools(annotations)
        with env, tools, pytest.raises(SystemExit):
            cli_standalone.cmd_layout(_args(attachment_key="A1", pages="all", json_out=True))
        assert json.loads(capsys.readouterr().out)["ok"] is False


class TestNotesListJson:
    def test_notes_listed_in_markdown_are_projected(self, capsys):
        """`notes list` found the note and `--json notes list` said count 0."""
        annotations = MagicMock()
        annotations.get_notes.return_value = (
            "# Notes for Item: ZJDYZVW4\n\n"
            "## Note 1 (from \"Attention is All you Need\")\n"
            "**Key:** CHENHXNA\n"
            "**Tags:** `reading`\n\n"
            "## Note 2 (from \"Attention is All you Need\")\n"
            "**Key:** `NOTE0002`\n"
        )
        backend = MagicMock()
        backend.get_items.side_effect = lambda keys: {
            key: {"key": key, "data": {"key": key, "itemType": "note",
                                       "note": f"<p>{key}</p>", "tags": []}}
            for key in keys
        }
        args = _args(subcommand="list", item_key="ZJDYZVW4", limit=20, full=False,
                     raw_html=False, json_out=True)
        env, tools = _with_tools(annotations)
        with env, tools, patch("zotero_mcp.cli_standalone._read_backend", return_value=backend):
            cli_standalone.cmd_notes(args)

        assert backend.get_items.call_args.args[0] == ["CHENHXNA", "NOTE0002"]
        data = json.loads(capsys.readouterr().out)["data"]
        assert data["count"] == 2


# ---------------------------------------------------------------------------
# Highlight geometry on a real PDF
# ---------------------------------------------------------------------------

LINE_1 = "Recurrent models typically factor computation along the symbol positions of input."
LINE_2 = "This inherently sequential nature precludes parallelization within training examples."
LINE_3 = "Memory constraints limit batching across examples at longer sequence lengths today."


@pytest.fixture
def prose_pdf(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i, line in enumerate((LINE_1, LINE_2, LINE_3)):
        page.insert_text((72, 100 + 14 * i), line, fontsize=10)
    path = str(tmp_path / "prose.pdf")
    doc.save(path)
    doc.close()
    return path


class TestHighlightClipping:
    def test_mid_line_match_covers_only_the_matched_words(self, prose_pdf):
        from zotero_mcp.pdf_utils import find_text_position, text_in_rects

        wanted = "sequential nature precludes"
        result = find_text_position(prose_pdf, 1, wanted)
        assert len(result["rects"]) == 1
        assert text_in_rects(prose_pdf, 0, result["rects"]) == wanted

    def test_long_match_across_lines_starts_and_ends_on_the_matched_words(self, prose_pdf):
        """Over 100 characters takes the anchor path, which used to box every
        span it touched -- here, all of lines 1 and 3."""
        from zotero_mcp.pdf_utils import find_text_position, text_in_rects

        wanted = ("positions of input. This inherently sequential nature precludes "
                  "parallelization within training examples. Memory constraints")
        assert len(wanted) > 100
        result = find_text_position(prose_pdf, 1, wanted)
        assert len(result["rects"]) == 3
        assert text_in_rects(prose_pdf, 0, result["rects"]) == wanted

    def test_fuzzy_match_is_clipped_too(self, prose_pdf):
        from zotero_mcp.pdf_utils import find_text_position, text_in_rects

        # One character off, so exact search fails and fuzzy matching runs.
        result = find_text_position(prose_pdf, 1, "inherently sequentail nature")
        covered = text_in_rects(prose_pdf, 0, result["rects"])
        assert "Recurrent" not in covered and "examples" not in covered
        assert "sequential" in covered

    def test_dry_run_reports_readable_matches_without_writing(self, prose_pdf, monkeypatch):
        from zotero_mcp.tools import annotations

        monkeypatch.setattr(annotations, "_fetch_attachment_file",
                            lambda _key, _tmp, **_kw: (prose_pdf, "prose.pdf", "pdf", None))
        monkeypatch.setattr(annotations, "_get_note_write_client",
                            lambda _op: pytest.fail("a dry run must not need write access"))
        results = annotations.create_annotations("ATT00001", [
            {"page": 1, "text": "sequential nature precludes"},
            {"page": 1, "text": "a sentence that is not in this document at all"},
            {"page": 1, "rect": [0.1, 0.1, 0.2, 0.2]},
        ], ctx=MagicMock(), dry_run=True)

        assert results[0]["ok"] and results[0]["matched_text"] == "sequential nature precludes"
        assert results[0]["page_found"] == 1
        assert not results[1]["ok"] and "Could not find text" in results[1]["error"]
        assert results[2]["ok"] and results[2]["type"] == "area"
        assert not any("annotation_key" in r for r in results)


class TestCreateAnnotations:
    """One fetch of the PDF and one write per 50 annotations, however many specs."""

    @pytest.fixture
    def writer(self, prose_pdf, monkeypatch):
        from zotero_mcp.tools import annotations

        fake = MagicMock()
        fake.create_items.side_effect = lambda items: {
            "success": {str(i): f"KEY{len(fake.create_items.call_args_list):02d}{i:02d}"
                        for i in range(len(items))},
            "failed": {},
        }
        fetches = []

        def fetch(key, tmpdir, **kwargs):
            fetches.append(key)
            return prose_pdf, "prose.pdf", "pdf", None

        monkeypatch.setattr(annotations, "_fetch_attachment_file", fetch)
        monkeypatch.setattr(annotations, "_get_note_write_client", lambda _op: (fake, None))
        return annotations, fake, fetches

    def test_placed_specs_are_written_in_one_request(self, writer):
        annotations, fake, fetches = writer
        results = annotations.create_annotations("ATT00001", [
            {"page": 1, "text": "sequential nature precludes", "color": "#2ea8e5", "tags": ["x"]},
            {"page": 1, "text": "nowhere in this document at all, not even close"},
            {"page": 1, "rect": [0.1, 0.1, 0.2, 0.2], "comment": "box"},
            {"page": 0, "text": "bad page"},
        ], ctx=MagicMock())

        assert fetches == ["ATT00001"]
        assert fake.create_items.call_count == 1
        payloads = fake.create_items.call_args.args[0]
        assert [p["annotationType"] for p in payloads] == ["highlight", "image"]
        assert payloads[0]["annotationColor"] == "#2ea8e5" and payloads[0]["tags"] == [{"tag": "x"}]
        assert payloads[1]["annotationComment"] == "box"
        assert [r["ok"] for r in results] == [True, False, True, False]
        assert results[0]["annotation_key"] == "KEY0100" and results[2]["annotation_key"] == "KEY0101"
        assert "positive integer" in results[3]["error"]

    def test_annotation_type_is_the_first_annotation_property(self, writer):
        """Zotero's local API answers 400 "annotationType must be set before
        other annotation properties" when any annotation* key precedes it.
        Every payload in the first live batch was refused this way."""
        annotations, fake, _fetches = writer
        annotations.create_annotations("ATT00001", [
            {"page": 1, "text": "sequential nature precludes", "comment": "c", "color": "#ff6666"},
            {"page": 1, "rect": [0.1, 0.1, 0.2, 0.2], "comment": "c"},
        ], ctx=MagicMock())
        for payload in fake.create_items.call_args.args[0]:
            annotation_keys = [k for k in payload if k.startswith("annotation")]
            assert annotation_keys[0] == "annotationType", annotation_keys

    def test_more_than_fifty_are_chunked(self, writer):
        annotations, fake, _fetches = writer
        specs = [{"page": 1, "rect": [0.1, 0.1, 0.1, 0.1]}] * 51
        results = annotations.create_annotations("ATT00001", specs, ctx=MagicMock())
        assert [len(c.args[0]) for c in fake.create_items.call_args_list] == [50, 1]
        assert all(r["ok"] for r in results)

    def test_a_refused_item_fails_only_itself(self, writer):
        annotations, fake, _fetches = writer
        fake.create_items.side_effect = lambda items: {
            "success": {"0": "GOOD0001"}, "failed": {"1": {"code": 400, "message": "bad color"}},
        }
        results = annotations.create_annotations("ATT00001", [
            {"page": 1, "rect": [0.1, 0.1, 0.1, 0.1]},
            {"page": 1, "rect": [0.2, 0.2, 0.1, 0.1], "color": "not-a-color"},
        ], ctx=MagicMock())
        assert results[0]["ok"] and not results[1]["ok"]
        assert "bad color" in results[1]["error"]

    def test_sticky_note_is_a_22pt_square_centered_on_the_point(self, writer):
        """Zotero's reader draws a note as a 22 pt square (PDF_NOTE_DIMENSIONS)."""
        annotations, fake, _fetches = writer
        results = annotations.create_annotations("ATT00001", [
            {"page": 1, "note": [0.5, 0.25], "comment": "Read this first", "color": "#f19837"},
        ], ctx=MagicMock())

        assert results[0]["ok"] and results[0]["type"] == "note"
        (payload,) = fake.create_items.call_args.args[0]
        assert [k for k in payload if k.startswith("annotation")][0] == "annotationType"
        assert payload["annotationType"] == "note"
        assert payload["annotationComment"] == "Read this first"
        position = json.loads(payload["annotationPosition"])
        assert position["pageIndex"] == 0
        # Point (306, 198) from the top of a 612 x 792 page is y = 594 in PDF space.
        assert position["rects"] == [[295.0, 583.0, 317.0, 605.0]]

    @pytest.mark.parametrize("spec, error", [
        ({"page": 1, "note": [0.5, 0.5]}, "needs a comment"),
        ({"page": 1, "note": [0.5, 1.5], "comment": "c"}, "normalized to 0-1"),
        ({"page": 1, "note": [0.5], "comment": "c"}, "normalized to 0-1"),
        ({"page": 1, "note": [0.5, 0.5], "rect": [0.1, 0.1, 0.1, 0.1], "comment": "c"}, "only one of"),
    ])
    def test_bad_note_specs_fail_alone(self, writer, spec, error):
        annotations, _fake, _fetches = writer
        (result,) = annotations.create_annotations("ATT00001", [spec], ctx=MagicMock())
        assert not result["ok"] and error in result["error"]

    def test_no_write_access_fails_every_spec_with_the_reason(self, prose_pdf, monkeypatch):
        from zotero_mcp.tools import annotations

        monkeypatch.setattr(annotations, "_get_note_write_client",
                            lambda _op: (None, "Error: Cannot perform write operations"))
        results = annotations.create_annotations(
            "ATT00001", [{"page": 1, "text": "a"}, {"page": 1, "text": "b"}], ctx=MagicMock())
        assert [r["error"] for r in results] == ["Error: Cannot perform write operations"] * 2


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def test_rule_only_table_is_detected_with_its_caption(tmp_path):
    """Booktabs tables have no vertical lines, so find_tables finds nothing."""
    from zotero_mcp.pdf_layout import detect_page_regions

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((108, 90), "Table 1: Maximum path lengths for different layer types.", fontsize=9)
    for y in (110, 128, 190):
        page.draw_line((118, y), (494, y), width=0.6)
    for i, row in enumerate(("Self-Attention  O(n2 d)  O(1)", "Recurrent  O(n d2)  O(n)",
                             "Convolutional  O(k n d2)  O(1)")):
        page.insert_text((124, 145 + 14 * i), row, fontsize=9)
    page.draw_line((108, 700), (250, 700), width=0.4)  # a lone footnote rule
    path = str(tmp_path / "table.pdf")
    doc.save(path)
    doc.close()

    regions = detect_page_regions(path, 1)["regions"]
    tables = [r for r in regions if r["source"] == "table"]
    assert len(tables) == 1
    x, y, w, h = tables[0]["bbox"]
    assert abs(x - 118 / 612) < 0.01 and abs((x + w) - 494 / 612) < 0.01
    assert abs(y - 110 / 792) < 0.01 and abs((y + h) - 190 / 792) < 0.01
    assert tables[0]["caption_label"] == "Table 1"


def test_ruled_table_boxes_ignores_partial_and_sparse_rules():
    from zotero_mcp.pdf_layout import _ruled_table_boxes

    def rule(x0, x1, y):
        return {"rect": fitz.Rect(x0, y, x1, y)}

    drawings = [
        rule(100, 500, 100), rule(100, 500, 100.2),  # drawn twice
        rule(300, 400, 115),                          # cmidrule
        rule(100, 500, 130),
        rule(101, 499, 180),
        rule(100, 250, 700),                          # footnote
        rule(100, 500, 400), rule(100, 500, 420),     # only two rules
    ]
    assert _ruled_table_boxes(drawings, 612, 792) == [(100, 100, 500, 180)]


def test_side_by_side_panels_join_the_captioned_figure():
    from zotero_mcp.pdf_layout import _absorb_uncaptioned_panels

    caption = {"label": "Figure 2", "kind": "figure", "text": "Figure 2: (left) ... (right) ...",
               "bbox": [0.176, 0.346, 0.648, 0.027]}
    right = {"source": "image", "bbox": [0.5666, 0.1044, 0.1965, 0.2331],
             "caption_label": "Figure 2", "caption_text": caption["text"], "confidence": "high"}
    left = {"source": "image", "bbox": [0.2859, 0.1187, 0.1047, 0.1607],
            "caption_label": None, "caption_text": None, "confidence": "low"}
    elsewhere = {"source": "image", "bbox": [0.2, 0.7, 0.2, 0.1],
                 "caption_label": None, "caption_text": None, "confidence": "low"}

    result = _absorb_uncaptioned_panels([right, left, elsewhere], [caption])

    assert len(result) == 2
    merged = result[0]
    assert merged["source"] == "merged" and merged["caption_label"] == "Figure 2"
    x, y, w, h = merged["bbox"]
    assert abs(x - 0.2859) < 1e-6 and abs((x + w) - 0.7631) < 1e-6
    assert result[1] is not elsewhere and result[1]["bbox"] == elsewhere["bbox"]


def test_table_captions_do_not_absorb_panels():
    from zotero_mcp.pdf_layout import _absorb_uncaptioned_panels

    caption = {"label": "Table 2", "kind": "table", "text": "Table 2", "bbox": [0.1, 0.5, 0.8, 0.02]}
    table = {"source": "table", "bbox": [0.1, 0.3, 0.4, 0.15], "caption_label": "Table 2"}
    other = {"source": "image", "bbox": [0.55, 0.3, 0.3, 0.15], "caption_label": None}
    assert len(_absorb_uncaptioned_panels([table, other], [caption])) == 2


# ---------------------------------------------------------------------------
# Math detection
# ---------------------------------------------------------------------------

class _FakeMathPage:
    """Just enough of a fitz page for scan_math: fonts, text blocks, size."""

    def __init__(self, fonts, blocks, width=612, height=792):
        self._fonts = fonts
        self._blocks = blocks
        self.rect = fitz.Rect(0, 0, width, height)
        self.text_extractions = 0

    def get_fonts(self):
        return [(1, "pfb", "Type1", name, "F1", "") for name in self._fonts]

    def get_text(self, kind, flags=0):
        self.text_extractions += 1
        return {"blocks": self._blocks}


def _block(*lines):
    return {"lines": [{"bbox": bbox, "spans": [{"text": t, "font": f} for t, f in spans]}
                      for bbox, spans in lines]}


class TestScanMath:
    def test_pages_without_math_fonts_are_skipped_before_extracting_text(self):
        from zotero_mcp.pdf_layout import scan_math

        page = _FakeMathPage(["NimbusRomNo9L-Regu"], [])
        assert scan_math(page) == ([], 0)
        assert page.text_extractions == 0

    def test_split_display_joins_into_one_numbered_equation(self):
        """AIAYN's Eq. (1): the fraction's denominator is its own block, and
        the equation number sits in a third."""
        from zotero_mcp.pdf_layout import scan_math

        page = _FakeMathPage(["ABCDEF+CMMI10", "CMR10", "CMSY10", "NimbusRomNo9L-Regu"], [
            _block(((72, 100, 540, 112), [("We compute the matrix of outputs as ", "NimbusRomNo9L-Regu"),
                                          ("Q", "CMMI10"), (", K", "CMMI10")])),
            _block(((220, 400, 377, 412), [("Attention(", "CMR10"), ("Q, K, V", "CMMI10"),
                                           (") = softmax(", "CMR10"), ("QK", "CMMI10")])),
            _block(((358, 407, 390, 420), [("√", "CMSY10"), ("dk", "CMMI10")]),
                   ((493, 407, 504, 419), [("(1)", "NimbusRomNo9L-Regu")])),
        ])
        equations, inline = scan_math(page)
        assert len(equations) == 1
        assert equations[0]["label"] == "(1)"
        x0, y0, x1, y1 = equations[0]["bbox"]
        assert x0 < 220 and x1 > 390 and y0 < 400 and y1 > 420
        assert inline == 3  # "Q" and ", K" in the prose block

    def test_displays_in_different_columns_stay_apart(self):
        from zotero_mcp.pdf_layout import scan_math

        page = _FakeMathPage(["CMMI10"], [
            _block(((60, 300, 280, 312), [("x = y + z", "CMMI10")])),
            _block(((330, 300, 550, 312), [("a = b + c", "CMMI10")])),
        ])
        equations, _ = scan_math(page)
        assert len(equations) == 2

    def test_numbers_go_to_the_display_in_their_own_column(self):
        """Two columns can put displays on one band; the right column's
        equation was labelled with the left column's number."""
        from zotero_mcp.pdf_layout import scan_math

        page = _FakeMathPage(["CMMI10"], [
            _block(((60, 500, 250, 512), [("p = x Q", "CMMI10")]),
                   ((268, 500, 282, 512), [("(14)", "Times")])),
            _block(((320, 500, 520, 512), [("s = y + x", "CMMI10")]),
                   ((536, 500, 550, 512), [("(17)", "Times")])),
        ])
        equations, _ = scan_math(page)
        assert sorted((eq["bbox"][0] < 300, eq["label"]) for eq in equations) == [
            (False, "(17)"), (True, "(14)"),
        ]

    def test_math_heavy_prose_is_not_a_display(self):
        from zotero_mcp.pdf_layout import scan_math

        page = _FakeMathPage(["CMMI10", "Times"], [
            _block(((72, 100, 540, 112), [("where ", "Times"), ("d", "CMMI10"),
                                          (" is the representation dimension of every layer", "Times")])),
        ])
        assert scan_math(page) == ([], 1)


def test_fragments_nested_in_a_table_are_dropped():
    """EGNN's Table 1 has a rule under every row; each band detected as a
    table of its own inside the real one."""
    from zotero_mcp.pdf_layout import _merge_candidate_regions

    regions = _merge_candidate_regions([
        {"source": "table", "bbox": [0.108, 0.104, 0.76, 0.025]},
        {"source": "table", "bbox": [0.091, 0.073, 0.794, 0.129]},
        {"source": "drawing", "bbox": [0.2, 0.08, 0.3, 0.05]},
        {"source": "image", "bbox": [0.1, 0.5, 0.3, 0.2]},
    ])
    assert [(r["source"], r["bbox"]) for r in regions] == [
        ("table", [0.091, 0.073, 0.794, 0.129]),
        ("image", [0.1, 0.5, 0.3, 0.2]),
    ]


# ---------------------------------------------------------------------------
# Reading pages as text with flags, or as images
# ---------------------------------------------------------------------------

@pytest.fixture
def two_page_pdf(tmp_path, monkeypatch):
    from zotero_mcp.tools import read_pdf

    doc = fitz.open()
    for i in range(2):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100), f"Page {i + 1} prose.", fontsize=11)
    path = str(tmp_path / "two.pdf")
    doc.save(path)
    doc.close()
    monkeypatch.setattr(read_pdf, "_get_pdf_path", lambda _key, _ctx: (path, "Paper", False))
    return path


class TestReadFlags:
    def _extracted(self, pages):
        return argparse.Namespace(page_numbers=tuple(range(len(pages))), pages=pages, needs_ocr=())

    def test_flags_name_equations_captions_and_inline_math(self, two_page_pdf, monkeypatch):
        from zotero_mcp import pdf_layout
        from zotero_mcp.tools import read_pdf

        scans = iter([
            ([{"bbox": (0, 0, 1, 1), "label": "(1)"}, {"bbox": (0, 0, 1, 1), "label": None}], 40),
            ([], 3),
        ])
        monkeypatch.setattr(pdf_layout, "scan_math", lambda _page: next(scans))
        flags = read_pdf._garbled_content_flags(two_page_pdf, self._extracted([
            "Some prose.\nFigure 2: (left) a diagram.",
            "Table 3: Variations on the model.\nTable 3 rows (A) vary heads.",
        ]))
        assert flags == {
            0: "> **Garbled in this text:** Equation (1), 1 unnumbered equation, Figure 2, inline math",
            1: "> **Garbled in this text:** Table 3",
        }

    def test_unreadable_file_means_no_flags_not_a_failed_read(self):
        from zotero_mcp.tools import read_pdf

        assert read_pdf._garbled_content_flags("/nonexistent.pdf", self._extracted(["x"])) == {}

    @pytest.mark.parametrize("surface,advice", [
        ("mcp", "format='image'"),
        ("cli", "zotero-cli read KEY00001 --start-page N --format image"),
    ])
    def test_advice_matches_the_surface(self, two_page_pdf, monkeypatch, surface, advice):
        from zotero_mcp import pdf_layout
        from zotero_mcp.tools import read_pdf

        monkeypatch.setattr(pdf_layout, "scan_math",
                            lambda _page: ([{"bbox": (0, 0, 1, 1), "label": "(2)"}], 0))
        text = read_pdf.read_pdf_text("KEY00001", 1, 2, ctx=MagicMock(), surface=surface)
        assert "> **Garbled in this text:** Equation (2)" in text
        assert advice in text

    def test_clean_pages_carry_no_advice(self, two_page_pdf):
        from zotero_mcp.tools import read_pdf

        text = read_pdf.read_pdf_text("KEY00001", 1, 2, ctx=MagicMock())
        assert "Garbled" not in text and "format='image'" not in text


class TestRenderPages:
    def test_pages_render_to_png_within_the_size_cap(self, two_page_pdf):
        from zotero_mcp.tools import read_pdf

        header, pages = read_pdf.render_pdf_pages("KEY00001", 1, 5, ctx=MagicMock())
        assert [p["page"] for p in pages] == [1, 2]
        assert "past the last page" in header
        for page in pages:
            assert page["png"].startswith(b"\x89PNG")
            assert max(page["width"], page["height"]) <= read_pdf._IMAGE_MAX_EDGE

    def test_rect_zooms_into_one_region(self, two_page_pdf):
        from zotero_mcp.tools import read_pdf

        _header, full = read_pdf.render_pdf_pages("KEY00001", 1, ctx=MagicMock())
        header, crop = read_pdf.render_pdf_pages("KEY00001", 1, rect="[0.1, 0.1, 0.4, 0.05]", ctx=MagicMock())
        assert len(crop) == 1 and "Region [0.1000, 0.1000, 0.4000, 0.0500] of page 1" in header
        # 40% of the page width, rendered larger than that share of the full page.
        assert crop[0]["width"] > 0.4 * full[0]["width"] * 1.5

    @pytest.mark.parametrize("kwargs,code", [
        (dict(rect=[0.5, 0.5, 0.6, 0.1]), "bad_rect"),
        (dict(rect="nope"), "bad_rect"),
        (dict(rect=[0.1, 0.1, 0.2, 0.2], end_page=2), "invalid_page_range"),
    ])
    def test_bad_requests_fail_with_a_code(self, two_page_pdf, kwargs, code):
        from zotero_mcp.tools import read_pdf

        with pytest.raises(read_pdf.PdfReadError) as exc:
            read_pdf.render_pdf_pages("KEY00001", 1, ctx=MagicMock(), **kwargs)
        assert exc.value.code == code

    def test_image_reads_are_capped_at_ten_pages(self, tmp_path, monkeypatch):
        from zotero_mcp.tools import read_pdf

        doc = fitz.open()
        for _ in range(12):
            doc.new_page()
        path = str(tmp_path / "long.pdf")
        doc.save(path)
        doc.close()
        monkeypatch.setattr(read_pdf, "_get_pdf_path", lambda _k, _c: (path, "Long", False))
        with pytest.raises(read_pdf.PdfReadError) as exc:
            read_pdf.render_pdf_pages("KEY00001", 1, 12, ctx=MagicMock())
        assert exc.value.code == "page_limit_exceeded"

    def test_mcp_tool_returns_header_then_images(self, two_page_pdf):
        from fastmcp.utilities.types import Image

        from zotero_mcp.tools import read_pdf

        result = read_pdf.read_pdf_pages("KEY00001", 1, 2, format="image", ctx=MagicMock())
        assert isinstance(result[0], str) and result[0].startswith("# Pages 1-2 of Paper")
        assert [type(part) for part in result[1:]] == [Image, Image]

    def test_mcp_tool_rejects_unknown_formats(self, two_page_pdf):
        from zotero_mcp.tools import read_pdf

        with pytest.raises(read_pdf.PdfReadError) as exc:
            read_pdf.read_pdf_pages("KEY00001", 1, format="pdf", ctx=MagicMock())
        assert exc.value.code == "bad_format"


class TestCliReadImages:
    def test_images_are_written_and_listed(self, tmp_path, capsys, monkeypatch):
        from zotero_mcp.tools import read_pdf

        monkeypatch.setattr(read_pdf, "render_pdf_pages", lambda *a, **k: (
            "# Pages 4-4 of Paper", [{"page": 4, "png": b"\x89PNGdata", "width": 10, "height": 20}]))
        args = _args(item_key="KEY00001", start_page=4, end_page=None, format="image",
                     rect="0.1,0.2,0.3,0.4", out=str(tmp_path), json_out=True)
        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"):
            cli_standalone.cmd_read(args)
        image = json.loads(capsys.readouterr().out)["data"]["images"][0]
        assert image["path"] == os.path.join(str(tmp_path), "KEY00001-p4-region.png")
        with open(image["path"], "rb") as handle:
            assert handle.read() == b"\x89PNGdata"

    def test_rect_without_image_format_is_a_usage_error(self):
        with pytest.raises(CliError) as exc:
            cli_standalone.cmd_read(_args(item_key="K", start_page=1, end_page=None,
                                          format="text", rect="0.1,0.1,0.2,0.2", out=None))
        assert exc.value.code == "bad_rect"
