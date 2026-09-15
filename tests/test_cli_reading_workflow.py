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
    def _run(self, tmp_path, capsys, specs, create, json_out=True, dry_run=False):
        path = tmp_path / "specs.jsonl"
        path.write_text("\n".join(json.dumps(s) for s in specs))
        annotations = MagicMock()
        annotations.create_annotation.side_effect = create
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

    def test_every_spec_is_attempted_and_reported(self, tmp_path, capsys):
        replies = iter([
            "Successfully created highlight annotation\n\n**Annotation Key:** AAAA1111",
            "Error: Could not find text on page 9",
            "Successfully created area annotation\n\n**Annotation Key:** BBBB2222",
        ])
        specs = [
            {"page": 1, "text": "one", "color": "red", "tags": ["t"]},
            {"page": 9, "text": "missing"},
            {"page": 3, "rect": "0.1,0.1,0.5,0.2", "attachment_key": "OTHER001"},
        ]
        annotations, out, code = self._run(tmp_path, capsys, specs, lambda **_: next(replies))

        assert code == 1  # one failure
        payload = json.loads(out)
        assert payload["ok"] is True
        data = payload["data"]
        assert (data["succeeded"], data["failed"], data["dry_run"]) == (2, 1, False)
        assert [r["ok"] for r in data["results"]] == [True, False, True]
        assert data["results"][0]["annotation_key"] == "AAAA1111"
        assert data["results"][2]["type"] == "area"
        assert "Could not find" in data["results"][1]["error"]

        calls = [c.kwargs for c in annotations.create_annotation.call_args_list]
        assert calls[0]["color"] == "#ff6666" and calls[0]["attachment_key"] == "ATT00001"
        assert calls[1]["color"] == "#ffd400"  # default
        assert calls[2]["rect"] == [0.1, 0.1, 0.5, 0.2] and calls[2]["attachment_key"] == "OTHER001"

    def test_bad_page_fails_that_spec_only(self, tmp_path, capsys):
        ok = "Successfully created\n\n**Annotation Key:** CCCC3333"
        annotations, out, code = self._run(
            tmp_path, capsys, [{"page": "two", "text": "x"}, {"page": 2, "text": "y"}],
            lambda **_: ok)
        data = json.loads(out)["data"]
        assert [r["ok"] for r in data["results"]] == [False, True]
        assert annotations.create_annotation.call_count == 1
        assert code == 1

    def test_all_succeeded_exits_zero_with_a_summary(self, tmp_path, capsys):
        ok = "Successfully created\n\n**Annotation Key:** DDDD4444"
        _annotations, out, code = self._run(
            tmp_path, capsys, [{"page": 1, "text": "x"}], lambda **_: ok, json_out=False)
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

    def test_preview_reports_readable_matches_without_writing(self, prose_pdf, monkeypatch):
        from zotero_mcp.tools import annotations

        def fake_fetch(_key, tmpdir, *, ctx):
            return prose_pdf, "prose.pdf", None

        monkeypatch.setattr(annotations, "_fetch_pdf_attachment", fake_fetch)
        results = annotations.preview_highlights("ATT00001", [
            {"page": 1, "text": "sequential nature precludes"},
            {"page": 1, "text": "a sentence that is not in this document at all"},
            {"page": 1, "rect": [0.1, 0.1, 0.2, 0.2]},
            {"page": 1, "text": "x", "attachment_key": "OTHER001"},
        ], ctx=MagicMock())

        assert results[0]["ok"] and results[0]["matched_text"] == "sequential nature precludes"
        assert results[0]["page_found"] == 1 and results[0]["lines"] == 1
        assert not results[1]["ok"]
        assert results[2] == {"index": 3, "page": 1, "ok": True, "type": "area"}
        assert not results[3]["ok"] and "different attachment" in results[3]["error"]


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
