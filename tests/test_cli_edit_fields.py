"""Generic metadata assignments must reach the existing schema-aware writer."""

from unittest.mock import MagicMock

import pytest

from zotero_mcp import cli_standalone


def _edit(monkeypatch, argv):
    args = cli_standalone.build_parser().parse_args(["edit", "ITEM1234", *argv])
    writer = MagicMock()
    writer.update_item.return_value = "updated"
    monkeypatch.setattr(cli_standalone, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(
        cli_standalone, "_import_tools", lambda: (None, None, None, writer, None)
    )
    cli_standalone.cmd_edit(args)
    return writer.update_item.call_args.kwargs


def test_repeatable_fields_preserve_values_and_existing_flags(monkeypatch):
    result = _edit(monkeypatch, [
        "--field", "seriesNumber=2",
        "--field", "url=https://example.org/?a=1&b=2",
        "--title", "A book",
        "--add-tags", "reviewed,important",
    ])
    assert result["fields"] == {
        "seriesNumber": "2", "url": "https://example.org/?a=1&b=2", "title": "A book"
    }
    assert result["add_tags"] == ["reviewed", "important"]


def test_software_version_and_empty_value_are_strings(monkeypatch):
    result = _edit(monkeypatch, [
        "--field", "versionNumber=v9.15", "--field", "extra="
    ])
    assert result["fields"] == {"versionNumber": "v9.15", "extra": ""}


def test_last_field_assignment_wins_and_overrides_named_flag(monkeypatch):
    result = _edit(monkeypatch, [
        "--field", "title=first", "--title", "named", "--field", "title=last"
    ])
    assert result["fields"] == {"title": "last"}


def test_raw_api_name_is_forwarded_alongside_named_flags(monkeypatch):
    result = _edit(monkeypatch, [
        "--publication-title", "old", "--field", "publicationTitle=new"
    ])
    assert result["fields"] == {"publication_title": "old", "publicationTitle": "new"}


@pytest.mark.parametrize("assignment", ["seriesNumber", "=2", "   =2"])
def test_malformed_assignment_is_rejected_before_dispatch(assignment, capsys):
    with pytest.raises(SystemExit) as exc:
        cli_standalone.build_parser().parse_args([
            "edit", "ITEM1234", "--field", assignment
        ])
    assert exc.value.code == 2
    assert "NAME=VALUE" in capsys.readouterr().err


def test_field_names_are_trimmed_but_values_are_preserved(monkeypatch):
    result = _edit(monkeypatch, ["--field", " extra =  text = more  "])
    assert result["fields"] == {"extra": "  text = more  "}
