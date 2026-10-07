import json
import sys
import types
from pathlib import Path

import pytest
from conftest import DummyContext

from zotero_mcp import cli_standalone
from zotero_mcp.tools import retrieval


def test_fulltext_failure_puts_error_before_metadata():
    result = retrieval._fulltext_error("Paper metadata", "accessing attachment: boom")

    assert result.startswith("Error: accessing attachment: boom")
    assert "Paper metadata" in result
    assert cli_standalone._reports_failure(result)


@pytest.fixture
def api_fallback(monkeypatch):
    """Drive get_item_fulltext down its download-and-convert fallback."""
    state = types.SimpleNamespace(
        download=types.SimpleNamespace(path=None, errors=["web: HTTP 404"], source=None),
        download_error=None,
    )
    item = {"key": "ITEM0001", "data": {"key": "ITEM0001", "itemType": "journalArticle", "title": "T"}}
    monkeypatch.setattr(retrieval._library, "get_library_backend",
                        lambda: types.SimpleNamespace(get_item=lambda key: item))
    monkeypatch.setattr(retrieval._client, "format_item_metadata", lambda *a, **k: "Paper metadata")
    monkeypatch.setattr(retrieval._utils, "is_local_mode", lambda: False)
    monkeypatch.setattr(retrieval._client, "get_zotero_client",
                        lambda: types.SimpleNamespace(fulltext_item=lambda key: {}))
    monkeypatch.setattr(retrieval._client, "get_attachment_details",
                        lambda zot, it: types.SimpleNamespace(
                            key="ATTACH01", content_type="application/pdf", filename="a.pdf"))
    monkeypatch.setattr(retrieval._client, "get_local_zotero_client", lambda: None)

    def download(*args, **kwargs):
        if state.download_error:
            raise state.download_error
        return state.download

    monkeypatch.setattr(retrieval._client, "download_attachment_file", download)
    return state


def _fulltext():
    return retrieval.get_item_fulltext(item_key="ITEM0001", ctx=DummyContext())


def test_download_failure_leads_with_error(api_fallback):
    result = _fulltext()
    assert result.startswith("Error: File download failed.")
    assert "web: HTTP 404" in result
    assert "Paper metadata" in result


def test_attachment_access_failure_leads_with_error(api_fallback):
    api_fallback.download_error = RuntimeError("disk on fire")
    result = _fulltext()
    assert result.startswith("Error: accessing attachment: disk on fire")
    assert "Paper metadata" in result


def test_conversion_failure_leads_with_error(api_fallback, monkeypatch, tmp_path):
    pdf = Path(tmp_path) / "a.pdf"
    pdf.write_bytes(b"%PDF")
    api_fallback.download = types.SimpleNamespace(path=pdf, errors=[], source="web")
    monkeypatch.setattr(retrieval, "extract_file", lambda *a, **k: None)
    result = _fulltext()
    assert result.startswith("Error: converting file to markdown: a.pdf")
    assert "Paper metadata" in result


@pytest.mark.parametrize("json_mode", [False, True])
def test_cli_fulltext_failure_exits_nonzero(api_fallback, monkeypatch, capsys, json_mode):
    """JSON mode wrapped the prose as data, which skipped the failure check."""
    monkeypatch.setattr(cli_standalone, "setup_zotero_environment", lambda: None)
    argv = ["zotero-cli", "get", "fulltext", "ITEM0001"]
    if json_mode:
        argv.insert(1, "--json")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        cli_standalone.main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    if json_mode:
        payload = json.loads(captured.out)
        assert payload["ok"] is False
        assert payload["error"]["code"] == "tool_error"
        assert "File download failed" in payload["error"]["message"]
    else:
        assert captured.out == ""
        assert "File download failed" in captured.err


def test_cli_json_fulltext_of_a_paper_titled_error_is_a_success(api_fallback, monkeypatch, capsys):
    """Only a bare leading "Error" is a failure; "# Error-correcting codes" is a title."""
    monkeypatch.setattr(retrieval._client, "format_item_metadata",
                        lambda *a, **k: "# Error-correcting codes")
    monkeypatch.setattr(retrieval._client, "get_zotero_client",
                        lambda: types.SimpleNamespace(
                            fulltext_item=lambda key: {"content": "Body text."}))
    monkeypatch.setattr(cli_standalone, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(sys, "argv", ["zotero-cli", "--json", "get", "fulltext", "ITEM0001"])
    cli_standalone.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert "Body text." in payload["data"]["text"]
