"""A failed page read must not reach a caller looking like a successful one.

`read_pdf_pages` used to return its failures as prose. Because a return value
is indistinguishable from content, both of its callers reported the failure as
success: `zotero-cli --json read` emitted `{"ok": true, ...}` and exited 0, and
the MCP tool answered `isError: false`. A script or agent consuming either had
to parse English to find out that nothing was read.

These tests pin the two surfaces separately, because they fail independently:
the CLI envelope is built by `cli_standalone.main`'s exception handler, and the
MCP flag is set by FastMCP. Both are downstream of the same raise, but only a
test on each proves the raise actually reaches it.

The reproduction in #528 needs no library, API key or running Zotero instance:
an invalid page range is rejected before anything is resolved.
"""

import asyncio
import json
import sys

import pytest
from fastmcp import Client, FastMCP
from mcp.types import CallToolResult

from zotero_mcp import cli_standalone
from zotero_mcp.extract import PAGE_SEPARATOR, ExtractedDoc
from zotero_mcp.tools import read_pdf as read_pdf_tools

ITEM_KEY = "TESTKEY1"
RANGE_MESSAGE = "end_page must be greater than or equal to start_page"


# ---------------------------------------------------------------------------
# CLI: `zotero-cli [--json] read`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("json_mode", [False, True], ids=["text", "json"])
def test_invalid_range_exits_nonzero(monkeypatch, capsys, json_mode):
    """The exact invocation from #528, in both output modes."""
    monkeypatch.setattr(cli_standalone, "setup_zotero_environment", lambda: None)
    monkeypatch.delenv("ZOTERO_CLI_DEBUG", raising=False)
    argv = ["zotero-cli", "read", ITEM_KEY, "--start-page", "2", "--end-page", "1"]
    if json_mode:
        argv.insert(1, "--json")
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc:
        cli_standalone.main()

    assert exc.value.code == 1, "a failed read must not exit 0"
    captured = capsys.readouterr()
    if json_mode:
        payload = json.loads(captured.out)
        assert payload["ok"] is False
        assert payload["command"] == "read"
        assert payload["error"]["code"] == "invalid_page_range"
        assert RANGE_MESSAGE in payload["error"]["message"]
        # A failure envelope carries `error`, never `data` -- the bug was that
        # this one carried `data` with the message inside it.
        assert "data" not in payload
        assert captured.err == ""
    else:
        assert captured.out == ""
        assert RANGE_MESSAGE in captured.err


@pytest.mark.parametrize(
    "arguments,code",
    [
        ({"item_key": ITEM_KEY, "start_page": 2, "end_page": 1}, "invalid_page_range"),
        ({"item_key": "", "start_page": 1}, "empty_item_key"),
    ],
)
def test_every_failure_carries_a_branchable_code(monkeypatch, capsys, arguments, code):
    """`ok` alone says a call failed; the code says which failure, so a caller
    can react without matching on the message."""
    monkeypatch.setattr(cli_standalone, "setup_zotero_environment", lambda: None)
    monkeypatch.delenv("ZOTERO_CLI_DEBUG", raising=False)
    argv = ["zotero-cli", "--json", "read", arguments["item_key"],
            "--start-page", str(arguments["start_page"])]
    if "end_page" in arguments:
        argv += ["--end-page", str(arguments["end_page"])]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc:
        cli_standalone.main()

    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == code


def test_a_successful_read_is_still_a_success(monkeypatch, capsys):
    """The fix must not turn the happy path into an error."""
    monkeypatch.setattr(cli_standalone, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(
        read_pdf_tools, "read_pdf_pages",
        lambda **_kwargs: "## Page 1\n\nBody text.",
    )
    monkeypatch.delenv("ZOTERO_CLI_DEBUG", raising=False)
    monkeypatch.setattr(
        sys, "argv",
        ["zotero-cli", "--json", "read", ITEM_KEY, "--start-page", "1"],
    )

    # A successful run returns from `main` rather than exiting, which is what
    # makes the process exit code 0.
    cli_standalone.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["data"]["text"] == "## Page 1\n\nBody text."
    assert "error" not in payload


# ---------------------------------------------------------------------------
# MCP: the tool result a client sees
# ---------------------------------------------------------------------------


def _patch_extract(monkeypatch, page_texts, total=None):
    """Stand in for the extraction seam with known page text."""
    total_pages = total if total is not None else len(page_texts)

    def _fake_page_count(_path):
        return total_pages

    def _fake_extract_pdf(_path, *, pages=None, max_pages=None):
        wanted = [p for p in (pages or range(total_pages)) if 0 <= p < total_pages]
        texts = [page_texts[p % len(page_texts)] for p in wanted]
        return ExtractedDoc(
            text=PAGE_SEPARATOR.join(texts),
            pages=tuple(texts),
            page_numbers=tuple(wanted),
            page_count=total_pages,
            source="pdf",
            needs_ocr=(),
        )

    monkeypatch.setattr(read_pdf_tools, "pdf_page_count", _fake_page_count)
    monkeypatch.setattr(read_pdf_tools, "extract_pdf", _fake_extract_pdf)


@pytest.fixture
def read_server():
    """A server carrying only this tool, so no application lifespan runs."""
    server = FastMCP("read_pdf failure protocol")
    # FastMCP 2's decorator returns a FunctionTool; 3 leaves the function intact.
    tool_function = getattr(read_pdf_tools.read_pdf_pages, "fn", read_pdf_tools.read_pdf_pages)
    server.tool(tool_function, name="zotero_read_pdf_pages")
    return server


def _call(server, arguments):
    async def run():
        async with Client(server) as client:
            return await client.call_tool_mcp("zotero_read_pdf_pages", arguments, timeout=5)

    return asyncio.run(run())


def _text(result):
    assert isinstance(result, CallToolResult)
    return "\n".join(block.text for block in result.content if block.type == "text")


def test_missing_attachment_is_an_mcp_error(read_server, monkeypatch):
    monkeypatch.setattr(read_pdf_tools, "_get_pdf_path", lambda _key, _ctx: None)

    result = _call(read_server, {"item_key": ITEM_KEY, "start_page": 1})

    assert result.isError is True
    assert "No PDF attachment found" in _text(result)


def test_invalid_range_is_an_mcp_error(read_server):
    """No patching needed: the range is rejected before any lookup."""
    result = _call(read_server, {"item_key": ITEM_KEY, "start_page": 2, "end_page": 1})

    assert result.isError is True
    assert RANGE_MESSAGE in _text(result)


def test_unreadable_pdf_is_an_mcp_error(read_server, monkeypatch):
    monkeypatch.setattr(
        read_pdf_tools, "_get_pdf_path", lambda _key, _ctx: ("/tmp/test.pdf", "Paper", True)
    )

    def _boom(_path):
        raise ValueError("Not a PDF: file is empty")

    monkeypatch.setattr(read_pdf_tools, "pdf_page_count", _boom)

    result = _call(read_server, {"item_key": ITEM_KEY, "start_page": 1})

    assert result.isError is True
    assert "Not a PDF" in _text(result)


def test_successful_read_is_not_an_mcp_error(read_server, monkeypatch):
    _patch_extract(monkeypatch, ["Page one body."], total=1)
    monkeypatch.setattr(
        read_pdf_tools, "_get_pdf_path", lambda _key, _ctx: ("/tmp/test.pdf", "Paper", True)
    )

    result = _call(read_server, {"item_key": ITEM_KEY, "start_page": 1})

    assert result.isError is False
    assert "Page one body." in _text(result)
