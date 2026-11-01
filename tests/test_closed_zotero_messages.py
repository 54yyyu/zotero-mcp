"""The rest of #744: what a closed Zotero looks like from the CLI.

The CLI printed every failure twice, the raw exception first: a tool logs
``ctx.error(f"Error adding by BibTeX: {e}")`` and then returns the same
failure through ``format_zotero_error``, and both reached stderr. And with
no write key, ``write_unavailable_message`` said the Zotero build "does not
expose the local write API" when Zotero was simply closed, because the
probe answered None for both.
"""

import sys
import time

import pytest

from zotero_mcp import cli_standalone
from zotero_mcp import client as _client
from zotero_mcp.tools import _helpers


def _run(monkeypatch, handler, argv=("config",)):
    monkeypatch.setitem(cli_standalone._CMD_MAP, argv[0], handler)
    monkeypatch.setattr(sys, "argv", ["zotero-cli", *argv])
    with pytest.raises(SystemExit) as exit_info:
        cli_standalone.main()
        raise SystemExit(0)
    return exit_info.value.code


def test_a_failure_the_tool_returns_is_printed_once(monkeypatch, capsys):
    def handler(args):
        ctx = cli_standalone._ctx(args)
        ctx.error("Error adding by BibTeX: [Errno 61] Connection refused")
        cli_standalone._out(args, "add", text=(
            "Error adding by BibTeX: Zotero is not running (nothing listening "
            "on 127.0.0.1:23119)."
        ))

    assert _run(monkeypatch, handler) == 1
    err = capsys.readouterr().err
    assert "Zotero is not running" in err
    assert "Errno 61" not in err
    assert err.count("Error adding by BibTeX") == 1


def test_other_errors_are_still_printed(monkeypatch, capsys):
    def handler(args):
        ctx = cli_standalone._ctx(args)
        ctx.error("Failed to update item ABCD1234: 400 bad field")
        cli_standalone._out(args, "batch", text="# Batch Tag Update Results\nItems updated: 1")

    assert _run(monkeypatch, handler) == 0
    captured = capsys.readouterr()
    assert "[ERROR] Failed to update item ABCD1234" in captured.err
    assert "Items updated: 1" in captured.out


def test_an_error_the_result_quotes_is_not_repeated(monkeypatch, capsys):
    def handler(args):
        cli_standalone._ctx(args).error("Error fetching collections: [Errno 61] refused")
        cli_standalone._out(args, "get", text=(
            "# Zotero Collections\n\nError fetching collections: [Errno 61] refused"
        ))

    _run(monkeypatch, handler)
    assert "[ERROR]" not in capsys.readouterr().err


def test_errors_are_printed_when_the_command_raises(monkeypatch, capsys):
    def handler(args):
        cli_standalone._ctx(args).error("Failed to fetch item K1: timeout")
        raise RuntimeError("boom")

    assert _run(monkeypatch, handler) == 1
    err = capsys.readouterr().err
    assert "[ERROR] Failed to fetch item K1" in err
    assert "Error: boom" in err


def test_verbose_prints_errors_as_they_happen(capsys):
    ctx = cli_standalone.CLIContext(verbose=True)
    ctx.error("Error x: raw")
    assert "[ERROR] Error x: raw" in capsys.readouterr().err


class TestWriteUnavailableMessage:
    def test_closed_zotero_says_so(self):
        _client._local_probe_cache.update(
            {"server_id": None, "reachable": False, "checked_at": time.monotonic()}
        )
        msg = _helpers.write_unavailable_message()
        assert "Zotero is not running" in msg
        assert "does not expose the local write API" not in msg
        assert "authorize-local" in msg and "ZOTERO_API_KEY" in msg

    def test_old_zotero_still_says_it_lacks_the_write_api(self):
        _client._local_probe_cache.update(
            {"server_id": None, "reachable": True, "checked_at": time.monotonic()}
        )
        assert "does not expose the local write API" in _helpers.write_unavailable_message()


class TestProbeTellsClosedFromOld:
    def test_refused_connection_is_unreachable(self, monkeypatch):
        _client._local_probe_cache.clear()

        def refuse(*_a, **_k):
            raise ConnectionRefusedError(61, "Connection refused")

        monkeypatch.setattr(_client, "_make_local_http_client", refuse)
        assert _client.probe_local_server_id() is None
        assert _client.local_server_unreachable() is True

    def test_an_answer_without_the_header_is_reachable(self, monkeypatch):
        _client._local_probe_cache.clear()

        class _Resp:
            headers = {}

        class _Http:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get(self, url):
                return _Resp()

        monkeypatch.setattr(_client, "_make_local_http_client", lambda *a, **k: _Http())
        assert _client.probe_local_server_id() is None
        assert _client.local_server_unreachable() is False
