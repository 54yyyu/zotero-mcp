"""Strict SQLite mode: a read that would reach the Zotero API is an error."""

import json

import pytest
import requests

from zotero_mcp import client as _client
from zotero_mcp import library, strict_sqlite
from zotero_mcp import utils as _utils


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    """No real config, no inherited setting, and the guard removed afterwards."""
    # setenv first so the variable is restored even if a test sets it directly.
    monkeypatch.setenv(strict_sqlite.ENV_VAR, "")
    monkeypatch.delenv(strict_sqlite.ENV_VAR)
    monkeypatch.setattr(strict_sqlite, "CONFIG_PATH", tmp_path / "config.json")
    yield
    strict_sqlite.deactivate()


@pytest.fixture
def http():
    return _client._pyzotero_httpx()


@pytest.fixture
def no_network(monkeypatch, http):
    """Answer below the guard, recording what got that far."""
    sent = []

    def handle(self, request):
        sent.append(f"{request.method} {request.url}")
        return http.Response(200, json=[], request=request)

    monkeypatch.setattr(http.HTTPTransport, "handle_request", handle)
    return sent


# -- which setting wins ------------------------------------------------------


@pytest.mark.parametrize(
    "flag, env, config, expected",
    [
        (None, None, None, False),
        (None, None, True, True),
        (None, "true", None, True),
        (None, "false", True, False),
        (True, "false", False, True),
        (False, "true", True, False),
    ],
)
def test_flag_beats_env_beats_config(monkeypatch, flag, env, config, expected):
    if env is not None:
        monkeypatch.setenv(strict_sqlite.ENV_VAR, env)
    if config is not None:
        strict_sqlite.CONFIG_PATH.write_text(json.dumps({"strict_sqlite": config}))
    strict_sqlite.apply_flag(flag)
    assert strict_sqlite.requested() is expected


def test_flag_reaches_child_processes_through_the_environment():
    import os

    strict_sqlite.apply_flag(True)
    assert os.environ[strict_sqlite.ENV_VAR] == "true"


def test_unreadable_config_means_off():
    strict_sqlite.CONFIG_PATH.write_text("{not json")
    assert strict_sqlite.requested() is False


# -- what the guard refuses --------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:23119/api/users/0/items",
        "http://localhost:23119/api/groups/5/collections",
        "https://api.zotero.org/users/123/items/ABC/file",
        "https://api.zotero.org/users/123/groups",
    ],
)
def test_library_get_is_refused_and_never_sent(no_network, http, url):
    strict_sqlite.activate()
    with pytest.raises(strict_sqlite.StrictSqliteViolation), http.Client() as c:
        c.get(url)
    assert no_network == []


def test_self_hosted_api_url_is_guarded(monkeypatch, no_network, http):
    monkeypatch.setenv("ZOTERO_API_URL", "https://zotero.example.org/base")
    strict_sqlite.activate()
    with pytest.raises(strict_sqlite.StrictSqliteViolation), http.Client() as c:
        c.get("https://zotero.example.org/base/users/1/items")


@pytest.mark.parametrize(
    "method, url",
    [
        ("POST", "http://127.0.0.1:23119/api/users/0/items"),
        ("PATCH", "https://api.zotero.org/users/1/items/ABC"),
        ("GET", "http://127.0.0.1:23119/api/"),
        ("GET", "https://api.zotero.org/schema"),
        ("GET", "https://api.zotero.org/items/new?itemType=book"),
        ("GET", "https://api.crossref.org/users/1/items"),
    ],
)
def test_writes_probes_and_other_services_pass(no_network, http, method, url):
    strict_sqlite.activate()
    with http.Client() as c:
        c.request(method, url)
    assert len(no_network) == 1


def test_better_bibtex_post_is_refused(monkeypatch):
    # Its JSON-RPC reads are POSTs and go through requests, not pyzotero's client.
    sent = []
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", lambda *a, **k: sent.append(a))
    strict_sqlite.activate()
    with pytest.raises(strict_sqlite.StrictSqliteViolation):
        requests.post("http://127.0.0.1:23119/better-bibtex/json-rpc", json={})
    assert sent == []


def test_exempt_client_may_read(no_network, http):
    strict_sqlite.activate()
    with http.Client() as c:
        strict_sqlite.exempt(type("Zot", (), {"client": c})(), "citation rendering")
        c.get("http://127.0.0.1:23119/api/users/0/items")
    assert len(no_network) == 1


def test_guard_is_off_unless_activated(no_network, http):
    with http.Client() as c:
        c.get("http://127.0.0.1:23119/api/users/0/items")
    assert len(no_network) == 1


def test_deactivate_restores_the_original_send(http):
    original = http.Client.send
    strict_sqlite.activate()
    assert http.Client.send is not original
    strict_sqlite.deactivate()
    assert http.Client.send is original


# -- how the error behaves ---------------------------------------------------


def test_violation_cannot_be_swallowed_by_except_exception(no_network, http):
    strict_sqlite.activate()
    with pytest.raises(strict_sqlite.StrictSqliteViolation), http.Client() as c:
        try:
            c.get("http://127.0.0.1:23119/api/users/0/items")
        except Exception:
            pytest.fail("except Exception caught the violation")


def test_message_names_this_call_site_and_the_request(no_network, http, monkeypatch):
    # Frames count as ours by module name, so make this test module one of ours.
    monkeypatch.setitem(globals(), "__name__", "zotero_mcp.tools.fake")
    strict_sqlite.activate()
    with pytest.raises(strict_sqlite.StrictSqliteViolation) as err, http.Client() as c:
        c.get("http://127.0.0.1:23119/api/users/0/items")
    assert "test_strict_sqlite.py:" in str(err.value)
    assert "GET" in str(err.value) and "/api/users/0/items" in str(err.value)


# -- the read port -----------------------------------------------------------


def test_strict_forces_the_sqlite_backend_setting(monkeypatch):
    monkeypatch.setattr(_utils, "is_local_mode", lambda: True)
    monkeypatch.setenv("ZOTERO_BACKEND", "api")
    strict_sqlite.activate()
    assert _utils.get_search_backend() == "sqlite"


def test_missing_database_is_a_violation_not_an_api_fallback(monkeypatch):
    monkeypatch.setattr(library, "_sqlite_reader", lambda: None)
    strict_sqlite.activate()
    with pytest.raises(strict_sqlite.StrictSqliteViolation, match="zotero.sqlite"):
        library.get_library_backend()


def test_declined_query_is_a_violation_not_an_api_retry():
    class Primary:
        name = "sqlite"

        def search_items(self):
            raise library.UnsupportedByBackend("a wildcard tag filter")

    def no_api():
        pytest.fail("strict mode asked the API")

    strict_sqlite.activate()
    backend = library.FallbackBackend(Primary(), no_api)
    with pytest.raises(strict_sqlite.StrictSqliteViolation, match="wildcard tag"):
        backend.search_items()


def test_declined_query_still_falls_back_when_off():
    class Primary:
        name = "sqlite"

        def search_items(self):
            raise library.UnsupportedByBackend("x")

    class Api:
        def search_items(self):
            return ["from api"]

    assert library.FallbackBackend(Primary(), Api).search_items() == ["from api"]


# -- exemptions --------------------------------------------------------------


def _read_items(zot):
    return zot.client.get("http://127.0.0.1:23119/api/users/0/items")


def test_clients_resolved_for_a_write_may_read(monkeypatch, no_network, http):
    from zotero_mcp.tools import _helpers

    zot = type("Zot", (), {"client": http.Client()})()
    monkeypatch.setattr(_utils, "is_local_mode", lambda: True)
    monkeypatch.setattr(_client, "get_local_write_client", lambda: zot)
    strict_sqlite.activate()
    read_zot, write_zot, _mode = _helpers.resolve_write_client()
    _read_items(read_zot)
    _read_items(write_zot)
    assert len(no_network) == 2


def test_bibliography_client_may_read(monkeypatch, no_network, http):
    from zotero_mcp.tools import _helpers

    zot = type("Zot", (), {"client": http.Client()})()
    monkeypatch.setattr(_client, "get_zotero_client", lambda: zot)
    strict_sqlite.activate()
    _read_items(_helpers._get_bibliography_client())
    assert len(no_network) == 1


def test_ordinary_read_client_is_not_exempt(monkeypatch, no_network, http):
    zot = type("Zot", (), {"client": http.Client()})()
    strict_sqlite.activate()
    with pytest.raises(strict_sqlite.StrictSqliteViolation):
        _read_items(zot)


def test_real_pyzotero_read_is_refused(no_network):
    # The guard has to catch pyzotero's own request path, not only a bare client.
    from pyzotero.zotero import Zotero

    strict_sqlite.activate()
    zot = Zotero(library_id="0", library_type="user", api_key=None, local=True)
    with pytest.raises(strict_sqlite.StrictSqliteViolation):
        zot.items()
    assert no_network == []


# -- boundaries --------------------------------------------------------------


def test_tool_reports_a_violation_as_a_tool_error():
    from fastmcp.exceptions import ToolError

    def tool():
        try:
            raise strict_sqlite.StrictSqliteViolation("refused")
        except Exception:
            return "no results"

    with pytest.raises(ToolError, match="refused"):
        strict_sqlite.surface_as_tool_error(tool)()


def test_registered_tool_reports_a_violation_as_an_mcp_error():
    import asyncio

    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    from zotero_mcp._app import _ZoteroMCP

    server = _ZoteroMCP("test")

    @server.tool()
    def reads() -> str:
        raise strict_sqlite.StrictSqliteViolation("refused GET")

    async def call():
        async with Client(server) as client:
            await client.call_tool("reads", {})

    with pytest.raises(ToolError, match="refused GET"):
        asyncio.run(call())


def _cli_handler_raises(monkeypatch):
    from zotero_mcp import cli_standalone

    def boom(args):
        raise strict_sqlite.StrictSqliteViolation("refused GET /x")

    monkeypatch.setitem(cli_standalone._CMD_MAP, "search", boom)


def test_zotero_cli_exits_3_and_activates_from_the_flag(monkeypatch, capsys):
    from zotero_mcp import cli_standalone

    _cli_handler_raises(monkeypatch)
    monkeypatch.setattr("sys.argv", ["zotero-cli", "search", "--strict-sqlite", "x"])
    with pytest.raises(SystemExit) as exit_info:
        cli_standalone.main()
    assert exit_info.value.code == strict_sqlite.EXIT_CODE
    assert strict_sqlite.is_active()
    assert "refused GET /x" in capsys.readouterr().err


def test_zotero_cli_json_error_envelope(monkeypatch, capsys):
    from zotero_mcp import cli_standalone

    _cli_handler_raises(monkeypatch)
    monkeypatch.setattr("sys.argv", ["zotero-cli", "--json", "search", "x"])
    with pytest.raises(SystemExit) as exit_info:
        cli_standalone.main()
    assert exit_info.value.code == strict_sqlite.EXIT_CODE
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "strict_sqlite_violation"


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["--strict-sqlite", "version"], "true"),
        (["version", "--strict-sqlite"], "true"),
        (["version", "--no-strict-sqlite"], "false"),
        (["--strict-sqlite", "version", "--no-strict-sqlite"], "false"),
    ],
)
def test_zotero_mcp_flag_works_before_and_after_the_command(monkeypatch, capsys, argv, expected):
    import os

    from zotero_mcp import cli

    monkeypatch.setattr("sys.argv", ["zotero-mcp", *argv])
    with pytest.raises(SystemExit):
        cli.main()
    assert os.environ[strict_sqlite.ENV_VAR] == expected


def test_zotero_mcp_exits_3_on_a_violation(monkeypatch, capsys):
    from zotero_mcp import cli

    def boom():
        raise strict_sqlite.StrictSqliteViolation("refused GET /x")

    monkeypatch.setattr(cli, "_run_cli", boom)
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == strict_sqlite.EXIT_CODE
    assert "refused GET /x" in capsys.readouterr().err


def test_setting_in_client_env_activates_after_env_setup(monkeypatch):
    from zotero_mcp import cli

    monkeypatch.setattr(cli, "load_standalone_env_vars", lambda: {strict_sqlite.ENV_VAR: "true"})
    monkeypatch.setattr(cli, "load_claude_desktop_env_vars", lambda: {})
    monkeypatch.setenv("ZOTERO_API_KEY", "x")
    cli.setup_zotero_environment()
    assert strict_sqlite.is_active()


def test_activation_says_so_on_stderr(capsys):
    strict_sqlite.activate()
    captured = capsys.readouterr()
    assert "Strict SQLite mode is on" in captured.err
    assert captured.out == ""  # stdout carries the MCP stdio protocol


def test_setup_keeps_the_permanent_setting(monkeypatch, tmp_path):
    from zotero_mcp import setup_helper

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    config = tmp_path / ".config" / "zotero-mcp" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"strict_sqlite": True}))
    setup_helper._write_standalone_config(local=True, api_key=None, library_id=None, library_type="user",
                                           semantic_config=None, no_claude=False)
    assert json.loads(config.read_text())["strict_sqlite"] is True


def test_read_tool_served_by_sqlite_works_in_strict_mode(monkeypatch, tmp_path, no_network):
    # The mode has to leave a tool working when SQLite can answer it.
    from unittest.mock import Mock

    import _search_corpus as corpus

    from zotero_mcp.local_db import LocalZoteroReader
    from zotero_mcp.tools import retrieval

    db = tmp_path / "zotero.sqlite"
    corpus.build_sqlite(db)
    reader = LocalZoteroReader(db_path=str(db))
    monkeypatch.setattr(library, "_sqlite_reader", lambda: reader)
    strict_sqlite.activate()
    try:
        assert "history" in retrieval.get_tags(ctx=Mock())
    finally:
        reader.close()
    assert no_network == []
