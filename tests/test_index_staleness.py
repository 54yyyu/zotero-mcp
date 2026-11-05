"""The search status says whether the index is stale.

An agent that searches a stale index gets confident answers that miss recent
additions. The status therefore compares the library version saved at the last
update with the library's version now: a read of the local Zotero database in
local mode, one API call in web mode. It loads no embedding model, and when the
check cannot be made it says so rather than claiming the index is current.

Every test points HOME and the config path at ``tmp_path`` and builds its own
Zotero database; none touches the user's real configuration or library.
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from zotero_mcp import index_staleness as st


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("ZOTERO_DB_PATH", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_TYPE", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    return home


@pytest.fixture(autouse=True)
def _no_real_zotero_api(monkeypatch):
    """No test reaches a real Zotero API; the default is 'unreachable'."""
    import importlib

    def unreachable():
        raise ConnectionError("local Zotero API not running (test default)")

    monkeypatch.setattr(importlib.import_module("zotero_mcp.client"), "get_zotero_client", unreachable)


def _config(tmp_path, versions=None, legacy=None):
    path = tmp_path / "config.json"
    section = {}
    if versions is not None:
        section["last_sync_versions"] = versions
    if legacy is not None:
        section["last_sync_version"] = legacy
    path.write_text(json.dumps({"semantic_search": section}))
    return str(path)


def _zotero_db(tmp_path, personal_version=0, group_id=None, group_version=0, items=0):
    path = tmp_path / "zotero.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE libraries (libraryID INTEGER PRIMARY KEY, type TEXT, editable INT,
            filesEditable INT, version INT NOT NULL DEFAULT 0, storageVersion INT, lastSync INT);
        CREATE TABLE groups (groupID INTEGER PRIMARY KEY, libraryID INT UNIQUE, name TEXT,
            description TEXT, version INT);
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, libraryID INT, key TEXT, version INT);
        """
    )
    conn.execute("INSERT INTO libraries VALUES (1, 'user', 1, 1, ?, 0, 0)", (personal_version,))
    if group_id is not None:
        conn.execute("INSERT INTO libraries VALUES (2, 'group', 1, 1, ?, 0, 0)", (group_version,))
        conn.execute("INSERT INTO groups VALUES (?, 2, 'G', '', 1)", (group_id,))
    conn.executemany(
        "INSERT INTO items (libraryID, key, version) VALUES (1, ?, 5)",
        [(f"K{i:07d}",) for i in range(items)],
    )
    conn.commit()
    conn.close()
    return str(path)


def test_equal_versions_are_up_to_date(tmp_path):
    cfg = _config(tmp_path, {"0": 120})
    db = _zotero_db(tmp_path, personal_version=120)
    result = st.check_index_staleness(cfg, group_id=0, local=True, db_path=db)
    assert result.state == "up_to_date"
    assert (result.saved, result.current) == (120, 120)


def test_newer_library_version_is_stale(tmp_path):
    cfg = _config(tmp_path, {"0": 100})
    db = _zotero_db(tmp_path, personal_version=130)
    result = st.check_index_staleness(cfg, group_id=0, local=True, db_path=db)
    assert result.state == "stale"
    text = st.format_staleness(result, update_command="zotero_update_search_database")
    assert "STALE" in text and "100" in text and "130" in text
    assert "zotero_update_search_database" in text


def test_group_library_uses_its_own_row_and_watermark(tmp_path):
    cfg = _config(tmp_path, {"0": 500, "77": 40})
    db = _zotero_db(tmp_path, personal_version=500, group_id=77, group_version=45)
    assert st.check_index_staleness(cfg, group_id=77, local=True, db_path=db).state == "stale"
    assert st.check_index_staleness(cfg, group_id=0, local=True, db_path=db).state == "up_to_date"


def test_group_missing_from_local_database_is_unknown_not_current(tmp_path):
    cfg = _config(tmp_path, {"99": 40})
    db = _zotero_db(tmp_path, personal_version=1)
    result = st.check_index_staleness(cfg, group_id=99, local=True, db_path=db)
    assert result.state == "unknown"
    assert "couldn't check" in st.format_staleness(result, update_command="x").lower()


def test_no_saved_version_for_the_library(tmp_path):
    cfg = _config(tmp_path, {"5": 10})
    db = _zotero_db(tmp_path, personal_version=3)
    result = st.check_index_staleness(cfg, group_id=0, local=True, db_path=db)
    assert result.state == "no_record"
    assert "update" in st.format_staleness(result, update_command="RUN-ME")
    assert "RUN-ME" in st.format_staleness(result, update_command="RUN-ME")


def test_legacy_scalar_counts_for_the_personal_library_only(tmp_path):
    cfg = _config(tmp_path, legacy=50)
    db = _zotero_db(tmp_path, personal_version=50, group_id=8, group_version=50)
    assert st.check_index_staleness(cfg, group_id=0, local=True, db_path=db).state == "up_to_date"
    assert st.check_index_staleness(cfg, group_id=8, local=True, db_path=db).state == "no_record"


def test_unreadable_local_database_says_could_not_check(tmp_path):
    cfg = _config(tmp_path, {"0": 10})
    bad = tmp_path / "bad.sqlite"
    bad.write_text("not a database")
    result = st.check_index_staleness(cfg, group_id=0, local=True, db_path=str(bad))
    assert result.state == "unknown"
    assert result.state != "up_to_date"


def test_missing_config_is_no_record_not_current(tmp_path):
    db = _zotero_db(tmp_path, personal_version=3)
    result = st.check_index_staleness(str(tmp_path / "absent.json"), group_id=0, local=True, db_path=db)
    assert result.state == "no_record"


def test_library_without_a_version_cannot_be_compared(tmp_path):
    # A library that never synced with zotero.org has version 0; "0 == 0" would
    # read as current whatever has been added since.
    cfg = _config(tmp_path, {"0": 0})
    db = _zotero_db(tmp_path, personal_version=0)
    result = st.check_index_staleness(cfg, group_id=0, local=True, db_path=db)
    assert result.state == "unknown"


def _patch_client(monkeypatch, factory):
    # Patch the module object `from zotero_mcp.client import ...` resolves to,
    # which another test's module reload can make differ from the package attribute.
    import importlib

    monkeypatch.setattr(importlib.import_module("zotero_mcp.client"), "get_zotero_client", factory)


class _FakeResponse:
    def __init__(self, status_code, version):
        self.status_code = status_code
        self.headers = {"last-modified-version": str(version)} if version is not None else {}


class _FakeHttp:
    def __init__(self, owner):
        self.owner = owner

    def get(self, url, params=None, timeout=None, **_):
        self.owner.calls += 1
        self.owner.last_request = (url, params, timeout)
        if self.owner.error:
            raise self.owner.error
        return _FakeResponse(self.owner.status, self.owner.version)


class _FakeWebClient:
    """Stands in for the Zotero client: the check does one GET through its
    HTTP client (not pyzotero's retrying ``last_modified_version``)."""

    endpoint = "http://example.invalid/api"
    library_type = "users"
    library_id = "0"

    def __init__(self, version=None, error=None, status=200):
        self.version, self.error, self.status, self.calls = version, error, status, 0
        self.last_request = None
        self.client = _FakeHttp(self)

    def last_modified_version(self):  # must not be used: it retries and ignores our timeout
        raise AssertionError("the check must not call pyzotero's last_modified_version")


def test_web_mode_makes_one_api_call(tmp_path, monkeypatch):
    fake = _FakeWebClient(version=210)
    _patch_client(monkeypatch, lambda: fake)
    cfg = _config(tmp_path, {"0": 200})
    result = st.check_index_staleness(cfg, group_id=0, local=False)
    assert result.state == "stale"
    assert (result.saved, result.current) == (200, 210)
    assert fake.calls == 1


def test_web_failure_says_could_not_check(tmp_path, monkeypatch):
    fake = _FakeWebClient(error=RuntimeError("503 from api.zotero.org"))
    _patch_client(monkeypatch, lambda: fake)
    cfg = _config(tmp_path, {"0": 200})
    result = st.check_index_staleness(cfg, group_id=0, local=False)
    assert result.state == "unknown"
    text = st.format_staleness(result, update_command="x")
    assert "couldn't check" in text.lower() and "503" in text
    assert "up to date" not in text.lower()


def test_web_mode_without_credentials_says_could_not_check(tmp_path, monkeypatch):
    def refuse():
        raise ValueError("Missing required environment variables")

    _patch_client(monkeypatch, refuse)
    cfg = _config(tmp_path, {"0": 200})
    assert st.check_index_staleness(cfg, group_id=0, local=False).state == "unknown"


# ---- unfinished batch runs -------------------------------------------------


def _manifest(config_dir, provider, run_id, entries, created="2026-11-01T10:00:00.000000Z", group_id=0):
    run = Path(config_dir) / f"{provider}_batches" / run_id
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "created_at": created, "group_id": group_id, "batches": entries})
    )


def test_unfinished_batch_run_is_reported_from_the_manifest(tmp_path):
    cfg = _config(tmp_path, {"0": 1})
    _manifest(
        tmp_path,
        "openai",
        "run-a",
        [
            {"batch_id": "b1", "status": "completed", "imported_at": "2026-11-02T00:00:00Z"},
            {"batch_id": "b2", "status": "in_progress"},
            {"batch_id": None, "status": "pending"},
        ],
    )
    found = st.unfinished_batch_runs(cfg, group_id=0)
    assert len(found) == 1
    assert found[0]["provider"] == "openai"
    assert (found[0]["total"], found[0]["imported"]) == (3, 1)
    text = "\n".join(st.format_unfinished_batches(found))
    assert "2 of 3" in text and "openai" in text.lower()


def test_finished_batch_run_is_not_reported(tmp_path):
    cfg = _config(tmp_path, {"0": 1})
    _manifest(tmp_path, "gemini", "run-b", [{"batch_id": "b1", "imported_at": "2026-11-02T00:00:00Z"}])
    assert st.unfinished_batch_runs(cfg, group_id=0) == []


def test_other_librarys_newer_run_does_not_hide_this_librarys_unfinished_one(tmp_path):
    cfg = _config(tmp_path, {"0": 1})
    _manifest(tmp_path, "openai", "run-old", [{"batch_id": "b", "status": "in_progress"}],
              created="2026-11-01T10:00:00.000000Z", group_id=0)
    _manifest(tmp_path, "openai", "run-new", [{"batch_id": "c", "imported_at": "x"}],
              created="2026-11-03T10:00:00.000000Z", group_id=77)
    mine = st.unfinished_batch_runs(cfg, group_id=0)
    assert [r["run_id"] for r in mine] == ["run-old"]
    assert st.unfinished_batch_runs(cfg, group_id=77) == []
    assert st.unfinished_batch_runs(cfg, group_id=5) == []


def test_dead_chunks_point_at_batch_status_not_batch_import(tmp_path):
    cfg = _config(tmp_path, {"0": 1})
    _manifest(tmp_path, "openai", "run-d", [
        {"batch_id": "b1", "status": "failed"},
        {"batch_id": "b2", "status": "expired"},
        {"batch_id": "b3", "status": "completed"},
    ])
    found = st.unfinished_batch_runs(cfg, group_id=0)
    assert (found[0]["dead"], found[0]["total"]) == (2, 3)
    text = "\n".join(st.format_unfinished_batches(found, "IMPORT-CMD", "STATUS-CMD"))
    assert "STATUS-CMD" in text and "cannot be imported" in text
    assert "IMPORT-CMD" in text  # the completed chunk can still be imported


def test_batch_report_does_not_import_chromadb(tmp_path):
    import subprocess

    cfg = _config(tmp_path, {"0": 1})
    _manifest(tmp_path, "openai", "r", [{"batch_id": "b", "status": "failed"}])
    _manifest(tmp_path, "gemini", "g", [{"batch_id": "b", "status": "JOB_STATE_EXPIRED"}])
    src = str(Path(st.__file__).resolve().parents[1])
    code = (
        f"import sys; sys.path.insert(0, {src!r})\n"
        "from zotero_mcp import index_staleness as st\n"
        f"r = st.unfinished_batch_runs({cfg!r}, group_id=0)\n"
        "print([x['dead'] for x in r], 'chromadb' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={"HOME": str(tmp_path), "PATH": ""}).stdout
    assert out.strip() == "[1, 1] False"


def test_run_with_only_dead_chunks_does_not_offer_import(tmp_path):
    cfg = _config(tmp_path, {"0": 1})
    _manifest(tmp_path, "gemini", "run-e", [{"batch_id": "b1", "status": "JOB_STATE_FAILED"}])
    text = "\n".join(st.format_unfinished_batches(
        st.unfinished_batch_runs(cfg, group_id=0), "IMPORT-CMD", "STATUS-CMD"))
    assert "IMPORT-CMD" not in text and "STATUS-CMD" in text


def test_no_batch_directories_means_nothing_to_report_and_none_created(tmp_path):
    cfg = _config(tmp_path, {"0": 1})
    assert st.unfinished_batch_runs(cfg, group_id=0) == []
    assert not (tmp_path / "openai_batches").exists()
    assert not (tmp_path / "gemini_batches").exists()


# ---- the status tool -------------------------------------------------------


def test_status_tool_includes_freshness_and_loads_no_embedding_model(tmp_path, monkeypatch, _isolated_home):
    pytest.importorskip("chromadb")
    from zotero_mcp.tools import search as search_tools

    cfg_dir = _isolated_home / ".config" / "zotero-mcp"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.json").write_text(json.dumps({"semantic_search": {"last_sync_versions": {"0": 7}}}))
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setattr(
        st,
        "check_index_staleness",
        lambda *a, **k: st.Staleness("stale", saved=7, current=9, detail=""),
    )
    monkeypatch.setattr(
        "zotero_mcp.chroma_client.read_collection_status",
        lambda *_a, **_k: {"name": "c", "count": 4, "initialized": True},
    )

    before = set(sys.modules)
    out = _call_status_tool(search_tools)
    loaded = set(sys.modules) - before
    assert "STALE" in out
    assert not {m for m in loaded if m.split(".")[0] in ("sentence_transformers", "torch")}


class _Ctx:
    def info(self, *_):
        pass

    def error(self, *_):
        pass


def _call_status_tool(search_tools):
    tool = search_tools.get_search_database_status
    fn = tool.fn if hasattr(tool, "fn") else tool
    return fn(ctx=_Ctx())


def test_check_is_fast_on_a_few_thousand_items(tmp_path):
    cfg = _config(tmp_path, {"0": 10})
    db = _zotero_db(tmp_path, personal_version=11, items=3000)
    start = time.perf_counter()
    result = st.check_index_staleness(cfg, group_id=0, local=True, db_path=db)
    assert result.state == "stale"
    assert time.perf_counter() - start < 2.0


# ---- the command-line status commands --------------------------------------


class _StubSearch:
    def get_database_status(self):
        return {
            "collection_info": {"name": "c", "count": 3},
            "update_config": {},
            "openai_batch": {},
            "chunking": {},
        }


def _run_cli_status(monkeypatch, tmp_path, capsys, which):
    pytest.importorskip("chromadb")
    import zotero_mcp.cli as cli
    import zotero_mcp.cli_standalone as standalone
    import zotero_mcp.semantic_search as semantic_search

    cfg = _config(tmp_path, {"0": 5})
    db = _zotero_db(tmp_path, personal_version=9)
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_DB_PATH", db)
    monkeypatch.setattr(cli, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(semantic_search, "create_semantic_search", lambda *a, **k: _StubSearch())
    if which == "db-status":
        monkeypatch.setattr(sys, "argv", ["zotero-mcp", "db-status", "--config-path", cfg])
        cli.main()
    else:
        monkeypatch.setattr(sys, "argv", ["zotero-cli", "db", "status", "--config-path", cfg])
        standalone.main()
    return capsys.readouterr().out


# ---- review round 1, minor findings ----------------------------------------


def test_legacy_scalar_follows_the_env_default_library(tmp_path, monkeypatch):
    # Mirrors _migrate_legacy_sync_version: the scalar belongs to the
    # env-configured default library, which may be a group.
    cfg = _config(tmp_path, legacy=50)
    monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "group")
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "8")
    assert st.saved_sync_version(cfg, 8) == 50
    assert st.saved_sync_version(cfg, 0) is None


def test_saved_version_ahead_of_the_library_says_so(tmp_path):
    cfg = _config(tmp_path, {"0": 130})
    db = _zotero_db(tmp_path, personal_version=120)
    result = st.check_index_staleness(cfg, group_id=0, local=True, db_path=db)
    assert result.state == "ahead"
    text = st.format_staleness(result, "RUN")
    assert "STALE" in text and "ahead" in text and "full scan" in text and "RUN" in text


def test_result_names_the_library_and_says_only_the_active_one_is_checked(tmp_path):
    cfg = _config(tmp_path, {"0": 5, "77": 5})
    db = _zotero_db(tmp_path, personal_version=5, group_id=77, group_version=5)
    personal = st.format_staleness(st.check_index_staleness(cfg, group_id=0, local=True, db_path=db), "x")
    group = st.format_staleness(st.check_index_staleness(cfg, group_id=77, local=True, db_path=db), "x")
    assert "personal library" in personal and "group 77" in group
    assert "active library" in personal and "other libraries are not checked" in personal


def test_check_active_library_follows_the_group_override(tmp_path, monkeypatch):
    import importlib

    client = importlib.import_module("zotero_mcp.client")
    cfg = _config(tmp_path, {"0": 5, "77": 5})
    db = _zotero_db(tmp_path, personal_version=5, group_id=77, group_version=9)
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_DB_PATH", db)
    monkeypatch.setattr(client, "_active_library_override", {"library_id": "77", "library_type": "group"})
    result = st.check_active_library(cfg)
    assert (result.state, result.current) == ("stale", 9)
    assert result.group_id == 77


@pytest.mark.parametrize(
    "info",
    [
        {"count": 0, "initialized": True},
        {"count": 0, "initialized": False},
        {"count": 5, "unreadable": True},
        {"count": 5, "error": "boom"},
    ],
)
def test_unusable_index_overrides_up_to_date(info):
    up = st.Staleness("up_to_date", saved=5, current=5, group_id=0)
    text = st.format_staleness(up, "RUN", index_problem=st.index_problem(info))
    assert "not meaningful" in text and "RUN" in text
    assert "up to date (" not in text


def test_usable_index_has_no_problem():
    assert st.index_problem({"count": 5, "initialized": True}) is None


def test_status_tool_does_not_say_up_to_date_for_an_empty_collection(tmp_path, monkeypatch, _isolated_home):
    pytest.importorskip("chromadb")
    from zotero_mcp.tools import search as search_tools

    cfg_dir = _isolated_home / ".config" / "zotero-mcp"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.json").write_text(json.dumps({"semantic_search": {"last_sync_versions": {"0": 7}}}))
    monkeypatch.setattr(
        st, "check_index_staleness", lambda *a, **k: st.Staleness("up_to_date", 7, 7, group_id=0)
    )
    monkeypatch.setattr(
        "zotero_mcp.chroma_client.read_collection_status",
        lambda *_a, **_k: {"name": "c", "count": 0, "initialized": True},
    )
    out = _call_status_tool(search_tools)
    assert "not meaningful" in out and "up to date (" not in out


# ---- local mode reads what update_database reads (review round 1) -----------


def _wal_db(tmp_path, name, checkpointed=100, in_wal=130):
    """A Zotero-like database whose newest library version sits only in the
    write-ahead log, as after a sync with Zotero running. Returns (path, conn);
    the connection stays open so the WAL is not checkpointed away."""
    path = tmp_path / name
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.executescript(
        """
        CREATE TABLE libraries (libraryID INTEGER PRIMARY KEY, type TEXT, editable INT,
            filesEditable INT, version INT NOT NULL DEFAULT 0, storageVersion INT, lastSync INT);
        CREATE TABLE groups (groupID INTEGER PRIMARY KEY, libraryID INT UNIQUE, name TEXT,
            description TEXT, version INT);
        """
    )
    conn.execute("INSERT INTO libraries VALUES (1, 'user', 1, 1, ?, 0, 0)", (checkpointed,))
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("UPDATE libraries SET version = ? WHERE libraryID = 1", (in_wal,))
    conn.commit()
    assert Path(str(path) + "-wal").stat().st_size > 0
    return str(path), conn


def test_local_mode_asks_the_local_api_like_update_does(tmp_path, monkeypatch):
    fake = _FakeWebClient(version=130)
    _patch_client(monkeypatch, lambda: fake)
    cfg = _config(tmp_path, {"0": 100})
    db = _zotero_db(tmp_path, personal_version=100)  # a stale file must not win
    result = st.check_index_staleness(cfg, group_id=0, local=True, db_path=db)
    assert (result.state, result.current) == ("stale", 130)
    assert fake.calls == 1


def test_the_request_is_one_get_with_a_short_per_request_timeout(tmp_path, monkeypatch):
    fake = _FakeWebClient(version=100)
    _patch_client(monkeypatch, lambda: fake)
    st.check_index_staleness(_config(tmp_path, {"0": 100}), group_id=0, local=True)
    url, params, timeout = fake.last_request
    assert url == "http://example.invalid/api/users/0/items"
    assert params["limit"] == 1
    assert timeout is not None and timeout <= 10


def _real_local_client_at(port):
    """A real pyzotero local client pointed at a throwaway loopback port."""
    from pyzotero import zotero

    from zotero_mcp.client import _make_local_http_client

    zot = zotero.Zotero(library_id="0", library_type="user", api_key=None, local=True,
                        client=_make_local_http_client())
    zot.endpoint = f"http://127.0.0.1:{port}/api"
    return zot


@pytest.fixture
def silent_port():
    """127.0.0.1 port that accepts connections and never answers."""
    import socket

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    held = []

    def accept():
        while True:
            try:
                held.append(srv.accept()[0])
            except OSError:
                return

    import threading

    threading.Thread(target=accept, daemon=True).start()
    yield srv.getsockname()[1]
    srv.close()
    for c in held:
        c.close()


def test_a_local_api_that_never_answers_is_bounded_by_our_timeout(tmp_path, monkeypatch, silent_port):
    assert silent_port != 23119
    _patch_client(monkeypatch, lambda: _real_local_client_at(silent_port))
    cfg = _config(tmp_path, {"0": 100})
    start = time.perf_counter()
    result = st.check_index_staleness(cfg, group_id=0, local=False)
    elapsed = time.perf_counter() - start
    assert result.state == "unknown" and "couldn't check" in st.format_staleness(result, "x")
    assert elapsed < 8, f"took {elapsed:.1f}s; pyzotero's own 30 s timeout is in effect"


def test_a_rate_limited_api_is_not_waited_out(tmp_path, monkeypatch):
    import http.server
    import threading

    class Limited(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(429)
            self.send_header("Retry-After", "30")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Limited)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        assert port != 23119
        _patch_client(monkeypatch, lambda: _real_local_client_at(port))
        start = time.perf_counter()
        result = st.check_index_staleness(_config(tmp_path, {"0": 100}), group_id=0, local=False)
        elapsed = time.perf_counter() - start
    finally:
        server.shutdown()
        server.server_close()
    assert result.state == "unknown" and "429" in result.detail
    assert elapsed < 3


def test_wal_not_read_in_place_is_never_up_to_date(tmp_path, monkeypatch):
    # The reviewer's reproduction: saved 100, live 130 only in the WAL, the
    # local API unreachable and snapshots disabled, so the file reads 100.
    db, conn = _wal_db(tmp_path, "wal.sqlite")
    monkeypatch.setenv("ZOTERO_MCP_DB_SNAPSHOT", "0")
    cfg = _config(tmp_path, {"0": 100})
    try:
        result = st.check_index_staleness(cfg, group_id=0, local=True, db_path=db)
    finally:
        conn.close()
    assert result.state == "unknown"
    assert "write-ahead" in result.detail
    assert "up to date" not in st.format_staleness(result, "x").lower().replace("brings it up to date", "")


def test_wal_in_snapshot_fallback_is_read_correctly(tmp_path, monkeypatch):
    db, conn = _wal_db(tmp_path, "wal2.sqlite")
    monkeypatch.delenv("ZOTERO_MCP_DB_SNAPSHOT", raising=False)
    cfg = _config(tmp_path, {"0": 100})
    try:
        result = st.check_index_staleness(cfg, group_id=0, local=True, db_path=db)
    finally:
        conn.close()
    assert (result.state, result.current) == ("stale", 130)


def test_throttled_old_snapshot_is_never_up_to_date(tmp_path, monkeypatch):
    db, conn = _wal_db(tmp_path, "wal3.sqlite", checkpointed=100, in_wal=110)
    monkeypatch.delenv("ZOTERO_MCP_DB_SNAPSHOT", raising=False)
    monkeypatch.setenv("ZOTERO_MCP_DB_SNAPSHOT_MIN_INTERVAL", "3600")
    cfg = _config(tmp_path, {"0": 110})
    try:
        assert st.check_index_staleness(cfg, group_id=0, local=True, db_path=db).state == "up_to_date"
        conn.execute("UPDATE libraries SET version = 130 WHERE libraryID = 1")
        conn.commit()
        # The cached copy still says 110, but the WAL has moved on.
        assert st.check_index_staleness(cfg, group_id=0, local=True, db_path=db).state == "unknown"
    finally:
        conn.close()


def test_fallback_to_a_complete_file_read_still_answers(tmp_path):
    cfg = _config(tmp_path, {"0": 100})
    db = _zotero_db(tmp_path, personal_version=100)
    assert st.check_index_staleness(cfg, group_id=0, local=True, db_path=db).state == "up_to_date"


@pytest.mark.parametrize("which", ["db-status", "db status"])
def test_cli_status_prints_freshness(monkeypatch, tmp_path, capsys, which):
    out = _run_cli_status(monkeypatch, tmp_path, capsys, which)
    assert "Index freshness: STALE" in out
    assert "update" in out
