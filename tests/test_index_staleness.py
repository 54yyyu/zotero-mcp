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
    return home


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


class _FakeWebClient:
    def __init__(self, version=None, error=None):
        self.version, self.error, self.calls = version, error, 0

    def last_modified_version(self):
        self.calls += 1
        if self.error:
            raise self.error
        return self.version


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


def _manifest(config_dir, provider, run_id, entries, created="2026-11-01T10:00:00.000000Z"):
    run = Path(config_dir) / f"{provider}_batches" / run_id
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "created_at": created, "batches": entries})
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
    found = st.unfinished_batch_runs(cfg)
    assert len(found) == 1
    assert found[0]["provider"] == "openai"
    assert (found[0]["total"], found[0]["imported"]) == (3, 1)
    text = "\n".join(st.format_unfinished_batches(found))
    assert "2 of 3" in text and "openai" in text.lower()


def test_finished_batch_run_is_not_reported(tmp_path):
    cfg = _config(tmp_path, {"0": 1})
    _manifest(tmp_path, "gemini", "run-b", [{"batch_id": "b1", "imported_at": "2026-11-02T00:00:00Z"}])
    assert st.unfinished_batch_runs(cfg) == []


def test_no_batch_directories_means_nothing_to_report_and_none_created(tmp_path):
    cfg = _config(tmp_path, {"0": 1})
    assert st.unfinished_batch_runs(cfg) == []
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

    class Ctx:
        def info(self, *_):
            pass

        def error(self, *_):
            pass

    out = search_tools.get_search_database_status.fn(ctx=Ctx()) if hasattr(
        search_tools.get_search_database_status, "fn"
    ) else search_tools.get_search_database_status(ctx=Ctx())
    assert "STALE" in out
    assert "sentence_transformers" not in sys.modules


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


@pytest.mark.parametrize("which", ["db-status", "db status"])
def test_cli_status_prints_freshness(monkeypatch, tmp_path, capsys, which):
    out = _run_cli_status(monkeypatch, tmp_path, capsys, which)
    assert "Index freshness: STALE" in out
    assert "update" in out
