"""An unreachable embedding API stops the update, and says so.

A university API may answer only on the campus network. These tests pin what
happens off it:

- a realtime update probes the API once before embedding anything, and only
  when there is something to embed; it stops if the API is unreachable, and
  stops mid-run too, keeping what it already wrote and leaving ``last_update``
  and the sync watermark alone;
- the failure is stored (time and host), shown by the status tools, cleared on
  the next successful update, and holds background runs off for an hour;
- a semantic search whose query cannot be embedded says so and points at
  keyword search;
- local embedding functions are never probed.

The OpenAI client is a fake that raises the SDK's real exception types. The
index is a real ChromaDB collection in a temporary directory. Nothing here
makes a network call.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")
openai = pytest.importorskip("openai")

from test_embedding_api_unreachable import (  # noqa: E402
    API_KEY,
    BASE_URL,
    HOST,
    FakeOpenAIClient,
    _no_wait_limiter,
    always,
    connection_error,
    status_error,
    timeout_error,
)

from zotero_mcp import config_light, semantic_search  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient  # noqa: E402
from zotero_mcp.embeddings.base import EmbeddingAPIUnreachable  # noqa: E402

# ---------------------------------------------------------------------------
# update_database against a real temporary index
# ---------------------------------------------------------------------------


class FakeZotero:
    def __init__(self, keys, version=7):
        self.keys = keys
        self.version = version

    def last_modified_version(self):
        return self.version

    def item_versions(self, **kwargs):
        return {k: 1 for k in self.keys}


def _items(n):
    return [
        {
            "key": f"ITEM{i:04d}",
            "data": {
                "title": f"Title {i}",
                "itemType": "journalArticle",
                "abstractNote": f"Abstract {i}",
                "creators": [],
            },
        }
        for i in range(n)
    ]


def _write_config(path, update_config=None, **extra):
    path.write_text(
        json.dumps(
            {
                "semantic_search": {
                    "update_config": update_config
                    or {
                        "auto_update": True,
                        "update_frequency": "startup",
                        "last_update": "2026-01-01T00:00:00",
                    },
                    **extra,
                }
            }
        )
    )


def _build(tmp_path, monkeypatch, script, items, max_parallel=None, update_config=None, **extra):
    config = tmp_path / "config.json"
    _write_config(config, update_config, **extra)
    chroma = ChromaClient(
        collection_name="unreachable_test",
        persist_directory=str(tmp_path / "chroma"),
        embedding_model="openai",
        embedding_config={
            "api_key": API_KEY,
            "base_url": BASE_URL,
            "request_batch_size": 10,
            **({"max_parallel_requests": max_parallel} if max_parallel else {}),
        },
    )
    ef = chroma.embedding_function
    ef.client = FakeOpenAIClient(script)
    ef.limiter = _no_wait_limiter()
    monkeypatch.setattr(
        semantic_search, "get_zotero_client", lambda: FakeZotero([i["key"] for i in items])
    )
    search = semantic_search.ZoteroSemanticSearch(
        chroma_client=chroma, config_path=str(config)
    )
    monkeypatch.setattr(search, "_get_items_from_source", lambda **kw: list(items))
    return search, chroma, ef, config


def _saved(config):
    return json.loads(Path(config).read_text())["semantic_search"]


def test_probe_failure_stops_before_embedding_anything(tmp_path, monkeypatch):
    items = _items(30)
    search, chroma, ef, config = _build(tmp_path, monkeypatch, always(connection_error), items)

    stats = search.update_database()

    assert HOST in stats["error"] and "VPN" in stats["error"]
    assert API_KEY not in stats["error"]
    # Only the probe went out: the 1-retry probe makes 2 attempts.
    assert ef.client.embeddings.calls == [["zotero-mcp connectivity check"]] * 2
    assert chroma.collection.count() == 0
    saved = _saved(config)
    assert saved["update_config"]["last_update"] == "2026-01-01T00:00:00"
    assert "last_sync_versions" not in saved and "last_sync_version" not in saved
    failure = saved["update_config"]["last_api_failure"]
    assert failure["host"] == HOST
    assert datetime.now() - datetime.fromisoformat(failure["time"]) < timedelta(minutes=1)
    assert API_KEY not in json.dumps(saved) and BASE_URL not in json.dumps(saved)


def test_nothing_to_embed_makes_no_provider_call(tmp_path, monkeypatch):
    search, chroma, ef, config = _build(tmp_path, monkeypatch, always(connection_error), [])

    stats = search.update_database()

    assert "error" not in stats
    assert ef.client.embeddings.calls == []


def _fail_after(successes, factory=connection_error):
    return lambda n, inputs: factory() if n > successes else None


@pytest.mark.parametrize("max_parallel", [None, 2])
def test_mid_run_failure_stops_and_keeps_written_records(tmp_path, monkeypatch, max_parallel):
    items = _items(60)
    # Probe + the first request succeed, then the network goes away.
    search, chroma, ef, config = _build(
        tmp_path, monkeypatch, _fail_after(2), items, max_parallel=max_parallel
    )

    stats = search.update_database()

    assert HOST in stats["error"]
    written = chroma.collection.count()
    assert 0 < written < 60
    # It stopped rather than pushing every remaining batch through retries.
    attempts = len(ef.client.embeddings.calls)
    assert attempts < 2 + (60 // 10) * (ef.max_retries + 1)
    saved = _saved(config)
    assert saved["update_config"]["last_update"] == "2026-01-01T00:00:00"
    assert "last_sync_versions" not in saved
    assert saved["update_config"]["last_api_failure"]["host"] == HOST


def test_403_mid_run_also_stops(tmp_path, monkeypatch):
    items = _items(60)
    search, chroma, ef, config = _build(
        tmp_path,
        monkeypatch,
        _fail_after(2, lambda: status_error(openai.PermissionDeniedError, 403)),
        items,
    )

    stats = search.update_database()

    assert "HTTP 403" in stats["error"]
    # probe, first batch, the failing batch, then one re-check that also fails
    assert len(ef.client.embeddings.calls) == 4


def test_other_errors_keep_todays_behaviour(tmp_path, monkeypatch):
    items = _items(30)
    bad = lambda n, inputs: (  # noqa: E731
        status_error(openai.BadRequestError, 400) if "Title 3" in " ".join(inputs) else None
    )
    search, chroma, ef, config = _build(tmp_path, monkeypatch, bad, items)

    stats = search.update_database()

    assert "error" not in stats
    assert stats["errors"] > 0
    assert chroma.collection.count() > 0
    assert _saved(config)["update_config"]["last_update"] != "2026-01-01T00:00:00"


def test_success_clears_stored_failure(tmp_path, monkeypatch):
    items = _items(5)
    update_config = {
        "auto_update": True,
        "update_frequency": "startup",
        "last_update": "2026-01-01T00:00:00",
        "last_api_failure": {"time": datetime.now().isoformat(), "host": HOST, "reason": "connection failed"},
    }
    search, chroma, ef, config = _build(
        tmp_path, monkeypatch, lambda n, i: None, items, update_config=update_config
    )

    stats = search.update_database()

    assert "error" not in stats
    assert chroma.collection.count() == 5
    assert "last_api_failure" not in _saved(config)["update_config"]


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


def test_search_returns_clear_message_when_query_cannot_be_embedded(tmp_path, monkeypatch):
    search, chroma, ef, config = _build(tmp_path, monkeypatch, always(connection_error), [])
    ef.max_retries = 0

    result = search.search("anything")

    assert result["results"] == []
    assert HOST in result["error"] and "VPN" in result["error"]
    assert result.get("embedding_api_unreachable") is True


def test_semantic_search_tool_suggests_keyword_search(monkeypatch):
    from conftest import DummyContext

    from zotero_mcp.tools import search as search_tools

    class StubSearch:
        def search(self, **kwargs):
            return {
                "results": [],
                "error": str(EmbeddingAPIUnreachable("openai", HOST, "connection failed")),
                "embedding_api_unreachable": True,
            }

    monkeypatch.setattr(semantic_search, "create_semantic_search", lambda *a, **k: StubSearch())
    monkeypatch.setattr(search_tools, "_maybe_fire_presearch_sync", lambda _s: None)

    out = search_tools.semantic_search(query="x", ctx=DummyContext())

    assert HOST in out
    assert "zotero_search_items" in out


# ---------------------------------------------------------------------------
# Status and background backoff
# ---------------------------------------------------------------------------


def _failure(minutes_ago):
    return {
        "time": (datetime.now() - timedelta(minutes=minutes_ago)).isoformat(),
        "host": HOST,
        "reason": "connection failed",
    }


def test_backoff_holds_background_runs_for_an_hour():
    due = {"auto_update": True, "update_frequency": "startup"}
    assert config_light.background_update_due(due) is True
    assert config_light.background_update_due({**due, "last_api_failure": _failure(5)}) is False
    assert config_light.background_update_due({**due, "last_api_failure": _failure(61)}) is True
    # The ordinary gate (and so "Should update" in status) is unchanged.
    assert config_light.should_update({**due, "last_api_failure": _failure(5)}) is True


def test_describe_api_failure():
    assert config_light.describe_api_failure({}) is None
    line = config_light.describe_api_failure({"last_api_failure": _failure(5)})
    assert HOST in line and "connection failed" in line and "paused" in line
    line = config_light.describe_api_failure({"last_api_failure": _failure(90)})
    assert HOST in line and "paused" not in line


def _home(monkeypatch, tmp_path, update_config):
    cfg_dir = tmp_path / ".config" / "zotero-mcp"
    cfg_dir.mkdir(parents=True)
    _write_config(cfg_dir / "config.json", update_config)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))


def test_startup_update_backs_off(tmp_path, monkeypatch):
    from zotero_mcp import _app

    _home(
        monkeypatch,
        tmp_path,
        {"auto_update": True, "update_frequency": "startup", "last_api_failure": _failure(5)},
    )
    assert _app._semantic_update_command() is None


def test_presearch_sync_backs_off(tmp_path, monkeypatch):
    from zotero_mcp.tools import search as search_tools

    config = tmp_path / "config.json"
    _write_config(
        config,
        {"auto_update": True, "update_frequency": "startup", "last_api_failure": _failure(5)},
    )

    class StubSearch:
        config_path = str(config)
        update_config = config_light.load_update_config(str(config))
        ran = False

        def should_update_database(self):
            return True

        def update_database(self, **kwargs):
            StubSearch.ran = True

    monkeypatch.setattr(search_tools, "_last_presearch_sync_ts", 0.0)
    started = []
    monkeypatch.setattr(search_tools._threading, "Thread", lambda **kw: started.append(kw) or _NoThread())
    search_tools._maybe_fire_presearch_sync(StubSearch())
    assert started == []


class _NoThread:
    def start(self):
        pass


def test_status_tool_shows_last_failure(tmp_path, monkeypatch):
    from conftest import DummyContext

    from zotero_mcp.tools import search as search_tools

    _home(monkeypatch, tmp_path, {"auto_update": False, "last_api_failure": _failure(5)})

    out = search_tools.get_search_database_status(ctx=DummyContext())

    assert "Embedding API" in out and HOST in out


def test_db_status_cli_shows_last_failure(tmp_path, monkeypatch, capsys):
    from zotero_mcp import cli

    config = tmp_path / "config.json"
    _write_config(config, {"auto_update": False, "last_api_failure": _failure(5)})

    class StubSearch:
        def get_database_status(self):
            return {
                "collection_info": {},
                "update_config": config_light.load_update_config(str(config)),
                "chunking": {"enabled": False, "effective": False},
            }

    monkeypatch.setattr(semantic_search, "create_semantic_search", lambda *a, **k: StubSearch())
    monkeypatch.setattr(cli, "setup_zotero_environment", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["zotero-mcp", "db-status", "--config-path", str(config)])

    cli.main()

    out = capsys.readouterr().out
    assert "embedding API failure" in out and HOST in out


def test_standalone_db_status_shows_last_failure(tmp_path, monkeypatch, capsys):
    from zotero_mcp import cli_standalone

    config = tmp_path / "config.json"
    _write_config(config, {"auto_update": False, "last_api_failure": _failure(5)})

    class StubSearch:
        def get_database_status(self):
            return {
                "collection_info": {},
                "update_config": config_light.load_update_config(str(config)),
                "chunking": {"enabled": False, "effective": False},
            }

    monkeypatch.setattr(semantic_search, "create_semantic_search", lambda *a, **k: StubSearch())
    cli_standalone_status(cli_standalone, monkeypatch, config)

    out = capsys.readouterr().out
    assert "embedding API failure" in out and HOST in out


def cli_standalone_status(module, monkeypatch, config):
    """Drive ``zotero-cli db status`` with a temporary config."""
    import argparse

    monkeypatch.setattr(module, "setup_zotero_environment", lambda *a, **k: None)
    args = argparse.Namespace(subcommand="status", config_path=str(config))
    module.cmd_db(args)


def test_local_embedding_functions_are_not_probed(tmp_path, monkeypatch):
    """Only RemoteEmbeddingFunction subclasses are probed; a local model
    (HuggingFace) runs exactly as before."""

    class LocalEF:
        probed = False

        def check_reachable(self):
            LocalEF.probed = True

        def __call__(self, documents):
            return [[1.0] for _ in documents]

    class Chroma:
        embedding_function = LocalEF()
        embedding_max_tokens = 8000
        written = []

        def truncate_text(self, text, max_tokens=None):
            return text

        def read_error(self):
            return None

        def get_existing_ids(self, ids):
            return set()

        def upsert_documents(self, documents, metadatas, ids):
            Chroma.written.extend(ids)

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: FakeZotero([]))
    # A temporary config_path keeps the fulltext cache (resolved next to the
    # config) out of the real ~/.config/zotero-mcp.
    config = tmp_path / "config.json"
    _write_config(config)
    search = semantic_search.ZoteroSemanticSearch(
        chroma_client=Chroma(), config_path=str(config)
    )
    monkeypatch.setattr(search, "_save_update_config", lambda *a, **k: None)
    monkeypatch.setattr(search, "_get_items_from_source", lambda **kw: _items(3))

    stats = search.update_database(limit=3)

    assert "error" not in stats
    assert LocalEF.probed is False
    assert len(Chroma.written) == 3


# ---------------------------------------------------------------------------
# Review round 1 regressions
# ---------------------------------------------------------------------------


def _populated(tmp_path, monkeypatch, n=20):
    items = _items(n)
    search, chroma, ef, config = _build(tmp_path, monkeypatch, lambda n, i: None, items)
    assert "error" not in search.update_database()
    assert chroma.collection.count() == n
    return search, chroma, ef, config


def test_force_rebuild_checks_the_api_before_emptying_the_index(tmp_path, monkeypatch):
    search, chroma, ef, config = _populated(tmp_path, monkeypatch)
    ef.client = FakeOpenAIClient(always(connection_error))

    stats = search.update_database(force_full_rebuild=True, allow_mass_deletion=True)

    assert HOST in stats["error"]
    assert chroma.collection.count() == 20, "the index was emptied before the API check"
    saved = _saved(config)
    assert saved["update_config"]["last_api_failure"]["host"] == HOST
    # Nothing was reset, so the watermark still describes the index.
    assert set(saved["last_sync_versions"].values()) == {7}
    assert "kept" in stats["error"]


def test_one_failing_batch_on_a_reachable_api_does_not_stop_the_run(tmp_path, monkeypatch):
    """A timeout on one batch while the API answers otherwise: the batch is
    set aside as before, and the run completes."""
    items = _items(60)

    def script(n, inputs):
        return timeout_error() if any("Title 35" in t for t in inputs) else None

    search, chroma, ef, config = _build(tmp_path, monkeypatch, script, items)

    stats = search.update_database()

    assert "error" not in stats
    assert chroma.collection.count() == 59
    assert stats["errors"] == 1
    assert "last_api_failure" not in _saved(config)["update_config"]


def test_unchanged_library_does_not_clear_the_failure(tmp_path, monkeypatch):
    """The early return contacts no API, so it cannot vouch for it."""
    search, chroma, ef, config = _populated(tmp_path, monkeypatch, n=3)
    search.update_config["last_api_failure"] = {
        "time": datetime.now().isoformat(), "host": HOST, "reason": "connection failed",
    }
    search._save_update_config()
    calls_before = len(ef.client.embeddings.calls)

    stats = search.update_database()

    assert "error" not in stats and stats["total_items"] == 0
    assert len(ef.client.embeddings.calls) == calls_before
    assert _saved(config)["update_config"]["last_api_failure"]["host"] == HOST


def test_stop_message_says_what_was_kept_and_what_to_run(tmp_path, monkeypatch):
    search, chroma, ef, config = _build(
        tmp_path, monkeypatch, always(connection_error), _items(5)
    )

    stats = search.update_database()

    assert "zotero-mcp update-db" in stats["error"]
    assert "kept" in stats["error"]


# ---------------------------------------------------------------------------
# Review round 2 regressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_parallel", [1, None])
def test_rebuild_that_loses_the_api_after_the_reset(tmp_path, monkeypatch, max_parallel):
    """The rebuild emptied the index, so the watermark goes to 0 and the
    message says how far it got; the next plain update-db finishes the job
    instead of reporting "library unchanged" on a part-filled index."""
    items = _items(30)
    search, chroma, ef, config = _build(
        tmp_path, monkeypatch, lambda n, i: None, items, max_parallel=max_parallel
    )
    assert "error" not in search.update_database()
    assert chroma.collection.count() == 30

    # The probe and the first request succeed, then the API goes away.
    ef.client = FakeOpenAIClient(_fail_after(2))
    stats = search.update_database(force_full_rebuild=True, allow_mass_deletion=True)

    written = chroma.collection.count()
    assert written < 30
    assert "stopped part way" in stats["error"]
    assert f"{written} item(s) are in the index now" in stats["error"]
    assert "zotero-mcp update-db" in stats["error"]
    assert "kept" not in stats["error"]
    assert set(_saved(config)["last_sync_versions"].values()) == {0}

    ef.client = FakeOpenAIClient(lambda n, i: None)
    stats = search.update_database()

    assert "error" not in stats
    assert stats["total_items"] == 30
    assert chroma.collection.count() == 30
    assert set(_saved(config)["last_sync_versions"].values()) == {7}


def test_watermark_is_zeroed_before_the_reset(tmp_path, monkeypatch):
    """Saved first, so a crash between the two errs toward a full scan."""
    search, chroma, ef, config = _populated(tmp_path, monkeypatch)
    seen = []
    real_reset = chroma.reset_collection

    def reset():
        seen.append(set(_saved(config)["last_sync_versions"].values()))
        real_reset()

    monkeypatch.setattr(chroma, "reset_collection", reset)

    search.update_database(force_full_rebuild=True, allow_mass_deletion=True)

    assert seen == [{0}]


def test_run_with_nothing_to_embed_keeps_the_failure(tmp_path, monkeypatch):
    """Only a run in which the API answered may clear the stored failure."""
    search, chroma, ef, config = _populated(tmp_path, monkeypatch, n=3)
    search.update_config["last_api_failure"] = {
        "time": datetime.now().isoformat(), "host": HOST, "reason": "connection failed",
    }
    search._save_update_config()
    monkeypatch.setattr(search, "_get_items_from_source", lambda **kw: [])
    calls_before = len(ef.client.embeddings.calls)

    stats = search.update_database(limit=5)

    assert "error" not in stats
    assert len(ef.client.embeddings.calls) == calls_before
    assert _saved(config)["update_config"]["last_api_failure"]["host"] == HOST
