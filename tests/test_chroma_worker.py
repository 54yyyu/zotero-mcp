"""The MCP server runs ChromaDB in a worker process that exits when idle.

ChromaDB keeps a loaded vector index in a process-wide cache that nothing in
its API releases, so the only way to give the memory back is to end the
process holding it. These tests run a real worker against a small real index:
calls are forwarded and their results, exceptions and iterators come back;
the worker stops after the idle time and the next call starts a new one; a
worker that dies is replaced; and the calling process never opens ChromaDB.
"""

import importlib.util
import json
import os
import signal
import sys
import time

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

import chromadb
import numpy as np
from chromadb.api.client import SharedSystemClient
from chromadb.config import Settings

from zotero_mcp import chroma_worker
from zotero_mcp.chroma_client import ChromaClient, create_chroma_client
from zotero_mcp.chroma_worker import WorkerChromaClient, WorkerExited

N = 30


@pytest.fixture()
def config(tmp_path, monkeypatch):
    """A populated index, a config pointing at it, and worker mode switched on."""
    persist = tmp_path / "chroma_db"
    client = chromadb.PersistentClient(path=str(persist), settings=Settings(anonymized_telemetry=False))
    client.create_collection("zotero_library").add(
        ids=[f"ITEM{i:04d}#0" for i in range(N)],
        embeddings=np.random.default_rng(5).random((N, 8)).astype(np.float32),
        documents=[f"document {i}" for i in range(N)],
        metadatas=[{"item_key": f"ITEM{i:04d}", "group_id": 0} for i in range(N)],
    )
    del client
    SharedSystemClient.clear_system_cache()

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"semantic_search": {
        "persist_directory": str(persist),
        "embedding_model": "default",
        "idle_unload_minutes": 10,  # opt-in: the default (0) keeps ChromaDB in-process
    }}))
    # The worker is `python -m zotero_mcp.chroma_worker`; make sure it imports
    # this checkout, as the server's own interpreter would.
    src = os.path.dirname(os.path.dirname(chroma_worker.__file__))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")])))
    monkeypatch.setattr(chroma_worker, "_enabled", True)
    monkeypatch.setattr(chroma_worker, "_shared", {})
    yield str(path)
    chroma_worker.stop_all()


def _in_process_chroma_systems() -> int:
    return len(SharedSystemClient._identifier_to_system)


def test_server_gets_one_worker_backed_client_per_config(config):
    first = create_chroma_client(config)
    second = create_chroma_client(config)

    assert isinstance(first, WorkerChromaClient)
    assert first is second


def test_a_changed_config_gets_a_fresh_worker(config):
    # Before the worker, every tool call built a client from the current
    # config; a shared worker must not keep serving the old one.
    first = create_chroma_client(config)
    assert first.embedding_model == "default"
    pid = first.worker_pid
    data = json.loads(open(config).read())
    data["semantic_search"]["embedding_model"] = "openai"
    data["semantic_search"]["embedding_config"] = {"api_key": "sk-test"}
    with open(config, "w") as f:
        json.dump(data, f)
    os.utime(config, (time.time() + 5, time.time() + 5))

    second = create_chroma_client(config)

    assert second is not first
    assert first.worker_pid is None and pid is not None
    assert create_chroma_client(config) is second


def test_calls_run_in_the_worker_not_in_this_process(config):
    client = create_chroma_client(config)

    assert client.get_collection_info()["count"] == N
    assert client.get_existing_ids(["ITEM0001#0", "NOPE#0"]) == {"ITEM0001#0"}
    assert client.embedding_model == "default"
    assert client.worker_pid not in (None, os.getpid())
    assert _in_process_chroma_systems() == 0


def test_exceptions_come_back_with_their_type(config):
    client = create_chroma_client(config)

    with pytest.raises(AttributeError):
        client.no_such_method()
    assert getattr(client, "no_such_attribute", "fallback") == "fallback"
    assert client.get_collection_info()["count"] == N  # the worker survived


def test_iterators_are_streamed_and_can_interleave_with_writes(config):
    client = create_chroma_client(config)
    seen = []

    for ids, metadatas in client.iter_metadatas(batch_size=7):
        seen += ids
        client.update_metadatas(ids, [{**m, "touched": True} for m in metadatas])

    assert sorted(seen) == sorted(f"ITEM{i:04d}#0" for i in range(N))
    assert client.get_document_metadata("ITEM0003")["touched"] is True


def test_a_value_that_cannot_cross_the_boundary_is_an_error_not_a_dead_worker(config):
    client = create_chroma_client(config)
    pid = client.get_collection_info() and client.worker_pid

    with pytest.raises(TypeError, match="collection"):
        client.collection

    assert client.get_collection_info()["count"] == N
    assert client.worker_pid == pid


@pytest.mark.skipif(sys.platform == "win32", reason="kills the worker with POSIX signals")
def test_idle_worker_exits_and_the_next_call_starts_a_new_one(config, monkeypatch):
    monkeypatch.setattr(chroma_worker, "_shared", {})
    client = chroma_worker.shared_client(config, idle_minutes=0.5 / 60)  # half a second
    client.get_collection_info()
    first = client.worker_pid

    deadline = time.monotonic() + 10
    while client.worker_pid is not None and time.monotonic() < deadline:
        time.sleep(0.1)
    assert client.worker_pid is None
    with pytest.raises(ProcessLookupError):
        os.kill(first, 0)

    assert client.get_collection_info()["count"] == N
    assert client.worker_pid not in (None, first)


@pytest.mark.skipif(sys.platform == "win32", reason="kills the worker with POSIX signals")
def test_a_worker_that_dies_is_replaced_on_the_next_call(config):
    client = create_chroma_client(config)
    client.get_collection_info()
    os.kill(client.worker_pid, signal.SIGKILL)
    time.sleep(0.2)

    assert client.get_collection_info()["count"] == N  # a fresh worker answers


@pytest.mark.skipif(sys.platform == "win32", reason="kills the worker with POSIX signals")
def test_a_call_in_flight_when_the_worker_dies_raises(config, monkeypatch):
    client = create_chroma_client(config)
    client.get_collection_info()
    pid = client.worker_pid

    def die_mid_call(stream, message):
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        raise BrokenPipeError

    monkeypatch.setattr(chroma_worker, "_send", die_mid_call)
    with pytest.raises(WorkerExited):
        client.get_collection_info()


@pytest.mark.parametrize("idle", [{"idle_unload_minutes": 0}, {}], ids=["zero", "unset"])
def test_zero_or_unset_idle_minutes_keeps_chromadb_in_process(tmp_path, monkeypatch, idle):
    monkeypatch.setattr(chroma_worker, "_enabled", True)
    monkeypatch.setattr(chroma_worker, "_shared", {})
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"semantic_search": {
        "persist_directory": str(tmp_path / "chroma_db"),
        **idle,
    }}))

    assert isinstance(create_chroma_client(str(path)), ChromaClient)
    SharedSystemClient.clear_system_cache()


def test_outside_the_server_chromadb_stays_in_process(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"semantic_search": {"persist_directory": str(tmp_path / "chroma_db")}}))

    assert isinstance(create_chroma_client(str(path)), ChromaClient)
    SharedSystemClient.clear_system_cache()


class _NeedsKeywords(Exception):
    """Pickles fine, cannot be unpickled: like the openai SDK's APIStatusError."""

    def __init__(self, message, *, response, body):
        super().__init__(message)


def test_an_exception_that_cannot_be_unpickled_still_reaches_the_caller_as_an_error(monkeypatch):
    import io

    class Boom:
        def fail(self):
            raise _NeedsKeywords("429 rate limited", response=None, body=None)

    monkeypatch.setattr("zotero_mcp.chroma_client.create_chroma_client", lambda config_path: Boom())
    requests, replies = io.BytesIO(), io.BytesIO()
    chroma_worker._send(requests, ("call", "fail", (), {}))
    requests.seek(0)

    chroma_worker._serve(None, requests, replies)

    replies.seek(0)
    status, error = chroma_worker._receive(replies)  # must not raise while reading
    assert status == "error"
    assert "429 rate limited" in str(error)


# --- the embedding function, which the server reads and calls itself ----------
#
# Indexing asks the client for ``embedding_function``: its
# ``max_parallel_requests`` picks the streaming pipeline (on by default for
# OpenAI and Gemini) and ``chars_per_token`` sizes passages. Forwarded by
# name, the attribute became a callable that runs in the worker with no such
# attributes, so every server-side update silently lost its parallelism.


@pytest.fixture()
def openai_config(config, tmp_path):
    """A config for an OpenAI index. No embedding call may leave this machine:
    the base_url is a closed local port, so a call made by mistake in the
    worker (which the test patches cannot reach) fails fast instead."""
    path = tmp_path / "openai_config.json"
    path.write_text(json.dumps({"semantic_search": {
        "persist_directory": str(tmp_path / "openai_chroma_db"),
        "embedding_model": "openai",
        "embedding_config": {"api_key": "sk-test", "base_url": "http://127.0.0.1:9"},
        "idle_unload_minutes": 10,
    }}))
    return str(path)


def test_the_embedding_function_reports_the_providers_own_settings(openai_config):
    ef = create_chroma_client(openai_config).embedding_function

    assert ef.max_parallel_requests == 4  # OpenAI's default: the streaming pipeline
    assert ef.chars_per_token == 3
    assert getattr(ef, "no_such_setting", "fallback") == "fallback"


def test_a_setting_is_read_from_the_worker_once(openai_config, monkeypatch):
    client = create_chroma_client(openai_config)
    ef = client.embedding_function
    assert ef.chars_per_token == 3

    def no_more_requests(message):
        raise AssertionError(f"asked the worker again: {message}")

    monkeypatch.setattr(client, "_request", no_more_requests)
    assert ef.chars_per_token == 3


def test_the_embedding_function_embeds_in_this_process_so_requests_can_overlap(openai_config, monkeypatch):
    seen = []

    def fake_create(model, config):
        seen.append((model, config.get("api_key")))
        return lambda documents: [[float(len(d))] for d in documents]

    monkeypatch.setattr("zotero_mcp.embeddings.registry.create_embedding_function", fake_create)
    client = create_chroma_client(openai_config)

    assert client.embedding_function(["ab", "abc"]) == [[2.0], [3.0]]
    assert seen == [("openai", "sk-test")]
    client.embedding_function(["x"])
    assert len(seen) == 1  # built once


def test_a_streaming_update_runs_through_the_worker(openai_config, monkeypatch):
    from zotero_mcp import semantic_search

    monkeypatch.setattr(
        "zotero_mcp.embeddings.registry.create_embedding_function",
        lambda model, config: (lambda documents: [[float(len(d)), 1.0] for d in documents]),
    )
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    client = create_chroma_client(openai_config)
    search = semantic_search.ZoteroSemanticSearch(chroma_client=client, config_path=openai_config)
    monkeypatch.setattr(search, "_save_update_config", lambda *a, **k: None)
    items = [
        {"key": f"ITEM{i:04d}", "data": {"title": f"Title {i}", "itemType": "journalArticle",
                                         "abstractNote": f"Abstract {i}", "creators": []}}
        for i in range(10)
    ]
    monkeypatch.setattr(search, "_get_items_from_source", lambda **kwargs: list(items))
    streamed = []
    original = search._stream_index_items
    monkeypatch.setattr(search, "_stream_index_items", lambda *a, **k: streamed.append(1) or original(*a, **k))

    stats = search.update_database()

    assert streamed, "the update did not use the streaming pipeline"
    assert stats["added_items"] == 10 and stats["errors"] == 0
    assert client.get_collection_info()["count"] == 10
