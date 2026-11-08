"""Opt-in background preload of the embedding model (ZOTERO_MCP_PRELOAD_EMBEDDINGS).

The first semantic search of a process imports torch and sentence-transformers,
loads the model and runs its first encode. With the variable set, ``serve``
does that in a background thread once the server has answered ``initialize``.
On Windows over stdio, that thread imports native libraries while the server
is blocked reading stdin, which is safe only because stdin has been moved off
descriptor 0 (#485, see test_stdin_off_descriptor_0.py).
"""

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from zotero_mcp import cli
from zotero_mcp.config_light import embedding_preload_requested


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", " on "])
def test_flag_turns_the_preload_on(value):
    assert embedding_preload_requested({"ZOTERO_MCP_PRELOAD_EMBEDDINGS": value}) is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_flag_is_off_otherwise(value):
    assert embedding_preload_requested({"ZOTERO_MCP_PRELOAD_EMBEDDINGS": value}) is False
    assert embedding_preload_requested({}) is False


@pytest.mark.parametrize(
    "flag,raw_config,expected",
    [
        ("1", '{"semantic_search": {"embedding_model": "qwen"}}', True),
        ("1", '{"semantic_search": {}}', False),  # semantic search not set up
        ("1", None, False),  # no config file
        ("", '{"semantic_search": {"embedding_model": "qwen"}}', False),  # default off
    ],
)
def test_preload_needs_the_flag_and_a_semantic_config(tmp_path, monkeypatch, flag, raw_config, expected):
    home = tmp_path / "home"
    (home / ".config" / "zotero-mcp").mkdir(parents=True)
    if raw_config is not None:
        (home / ".config" / "zotero-mcp" / "config.json").write_text(raw_config)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setenv("ZOTERO_MCP_PRELOAD_EMBEDDINGS", flag)
    assert cli._embedding_preload_wanted() is expected


def test_preload_starts_once_and_only_after_initialize(monkeypatch):
    pytest.importorskip("chromadb")
    from fastmcp import Client, FastMCP

    from zotero_mcp import semantic_search

    calls = []
    done = threading.Event()

    def fake_preload(config_path):
        calls.append((config_path, threading.current_thread().name))
        done.set()
        return True

    monkeypatch.setattr(semantic_search, "preload_embedding_model", fake_preload)
    mcp = FastMCP("preload-test")

    @mcp.tool
    def ping() -> str:
        return "pong"

    cli._preload_embeddings_after_initialize(mcp)
    time.sleep(0.2)
    assert calls == []  # registering the middleware starts nothing

    async def two_sessions():
        for _ in range(2):
            async with Client(mcp) as client:
                assert (await client.call_tool("ping")).data == "pong"

    asyncio.run(two_sessions())
    assert done.wait(10)
    time.sleep(0.2)
    assert len(calls) == 1
    assert calls[0][1] == "zmcp-embedding-preload"
    assert calls[0][0].endswith("config.json")


class _FakeEmbeddingFunction:
    def __init__(self, provider):
        self.provider = provider
        self.inputs = []

    def name(self):
        return self.provider

    def __call__(self, texts):
        self.inputs.append(texts)
        return [[0.0, 1.0] for _ in texts]


def _patch_search(monkeypatch, embedding_function):
    pytest.importorskip("chromadb")
    from zotero_mcp import semantic_search

    search = SimpleNamespace(chroma_client=SimpleNamespace(embedding_function=embedding_function))
    monkeypatch.setattr(semantic_search, "create_semantic_search", lambda config_path: search)
    return semantic_search


@pytest.mark.parametrize("provider", ["huggingface", "default"])
def test_local_models_are_loaded_and_warmed(monkeypatch, provider):
    ef = _FakeEmbeddingFunction(provider)
    semantic_search = _patch_search(monkeypatch, ef)
    assert semantic_search.preload_embedding_model("config.json") is True
    assert ef.inputs == [["warm-up"]]


@pytest.mark.parametrize("provider", ["openai", "gemini", "ollama"])
def test_api_providers_get_no_warm_up_request(monkeypatch, provider):
    ef = _FakeEmbeddingFunction(provider)
    semantic_search = _patch_search(monkeypatch, ef)
    assert semantic_search.preload_embedding_model("config.json") is False
    assert ef.inputs == []


def test_a_failed_preload_never_raises(monkeypatch):
    pytest.importorskip("chromadb")
    from zotero_mcp import semantic_search

    def broken(config_path):
        raise RuntimeError("index unreadable")

    monkeypatch.setattr(semantic_search, "create_semantic_search", broken)
    assert semantic_search.preload_embedding_model("config.json") is False


def test_worker_mode_loads_nothing_in_the_server(monkeypatch):
    """With idle_unload_minutes, the model lives in the ChromaDB worker process."""
    pytest.importorskip("chromadb")
    from zotero_mcp import semantic_search
    from zotero_mcp.chroma_worker import WorkerChromaClient

    worker = WorkerChromaClient("config.json", idle_seconds=600)  # starts nothing until called
    search = SimpleNamespace(chroma_client=worker)
    monkeypatch.setattr(semantic_search, "create_semantic_search", lambda config_path: search)

    assert semantic_search.preload_embedding_model("config.json") is False
    assert worker._proc is None  # the worker was not started
