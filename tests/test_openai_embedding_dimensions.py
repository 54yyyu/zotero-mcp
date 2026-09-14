"""Matryoshka ``dimensions`` support on the OpenAI embedding function.

``text-embedding-3-*`` models accept a ``dimensions`` argument that returns
shorter vectors with near-lossless semantic quality (≈67% ChromaDB storage
saved at 1024 vs the 3072 default). These tests pin the contract:

- the parameter is forwarded to ``embeddings.create`` when set, and omitted
  when it is not — some OpenAI-compatible backends reject it outright;
- it round-trips through ``get_config``/``build_from_config`` so a persisted
  collection rebuilt by name keeps its dimensionality.

Construction-based tests are guarded with ``importorskip("openai")``; the
request-shape tests build the instance via ``__new__`` with a fake client,
so they run without the optional ``openai`` package (CI does not install
it).
"""

import threading

import pytest

pytest.importorskip("chromadb")

from zotero_mcp.embeddings.providers.openai import OpenAIEmbeddingFunction  # noqa: E402


def _make(dimensions=None):
    """Build an OpenAIEmbeddingFunction with a fake client, bypassing __init__."""
    ef = OpenAIEmbeddingFunction.__new__(OpenAIEmbeddingFunction)
    ef.model_name = "text-embedding-3-large"
    ef.base_url = None
    ef.request_batch_size = 64
    ef.rate_limit_rps = None
    ef.dimensions = dimensions
    ef.max_parallel_requests = 1
    ef.max_retries = 0
    ef._rate_lock = threading.Lock()
    ef._last_request_ts = 0.0

    calls = []

    class _Resp:
        def __init__(self, items):
            self.data = [type("D", (), {"embedding": [float(x)]}) for x in items]

    class _Embeddings:
        @staticmethod
        def create(**kwargs):
            calls.append(kwargs)
            return _Resp(kwargs["input"])

    class _Client:
        embeddings = _Embeddings()

    ef.client = _Client()
    return ef, calls


def test_dimensions_forwarded_to_create_when_set():
    ef, calls = _make(dimensions=1024)
    ef([0, 1])
    assert calls and all(c["dimensions"] == 1024 for c in calls)


def test_dimensions_omitted_when_none():
    """Backends without Matryoshka support would 400 on the parameter."""
    ef, calls = _make(dimensions=None)
    ef([0])
    assert len(calls) == 1
    assert "dimensions" not in calls[0]


def test_get_config_roundtrips_dimensions():
    ef, _ = _make(dimensions=1024)
    assert ef.get_config()["dimensions"] == 1024


def test_build_from_config_restores_dimensions(monkeypatch):
    """A persisted collection rebuilt by name must keep its dimensionality."""
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-network")

    ef = OpenAIEmbeddingFunction.build_from_config(
        {"model_name": "text-embedding-3-large", "dimensions": 1024}
    )
    assert ef.dimensions == 1024


def test_registry_factory_forwards_dimensions(monkeypatch):
    """embedding_config.dimensions reaches the constructed function."""
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-network")

    from zotero_mcp.embeddings.registry import create_embedding_function

    ef = create_embedding_function(
        "openai", {"model_name": "text-embedding-3-large", "dimensions": 1024}
    )
    assert ef.dimensions == 1024
