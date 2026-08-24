"""Live tests against a real local Ollama server.

Requires ``ollama serve`` running with ``nomic-embed-text`` pulled (768-dim
model); see the ``ollama_available`` fixture in ``tests/live/conftest.py``.
Collected but skipped by default — run with
``ZOTERO_MCP_LIVE_TESTS=1 uv run pytest tests/live/test_ollama_live.py -v``.

The machine this suite runs on may have a large embedding job in flight
(CPU-loaded), so most tests carry a generous ``@pytest.mark.timeout`` above
the global 30s default.
"""

import sys
import time

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")
import requests  # noqa: E402

from zotero_mcp.embeddings.providers.ollama import OllamaEmbeddingFunction  # noqa: E402
from zotero_mcp.embeddings.registry import create_embedding_function  # noqa: E402

MODEL = "nomic-embed-text"
DIM = 768


def _texts(n: int) -> list[str]:
    return [f"live ollama test document number {i}" for i in range(n)]


def _as_float_lists(vectors) -> list[list[float]]:
    return [[float(x) for x in vec] for vec in vectors]


@pytest.mark.timeout(30)
def test_registry_builds_ollama_ef(ollama_available):
    """create_embedding_function("ollama", ...) is the exact production path
    ChromaClient uses to build the realtime EF."""
    ef = create_embedding_function("ollama", {"model_name": MODEL, "base_url": ollama_available})
    assert isinstance(ef, OllamaEmbeddingFunction)
    assert ef.name() == "ollama"


@pytest.mark.timeout(45)
def test_sub_batching_issues_expected_request_count(ollama_available, count_requests_post):
    ef = OllamaEmbeddingFunction(model_name=MODEL, base_url=ollama_available, request_batch_size=3)
    texts = _texts(12)

    vectors = ef(texts)

    assert len(count_requests_post) == 4, "12 texts / batch 3 must issue exactly 4 requests"
    assert len(vectors) == 12
    for vec in vectors:
        assert len(vec) == DIM


@pytest.mark.timeout(60)
def test_parallel_matches_sequential(ollama_available, count_requests_post):
    """max_parallel_requests=4 must return bit-identical results to the
    sequential path, addressed by index so request completion order never
    matters, and must still issue exactly 4 requests (one per sub-batch)."""
    texts = _texts(12)

    sequential_ef = OllamaEmbeddingFunction(model_name=MODEL, base_url=ollama_available, request_batch_size=3)
    sequential = _as_float_lists(sequential_ef(texts))
    count_requests_post.clear()

    parallel_ef = OllamaEmbeddingFunction(
        model_name=MODEL, base_url=ollama_available, request_batch_size=3, max_parallel_requests=4
    )
    parallel = _as_float_lists(parallel_ef(texts))

    assert len(count_requests_post) == 4
    assert sequential == parallel


@pytest.mark.timeout(60)
def test_rate_limiter_paces_requests(ollama_available):
    """rate_limit_rps=2 with burst=max(4, max_parallel_requests)=4: the first
    4 embed_query calls spend burst tokens immediately, calls 5-6 must wait
    on the token bucket. Six calls should take at least ~1s total."""
    ef = OllamaEmbeddingFunction(model_name=MODEL, base_url=ollama_available, rate_limit_rps=2)

    start = time.monotonic()
    for _ in range(6):
        vec = ef.embed_query("ping")
        assert len(vec) == DIM
    elapsed = time.monotonic() - start

    assert elapsed >= 0.9, f"expected rate-limited pacing to take >= 0.9s, took {elapsed:.3f}s"


@pytest.mark.timeout(30)
def test_embed_query_returns_768_floats(ollama_available):
    ef = OllamaEmbeddingFunction(model_name=MODEL, base_url=ollama_available)
    vec = ef.embed_query("a live embedding query")
    assert len(vec) == DIM
    assert all(isinstance(x, float) for x in vec)


@pytest.mark.timeout(30)
def test_config_round_trip(ollama_available, cosine_similarity):
    ef = OllamaEmbeddingFunction(
        model_name=MODEL, base_url=ollama_available, request_batch_size=5, max_parallel_requests=2
    )
    cfg = ef.get_config()
    rebuilt = OllamaEmbeddingFunction.build_from_config(cfg)

    assert rebuilt.model_name == ef.model_name
    assert rebuilt.base_url == ef.base_url
    assert rebuilt.request_batch_size == ef.request_batch_size
    assert rebuilt.max_parallel_requests == ef.max_parallel_requests

    text = "config round trip probe text"
    v1 = ef.embed_query(text)
    v2 = rebuilt.embed_query(text)
    assert cosine_similarity(v1, v2) >= 0.999


@pytest.mark.timeout(15)
def test_unknown_model_fails_fast(ollama_available):
    """A 404 (unknown model) is not retryable, so this must fail almost
    immediately rather than burning through max_retries."""
    ef = OllamaEmbeddingFunction(model_name="definitely-not-a-model-xyz", base_url=ollama_available)

    start = time.monotonic()
    with pytest.raises(requests.HTTPError):
        ef.embed_query("hello")
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"expected fail-fast (no retry storm), took {elapsed:.3f}s"
