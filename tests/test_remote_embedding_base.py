"""Unit tests for RemoteEmbeddingFunction: sub-batching, parallel ordering,
retries, and the config round-trip for the three concrete providers.

No network, no real sleeping — retry delays go through a limiter built with
an injected no-op sleep, and providers are built with dummy api keys so their
constructors (which only build an SDK client object, never call the network)
succeed hermetically.
"""

import random
import sys
import time

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")

from zotero_mcp.embeddings.base import RemoteEmbeddingFunction  # noqa: E402
from zotero_mcp.embeddings.ratelimit import AdaptiveRateLimiter  # noqa: E402

# ---------------------------------------------------------------------------
# A minimal concrete subclass for exercising the base class in isolation.
# ---------------------------------------------------------------------------


class _FakeRemoteEF(RemoteEmbeddingFunction):
    default_request_batch_size = 2

    def __init__(self, embed_batch, classify_error=None, **common_kwargs):
        self._embed_batch_impl = embed_batch
        self._classify_error_impl = classify_error
        self._init_common(
            model_name="fake-model",
            base_url=None,
            request_batch_size=common_kwargs.pop("request_batch_size", None),
            rate_limit_rps=common_kwargs.pop("rate_limit_rps", None),
            max_parallel_requests=common_kwargs.pop("max_parallel_requests", None),
            max_retries=common_kwargs.pop("max_retries", None),
        )

    def _embed_batch(self, texts, is_query=False):
        return self._embed_batch_impl(texts, is_query=is_query)

    def _classify_error(self, exc):
        if self._classify_error_impl is not None:
            return self._classify_error_impl(exc)
        return super()._classify_error(exc)


def _no_sleep(_seconds):
    pass


def _fast_limiter():
    """A limiter that never actually paces or waits, for tests that only
    care about ordering/retry logic, not rate limiting."""
    return AdaptiveRateLimiter(initial_rps=None, sleep=_no_sleep)


# ---------------------------------------------------------------------------
# Ordering under parallelism
# ---------------------------------------------------------------------------


def test_output_order_preserved_with_parallel_workers():
    """max_parallel_requests=4 with randomized completion delays must not
    scramble output order: index-addressed slots reassemble input order."""

    def embed_batch(texts, is_query=False):
        # Small random delay to encourage out-of-order completion across
        # sub-batches; kept tiny so the test stays fast.
        time.sleep(random.uniform(0.001, 0.01))
        return [[float(t)] for t in texts]

    ef = _FakeRemoteEF(embed_batch, request_batch_size=3, max_parallel_requests=4)
    ef.limiter = _fast_limiter()

    inputs = list(range(37))
    result = ef([str(i) for i in inputs])

    assert [r[0] for r in result] == [float(i) for i in inputs]


def test_sequential_when_max_parallel_requests_is_1():
    calls = []

    def embed_batch(texts, is_query=False):
        calls.append(list(texts))
        return [[float(t)] for t in texts]

    ef = _FakeRemoteEF(embed_batch, request_batch_size=2, max_parallel_requests=1)
    ef.limiter = _fast_limiter()

    result = ef(["0", "1", "2", "3", "4"])
    assert calls == [["0", "1"], ["2", "3"], ["4"]]
    assert [r[0] for r in result] == [0.0, 1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# Sub-batch slicing
# ---------------------------------------------------------------------------


def test_request_batch_size_none_sends_one_request():
    calls = []

    def embed_batch(texts, is_query=False):
        calls.append(list(texts))
        return [[0.0] for _ in texts]

    ef = _FakeRemoteEF(embed_batch, request_batch_size=None)
    ef.limiter = _fast_limiter()
    # Force single-request mode regardless of the test class's own default
    # (both the resolved instance value and the class-default fallback used
    # by __call__ when request_batch_size is falsy).
    ef.request_batch_size = None
    ef.default_request_batch_size = None

    ef(["a", "b", "c"])
    assert calls == [["a", "b", "c"]]


def test_empty_input_issues_no_requests():
    """Our __call__ short-circuits to [] before ever calling _embed_batch.

    ChromaDB's own EmbeddingFunction wrapper separately rejects an empty
    *result* (``normalize_embeddings`` raises on a zero-length list) — that
    is orthogonal to the guard under test here and varies by ChromaDB
    version (see the same caveat in tests/test_ollama_embedding.py), so it
    is not asserted; only "no request was issued" is.
    """
    calls = []

    def embed_batch(texts, is_query=False):
        calls.append(list(texts))
        return []

    ef = _FakeRemoteEF(embed_batch, request_batch_size=None)
    ef.limiter = _fast_limiter()

    try:
        ef([])
    except Exception:
        pass
    assert calls == []


def test_subbatch_slicing_math():
    calls = []

    def embed_batch(texts, is_query=False):
        calls.append(len(texts))
        return [[0.0] for _ in texts]

    ef = _FakeRemoteEF(embed_batch, request_batch_size=10, max_parallel_requests=1)
    ef.limiter = _fast_limiter()

    ef([str(i) for i in range(25)])
    assert calls == [10, 10, 5]


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------


class _Retryable(Exception):
    pass


class _NonRetryable(Exception):
    pass


def test_retry_succeeds_on_second_attempt_and_honors_retry_after():
    attempts = []

    def embed_batch(texts, is_query=False):
        attempts.append(texts)
        if len(attempts) == 1:
            raise _Retryable("throttled")
        return [[1.0] for _ in texts]

    def classify(exc):
        if isinstance(exc, _Retryable):
            return True, 3.5
        return False, None

    ef = _FakeRemoteEF(embed_batch, classify_error=classify, request_batch_size=None, max_retries=2)
    # A clock/sleep pair where sleeping actually advances the fake clock (as
    # a real time.sleep would advance real monotonic time) — otherwise the
    # retry's own post-throttle acquire() sees zero elapsed time and adds an
    # unrelated pacing wait on top of the honored retry_after.
    waits = []
    clock = {"t": 0.0}

    def fake_clock():
        return clock["t"]

    def fake_sleep(seconds):
        waits.append(seconds)
        clock["t"] += seconds

    ef.limiter = AdaptiveRateLimiter(initial_rps=None, clock=fake_clock, sleep=fake_sleep)

    result = ef(["x"])
    assert len(attempts) == 2
    assert result == [[1.0]]
    assert waits == [3.5]


def test_non_retryable_error_propagates_immediately():
    attempts = []

    def embed_batch(texts, is_query=False):
        attempts.append(texts)
        raise _NonRetryable("bad request")

    def classify(exc):
        return False, None

    ef = _FakeRemoteEF(embed_batch, classify_error=classify, request_batch_size=None, max_retries=5)
    ef.limiter = _fast_limiter()

    with pytest.raises(_NonRetryable):
        ef(["x"])
    assert len(attempts) == 1


def test_max_retries_exhaustion_reraises():
    attempts = []

    def embed_batch(texts, is_query=False):
        attempts.append(texts)
        raise _Retryable("still throttled")

    def classify(exc):
        return True, 0.001

    ef = _FakeRemoteEF(embed_batch, classify_error=classify, request_batch_size=None, max_retries=3)
    ef.limiter = AdaptiveRateLimiter(initial_rps=None, sleep=_no_sleep)

    with pytest.raises(_Retryable):
        ef(["x"])
    # Initial attempt + 3 retries = 4 total calls.
    assert len(attempts) == 4


# ---------------------------------------------------------------------------
# Config round-trip for the three concrete providers
# ---------------------------------------------------------------------------


def test_openai_build_from_config_roundtrips_all_fields(monkeypatch):
    pytest.importorskip("openai")
    from zotero_mcp.chroma_client import OpenAIEmbeddingFunction

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-network")
    ef = OpenAIEmbeddingFunction(
        model_name="text-embedding-3-large",
        request_batch_size=32,
        rate_limit_rps=7.5,
        max_parallel_requests=3,
        max_retries=2,
    )
    cfg = ef.get_config()
    rebuilt = OpenAIEmbeddingFunction.build_from_config({**cfg, "api_key": "test-key-no-network"})
    rebuilt_cfg = rebuilt.get_config()
    assert rebuilt_cfg["model_name"] == "text-embedding-3-large"
    assert rebuilt_cfg["request_batch_size"] == 32
    assert rebuilt_cfg["rate_limit_rps"] == 7.5
    assert rebuilt_cfg["max_parallel_requests"] == 3
    assert rebuilt_cfg["max_retries"] == 2


def test_openai_build_from_config_defaults_old_persisted_config(monkeypatch):
    pytest.importorskip("openai")
    from zotero_mcp.chroma_client import OpenAIEmbeddingFunction

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-network")
    # Pre-refactor persisted shape: only model_name/base_url.
    old_config = {"model_name": "text-embedding-3-small", "base_url": None}
    ef = OpenAIEmbeddingFunction.build_from_config(old_config)
    cfg = ef.get_config()
    assert cfg["request_batch_size"] == OpenAIEmbeddingFunction.DEFAULT_REQUEST_BATCH_SIZE
    assert cfg["rate_limit_rps"] is None
    assert cfg["max_parallel_requests"] == OpenAIEmbeddingFunction.DEFAULT_MAX_PARALLEL_REQUESTS
    assert cfg["max_retries"] == 5


def test_gemini_build_from_config_roundtrips_all_fields(monkeypatch):
    pytest.importorskip("google.genai")
    from zotero_mcp.chroma_client import GeminiEmbeddingFunction

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-no-network")
    ef = GeminiEmbeddingFunction(
        model_name="gemini-embedding-001",
        request_batch_size=50,
        rate_limit_rps=3.0,
        max_parallel_requests=2,
        max_retries=1,
    )
    cfg = ef.get_config()
    rebuilt = GeminiEmbeddingFunction.build_from_config({**cfg, "api_key": "test-key-no-network"})
    rebuilt_cfg = rebuilt.get_config()
    assert rebuilt_cfg["model_name"] == "gemini-embedding-001"
    assert rebuilt_cfg["request_batch_size"] == 50
    assert rebuilt_cfg["rate_limit_rps"] == 3.0
    assert rebuilt_cfg["max_parallel_requests"] == 2
    assert rebuilt_cfg["max_retries"] == 1


def test_gemini_build_from_config_defaults_old_persisted_config(monkeypatch):
    pytest.importorskip("google.genai")
    from zotero_mcp.chroma_client import GeminiEmbeddingFunction

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-no-network")
    old_config = {"model_name": "gemini-embedding-001", "base_url": None}
    ef = GeminiEmbeddingFunction.build_from_config(old_config)
    cfg = ef.get_config()
    assert cfg["request_batch_size"] == GeminiEmbeddingFunction.GEMINI_MAX_BATCH
    assert cfg["rate_limit_rps"] is None
    assert cfg["max_parallel_requests"] == GeminiEmbeddingFunction.DEFAULT_MAX_PARALLEL_REQUESTS
    assert cfg["max_retries"] == 5



def test_ollama_build_from_config_roundtrips_all_fields():
    from zotero_mcp.chroma_client import OllamaEmbeddingFunction

    ef = OllamaEmbeddingFunction(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434",
        request_batch_size=16,
        rate_limit_rps=2.0,
        max_parallel_requests=4,
        max_retries=3,
    )
    cfg = ef.get_config()
    rebuilt = OllamaEmbeddingFunction.build_from_config(cfg)
    rebuilt_cfg = rebuilt.get_config()
    assert rebuilt_cfg["model_name"] == "nomic-embed-text"
    assert rebuilt_cfg["base_url"] == "http://localhost:11434"
    assert rebuilt_cfg["request_batch_size"] == 16
    assert rebuilt_cfg["rate_limit_rps"] == 2.0
    assert rebuilt_cfg["max_parallel_requests"] == 4
    assert rebuilt_cfg["max_retries"] == 3


def test_ollama_build_from_config_defaults_old_persisted_config():
    from zotero_mcp.chroma_client import OllamaEmbeddingFunction

    old_config = {"model_name": "qwen3-embedding", "base_url": "http://localhost:11434"}
    ef = OllamaEmbeddingFunction.build_from_config(old_config)
    cfg = ef.get_config()
    assert cfg["request_batch_size"] is None
    assert cfg["rate_limit_rps"] is None
    assert cfg["max_parallel_requests"] == 1
    assert cfg["max_retries"] == 5


# ---------------------------------------------------------------------------
# Removed-feature guard
# ---------------------------------------------------------------------------


def test_service_tier_fully_removed():
    """service_tier was removed: OpenAI's flex processing never covered the
    embeddings endpoint, and the SDK's embeddings.create rejects the kwarg
    with a TypeError. Old persisted configs may still carry the key —
    build_from_config must ignore it rather than crash."""
    pytest.importorskip("openai")
    from zotero_mcp.chroma_client import OpenAIEmbeddingFunction

    ef = OpenAIEmbeddingFunction.build_from_config(
        {"api_key": "test-key-no-network", "service_tier": "flex"}
    )
    assert not hasattr(ef, "service_tier")
    assert "service_tier" not in ef.get_config()
