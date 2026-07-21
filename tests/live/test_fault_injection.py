"""Fault-injection tests against a local stub HTTP server (pytest-httpserver).

Unlike the rest of ``tests/live/``, these tests never talk to a real network
service — the server is a local stub bound to ``127.0.0.1`` on an ephemeral
port. They still live under ``tests/live/`` (and are still gated behind
``ZOTERO_MCP_LIVE_TESTS=1`` by the module-wide hook in ``conftest.py``, since
they exercise the same retry/backoff machinery the rest of this suite does)
but deliberately do NOT use the ``ollama_available`` fixture — no real Ollama
server needs to be running for these to pass.

Collected but skipped by default; run with
``ZOTERO_MCP_LIVE_TESTS=1 uv run pytest tests/live/test_fault_injection.py -v``.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")

import requests  # noqa: E402
from pytest_httpserver import HTTPServer  # noqa: E402
from werkzeug.wrappers import Request, Response  # noqa: E402

from zotero_mcp.embeddings.providers.ollama import OllamaEmbeddingFunction  # noqa: E402
from zotero_mcp.embeddings.providers.openai import OpenAIEmbeddingFunction  # noqa: E402

OLLAMA_SUCCESS_BODY = {"embeddings": [[0.1, 0.2, 0.3]]}
OPENAI_SUCCESS_BODY = {
    "data": [{"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}],
    "model": "text-embedding-3-small",
    "object": "list",
    "usage": {"prompt_tokens": 1, "total_tokens": 1},
}


def _json_response(payload: dict[str, Any], status: int = 200, headers: dict[str, str] | None = None) -> Response:
    import json

    return Response(json.dumps(payload), status=status, content_type="application/json", headers=headers or {})


def _fail_then_succeed_handler(fail_status: int, fail_times: int = 1):
    """Return a stateful werkzeug handler: fails ``fail_times`` times with
    ``fail_status``, then serves a valid Ollama /api/embed success body.

    Stateful via a closure over a mutable counter (pytest-httpserver's
    ``respond_with_handler`` takes an arbitrary callable, so this is the
    documented way to script a sequence of responses for the same route).
    """
    state = {"count": 0}

    def handler(request: Request) -> Response:
        state["count"] += 1
        if state["count"] <= fail_times:
            return _json_response({"error": f"stub failure #{state['count']}"}, status=fail_status)
        return _json_response(OLLAMA_SUCCESS_BODY)

    return handler


def _always_fail_handler(status: int):
    def handler(request: Request) -> Response:
        return _json_response({"error": "stub always fails"}, status=status)

    return handler


@pytest.mark.timeout(15)
def test_ollama_429_then_success(httpserver: HTTPServer):
    """A single 429 followed by a 200 must be retried transparently.

    The rate limiter starts unarmed; its first backoff (no Retry-After header
    from Ollama's error path, since ``_classify_error`` never parses one) is
    ~1s (2**0), hence the generous timeout override.
    """
    httpserver.expect_request("/api/embed", method="POST").respond_with_handler(
        _fail_then_succeed_handler(fail_status=429)
    )

    ef = OllamaEmbeddingFunction(model_name="stub-model", base_url=httpserver.url_for(""), max_retries=1)
    vectors = ef(["hello"])

    # chromadb's EmbeddingFunction.__call__ wraps the return value as numpy
    # arrays (validate_embeddings), so compare via plain-list conversion
    # rather than `vectors == [[...]]` (numpy array equality is elementwise
    # and raises on a bare `assert` truth-value check).
    assert [list(v) for v in vectors] == [[0.1, 0.2, 0.3]]
    assert len(httpserver.log) == 2, "expected exactly 2 requests (1 failure + 1 success)"


@pytest.mark.timeout(15)
def test_ollama_500_then_success(httpserver: HTTPServer):
    """A single 5xx followed by a 200 must also be retried (Ollama's
    ``_classify_error`` treats any status >= 500 as retryable)."""
    httpserver.expect_request("/api/embed", method="POST").respond_with_handler(
        _fail_then_succeed_handler(fail_status=500)
    )

    ef = OllamaEmbeddingFunction(model_name="stub-model", base_url=httpserver.url_for(""), max_retries=1)
    vectors = ef(["hello"])

    assert [list(v) for v in vectors] == [[0.1, 0.2, 0.3]]
    assert len(httpserver.log) == 2, "expected exactly 2 requests (1 failure + 1 success)"


@pytest.mark.timeout(15)
def test_ollama_exhausted_retries_raises(httpserver: HTTPServer):
    """A server that always 429s must exhaust ``max_retries`` and raise
    ``requests.HTTPError`` after exactly ``max_retries + 1`` requests."""
    httpserver.expect_request("/api/embed", method="POST").respond_with_handler(_always_fail_handler(429))

    ef = OllamaEmbeddingFunction(model_name="stub-model", base_url=httpserver.url_for(""), max_retries=1)

    with pytest.raises(requests.HTTPError):
        ef(["hello"])

    assert len(httpserver.log) == 2, "expected exactly 2 requests (initial attempt + 1 retry, then raise)"


@pytest.mark.timeout(15)
def test_openai_honors_retry_after(httpserver: HTTPServer):
    """A 429 carrying ``Retry-After: 1`` must make our own retry loop
    (``RemoteEmbeddingFunction._embed_with_retry`` -> ``AdaptiveRateLimiter``)
    wait ~1s before the retry, and the retried request must succeed.

    Empirical findings (see probe scripts used during implementation, not
    checked in):
      - With ``base_url=httpserver.url_for("")`` (i.e. no ``/v1`` suffix,
        matching how ``OpenAIEmbeddingFunction.__init__`` passes ``base_url``
        straight through to ``openai.OpenAI(base_url=...)``), the SDK POSTs
        to ``/embeddings`` directly -- it does NOT insert a ``/v1`` segment
        itself; that only lives in the SDK's own *default* base URL
        (``https://api.openai.com/v1``). So the stub route must be
        registered at ``/embeddings``, not ``/v1/embeddings``.
      - ``openai.OpenAI`` defaults to its OWN internal retry loop
        (``max_retries=2``), which also inspects ``Retry-After`` and would
        retry a 429 *before* our code ever sees an exception -- so without
        disabling it, this test cannot tell whether the ~1s wait + 2nd
        request came from our ``AdaptiveRateLimiter``/``_classify_error``
        path or from the SDK's own opaque retry. Disabling it with
        ``ef.client = ef.client.with_options(max_retries=0)`` (no ``src/``
        change needed -- ``with_options`` returns a client copy) makes the
        first 429 propagate as ``openai.RateLimitError`` straight into
        ``RemoteEmbeddingFunction._embed_with_retry``, so the observed
        retry-and-succeed is provably OUR retry honoring ``Retry-After``.
    """
    state = {"count": 0}
    timestamps: list[float] = []

    def handler(request: Request) -> Response:
        timestamps.append(time.monotonic())
        state["count"] += 1
        if state["count"] == 1:
            return _json_response(
                {"error": {"message": "rate limited", "type": "rate_limit_error"}},
                status=429,
                headers={"Retry-After": "1"},
            )
        return _json_response(OPENAI_SUCCESS_BODY)

    httpserver.expect_request("/embeddings", method="POST").respond_with_handler(handler)

    ef = OpenAIEmbeddingFunction(
        model_name="text-embedding-3-small",
        api_key="sk-test-fake",
        base_url=httpserver.url_for(""),
        max_retries=1,
    )
    # See docstring above: disable the OpenAI SDK's own internal retry so
    # this test exercises only RemoteEmbeddingFunction's retry/backoff path.
    ef.client = ef.client.with_options(max_retries=0)

    vectors = ef(["hello"])

    assert [list(v) for v in vectors] == [[0.1, 0.2]]
    assert len(httpserver.log) == 2, "expected exactly 2 requests (1 throttle + 1 success)"
    assert state["count"] == 2
    gap = timestamps[1] - timestamps[0]
    assert gap >= 0.9, f"expected Retry-After (1s) to be honored, gap was {gap:.3f}s"
    assert gap < 5.0, f"gap suspiciously large ({gap:.3f}s) -- possible extra backoff stacking"
