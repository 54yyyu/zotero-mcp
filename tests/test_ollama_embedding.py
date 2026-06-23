"""Unit tests for ``OllamaEmbeddingFunction``'s HTTP contract.

Ollama deprecated the single-input ``/api/embeddings`` route in favour of
``/api/embed``, which accepts a batch under ``input`` and returns a list of
vectors under ``embeddings``. These tests pin that contract without requiring a
live Ollama server.

Note: ChromaDB's ``EmbeddingFunction`` base wraps ``__call__`` with
``validate_embeddings(normalize_embeddings(...))``, so the values returned to
the caller are numpy arrays. Assertions below account for that.
"""

import pytest

requests = pytest.importorskip("requests")
# OllamaEmbeddingFunction subclasses chromadb's EmbeddingFunction; numpy ships
# with chromadb and is used to compare the (normalized) return value.
pytest.importorskip("chromadb")
import numpy as np  # noqa: E402

from zotero_mcp.chroma_client import OllamaEmbeddingFunction  # noqa: E402


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_call_uses_api_embed_and_sends_one_batched_request(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        # /api/embed returns one vector per input, in order.
        return _FakeResponse({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    monkeypatch.setattr(requests, "post", fake_post)

    ef = OllamaEmbeddingFunction(
        model_name="nomic-embed-text", base_url="http://localhost:11434"
    )
    result = ef(["alpha", "beta"])

    # The whole batch goes out in a single request to the new endpoint, with the
    # texts under ``input`` (not one ``prompt`` request per document).
    assert len(calls) == 1, "the whole batch must be sent in a single request"
    assert calls[0]["url"] == "http://localhost:11434/api/embed"
    assert calls[0]["json"] == {
        "model": "nomic-embed-text",
        "input": ["alpha", "beta"],
    }
    assert np.allclose(np.asarray(result, dtype=float), [[0.1, 0.2], [0.3, 0.4]])


def test_call_empty_input_issues_no_http_request(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append((a, k)))

    # The guard must short-circuit *before* any HTTP request. (ChromaDB's EF
    # wrapper may separately reject an empty result; that is orthogonal to the
    # guard under test and varies by ChromaDB version, so it is not asserted.)
    try:
        OllamaEmbeddingFunction()([])
    except Exception:
        pass

    assert calls == [], "empty input must not issue an HTTP request"


def test_call_raises_when_response_lacks_embeddings(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: _FakeResponse({"error": "model not found"}),
    )

    with pytest.raises(ValueError, match="no 'embeddings' field"):
        OllamaEmbeddingFunction()(["x"])
