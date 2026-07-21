"""Tests for the Voyage AI embedding provider (Phase 5 extensibility proof).

Voyage is the first entirely new provider added after the embedding-provider
refactor: it needed exactly one new module
(``embeddings/providers/voyage.py``) and one ``register_provider`` call in
``embeddings/registry.py`` — zero edits to ``semantic_search.py`` or
``cli.py``. The last test in this file asserts that claim directly by
reading those two modules' source text.

Mirrors the patterns in ``tests/test_provider_registry.py`` (stub
``ChromaClient`` for env-merge tests) and
``tests/test_embedding_function_registration.py``/``tests/test_ollama_embedding.py``
(monkeypatched ``requests.post`` for request-shape assertions, no real
network).
"""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

# chromadb is an optional extra (``[semantic]``); skip where it isn't installed.
pytest.importorskip("chromadb")

from zotero_mcp import chroma_client  # noqa: E402
from zotero_mcp.embeddings.providers.voyage import VoyageEmbeddingFunction  # noqa: E402
from zotero_mcp.embeddings.registry import (  # noqa: E402
    PROVIDERS,
    batch_capable_providers,
    resolve_provider,
)

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_voyage_registered_by_name():
    assert "voyage" in PROVIDERS
    assert PROVIDERS["voyage"].name == "voyage"
    assert PROVIDERS["voyage"].default_model == "voyage-3.5"


def test_resolve_provider_voyage_direct_name():
    spec, extra = resolve_provider("voyage")
    assert spec.name == "voyage"
    assert extra == {}


def test_batch_capable_providers_unchanged_by_voyage():
    """Voyage registers with no BatchAdapter; it must not appear in the
    batch-capable list (which drives CLI --provider choices for batch-status/
    batch-import)."""
    assert "voyage" not in batch_capable_providers()


# ---------------------------------------------------------------------------
# create_chroma_client env merge (VOYAGE_API_KEY / _EMBEDDING_MODEL / _BASE_URL)
# ---------------------------------------------------------------------------


def _stub_chroma_client_class(monkeypatch):
    captured = {}

    class _StubChromaClient:
        def __init__(self, collection_name, embedding_model, embedding_config):
            captured["collection_name"] = collection_name
            captured["embedding_model"] = embedding_model
            captured["embedding_config"] = embedding_config

    monkeypatch.setattr(chroma_client, "ChromaClient", _StubChromaClient)
    return captured


def _clear_voyage_env(monkeypatch):
    for var in (
        "ZOTERO_EMBEDDING_MODEL",
        "VOYAGE_API_KEY",
        "VOYAGE_EMBEDDING_MODEL",
        "VOYAGE_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_voyage_env_fills_gaps_but_config_json_wins(monkeypatch, tmp_path):
    _clear_voyage_env(monkeypatch)
    captured = _stub_chroma_client_class(monkeypatch)
    monkeypatch.setenv("VOYAGE_API_KEY", "env-key")
    monkeypatch.setenv("VOYAGE_EMBEDDING_MODEL", "env-model")
    monkeypatch.setenv("VOYAGE_BASE_URL", "http://env-base")

    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"semantic_search": {"embedding_model": "voyage", '
        '"embedding_config": {"model_name": "config-model"}}}'
    )

    chroma_client.create_chroma_client(str(config_path))

    ec = captured["embedding_config"]
    assert ec["model_name"] == "config-model"  # config.json wins
    assert ec["api_key"] == "env-key"  # env fills gap
    assert ec["base_url"] == "http://env-base"  # env fills gap


def test_voyage_missing_api_key_leaves_embedding_config_unassigned(monkeypatch, tmp_path):
    _clear_voyage_env(monkeypatch)
    captured = _stub_chroma_client_class(monkeypatch)

    config_path = tmp_path / "config.json"
    config_path.write_text('{"semantic_search": {"embedding_model": "voyage"}}')

    chroma_client.create_chroma_client(str(config_path))

    # requires_api_key=True and no VOYAGE_API_KEY anywhere -> merged config
    # never written back, matching openai/gemini's behavior.
    assert captured["embedding_config"] == {}


# ---------------------------------------------------------------------------
# _embed_batch request shape
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _make_ef(**overrides):
    kwargs = {"model_name": "voyage-3.5", "api_key": "test-key"}
    kwargs.update(overrides)
    return VoyageEmbeddingFunction(**kwargs)


def test_embed_batch_document_input_type(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeResponse({"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]})

    import requests

    monkeypatch.setattr(requests, "post", fake_post)

    ef = _make_ef()
    result = ef._embed_batch(["doc one", "doc two"], is_query=False)

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["json"]["input_type"] == "document"
    assert captured["json"]["model"] == "voyage-3.5"
    assert captured["json"]["input"] == ["doc one", "doc two"]
    assert captured["url"].endswith("/embeddings")
    assert captured["headers"]["Authorization"] == "Bearer test-key"


def test_embed_batch_query_input_type(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return _FakeResponse({"data": [{"embedding": [0.5, 0.6]}]})

    import requests

    monkeypatch.setattr(requests, "post", fake_post)

    ef = _make_ef()
    result = ef._embed_batch(["a query"], is_query=True)

    assert result == [[0.5, 0.6]]
    assert captured["json"]["input_type"] == "query"


def test_missing_api_key_raises():
    import os

    old = os.environ.pop("VOYAGE_API_KEY", None)
    try:
        with pytest.raises(ValueError):
            VoyageEmbeddingFunction(model_name="voyage-3.5", api_key=None)
    finally:
        if old is not None:
            os.environ["VOYAGE_API_KEY"] = old


# ---------------------------------------------------------------------------
# get_config / build_from_config round trip
# ---------------------------------------------------------------------------


def test_config_round_trip():
    ef = _make_ef(base_url="http://custom-base", request_batch_size=32, rate_limit_rps=2.0,
                  max_parallel_requests=3, max_retries=2)
    cfg = ef.get_config()
    assert cfg["model_name"] == "voyage-3.5"
    assert cfg["base_url"] == "http://custom-base"
    assert cfg["request_batch_size"] == 32
    assert cfg["rate_limit_rps"] == 2.0
    assert cfg["max_parallel_requests"] == 3
    assert cfg["max_retries"] == 2

    rebuilt = VoyageEmbeddingFunction.build_from_config({**cfg, "api_key": "test-key"})
    assert rebuilt.model_name == "voyage-3.5"
    assert rebuilt.base_url == "http://custom-base"
    assert rebuilt.request_batch_size == 32


def test_build_from_config_handles_persisted_config_without_new_keys():
    """A config persisted before request_batch_size/rate_limit_rps/
    max_parallel_requests/max_retries existed must still build, falling back
    to class defaults for the new fields."""
    persisted = {"model_name": "voyage-3.5", "base_url": None, "api_key": "test-key"}

    ef = VoyageEmbeddingFunction.build_from_config(persisted)

    assert ef.model_name == "voyage-3.5"
    cfg = ef.get_config()
    assert cfg["request_batch_size"] == VoyageEmbeddingFunction.default_request_batch_size
    assert cfg["rate_limit_rps"] is None


# ---------------------------------------------------------------------------
# The extensibility proof: zero edits to orchestration/CLI
# ---------------------------------------------------------------------------


def test_voyage_added_with_zero_orchestration_or_cli_edits():
    """The whole point of Phase 5's registry generalization: a brand-new
    provider is reachable (registered, resolvable, env-mergeable) with no
    edits whatsoever to semantic_search.py or cli.py — everything routes
    through the registry generically."""
    import zotero_mcp.cli as cli_module
    import zotero_mcp.semantic_search as semantic_search_module

    assert "voyage" in PROVIDERS

    for module in (semantic_search_module, cli_module):
        with open(module.__file__, encoding="utf-8") as fh:
            source = fh.read()
        assert "voyage" not in source.lower()
