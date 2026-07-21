"""Tests for the provider registry (embeddings/registry.py), Phase 2 of the
embedding-provider refactor.

Two things are pinned here:
  1. ``resolve_provider`` reproduces the old ``ChromaClient._create_embedding_function``
     if/elif chain exactly (which spec + which extra config a model string implies).
  2. ``create_chroma_client``'s generic env-merge block (driven by the resolved
     spec's ``EnvSpec``) reproduces the old per-provider merge blocks exactly:
     config.json wins, env fills gaps, missing api_key leaves embedding_config
     unassigned for openai/gemini, ollama merges unconditionally, and
     ZOTERO_EMBEDDING_MODEL only applies when the model is "default"/absent.

HF models are never actually instantiated here (SentenceTransformer would try
to download real weights) — HF-routed cases are checked by asserting the
resolved spec/extra config, and by monkeypatching the factory to confirm it
would be invoked with the right merged config, never by calling
``ef_factory`` for real.
"""

import pytest

# chromadb is an optional extra (``[semantic]``); skip where it isn't installed.
pytest.importorskip("chromadb")

from zotero_mcp import chroma_client  # noqa: E402
from zotero_mcp.embeddings.registry import PROVIDERS, resolve_provider  # noqa: E402

# ---------------------------------------------------------------------------
# resolve_provider
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "embedding_model, expected_name, expected_extra",
    [
        ("openai", "openai", {}),
        ("gemini", "gemini", {}),
        ("ollama", "ollama", {}),
        ("qwen", "huggingface", {"model_name": "Qwen/Qwen3-Embedding-0.6B"}),
        (
            "embeddinggemma",
            "huggingface",
            {"model_name": "google/embeddinggemma-300m"},
        ),
        (
            "sentence-transformers/all-mpnet-base-v2",
            "huggingface",
            {"model_name": "sentence-transformers/all-mpnet-base-v2"},
        ),
        ("default", "default", {}),
    ],
)
def test_resolve_provider_matches_old_if_elif_chain(
    embedding_model, expected_name, expected_extra
):
    spec, extra = resolve_provider(embedding_model)
    assert spec.name == expected_name
    assert extra == expected_extra


def test_explicit_model_name_wins_over_alias_default():
    """embedding_config.get("model_name", <alias default>) semantics: the
    caller merges as {**extra, **embedding_config}, so an explicit
    model_name in embedding_config must win over the alias's default."""
    spec, extra = resolve_provider("qwen")
    merged = {**extra, **{"model_name": "custom/explicit-model"}}
    assert merged["model_name"] == "custom/explicit-model"
    assert spec.name == "huggingface"


# ---------------------------------------------------------------------------
# _create_embedding_function selection (factory identity via monkeypatch, no
# real HF/network construction)
# ---------------------------------------------------------------------------


class _DummyEF:
    def __init__(self, config):
        self.config = config


def _patch_huggingface_factory(monkeypatch, fake_factory):
    """ProviderSpec is frozen, so swapping its factory means registering a
    replacement spec (via dataclasses.replace) under the same dict key
    rather than assigning to the existing instance's attribute."""
    import dataclasses

    from zotero_mcp.embeddings import registry

    original = registry.PROVIDERS["huggingface"]
    patched = dataclasses.replace(original, ef_factory=fake_factory)
    monkeypatch.setitem(registry.PROVIDERS, "huggingface", patched)


def test_create_embedding_function_uses_resolved_factory(monkeypatch):
    """ChromaClient._create_embedding_function must call
    spec.ef_factory({**extra, **self.embedding_config}) — verified by
    monkeypatching the registered huggingface spec's factory rather than
    constructing a real SentenceTransformer model."""
    captured = {}

    def fake_factory(config):
        captured.update(config)
        return _DummyEF(config)

    _patch_huggingface_factory(monkeypatch, fake_factory)

    client = chroma_client.ChromaClient.__new__(chroma_client.ChromaClient)
    client.embedding_model = "qwen"
    client.embedding_config = {"request_batch_size": 8}

    ef = client._create_embedding_function()

    assert isinstance(ef, _DummyEF)
    assert captured["model_name"] == "Qwen/Qwen3-Embedding-0.6B"
    assert captured["request_batch_size"] == 8


def test_create_embedding_function_explicit_config_wins(monkeypatch):
    captured = {}

    def fake_factory(config):
        captured.update(config)
        return _DummyEF(config)

    _patch_huggingface_factory(monkeypatch, fake_factory)

    client = chroma_client.ChromaClient.__new__(chroma_client.ChromaClient)
    client.embedding_model = "embeddinggemma"
    client.embedding_config = {"model_name": "explicit/override"}

    client._create_embedding_function()

    assert captured["model_name"] == "explicit/override"


def test_create_embedding_function_openai_constructs_real_ef(monkeypatch):
    """openai/gemini construct real EF objects (constructors don't hit the
    network) with a dummy api_key, unlike the HF path above."""
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-fallback")

    client = chroma_client.ChromaClient.__new__(chroma_client.ChromaClient)
    client.embedding_model = "openai"
    client.embedding_config = {"api_key": "sk-test-dummy", "model_name": "text-embedding-3-small"}

    ef = client._create_embedding_function()

    assert isinstance(ef, chroma_client.OpenAIEmbeddingFunction)
    assert ef.model_name == "text-embedding-3-small"
    assert ef.api_key == "sk-test-dummy"


def test_create_embedding_function_gemini_constructs_real_ef(monkeypatch):
    pytest.importorskip("google.genai")

    client = chroma_client.ChromaClient.__new__(chroma_client.ChromaClient)
    client.embedding_model = "gemini"
    client.embedding_config = {"api_key": "dummy-gemini-key"}

    ef = client._create_embedding_function()

    assert isinstance(ef, chroma_client.GeminiEmbeddingFunction)
    assert ef.model_name == "gemini-embedding-001"


def test_create_embedding_function_default_sets_max_input_tokens():
    client = chroma_client.ChromaClient.__new__(chroma_client.ChromaClient)
    client.embedding_model = "default"
    client.embedding_config = {}

    ef = client._create_embedding_function()

    assert ef.max_input_tokens == 256


# ---------------------------------------------------------------------------
# create_chroma_client env-merge semantics
# ---------------------------------------------------------------------------


def _stub_chroma_client_class(monkeypatch):
    """Replace ChromaClient with a stub that records constructor kwargs
    instead of touching a real ChromaDB directory."""
    captured = {}

    class _StubChromaClient:
        def __init__(self, collection_name, embedding_model, embedding_config):
            captured["collection_name"] = collection_name
            captured["embedding_model"] = embedding_model
            captured["embedding_config"] = embedding_config

    monkeypatch.setattr(chroma_client, "ChromaClient", _StubChromaClient)
    return captured


def _clear_provider_env(monkeypatch):
    for var in (
        "ZOTERO_EMBEDDING_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_EMBEDDING_MODEL",
        "OPENAI_BASE_URL",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_EMBEDDING_MODEL",
        "GEMINI_BASE_URL",
        "OLLAMA_EMBEDDING_MODEL",
        "OLLAMA_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_openai_env_fills_gaps_but_config_json_wins(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    captured = _stub_chroma_client_class(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "env-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env-base")

    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"semantic_search": {"embedding_model": "openai", '
        '"embedding_config": {"model_name": "config-model"}}}'
    )

    chroma_client.create_chroma_client(str(config_path))

    ec = captured["embedding_config"]
    assert ec["model_name"] == "config-model"  # config.json wins
    assert ec["api_key"] == "env-key"  # env fills gap
    assert ec["base_url"] == "http://env-base"  # env fills gap


def test_openai_missing_api_key_leaves_embedding_config_unassigned(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    captured = _stub_chroma_client_class(monkeypatch)

    config_path = tmp_path / "config.json"
    config_path.write_text('{"semantic_search": {"embedding_model": "openai"}}')

    chroma_client.create_chroma_client(str(config_path))

    # No OPENAI_API_KEY anywhere and none in config.json -> merged config is
    # never written back; embedding_config stays the file's original (empty).
    assert captured["embedding_config"] == {}


def test_gemini_api_key_prefers_gemini_over_google(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    captured = _stub_chroma_client_class(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    config_path = tmp_path / "config.json"
    config_path.write_text('{"semantic_search": {"embedding_model": "gemini"}}')

    chroma_client.create_chroma_client(str(config_path))

    assert captured["embedding_config"]["api_key"] == "gemini-key"


def test_gemini_falls_back_to_google_api_key(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    captured = _stub_chroma_client_class(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    config_path = tmp_path / "config.json"
    config_path.write_text('{"semantic_search": {"embedding_model": "gemini"}}')

    chroma_client.create_chroma_client(str(config_path))

    assert captured["embedding_config"]["api_key"] == "google-key"


def test_gemini_missing_api_key_leaves_embedding_config_unassigned(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    captured = _stub_chroma_client_class(monkeypatch)

    config_path = tmp_path / "config.json"
    config_path.write_text('{"semantic_search": {"embedding_model": "gemini"}}')

    chroma_client.create_chroma_client(str(config_path))

    assert captured["embedding_config"] == {}


def test_ollama_merges_unconditionally_without_api_key(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    captured = _stub_chroma_client_class(monkeypatch)

    config_path = tmp_path / "config.json"
    config_path.write_text('{"semantic_search": {"embedding_model": "ollama"}}')

    chroma_client.create_chroma_client(str(config_path))

    ec = captured["embedding_config"]
    assert ec["model_name"] == "qwen3-embedding"  # default, no api_key required
    assert "api_key" not in ec


def test_ollama_env_base_url_fills_gap(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    captured = _stub_chroma_client_class(monkeypatch)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama-env:11434")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "env-ollama-model")

    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"semantic_search": {"embedding_model": "ollama", '
        '"embedding_config": {"base_url": "http://config-base:11434"}}}'
    )

    chroma_client.create_chroma_client(str(config_path))

    ec = captured["embedding_config"]
    assert ec["base_url"] == "http://config-base:11434"  # config.json wins
    assert ec["model_name"] == "env-ollama-model"  # env fills gap


def test_huggingface_model_skips_env_merge_entirely(monkeypatch, tmp_path):
    """HF model names have no EnvSpec entries — the merge block must be a
    complete no-op, leaving embedding_config exactly as loaded from file."""
    _clear_provider_env(monkeypatch)
    captured = _stub_chroma_client_class(monkeypatch)
    # Even if provider env vars happen to be set, they must not leak in.
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-appear")

    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"semantic_search": {"embedding_model": "some-hf/model"}}'
    )

    chroma_client.create_chroma_client(str(config_path))

    assert captured["embedding_config"] == {}


def test_default_model_skips_env_merge_entirely(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    captured = _stub_chroma_client_class(monkeypatch)

    config_path = tmp_path / "config.json"
    config_path.write_text('{"semantic_search": {"embedding_model": "default"}}')

    chroma_client.create_chroma_client(str(config_path))

    assert captured["embedding_config"] == {}


def test_zotero_embedding_model_env_applies_only_when_default_or_absent(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    captured = _stub_chroma_client_class(monkeypatch)
    monkeypatch.setenv("ZOTERO_EMBEDDING_MODEL", "ollama")

    # config.json explicitly names a concrete provider -> env var must NOT
    # override it (guards against a stale ZOTERO_EMBEDDING_MODEL downgrading
    # an explicitly configured provider).
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"semantic_search": {"embedding_model": "gemini", '
        '"embedding_config": {"api_key": "explicit-key"}}}'
    )

    chroma_client.create_chroma_client(str(config_path))

    assert captured["embedding_model"] == "gemini"


def test_zotero_embedding_model_env_fills_default_placeholder(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    captured = _stub_chroma_client_class(monkeypatch)
    monkeypatch.setenv("ZOTERO_EMBEDDING_MODEL", "ollama")

    config_path = tmp_path / "config.json"
    config_path.write_text('{"semantic_search": {"embedding_model": "default"}}')

    chroma_client.create_chroma_client(str(config_path))

    assert captured["embedding_model"] == "ollama"
    assert captured["embedding_config"]["model_name"] == "qwen3-embedding"


def test_zotero_embedding_model_env_fills_absent_model(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    captured = _stub_chroma_client_class(monkeypatch)
    monkeypatch.setenv("ZOTERO_EMBEDDING_MODEL", "ollama")

    config_path = tmp_path / "config.json"
    config_path.write_text('{"semantic_search": {}}')

    chroma_client.create_chroma_client(str(config_path))

    assert captured["embedding_model"] == "ollama"


# ---------------------------------------------------------------------------
# Registered spec sanity
# ---------------------------------------------------------------------------


def test_all_expected_providers_registered():
    assert set(PROVIDERS) == {"openai", "gemini", "ollama", "huggingface", "default"}


def test_chars_per_token_values():
    # Mirror OPENAI_CHARS_PER_TOKEN / GEMINI_CHARS_PER_TOKEN in the batch
    # modules (not imported here to avoid a registry -> batch module dependency).
    assert PROVIDERS["openai"].chars_per_token == 3.0
    assert PROVIDERS["gemini"].chars_per_token == 3.5
    assert PROVIDERS["ollama"].chars_per_token == 4.0


def test_batch_capable_providers_lists_openai_and_gemini():
    """Phase 4 wires ``attach_batch_adapter`` at ``openai_batch``/``gemini_batch``
    import time (see those modules' module-scope calls), so importing them
    here mutates the process-wide ``PROVIDERS`` registry — matching what
    happens in any real process that touches batch functionality."""
    import zotero_mcp.gemini_batch  # noqa: F401
    import zotero_mcp.openai_batch  # noqa: F401
    from zotero_mcp.embeddings.registry import batch_capable_providers

    assert PROVIDERS["openai"].batch is not None
    assert PROVIDERS["gemini"].batch is not None
    assert batch_capable_providers() == ["openai", "gemini"]
