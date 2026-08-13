"""Unit tests for the embedding provider registry itself.

Behavioural equivalence with the if/elif chains this registry replaced is
covered in ``test_embedding_provider_resolution.py``, which is deliberately
written against ``chroma_client`` only. This file tests the registry's own API:
the shape ``resolve_provider`` returns, and the invariants a new provider has
to respect.
"""

import pytest

pytest.importorskip("chromadb")

from zotero_mcp.embeddings.registry import (  # noqa: E402
    PROVIDERS,
    EnvSpec,
    ProviderSpec,
    merge_env_config,
    register_provider,
    resolve_provider,
)


def _effective_config(embedding_model, embedding_config):
    """Apply the merge rule that ``create_embedding_function`` uses."""
    _, defaults, overrides = resolve_provider(embedding_model)
    return {**defaults, **embedding_config, **overrides}


@pytest.mark.parametrize("name", ["openai", "gemini", "ollama", "huggingface", "default"])
def test_every_expected_provider_is_registered(name):
    """These names are persisted in collection configs and cannot be renamed."""
    assert name in PROVIDERS
    assert PROVIDERS[name].name == name


@pytest.mark.parametrize("name", ["openai", "gemini", "ollama"])
def test_a_provider_name_resolves_to_itself_with_no_extras(name):
    spec, defaults, overrides = resolve_provider(name)

    assert spec is PROVIDERS[name]
    assert defaults == {}
    assert overrides == {}


def test_aliases_resolve_to_huggingface_as_a_default():
    """An alias supplies model_name as a *default*, so config still wins."""
    spec, defaults, overrides = resolve_provider("qwen")

    assert spec is PROVIDERS["huggingface"]
    assert defaults == {"model_name": "Qwen/Qwen3-Embedding-0.6B"}
    assert overrides == {}
    assert _effective_config("qwen", {"model_name": "x"})["model_name"] == "x"


def test_a_bare_model_string_resolves_as_an_override():
    """A bare model string supplies model_name as an *override*, beating config.

    The asymmetry against the alias case above is why resolve_provider returns
    two dicts instead of one.
    """
    spec, defaults, overrides = resolve_provider("BAAI/bge-m3")

    assert spec is PROVIDERS["huggingface"]
    assert defaults == {}
    assert overrides == {"model_name": "BAAI/bge-m3"}
    assert _effective_config("BAAI/bge-m3", {"model_name": "x"})["model_name"] == "BAAI/bge-m3"


def test_default_resolves_to_the_builtin_spec():
    spec, defaults, overrides = resolve_provider("default")

    assert spec is PROVIDERS["default"]
    assert (defaults, overrides) == ({}, {})


def test_local_providers_declare_no_environment_wiring():
    for name in ("huggingface", "default"):
        assert not PROVIDERS[name].env.reads_environment()


def test_merge_env_config_is_identity_for_providers_without_env_wiring():
    original = {"chunk_size": 512}

    assert merge_env_config("default", original) is original
    assert merge_env_config("qwen", original) is original


def test_a_newly_registered_provider_is_reachable_by_its_bare_name():
    """Adding a provider must not require editing resolve_provider.

    This is the extensibility the registry exists to buy: PR 3 and PR 4 both
    add providers, and neither should have to touch the resolution logic.
    """
    sentinel = object()
    spec = ProviderSpec(
        name="test-provider",
        default_model="test-model",
        ef_factory=lambda config: sentinel,
        env=EnvSpec(api_key_vars=("TEST_PROVIDER_API_KEY",), requires_api_key=True),
    )
    try:
        register_provider(spec)

        resolved, defaults, overrides = resolve_provider("test-provider")

        assert resolved is spec
        assert (defaults, overrides) == ({}, {})
    finally:
        del PROVIDERS["test-provider"]

    # ...and once unregistered it falls back to being a HuggingFace model name.
    assert resolve_provider("test-provider")[0] is PROVIDERS["huggingface"]


def test_merge_env_config_reads_api_key_vars_in_order(monkeypatch):
    spec = ProviderSpec(
        name="test-provider",
        default_model="test-model",
        ef_factory=lambda config: None,
        env=EnvSpec(
            api_key_vars=("TEST_FIRST_KEY", "TEST_SECOND_KEY"),
            model_var="TEST_MODEL",
            requires_api_key=True,
        ),
    )
    try:
        register_provider(spec)
        monkeypatch.delenv("TEST_FIRST_KEY", raising=False)
        monkeypatch.setenv("TEST_SECOND_KEY", "second")
        monkeypatch.delenv("TEST_MODEL", raising=False)

        merged = merge_env_config("test-provider", {})

        assert merged == {"api_key": "second", "model_name": "test-model"}

        monkeypatch.setenv("TEST_FIRST_KEY", "first")
        assert merge_env_config("test-provider", {})["api_key"] == "first"
    finally:
        del PROVIDERS["test-provider"]
