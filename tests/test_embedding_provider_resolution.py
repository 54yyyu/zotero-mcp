"""Characterization tests for embedding-provider resolution and env merging.

These pin the two behaviours that used to be spelled out as parallel if/elif
chains keyed on ``embedding_model``:

- ``ChromaClient._create_embedding_function`` — which class gets built, with
  which arguments, for a given model string.
- ``create_chroma_client`` — which environment variables are merged into
  ``embedding_config``, and when the merged result is kept.

The env-merge half in particular had no coverage at all, which is what made it
risky to consolidate. Everything here is written against the public surface
(``chroma_client.ChromaClient`` / ``chroma_client.create_chroma_client``) and
never imports the registry, so this file passes unchanged both before and after
the providers moved into ``zotero_mcp.embeddings`` — it describes behaviour, not
structure.

Construction is intercepted by patching the embedding function class's own
``__init__`` rather than the name the caller looks up, because those namespaces
differ between the two implementations while the class object does not. That
also keeps the tests offline: no sentence-transformers download, no ONNX model
fetch, no API client.
"""

import json

import pytest

pytest.importorskip("chromadb")

from zotero_mcp import chroma_client  # noqa: E402

# Every environment variable the code under test consults. Cleared before each
# test so a developer's real shell (or a leaked GOOGLE_API_KEY) cannot change
# the outcome.
_ENV_VARS = (
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
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Which embedding function does a model string build?
# ---------------------------------------------------------------------------


def _client(embedding_model, embedding_config=None):
    """A ChromaClient without __init__'s on-disk PersistentClient side effects.

    ``_create_embedding_function`` reads only these two attributes.
    """
    client = chroma_client.ChromaClient.__new__(chroma_client.ChromaClient)
    client.embedding_model = embedding_model
    client.embedding_config = embedding_config or {}
    return client


@pytest.fixture
def hf_args(monkeypatch):
    """Capture the arguments HuggingFaceEmbeddingFunction is constructed with.

    Patching ``__init__`` on the class itself (not on a module-level name)
    intercepts construction no matter which module resolved the class.
    """
    seen = {}

    def fake_init(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        seen["model_name"] = model_name

    monkeypatch.setattr(
        chroma_client.HuggingFaceEmbeddingFunction, "__init__", fake_init
    )
    return seen


def test_qwen_alias_expands_to_the_concrete_model(hf_args):
    _client("qwen")._create_embedding_function()
    assert hf_args["model_name"] == "Qwen/Qwen3-Embedding-0.6B"


def test_embeddinggemma_alias_expands_to_the_concrete_model(hf_args):
    _client("embeddinggemma")._create_embedding_function()
    assert hf_args["model_name"] == "google/embeddinggemma-300m"


def test_an_explicit_model_name_overrides_an_alias(hf_args):
    """For the aliases, embedding_config.model_name wins over the expansion."""
    _client("qwen", {"model_name": "BAAI/bge-m3"})._create_embedding_function()
    assert hf_args["model_name"] == "BAAI/bge-m3"


def test_a_bare_model_string_beats_an_explicit_model_name(hf_args):
    """...but for a bare model string the string itself wins.

    The opposite direction from the alias case above. Asymmetric, and only
    reachable by configuring both at once, but it is the established
    behaviour, so it is pinned here rather than quietly normalized.
    """
    _client(
        "sentence-transformers/all-mpnet-base-v2", {"model_name": "BAAI/bge-m3"}
    )._create_embedding_function()
    assert hf_args["model_name"] == "sentence-transformers/all-mpnet-base-v2"


def test_the_literal_string_huggingface_is_treated_as_a_model_name(hf_args):
    """A legacy quirk: "huggingface" does not select the provider generically.

    It falls through to the bare-model-string rule, so it is passed to
    sentence-transformers as a model name (where it fails to load). Preserved
    deliberately — changing it would alter behaviour, not just structure.
    """
    _client("huggingface")._create_embedding_function()
    assert hf_args["model_name"] == "huggingface"


def test_default_uses_chromadbs_builtin_capped_at_256_tokens(monkeypatch):
    from chromadb.utils import embedding_functions

    class _FakeDefaultEF:
        pass

    monkeypatch.setattr(
        embedding_functions, "DefaultEmbeddingFunction", _FakeDefaultEF
    )

    ef = _client("default")._create_embedding_function()

    assert isinstance(ef, _FakeDefaultEF)
    # all-MiniLM-L6-v2's max_seq_length; the pipeline truncates against it.
    assert ef.max_input_tokens == 256


def test_openai_receives_every_configured_knob():
    pytest.importorskip("openai")

    ef = _client(
        "openai",
        {
            "model_name": "text-embedding-3-large",
            "api_key": "test-key-no-network",
            "base_url": "https://example.invalid/v1",
            "request_batch_size": 8,
            "rate_limit_rps": 2.5,
        },
    )._create_embedding_function()

    assert isinstance(ef, chroma_client.OpenAIEmbeddingFunction)
    assert ef.model_name == "text-embedding-3-large"
    assert ef.api_key == "test-key-no-network"
    assert ef.base_url == "https://example.invalid/v1"
    assert ef.request_batch_size == 8
    assert ef.rate_limit_rps == 2.5


def test_openai_falls_back_to_its_default_model(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-network")

    ef = _client("openai")._create_embedding_function()

    assert ef.model_name == "text-embedding-3-small"
    assert ef.request_batch_size == (
        chroma_client.OpenAIEmbeddingFunction.DEFAULT_REQUEST_BATCH_SIZE
    )
    assert ef.rate_limit_rps is None


# ---------------------------------------------------------------------------
# Which environment variables reach embedding_config?
# ---------------------------------------------------------------------------


@pytest.fixture
def built_config(monkeypatch, tmp_path):
    """Run create_chroma_client and return the kwargs it passed to ChromaClient."""
    captured = {}

    class _RecordingChromaClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(chroma_client, "ChromaClient", _RecordingChromaClient)

    def run(semantic_search):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"semantic_search": semantic_search}))
        chroma_client.create_chroma_client(str(path))
        return captured

    return run


def test_openai_env_fills_every_gap(monkeypatch, built_config):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.invalid/v1")

    config = built_config({"embedding_model": "openai"})

    assert config["embedding_config"] == {
        "api_key": "env-key",
        "model_name": "text-embedding-3-large",
        "base_url": "https://env.invalid/v1",
    }


def test_config_json_wins_over_the_environment(monkeypatch, built_config):
    """A stale env var must never displace an explicitly configured value."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.invalid/v1")

    config = built_config(
        {
            "embedding_model": "openai",
            "embedding_config": {
                "api_key": "file-key",
                "model_name": "text-embedding-3-small",
                "base_url": "https://file.invalid/v1",
            },
        }
    )

    assert config["embedding_config"] == {
        "api_key": "file-key",
        "model_name": "text-embedding-3-small",
        "base_url": "https://file.invalid/v1",
    }


def test_openai_model_name_defaults_when_no_env_var_is_set(built_config):
    """The api key is still required for the merged config to be kept..."""
    config = built_config(
        {"embedding_model": "openai", "embedding_config": {"api_key": "file-key"}}
    )

    assert config["embedding_config"]["model_name"] == "text-embedding-3-small"
    assert "base_url" not in config["embedding_config"]


def test_openai_without_any_api_key_leaves_the_config_untouched(built_config):
    """...and without one, the merge is discarded rather than half-applied.

    No api key means the embedding function will refuse to construct anyway;
    the point is that the defaulted model_name is not silently written back.
    """
    config = built_config(
        {"embedding_model": "openai", "embedding_config": {"chunk_size": 512}}
    )

    assert config["embedding_config"] == {"chunk_size": 512}


def test_gemini_prefers_gemini_api_key_over_google_api_key(monkeypatch, built_config):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    config = built_config({"embedding_model": "gemini"})

    assert config["embedding_config"]["api_key"] == "gemini-key"
    assert config["embedding_config"]["model_name"] == "gemini-embedding-001"


def test_gemini_falls_back_to_google_api_key(monkeypatch, built_config):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    config = built_config({"embedding_model": "gemini"})

    assert config["embedding_config"]["api_key"] == "google-key"


def test_ollama_merges_without_needing_an_api_key(monkeypatch, built_config):
    """Ollama keeps the merged config unconditionally — it has no api key."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu:11434")

    config = built_config({"embedding_model": "ollama"})

    assert config["embedding_config"] == {
        "model_name": "qwen3-embedding",
        "base_url": "http://gpu:11434",
    }


def test_ollama_model_name_defaults_with_no_env_at_all(built_config):
    config = built_config({"embedding_model": "ollama"})

    assert config["embedding_config"] == {"model_name": "qwen3-embedding"}


@pytest.mark.parametrize(
    "embedding_model", ["default", "qwen", "embeddinggemma", "BAAI/bge-m3"]
)
def test_providers_without_env_wiring_ignore_the_environment(
    monkeypatch, built_config, embedding_model
):
    """The local providers had no branch in the old chain, and still read nothing.

    A leaked OPENAI_API_KEY/GOOGLE_API_KEY from another tool must not seep into
    a locally-embedded configuration.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu:11434")

    config = built_config(
        {"embedding_model": embedding_model, "embedding_config": {"chunk_size": 512}}
    )

    assert config["embedding_config"] == {"chunk_size": 512}


def test_env_model_only_fills_in_when_config_is_left_at_default(
    monkeypatch, built_config
):
    """ZOTERO_EMBEDDING_MODEL must not downgrade an explicitly configured provider."""
    monkeypatch.setenv("ZOTERO_EMBEDDING_MODEL", "default")

    config = built_config(
        {"embedding_model": "ollama", "embedding_config": {"model_name": "bge-m3"}}
    )

    assert config["embedding_model"] == "ollama"
