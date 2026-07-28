"""Regression test: custom embedding functions must be registered with ChromaDB.

ChromaDB >=1.x reconstructs a collection's embedding function *by name* from the
persisted config when it reopens a collection (e.g. the ``collection.configuration``
property used inside the upsert path). It looks the name up in
``chromadb.utils.embedding_functions.known_embedding_functions`` and calls that
class's ``build_from_config``.

Our custom embedding functions report names that either collide with ChromaDB's
own built-ins (``"openai"``, ``"huggingface"``) or are absent from the registry
(``"gemini"``). If they are not registered, ChromaDB resolves ``"openai"`` to its
*built-in* OpenAIEmbeddingFunction, whose ``build_from_config`` expects an
``api_key_env_var`` key and asserts on our ``{model_name, base_url}`` config:

    Could not build embedding function openai from config
    {'base_url': None, 'model_name': 'text-embedding-3-small'}:
    This code should not be reached

which surfaced as "19 errors" on every ``update-db`` against an existing index.

Fix: decorate the custom classes with ``@register_embedding_function`` so the
registry maps their names to *our* classes (and our compatible build_from_config).
"""

import pytest

# chromadb is an optional extra (``[semantic]``); skip where it isn't installed.
chromadb = pytest.importorskip("chromadb")  # noqa: F841

from chromadb.utils.embedding_functions import (  # noqa: E402
    known_embedding_functions,
)

from zotero_mcp import chroma_client  # noqa: E402


@pytest.mark.parametrize(
    "name, cls_attr",
    [
        ("openai", "OpenAIEmbeddingFunction"),
        ("gemini", "GeminiEmbeddingFunction"),
        ("huggingface", "HuggingFaceEmbeddingFunction"),
        ("ollama", "OllamaEmbeddingFunction"),
    ],
)
def test_custom_embedding_functions_are_registered(name, cls_attr):
    """Importing chroma_client must register our EFs under their names.

    Without this, ``known_embedding_functions["openai"]`` resolves to ChromaDB's
    incompatible built-in and breaks reload/upsert of an existing collection.
    """
    assert name in known_embedding_functions, (
        f"{name!r} not registered; ChromaDB cannot rebuild the embedding "
        "function from a persisted collection's config."
    )
    assert known_embedding_functions[name] is getattr(chroma_client, cls_attr)


def test_openai_build_from_config_handles_persisted_config(monkeypatch):
    """The exact operation that failed for existing indexes must now succeed.

    ChromaDB stores our OpenAI EF config as ``{"model_name": ..., "base_url": ...}``
    (see ``OpenAIEmbeddingFunction.get_config``). Rebuilding from that config via
    the registry previously hit ChromaDB's built-in and raised
    "This code should not be reached".
    """
    pytest.importorskip("openai")  # __init__ constructs an openai.OpenAI client
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-network")

    persisted = {"model_name": "text-embedding-3-small", "base_url": None}
    ef = known_embedding_functions["openai"].build_from_config(persisted)

    assert isinstance(ef, chroma_client.OpenAIEmbeddingFunction)
    # Configs persisted before request_batch_size/rate_limit_rps existed must
    # still rebuild, falling back to defaults for the new fields.
    cfg = ef.get_config()
    assert cfg["model_name"] == "text-embedding-3-small"
    assert cfg["base_url"] is None
    assert cfg["request_batch_size"] == chroma_client.OpenAIEmbeddingFunction.DEFAULT_REQUEST_BATCH_SIZE
    assert cfg["rate_limit_rps"] is None


# ---------------------------------------------------------------------------
# Issue #382: the persisted config must be buildable by BOTH classes
#
# The registry is a plain last-write-wins dict, so on some ChromaDB versions
# the *built-in* class answers the "ollama" lookup when a persisted collection
# config is rebuilt at query time. It reads url/model_name/timeout and asserts
# "This code should not be reached" on our {base_url, model_name} config, so
# indexing worked but every query failed. The stored config now carries both
# spellings, and our build_from_config accepts either.
# ---------------------------------------------------------------------------


def _builtin(name):
    """ChromaDB's own embedding function class for *name* (bypassing the registry)."""
    from chromadb.utils.embedding_functions import ollama_embedding_function

    return {"ollama": ollama_embedding_function.OllamaEmbeddingFunction}[name]


def test_ollama_config_carries_the_builtin_required_keys():
    """Our stored config must satisfy ChromaDB's built-in ollama EF."""
    ef = chroma_client.OllamaEmbeddingFunction(
        model_name="bge-m3", base_url="http://localhost:11434"
    )
    config = ef.get_config()

    # Keys the built-in's build_from_config asserts on.
    for key in ("url", "model_name", "timeout"):
        assert config.get(key) is not None, f"{key!r} missing from the persisted config"
    assert config["url"] == config["base_url"] == "http://localhost:11434"

    # The built-in's own JSON schema must accept the subset it reads.
    _builtin("ollama").validate_config(
        {k: config[k] for k in ("url", "model_name", "timeout")}
    )


def test_builtin_ollama_never_asserts_on_our_config():
    """If the built-in wins the registry lookup, it must still build.

    Constructing it may fail for unrelated reasons (the ``ollama`` python
    package is not a dependency of this project), but it must never fail with
    the "This code should not be reached" assertion that broke every query.
    """
    config = chroma_client.OllamaEmbeddingFunction(model_name="bge-m3").get_config()
    try:
        _builtin("ollama").build_from_config(config)
    except AssertionError as e:  # pragma: no cover - the bug this test pins
        pytest.fail(f"built-in ollama EF rejected our config: {e}")
    except Exception:
        pass  # e.g. the optional `ollama` package is not installed — not our bug


@pytest.mark.parametrize(
    "persisted",
    [
        # Written by us (pre- and post-fix shapes) ...
        {"model_name": "bge-m3", "base_url": "http://localhost:11434"},
        {
            "model_name": "bge-m3",
            "base_url": "http://localhost:11434",
            "url": "http://localhost:11434",
            "timeout": 60,
        },
        # ... and by ChromaDB's built-in.
        {"model_name": "bge-m3", "url": "http://localhost:11434", "timeout": 60},
    ],
)
def test_ollama_build_from_config_accepts_either_key_spelling(persisted):
    """``build_from_config`` must work with ``base_url`` or ``url`` (or both)."""
    ef = known_embedding_functions["ollama"].build_from_config(persisted)

    assert isinstance(ef, chroma_client.OllamaEmbeddingFunction)
    assert ef.model_name == "bge-m3"
    assert ef.base_url == "http://localhost:11434"
    assert ef.timeout == persisted.get(
        "timeout", chroma_client.OllamaEmbeddingFunction.DEFAULT_TIMEOUT
    )
    # Round-trips: ChromaDB calls build_from_config(get_config()) in is_legacy().
    assert ef.get_config() == ef.build_from_config(ef.get_config()).get_config()


def test_openai_and_huggingface_configs_carry_builtin_required_keys():
    """Same defence for the other two names that collide with built-ins.

    Both built-ins assert without ``api_key_env_var``; "gemini" has no built-in
    of that name, so its config needs no extra keys.

    The instances are built with ``__new__`` so no model is downloaded and no
    API client is constructed — only ``get_config`` is under test.
    """
    ef = chroma_client.HuggingFaceEmbeddingFunction.__new__(
        chroma_client.HuggingFaceEmbeddingFunction
    )
    ef.model_name = "Qwen/Qwen3-Embedding-0.6B"
    assert ef.get_config().get("api_key_env_var")

    oai = chroma_client.OpenAIEmbeddingFunction.__new__(
        chroma_client.OpenAIEmbeddingFunction
    )
    oai.model_name = "text-embedding-3-small"
    oai.base_url = None
    oai.request_batch_size = 64
    oai.rate_limit_rps = None
    assert oai.get_config().get("api_key_env_var")


def test_ensure_embedding_functions_registered_reclaims_the_name():
    """A late-registering built-in must not keep our name (issue #382)."""
    original = known_embedding_functions["ollama"]
    try:
        known_embedding_functions["ollama"] = _builtin("ollama")
        chroma_client.ensure_embedding_functions_registered()
        assert known_embedding_functions["ollama"] is chroma_client.OllamaEmbeddingFunction
    finally:
        known_embedding_functions["ollama"] = original
