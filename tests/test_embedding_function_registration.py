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
