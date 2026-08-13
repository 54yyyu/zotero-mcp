"""Persist/reload round-trip through a real ``ChromaClient`` on a scratch
directory (never the production ``~/.config/zotero-mcp/chroma_db``).

Exercises the exact production persistence path: an embedding function is
built, documents are upserted (embedding them for real), the client is torn
down, and a fresh ``ChromaClient`` is opened on the same directory to prove
ChromaDB correctly rebuilds the embedding function from its persisted config
and that reloaded vectors match the originally stored ones.

Requires a real backend: the Ollama variant needs ``ollama serve`` with
``nomic-embed-text`` pulled (see ``ollama_available`` in conftest.py); the
config-matched variant needs a ``~/.config/zotero-mcp/config.json`` whose
``semantic_search.embedding_model`` names a live-testable provider (this
machine: openai) via the ``configured_provider`` gate. Collected but skipped
by default; run with
``ZOTERO_MCP_LIVE_TESTS=1 uv run pytest tests/live/test_chroma_roundtrip_live.py -v``.
"""

from __future__ import annotations

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")

from zotero_mcp.chroma_client import ChromaClient  # noqa: E402

OLLAMA_MODEL = "nomic-embed-text"

DOC_NEURAL = "Neural networks are computational models inspired by biological brains, used for pattern recognition."
DOC_MEDIEVAL = "Medieval history covers the period from the fall of Rome to the beginning of the Renaissance."


@pytest.mark.timeout(90)
def test_persist_and_reload_ollama(tmp_path, ollama_available, cosine_similarity):
    persist_dir = str(tmp_path / "chroma_db")

    client = ChromaClient(
        collection_name="zotero_library",
        persist_directory=persist_dir,
        embedding_model="ollama",
        embedding_config={"model_name": OLLAMA_MODEL, "base_url": ollama_available},
    )
    client.upsert_documents(
        documents=[DOC_NEURAL, DOC_MEDIEVAL],
        metadatas=[{"title": "Neural Nets"}, {"title": "Medieval History"}],
        ids=["ITEM1", "ITEM2"],
    )

    stored = client.collection.get(ids=["ITEM1"], include=["embeddings"])
    stored_embedding = list(stored["embeddings"][0])

    # Tear down and reopen a fresh client on the same directory -- this is
    # the scenario ChromaDB's "rebuild embedding function from persisted
    # config" path exists for.
    del client

    reloaded = ChromaClient(
        collection_name="zotero_library",
        persist_directory=persist_dir,
        embedding_model="ollama",
        embedding_config={"model_name": OLLAMA_MODEL, "base_url": ollama_available},
    )

    assert reloaded.embedding_function.name() == "ollama"
    assert reloaded.embedding_function.model_name == OLLAMA_MODEL

    results = reloaded.search(query_texts=["neural networks"], n_results=2)
    assert results["ids"][0][0] == "ITEM1"

    reloaded_get = reloaded.collection.get(ids=["ITEM1"], include=["embeddings"])
    reloaded_embedding = list(reloaded_get["embeddings"][0])

    similarity = cosine_similarity(stored_embedding, reloaded_embedding)
    assert similarity >= 0.999, f"expected stored vs reloaded ITEM1 embedding cosine >= 0.999, got {similarity}"


@pytest.mark.timeout(90)
def test_persist_and_reload_configured_provider(tmp_path, configured_provider, cosine_similarity):
    """Same persist/reload/query flow, but using whatever provider is
    actually configured in production (~/.config/zotero-mcp/config.json) --
    the config-match gate skips cleanly if this machine isn't configured for
    a live-testable provider. Two tiny documents: cost is negligible and
    intentional (this is precisely the production persistence path)."""
    production_config = configured_provider("openai")
    persist_dir = str(tmp_path / "chroma_db")

    client = ChromaClient(
        collection_name="zotero_library",
        persist_directory=persist_dir,
        embedding_model="openai",
        embedding_config=production_config,
    )
    client.upsert_documents(
        documents=[DOC_NEURAL, DOC_MEDIEVAL],
        metadatas=[{"title": "Neural Nets"}, {"title": "Medieval History"}],
        ids=["ITEM1", "ITEM2"],
    )

    stored = client.collection.get(ids=["ITEM1"], include=["embeddings"])
    stored_embedding = list(stored["embeddings"][0])

    del client

    reloaded = ChromaClient(
        collection_name="zotero_library",
        persist_directory=persist_dir,
        embedding_model="openai",
        embedding_config=production_config,
    )

    assert reloaded.embedding_function.name() == "openai"
    assert reloaded.embedding_function.model_name == production_config.get("model_name")

    results = reloaded.search(query_texts=["neural networks"], n_results=2)
    assert results["ids"][0][0] == "ITEM1"

    reloaded_get = reloaded.collection.get(ids=["ITEM1"], include=["embeddings"])
    reloaded_embedding = list(reloaded_get["embeddings"][0])

    similarity = cosine_similarity(stored_embedding, reloaded_embedding)
    assert similarity >= 0.999, f"expected stored vs reloaded ITEM1 embedding cosine >= 0.999, got {similarity}"
