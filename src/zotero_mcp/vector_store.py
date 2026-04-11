"""
Vector store abstraction layer.

Provides a ``VectorStoreClient`` protocol that both ``ChromaClient`` and
``PineconeClient`` satisfy, plus a factory function that creates the right
backend based on configuration.
"""

import json
import logging
import os
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorStoreClient(Protocol):
    """Protocol defining the vector store client interface.

    Both ``ChromaClient`` and ``PineconeClient`` satisfy this protocol so
    they can be used interchangeably by ``ZoteroSemanticSearch``.
    """

    embedding_model: str

    @property
    def embedding_max_tokens(self) -> int: ...

    def truncate_text(self, text: str, max_tokens: int | None = None) -> str: ...

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None: ...

    def upsert_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None: ...

    def search(
        self,
        query_texts: list[str],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def delete_documents(self, ids: list[str]) -> None: ...

    def get_collection_info(self) -> dict[str, Any]: ...

    def reset_collection(self) -> None: ...

    def document_exists(self, doc_id: str) -> bool: ...

    def get_document_metadata(self, doc_id: str) -> dict[str, Any] | None: ...

    def get_existing_ids(self, ids: list[str]) -> set[str]: ...


def get_vector_backend(config_path: str | None = None) -> str:
    """Determine which vector backend to use.

    Resolution order:
    1. ``ZOTERO_VECTOR_BACKEND`` environment variable
    2. ``semantic_search.vector_backend`` in the config JSON file
    3. Default: ``"chroma"``
    """
    backend = os.getenv("ZOTERO_VECTOR_BACKEND", "").lower()
    if backend:
        return backend

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            backend = cfg.get("semantic_search", {}).get("vector_backend", "").lower()
            if backend:
                return backend
        except Exception:
            pass

    return "chroma"


def create_vector_client(config_path: str | None = None) -> VectorStoreClient:
    """Create a vector store client based on configuration.

    Reads the ``vector_backend`` setting from config or the
    ``ZOTERO_VECTOR_BACKEND`` environment variable.  Defaults to
    ``"chroma"`` for backward compatibility.

    Supported backends: ``"chroma"``, ``"pinecone"``.
    """
    backend = get_vector_backend(config_path)

    if backend == "pinecone":
        from .pinecone_client import create_pinecone_client

        return create_pinecone_client(config_path)
    elif backend == "chroma":
        from .chroma_client import create_chroma_client

        return create_chroma_client(config_path)
    else:
        raise ValueError(f"Unknown vector backend: '{backend}'. Supported backends: 'chroma', 'pinecone'")
