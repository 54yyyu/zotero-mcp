"""
Pinecone client for semantic search functionality.

This module provides a Pinecone-backed vector database for semantic search
over Zotero libraries. It implements the same interface as ChromaClient,
allowing it to be used as a drop-in replacement.
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Maximum bytes for document text stored in Pinecone metadata.
# Pinecone allows 40KB total metadata per vector; we reserve ~5KB for
# structured fields and store up to 35KB of document text.
_MAX_DOC_TEXT_BYTES = 35_000

# Known embedding dimensions for auto-creating Pinecone indexes.
_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "default": 384,  # all-MiniLM-L6-v2
    "openai": 1536,  # text-embedding-3-small
    "gemini": 768,  # gemini-embedding-001
    "qwen": 1024,  # Qwen3-Embedding-0.6B
    "embeddinggemma": 384,  # embeddinggemma-300m
}


# ---------------------------------------------------------------------------
# Standalone embedding functions (no chromadb dependency)
# ---------------------------------------------------------------------------


class _BaseEmbedding:
    """Base class for embedding functions used by PineconeClient."""

    max_input_tokens: int = 8000
    model_name: str = ""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.__call__([text])[0]

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately *max_tokens* tokens."""
        max_chars = max_tokens * 3
        return text[:max_chars] if len(text) > max_chars else text


class _OpenAIEmbedding(_BaseEmbedding):
    max_input_tokens = 8000

    def __init__(
        self, model_name: str = "text-embedding-3-small", api_key: str | None = None, base_url: str | None = None
    ):
        self.model_name = model_name
        import openai

        kwargs: dict[str, Any] = {"api_key": api_key or os.getenv("OPENAI_API_KEY")}
        resolved_base = base_url or os.getenv("OPENAI_BASE_URL")
        if resolved_base:
            kwargs["base_url"] = resolved_base
        self.client = openai.OpenAI(**kwargs)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [data.embedding for data in response.data]

    def truncate(self, text: str, max_tokens: int) -> str:
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            if len(tokens) > max_tokens:
                return enc.decode(tokens[:max_tokens])
        except ImportError:
            max_chars = max_tokens * 3
            if len(text) > max_chars:
                return text[:max_chars]
        return text


class _GeminiEmbedding(_BaseEmbedding):
    max_input_tokens = 2000

    def __init__(
        self, model_name: str = "gemini-embedding-001", api_key: str | None = None, base_url: str | None = None
    ):
        self.model_name = model_name
        from google import genai
        from google.genai import types

        resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        kwargs: dict[str, Any] = {"api_key": resolved_key}
        resolved_base = base_url or os.getenv("GEMINI_BASE_URL")
        if resolved_base:
            kwargs["http_options"] = types.HttpOptions(baseUrl=resolved_base)
        self.client = genai.Client(**kwargs)
        self._types = types

    def __call__(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=[text],
                config=self._types.EmbedContentConfig(
                    task_type="retrieval_document",
                    title="Zotero library document",
                ),
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=[text],
            config=self._types.EmbedContentConfig(task_type="retrieval_query"),
        )
        return response.embeddings[0].values

    def truncate(self, text: str, max_tokens: int) -> str:
        max_chars = max_tokens * 4
        return text[:max_chars] if len(text) > max_chars else text


class _HuggingFaceEmbedding(_BaseEmbedding):
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.model_name = model_name
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.max_input_tokens = getattr(self.model, "max_seq_length", 500)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def truncate(self, text: str, max_tokens: int) -> str:
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) > max_tokens:
                return tokenizer.decode(encoded[:max_tokens])
        else:
            max_chars = max_tokens * 2
            if len(text) > max_chars:
                return text[:max_chars]
        return text


class _DefaultEmbedding(_BaseEmbedding):
    """Uses sentence-transformers all-MiniLM-L6-v2 (same as ChromaDB default)."""

    max_input_tokens = 256

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model_name = "all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def truncate(self, text: str, max_tokens: int) -> str:
        max_chars = max_tokens * 2
        return text[:max_chars] if len(text) > max_chars else text


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Ensure all metadata values are Pinecone-compatible types.

    Pinecone accepts: str, int, float, bool, and list[str].
    None values and unsupported types are dropped.
    """
    clean: dict[str, Any] = {}
    for k, v in metadata.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            clean[k] = v
        else:
            # Convert to string as fallback
            clean[k] = str(v)
    return clean


def _translate_filter(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate ChromaDB-style metadata filters to Pinecone format.

    ChromaDB: ``{"item_type": "journalArticle"}``  (shorthand equality)
    Pinecone: ``{"item_type": {"$eq": "journalArticle"}}``

    Logical operators (``$and``, ``$or``) and explicit operator dicts are
    passed through unchanged.
    """
    if not where:
        return None

    translated: dict[str, Any] = {}
    for key, value in where.items():
        if key.startswith("$"):
            # Logical operators ($and, $or, $not)
            if isinstance(value, list):
                translated[key] = [_translate_filter(f) for f in value]
            else:
                translated[key] = value
        elif isinstance(value, dict):
            # Already in operator format, e.g. {"$gt": 5}
            translated[key] = value
        else:
            # Simple equality shorthand → explicit $eq
            translated[key] = {"$eq": value}
    return translated


# ---------------------------------------------------------------------------
# PineconeClient
# ---------------------------------------------------------------------------


class PineconeClient:
    """Pinecone vector database client for Zotero semantic search.

    Implements the same public interface as ``ChromaClient`` so it can be
    used as a drop-in alternative backend.
    """

    def __init__(
        self,
        collection_name: str = "zotero_library",
        embedding_model: str = "default",
        embedding_config: dict[str, Any] | None = None,
        api_key: str | None = None,
        index_name: str | None = None,
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_config = embedding_config or {}
        self.namespace = collection_name  # map collection → Pinecone namespace

        # Resolve API key
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Pinecone API key is required. Set PINECONE_API_KEY environment "
                "variable or pass api_key to PineconeClient."
            )

        from pinecone import Pinecone

        self.pc = Pinecone(api_key=self.api_key)

        # Embedding function (standalone, no chromadb dependency)
        self.embedding_function = self._create_embedding_function()

        # Resolve index name
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "zotero-library")

        # Ensure index exists (creates if needed)
        self._ensure_index(cloud, region)
        self.index = self.pc.Index(self.index_name)

        logger.info(
            f"Pinecone client initialized: index={self.index_name}, "
            f"namespace={self.namespace}, model={self.embedding_model}"
        )

    # ----- embedding setup ---------------------------------------------------

    def _create_embedding_function(self) -> _BaseEmbedding:
        """Create the appropriate embedding function."""
        if self.embedding_model == "openai":
            return _OpenAIEmbedding(
                model_name=self.embedding_config.get("model_name", "text-embedding-3-small"),
                api_key=self.embedding_config.get("api_key"),
                base_url=self.embedding_config.get("base_url"),
            )
        elif self.embedding_model == "gemini":
            return _GeminiEmbedding(
                model_name=self.embedding_config.get("model_name", "gemini-embedding-001"),
                api_key=self.embedding_config.get("api_key"),
                base_url=self.embedding_config.get("base_url"),
            )
        elif self.embedding_model == "qwen":
            return _HuggingFaceEmbedding(
                model_name=self.embedding_config.get("model_name", "Qwen/Qwen3-Embedding-0.6B"),
            )
        elif self.embedding_model == "embeddinggemma":
            return _HuggingFaceEmbedding(
                model_name=self.embedding_config.get("model_name", "google/embeddinggemma-300m"),
            )
        elif self.embedding_model not in ("default", "openai", "gemini"):
            # Treat any other value as a HuggingFace model name
            return _HuggingFaceEmbedding(model_name=self.embedding_model)
        else:
            return _DefaultEmbedding()

    def _get_embedding_dimension(self) -> int:
        """Determine the embedding vector dimension for index creation."""
        model_name = self.embedding_config.get("model_name", "")

        if self.embedding_model in _EMBEDDING_DIMENSIONS:
            # Handle OpenAI large model variant
            if self.embedding_model == "openai" and "large" in model_name:
                return 3072
            return _EMBEDDING_DIMENSIONS[self.embedding_model]

        # Unknown model — generate a test embedding to discover dimension
        test_vec = self.embedding_function.embed_query("dimension test")
        return len(test_vec)

    # ----- index management --------------------------------------------------

    def _ensure_index(self, cloud: str, region: str) -> None:
        """Create the Pinecone index if it does not already exist."""
        existing_names = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name in existing_names:
            return

        dimension = self._get_embedding_dimension()
        logger.info(f"Creating Pinecone index '{self.index_name}' (dim={dimension}, cloud={cloud}, region={region})")

        from pinecone import ServerlessSpec

        self.pc.create_index(
            name=self.index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

        # Wait for the index to become ready
        import time

        for _ in range(120):  # max ~2 minutes
            desc = self.pc.describe_index(self.index_name)
            if desc.status.get("ready", False):
                break
            time.sleep(1)
        else:
            logger.warning("Pinecone index creation timed out; continuing anyway")

    # ----- public interface (mirrors ChromaClient) ---------------------------

    @property
    def embedding_max_tokens(self) -> int:
        return getattr(self.embedding_function, "max_input_tokens", 8000)

    def truncate_text(self, text: str, max_tokens: int | None = None) -> str:
        if max_tokens is None:
            max_tokens = self.embedding_max_tokens
        if hasattr(self.embedding_function, "truncate"):
            return self.embedding_function.truncate(text, max_tokens)
        # Fallback
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            if len(tokens) > max_tokens:
                return enc.decode(tokens[:max_tokens])
        except Exception:
            max_chars = max_tokens * 2
            if len(text) > max_chars:
                return text[:max_chars]
        return text

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        self.upsert_documents(documents, metadatas, ids)

    def upsert_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        if not documents:
            return

        # Generate embeddings
        embeddings = self.embedding_function(documents)

        vectors: list[dict[str, Any]] = []
        for i in range(len(documents)):
            metadata = _sanitize_metadata(metadatas[i])

            # Store document text in metadata (respecting size limit)
            doc_text = documents[i]
            encoded = doc_text.encode("utf-8")
            if len(encoded) > _MAX_DOC_TEXT_BYTES:
                doc_text = encoded[:_MAX_DOC_TEXT_BYTES].decode("utf-8", errors="ignore")
            metadata["_document_text"] = doc_text

            values = embeddings[i]
            if hasattr(values, "tolist"):
                values = values.tolist()

            vectors.append({"id": ids[i], "values": values, "metadata": metadata})

        # Pinecone recommends max 100 vectors per upsert call
        batch_size = 100
        for start in range(0, len(vectors), batch_size):
            batch = vectors[start : start + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)

        logger.info(f"Upserted {len(documents)} documents to Pinecone")

    def search(
        self,
        query_texts: list[str],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        all_ids: list[list[str]] = []
        all_distances: list[list[float]] = []
        all_documents: list[list[str]] = []
        all_metadatas: list[list[dict[str, Any]]] = []

        pinecone_filter = _translate_filter(where)

        for query in query_texts:
            query_vector = self.embedding_function.embed_query(query)
            if hasattr(query_vector, "tolist"):
                query_vector = query_vector.tolist()

            query_kwargs: dict[str, Any] = {
                "namespace": self.namespace,
                "vector": query_vector,
                "top_k": n_results,
                "include_metadata": True,
            }
            if pinecone_filter:
                query_kwargs["filter"] = pinecone_filter

            response = self.index.query(**query_kwargs)

            ids: list[str] = []
            distances: list[float] = []
            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []

            for match in response.matches:
                ids.append(match.id)
                # Pinecone returns cosine similarity (0–1, higher = closer).
                # ChromaDB returns L2 distances (lower = closer).
                # semantic_search.py computes: similarity = 1 - distance
                # So we convert: distance = 1 - score
                distances.append(1.0 - (match.score or 0.0))

                metadata = dict(match.metadata) if match.metadata else {}
                doc_text = metadata.pop("_document_text", "")
                documents.append(doc_text)
                metadatas.append(metadata)

            all_ids.append(ids)
            all_distances.append(distances)
            all_documents.append(documents)
            all_metadatas.append(metadatas)

        logger.info(f"Pinecone search returned {len(all_ids[0]) if all_ids else 0} results")

        return {
            "ids": all_ids,
            "distances": all_distances,
            "documents": all_documents,
            "metadatas": all_metadatas,
        }

    def delete_documents(self, ids: list[str]) -> None:
        if not ids:
            return
        self.index.delete(ids=ids, namespace=self.namespace)
        logger.info(f"Deleted {len(ids)} documents from Pinecone")

    def get_collection_info(self) -> dict[str, Any]:
        try:
            stats = self.index.describe_index_stats()
            ns_stats = stats.namespaces.get(self.namespace)
            count = ns_stats.vector_count if ns_stats else 0
            return {
                "name": self.collection_name,
                "count": count,
                "embedding_model": self.embedding_model,
                "persist_directory": f"pinecone://{self.index_name}/{self.namespace}",
                "backend": "pinecone",
            }
        except Exception as e:
            logger.error(f"Error getting Pinecone collection info: {e}")
            return {
                "name": self.collection_name,
                "count": 0,
                "embedding_model": self.embedding_model,
                "persist_directory": f"pinecone://{self.index_name}/{self.namespace}",
                "backend": "pinecone",
                "error": str(e),
            }

    def reset_collection(self) -> None:
        """Delete all vectors in the current namespace."""
        self.index.delete(delete_all=True, namespace=self.namespace)
        logger.info(f"Reset Pinecone namespace '{self.namespace}'")

    def document_exists(self, doc_id: str) -> bool:
        try:
            result = self.index.fetch(ids=[doc_id], namespace=self.namespace)
            return doc_id in (result.vectors or {})
        except Exception:
            return False

    def get_document_metadata(self, doc_id: str) -> dict[str, Any] | None:
        try:
            result = self.index.fetch(ids=[doc_id], namespace=self.namespace)
            vectors = result.vectors or {}
            if doc_id in vectors:
                metadata = dict(vectors[doc_id].metadata or {})
                metadata.pop("_document_text", None)
                return metadata
            return None
        except Exception:
            return None

    def get_existing_ids(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        existing: set[str] = set()
        try:
            # Pinecone fetch supports up to 1000 IDs per call
            for start in range(0, len(ids), 1000):
                batch = ids[start : start + 1000]
                result = self.index.fetch(ids=batch, namespace=self.namespace)
                existing.update((result.vectors or {}).keys())
        except Exception:
            pass
        return existing

    def get_all_documents(
        self,
        limit: int = 20,
        include_documents: bool = False,
    ) -> dict[str, Any]:
        """Retrieve a sample of stored vectors (for inspection/debugging).

        Returns a dict matching the structure expected by the CLI db-inspect
        command: ``{"ids": [...], "metadatas": [...], "documents": [...]}``.

        Pinecone does not support listing all vectors efficiently; we use the
        list endpoint (pagination over IDs) to fetch up to *limit* vectors.
        """
        ids_to_fetch: list[str] = []
        try:
            paginator = self.index.list(namespace=self.namespace, limit=limit)
            for page in paginator:
                ids_to_fetch.extend(page)
                if len(ids_to_fetch) >= limit:
                    break
        except Exception as e:
            logger.warning(f"Pinecone list failed: {e}")
            return {"ids": [], "metadatas": [], "documents": []}

        ids_to_fetch = ids_to_fetch[:limit]
        if not ids_to_fetch:
            return {"ids": [], "metadatas": [], "documents": []}

        result = self.index.fetch(ids=ids_to_fetch, namespace=self.namespace)
        vectors = result.vectors or {}

        out_ids: list[str] = []
        out_metadatas: list[dict[str, Any]] = []
        out_documents: list[str] = []

        for vid in ids_to_fetch:
            vec = vectors.get(vid)
            if not vec:
                continue
            out_ids.append(vid)
            metadata = dict(vec.metadata or {})
            doc_text = metadata.pop("_document_text", "")
            out_metadatas.append(metadata)
            out_documents.append(doc_text if include_documents else "")

        return {"ids": out_ids, "metadatas": out_metadatas, "documents": out_documents}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_pinecone_client(config_path: str | None = None) -> PineconeClient:
    """Create a PineconeClient instance from configuration.

    Reads settings from *config_path* (JSON) and environment variables,
    mirroring the behaviour of ``create_chroma_client``.
    """
    config: dict[str, Any] = {
        "collection_name": "zotero_library",
        "embedding_model": "default",
        "embedding_config": {},
    }
    pinecone_config: dict[str, Any] = {}

    # Load from file
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                file_config = json.load(f)
                ss = file_config.get("semantic_search", {})
                config.update({k: v for k, v in ss.items() if k in config})
                pinecone_config = ss.get("pinecone", {})
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {e}")

    # Environment variable overrides
    env_embedding_model = os.getenv("ZOTERO_EMBEDDING_MODEL")
    if env_embedding_model:
        config["embedding_model"] = env_embedding_model

    # Resolve embedding config from env (same logic as chroma_client)
    if config["embedding_model"] == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        base_url = os.getenv("OPENAI_BASE_URL")
        if api_key:
            config["embedding_config"] = {"api_key": api_key, "model_name": model}
            if base_url:
                config["embedding_config"]["base_url"] = base_url
    elif config["embedding_model"] == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        base_url = os.getenv("GEMINI_BASE_URL")
        if api_key:
            config["embedding_config"] = {"api_key": api_key, "model_name": model}
            if base_url:
                config["embedding_config"]["base_url"] = base_url

    return PineconeClient(
        collection_name=config["collection_name"],
        embedding_model=config["embedding_model"],
        embedding_config=config["embedding_config"],
        api_key=pinecone_config.get("api_key"),
        index_name=pinecone_config.get("index_name"),
        cloud=pinecone_config.get("cloud", "aws"),
        region=pinecone_config.get("region", "us-east-1"),
    )
