"""
ChromaDB client for semantic search functionality.

This module provides persistent vector database storage for semantic search
over Zotero libraries.

The embedding functions themselves now live in :mod:`zotero_mcp.embeddings`.
They are re-exported below because ``zotero_mcp.chroma_client`` is where every
caller — and every existing test — imports them from, and because ChromaDB's
registry maps a persisted collection's embedding-function name to a specific
class object, so the re-exported name has to *be* the registered class.
"""

import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Re-exported for backward compatibility; see the module docstring.
from zotero_mcp.embeddings.providers import (  # noqa: F401
    CUSTOM_EMBEDDING_FUNCTIONS,
    GeminiEmbeddingFunction,
    HuggingFaceEmbeddingFunction,
    OllamaEmbeddingFunction,
    OpenAIEmbeddingFunction,
    ensure_embedding_functions_registered,
)
from zotero_mcp.embeddings.registry import create_embedding_function, merge_env_config
from zotero_mcp.utils import install_hint, suppress_stdout

try:
    import chromadb
    from chromadb import Documents, EmbeddingFunction, Embeddings
    from chromadb.config import Settings
except ImportError as e:
    raise ImportError(
        f"chromadb is required for semantic search. {install_hint('semantic')}"
    ) from e

logger = logging.getLogger(__name__)


class ChromaClient:
    """ChromaDB client for Zotero semantic search."""

    def __init__(self,
                 collection_name: str = "zotero_library",
                 persist_directory: str | None = None,
                 embedding_model: str = "default",
                 embedding_config: dict[str, Any] | None = None):
        """
        Initialize ChromaDB client.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Model to use for embeddings ('default', 'openai', 'gemini', 'ollama', 'qwen', 'embeddinggemma', or HuggingFace model name)
            embedding_config: Configuration for the embedding model
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_config = embedding_config or {}

        # Set up persistent directory
        if persist_directory is None:
            # Use user's config directory by default
            config_dir = Path.home() / ".config" / "zotero-mcp"
            config_dir.mkdir(parents=True, exist_ok=True)
            persist_directory = str(config_dir / "chroma_db")

        self.persist_directory = persist_directory

        # Make sure our classes — not ChromaDB's same-named built-ins — answer
        # the registry lookup used when a persisted collection config is
        # rebuilt below (issue #382).
        ensure_embedding_functions_registered()

        # Initialize ChromaDB client with stdout suppression
        with suppress_stdout():
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Set up embedding function
            self.embedding_function = self._create_embedding_function()

            # Get or create collection with the configured embedding function.
            # If the user switched embedding models, the persisted collection
            # will have stale config.  Detect the mismatch and drop/recreate.
            try:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )

                # ChromaDB may silently persist the old embedding function config.
                # Check if the stored config matches what we want; if not, recreate.
                stored_config = getattr(self.collection, 'metadata', {}) or {}
                if not stored_config:
                    # Try reading config from the collection's config_json_str
                    try:
                        import json as _json
                        rows = self.client._sysdb.get_collections(name=self.collection_name)
                        if rows:
                            raw = getattr(rows[0], 'config_json_str', None) or '{}'
                            cfg = _json.loads(raw)
                            ef_cfg = cfg.get('embedding_function', {}).get('config', {})
                            stored_model = ef_cfg.get('model_name', '')
                            # Compare stored model with configured model
                            configured_model = getattr(self.embedding_function, 'model_name', None)
                            if stored_model and configured_model and stored_model != configured_model:
                                logger.warning(
                                    f"Stored embedding model '{stored_model}' differs from "
                                    f"configured '{configured_model}'. Resetting collection."
                                )
                                self.client.delete_collection(name=self.collection_name)
                                self.collection = self.client.create_collection(
                                    name=self.collection_name,
                                    embedding_function=self.embedding_function
                                )
                    except Exception:
                        pass  # Best-effort check; proceed with existing collection

            except Exception as e:
                if "embedding function conflict" in str(e).lower():
                    logger.warning(
                        f"Embedding model changed to '{self.embedding_model}'. "
                        "Resetting collection for rebuild."
                    )
                    self.client.delete_collection(name=self.collection_name)
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        embedding_function=self.embedding_function
                    )
                else:
                    raise

    def _create_embedding_function(self) -> EmbeddingFunction:
        """Create the appropriate embedding function based on configuration.

        The provider registry owns the model-string -> embedding-function
        mapping; see :func:`zotero_mcp.embeddings.registry.resolve_provider`
        for the resolution rules.
        """
        return create_embedding_function(self.embedding_model, self.embedding_config)

    @property
    def embedding_max_tokens(self) -> int:
        """Maximum input tokens supported by the configured embedding model."""
        return getattr(self.embedding_function, "max_input_tokens", 8000)

    def truncate_text(self, text: str, max_tokens: int | None = None) -> str:
        """Truncate text using the embedding function's model-aware tokenizer.

        Falls back to tiktoken cl100k_base or character estimation if the
        embedding function does not provide a truncate method.
        """
        if max_tokens is None:
            max_tokens = self.embedding_max_tokens
        if hasattr(self.embedding_function, 'truncate'):
            return self.embedding_function.truncate(text, max_tokens)
        # Fallback for default ChromaDB embedding function
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text, disallowed_special=())
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                text = enc.decode(tokens)
        except Exception:
            max_chars = max_tokens * 2
            if len(text) > max_chars:
                text = text[:max_chars]
        return text

    def add_documents(self,
                     documents: list[str],
                     metadatas: list[dict[str, Any]],
                     ids: list[str]) -> None:
        """
        Add documents to the collection.

        Args:
            documents: List of document texts to embed
            metadatas: List of metadata dictionaries for each document
            ids: List of unique IDs for each document
        """
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to ChromaDB collection")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise

    def upsert_documents(self,
                        documents: list[str],
                        metadatas: list[dict[str, Any]],
                        ids: list[str]) -> None:
        """
        Upsert (update or insert) documents to the collection.

        Args:
            documents: List of document texts to embed
            metadatas: List of metadata dictionaries for each document
            ids: List of unique IDs for each document
        """
        try:
            # ChromaDB rejects batches larger than its max_batch_size
            # (~5461). With passage-chunking enabled a batch of 25 books
            # easily exceeds that, so split instead of failing.
            try:
                max_batch = int(self.client.get_max_batch_size())
            except Exception:
                max_batch = 5000
            for i in range(0, len(ids), max_batch):
                self.collection.upsert(
                    documents=documents[i:i + max_batch],
                    metadatas=metadatas[i:i + max_batch],
                    ids=ids[i:i + max_batch]
                )
            logger.info(f"Upserted {len(documents)} documents to ChromaDB collection")
        except Exception as e:
            logger.error(f"Error upserting documents to ChromaDB: {e}")
            raise

    def upsert_embeddings(self,
                         documents: list[str],
                         metadatas: list[dict[str, Any]],
                         ids: list[str],
                         embeddings: list[list[float]]) -> None:
        """
        Upsert documents with precomputed embeddings.

        Used by OpenAI Batch API imports so ChromaDB stores the vectors
        returned asynchronously without calling the realtime embeddings API.
        """
        try:
            # Same max_batch_size constraint as upsert_documents.
            try:
                max_batch = int(self.client.get_max_batch_size())
            except Exception:
                max_batch = 5000
            for i in range(0, len(ids), max_batch):
                self.collection.upsert(
                    documents=documents[i:i + max_batch],
                    metadatas=metadatas[i:i + max_batch],
                    ids=ids[i:i + max_batch],
                    embeddings=embeddings[i:i + max_batch],
                )
            logger.info(f"Upserted {len(documents)} precomputed embeddings to ChromaDB collection")
        except Exception as e:
            logger.error(f"Error upserting precomputed embeddings to ChromaDB: {e}")
            raise

    def search(self,
               query_texts: list[str],
               n_results: int = 10,
               where: dict[str, Any] | None = None,
               where_document: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Search for similar documents.

        Args:
            query_texts: List of query texts
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions

        Returns:
            Search results from ChromaDB
        """
        try:
            query_kwargs = {
                "n_results": n_results,
                "where": where,
                "where_document": where_document,
            }

            # Use embed_query for our custom embedding functions that implement
            # correct query-time task types (e.g. Gemini retrieval_query).
            # Do NOT use embed_query on ChromaDB's DefaultEmbeddingFunction —
            # its embed_query returns chunked results, not a single vector.
            _is_custom_ef = isinstance(
                self.embedding_function,
                (OpenAIEmbeddingFunction, GeminiEmbeddingFunction, HuggingFaceEmbeddingFunction, OllamaEmbeddingFunction),
            )
            if _is_custom_ef and hasattr(self.embedding_function, 'embed_query') and query_texts:
                query_embeddings = []
                for qt in query_texts:
                    emb = self.embedding_function.embed_query(qt)
                    # Ensure plain Python floats (some providers return numpy)
                    if hasattr(emb, 'tolist'):
                        emb = emb.tolist()
                    query_embeddings.append(emb)
                query_kwargs["query_embeddings"] = query_embeddings
            else:
                query_kwargs["query_texts"] = query_texts

            results = self.collection.query(**query_kwargs)
            logger.info(f"Semantic search returned {len(results.get('ids', [[]])[0])} results")
            return results
        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            raise

    def delete_documents(self, ids: list[str]) -> None:
        """
        Delete documents from the collection.

        Args:
            ids: List of document IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from ChromaDB collection")
        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {e}")
            raise

    def delete_item_chunks(self, item_key: str, group_id: int | None = None) -> None:
        """Delete all passage chunks belonging to one item (chunked collections).

        Passage chunks carry ``parent_item_key`` in their metadata; deleting by
        that key clears every ``<item_key>#<n>`` entry for the item before its
        chunks are re-upserted, so a document that shrank to fewer passages
        never leaves orphaned chunks behind. No-op-safe on item-level
        collections (nothing matches the filter).

        Args:
            item_key: Parent item whose chunks to delete.
            group_id: When given, restrict the delete to chunks attributed to
                that library — the deletion pass passes its run scope so a
                mixed-attribution chunk set (partial rewrite, key collision)
                never loses another library's chunks.
        """
        where: dict[str, Any] = {"parent_item_key": item_key}
        if group_id is not None:
            where = {"$and": [{"parent_item_key": item_key}, {"group_id": int(group_id)}]}
        try:
            self.collection.delete(where=where)
        except Exception as e:
            logger.warning(f"delete_item_chunks({item_key}) failed: {e}")

    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "name": self.collection_name,
                "count": 0,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory,
                "error": str(e)
            }

    def reset_collection(self) -> None:
        """Reset (clear) the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Reset ChromaDB collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the collection."""
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except Exception:
            return False

    def get_document_metadata(self, doc_id: str) -> dict[str, Any] | None:
        """
        Get metadata for an item if it is indexed.

        With passage chunking enabled, an item is stored only under its chunk
        ids (``<key>#<n>``) and never under the bare item key, so an exact-id
        lookup on the key alone misses every chunked item. Chunk 0 carries the
        same item-level metadata (``date_modified``, ``has_fulltext``) that
        callers need, so fall back to it.

        Args:
            doc_id: Item key (or full document id) to look up

        Returns:
            Metadata dictionary if the item is indexed, None otherwise
        """
        try:
            result = self.collection.get(ids=[doc_id, f"{doc_id}#0"], include=["metadatas"])
            if result['ids'] and result['metadatas']:
                return result['metadatas'][0]
            return None
        except Exception:
            return None

    def get_existing_ids(self, ids: list[str]) -> set[str]:
        """Return the subset of ids that already exist in the collection."""
        if not ids:
            return set()
        try:
            result = self.collection.get(ids=ids, include=[])
            return set(result.get("ids", []))
        except Exception:
            return set()

    def get_all_ids(self, where: dict[str, Any] | None = None) -> set[str]:
        """Return every id currently stored in the collection.

        Used by incremental sync to compute deletions: items in the local
        collection but no longer present in the Zotero library.

        Args:
            where: Optional ChromaDB metadata filter (e.g.
                ``{"group_id": 0}``) applied DB-side. Documents missing a
                filtered key never match, so callers scoping by ``group_id``
                structurally exclude unattributed documents.

        Errors return an empty set, which every caller treats as "nothing
        eligible" — deletion passes fail toward deleting nothing.
        """
        try:
            result = self.collection.get(where=where, include=[])
            return set(result.get("ids", []))
        except Exception as e:
            logger.error(f"Error listing collection ids: {e}")
            return set()

    def iter_metadatas(self, batch_size: int = 500) -> Iterator[tuple[list[str], list[dict[str, Any]]]]:
        """Stream ``(ids, metadatas)`` over the whole collection in batches.

        Snapshots the id set first, then pages metadata with
        ``collection.get(ids=chunk)`` — deliberately not ``limit``/``offset``,
        whose ordering across pages is an undocumented implementation detail,
        and never one unbounded ids list (the "too many SQL variables"
        failure, #368). Callers may update already-yielded documents between
        batches (the group_id backfill does); a snapshot makes that safe by
        construction. Documents deleted mid-iteration are simply absent from
        their page.

        Backend failures RAISE — deliberately not routed through
        ``get_all_ids``, whose swallow-into-empty-set contract is safe for
        deletion ("nothing eligible") but inverts here: the backfill's
        one-time schema gate closes permanently on "success", so an error
        masquerading as an empty collection would silently disable the
        migration with zero documents tagged.
        """
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        # Cap at the same order as the backend batch limit so a caller cannot
        # accidentally defeat the bounded-ids protection this method exists
        # to provide.
        batch_size = min(batch_size, 5000)
        all_ids = sorted(self.collection.get(include=[]).get("ids") or [])
        for start in range(0, len(all_ids), batch_size):
            chunk = all_ids[start:start + batch_size]
            result = self.collection.get(ids=chunk, include=["metadatas"])
            ids = result.get("ids") or []
            if not ids:
                continue
            metadatas = result.get("metadatas") or [{} for _ in ids]
            yield ids, metadatas

    def update_metadatas(self, ids: list[str], metadatas: list[dict[str, Any]]) -> None:
        """Metadata-only update for existing documents; never re-embeds.

        Callers must pass complete metadata dicts: chromadb 1.5.x merges
        the given keys into existing metadata, while other versions have
        replaced the whole dict — full dicts behave identically under both.
        Split under the backend's max batch size like ``upsert_documents``
        (#369).
        """
        if not ids:
            return
        try:
            try:
                max_batch = int(self.client.get_max_batch_size())
            except Exception:
                max_batch = 5000
            if max_batch < 1:
                max_batch = 5000
            for i in range(0, len(ids), max_batch):
                self.collection.update(
                    ids=ids[i:i + max_batch],
                    metadatas=metadatas[i:i + max_batch],
                )
            logger.info(f"Updated metadata for {len(ids)} documents")
        except Exception as e:
            logger.error(f"Error updating document metadata: {e}")
            raise


def create_chroma_client(config_path: str | None = None) -> ChromaClient:
    """
    Create a ChromaClient instance from configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured ChromaClient instance
    """
    # Default configuration
    config = {
        "collection_name": "zotero_library",
        "embedding_model": "default",
        "embedding_config": {}
    }

    # Load configuration from file if it exists
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                file_config = json.load(f)
                config.update(file_config.get("semantic_search", {}))
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {e}")

    # Pick the embedding model. config.json is the richer, authoritative source
    # (it also carries the matching api_key / base_url / model_name), so it wins
    # over the ZOTERO_EMBEDDING_MODEL env var whenever it names a concrete model.
    # The env var only fills the gap when config.json is absent or left at the
    # "default" placeholder — which is the normal Claude Desktop case.
    #
    # This deliberately guards against a stale env value silently downgrading an
    # explicitly configured provider: Claude Desktop passes its server `env`
    # block on every launch and can rewrite that file on its own, so a leftover
    # ZOTERO_EMBEDDING_MODEL=default there would otherwise override a Gemini/
    # OpenAI config.json and break search with an opaque embedding-dimension
    # mismatch against the persisted collection.
    env_embedding_model = os.getenv("ZOTERO_EMBEDDING_MODEL")
    if env_embedding_model and config.get("embedding_model", "default") in (None, "default"):
        config["embedding_model"] = env_embedding_model

    # Merge embedding config from environment (config.json wins, env fills gaps).
    # Precedence: explicit config.json value > env var > hardcoded default.
    # Previous code unconditionally REPLACED config["embedding_config"] with env
    # values, silently dropping model_name from config.json whenever any
    # provider env var (e.g. GOOGLE_API_KEY leaked from another tool) was set.
    # Which variables each provider reads, and whether the merged result is
    # kept, is declared by its EnvSpec in the provider registry.
    config["embedding_config"] = merge_env_config(
        config["embedding_model"], config.get("embedding_config")
    )

    return ChromaClient(
        collection_name=config["collection_name"],
        embedding_model=config["embedding_model"],
        embedding_config=config["embedding_config"]
    )


class _NoEmbeddingFunction(EmbeddingFunction):
    """Placeholder embedding function used for read-only status reads.

    Passing an explicit embedding function to ``get_collection`` stops ChromaDB
    from reconstructing the collection's persisted embedding function — which,
    for the default backend, eagerly downloads the ~80MB ONNX MiniLM model.
    Counting rows never embeds anything, so this is never actually called; it
    raises if it ever is, to make misuse loud rather than silently wrong.

    ``name()`` MUST return ``"default"``. ChromaDB >=1.x validates the supplied
    embedding function against the collection's persisted config in
    ``validate_embedding_function_conflict_on_get`` and raises a ``ValueError``
    whenever the supplied ``name()`` differs from the persisted one — *unless*
    the supplied name is ``"default"``, which short-circuits the check. Without
    this, opening a collection that was built with any real backend (default,
    openai, gemini, ...) raises a conflict; ``read_collection_status`` then
    swallowed that error and reported "0 documents / not initialized" against a
    fully populated database (issue #362).
    """

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:  # pragma: no cover - never invoked
        raise RuntimeError("embedding is unavailable in status-only mode")

    @staticmethod
    def name() -> str:
        return "default"


def read_collection_status(
    config_path: str | None = None,
    *,
    persist_directory: str | None = None,
) -> dict[str, Any]:
    """Read ChromaDB collection stats WITHOUT loading an embedding model.

    The full :class:`ChromaClient` constructor builds the embedding function,
    which for the default backend downloads the ONNX MiniLM model on first use —
    turning a read-only status check into a multi-minute (or network-blocked,
    indefinite) hang. Reporting readiness only needs the row count and the
    configured model name, neither of which requires the model itself. This
    opens the persisted database directly and reads the count, mirroring the
    shape returned by :meth:`ChromaClient.get_collection_info`.

    ``persist_directory`` defaults to ``ChromaClient``'s location
    (``~/.config/zotero-mcp/chroma_db``); it is parameterised for testing.
    """
    collection_name = "zotero_library"
    embedding_model = "default"

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                semantic_cfg = json.load(f).get("semantic_search", {})
            collection_name = semantic_cfg.get("collection_name", collection_name)
            embedding_model = semantic_cfg.get("embedding_model", embedding_model)
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {e}")

    # Mirror create_chroma_client's precedence: config.json wins; the env var
    # only fills in when the file left the model at the "default" placeholder.
    env_model = os.getenv("ZOTERO_EMBEDDING_MODEL")
    if env_model and embedding_model in (None, "default"):
        embedding_model = env_model

    if persist_directory is None:
        persist_directory = str(Path.home() / ".config" / "zotero-mcp" / "chroma_db")
    base = {
        "name": collection_name,
        "embedding_model": embedding_model,
        "persist_directory": persist_directory,
    }

    try:
        with suppress_stdout():
            client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            try:
                collection = client.get_collection(
                    name=collection_name,
                    embedding_function=_NoEmbeddingFunction(),
                )
            except Exception:
                # Collection does not exist yet — database not initialized.
                return {**base, "count": 0, "initialized": False}
            count = collection.count()
        return {**base, "count": count, "initialized": True}
    except Exception as e:
        logger.error(f"Error reading collection status: {e}")
        return {**base, "count": 0, "error": str(e)}
