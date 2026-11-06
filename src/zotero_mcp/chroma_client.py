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
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from zotero_mcp.config import resolve_chroma_dir

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
from zotero_mcp.utils import ensure_private_dir, install_hint, suppress_stdout

try:
    import chromadb
    from chromadb import EmbeddingFunction
    from chromadb.config import Settings
except ImportError as e:
    raise ImportError(
        f"chromadb is required for semantic search. {install_hint('semantic')}"
    ) from e

logger = logging.getLogger(__name__)

ARTICLES = "articles"
PASSAGES = "passages"


def passage_collection_name(collection_name: str) -> str:
    """The collection holding the passages of the article collection ``collection_name``."""
    return f"{collection_name}_passages"


class ChromaClient:
    """ChromaDB client for Zotero semantic search.

    An index holds up to two collections, using the same embedding model:
    ``collection_name`` for articles (one record per item, id ``KEY``) and
    :func:`passage_collection_name` for passages (id ``KEY#n``). Keeping them
    apart means an article search never loads passage vectors: ChromaDB loads a
    collection's whole vector index on first use, deleted records included.

    The passage collection is opened on first use and not before, so an index
    that never had passages never gets an empty collection.

    A chunked index built before the split holds its passages in the article
    collection. It keeps working that way until it is migrated:
    :meth:`separate_passages` is False for it and every id stays there.
    """

    #: Whether writes go to the collection their id belongs to.
    route_passages = False

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
        if not persist_directory:
            # Use user's config directory by default
            ensure_private_dir(Path.home() / ".config" / "zotero-mcp")

        self.persist_directory = str(resolve_chroma_dir(persist_directory))
        if sys.platform == "win32" and not os.path.abspath(self.persist_directory).isascii():
            # Below hnsw:sync_threshold (1000) the index lives in SQLite and
            # works; past it the HNSW files go to a garbled path and every
            # later read fails (#617).
            logger.warning(
                "ChromaDB cannot reliably store its index under a non-ASCII path "
                "on Windows (%s). Set semantic_search.persist_directory in "
                "~/.config/zotero-mcp/config.json to an ASCII-only directory.",
                self.persist_directory,
            )

        # Make sure our classes — not ChromaDB's same-named built-ins — answer
        # the registry lookup used when a persisted collection config is
        # rebuilt below (issue #382).
        ensure_embedding_functions_registered()
        self._open()

    @property
    def passage_collection_name(self) -> str:
        return passage_collection_name(self.collection_name)

    def _open(self) -> None:
        """Open the persistent client and the collection in ``persist_directory``."""
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

            self.collection = self._open_collection(self.collection_name)
            self._passage_collection = None
            self._unmigrated: bool | None = None

    def _open_collection(self, name: str):
        """Get or create collection ``name`` with the configured embedding function.

        If the user switched embedding models, the persisted collection will
        have stale config. Detect the mismatch and drop/recreate.
        """
        try:
            collection = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_function
            )

            # ChromaDB may silently persist the old embedding function config.
            # Check if the stored config matches what we want; if not, recreate.
            stored_config = getattr(collection, 'metadata', {}) or {}
            if not stored_config:
                # Try reading config from the collection's config_json_str
                try:
                    import json as _json
                    rows = self.client._sysdb.get_collections(name=name)
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
                            self.client.delete_collection(name=name)
                            collection = self.client.create_collection(
                                name=name,
                                embedding_function=self.embedding_function
                            )
                except Exception:
                    pass  # Best-effort check; proceed with existing collection

        except Exception as e:
            if "embedding function conflict" not in str(e).lower():
                raise
            logger.warning(
                f"Embedding model changed to '{self.embedding_model}'. "
                "Resetting collection for rebuild."
            )
            self.client.delete_collection(name=name)
            collection = self.client.create_collection(
                name=name,
                embedding_function=self.embedding_function
            )
        return collection

    def index_layout(self) -> dict[str, Any]:
        """Record counts per level, read from SQLite without loading a vector index.

        Also picks up a change made by another process: a long-running client
        (the server's worker) drops handles to a collection that was renamed
        or recreated, and re-decides where passages belong.
        """
        layout = read_index_layout(self.persist_directory, self.collection_name)
        article_id = layout.get("article_collection_id")
        if article_id and article_id != str(getattr(self.collection, "id", article_id)):
            with suppress_stdout():
                self.collection = self._open_collection(self.collection_name)
        passage_id = layout.get("passage_collection_id")
        if self._passage_collection is not None and passage_id != str(self._passage_collection.id):
            self._passage_collection = None
        self._unmigrated = layout["unmigrated_passages"] > 0
        return layout

    def separate_passages(self) -> bool:
        """Whether passages live in their own collection in this index.

        False for an index with passages still in the article collection,
        which goes on working that way until it is migrated.
        """
        if not self.route_passages:
            return False
        if self._unmigrated is None:
            self.index_layout()
        return not self._unmigrated

    def _passages(self):
        """The passage collection, created if this is the first use."""
        if self._passage_collection is None:
            with suppress_stdout():
                self._passage_collection = self._open_collection(self.passage_collection_name)
        return self._passage_collection

    def _existing_passages(self):
        """The passage collection if it exists, without creating it."""
        if not self.separate_passages():
            return None
        if self._passage_collection is None and not self._passage_collection_exists():
            return None
        return self._passages()

    def _passage_collection_exists(self) -> bool:
        db = Path(self.persist_directory) / "chroma.sqlite3"
        if not db.exists():
            return False
        con = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True, timeout=5)
        try:
            row = con.execute(
                "SELECT 1 FROM collections WHERE name = ?", (self.passage_collection_name,)
            ).fetchone()
        finally:
            con.close()
        return row is not None

    def collections(self) -> list[Any]:
        """Every collection of this index that exists: articles, then passages."""
        passages = self._existing_passages()
        return [self.collection] if passages is None else [self.collection, passages]

    def _group_existing(self, ids: list[str]) -> list[tuple[Any, list[str]]]:
        """``(collection, ids)`` for reads and deletes, which must not create a collection."""
        passages = self._existing_passages()
        if not self.separate_passages():
            return [(self.collection, ids)] if ids else []
        groups = []
        articles = [doc_id for doc_id in ids if "#" not in doc_id]
        if articles:
            groups.append((self.collection, articles))
        chunks = [doc_id for doc_id in ids if "#" in doc_id]
        if chunks and passages is not None:
            groups.append((passages, chunks))
        return groups

    def _collection_for_level(self, level: str):
        return self._passages() if level == PASSAGES and self.separate_passages() else self.collection

    def _group_by_collection(self, ids: list[str]) -> list[tuple[Any, list[int]]]:
        """``(collection, positions in ids)`` for each collection that ``ids`` touch."""
        if not self.separate_passages():
            return [(self.collection, list(range(len(ids))))] if ids else []
        articles = [i for i, doc_id in enumerate(ids) if "#" not in doc_id]
        passages = [i for i, doc_id in enumerate(ids) if "#" in doc_id]
        groups = []
        if articles:
            groups.append((self.collection, articles))
        if passages:
            groups.append((self._passages(), passages))
        return groups

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
            for collection, positions in self._group_by_collection(ids):
                collection.add(
                    documents=[documents[i] for i in positions],
                    metadatas=[metadatas[i] for i in positions],
                    ids=[ids[i] for i in positions],
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
            for collection, positions in self._group_by_collection(ids):
                for start in range(0, len(positions), max_batch):
                    part = positions[start:start + max_batch]
                    collection.upsert(
                        documents=[documents[i] for i in part],
                        metadatas=[metadatas[i] for i in part],
                        ids=[ids[i] for i in part],
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
            for collection, positions in self._group_by_collection(ids):
                for start in range(0, len(positions), max_batch):
                    part = positions[start:start + max_batch]
                    collection.upsert(
                        documents=[documents[i] for i in part],
                        metadatas=[metadatas[i] for i in part],
                        ids=[ids[i] for i in part],
                        embeddings=[embeddings[i] for i in part],
                    )
            logger.info(f"Upserted {len(documents)} precomputed embeddings to ChromaDB collection")
        except Exception as e:
            logger.error(f"Error upserting precomputed embeddings to ChromaDB: {e}")
            raise

    def search(self,
               query_texts: list[str],
               n_results: int = 10,
               where: dict[str, Any] | None = None,
               where_document: dict[str, Any] | None = None,
               level: str = ARTICLES) -> dict[str, Any]:
        """
        Search for similar documents.

        Args:
            query_texts: List of query texts
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions
            level: Collection to search, ``ARTICLES`` or ``PASSAGES``. Only
                the collection searched has its vector index loaded.

        Returns:
            Search results from ChromaDB
        """
        try:
            query_kwargs = {
                "n_results": n_results,
                "where": where,
                "where_document": where_document,
            }

            # Embed queries ourselves through embed_query_text, so our custom
            # embedding functions apply their query-time tuning (e.g. Gemini's
            # retrieval_query task type). Only our own classes have that
            # method; anything else — ChromaDB's DefaultEmbeddingFunction
            # included — is handed query_texts and left to embed them itself.
            _is_custom_ef = isinstance(self.embedding_function, CUSTOM_EMBEDDING_FUNCTIONS)
            if _is_custom_ef and query_texts:
                query_embeddings = []
                for qt in query_texts:
                    emb = self.embedding_function.embed_query_text(qt)
                    # Ensure plain Python floats (some providers return numpy)
                    if hasattr(emb, 'tolist'):
                        emb = emb.tolist()
                    query_embeddings.append(emb)
                query_kwargs["query_embeddings"] = query_embeddings
            else:
                query_kwargs["query_texts"] = query_texts

            results = self._collection_for_level(level).query(**query_kwargs)
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
            for collection, part in self._group_existing(ids):
                collection.delete(ids=part)
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
            for collection in self.collections():
                collection.delete(where=where)
        except Exception as e:
            logger.warning(f"delete_item_chunks({item_key}) failed: {e}")

    def prune_item_chunks(self, item_key: str, keep: int) -> None:
        """Delete an item's passages from ``chunk_index >= keep`` onward.

        Called after an item's new passages ``<item_key>#0..keep-1`` have been
        upserted, so a document that shrank leaves no orphaned tail, while a
        failed re-embed leaves the old passages searchable instead of none.
        On an index with passages in the article collection, also drops a bare
        ``<item_key>`` document left by a pre-chunking index. Where passages
        have their own collection, that record is the item's article vector.
        """
        where = {"$and": [{"parent_item_key": item_key}, {"chunk_index": {"$gte": int(keep)}}]}
        try:
            for collection in self.collections():
                collection.delete(where=where)
            if not self.separate_passages():
                self.collection.delete(ids=[item_key])
        except Exception as e:
            logger.warning(f"prune_item_chunks({item_key}) failed: {e}")

    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection."""
        try:
            # Counted in SQLite: count() would load each collection's whole
            # vector index just to report a number. Reading the collection
            # afterwards still reports a damaged index (#747).
            self.collection.get(limit=1, include=[])
            layout = self.index_layout()
            return {
                "name": self.collection_name,
                "count": layout["article_records"] + layout["passage_records"],
                "article_records": layout["article_records"],
                "passage_records": layout["passage_records"],
                "unmigrated_passages": layout["unmigrated_passages"],
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
                "error": str(e),
                **unreadable_index_facts(self.persist_directory),
            }

    def read_error(self) -> str | None:
        """Why the collection cannot be read, or None when it can.

        Counting reads the metadata segment, which is the part an interrupted
        write leaves unreadable (#747).
        """
        try:
            for collection in self.collections():
                collection.count()
            return None
        except Exception as e:
            return str(e)

    def move_aside(self) -> str:
        """Rename the index directory to ``<dir>.broken-<timestamp>`` and start an empty one.

        For an index ChromaDB cannot read: deleting it through ChromaDB would
        run into the same error, and a rename keeps the data for ``repair-db``
        or a manual look. Returns the new name of the old directory.
        """
        from datetime import datetime

        self.collection = None
        self.client = None
        release_chroma_clients()
        current = Path(self.persist_directory)
        moved = current.with_name(f"{current.name}.broken-{datetime.now():%Y%m%d-%H%M%S}")
        try:
            os.rename(current, moved)
        except PermissionError as e:  # Windows: ChromaDB or another process still has the files open
            raise RuntimeError(
                f"Could not move the unreadable index at {current} aside ({e}). Stop every Zotero MCP "
                f"server using it and run `zotero-mcp repair-db`, or delete the folder and run update-db."
            ) from e
        self._open()
        return str(moved)

    def reset_collection(self) -> None:
        """Clear the index: the article collection, and the passage collection if any."""
        try:
            if self._passage_collection_exists():
                self.client.delete_collection(name=self.passage_collection_name)
            self._passage_collection = None
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            self._unmigrated = False
            logger.info(f"Reset ChromaDB collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the collection."""
        try:
            return any(
                len(collection.get(ids=part)['ids']) > 0
                for collection, part in self._group_existing([doc_id])
            )
        except Exception:
            return False

    def get_document_metadata(self, doc_id: str) -> dict[str, Any] | None:
        """
        Get metadata for an item if it is indexed.

        An item may be stored only under its chunk ids (``<key>#<n>``), so an
        exact-id lookup on the key alone would miss it. Chunk 0 carries the
        same item-level metadata (``date_modified``, ``has_fulltext``) that
        callers need, so fall back to it, in whichever collection it is.

        Args:
            doc_id: Item key (or full document id) to look up

        Returns:
            Metadata dictionary if the item is indexed, None otherwise
        """
        try:
            for collection, part in self._group_existing([doc_id, f"{doc_id}#0"]):
                result = collection.get(ids=part, include=["metadatas"])
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
            found: set[str] = set()
            for collection, part in self._group_existing(ids):
                # Bounded, for the "too many SQL variables" failure (#368):
                # a run can ask about every passage of a large library.
                for start in range(0, len(part), 5000):
                    result = collection.get(ids=part[start:start + 5000], include=[])
                    found.update(result.get("ids", []))
            return found
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
            found: set[str] = set()
            for collection in self.collections():
                found.update(collection.get(where=where, include=[]).get("ids", []))
            return found
        except Exception as e:
            logger.error(f"Error listing collection ids: {e}")
            return set()

    def count_outside_library(self, group_id: int) -> int:
        """Documents a reset would take from anyone but library ``group_id``.

        Unlike :meth:`get_all_ids` this raises on a read failure, for the guard
        in front of a reset: not knowing must not read as "there are none".
        Counts by subtraction, because unattributed documents never match a
        ``group_id`` filter and they would be lost too. Covers the passage
        collection as well, which a reset deletes with the articles.
        """
        outside = 0
        for collection in self.collections():
            everything = collection.get(include=[])
            ours = collection.get(where={"group_id": int(group_id)}, include=[])
            outside += len(set(everything.get("ids", [])) - set(ours.get("ids", [])))
        return outside

    def has_passage_documents(self) -> bool:
        """Does the index hold passage-chunked documents, in either collection?
        Raises on a read failure."""
        for collection in self.collections():
            found = collection.get(where={"chunk_index": {"$gte": 0}}, limit=1, include=[])
            if found.get("ids"):
                return True
        return False

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
        for collection in self.collections():
            all_ids = sorted(collection.get(include=[]).get("ids") or [])
            for start in range(0, len(all_ids), batch_size):
                chunk = all_ids[start:start + batch_size]
                result = collection.get(ids=chunk, include=["metadatas"])
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
            by_id = dict(zip(ids, metadatas))
            for collection, part in self._group_existing(ids):
                for start in range(0, len(part), max_batch):
                    window = part[start:start + max_batch]
                    collection.update(ids=window, metadatas=[by_id[doc_id] for doc_id in window])
            logger.info(f"Updated metadata for {len(ids)} documents")
        except Exception as e:
            logger.error(f"Error updating document metadata: {e}")
            raise


def create_chroma_client(
    config_path: str | None = None,
    *,
    persist_directory: str | None = None,
) -> ChromaClient:
    """
    Create a ChromaClient instance from configuration.

    Args:
        config_path: Path to configuration file
        persist_directory: Build the client on this directory instead of the
            configured one (``repair-db`` writes its rebuilt index here)

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

    # In the MCP server, ChromaDB runs in a worker process that exits when
    # idle, because the index it loads is never released in-process.
    if persist_directory is None:
        from zotero_mcp import chroma_worker

        idle = config.get("idle_unload_minutes", chroma_worker.DEFAULT_IDLE_UNLOAD_MINUTES)
        worker = chroma_worker.shared_client(config_path, idle)
        if worker is not None:
            return worker

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
        persist_directory=persist_directory or config.get("persist_directory"),
        embedding_model=config["embedding_model"],
        embedding_config=config["embedding_config"]
    )


def release_chroma_clients() -> None:
    """Drop ChromaDB's per-path client cache, closing the files it holds open.

    Needed before an index directory is renamed: a later client for the same
    path would otherwise reuse the cached one, still bound to the old files.
    """
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception as e:  # pragma: no cover - best effort
        logger.debug(f"could not clear the ChromaDB client cache: {e}")


def unreadable_index_facts(persist_directory: str) -> dict[str, Any]:
    """Size and last write of an index ChromaDB failed to read, if one is on disk.

    Empty when there is no ``chroma.sqlite3``: then the failure is not a
    damaged index, and the caller's plain error is the right message.
    """
    db = Path(persist_directory) / "chroma.sqlite3"
    try:
        if db.stat().st_size == 0:
            return {}
        size = sum(f.stat().st_size for f in Path(persist_directory).rglob("*") if f.is_file())
        return {"unreadable": True, "size_bytes": size, "last_written": db.stat().st_mtime}
    except OSError:
        return {}


def unreadable_index_message(info: dict[str, Any], last_update: str | None) -> str:
    """What to tell the user about an index that exists but cannot be read (#747).

    The last write against the last successful update tells an interrupted
    update (written later) from damage of some other kind.
    """
    from datetime import datetime, timedelta

    written = datetime.fromtimestamp(info["last_written"])
    try:
        last = datetime.fromisoformat(last_update) if last_update else None
    except ValueError:
        last = None
    late = last is not None and written - last > timedelta(minutes=5)
    lines = [
        f"The semantic index exists ({info['size_bytes'] / 2**30:.1f} GB) but ChromaDB cannot read it:",
        f"  {info.get('error', 'unknown error')}",
        f"Last successful update: {last:%Y-%m-%d %H:%M}" if last else "Last successful update: never recorded",
        f"Index last written:     {written:%Y-%m-%d %H:%M}"
        + (" (after the last successful update; an interrupted update is the usual cause)" if late else ""),
        f"Index location:         {info.get('persist_directory', '')}",
        "To recover it without re-embedding: zotero-mcp repair-db",
        "To re-embed everything instead:      zotero-mcp update-db --force-rebuild",
    ]
    return "\n".join(lines)


def _level_counts(con: sqlite3.Connection, collection_name: str) -> tuple[str | None, int, int]:
    """(collection id, records with a bare id, records with a ``KEY#n`` id)."""
    row = con.execute("SELECT id FROM collections WHERE name = ?", (collection_name,)).fetchone()
    if row is None:
        return None, 0, 0
    total, chunks = con.execute(
        """
        SELECT count(*), coalesce(sum(e.embedding_id LIKE '%#%'), 0)
        FROM embeddings e JOIN segments s ON e.segment_id = s.id
        WHERE s.collection = ? AND s.scope = 'METADATA'
        """,
        (row[0],),
    ).fetchone()
    return row[0], total - chunks, chunks


def _passage_counts(con: sqlite3.Connection, collection_name: str) -> tuple[str | None, int, int]:
    """(collection id, records, items) of the passage collection, cheaply.

    Runs on every search, so it counts items without scanning a million ids.
    Passage ids are ``KEY#n`` and sorted in ChromaDB's (segment, id) index, so
    an item's passages sit together: each step seeks past ``KEY#`` to the next
    item. That makes one index seek per item, and counts an item whose first
    passage failed to write, which counting ``chunk_index = 0`` would miss.
    """
    row = con.execute("SELECT id FROM collections WHERE name = ?", (collection_name,)).fetchone()
    if row is None:
        return None, 0, 0
    segment = con.execute(
        "SELECT id FROM segments WHERE collection = ? AND scope = 'METADATA'", (row[0],)
    ).fetchone()
    if segment is None:
        return row[0], 0, 0
    records = con.execute("SELECT count(*) FROM embeddings WHERE segment_id = ?", (segment[0],)).fetchone()[0]
    # '$' is the character after '#', so "KEY$" sorts after every "KEY#n".
    items = con.execute(
        """
        WITH RECURSIVE item(id) AS (
            SELECT (SELECT embedding_id FROM embeddings WHERE segment_id = :segment
                    ORDER BY embedding_id LIMIT 1)
            UNION ALL
            SELECT (SELECT embedding_id FROM embeddings
                    WHERE segment_id = :segment AND embedding_id > CASE
                        WHEN instr(item.id, '#') THEN substr(item.id, 1, instr(item.id, '#') - 1)
                        ELSE item.id END || '$'
                    ORDER BY embedding_id LIMIT 1)
            FROM item WHERE item.id IS NOT NULL
        )
        SELECT count(id) FROM item
        """,
        {"segment": segment[0]},
    ).fetchone()[0]
    return row[0], records, items


def read_index_layout(persist_directory: str, collection_name: str) -> dict[str, Any]:
    """Count an index's records per level from SQLite, read-only.

    Never asks ChromaDB: counting through it loads the whole vector index of
    the collection counted, which an article search must not do to passages.

    ``passage_records`` covers both places a passage can be, and
    ``unmigrated_passages`` is how many of them are still in the article
    collection. Those are only "unmigrated" while the passage collection is
    empty; once it holds records the index has been split, so any left behind
    are reported but no longer decide where passages go.
    """
    layout: dict[str, Any] = {
        "article_records": 0,
        "passage_records": 0,
        "passage_items": 0,
        "unmigrated_passages": 0,
        "passages_in_article_collection": 0,
        "passage_collection_exists": False,
        "article_collection_id": None,
        "passage_collection_id": None,
    }
    db = Path(persist_directory) / "chroma.sqlite3"
    if not db.exists():
        return layout
    con = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True, timeout=5)
    try:
        article_id, bare, in_articles = _level_counts(con, collection_name)
        passage_id, in_passages, passage_items = _passage_counts(con, passage_collection_name(collection_name))
    finally:
        con.close()
    layout.update(
        article_records=bare,
        passage_records=in_articles + in_passages,
        passage_items=passage_items,
        unmigrated_passages=0 if in_passages else in_articles,
        passages_in_article_collection=in_articles,
        passage_collection_exists=passage_id is not None,
        article_collection_id=article_id,
        passage_collection_id=passage_id,
    )
    return layout


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

    ``persist_directory`` defaults to the configured location, as
    ``create_chroma_client`` resolves it; it is parameterised for testing.
    """
    collection_name = "zotero_library"
    embedding_model = "default"
    configured_dir = None

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                semantic_cfg = json.load(f).get("semantic_search", {})
            collection_name = semantic_cfg.get("collection_name", collection_name)
            embedding_model = semantic_cfg.get("embedding_model", embedding_model)
            configured_dir = semantic_cfg.get("persist_directory")
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {e}")

    # Mirror create_chroma_client's precedence: config.json wins; the env var
    # only fills in when the file left the model at the "default" placeholder.
    env_model = os.getenv("ZOTERO_EMBEDDING_MODEL")
    if env_model and embedding_model in (None, "default"):
        embedding_model = env_model

    if persist_directory is None:
        persist_directory = str(resolve_chroma_dir(configured_dir))
    base = {
        "name": collection_name,
        "embedding_model": embedding_model,
        "persist_directory": persist_directory,
    }

    # Read the counts from chroma.sqlite3 directly, read-only. Asking ChromaDB
    # for count() loads the collection's whole vector index into this process
    # and keeps it there (gigabytes for a large full-text index), and on a
    # damaged index it writes while retrying its pending log (#747).
    db = Path(persist_directory) / "chroma.sqlite3"
    if not db.exists():
        return {**base, "count": 0, "initialized": False}
    try:
        if problem := sqlite_page_count_problem(db):
            raise sqlite3.DatabaseError(problem)
        con = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True, timeout=5)
        try:
            names = (collection_name, passage_collection_name(collection_name))
            rows = [con.execute("SELECT id FROM collections WHERE name = ?", (n,)).fetchone() for n in names]
            if rows[0] is None:
                return {**base, "count": 0, "initialized": False}
            pending = sum(
                con.execute(
                    """
                    SELECT count(*) FROM embeddings_queue
                    WHERE topic LIKE '%' || :c AND seq_id > coalesce(
                        (SELECT min(m.seq_id) FROM max_seq_id m JOIN segments s ON m.segment_id = s.id
                         WHERE s.collection = :c), 0)
                    """,
                    {"c": row[0]},
                ).fetchone()[0]
                for row in rows
                if row is not None
            )
        finally:
            con.close()
        layout = read_index_layout(persist_directory, collection_name)
        result = {
            **base,
            "count": layout["article_records"] + layout["passage_records"],
            "article_records": layout["article_records"],
            "passage_records": layout["passage_records"],
            "unmigrated_passages": layout["unmigrated_passages"],
            "initialized": True,
        }
        if pending:
            # Writes ChromaDB has logged but not applied yet; it applies them
            # the next time the index is opened.
            result["pending_writes"] = pending
        return result
    except Exception as e:
        logger.error(f"Error reading collection status: {e}")
        return {**base, "count": 0, "error": str(e), **unreadable_index_facts(persist_directory)}


def sqlite_page_count_problem(db: Path) -> str | None:
    """Say so if an SQLite file is longer than its header's page count; None if not.

    SQLite never leaves a healthy rollback-journal database like that. It is
    what a write that grew the file and was cut off before the header was
    updated leaves behind, and SQLite then reports the tables that point past
    the counted pages as "malformed" (#747). The check reads 100 bytes, so it
    costs nothing on a 20 GB index. WAL databases are skipped: their file and
    header lag the log by design.
    """
    if Path(f"{db}-wal").exists():
        return None
    with open(db, "rb") as f:
        header = f.read(100)
    if len(header) < 100 or not header.startswith(b"SQLite format 3\x00"):
        return "not an SQLite database"
    page_size = int.from_bytes(header[16:18], "big")
    page_size = 65536 if page_size == 1 else page_size
    page_count = int.from_bytes(header[28:32], "big")
    trusted = header[24:28] == header[92:96]  # change counter == version-valid-for
    file_pages = db.stat().st_size // page_size
    if trusted and page_count < file_pages:
        return (
            f"the database file holds {file_pages} pages but its header counts {page_count}; "
            "a write was cut off before it finished"
        )
    return None
