"""
Semantic search functionality for Zotero MCP.

This module provides semantic search capabilities by integrating ChromaDB
with the existing Zotero client to enable vector-based similarity search
over research libraries.
"""

import contextlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import tiktoken

    _tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    tiktoken = None
    _tokenizer = None


from . import openai_batch
from .chroma_client import ChromaClient, create_chroma_client
from .client import get_zotero_client
from .local_db import LocalZoteroReader
from .utils import format_creators, is_local_mode, suppress_stdout

logger = logging.getLogger(__name__)


def _pid_is_alive(pid: int) -> bool:
    """Best-effort liveness check for a process id (POSIX ``kill(pid, 0)``)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    except Exception:
        return False
    return True


def read_lock_holder(lock_path: Path) -> tuple[int | None, bool]:
    """Return ``(holder_pid, alive)`` for the update lock, for diagnostics.

    ``holder_pid`` is None when the file is missing/unparseable. ``alive`` says
    whether that pid still exists — a held lock whose holder is dead would be a
    genuinely stale lock (flock releases those automatically on POSIX, so this
    is purely to make the user-facing "skipped" message precise).
    """
    try:
        raw = lock_path.read_text().strip()
        pid = int(raw)
    except Exception:
        return None, False
    return pid, _pid_is_alive(pid)


def _force_update_requested() -> bool:
    """Whether the user asked to bypass the cross-process update lock."""
    return os.getenv("ZOTERO_MCP_FORCE_UPDATE", "").strip().lower() in {"1", "true", "yes"}


@contextlib.contextmanager
def _acquire_update_lock(lock_path: Path):
    """Non-blocking exclusive flock over an update-database run.

    Yields True if the lock was acquired (caller should proceed), False if
    another process already holds it (caller should skip). This prevents the
    MCP server's auto-update in ``server_lifespan`` from racing a manual
    ``zotero-mcp update-db`` invocation on the same ChromaDB collection.

    Setting ``ZOTERO_MCP_FORCE_UPDATE=1`` bypasses the lock entirely — an
    escape hatch for the rare case where a lock appears stuck (e.g. a crashed
    holder on a filesystem with quirky flock semantics) and the user knowingly
    accepts the small double-work risk.

    Windows lacks ``fcntl``; on that platform the function degrades to a
    no-op and yields True so behaviour matches pre-lock releases.
    """
    if _force_update_requested():
        yield True
        return

    try:
        import fcntl
    except ImportError:
        yield True
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        fd = open(lock_path, "w")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        # Record our pid so a concurrent invocation can report the holder.
        try:
            fd.seek(0)
            fd.truncate()
            fd.write(str(os.getpid()))
            fd.flush()
        except Exception:
            pass
        yield True
    finally:
        if fd is not None:
            fd.close()


def _truncate_to_tokens(text: str, max_tokens: int = 8000) -> str:
    """Truncate text to fit within embedding model token limit.

    Uses tiktoken for accurate token counting when available,
    falls back to conservative character-based estimation.
    """
    if _tokenizer is not None:
        tokens = _tokenizer.encode(text, disallowed_special=())
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = _tokenizer.decode(tokens)
    else:
        # Fallback: conservative char limit (~1.5 chars/token for non-Latin scripts)
        max_chars = max_tokens * 2
        if len(text) > max_chars:
            text = text[:max_chars]
    return text


_DEFAULT_UPDATE_CONFIG = {
    "auto_update": False,
    "update_frequency": "manual",
    "last_update": None,
    "update_days": 7,
}


def load_update_config(config_path: str | None) -> dict[str, Any]:
    """Read the semantic-search ``update_config`` block from disk.

    Pure file read with no ChromaDB or embedding-model side effects, so it is
    safe on the read-only status path. Returns defaults when the file is
    missing or unreadable.
    """
    config = dict(_DEFAULT_UPDATE_CONFIG)
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                file_config = json.load(f)
            config.update(file_config.get("semantic_search", {}).get("update_config", {}))
        except Exception as e:
            logger.warning(f"Error loading update config: {e}")
    return config


def should_update(update_config: dict[str, Any]) -> bool:
    """Decide whether an auto-update is due from ``update_config`` alone.

    Pure function of the config dict (and the wall clock) — no I/O, no model
    load — so both :class:`ZoteroSemanticSearch` and the status tool can share
    one source of truth.
    """
    if not update_config.get("auto_update", False):
        return False

    frequency = update_config.get("update_frequency", "manual")

    if frequency == "manual":
        return False
    elif frequency == "startup":
        return True
    elif frequency == "daily":
        last_update = update_config.get("last_update")
        if not last_update:
            return True
        return datetime.now() - datetime.fromisoformat(last_update) >= timedelta(days=1)
    elif frequency.startswith("every_"):
        try:
            days = int(frequency.split("_")[1])
            last_update = update_config.get("last_update")
            if not last_update:
                return True
            return datetime.now() - datetime.fromisoformat(last_update) >= timedelta(days=days)
        except (ValueError, IndexError):
            return False

    return False


# ---------------------------------------------------------------------------
# Passage-level chunking (Tier-1 grounded retrieval)
# ---------------------------------------------------------------------------

# Sentinel separating PDF pages in extracted fulltext, when present. Page-aware
# extractors may insert a form-feed between pages; the chunker uses it to map a
# character offset back to a 1-indexed page. Absent it, only char offsets are
# reported and ``page`` is omitted from passage metadata.
_PAGE_SEPARATOR = "\f"


def split_into_passages(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 200,
    max_chunks: int = 20,
) -> list[tuple[str, int, int]]:
    """Split *text* into overlapping passages on natural boundaries.

    Pure function (no I/O, no model load) so it is unit-testable in isolation.
    Returns a list of ``(passage_text, char_start, char_end)`` tuples with
    character offsets into the original string. Each window targets
    ``chunk_size`` characters but is snapped back to the nearest paragraph or
    sentence boundary in its second half so passages read as coherent quotes.
    Consecutive windows overlap by ``overlap`` characters so a relevant span
    straddling a boundary is still captured whole in one of them. At most
    ``max_chunks`` passages are produced (a guard against pathologically long
    documents inflating the index).
    """
    text = (text or "").strip()
    if not text:
        return []
    if overlap >= chunk_size:
        overlap = chunk_size // 4

    passages: list[tuple[str, int, int]] = []
    start = 0
    n = len(text)
    while start < n and len(passages) < max_chunks:
        end = min(n, start + chunk_size)
        if end < n:
            window = text[start:end]
            for sep in ("\n\n", ". ", ".\n", "\n", " "):
                idx = window.rfind(sep)
                if idx != -1 and idx >= int(chunk_size * 0.5):
                    end = start + idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            passages.append((chunk, start, min(end, n)))
        if end >= n:
            break
        new_start = end - overlap
        # Guarantee forward progress even when overlap is large.
        start = new_start if new_start > start else end
    return passages


def _page_for_offset(text: str, offset: int) -> int | None:
    """Return the 1-indexed page containing *offset*, or None if unknowable.

    Only meaningful when *text* carries ``_PAGE_SEPARATOR`` form-feed page
    breaks (page-aware extraction). Returns None otherwise so callers can omit
    a page field rather than report a misleading one.
    """
    if _PAGE_SEPARATOR not in text:
        return None
    return text.count(_PAGE_SEPARATOR, 0, max(0, offset)) + 1


def best_snippet(query: str, text: str, width: int = 320) -> tuple[str, int]:
    """Return the ``width``-char window of *text* richest in query terms.

    Used to surface a *grounded* quote — the part of a matched document that
    actually overlaps the query — instead of a blind head-truncation. Returns
    ``(snippet, char_start)``. Falls back to the head of the text when no query
    term appears. Pure and dependency-free (lexical overlap only).
    """
    text = text or ""
    if not text.strip():
        return "", 0
    if len(text) <= width:
        return text.strip(), 0
    terms = [t for t in re.findall(r"\w+", (query or "").lower()) if len(t) > 2]
    if not terms:
        return text[:width].strip(), 0
    lowered = text.lower()
    # Score each candidate window anchored at a query-term hit; keep the best.
    best_start = 0
    best_score = -1
    for m in re.finditer(r"\w+", lowered):
        if m.group(0) not in terms:
            continue
        start = max(0, m.start() - width // 3)
        window = lowered[start : start + width]
        score = sum(window.count(t) for t in terms)
        if score > best_score:
            best_score = score
            best_start = start
    snippet = text[best_start : best_start + width].strip()
    return snippet, best_start


class CrossEncoderReranker:
    """Optional cross-encoder re-ranker for semantic search results."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[int]:
        """Re-rank documents by relevance to query.

        Returns indices of top_k documents in descending relevance order.
        """
        return [idx for idx, _ in self.rerank_with_scores(query, documents, top_k)]

    def rerank_with_scores(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        """Re-rank documents, returning ``(index, score)`` pairs.

        Scores are the raw cross-encoder relevance logits, surfaced so search
        results can report *why* an item ranked where it did, not just an
        opaque order.
        """
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in ranked[:top_k]]


class ZoteroSemanticSearch:
    """Semantic search interface for Zotero libraries using ChromaDB."""

    def __init__(
        self, chroma_client: ChromaClient | None = None, config_path: str | None = None, db_path: str | None = None
    ):
        """
        Initialize semantic search.

        Args:
            chroma_client: Optional ChromaClient instance
            config_path: Path to configuration file
            db_path: Optional path to Zotero database (overrides config file)
        """
        self.chroma_client = chroma_client or create_chroma_client(config_path)
        self.zotero_client = get_zotero_client()
        self.config_path = config_path
        self.db_path = db_path  # CLI override for Zotero database path
        # Item keys seen by the most recent local sqlite scan (set by
        # _get_items_from_local_db); used to verify watermark promotion.
        self._last_scan_snapshot_keys: set[str] | None = None

        # Load update configuration
        self.update_config = self._load_update_config()

        # Reranker (lazy-initialized on first search)
        self._reranker: CrossEncoderReranker | None = None
        self._reranker_config = self._load_reranker_config()

        # Passage-level chunking (opt-in; default off preserves item-level
        # indexing and existing collections byte-for-byte).
        self._chunking_config = self._load_chunking_config()

    def _load_chunking_config(self) -> dict[str, Any]:
        """Load passage-chunking configuration from file or use defaults.

        When ``enabled`` is true, each item is indexed as several overlapping
        passages (id ``<item_key>#<n>``) instead of one item-level vector, so
        semantic search returns grounded passage quotes and long PDFs are
        searchable past the single-vector truncation limit. Off by default.
        """
        config: dict[str, Any] = {
            "enabled": False,
            "chunk_size": 1500,
            "overlap": 200,
            "max_chunks_per_item": 20,
        }
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    file_config = json.load(f)
                    config.update(file_config.get("semantic_search", {}).get("chunking", {}))
            except Exception as e:
                logger.warning(f"Error loading chunking config: {e}")
        return config

    @property
    def _chunking_enabled(self) -> bool:
        return bool(self._chunking_config.get("enabled", False))

    def _load_reranker_config(self) -> dict[str, Any]:
        """Load reranker configuration from file or use defaults."""
        config: dict[str, Any] = {
            "enabled": False,
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "candidate_multiplier": 3,
        }
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    file_config = json.load(f)
                    config.update(file_config.get("semantic_search", {}).get("reranker", {}))
            except Exception as e:
                logger.warning(f"Error loading reranker config: {e}")
        return config

    def _get_reranker(self) -> CrossEncoderReranker | None:
        """Get the reranker instance, lazily initializing if enabled."""
        if not self._reranker_config.get("enabled", False):
            return None
        if self._reranker is None:
            model = self._reranker_config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._reranker = CrossEncoderReranker(model_name=model)
        return self._reranker

    def _load_update_config(self) -> dict[str, Any]:
        """Load update configuration from file or use defaults."""
        return load_update_config(self.config_path)

    def _load_include_fulltext_setting(self) -> bool:
        """Whether to fetch fulltext via the Zotero web API during indexing.

        Defaults to True so existing users auto-upgrade to fulltext indexing on
        their next sync. Users can opt out by setting
        `semantic_search.include_fulltext: false` in the config file.
        Local mode (`ZOTERO_LOCAL=true`) keeps using `extract_fulltext` via
        the local sqlite DB; this setting only governs web-API ingestion.
        """
        if not self.config_path or not os.path.exists(self.config_path):
            return True
        try:
            with open(self.config_path) as f:
                file_config = json.load(f)
                value = file_config.get("semantic_search", {}).get("include_fulltext", True)
                return bool(value)
        except Exception as e:
            logger.warning(f"Error loading include_fulltext setting: {e}")
            return True

    def _load_openai_batch_enabled(self) -> bool:
        """Whether OpenAI Batch API indexing is enabled by semantic config."""
        if not self.config_path or not os.path.exists(self.config_path):
            return False
        try:
            with open(self.config_path) as f:
                file_config = json.load(f)
                value = (
                    file_config
                    .get("semantic_search", {})
                    .get("openai_batch", {})
                    .get("enabled", False)
                )
                return bool(value)
        except Exception as e:
            logger.warning(f"Error loading OpenAI batch setting: {e}")
            return False

    def _resolve_openai_batch_enabled(self, use_openai_batch: bool | None) -> bool:
        """Resolve CLI override + config default for OpenAI batch indexing."""
        requested = self._load_openai_batch_enabled() if use_openai_batch is None else use_openai_batch
        return bool(requested and self.chroma_client.embedding_model == "openai")

    def _load_last_sync_version(self) -> int:
        """Last Zotero library version fully indexed into ChromaDB.

        Zero means "no prior successful sync; bootstrap required". Used to
        drive since-based incremental ingest via pyzotero's
        `item_versions(since=V)` and `new_fulltext(since=V)`.
        """
        if not self.config_path or not os.path.exists(self.config_path):
            return 0
        try:
            with open(self.config_path) as f:
                file_config = json.load(f)
                value = file_config.get("semantic_search", {}).get("last_sync_version", 0)
                return int(value) if value is not None else 0
        except Exception as e:
            logger.warning(f"Error loading last_sync_version: {e}")
            return 0

    def _save_update_config(self, last_sync_version: int | None = None) -> None:
        """Save update configuration and optionally update last_sync_version."""
        if not self.config_path:
            return

        config_dir = Path(self.config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)

        # Load existing config or create new one
        full_config = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    full_config = json.load(f)
            except Exception:
                pass

        # Update semantic search config
        if "semantic_search" not in full_config:
            full_config["semantic_search"] = {}

        full_config["semantic_search"]["update_config"] = self.update_config
        if last_sync_version is not None:
            full_config["semantic_search"]["last_sync_version"] = int(last_sync_version)

        try:
            with open(self.config_path, "w") as f:
                json.dump(full_config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving update config: {e}")

    def _create_document_text(self, item: dict[str, Any]) -> str:
        """
        Create searchable text from a Zotero item.

        Args:
            item: Zotero item dictionary

        Returns:
            Combined text for embedding
        """
        data = item.get("data", {})
        item_type = data.get("itemType", "")

        # Annotations have no title / creators / abstract — they have
        # ``annotationText`` (the highlighted passage) and an optional
        # ``annotationComment``. The previous "title + creators + abstract"
        # template fell back to ``format_creators([])`` → "No authors
        # listed", which then embedded identically for every annotation in
        # the library, collapsing them all to a single vector and
        # dominating every semantic-search result (#287).
        if item_type == "annotation":
            return self._create_annotation_document_text(data)

        # Extract key fields for semantic search
        title = data.get("title", "")
        abstract = data.get("abstractNote", "")

        # Format creators as text
        creators = data.get("creators", [])
        creators_text = format_creators(creators)

        # Additional searchable content
        extra_fields = []

        # Publication details
        if publication := data.get("publicationTitle"):
            extra_fields.append(publication)

        # Tags
        if tags := data.get("tags"):
            tag_text = " ".join([tag.get("tag", "") for tag in tags])
            extra_fields.append(tag_text)

        # Note content (if available)
        if note := data.get("note"):
            # Clean HTML from notes
            import re

            note_text = re.sub(r"<[^>]+>", "", note)
            extra_fields.append(note_text)

        # Combine all text fields
        text_parts = [title, creators_text, abstract] + extra_fields
        return " ".join(filter(None, text_parts))

    def _create_annotation_document_text(self, data: dict[str, Any]) -> str:
        """Build the embedding text for an annotation item.

        Combines ``annotationText`` (highlighted passage) and
        ``annotationComment`` (user's commentary), plus any tags. Returns
        the empty string when nothing meaningful is present so the caller
        can decide to skip the item rather than embedding noise.
        """
        parts: list[str] = []
        if highlighted := (data.get("annotationText") or "").strip():
            parts.append(highlighted)
        if comment := (data.get("annotationComment") or "").strip():
            parts.append(comment)
        if tags := data.get("tags"):
            tag_text = " ".join(t.get("tag", "") for t in tags if t.get("tag"))
            if tag_text:
                parts.append(tag_text)
        return " ".join(parts)

    def _create_metadata(self, item: dict[str, Any]) -> dict[str, Any]:
        """
        Create metadata for a Zotero item.

        Args:
            item: Zotero item dictionary

        Returns:
            Metadata dictionary for ChromaDB
        """
        data = item.get("data", {})

        metadata = {
            "item_key": item.get("key", ""),
            "item_type": data.get("itemType", ""),
            "title": data.get("title", ""),
            "date": data.get("date", ""),
            "date_added": data.get("dateAdded", ""),
            "date_modified": data.get("dateModified", ""),
            "creators": format_creators(data.get("creators", [])),
            "publication": data.get("publicationTitle", ""),
            "url": data.get("url", ""),
            "doi": data.get("DOI", ""),
        }
        # If fulltext was extracted (or attempted), mark it so incremental
        # updates don't keep re-trying items that failed extraction
        if data.get("fulltext"):
            metadata["has_fulltext"] = True
            if data.get("fulltextSource"):
                metadata["fulltext_source"] = data.get("fulltextSource")
        elif data.get("fulltext_attempted"):
            # Extraction was attempted but failed (timeout, empty, etc.)
            # Mark so we don't retry on every incremental update
            metadata["has_fulltext"] = "failed"

        # Add tags as a single string
        if tags := data.get("tags"):
            metadata["tags"] = " ".join([tag.get("tag", "") for tag in tags])
        else:
            metadata["tags"] = ""

        # Add citation key if available
        extra = data.get("extra", "")
        citation_key = ""
        for line in extra.split("\n"):
            if line.lower().startswith(("citation key:", "citationkey:")):
                citation_key = line.split(":", 1)[1].strip()
                break
        metadata["citation_key"] = citation_key

        return metadata

    def should_update_database(self) -> bool:
        """Check if the database should be updated based on configuration."""
        return should_update(self.update_config)

    def _get_items_from_source(
        self,
        limit: int | None = None,
        extract_fulltext: bool = False,
        chroma_client: ChromaClient | None = None,
        force_rebuild: bool = False,
        include_fulltext_via_api: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Get items from either local database or API.

        When extract_fulltext=True, requires local mode (ZOTERO_LOCAL=true);
        raises RuntimeError if local mode is not enabled. This path reads the
        local Zotero sqlite database and extracts PDF text on-disk.

        When include_fulltext_via_api=True (web-API mode), fetches the
        server-side extracted fulltext that Zotero cloud has already built
        for each PDF — no local files required.

        Otherwise uses API metadata only (fastest, title/abstract/tags).

        Args:
            limit: Optional limit on number of items
            extract_fulltext: Whether to extract fulltext from the local sqlite DB
            chroma_client: ChromaDB client to check for existing documents (None to skip checks)
            force_rebuild: Whether to force extraction even if item exists
            include_fulltext_via_api: Fetch fulltext via the Zotero web API

        Returns:
            List of items in API-compatible format
        """
        if extract_fulltext:
            if not is_local_mode():
                raise RuntimeError(
                    "Fulltext extraction requires local mode but ZOTERO_LOCAL is not enabled. "
                    "Set ZOTERO_LOCAL=true or run 'zotero-mcp setup' to enable local mode."
                )
            return self._get_items_from_local_db(
                limit, extract_fulltext=extract_fulltext, chroma_client=chroma_client, force_rebuild=force_rebuild
            )
        else:
            return self._get_items_from_api(limit, include_fulltext=include_fulltext_via_api)

    def _get_items_from_local_db(
        self,
        limit: int | None = None,
        extract_fulltext: bool = False,
        chroma_client: ChromaClient | None = None,
        force_rebuild: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Get items from local Zotero database.

        Args:
            limit: Optional limit on number of items
            extract_fulltext: Whether to extract fulltext content
            chroma_client: ChromaDB client to check for existing documents (None to skip checks)
            force_rebuild: Whether to force extraction even if item exists

        Returns:
            List of items in API-compatible format
        """
        logger.info("Fetching items from local Zotero database...")

        try:
            # Load per-run config, including extraction limits and db path if provided
            pdf_max_pages = None
            pdf_timeout = 30
            zotero_db_path = self.db_path  # CLI override takes precedence
            # If semantic_search config file exists, prefer its setting
            try:
                if self.config_path and os.path.exists(self.config_path):
                    with open(self.config_path) as _f:
                        _cfg = json.load(_f)
                        semantic_cfg = _cfg.get("semantic_search", {})
                        extraction_cfg = semantic_cfg.get("extraction", {})
                        pdf_max_pages = extraction_cfg.get("pdf_max_pages")
                        pdf_timeout = extraction_cfg.get("pdf_timeout", 30)
                        # Use config db_path only if no CLI override
                        if not zotero_db_path:
                            zotero_db_path = semantic_cfg.get("zotero_db_path")
            except Exception:
                pass

            with (
                suppress_stdout(),
                LocalZoteroReader(
                    db_path=zotero_db_path, pdf_max_pages=pdf_max_pages, pdf_timeout=pdf_timeout
                ) as reader,
            ):
                # Capture the snapshot's full key set on the SAME connection
                # this scan uses. The staleness check after the (potentially
                # long) extraction must compare against what this scan could
                # actually see — a fresh read taken later could already
                # include rows from a WAL checkpoint that landed mid-scan.
                self._last_scan_snapshot_keys = reader.get_all_item_keys()
                # Phase 1: fetch metadata only (fast)
                sys.stderr.write("Scanning local Zotero database for items...\n")
                local_items = reader.get_items_with_text(limit=limit, include_fulltext=False)
                candidate_count = len(local_items)
                sys.stderr.write(f"Found {candidate_count} candidate items.\n")

                # Optional deduplication: if preprint and journalArticle share a DOI/title, keep journalArticle
                # Build index by (normalized DOI or normalized title)
                def norm(s: str | None) -> str | None:
                    if not s:
                        return None
                    return "".join(s.lower().split())

                key_to_best = {}
                for it in local_items:
                    doi_key = ("doi", norm(getattr(it, "doi", None))) if getattr(it, "doi", None) else None
                    title_key = ("title", norm(getattr(it, "title", None))) if getattr(it, "title", None) else None

                    def consider(k):
                        if not k:
                            return
                        cur = key_to_best.get(k)
                        # Prefer journalArticle over preprint; otherwise keep first
                        if cur is None:
                            key_to_best[k] = it
                        else:
                            prefer_types = {"journalArticle": 2, "preprint": 1}
                            cur_score = prefer_types.get(getattr(cur, "item_type", ""), 0)
                            new_score = prefer_types.get(getattr(it, "item_type", ""), 0)
                            if new_score > cur_score:
                                key_to_best[k] = it

                    consider(doi_key)
                    consider(title_key)

                # If a preprint loses against a journal article for same DOI/title, drop it
                filtered_items = []
                for it in local_items:
                    # If there is a journalArticle alternative for same DOI or title, and this is preprint, drop
                    if getattr(it, "item_type", None) == "preprint":
                        k_doi = ("doi", norm(getattr(it, "doi", None))) if getattr(it, "doi", None) else None
                        k_title = ("title", norm(getattr(it, "title", None))) if getattr(it, "title", None) else None
                        drop = False
                        for k in (k_doi, k_title):
                            if not k:
                                continue
                            best = key_to_best.get(k)
                            if (
                                best is not None
                                and best is not it
                                and getattr(best, "item_type", None) == "journalArticle"
                            ):
                                drop = True
                                break
                        if drop:
                            continue
                    filtered_items.append(it)

                local_items = filtered_items
                total_to_extract = len(local_items)
                if total_to_extract != candidate_count:
                    try:
                        sys.stderr.write(
                            f"After filtering/dedup: {total_to_extract} items to process. Extracting content...\n"
                        )
                    except Exception:
                        pass
                else:
                    try:
                        sys.stderr.write("Extracting content...\n")
                    except Exception:
                        pass

                # Phase 2: selectively extract fulltext only when requested
                if extract_fulltext:
                    extracted = 0
                    skipped_existing = 0
                    updated_existing = 0
                    items_to_process = []

                    consecutive_timeouts = 0
                    MAX_CONSECUTIVE_TIMEOUTS = 5
                    _extraction_stopped = False  # Set True when circuit breaker trips

                    total_local = len(local_items)
                    _skipped_pdfs = []  # Collect timeout/error names for summary
                    _skipped_failed = []  # Items skipped because extraction previously failed

                    # Show startup note
                    try:
                        sys.stderr.write(
                            "\n  Note: Most papers take 1-3 seconds. Some larger or complex PDFs\n"
                            "  may take up to 30 seconds. Password-protected or corrupted files\n"
                            "  will be skipped automatically. The system moves on to the next\n"
                            "  paper if a file can't be processed in time.\n\n"
                        )
                        sys.stderr.flush()
                    except Exception:
                        pass

                    # Temporarily suppress local_db logger to prevent timeout warnings
                    # from disrupting the progress line — we collect them ourselves
                    _local_db_logger = logging.getLogger("zotero_mcp.local_db")
                    _prev_level = _local_db_logger.level
                    _local_db_logger.setLevel(logging.CRITICAL)

                    for item_idx, it in enumerate(local_items, 1):
                        # Build display string: Author (Year) — Title
                        title = getattr(it, "title", "") or ""
                        creators = getattr(it, "creators", "") or ""
                        date = getattr(it, "date_added", "") or ""
                        first_author = ""
                        if creators:
                            first_author = creators.split(";")[0].split(",")[0].strip()
                            if first_author:
                                first_author += " et al." if ";" in creators else ""
                        year = ""
                        if date and len(date) >= 4:
                            year = date[:4]
                        citation = ""
                        if first_author and year:
                            citation = f"{first_author} ({year}) — "
                        elif first_author:
                            citation = f"{first_author} — "
                        display = f"{citation}{title}"
                        if len(display) > 60:
                            display = display[:57] + "..."

                        # Single-line progress with \r overwrite
                        # MUST fit within terminal width to prevent wrapping
                        try:
                            try:
                                term_width = os.get_terminal_size().columns
                            except (OSError, ValueError):
                                term_width = 80
                            # Build the line and truncate to terminal width - 1
                            # (- 1 to prevent the cursor from wrapping to next line)
                            max_len = term_width - 1
                            status_parts = []
                            if skipped_existing > 0:
                                status_parts.append(f"{skipped_existing} up to date")
                            if extracted > 0:
                                status_parts.append(f"{extracted} extracted")
                            status = f" ({', '.join(status_parts)})" if status_parts else ""
                            prefix = f"  Processing {item_idx}/{total_local}{status} — "
                            # Truncate display to fit remaining space
                            remaining = max_len - len(prefix) - 3  # -3 for "..."
                            if remaining > 0 and display and len(display) > remaining:
                                display = display[:remaining] + "..."
                            line = f"{prefix}{display or 'working...'}"
                            if len(line) > max_len:
                                line = line[:max_len]
                            sys.stderr.write(f"\r{line}{' ' * max(0, max_len - len(line))}")
                            sys.stderr.flush()
                        except Exception:
                            pass

                        should_extract = True

                        # CHECK IF ITEM ALREADY EXISTS (unless force_rebuild or no client)
                        if chroma_client and not force_rebuild:
                            existing_metadata = chroma_client.get_document_metadata(it.key)
                            if existing_metadata:
                                chroma_has_fulltext = existing_metadata.get("has_fulltext", False)
                                local_has_fulltext = len(reader.get_fulltext_meta_for_item(it.item_id)) > 0

                                # Skip if extraction previously failed AND the item hasn't been
                                # modified since (handles case where user replaces a bad PDF)
                                if chroma_has_fulltext == "failed":
                                    chroma_date = existing_metadata.get("date_modified", "")
                                    item_date = getattr(it, "date_modified", "") or ""
                                    if chroma_date == item_date:
                                        # Same modification date — don't retry failed extraction
                                        should_extract = False
                                        skipped_existing += 1
                                        _skipped_failed.append(display or f"item {it.key}")
                                    else:
                                        # Item was modified since last failure — retry
                                        updated_existing += 1
                                elif not chroma_has_fulltext and local_has_fulltext:
                                    # Document exists but lacks fulltext - we need to update it
                                    updated_existing += 1
                                else:
                                    should_extract = False
                                    skipped_existing += 1

                        if should_extract:
                            # Extract fulltext if item doesn't have it yet
                            # (skip if circuit breaker has tripped)
                            if not getattr(it, "fulltext", None) and not _extraction_stopped:
                                text = reader.extract_fulltext_for_item(it.item_id)
                                # Circuit breaker: stop PDF extraction after consecutive timeouts
                                if isinstance(text, tuple) and len(text) == 2 and text[1] == "timeout":
                                    _skipped_pdfs.append(display or f"item {it.key}")
                                    consecutive_timeouts += 1
                                    if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                                        logger.warning(
                                            f"Stopping PDF extraction after {MAX_CONSECUTIVE_TIMEOUTS} "
                                            f"consecutive timeouts — remaining items will use metadata only"
                                        )
                                        try:
                                            sys.stderr.write(
                                                f"\n  Warning: PDF extraction stopped after {MAX_CONSECUTIVE_TIMEOUTS} "
                                                f"consecutive timeouts. Remaining items will be indexed with "
                                                f"metadata only (titles, abstracts, authors).\n\n"
                                            )
                                        except Exception:
                                            pass
                                        _extraction_stopped = True
                                    # Don't skip the item — still add it with metadata only
                                    it._fulltext_attempted = True  # Mark so metadata knows extraction was tried
                                else:
                                    # Reset counter on successful extraction
                                    if text:
                                        consecutive_timeouts = 0
                                    if text:
                                        # Support new (text, source) return format
                                        if isinstance(text, tuple) and len(text) == 2:
                                            it.fulltext, it.fulltext_source = text[0], text[1]
                                        else:
                                            it.fulltext = text
                                    else:
                                        # Extraction returned empty — mark as attempted
                                        it._fulltext_attempted = True
                            extracted += 1
                            items_to_process.append(it)

                            # (progress shown inline above via \r)

                    # Restore local_db logger
                    _local_db_logger.setLevel(_prev_level)

                    # Clear progress line and show extraction summary
                    try:
                        sys.stderr.write(f"\r{' ' * 120}\r")  # Clear progress line
                        parts = [f"  Extraction complete: {extracted} items to index"]
                        if skipped_existing > 0:
                            parts.append(f"{skipped_existing} already up to date")
                        sys.stderr.write(", ".join(parts) + "\n")
                        if updated_existing > 0:
                            sys.stderr.write(f"  ({updated_existing} items updated with new fulltext)\n")
                        if _skipped_pdfs:
                            sys.stderr.write(f"  Skipped {len(_skipped_pdfs)} PDF(s) (timed out):\n")
                            for name in _skipped_pdfs:
                                sys.stderr.write(f"    - {name}\n")
                        if _skipped_failed:
                            sys.stderr.write(
                                f"  {len(_skipped_failed)} item(s) skipped (PDF extraction previously failed):\n"
                            )
                            for name in _skipped_failed[:5]:  # Show first 5
                                sys.stderr.write(f"    - {name}\n")
                            if len(_skipped_failed) > 5:
                                sys.stderr.write(f"    ... and {len(_skipped_failed) - 5} more\n")
                            sys.stderr.write("  (To retry these, run with --force-rebuild)\n")
                    except Exception:
                        pass

                    # Replace local_items with filtered list
                    local_items = items_to_process
                else:
                    # Skip fulltext extraction for faster processing
                    for it in local_items:
                        it.fulltext = None
                        it.fulltext_source = None

                # Convert to API-compatible format
                api_items = []
                for item in local_items:
                    # Create API-compatible item structure
                    api_item = {
                        "key": item.key,
                        "version": 0,  # Local items don't have versions
                        "data": {
                            "key": item.key,
                            "itemType": getattr(item, "item_type", None) or "journalArticle",
                            "title": item.title or "",
                            "abstractNote": item.abstract or "",
                            "extra": item.extra or "",
                            # Include fulltext only when extracted
                            "fulltext": getattr(item, "fulltext", None) or "" if extract_fulltext else "",
                            "fulltextSource": getattr(item, "fulltext_source", None) or "" if extract_fulltext else "",
                            # Flag if extraction was attempted but failed (timeout, empty)
                            "fulltext_attempted": getattr(item, "_fulltext_attempted", False),
                            "dateAdded": item.date_added,
                            "dateModified": item.date_modified,
                            "creators": self._parse_creators_string(item.creators) if item.creators else [],
                        },
                    }

                    # Add notes if available
                    if item.notes:
                        api_item["data"]["notes"] = item.notes

                    api_items.append(api_item)

                logger.info(f"Retrieved {len(api_items)} items from local database")
                return api_items

        except Exception as e:
            logger.error(f"Error reading from local database: {e}")
            logger.info("Falling back to API...")
            return self._get_items_from_api(limit)

    def _parse_creators_string(self, creators_str: str) -> list[dict[str, str]]:
        """
        Parse creators string from local DB into API format.

        Args:
            creators_str: String like "Smith, John; Doe, Jane"

        Returns:
            List of creator objects
        """
        if not creators_str:
            return []

        creators = []
        for creator in creators_str.split(";"):
            creator = creator.strip()
            if not creator:
                continue

            if "," in creator:
                last, first = creator.split(",", 1)
                creators.append({"creatorType": "author", "firstName": first.strip(), "lastName": last.strip()})
            else:
                creators.append({"creatorType": "author", "name": creator})

        return creators

    def _fetch_fulltext_via_web_api(self, item_key: str) -> tuple[str, str]:
        """Fetch fulltext for a top-level item via the Zotero web API.

        Zotero's cloud keeps a server-side extracted text for every PDF that
        the desktop client has ever indexed. Web-API mode can retrieve that
        text without needing the PDF file to be present locally.

        The fulltext usually lives on the PDF attachment child, not the
        parent. We first try the parent's own key (covers the case where the
        parent is itself an attachment), then cascade through PDF attachment
        children.

        Returns:
            (text, source) where source describes which endpoint supplied the
            text (e.g. "web-api:parent", "web-api:attachment:<key>"). Empty
            strings mean no fulltext is available for this item.
        """

        def _extract_content(resp: Any) -> str:
            if isinstance(resp, dict):
                return str(resp.get("content", "") or "")
            if isinstance(resp, str):
                return resp
            return ""

        # 1. Try the item itself (works when item_key IS the attachment key).
        try:
            resp = self.zotero_client.fulltext_item(item_key)
            text = _extract_content(resp)
            if text.strip():
                return text, "web-api:parent"
        except Exception as e:
            logger.debug(f"fulltext_item({item_key}) failed: {e}")

        # 2. Walk PDF attachment children and try each in order.
        try:
            children = self.zotero_client.children(item_key) or []
        except Exception as e:
            logger.debug(f"children({item_key}) failed: {e}")
            children = []

        for child in children:
            data = child.get("data", {}) if isinstance(child, dict) else {}
            if data.get("itemType") != "attachment":
                continue
            if data.get("contentType") != "application/pdf":
                continue
            child_key = child.get("key") or data.get("key")
            if not child_key:
                continue
            try:
                resp = self.zotero_client.fulltext_item(child_key)
            except Exception as e:
                logger.debug(f"fulltext_item({child_key}) failed: {e}")
                continue
            text = _extract_content(resp)
            if text.strip():
                return text, f"web-api:attachment:{child_key}"

        return "", ""

    def _attach_web_fulltext(self, items: list[dict[str, Any]]) -> None:
        """Populate `data.fulltext` on each item in place using the web API."""
        total = len(items)
        if not total:
            return
        try:
            sys.stderr.write(f"\nFetching fulltext for {total} items via web API...\n")
            sys.stderr.flush()
        except Exception:
            pass
        fetched = 0
        for idx, item in enumerate(items, 1):
            key = item.get("key", "")
            data = item.setdefault("data", {})
            # Skip items that obviously can't have fulltext
            if data.get("itemType") in {"note", "annotation"}:
                data["fulltext_attempted"] = True
                continue
            if not key:
                continue
            text, source = self._fetch_fulltext_via_web_api(key)
            if text:
                data["fulltext"] = text
                data["fulltextSource"] = source
                fetched += 1
            else:
                data["fulltext_attempted"] = True
            if idx % 25 == 0 or idx == total:
                try:
                    sys.stderr.write(f"\r  Fulltext: {idx}/{total} items checked, {fetched} with text")
                    sys.stderr.flush()
                except Exception:
                    pass
        try:
            sys.stderr.write("\n")
        except Exception:
            pass

    def _get_items_from_api(self, limit: int | None = None, include_fulltext: bool = False) -> list[dict[str, Any]]:
        """
        Get items from Zotero API (original implementation).

        Args:
            limit: Optional limit on number of items
            include_fulltext: If True, fetch server-side extracted PDF text
                via pyzotero's fulltext_item endpoint for each returned
                top-level item. Enables full-text semantic indexing without
                requiring local Zotero mode.

        Returns:
            List of items from API
        """
        logger.info("Fetching items from Zotero API...")

        # Fetch items in batches to handle large libraries
        batch_size = 100
        start = 0
        all_items = []

        while True:
            batch_params = {"start": start, "limit": batch_size}
            if limit and len(all_items) >= limit:
                break

            try:
                items = self.zotero_client.items(**batch_params)
            except Exception as e:
                if "Connection refused" in str(e):
                    error_msg = (
                        "Cannot connect to Zotero local API. Please ensure:\n"
                        "1. Zotero is running\n"
                        "2. Local API is enabled in Zotero Preferences > Advanced > Enable HTTP server\n"
                        "3. The local API port (default 23119) is not blocked"
                    )
                    raise Exception(error_msg) from e
                else:
                    raise Exception(f"Zotero API connection error: {e}") from e
            if not items:
                break

            # Filter out attachments and notes by default
            filtered_items = [
                item for item in items if item.get("data", {}).get("itemType") not in ["attachment", "note"]
            ]

            all_items.extend(filtered_items)
            start += batch_size

            if len(items) < batch_size:
                break

        if limit:
            all_items = all_items[:limit]

        if include_fulltext:
            self._attach_web_fulltext(all_items)

        logger.info(f"Retrieved {len(all_items)} items from API")
        return all_items

    def _get_changed_items_from_api(
        self, since_version: int, include_fulltext: bool = False
    ) -> tuple[list[dict[str, Any]], set[str]]:
        """Fetch only items changed in the Zotero library since a given version.

        Uses pyzotero's `item_versions(since=V)` to discover changed top-level
        item keys, then fetches their full payloads one at a time. When
        `include_fulltext` is True, also fetches server-side extracted text
        for each changed item.

        Returns:
            (changed_items, all_current_top_level_keys). The second element
            powers deletion detection: any id present in the ChromaDB
            collection but absent from it has been removed from the library.
        """
        logger.info(f"Fetching changed items since library version {since_version}...")
        try:
            changed_versions = self.zotero_client.item_versions(since=since_version) or {}
        except Exception as e:
            raise Exception(f"Failed to fetch item_versions(since={since_version}): {e}") from e

        try:
            current_versions = self.zotero_client.item_versions() or {}
        except Exception as e:
            logger.warning(f"Failed to fetch current item_versions for deletion check: {e}")
            current_versions = {}
        current_keys = set(current_versions.keys())

        if not changed_versions:
            return [], current_keys

        changed_items: list[dict[str, Any]] = []
        for key in changed_versions.keys():
            try:
                item = self.zotero_client.item(key)
            except Exception as e:
                logger.debug(f"item({key}) failed during incremental fetch: {e}")
                continue
            if not item:
                continue
            item_type = item.get("data", {}).get("itemType")
            # Don't index attachments/notes as standalone entries; only
            # top-level research items participate in semantic search.
            if item_type in {"attachment", "note", "annotation"}:
                continue
            changed_items.append(item)

        if include_fulltext and changed_items:
            self._attach_web_fulltext(changed_items)

        return changed_items, current_keys

    def _verify_local_snapshot_version(self, target_sync_version: int) -> int | None:
        """Decide whether the local sqlite snapshot supports promoting the
        API-derived sync watermark.

        The local-extraction scan reads zotero.sqlite with `immutable=1`,
        which cannot see rows still sitting in an un-checkpointed WAL file.
        The API (served by the running Zotero) *does* see them, so its
        library version may cover items the scan never returned. Promoting
        `last_sync_version` in that state makes every later incremental
        update skip those items forever (issue #292).

        Returns:
            `target_sync_version` if every item key known to the API is
            present in the sqlite snapshot, otherwise None (keep the
            previous watermark so the next update re-covers the gap).
        """
        try:
            api_keys = set((self.zotero_client.item_versions() or {}).keys())

            # Prefer the key set captured by the scan's own connection: a
            # fresh read here could already see rows from a WAL checkpoint
            # that landed mid-scan, masking the very staleness we check for.
            snapshot_keys = getattr(self, "_last_scan_snapshot_keys", None)
            if snapshot_keys is None:
                zotero_db_path = self.db_path  # CLI override takes precedence
                if not zotero_db_path and self.config_path and os.path.exists(self.config_path):
                    try:
                        with open(self.config_path) as f:
                            zotero_db_path = (
                                json.load(f).get("semantic_search", {}).get("zotero_db_path")
                            )
                    except Exception:
                        pass
                with LocalZoteroReader(db_path=zotero_db_path) as reader:
                    snapshot_keys = reader.get_all_item_keys()
        except Exception as e:
            logger.warning(
                f"Could not verify local snapshot completeness ({e}); "
                "keeping previous sync watermark."
            )
            return None

        missing = api_keys - snapshot_keys
        if missing:
            logger.warning(
                f"{len(missing)} item(s) are visible via the Zotero API but "
                "missing from the local sqlite snapshot (immutable reads "
                "cannot see un-checkpointed WAL data); keeping previous sync "
                "watermark so the next update can pick them up."
            )
            return None
        return target_sync_version

    def _prepare_index_records(self, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Prepare ChromaDB records without embedding or writing them."""
        stats = {"processed": 0, "skipped": 0, "errors": 0}
        records: list[dict[str, Any]] = []

        for item in items:
            try:
                item_key = item.get("key", "")
                if not item_key:
                    stats["skipped"] += 1
                    continue

                fulltext = item.get("data", {}).get("fulltext", "")
                structured_text = self._create_document_text(item)
                if fulltext.strip():
                    doc_text = (structured_text + "\n\n" + fulltext) if structured_text.strip() else fulltext
                else:
                    doc_text = structured_text
                metadata = self._create_metadata(item)

                if not doc_text.strip():
                    stats["skipped"] += 1
                    continue

                doc_text = self.chroma_client.truncate_text(doc_text)
                records.append({"id": item_key, "document": doc_text, "metadata": metadata})
                stats["processed"] += 1

            except Exception as e:
                logger.error(f"Error processing item {item.get('key', 'unknown')}: {e}")
                stats["errors"] += 1

        return records, stats

    def _submit_openai_batch_index(
        self,
        items: list[dict[str, Any]],
        force_full_rebuild: bool,
        target_sync_version: int | None,
        stats: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepare records and submit asynchronous OpenAI embedding batches."""
        records, prepare_stats = self._prepare_index_records(items)
        stats["processed_items"] += prepare_stats["processed"]
        stats["skipped_items"] += prepare_stats["skipped"]
        stats["errors"] += prepare_stats["errors"]

        if not records:
            stats["batch_submitted"] = False
            stats["batch_error"] = "No documents were prepared for OpenAI Batch API submission"
            return stats

        ids = [record["id"] for record in records]
        existing_ids = self.chroma_client.get_existing_ids(ids) if ids and not force_full_rebuild else set()
        model_name = self.chroma_client.embedding_config.get("model_name", "text-embedding-3-small")
        manifest = openai_batch.submit_embedding_batches(
            records=records,
            model_name=model_name,
            embedding_config=self.chroma_client.embedding_config,
            config_path=self.config_path,
            force_full_rebuild=force_full_rebuild,
            target_sync_version=target_sync_version,
        )
        stats["batch_submitted"] = True
        stats["batch_run_id"] = manifest["run_id"]
        stats["batch_manifest"] = manifest["manifest_path"]
        stats["batch_ids"] = [batch["batch_id"] for batch in manifest.get("batches", [])]
        stats["submitted_items"] = len(records)
        stats["estimated_updated_items"] = len(existing_ids)
        stats["estimated_added_items"] = len(ids) - len(existing_ids)
        return stats

    def update_database(
        self,
        force_full_rebuild: bool = False,
        limit: int | None = None,
        extract_fulltext: bool = False,
        include_fulltext: bool | None = None,
        use_openai_batch: bool | None = None,
    ) -> dict[str, Any]:
        """
        Update the semantic search database with Zotero items.

        Args:
            force_full_rebuild: Whether to rebuild the entire database
            limit: Limit number of items to process (for testing)
            extract_fulltext: Whether to extract fulltext content from the
                local Zotero sqlite database (requires ZOTERO_LOCAL=true)
            include_fulltext: Whether to fetch server-side extracted
                fulltext via the Zotero web API. Defaults to the
                `semantic_search.include_fulltext` config setting (True
                unless explicitly disabled). Ignored in local mode since
                `extract_fulltext` provides richer local extraction.
            use_openai_batch: Override for OpenAI Batch API indexing. None
                uses `semantic_search.openai_batch.enabled`.

        Returns:
            Update statistics
        """
        logger.info("Starting database update...")
        start_time = datetime.now()

        stats = {
            "total_items": 0,
            "processed_items": 0,
            "added_items": 0,
            "updated_items": 0,
            "recovered_items": 0,
            "skipped_items": 0,
            "deleted_items": 0,
            "errors": 0,
            "start_time": start_time.isoformat(),
            "duration": None,
        }

        # Guard against concurrent rebuilds: the MCP server auto-launches
        # update_database on startup while the user may also run
        # `zotero-mcp update-db` manually. A cross-process flock avoids
        # double work and potential ChromaDB corruption.
        lock_path = Path.home() / ".config" / "zotero-mcp" / "update.lock"
        lock_cm = _acquire_update_lock(lock_path)
        acquired = lock_cm.__enter__()
        if not acquired:
            lock_cm.__exit__(None, None, None)
            holder_pid, holder_alive = read_lock_holder(lock_path)
            if holder_pid and not holder_alive:
                logger.warning(
                    "Update lock at %s is held by dead pid %s (stale). "
                    "flock should have released it; set ZOTERO_MCP_FORCE_UPDATE=1 "
                    "to bypass if this persists.",
                    lock_path,
                    holder_pid,
                )
            else:
                logger.warning(
                    "Another semantic-search update is already running "
                    "(lock held at %s by pid %s); skipping this invocation. "
                    "This is expected when the MCP server's background sync is "
                    "active. Set ZOTERO_MCP_FORCE_UPDATE=1 to override.",
                    lock_path,
                    holder_pid if holder_pid else "unknown",
                )
            stats["duration"] = "0:00:00"
            stats["skipped_reason"] = "another_update_in_progress"
            return stats

        try:
            # Resolve include_fulltext default from config if not specified
            if include_fulltext is None:
                include_fulltext = self._load_include_fulltext_setting()

            # Web-API fulltext only applies when not using the local sqlite
            # extractor (extract_fulltext=True takes precedence in local mode)
            include_fulltext_via_api = include_fulltext and not extract_fulltext
            use_openai_batch = self._resolve_openai_batch_enabled(use_openai_batch)

            # In batch mode, defer destructive rebuilds until import so the
            # existing search index remains usable while the batch runs.
            if force_full_rebuild and not use_openai_batch:
                logger.info("Force rebuilding database...")
                self.chroma_client.reset_collection()

            # Decide whether to use since-based incremental ingest.
            # Incremental requires: not a forced rebuild, not a local-extraction
            # run (incremental path covers web-API metadata and optionally
            # fulltext only), not a test limit, and a known prior sync version.
            last_sync_version = self._load_last_sync_version() if not force_full_rebuild else 0
            use_incremental = (
                not force_full_rebuild and not extract_fulltext and limit is None and last_sync_version > 0
            )

            target_sync_version: int | None = None
            all_items: list[dict[str, Any]] = []
            if use_incremental:
                try:
                    target_sync_version = self.zotero_client.last_modified_version()
                except Exception as e:
                    logger.warning(f"last_modified_version() failed, falling back to full scan: {e}")
                    use_incremental = False

            if use_incremental and target_sync_version == last_sync_version:
                # No changes since last sync; skip ingest but still touch last_update
                try:
                    sys.stderr.write(
                        f"\nLibrary unchanged since last sync (version {last_sync_version}); no items to reindex.\n"
                    )
                except Exception:
                    pass
                self.update_config["last_update"] = datetime.now().isoformat()
                self._save_update_config(last_sync_version=target_sync_version)
                end_time = datetime.now()
                stats["duration"] = str(end_time - start_time)
                stats["end_time"] = end_time.isoformat()
                return stats

            if use_incremental:
                all_items, current_library_keys = self._get_changed_items_from_api(
                    since_version=last_sync_version,
                    include_fulltext=include_fulltext_via_api,
                )
                # Delete collection entries that are no longer present in the
                # library. Map any chunk ids (``<key>#<n>``) back to item keys
                # so deletion works identically whether or not chunking is on.
                try:
                    stored_ids = self.chroma_client.get_all_ids()
                    stored_item_keys = {i.split("#", 1)[0] for i in stored_ids}
                    to_delete_keys = [k for k in (stored_item_keys - current_library_keys) if k]
                    if to_delete_keys:
                        if self._chunking_enabled and hasattr(self.chroma_client, "delete_item_chunks"):
                            for k in to_delete_keys:
                                self.chroma_client.delete_item_chunks(k)
                        else:
                            self.chroma_client.delete_documents(to_delete_keys)
                        stats["deleted_items"] = len(to_delete_keys)
                        try:
                            sys.stderr.write(f"\nDeleted {len(to_delete_keys)} items no longer present in Zotero.\n")
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"Deletion pass failed: {e}")
            else:
                # Full scan: bootstrap or forced rebuild.
                # Capture the library version BEFORE scanning so any changes
                # made during the scan will be picked up by the next
                # incremental run. Skipping this after a force_full_rebuild
                # would leave last_sync_version stale and the next
                # incremental run would miss items that haven't changed
                # since the old watermark (because they were just deleted
                # along with the collection).
                try:
                    target_sync_version = self.zotero_client.last_modified_version()
                except Exception as e:
                    logger.warning(f"last_modified_version() failed: {e}")
                    target_sync_version = None
                all_items = self._get_items_from_source(
                    limit=limit,
                    extract_fulltext=extract_fulltext,
                    chroma_client=self.chroma_client if not force_full_rebuild else None,
                    force_rebuild=force_full_rebuild,
                    include_fulltext_via_api=include_fulltext_via_api,
                )
                # The local-extraction scan may lag behind the API version
                # captured above (immutable sqlite reads skip WAL contents);
                # only promote the watermark if the snapshot was complete.
                if extract_fulltext and target_sync_version is not None:
                    target_sync_version = self._verify_local_snapshot_version(
                        target_sync_version
                    )

            stats["total_items"] = len(all_items)
            logger.info(f"Found {stats['total_items']} items to process")

            if use_openai_batch:
                stats["batch_mode"] = True
                try:
                    sys.stderr.write(f"\nSubmitting {len(all_items)} items to OpenAI Batch API...\n")
                    sys.stderr.flush()
                except Exception:
                    pass
                stats = self._submit_openai_batch_index(
                    all_items,
                    force_full_rebuild=force_full_rebuild,
                    target_sync_version=target_sync_version,
                    stats=stats,
                )
                try:
                    batch_ids = ", ".join(stats.get("batch_ids", []))
                    sys.stderr.write(
                        "  Submitted OpenAI embedding batch"
                        f"{'es' if len(stats.get('batch_ids', [])) != 1 else ''}: {batch_ids}\n"
                    )
                    sys.stderr.write("  Run 'zotero-mcp openai-batch-status' to check progress.\n")
                    sys.stderr.write("  Run 'zotero-mcp openai-batch-import' after the batch completes.\n")
                except Exception:
                    pass
                end_time = datetime.now()
                stats["duration"] = str(end_time - start_time)
                stats["end_time"] = end_time.isoformat()
                return stats

            # User-friendly progress reporting
            total = stats["total_items"] = len(all_items)
            try:
                sys.stderr.write(f"\nIndexing {total} items...\n\n")
                sys.stderr.flush()
            except Exception:
                pass

            # Process items in batches
            # Keep batch size under OpenAI's 300k token-per-request limit
            # (25 × 8000 max tokens = 200k, well within the limit)
            batch_size = 25
            seen_items = 0
            _failed_docs = []  # Collect failures for end-of-run retry
            for i in range(0, len(all_items), batch_size):
                batch = all_items[i : i + batch_size]

                # Show per-item progress within this batch
                for item in batch:
                    seen_items += 1
                    title = item.get("data", {}).get("title", "")
                    if title and len(title) > 60:
                        title = title[:57] + "..."
                    pct = int(seen_items / total * 100) if total else 0
                    try:
                        sys.stderr.write(f"\r  [{pct:3d}%] {seen_items}/{total} — {title or 'processing...'}")
                        sys.stderr.flush()
                    except Exception:
                        pass

                batch_stats = self._process_item_batch(batch, force_full_rebuild, _failed_docs)

                stats["processed_items"] += batch_stats["processed"]
                stats["added_items"] += batch_stats["added"]
                stats["updated_items"] += batch_stats["updated"]
                stats["skipped_items"] += batch_stats["skipped"]
                stats["errors"] += batch_stats["errors"]

                logger.info(
                    f"Processed {seen_items}/{total} items (added: {stats['added_items']}, skipped: {stats['skipped_items']})"
                )

            # Retry any documents that failed during the main run
            if _failed_docs:
                try:
                    sys.stderr.write(f"\r{' ' * 120}\r")
                    sys.stderr.write(f"\n  Retrying {len(_failed_docs)} failed items...\n")
                except Exception:
                    pass

                import time as _retry_time

                _retry_time.sleep(1)  # Brief pause before retry

                retry_ok = 0
                retry_fail = 0
                for doc, meta, doc_id in _failed_docs:
                    try:
                        self.chroma_client.upsert_documents([doc], [meta], [doc_id])
                        retry_ok += 1
                        stats["errors"] -= 1  # Remove from error count
                        # Don't classify as added vs updated — when the
                        # original batch failed, the add/update lookup never
                        # ran, so we don't know which category it belongs in.
                        # Track recovered items in their own bucket.
                        stats["recovered_items"] += 1
                    except Exception as e2:
                        retry_fail += 1
                        logger.error(f"Retry failed for {doc_id}: {e2}")

                try:
                    sys.stderr.write(f"  Retry: {retry_ok} recovered, {retry_fail} still failed\n")
                except Exception:
                    pass

            # Clear the progress line and show summary
            try:
                sys.stderr.write(f"\r{' ' * 120}\r")  # Clear line
                summary = (
                    f"  Done: {stats['processed_items']} indexed, "
                    f"{stats['skipped_items']} skipped, "
                    f"{stats['errors']} errors"
                )
                if stats["recovered_items"]:
                    summary += f", {stats['recovered_items']} recovered"
                sys.stderr.write(summary + "\n")
            except Exception:
                pass

            # Update last update time, and promote last_sync_version on success
            self.update_config["last_update"] = datetime.now().isoformat()
            self._save_update_config(last_sync_version=target_sync_version)

            end_time = datetime.now()
            stats["duration"] = str(end_time - start_time)
            stats["end_time"] = end_time.isoformat()

            logger.info(f"Database update completed in {stats['duration']}")
            return stats

        except Exception as e:
            logger.error(f"Error updating database: {e}")
            stats["error"] = str(e)
            end_time = datetime.now()
            stats["duration"] = str(end_time - start_time)
            return stats
        finally:
            # Release the update flock on every exit path. Paired with the
            # __enter__ call above; the "not acquired" branch releases
            # separately before its early return, so this finally only runs
            # for the path where we actually hold the lock.
            lock_cm.__exit__(None, None, None)

    def _process_item_batch(
        self,
        items: list[dict[str, Any]],
        force_rebuild: bool = False,
        _failed_docs: list | None = None,
    ) -> dict[str, int]:
        """Process a batch of items.

        _failed_docs: optional list (passed by reference from update_database)
        that collects (doc_text, metadata, doc_id) tuples for batches that fail
        mid-run. Without this, the retry path at update_database:839-865 is
        dead code — a NameError raised here would crash the whole reindex,
        making every transient ChromaDB error fatal instead of recoverable.
        """
        stats = {"processed": 0, "added": 0, "updated": 0, "skipped": 0, "errors": 0}

        chunking = self._chunking_enabled
        chunk_size = int(self._chunking_config.get("chunk_size", 1500))
        overlap = int(self._chunking_config.get("overlap", 200))
        max_chunks = int(self._chunking_config.get("max_chunks_per_item", 20))

        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []
        # One entry per *item* successfully prepared (not per chunk) so add/
        # update accounting stays item-granular regardless of chunking.
        item_keys_order: list[str] = []

        for item in items:
            try:
                item_key = item.get("key", "")
                if not item_key:
                    stats["skipped"] += 1
                    continue

                # Create document text and metadata
                # Always include structured fields; append fulltext when available
                fulltext = item.get("data", {}).get("fulltext", "")
                structured_text = self._create_document_text(item)
                if fulltext.strip():
                    doc_text = (structured_text + "\n\n" + fulltext) if structured_text.strip() else fulltext
                else:
                    doc_text = structured_text
                metadata = self._create_metadata(item)

                if not doc_text.strip():
                    stats["skipped"] += 1
                    continue

                if chunking:
                    # Index one vector per overlapping passage so search can
                    # return a grounded quote and long PDFs stay searchable
                    # past the single-vector truncation limit.
                    passages = split_into_passages(doc_text, chunk_size, overlap, max_chunks)
                    if not passages:
                        stats["skipped"] += 1
                        continue
                    n_chunks = len(passages)
                    for ci, (chunk_text, c0, c1) in enumerate(passages):
                        cmeta = dict(metadata)
                        cmeta["parent_item_key"] = item_key
                        cmeta["chunk_index"] = ci
                        cmeta["n_chunks"] = n_chunks
                        cmeta["char_start"] = c0
                        cmeta["char_end"] = c1
                        page = _page_for_offset(doc_text, c0)
                        if page is not None:
                            cmeta["page"] = page
                        documents.append(self.chroma_client.truncate_text(chunk_text))
                        metadatas.append(cmeta)
                        ids.append(f"{item_key}#{ci}")
                else:
                    # Truncate to fit the configured embedding model's token limit
                    documents.append(self.chroma_client.truncate_text(doc_text))
                    metadatas.append(metadata)
                    ids.append(item_key)

                item_keys_order.append(item_key)
                stats["processed"] += 1

            except Exception as e:
                logger.error(f"Error processing item {item.get('key', 'unknown')}: {e}")
                stats["errors"] += 1

        # Add documents to ChromaDB if any
        if documents:
            # Which items already existed (drives added-vs-updated). When
            # chunking, also clear an item's stale passages before re-adding so
            # a shrinking document never leaves orphaned chunks behind.
            existing_item_keys: set[str] = set()
            if not force_rebuild:
                if chunking:
                    probe_ids = [f"{k}#0" for k in item_keys_order]
                    existing_chunk0 = self.chroma_client.get_existing_ids(probe_ids)
                    existing_item_keys = {cid.split("#", 1)[0] for cid in existing_chunk0}
                    if hasattr(self.chroma_client, "delete_item_chunks"):
                        for k in dict.fromkeys(item_keys_order):
                            try:
                                self.chroma_client.delete_item_chunks(k)
                            except Exception as e:
                                logger.debug(f"delete_item_chunks({k}) failed: {e}")
                else:
                    existing_item_keys = self.chroma_client.get_existing_ids(ids)

            try:
                self.chroma_client.upsert_documents(documents, metadatas, ids)
                for k in item_keys_order:
                    if k in existing_item_keys:
                        stats["updated"] += 1
                    else:
                        stats["added"] += 1
            except Exception as e:
                # Batch failed — collect failures for end-of-run retry.
                # ChromaDB's ONNX tokenizer can fail intermittently in bursts;
                # retrying immediately usually fails too. Collecting failures
                # and retrying after all batches are done is more effective.
                logger.warning(f"Batch upsert failed ({e}), saving for retry")
                if _failed_docs is not None:
                    for j in range(len(documents)):
                        _failed_docs.append((documents[j], metadatas[j], ids[j]))
                    # Count them as errors so stats are accurate
                    stats["errors"] += len(documents)
                else:
                    # No retry list — this is the legacy crash path; re-raise
                    # so caller sees the real error instead of hiding it.
                    raise

        return stats

    def get_openai_batch_status(self, batch_ids: list[str] | None = None) -> dict[str, Any]:
        """Refresh and return OpenAI Batch API status for the latest run or selected batches."""
        selected_ids = set(batch_ids or [])
        manifest = openai_batch.find_manifest(
            config_path=self.config_path,
            batch_id=next(iter(selected_ids), None),
        )
        manifest = openai_batch.refresh_manifest_status(
            manifest,
            embedding_config=self.chroma_client.embedding_config,
            batch_ids=selected_ids or None,
        )
        batches = [
            batch for batch in manifest.get("batches", [])
            if not selected_ids or batch.get("batch_id") in selected_ids
        ]
        missing_ids = selected_ids - {batch.get("batch_id") for batch in batches}
        if missing_ids:
            raise FileNotFoundError(f"No OpenAI batch manifest entries found for: {', '.join(sorted(missing_ids))}")
        return {
            "run_id": manifest.get("run_id"),
            "manifest_path": manifest.get("manifest_path"),
            "model": manifest.get("model"),
            "force_full_rebuild": manifest.get("force_full_rebuild", False),
            "batches": batches,
        }

    def import_openai_batch(self, batch_ids: list[str] | None = None) -> dict[str, Any]:
        """Import completed OpenAI Batch API embeddings into ChromaDB."""
        selected_ids = set(batch_ids or [])
        manifest = openai_batch.find_manifest(
            config_path=self.config_path,
            batch_id=next(iter(selected_ids), None),
        )
        manifest = openai_batch.refresh_manifest_status(
            manifest,
            embedding_config=self.chroma_client.embedding_config,
            batch_ids=selected_ids or None,
        )

        all_batches = manifest.get("batches", [])
        batches = [
            batch for batch in all_batches
            if not selected_ids or batch.get("batch_id") in selected_ids
        ]
        missing_ids = selected_ids - {batch.get("batch_id") for batch in batches}
        if missing_ids:
            raise FileNotFoundError(f"No OpenAI batch manifest entries found for: {', '.join(sorted(missing_ids))}")
        if not batches:
            raise ValueError("No matching OpenAI batches found in the local manifest")
        if manifest.get("force_full_rebuild") and selected_ids and len(batches) != len(all_batches):
            raise RuntimeError("Force-rebuild OpenAI batch runs must be imported as a complete run")
        if manifest.get("force_full_rebuild"):
            incomplete = [
                batch.get("batch_id")
                for batch in all_batches
                if not batch.get("imported_at") and batch.get("status") != "completed"
            ]
            if incomplete:
                raise RuntimeError(
                    "Force-rebuild OpenAI batch runs can only be imported after all batches complete: "
                    + ", ".join(incomplete)
                )

        stats = {
            "run_id": manifest.get("run_id"),
            "manifest_path": manifest.get("manifest_path"),
            "batches_seen": len(batches),
            "batches_imported": 0,
            "batches_skipped": 0,
            "imported_items": 0,
            "added_items": 0,
            "updated_items": 0,
            "failed_items": 0,
            "missing_items": 0,
            "errors": [],
        }

        lock_path = Path.home() / ".config" / "zotero-mcp" / "update.lock"
        lock_cm = _acquire_update_lock(lock_path)
        acquired = lock_cm.__enter__()
        if not acquired:
            lock_cm.__exit__(None, None, None)
            raise RuntimeError(f"Another semantic-search update is already running (lock held at {lock_path})")

        try:
            already_imported = any(batch.get("imported_at") for batch in all_batches)
            if (
                manifest.get("force_full_rebuild")
                and not already_imported
                and any(not batch.get("imported_at") for batch in batches)
            ):
                self.chroma_client.reset_collection()

            client = openai_batch.create_openai_client(self.chroma_client.embedding_config)
            for batch in batches:
                if batch.get("imported_at"):
                    stats["batches_skipped"] += 1
                    continue
                if batch.get("status") != "completed":
                    stats["batches_skipped"] += 1
                    stats["errors"].append({
                        "batch_id": batch.get("batch_id"),
                        "error": f"Batch status is {batch.get('status')}, not completed",
                    })
                    continue
                output_file_id = batch.get("output_file_id")
                if not output_file_id:
                    stats["batches_skipped"] += 1
                    stats["errors"].append({"batch_id": batch.get("batch_id"), "error": "Missing output_file_id"})
                    continue

                output_path = Path(batch["records_path"]).with_name(Path(batch["records_path"]).stem + "-output.jsonl")
                if output_path.exists():
                    output_text = output_path.read_text(encoding="utf-8")
                else:
                    output_text = openai_batch.download_file_text(client, output_file_id, output_path)
                embeddings_by_id, row_failures = openai_batch.parse_embedding_output(output_text)

                if batch.get("error_file_id"):
                    error_path = Path(batch["records_path"]).with_name(Path(batch["records_path"]).stem + "-errors.jsonl")
                    error_text = openai_batch.download_file_text(client, batch["error_file_id"], error_path)
                    row_failures.extend(openai_batch.parse_error_output(error_text))

                records = {record["id"]: record for record in openai_batch.read_jsonl(Path(batch["records_path"]))}
                ids = [doc_id for doc_id in embeddings_by_id if doc_id in records]
                unexpected_output_ids = [doc_id for doc_id in embeddings_by_id if doc_id not in records]
                failure_ids = {
                    failure.get("custom_id")
                    for failure in row_failures
                    if failure.get("custom_id")
                }
                missing_result_ids = [
                    doc_id
                    for doc_id in records
                    if doc_id not in embeddings_by_id and doc_id not in failure_ids
                ]
                missing_errors = [
                    {"custom_id": doc_id, "error": "Batch output returned an embedding for an unknown record"}
                    for doc_id in unexpected_output_ids
                ] + [
                    {"custom_id": doc_id, "error": "No embedding or error row returned for batch record"}
                    for doc_id in missing_result_ids
                ]
                stats["missing_items"] += len(unexpected_output_ids) + len(missing_result_ids)
                stats["failed_items"] += len(row_failures)
                stats["errors"].extend(row_failures)
                stats["errors"].extend(missing_errors)

                if ids:
                    existing_ids = self.chroma_client.get_existing_ids(ids)
                    self.chroma_client.upsert_embeddings(
                        documents=[records[doc_id]["document"] for doc_id in ids],
                        metadatas=[records[doc_id]["metadata"] for doc_id in ids],
                        ids=ids,
                        embeddings=[embeddings_by_id[doc_id] for doc_id in ids],
                    )
                    stats["imported_items"] += len(ids)
                    stats["updated_items"] += len(existing_ids)
                    stats["added_items"] += len(ids) - len(existing_ids)

                batch["imported_at"] = datetime.now().isoformat()
                batch["imported_count"] = len(ids)
                stats["batches_imported"] += 1

            openai_batch.save_manifest(manifest)
            if all(batch.get("imported_at") for batch in all_batches):
                self.update_config["last_update"] = datetime.now().isoformat()
                self._save_update_config(last_sync_version=manifest.get("target_sync_version"))
            return stats
        finally:
            lock_cm.__exit__(None, None, None)

    def search(self,
               query: str,
               limit: int = 10,
               filters: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Perform semantic search over the Zotero library.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            Search results with Zotero item details
        """
        try:
            # Over-fetch candidates when re-ranking and/or chunking are on.
            reranker = self._get_reranker()
            fetch_limit = limit
            if self._chunking_enabled:
                # Passages are grouped back to items downstream, so fetch
                # several chunks per desired item to still surface ~limit
                # distinct papers.
                fetch_limit = max(fetch_limit, limit * 4)
            if reranker:
                multiplier = self._reranker_config.get("candidate_multiplier", 3)
                fetch_limit = max(fetch_limit, limit * multiplier)

            # Perform semantic search
            results = self.chroma_client.search(query_texts=[query], n_results=fetch_limit, where=filters)

            # Re-rank results with cross-encoder if enabled. With chunking we
            # rerank ALL candidates (grouping to `limit` items happens in
            # enrichment); without chunking we keep the historical top-k=limit.
            if reranker and results.get("documents") and results["documents"][0]:
                documents = results["documents"][0]
                top_k = len(documents) if self._chunking_enabled else limit
                ranked_indices = reranker.rerank(query, documents, top_k=top_k)
                for key in ["ids", "distances", "documents", "metadatas"]:
                    if results.get(key) and results[key][0]:
                        results[key][0] = [results[key][0][i] for i in ranked_indices]

            # Enrich results with full Zotero item data, grouping passages back
            # to their parent items and capping at `limit` distinct papers.
            enriched_results = self._enrich_search_results(results, query, limit)

            return {
                "query": query,
                "limit": limit,
                "filters": filters,
                "results": enriched_results,
                "total_found": len(enriched_results),
            }

        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            return {
                "query": query,
                "limit": limit,
                "filters": filters,
                "results": [],
                "total_found": 0,
                "error": str(e),
            }

    def _enrich_search_results(
        self, chroma_results: dict[str, Any], query: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Enrich ChromaDB results with full Zotero item data.

        Chunk-aware: when the collection is indexed as passages, ids look like
        ``<item_key>#<n>``. Results are grouped back to their parent item — the
        first (best-ranked) passage per item wins — and capped at ``limit``
        distinct items. For every hit a grounded ``matched_passage`` quote and,
        when available, the passage's character offset and page are attached so
        callers can cite precisely. Item-level collections (ids without ``#``)
        flow through unchanged.
        """
        enriched: list[dict[str, Any]] = []

        if not chroma_results.get("ids") or not chroma_results["ids"][0]:
            return enriched

        ids = chroma_results["ids"][0]
        distances = chroma_results.get("distances", [[]])[0]
        documents = chroma_results.get("documents", [[]])[0]
        metadatas = chroma_results.get("metadatas", [[]])[0]

        seen_items: set[str] = set()
        for i, raw_id in enumerate(ids):
            item_key = raw_id.split("#", 1)[0]
            if item_key in seen_items:
                continue
            seen_items.add(item_key)

            distance = distances[i] if i < len(distances) else None
            document = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}

            passage, passage_offset = best_snippet(query, document)

            enriched_result: dict[str, Any] = {
                "item_key": item_key,
                "similarity_score": (1 - distance) if distance is not None else 0,
                "matched_text": document,
                "matched_passage": passage,
                "metadata": meta if isinstance(meta, dict) else {},
                "query": query,
            }
            # Passage provenance — present only on a chunk-indexed collection.
            if isinstance(meta, dict):
                for mk in ("chunk_index", "n_chunks", "char_start", "char_end", "page"):
                    if mk in meta:
                        enriched_result[mk] = meta[mk]
            if "char_start" not in enriched_result and passage_offset:
                enriched_result["passage_offset"] = passage_offset

            try:
                enriched_result["zotero_item"] = self.zotero_client.item(item_key)
            except Exception as e:
                logger.error(f"Error enriching result for item {item_key}: {e}")
                enriched_result["error"] = f"Could not fetch full item data: {e}"

            enriched.append(enriched_result)
            if limit and len(enriched) >= limit:
                break

        return enriched

    def get_database_status(self) -> dict[str, Any]:
        """Get status information about the semantic search database."""
        collection_info = self.chroma_client.get_collection_info()

        return {
            "collection_info": collection_info,
            "update_config": self.update_config,
            "openai_batch": {
                "enabled": self._load_openai_batch_enabled(),
                "active": self._resolve_openai_batch_enabled(None),
            },
            "should_update": self.should_update_database(),
            "last_update": self.update_config.get("last_update"),
        }

    def delete_item(self, item_key: str) -> bool:
        """Delete an item from the semantic search database."""
        try:
            self.chroma_client.delete_documents([item_key])
            return True
        except Exception as e:
            logger.error(f"Error deleting item {item_key}: {e}")
            return False


def create_semantic_search(config_path: str | None = None, db_path: str | None = None) -> ZoteroSemanticSearch:
    """
    Create a ZoteroSemanticSearch instance.

    Args:
        config_path: Path to configuration file
        db_path: Optional path to Zotero database (overrides config file)

    Returns:
        Configured ZoteroSemanticSearch instance
    """
    return ZoteroSemanticSearch(config_path=config_path, db_path=db_path)
