"""
Semantic search functionality for Zotero MCP.

This module provides semantic search capabilities by integrating ChromaDB
with the existing Zotero client to enable vector-based similarity search
over research libraries.
"""

import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import logging

try:
    import tiktoken
    _tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    tiktoken = None
    _tokenizer = None

from pyzotero import zotero

from .chroma_client import ChromaClient, create_chroma_client
from .client import get_zotero_client
from .utils import format_creators, is_local_mode
from .local_db import LocalZoteroReader, get_local_zotero_reader

logger = logging.getLogger(__name__)


from zotero_mcp.utils import suppress_stdout


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


class CrossEncoderReranker:
    """Optional cross-encoder re-ranker for semantic search results."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[int]:
        """Re-rank documents by relevance to query.

        Returns indices of top_k documents in descending relevance order.
        """
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked_indices[:top_k]


class ZoteroSemanticSearch:
    """Semantic search interface for Zotero libraries using ChromaDB."""

    def __init__(self,
                 chroma_client: ChromaClient | None = None,
                 config_path: str | None = None,
                 db_path: str | None = None):
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

        # Load update configuration
        self.update_config = self._load_update_config()

        # Reranker (lazy-initialized on first search)
        self._reranker: CrossEncoderReranker | None = None
        self._reranker_config = self._load_reranker_config()

        # Embedding-request throttle state (thread-safe: the indexing loop
        # runs batches sequentially but the pre-search fire-and-forget sync
        # in tools/search.py kicks off another indexer thread).
        self._embed_throttle_lock = threading.Lock()
        self._last_embed_ts: float = 0.0

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
        config = {
            "auto_update": False,
            "update_frequency": "manual",
            "last_update": None,
            "update_days": 7
        }

        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    file_config = json.load(f)
                    config.update(file_config.get("semantic_search", {}).get("update_config", {}))
            except Exception as e:
                logger.warning(f"Error loading update config: {e}")

        return config

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

    def _load_chunking_settings(self) -> dict[str, int]:
        """Chunk window / overlap sizes in tiktoken cl100k_base tokens.

        Default 1500-token window with 225-token (15%) overlap fits
        comfortably inside OpenAI text-embedding-3-small (8192 ctx),
        SiliconFlow bge-m3 (8192 ctx), and leaves headroom for prepended
        structured metadata.
        """
        defaults = {"window": 1500, "overlap": 225}
        if not self.config_path or not os.path.exists(self.config_path):
            return defaults
        try:
            with open(self.config_path) as f:
                file_config = json.load(f)
                overrides = file_config.get("semantic_search", {}).get("chunking", {})
                if isinstance(overrides, dict):
                    defaults.update({k: int(v) for k, v in overrides.items()
                                     if k in defaults})
        except Exception as e:
            logger.warning(f"Error loading chunking settings: {e}")
        return defaults

    def _load_embedding_rate_limit(self) -> float | None:
        """Max embedding HTTP requests per second, or None for no throttle.

        Useful when driving a rate-limited OpenAI-compatible endpoint such as
        SiliconFlow or a free-tier OpenAI key. Default is None (no throttle).
        """
        if not self.config_path or not os.path.exists(self.config_path):
            return None
        try:
            with open(self.config_path) as f:
                file_config = json.load(f)
                val = file_config.get("semantic_search", {}).get("embedding_rate_limit_rps")
                if val is None:
                    return None
                rps = float(val)
                return rps if rps > 0 else None
        except Exception:
            return None

    def _throttle_embedding_request(self) -> None:
        """Sleep as needed to respect `embedding_rate_limit_rps`.

        Call this immediately before every ChromaDB upsert/add that will
        trigger a remote embedding request. Each upsert batches many docs
        into a single POST, so throttling at the upsert boundary matches
        the per-request rate limit enforced by the provider.
        """
        rps = self._load_embedding_rate_limit()
        if not rps:
            return
        with self._embed_throttle_lock:
            now = time.monotonic()
            min_interval = 1.0 / rps
            wait = min_interval - (now - self._last_embed_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_embed_ts = time.monotonic()

    def _chunk_document(self, text: str,
                        window: int = 1500,
                        overlap: int = 225) -> list[str]:
        """Split text into overlapping token windows.

        Uses tiktoken (cl100k_base) for token counting so window sizes are
        aligned with the tokenizer used by OpenAI and OpenAI-compatible
        embedding providers (SiliconFlow, Mistral, etc.). Falls back to a
        conservative 4-char-per-token approximation when tiktoken is
        unavailable.

        Short inputs (≤ window tokens) return a single-chunk list — callers
        must still wrap in a list to keep the ingest loop uniform.
        """
        if not text or not text.strip():
            return []
        if window <= 0:
            return [text.strip()]
        # Clamp overlap to < window to prevent infinite loops on malformed config
        overlap = max(0, min(overlap, window - 1))
        step = max(1, window - overlap)

        if _tokenizer is None:
            char_window = window * 4
            char_step = step * 4
            chunks = []
            for i in range(0, len(text), char_step):
                piece = text[i:i + char_window].strip()
                if piece:
                    chunks.append(piece)
                if i + char_window >= len(text):
                    break
            return chunks or [text.strip()]

        tokens = _tokenizer.encode(text, disallowed_special=())
        if len(tokens) <= window:
            return [text.strip()]

        chunks: list[str] = []
        for i in range(0, len(tokens), step):
            window_tokens = tokens[i:i + window]
            if not window_tokens:
                break
            piece = _tokenizer.decode(window_tokens).strip()
            if piece:
                chunks.append(piece)
            if i + window >= len(tokens):
                break
        return chunks

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
            with open(self.config_path, 'w') as f:
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
            note_text = re.sub(r'<[^>]+>', '', note)
            extra_fields.append(note_text)

        # Combine all text fields
        text_parts = [title, creators_text, abstract] + extra_fields
        return " ".join(filter(None, text_parts))

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
        if not self.update_config.get("auto_update", False):
            return False

        frequency = self.update_config.get("update_frequency", "manual")

        if frequency == "manual":
            return False
        elif frequency == "startup":
            return True
        elif frequency == "daily":
            last_update = self.update_config.get("last_update")
            if not last_update:
                return True

            last_update_date = datetime.fromisoformat(last_update)
            return datetime.now() - last_update_date >= timedelta(days=1)
        elif frequency.startswith("every_"):
            try:
                days = int(frequency.split("_")[1])
                last_update = self.update_config.get("last_update")
                if not last_update:
                    return True

                last_update_date = datetime.fromisoformat(last_update)
                return datetime.now() - last_update_date >= timedelta(days=days)
            except (ValueError, IndexError):
                return False

        return False

    def _get_items_from_source(self,
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
                limit,
                extract_fulltext=extract_fulltext,
                chroma_client=chroma_client,
                force_rebuild=force_rebuild
            )
        else:
            return self._get_items_from_api(limit, include_fulltext=include_fulltext_via_api)

    def _get_items_from_local_db(self, limit: int | None = None, extract_fulltext: bool = False, chroma_client: ChromaClient | None = None, force_rebuild: bool = False) -> list[dict[str, Any]]:
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
                        semantic_cfg = _cfg.get('semantic_search', {})
                        extraction_cfg = semantic_cfg.get('extraction', {})
                        pdf_max_pages = extraction_cfg.get('pdf_max_pages')
                        pdf_timeout = extraction_cfg.get('pdf_timeout', 30)
                        # Use config db_path only if no CLI override
                        if not zotero_db_path:
                            zotero_db_path = semantic_cfg.get('zotero_db_path')
            except Exception:
                pass

            with suppress_stdout(), LocalZoteroReader(db_path=zotero_db_path, pdf_max_pages=pdf_max_pages, pdf_timeout=pdf_timeout) as reader:
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
                            if best is not None and best is not it and getattr(best, "item_type", None) == "journalArticle":
                                drop = True
                                break
                        if drop:
                            continue
                    filtered_items.append(it)

                local_items = filtered_items
                total_to_extract = len(local_items)
                if total_to_extract != candidate_count:
                    try:
                        sys.stderr.write(f"After filtering/dedup: {total_to_extract} items to process. Extracting content...\n")
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
                            sys.stderr.write(f"  {len(_skipped_failed)} item(s) skipped (PDF extraction previously failed):\n")
                            for name in _skipped_failed[:5]:  # Show first 5
                                sys.stderr.write(f"    - {name}\n")
                            if len(_skipped_failed) > 5:
                                sys.stderr.write(f"    ... and {len(_skipped_failed) - 5} more\n")
                            sys.stderr.write(f"  (To retry these, run with --force-rebuild)\n")
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
                            "itemType": getattr(item, 'item_type', None) or "journalArticle",
                            "title": item.title or "",
                            "abstractNote": item.abstract or "",
                            "extra": item.extra or "",
                            # Include fulltext only when extracted
                            "fulltext": getattr(item, 'fulltext', None) or "" if extract_fulltext else "",
                            "fulltextSource": getattr(item, 'fulltext_source', None) or "" if extract_fulltext else "",
                            # Flag if extraction was attempted but failed (timeout, empty)
                            "fulltext_attempted": getattr(item, '_fulltext_attempted', False),
                            "dateAdded": item.date_added,
                            "dateModified": item.date_modified,
                            "creators": self._parse_creators_string(item.creators) if item.creators else []
                        }
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
        for creator in creators_str.split(';'):
            creator = creator.strip()
            if not creator:
                continue

            if ',' in creator:
                last, first = creator.split(',', 1)
                creators.append({
                    "creatorType": "author",
                    "firstName": first.strip(),
                    "lastName": last.strip()
                })
            else:
                creators.append({
                    "creatorType": "author",
                    "name": creator
                })

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
                    sys.stderr.write(
                        f"\r  Fulltext: {idx}/{total} items checked, "
                        f"{fetched} with text"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
        try:
            sys.stderr.write("\n")
        except Exception:
            pass

    def _get_items_from_api(self,
                            limit: int | None = None,
                            include_fulltext: bool = False) -> list[dict[str, Any]]:
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

            # Filter out attachments, notes, and annotations. Annotations
            # (PDF highlights / user comments) are top-level in the pyzotero
            # /items listing but are semantically noise — they have no body
            # text of their own and clog up semantic search results. The
            # since-based fetch and itemKey batch-fetch already filter
            # annotation too; keeping all three paths consistent.
            filtered_items = [
                item for item in items
                if item.get("data", {}).get("itemType") not in ["attachment", "note", "annotation"]
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

    def _get_changed_items_from_api(self,
                                    since_version: int,
                                    include_fulltext: bool = False
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

    def _fetch_items_by_keys(self, keys: list[str]) -> list[dict[str, Any]]:
        """Bulk-fetch top-level items by itemKey, batched through the
        Zotero web API's `itemKey=K1,K2,...` query parameter.

        Zotero caps the itemKey response at 50 items per request, which is
        still 50× faster than per-key fetches for large backfills. Falls
        back to per-key fetch if a batched request raises so a single bad
        key can't stall the whole backfill. Attachments / notes / inline
        annotations are filtered out to match the rest of the ingest
        pipeline's assumptions.
        """
        if not keys:
            return []
        BATCH = 50
        results: list[dict[str, Any]] = []
        for i in range(0, len(keys), BATCH):
            batch = keys[i:i + BATCH]
            try:
                self.zotero_client.add_parameters(
                    itemKey=",".join(batch),
                    limit=len(batch),
                )
                items = self.zotero_client.items() or []
            except Exception as e:
                logger.debug(
                    f"Batched itemKey fetch failed ({e}); falling back to "
                    f"per-key for this batch of {len(batch)}"
                )
                items = []
                for key in batch:
                    try:
                        one = self.zotero_client.item(key)
                        if one:
                            items.append(one)
                    except Exception as e2:
                        logger.debug(f"item({key}) failed in fallback: {e2}")
            for it in items:
                itype = it.get("data", {}).get("itemType")
                if itype in {"attachment", "note", "annotation"}:
                    continue
                results.append(it)
        return results

    def update_database(self,
                       force_full_rebuild: bool = False,
                       limit: int | None = None,
                       extract_fulltext: bool = False,
                       include_fulltext: bool | None = None) -> dict[str, Any]:
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
            "gap_filled_items": 0,
            "cleaned_annotation_chunks": 0,
            "errors": 0,
            "start_time": start_time.isoformat(),
            "duration": None
        }

        try:
            # Resolve include_fulltext default from config if not specified
            if include_fulltext is None:
                include_fulltext = self._load_include_fulltext_setting()

            # Web-API fulltext only applies when not using the local sqlite
            # extractor (extract_fulltext=True takes precedence in local mode)
            include_fulltext_via_api = include_fulltext and not extract_fulltext

            # Migrate from pre-chunking id format: PR1 stored one id per item
            # (`<key>`); PR2 stores one per chunk (`<key>__<i>`). Mixing both
            # pollutes dedup / rerank. Detect and upgrade to a fresh rebuild
            # so the user never has to run --force-rebuild manually.
            #
            # The "empty collection but cached_sync > 0" case (e.g. after an
            # embedding-model dim change triggered a silent collection
            # reset) doesn't need a force-rebuild here because the
            # diff-driven incremental path's gap-fill naturally handles it:
            # stored_parents is empty -> missing_keys = all library keys ->
            # every item gets ingested, with no need to throw away
            # progress.
            if not force_full_rebuild:
                try:
                    existing_ids = self.chroma_client.get_all_ids()
                    if existing_ids:
                        sample = list(existing_ids)[:20]
                        if not any("__" in i for i in sample):
                            logger.info(
                                "Legacy pre-chunking id format detected; "
                                "rebuilding collection."
                            )
                            try:
                                sys.stderr.write(
                                    "\nCollection was built by an older (pre-chunking) "
                                    "version; rebuilding to enable paragraph-level "
                                    "semantic search...\n"
                                )
                            except Exception:
                                pass
                            force_full_rebuild = True
                except Exception as e:
                    logger.debug(f"Legacy id-format check failed: {e}")

            # Reset collection if force rebuild
            if force_full_rebuild:
                logger.info("Force rebuilding database...")
                self.chroma_client.reset_collection()
            else:
                # Self-healing: purge annotation chunks. A past version of
                # `_get_items_from_api` excluded only attachment+note and
                # let the `annotation` itemType through, producing
                # thousands of useless zero-text entries. Now that the
                # filter is tightened, sweep any stragglers on every run.
                try:
                    cleaned = self.chroma_client.delete_documents_by_item_type("annotation")
                    if cleaned:
                        stats["cleaned_annotation_chunks"] = cleaned
                        try:
                            sys.stderr.write(
                                f"\nCleaned up {cleaned} stale annotation chunks "
                                f"(past ingest-filter bug).\n"
                            )
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug(f"Annotation cleanup failed: {e}")

            # Decide whether to use since-based incremental ingest.
            # Incremental requires: not a forced rebuild, not a local-extraction
            # run (incremental path covers web-API metadata and optionally
            # fulltext only), not a test limit, and a known prior sync version.
            last_sync_version = self._load_last_sync_version() if not force_full_rebuild else 0
            use_incremental = (
                not force_full_rebuild
                and not extract_fulltext
                and limit is None
                and last_sync_version > 0
            )

            target_sync_version: int | None = None
            all_items: list[dict[str, Any]] = []
            if use_incremental:
                try:
                    target_sync_version = self.zotero_client.last_modified_version()
                except Exception as e:
                    logger.warning(f"last_modified_version() failed, falling back to full scan: {e}")
                    use_incremental = False

            if use_incremental:
                # Diff-driven incremental path:
                #   changed = items whose version > last_sync_version
                #   missing = items currently in the library but not in our
                #             ChromaDB (resume case: killed mid-rebuild,
                #             embedding-model change reset, etc.)
                #   deleted = items present in ChromaDB but no longer in the
                #             library
                # The ingest set is `changed ∪ missing`; if both that and
                # `deleted` are empty the run is a true noop.
                changed_items, current_library_keys = self._get_changed_items_from_api(
                    since_version=last_sync_version,
                    include_fulltext=include_fulltext_via_api,
                )

                # Gap-fill: items in the library but missing from ChromaDB.
                # This is what makes resume work — a killed rebuild leaves
                # ChromaDB partially populated without advancing
                # last_sync_version, and the next run needs to pick up the
                # holes instead of declaring "library unchanged; noop".
                stored_ids = self.chroma_client.get_all_ids()
                stored_parents = {
                    i.split("__", 1)[0] if "__" in i else i
                    for i in stored_ids
                }
                already_queued = {it.get("key") for it in changed_items if it.get("key")}
                missing_keys = current_library_keys - stored_parents - already_queued
                gap_items: list[dict[str, Any]] = []
                if missing_keys:
                    try:
                        sys.stderr.write(
                            f"\nGap fill: {len(missing_keys)} items in library "
                            f"missing from ChromaDB; fetching...\n"
                        )
                    except Exception:
                        pass
                    gap_items = self._fetch_items_by_keys(sorted(missing_keys))
                    if include_fulltext_via_api and gap_items:
                        self._attach_web_fulltext(gap_items)
                stats["gap_filled_items"] = len(gap_items)

                all_items = changed_items + gap_items

                # Delete collection entries that are no longer present in the
                # library. With chunking, stored ids are `<parent>__<i>` so we
                # must group by parent before computing the diff.
                try:
                    deleted_parents = [k for k in (stored_parents - current_library_keys) if k]
                    if deleted_parents:
                        total_deleted_chunks = 0
                        for pkey in deleted_parents:
                            total_deleted_chunks += self.chroma_client.delete_documents_by_parent(pkey)
                        stats["deleted_items"] = len(deleted_parents)
                        try:
                            sys.stderr.write(
                                f"\nDeleted {len(deleted_parents)} items "
                                f"({total_deleted_chunks} chunks) no longer present in Zotero.\n"
                            )
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"Deletion pass failed: {e}")

                # True noop: nothing to add, nothing to delete. Still advance
                # the watermark so subsequent runs short-circuit immediately.
                if not all_items and not stats["deleted_items"]:
                    try:
                        sys.stderr.write(
                            f"\nLibrary fully synced (version {target_sync_version}); "
                            f"nothing to reindex.\n"
                        )
                    except Exception:
                        pass
                    self.update_config["last_update"] = datetime.now().isoformat()
                    self._save_update_config(last_sync_version=target_sync_version)
                    end_time = datetime.now()
                    stats["duration"] = str(end_time - start_time)
                    stats["end_time"] = end_time.isoformat()
                    return stats
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

            stats["total_items"] = len(all_items)
            logger.info(f"Found {stats['total_items']} items to process")
            # User-friendly progress reporting
            total = stats['total_items'] = len(all_items)
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
                batch = all_items[i:i + batch_size]

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

                logger.info(f"Processed {seen_items}/{total} items (added: {stats['added_items']}, skipped: {stats['skipped_items']})")

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
                    # Retry must respect the same rate limit as the main
                    # ingest path. Without this throttle, a large retry
                    # burst can hammer SiliconFlow / other rate-limited
                    # providers with N unthrottled HTTP requests and
                    # re-trigger 429s — each of which the retry loop logs
                    # as a permanent failure and moves on, leaving gaps.
                    self._throttle_embedding_request()
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

    def _process_item_batch(
        self,
        items: list[dict[str, Any]],
        force_rebuild: bool = False,
        _failed_docs: list | None = None,
    ) -> dict[str, int]:
        """Process a batch of items into chunked embeddings.

        Each Zotero item is split into one-or-more overlapping token chunks
        via `_chunk_document`. Chunk ids follow `<item_key>__<chunk_index>`
        and each chunk's metadata carries `parent_item_key` so the search
        handler can groupby back to one result per paper.

        Before upserting fresh chunks for an item we delete any pre-existing
        chunks for that parent so the chunk count can shrink between runs
        (fulltext re-extraction sometimes yields less text).

        _failed_docs: optional list (passed by reference from update_database)
        that collects (doc_text, metadata, doc_id) tuples for batches that
        fail mid-run. Without this, a transient ChromaDB error crashes the
        whole reindex instead of surviving via the retry path.
        """
        stats = {"processed": 0, "added": 0, "updated": 0, "skipped": 0, "errors": 0}

        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []
        parents_touched: list[str] = []
        chunk_settings = self._load_chunking_settings()

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
                base_metadata = self._create_metadata(item)

                if not doc_text.strip():
                    stats["skipped"] += 1
                    continue

                # Split into token-bounded chunks. Unlike the old truncate-only
                # path, chunking preserves the entire fulltext across multiple
                # embeddings so paragraph-level semantic matches can land
                # anywhere in the paper, not just the first 8k tokens.
                chunks = self._chunk_document(
                    doc_text,
                    window=chunk_settings["window"],
                    overlap=chunk_settings["overlap"],
                )
                if not chunks:
                    stats["skipped"] += 1
                    continue

                parents_touched.append(item_key)
                total_chunks = len(chunks)
                for idx, chunk_text in enumerate(chunks):
                    chunk_meta = dict(base_metadata)
                    chunk_meta["parent_item_key"] = item_key
                    chunk_meta["chunk_index"] = idx
                    chunk_meta["total_chunks"] = total_chunks
                    documents.append(chunk_text)
                    metadatas.append(chunk_meta)
                    ids.append(f"{item_key}__{idx}")

                stats["processed"] += 1

            except Exception as e:
                logger.error(f"Error processing item {item.get('key', 'unknown')}: {e}")
                stats["errors"] += 1

        if not documents:
            return stats

        # For non-rebuild runs: clear any stale chunks for these parents so
        # chunk count can shrink. On force_rebuild the collection was already
        # reset upstream, so no pre-existing chunks exist.
        if not force_rebuild:
            for pkey in parents_touched:
                try:
                    self.chroma_client.delete_documents_by_parent(pkey)
                except Exception as e:
                    logger.debug(f"Pre-upsert cleanup for {pkey} failed: {e}")

        # Track which parents were already in the collection, for added/updated stats
        pre_existing_parents: set[str] = set()
        if not force_rebuild:
            # get_existing_ids expects chunk ids; sample the first chunk of
            # each parent to decide "new" vs "update" without a separate
            # collection scan. This is a statistics-only signal.
            sample_ids = [f"{p}__0" for p in parents_touched]
            try:
                pre_existing_parents = {
                    i.split("__", 1)[0]
                    for i in self.chroma_client.get_existing_ids(sample_ids)
                }
            except Exception:
                pre_existing_parents = set()

        try:
            self._throttle_embedding_request()
            self.chroma_client.upsert_documents(documents, metadatas, ids)
            for pkey in parents_touched:
                if pkey in pre_existing_parents:
                    stats["updated"] += 1
                else:
                    stats["added"] += 1
        except Exception as e:
            logger.warning(f"Batch upsert failed ({e}), saving for retry")
            if _failed_docs is not None:
                for j in range(len(documents)):
                    _failed_docs.append((documents[j], metadatas[j], ids[j]))
                stats["errors"] += len(parents_touched)
            else:
                raise

        return stats

    def search(self,
               query: str,
               limit: int = 10,
               filters: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Perform semantic search over the Zotero library.

        Because items are stored as multiple chunks (one per ~1500 tokens),
        a naive top-N query can return the same paper multiple times. We
        oversample from ChromaDB, groupby `parent_item_key`, keep the best
        chunk per parent, then truncate to the user's requested limit.

        Args:
            query: Search query text
            limit: Maximum number of results to return (counted in unique papers)
            filters: Optional metadata filters

        Returns:
            Search results with Zotero item details
        """
        try:
            reranker = self._get_reranker()
            rerank_mult = self._reranker_config.get("candidate_multiplier", 3) if reranker else 1
            target_candidates = max(limit, 1) * rerank_mult
            # Oversample generously: same paper may contribute many chunks,
            # so the first N hits can all come from 1-2 papers. A 5x
            # oversample on top of rerank's multiplier balances recall vs.
            # network cost.
            chroma_limit = max(target_candidates * 5, 50)

            results = self.chroma_client.search(
                query_texts=[query],
                n_results=chroma_limit,
                where=filters
            )

            # Collapse by parent: one best chunk per paper
            results = self._dedupe_by_parent(results, keep=target_candidates)

            # Re-rank survives after dedup since it works on (query, doc)
            # pairs which are independent of chunk identity.
            if reranker and results.get("documents") and results["documents"][0]:
                documents = results["documents"][0]
                ranked_indices = reranker.rerank(query, documents, top_k=limit)
                for key in ["ids", "distances", "documents", "metadatas"]:
                    if results.get(key) and results[key][0]:
                        results[key][0] = [results[key][0][i] for i in ranked_indices]
            else:
                # Without a reranker, dedupe already sorted by distance; just
                # clip to the user-visible limit.
                for key in ["ids", "distances", "documents", "metadatas"]:
                    if results.get(key) and results[key][0]:
                        results[key][0] = results[key][0][:limit]

            enriched_results = self._enrich_search_results(results, query)

            return {
                "query": query,
                "limit": limit,
                "filters": filters,
                "results": enriched_results,
                "total_found": len(enriched_results)
            }

        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            return {
                "query": query,
                "limit": limit,
                "filters": filters,
                "results": [],
                "total_found": 0,
                "error": str(e)
            }

    def _dedupe_by_parent(self, chroma_results: dict[str, Any], keep: int) -> dict[str, Any]:
        """Collapse chunk hits so each parent_item_key appears at most once.

        Keeps the chunk with the smallest distance (highest similarity)
        per parent. Sorts the survivors ascending by distance and truncates
        to `keep`. Falls back to the id's pre-`__` prefix when a metadata
        record lacks `parent_item_key` (e.g. entries indexed before the
        chunking migration).
        """
        if not chroma_results.get("ids") or not chroma_results["ids"][0]:
            return chroma_results

        ids = chroma_results["ids"][0]
        distances = chroma_results.get("distances", [[]])[0] or []
        documents = chroma_results.get("documents", [[]])[0] or []
        metadatas = chroma_results.get("metadatas", [[]])[0] or []

        best_per_parent: dict[str, int] = {}  # parent -> original index
        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            pkey = (meta or {}).get("parent_item_key")
            if not pkey:
                pkey = doc_id.split("__", 1)[0] if "__" in doc_id else doc_id
            dist = distances[i] if i < len(distances) else float("inf")
            cur = best_per_parent.get(pkey)
            if cur is None:
                best_per_parent[pkey] = i
                continue
            cur_dist = distances[cur] if cur < len(distances) else float("inf")
            if dist < cur_dist:
                best_per_parent[pkey] = i

        kept_indices = sorted(
            best_per_parent.values(),
            key=lambda i: distances[i] if i < len(distances) else float("inf"),
        )[:keep]

        return {
            "ids": [[ids[i] for i in kept_indices]],
            "distances": [[distances[i] for i in kept_indices]] if distances else [[]],
            "documents": [[documents[i] for i in kept_indices]] if documents else [[]],
            "metadatas": [[metadatas[i] for i in kept_indices]] if metadatas else [[]],
        }

    def _enrich_search_results(self, chroma_results: dict[str, Any], query: str) -> list[dict[str, Any]]:
        """Enrich ChromaDB results with full Zotero item data.

        `ids` in chroma_results are chunk-scoped (`<parent>__<index>`).
        We resolve each to its parent Zotero item for the web-API lookup
        and surface that parent key as the result's `item_key`.
        """
        enriched = []

        if not chroma_results.get("ids") or not chroma_results["ids"][0]:
            return enriched

        ids = chroma_results["ids"][0]
        distances = chroma_results.get("distances", [[]])[0]
        documents = chroma_results.get("documents", [[]])[0]
        metadatas = chroma_results.get("metadatas", [[]])[0]

        for i, chunk_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            parent_key = (meta or {}).get("parent_item_key")
            if not parent_key:
                parent_key = chunk_id.split("__", 1)[0] if "__" in chunk_id else chunk_id
            try:
                # Get full item data from Zotero
                zotero_item = self.zotero_client.item(parent_key)

                enriched_result = {
                    "item_key": parent_key,
                    "chunk_id": chunk_id,
                    "chunk_index": (meta or {}).get("chunk_index"),
                    "similarity_score": 1 - distances[i] if i < len(distances) else 0,
                    "matched_text": documents[i] if i < len(documents) else "",
                    "metadata": meta,
                    "zotero_item": zotero_item,
                    "query": query
                }

                enriched.append(enriched_result)

            except Exception as e:
                logger.error(f"Error enriching result for item {parent_key}: {e}")
                # Include basic result even if enrichment fails
                enriched.append({
                    "item_key": parent_key,
                    "chunk_id": chunk_id,
                    "chunk_index": (meta or {}).get("chunk_index"),
                    "similarity_score": 1 - distances[i] if i < len(distances) else 0,
                    "matched_text": documents[i] if i < len(documents) else "",
                    "metadata": meta,
                    "query": query,
                    "error": f"Could not fetch full item data: {e}"
                })

        return enriched

    def get_database_status(self) -> dict[str, Any]:
        """Get status information about the semantic search database."""
        collection_info = self.chroma_client.get_collection_info()

        return {
            "collection_info": collection_info,
            "update_config": self.update_config,
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
