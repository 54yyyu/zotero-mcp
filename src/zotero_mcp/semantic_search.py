"""
Semantic search functionality for Zotero MCP.

This module provides semantic search capabilities by integrating ChromaDB
with the existing Zotero client to enable vector-based similarity search
over research libraries.
"""

import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

from pyzotero import zotero
from dotenv import load_dotenv

from .chroma_client import ChromaClient, create_chroma_client
from .client import get_zotero_client
from .utils import format_creators, is_local_mode
from .local_db import LocalZoteroReader, get_local_zotero_reader
from .markdown_processor import chunk_markdown
from .md_store import MarkdownStore
from .locator_store import LocatorStore

logger = logging.getLogger(__name__)

# Weight for meta chunk scores in score fusion.
# Meta chunks (title, abstract, authors) are weighted slightly lower than content
# because they tend to have higher baseline similarity scores. The 0.85 factor
# balances relevance from both meta and content chunks.
META_CHUNK_SCORE_WEIGHT = 0.85


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout temporarily."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


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
        load_dotenv()
        self.chroma_client = chroma_client or create_chroma_client(config_path)
        self.zotero_client = get_zotero_client()
        self.config_path = config_path
        self.db_path = db_path  # CLI override for Zotero database path
        self.semantic_config = self._load_semantic_config()
        self.extraction_mode = self._resolve_extraction_mode(self.semantic_config)
        self.mineru_config = self._resolve_mineru_config(self.semantic_config.get("mineru", {}))
        self.meta_chunk_enabled = self.extraction_mode == "mineru"

        # Lazily/conditionally initialize stores to avoid creating files
        # (e.g., ~/.config/zotero-mcp/md_store, locator.db) when not needed.
        self.md_store: MarkdownStore | None = None
        self.locator_store: LocatorStore | None = None

        # Initialize stores only when MinerU mode is enabled
        if self.meta_chunk_enabled:
            self.md_store = MarkdownStore(self.semantic_config.get("md_store", {}).get("base_dir"))
            self.locator_store = LocatorStore(self.semantic_config.get("locator_db", {}).get("path"))

        # Load update configuration
        self.update_config = self._load_update_config()

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

    def _load_semantic_config(self) -> dict[str, Any]:
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    file_config = json.load(f)
                return file_config.get("semantic_search", {}) or {}
            except Exception as e:
                logger.warning(f"Error loading semantic config: {e}")
        return {}

    def _resolve_extraction_mode(self, semantic_cfg: dict[str, Any]) -> str:
        mode = str((semantic_cfg or {}).get("extraction_mode", "")).strip().lower()
        if mode in {"local", "mineru"}:
            return mode
        # Backward compatibility for old configs.
        mineru_cfg = (semantic_cfg or {}).get("mineru", {}) or {}
        if mineru_cfg.get("enabled"):
            return "mineru"
        return "local"

    def _resolve_mineru_config(self, mineru_cfg: dict[str, Any]) -> dict[str, Any]:
        cfg = dict(mineru_cfg or {})
        # Env fallback for local testing.
        env_tokens = os.getenv("MINERU_TOKENS", "").strip()
        single_token = os.getenv("MINERU_TOKEN", "").strip()
        tokens = cfg.get("tokens") or []
        if not tokens and env_tokens:
            tokens = [t.strip() for t in env_tokens.split(",") if t.strip()]
        if not tokens and single_token:
            tokens = [single_token]
        cfg["tokens"] = tokens
        # Default to enabled when tokens are available unless explicitly disabled.
        explicit_enabled = cfg.get("enabled", None)
        if explicit_enabled is None:
            cfg["enabled"] = bool(tokens)
        else:
            cfg["enabled"] = bool(explicit_enabled) and bool(tokens)
        return cfg

    def _save_update_config(self) -> None:
        """Save update configuration to file."""
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
            "attachment_key": data.get("attachmentKey", ""),
        }
        # If local fulltext field exists, add markers so we can filter later
        if data.get("fulltext"):
            metadata["has_fulltext"] = True
            if data.get("fulltextSource"):
                metadata["fulltext_source"] = data.get("fulltextSource")

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

    def _get_items_from_source(self, limit: int | None = None, extract_fulltext: bool = False, chroma_client: ChromaClient | None = None, force_rebuild: bool = False) -> list[dict[str, Any]]:
        """
        Get items from either local database or API.

        Uses local database only when both extract_fulltext=True and is_local_mode().
        Otherwise uses API (faster, metadata-only).

        Args:
            limit: Optional limit on number of items
            extract_fulltext: Whether to extract fulltext content
            chroma_client: ChromaDB client to check for existing documents (None to skip checks)
            force_rebuild: Whether to force extraction even if item exists

        Returns:
            List of items in API-compatible format
        """
        if extract_fulltext and is_local_mode():
            return self._get_items_from_local_db(
                limit,
                extract_fulltext=extract_fulltext,
                chroma_client=chroma_client,
                force_rebuild=force_rebuild
            )
        else:
            return self._get_items_from_api(limit)

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
            zotero_db_path = self.db_path  # CLI override takes precedence
            mineru_cfg = dict(self.mineru_config)
            if self.extraction_mode != "mineru":
                mineru_cfg["enabled"] = False
            # If semantic_search config file exists, prefer its setting
            try:
                if self.config_path and os.path.exists(self.config_path):
                    with open(self.config_path) as _f:
                        _cfg = json.load(_f)
                        semantic_cfg = _cfg.get('semantic_search', {})
                        pdf_max_pages = semantic_cfg.get('extraction', {}).get('pdf_max_pages')
                        # Use config db_path only if no CLI override
                        if not zotero_db_path:
                            zotero_db_path = semantic_cfg.get('zotero_db_path')
                        if semantic_cfg.get("mineru"):
                            mineru_cfg = self._resolve_mineru_config(semantic_cfg.get("mineru", {}))
                        mode = self._resolve_extraction_mode(semantic_cfg)
                        if mode != "mineru":
                            mineru_cfg["enabled"] = False
            except Exception:
                pass

            with suppress_stdout(), LocalZoteroReader(
                db_path=zotero_db_path,
                pdf_max_pages=pdf_max_pages,
                mineru_config=mineru_cfg,
                md_store_base_dir=str(self.md_store.base_dir) if self.extraction_mode == "mineru" else None,
            ) as reader:
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
                    progress_every = 1 if self.extraction_mode == "mineru" else 25

                    for it in local_items:
                        should_extract = True

                        # CHECK IF ITEM ALREADY EXISTS (unless force_rebuild or no client)
                        if chroma_client and not force_rebuild:
                            existing_metadata = chroma_client.get_document_metadata(it.key)
                            if existing_metadata:
                                chroma_has_fulltext = existing_metadata.get("has_fulltext", False)
                                local_has_fulltext = len(reader.get_fulltext_meta_for_item(it.item_id)) > 0

                                # Skip only if chroma does not have the fulltext embedding but local does (e.g. the users updated it)
                                if not chroma_has_fulltext and local_has_fulltext:
                                    # Document exists but lacks fulltext - we need to update it
                                    updated_existing += 1
                                else:
                                    should_extract = False
                                    skipped_existing += 1

                        if should_extract:
                            # Extract fulltext if item doesn't have it yet
                            if not getattr(it, "fulltext", None):
                                text = reader.extract_fulltext_for_item(it.item_id, it.key)
                                if text:
                                    # Support tuple return format:
                                    # (text, source) or (text, source, attachment_key)
                                    if isinstance(text, tuple):
                                        if len(text) >= 2:
                                            it.fulltext, it.fulltext_source = text[0], text[1]
                                        if len(text) >= 3:
                                            setattr(it, "attachment_key", text[2])
                                    else:
                                        it.fulltext = text
                            extracted += 1
                            items_to_process.append(it)

                            if extracted % progress_every == 0 and total_to_extract:
                                try:
                                    sys.stderr.write(
                                        f"Fulltext extraction progress: {extracted}/{total_to_extract} "
                                        f"(skipped {skipped_existing}, updating {updated_existing})\n"
                                    )
                                except Exception:
                                    pass

                    # Replace local_items with filtered list
                    local_items = items_to_process

                    # Report final stats
                    if skipped_existing > 0 or updated_existing > 0:
                        try:
                            msg_parts = []
                            if skipped_existing > 0:
                                msg_parts.append(f"Skipped {skipped_existing} items with up to date embeddings")
                            if updated_existing > 0:
                                msg_parts.append(f"Updated {updated_existing} items with new fulltext")
                            sys.stderr.write(", ".join(msg_parts) + "\n")
                        except Exception:
                            pass
                    if total_to_extract:
                        try:
                            sys.stderr.write(
                                f"Fulltext extraction completed: {extracted}/{total_to_extract}\n"
                            )
                        except Exception:
                            pass
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
                            "attachmentKey": getattr(item, 'attachment_key', None) or "",
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

    def _get_items_from_api(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get items from Zotero API (original implementation).

        Args:
            limit: Optional limit on number of items

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
                item for item in items
                if item.get("data", {}).get("itemType") not in ["attachment", "note"]
            ]

            all_items.extend(filtered_items)
            start += batch_size

            if len(items) < batch_size:
                break

        if limit:
            all_items = all_items[:limit]

        logger.info(f"Retrieved {len(all_items)} items from API")
        return all_items

    def update_database(self,
                       force_full_rebuild: bool = False,
                       limit: int | None = None,
                       extract_fulltext: bool = False) -> dict[str, Any]:
        """
        Update the semantic search database with Zotero items.

        Args:
            force_full_rebuild: Whether to rebuild the entire database
            limit: Limit number of items to process (for testing)
            extract_fulltext: Whether to extract fulltext content from local database

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
            "skipped_items": 0,
            "errors": 0,
            "start_time": start_time.isoformat(),
            "duration": None
        }

        try:
            # Reset collection if force rebuild
            if force_full_rebuild:
                logger.info("Force rebuilding database...")
                self.chroma_client.reset_collection()
                if self.locator_store is not None:
                    self.locator_store.reset()

            # Get all items from either local DB or API
            all_items = self._get_items_from_source(
                limit=limit,
                extract_fulltext=extract_fulltext,
                chroma_client=self.chroma_client if not force_full_rebuild else None,
                force_rebuild=force_full_rebuild
            )

            stats["total_items"] = len(all_items)
            logger.info(f"Found {stats['total_items']} items to process")
            # Immediate progress line so users see counts up-front
            try:
                sys.stderr.write(f"Total items to index: {stats['total_items']}\n")
            except Exception:
                pass

            # Process items in batches
            batch_size = 50
            # Track next milestone for progress printing (every 10 items)
            next_milestone = 10 if stats["total_items"] >= 10 else stats["total_items"]
            # Count of items seen (including skipped), used for progress milestones
            seen_items = 0
            for i in range(0, len(all_items), batch_size):
                batch = all_items[i:i + batch_size]
                batch_stats = self._process_item_batch(batch, force_full_rebuild)

                stats["processed_items"] += batch_stats["processed"]
                stats["added_items"] += batch_stats["added"]
                stats["updated_items"] += batch_stats["updated"]
                stats["skipped_items"] += batch_stats["skipped"]
                stats["errors"] += batch_stats["errors"]
                seen_items += len(batch)

                logger.info(f"Processed {seen_items}/{stats['total_items']} items (added: {stats['added_items']}, skipped: {stats['skipped_items']})")
                # Print progress every 10 seen items (even if all are skipped)
                try:
                    while seen_items >= next_milestone and next_milestone > 0:
                        sys.stderr.write(f"Processed: {next_milestone}/{stats['total_items']} added:{stats['added_items']} skipped:{stats['skipped_items']} errors:{stats['errors']}\n")
                        next_milestone += 10
                        if next_milestone > stats["total_items"]:
                            next_milestone = stats["total_items"]
                            break
                except Exception:
                    pass

            # Update last update time
            self.update_config["last_update"] = datetime.now().isoformat()
            self._save_update_config()

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

    def _process_item_batch(self, items: list[dict[str, Any]], force_rebuild: bool = False) -> dict[str, int]:
        """Process a batch of items."""
        stats = {"processed": 0, "added": 0, "updated": 0, "skipped": 0, "errors": 0}
        dual_index_mode = self.meta_chunk_enabled

        documents = []
        metadatas = []
        ids = []
        locator_records: list[dict[str, Any]] = []

        for item in items:
            try:
                item_key = item.get("key", "")
                if not item_key:
                    stats["skipped"] += 1
                    continue

                fulltext = item.get("data", {}).get("fulltext", "")
                if dual_index_mode and fulltext.strip():
                    # In dual-index mode, each update rewrites all chunks for the item.
                    try:
                        self.chroma_client.collection.delete(where={"item_key": item_key})  # type: ignore[attr-defined]
                    except Exception as e:
                        logger.warning(
                            "Failed to delete existing chunks for item %s from Chroma collection: %s",
                            item_key,
                            e,
                        )
                        stats["errors"] += 1
                        # Skip further processing for this item to avoid mixing old and new chunks.
                        continue
                    if self.locator_store is not None:
                        self.locator_store.delete_item(item_key)
                    chunk_docs, chunk_meta, chunk_ids, chunk_locators = self._build_chunk_records(item)
                    meta_doc = self._build_metadata_chunk_record(item) if dual_index_mode else None
                    if not chunk_docs and not meta_doc:
                        stats["skipped"] += 1
                        continue
                    if chunk_docs:
                        documents.extend(chunk_docs)
                        metadatas.extend(chunk_meta)
                        ids.extend(chunk_ids)
                        locator_records.extend(chunk_locators)
                    if meta_doc:
                        documents.append(meta_doc[0])
                        metadatas.append(meta_doc[1])
                        ids.append(meta_doc[2])
                    stats["processed"] += 1
                    continue

                # Preserve original item-level indexing behavior outside dual-index mode.
                doc_text = fulltext if fulltext.strip() else self._create_document_text(item)
                if not doc_text.strip():
                    stats["skipped"] += 1
                    continue
                metadata = self._create_metadata(item)
                if dual_index_mode:
                    meta_doc = self._build_metadata_chunk_record(item)
                    if not meta_doc:
                        stats["skipped"] += 1
                        continue
                    documents.append(meta_doc[0])
                    metadatas.append(meta_doc[1])
                    ids.append(meta_doc[2])
                else:
                    documents.append(doc_text)
                    metadatas.append(metadata)
                    ids.append(item_key)

                stats["processed"] += 1

            except Exception as e:
                logger.error(f"Error processing item {item.get('key', 'unknown')}: {e}")
                stats["errors"] += 1

        # Add documents to ChromaDB if any
        if documents:
            try:
                existing_ids: set[str] = set()
                if not force_rebuild and not dual_index_mode:
                    existing_ids = self.chroma_client.get_existing_ids(ids)

                self.chroma_client.upsert_documents(documents, metadatas, ids)
                if dual_index_mode and self.locator_store is not None:
                    self.locator_store.upsert_many(locator_records)
                    stats["added"] += len(documents)
                else:
                    for doc_id in ids:
                        if doc_id in existing_ids:
                            stats["updated"] += 1
                        else:
                            stats["added"] += 1
            except Exception as e:
                logger.error(f"Error adding documents to ChromaDB: {e}")
                stats["errors"] += len(documents)

        return stats

    def _build_metadata_chunk_record(
        self, item: dict[str, Any]
    ) -> tuple[str, dict[str, Any], str] | None:
        item_key = item.get("key", "")
        if not item_key:
            return None
        doc_text = self._create_document_text(item)
        if not doc_text.strip():
            return None
        meta = self._create_metadata(item)
        meta.update(
            {
                "chunk_kind": "meta",
                "chunk_index": 0,
                "attachment_key": "meta",
                "source_type": "metadata",
                "section_path": "metadata",
                "char_start": 0,
                "char_end": len(doc_text),
                "page_start": -1,
                "page_end": -1,
            }
        )
        chunk_id = f"{item_key}:meta:0"
        return doc_text, meta, chunk_id

    def _build_chunk_records(
        self, item: dict[str, Any]
    ) -> tuple[list[str], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
        # Defensive check: ensure md_store is initialized
        if self.md_store is None:
            return [], [], [], []

        data = item.get("data", {})
        item_key = item.get("key", "")
        attachment_key = data.get("attachmentKey", "default")
        fulltext = data.get("fulltext", "") or ""
        source = data.get("fulltextSource", "") or "pdf"
        base_meta = self._create_metadata(item)

        if not fulltext.strip():
            return [], [], [], []

        md_path = ""
        md_hash = ""
        # md_store.write() computes the hash internally, so no need to compute it here
        if source == "mineru_md":
            md_path, md_hash = self.md_store.write(item_key, attachment_key, fulltext)
            chunks = chunk_markdown(fulltext)
        else:
            chunks = chunk_markdown(fulltext)
            md_path, md_hash = self.md_store.write(item_key, attachment_key, fulltext)

        if not chunks:
            return [], [], [], []

        docs: list[str] = []
        metas: list[dict[str, Any]] = []
        ids: list[str] = []
        locators: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{item_key}:{attachment_key}:{idx}"
            chunk_text = chunk.text.strip()
            if not chunk_text:
                continue
            preview = chunk_text[:500]
            meta = dict(base_meta)
            meta.update(
                {
                    "chunk_kind": "content",
                    "chunk_index": idx,
                    "attachment_key": attachment_key,
                    "source_type": source,
                    "section_path": chunk.section_path,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "page_start": chunk.page_start or -1,
                    "page_end": chunk.page_end or -1,
                    "md_hash": md_hash,
                }
            )
            docs.append(preview)
            metas.append(meta)
            ids.append(chunk_id)
            locators.append(
                {
                    "chunk_id": chunk_id,
                    "item_key": item_key,
                    "attachment_key": attachment_key,
                    "md_store_path": md_path,
                    "char_start": int(chunk.char_start),
                    "char_end": int(chunk.char_end),
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section_path": chunk.section_path,
                    "md_hash": md_hash,
                    "pdf_hash": "",
                }
            )
        return docs, metas, ids, locators

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
            # Perform semantic search
            results = self.chroma_client.search(
                query_texts=[query],
                n_results=limit,
                where=filters
            )

            # Enrich results with full Zotero item data
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

    def _enrich_search_results(self, chroma_results: dict[str, Any], query: str) -> list[dict[str, Any]]:
        """Enrich ChromaDB results with full Zotero item data."""
        enriched: list[dict[str, Any]] = []

        if not chroma_results.get("ids") or not chroma_results["ids"][0]:
            return enriched

        ids = chroma_results["ids"][0]
        distances = chroma_results.get("distances", [[]])[0]
        documents = chroma_results.get("documents", [[]])[0]
        metadatas = chroma_results.get("metadatas", [[]])[0]

        # Preserve the original behavior for legacy item-level documents.
        has_chunk_docs = any(":" in str(doc_id) for doc_id in ids)
        if not has_chunk_docs and not self.meta_chunk_enabled:
            for i, item_key in enumerate(ids):
                try:
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    raw_id = item_key
                    zotero_item = self.zotero_client.item(item_key)
                    matched_text = documents[i] if i < len(documents) else ""
                    # Only attempt locator enrichment if stores are initialized (MinerU mode)
                    locator = None
                    if self.locator_store and self.md_store:
                        locator = self.locator_store.get(raw_id)
                        if locator:
                            try:
                                md_text = self.md_store.read(locator["md_store_path"])
                                cs = int(locator["char_start"])
                                ce = int(locator["char_end"])
                                matched_text = md_text[max(0, cs): max(cs, ce)].strip() or matched_text
                            except Exception:
                                # Best-effort enrichment from markdown; fall back to original matched_text.
                                logger.debug(
                                    "Failed to read or apply locator for %s", raw_id, exc_info=True
                                )

                    enriched.append(
                        {
                            "item_key": item_key,
                            "chunk_id": raw_id,
                            "similarity_score": 1 - distances[i] if i < len(distances) else 0,
                            "matched_text": matched_text,
                            "metadata": metadata,
                            "locator": locator or {},
                            "zotero_item": zotero_item,
                            "query": query,
                        }
                    )
                except Exception as e:
                    logger.error(f"Error enriching result for item {item_key}: {e}")
                    enriched.append(
                        {
                            "item_key": item_key,
                            "similarity_score": 1 - distances[i] if i < len(distances) else 0,
                            "matched_text": documents[i] if i < len(documents) else "",
                            "metadata": metadatas[i] if i < len(metadatas) else {},
                            "query": query,
                            "error": f"Could not fetch full item data: {e}",
                        }
                    )
            return enriched

        grouped_hits: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "content_hits": [],
                "meta_hits": [],
                "best_content_score": 0.0,
                "best_meta_score": 0.0,
            }
        )

        for i, raw_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            item_key = metadata.get("item_key") or str(raw_id).split(":", 1)[0]
            score = 1 - distances[i] if i < len(distances) else 0
            matched_text = documents[i] if i < len(documents) else ""
            chunk_kind = metadata.get("chunk_kind", "meta" if ":meta:" in str(raw_id) else "content")
            locator = None
            if self.locator_store and self.md_store:
                locator = self.locator_store.get(raw_id)
                if locator and chunk_kind == "content":
                    try:
                        md_text = self.md_store.read(locator["md_store_path"])
                        cs = int(locator["char_start"])
                        ce = int(locator["char_end"])
                        matched_text = md_text[max(0, cs): max(cs, ce)].strip() or matched_text
                    except Exception:
                        pass

            hit = {
                "chunk_id": raw_id,
                "similarity_score": score,
                "matched_text": matched_text,
                "metadata": metadata,
                "locator": locator or {},
                "chunk_kind": chunk_kind,
            }
            if chunk_kind == "meta":
                grouped_hits[item_key]["meta_hits"].append(hit)
                grouped_hits[item_key]["best_meta_score"] = max(grouped_hits[item_key]["best_meta_score"], score)
            else:
                grouped_hits[item_key]["content_hits"].append(hit)
                grouped_hits[item_key]["best_content_score"] = max(grouped_hits[item_key]["best_content_score"], score)

        for item_key, bucket in grouped_hits.items():
            try:
                zotero_item = self.zotero_client.item(item_key)
                content_hits = sorted(bucket["content_hits"], key=lambda x: x["similarity_score"], reverse=True)
                meta_hits = sorted(bucket["meta_hits"], key=lambda x: x["similarity_score"], reverse=True)
                best_content = float(bucket["best_content_score"])
                best_meta = float(bucket["best_meta_score"])
                item_score = max(best_content, META_CHUNK_SCORE_WEIGHT * best_meta)
                primary = content_hits[0] if content_hits else (meta_hits[0] if meta_hits else None)
                if not primary:
                    continue
                enriched.append(
                    {
                        "item_key": item_key,
                        "chunk_id": primary["chunk_id"],
                        "similarity_score": item_score,
                        "matched_text": primary["matched_text"],
                        "metadata": primary["metadata"],
                        "locator": primary["locator"],
                        "zotero_item": zotero_item,
                        "query": query,
                        "content_evidence": content_hits[:2],
                        "meta_evidence": meta_hits[:1],
                    }
                )

            except Exception as e:
                logger.error(f"Error enriching result for item {item_key}: {e}")
                enriched.append({
                    "item_key": item_key,
                    "similarity_score": max(
                        float(bucket["best_content_score"]),
                        META_CHUNK_SCORE_WEIGHT * float(bucket["best_meta_score"]),
                    ),
                    "matched_text": "",
                    "metadata": {},
                    "query": query,
                    "error": f"Could not fetch full item data: {e}"
                })

        enriched.sort(key=lambda r: float(r.get("similarity_score", 0)), reverse=True)
        return enriched

    def get_database_status(self) -> dict[str, Any]:
        """Get status information about the semantic search database."""
        collection_info = self.chroma_client.get_collection_info()

        return {
            "collection_info": collection_info,
            "update_config": self.update_config,
            "should_update": self.should_update_database(),
            "last_update": self.update_config.get("last_update"),
            "locator_count": self.locator_store.count() if self.locator_store else 0,
            "md_store_dir": str(self.md_store.base_dir) if self.md_store else "",
            "extraction_mode": self.extraction_mode,
            "meta_chunk_enabled": self.meta_chunk_enabled,
            "mineru_enabled": bool(self.mineru_config.get("enabled", False)),
            "mineru_token_count": len(self.mineru_config.get("tokens", [])),
        }

    def delete_item(self, item_key: str) -> bool:
        """Delete an item from the semantic search database."""
        success = False
        try:
            try:
                # Primary deletion: delete all chunks/documents with matching item_key metadata.
                self.chroma_client.collection.delete(where={"item_key": item_key})  # type: ignore[attr-defined]
                success = True
            except Exception as e:
                # The metadata-filter delete can fail in some configurations (e.g. chunk mode)
                # because document IDs are like "ITEM:attachment:0", not just "ITEM".
                logger.warning(
                    "Failed to delete chunks for item %s via metadata filter: %s; "
                    "this may be expected in chunk mode where document IDs differ from item keys.",
                    item_key,
                    e,
                )
                # Fallback for non-chunk (legacy) documents: delete by document ID.
                try:
                    self.chroma_client.delete_documents([item_key])
                    success = True
                except Exception as e_fallback:
                    logger.warning(
                        "Fallback deletion by document ID also failed for item %s: %s",
                        item_key,
                        e_fallback,
                    )
            if self.locator_store is not None:
                self.locator_store.delete_item(item_key)
            return success
        except Exception as e:
            logger.error(f"Error deleting item {item_key}: {e}")
            return False

    def close(self) -> None:
        """Close database connections and release resources."""
        if self.locator_store is not None:
            self.locator_store.close()

    def __enter__(self) -> "ZoteroSemanticSearch":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


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
