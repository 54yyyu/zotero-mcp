"""Typed configuration for zotero-mcp.

The server configuration lives at ``~/.config/zotero-mcp/config.json``.
This module defines stdlib dataclasses for every section and a single
loader that deserializes the JSON file into a typed ``ZoteroMcpConfig``
instance (using pydantic's ``TypeAdapter`` for validation / defaults).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

ZOTERO_MCP_CONFIG_PATH = Path.home() / ".config" / "zotero-mcp" / "config.json"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Semantic-search sub-sections
# ---------------------------------------------------------------------------


@dataclass
class ExtractionConfig:
    pdf_max_pages: int | None = None
    fulltext_display_max_pages: int | None = None


@dataclass
class UpdateConfig:
    auto_update: bool = False
    update_frequency: str = "manual"
    update_days: int | None = None
    last_update: str | None = None


@dataclass
class OpenAIBatchConfig:
    enabled: bool = False


@dataclass
class RerankerConfig:
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidate_multiplier: int = 3


@dataclass
class ChunkingConfig:
    enabled: bool = False
    chunk_size: int = 1500
    overlap: int = 200
    max_chunks_per_item: int = 20


@dataclass
class SemanticSearchConfig:
    embedding_model: str = "default"
    embedding_config: dict[str, Any] = field(default_factory=dict)
    openai_batch: OpenAIBatchConfig = field(default_factory=OpenAIBatchConfig)
    update_config: UpdateConfig = field(default_factory=UpdateConfig)
    zotero_db_path: str | None = None
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    include_fulltext: bool = True
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    # Per-library incremental-sync watermarks, keyed by group_id as a string
    # ("0" = personal library). Every Zotero library has its own independent
    # version counter, so one shared scalar corrupted sync state whenever the
    # active library changed (#393).
    last_sync_versions: dict[str, int] = field(default_factory=dict)
    # Legacy pre-#393 scalar. Kept for backward compatibility: it is migrated
    # on read and mirrored on write for the personal library only.
    last_sync_version: int = 0


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class ZoteroMcpConfig:
    zotero_db_path: str | None = None
    client_env: dict[str, str] = field(default_factory=dict)
    semantic_search: SemanticSearchConfig = field(
        default_factory=SemanticSearchConfig,
    )

    def resolve_zotero_db_path(self) -> str | None:
        """Return the Zotero database path with correct precedence.

        Checks top-level ``zotero_db_path`` first, then falls back to
        ``semantic_search.zotero_db_path`` for backward compatibility.
        Returns ``None`` when neither is set, letting
        ``LocalZoteroReader._find_zotero_db()`` auto-detect.
        """
        return self.zotero_db_path or self.semantic_search.zotero_db_path


_config_adapter: TypeAdapter[ZoteroMcpConfig] = TypeAdapter(ZoteroMcpConfig)


def _strip_nulls(obj: Any) -> Any:
    """Recursively remove keys with ``None`` values from nested dicts.

    Allows ``"extraction": null`` in config without causing a validation
    error — the key is treated as absent and the dataclass default fills in.
    """
    if isinstance(obj, dict):
        return {k: _strip_nulls(v) for k, v in obj.items() if v is not None}
    return obj


def load_config() -> ZoteroMcpConfig:
    """Load and validate ``~/.config/zotero-mcp/config.json``.

    Return a ``ZoteroMcpConfig`` with defaults filled in for any
    missing keys. A missing file, parse error, or validation error
    yields the default (all-empty) config so callers never need to
    guard against ``None``.
    """
    raw: dict[str, Any] = {}
    if ZOTERO_MCP_CONFIG_PATH.exists():
        try:
            with open(ZOTERO_MCP_CONFIG_PATH, encoding="utf-8") as f:
                raw = json.load(f) or {}
        except (json.JSONDecodeError, OSError):
            pass
    try:
        return _config_adapter.validate_python(_strip_nulls(raw))
    except ValidationError:
        logger.warning("Invalid zotero-mcp config, falling back to defaults")
        return ZoteroMcpConfig()
