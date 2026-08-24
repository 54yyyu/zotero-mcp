"""Config reads that must not pull ChromaDB in.

Everything here answers "is the expensive thing worth doing?" from the config
file alone. It lives outside ``semantic_search`` on purpose: that module
imports ChromaDB (and, transitively, numpy) at import time, so asking the
question from there costs the whole heavy dependency chain even when the
answer is no. On Windows that import, running in a startup worker thread,
wedges the process for the length of the first tool call (#485).

There were two such gates and both had the import above the check rather than
below it: the auto-update check in ``_app.server_lifespan``, and the reranker
warmup in ``cli.serve``. Anything else that decides whether to do semantic
work belongs here too, for the same reason.

Nothing here does I/O beyond reading the config JSON, and nothing here imports
a third-party package. ``config.py`` holds the typed, validated view of the
same file; this is the cheap one, for the paths that run before the decision
to load anything heavy has been made.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_UPDATE_CONFIG: dict[str, Any] = {
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


_DEFAULT_RERANKER_CONFIG: dict[str, Any] = {
    "enabled": False,
    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "candidate_multiplier": 3,
}


def load_reranker_config(config_path: str | None) -> dict[str, Any]:
    """Read the semantic-search ``reranker`` block from disk.

    Pure file read with no model load, so the server can consult it (e.g. to
    decide whether to warm up) without paying the cross-encoder cost.
    """
    config = dict(_DEFAULT_RERANKER_CONFIG)
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                file_config = json.load(f)
            config.update(file_config.get("semantic_search", {}).get("reranker", {}))
        except Exception as e:
            logger.warning(f"Error loading reranker config: {e}")
    return config


def reranker_enabled(config_path: str | None) -> bool:
    """Whether a reranker warmup is worth doing, from the config alone.

    The gate ``warmup_reranker`` applies internally, hoisted so a caller can
    apply it *before* importing ``semantic_search`` rather than after (#485).
    """
    return bool(load_reranker_config(config_path).get("enabled", False))
