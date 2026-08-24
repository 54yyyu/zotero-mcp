"""Whether a semantic-search auto-update is due, decided from config alone.

This lives outside ``semantic_search`` on purpose. That module imports
ChromaDB (and, transitively, numpy) at import time, so asking "is an update
due?" from there costs the whole heavy dependency chain even when the answer
is no. On Windows that import, running in the lifespan's worker thread, wedges
the process for the length of the first tool call (#485).

Nothing here does I/O beyond reading the config JSON, and nothing here imports
a third-party package.
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
