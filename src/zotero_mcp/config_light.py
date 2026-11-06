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


def batch_notice_path(config_path: str) -> str:
    """Where the last batch collect error and deferral reason are kept."""
    return os.path.join(os.path.dirname(os.path.abspath(config_path)), "batch_notices.json")


def load_batch_notice(config_path: str | None, library_key: str) -> dict[str, Any]:
    """What a background batch collect or a deferred update left for the user.

    Background runs have nobody watching their output, so this is how a stuck
    run gets told. ``{}`` when there is nothing to say, or nothing readable.
    """
    if not config_path:
        return {}
    try:
        with open(batch_notice_path(config_path), encoding="utf-8") as f:
            notice = json.load(f).get(str(library_key))
    except (OSError, ValueError, AttributeError):
        return {}
    return notice if isinstance(notice, dict) else {}


def batch_notice_line(notice: dict[str, Any]) -> str | None:
    """One line for the output of a search, or None with nothing to say."""
    messages = [*notice.get("notes", []), *([notice["deferred"]] if notice.get("deferred") else [])]
    if not messages:
        return None
    first = " ".join(str(messages[0]).split())
    if len(first) > 160:
        first = first[:157] + "..."
    return f"{first} See zotero_get_search_database_status."


def auto_update_enabled(update_config: dict[str, Any]) -> bool:
    """Did the user ask for automatic updates, whether or not one is due now?"""
    return bool(update_config.get("auto_update", False)) and update_config.get("update_frequency", "manual") != "manual"


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


#: How long background updates wait after the embedding API was unreachable.
#: A manual ``update-db`` always runs; only the server's own syncs back off.
API_FAILURE_BACKOFF = timedelta(hours=1)


def api_unreachable_message(host: str, reason: str) -> str:
    """The one message for an embedding API that cannot be reached.

    Names the host and what went wrong. For a server on this machine it says
    to check that the server is running; otherwise it suggests the network
    or VPN, because some APIs answer only from particular networks, without
    claiming the request was blocked by address. A 401 leads with the key.
    """
    if host in _LOOPBACK_HOSTS:
        return (
            f"Could not reach the embedding server at {host} ({reason}). "
            "Check that the local server (for example Ollama) is running."
        )
    network = (
        "if this API only accepts requests from a particular network, "
        "connect to it or its VPN"
    )
    if reason == "HTTP 401":
        return (
            f"The embedding API at {host} rejected the API key (HTTP 401). "
            f"Check the API key; {network}."
        )
    if reason == "HTTP 403":
        return (
            f"The embedding API at {host} refused the request (HTTP 403). "
            f"Check your network connection; {network}. "
            "If you are on the right network, check the API key and its permissions."
        )
    return (
        f"Could not reach the embedding API at {host} ({reason}). "
        f"Check your network connection; {network}."
    )


#: Hosts that name this machine; for these, VPN advice would be misleading.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def api_failure_backoff_until(
    update_config: dict[str, Any], now: datetime | None = None
) -> datetime | None:
    """When background updates may resume, or None if they are not held off.

    Reads the ``last_api_failure`` record a realtime update stores when the
    embedding API could not be reached; a successful update removes it.
    """
    failure = update_config.get("last_api_failure")
    if not isinstance(failure, dict):
        return None
    try:
        until = datetime.fromisoformat(str(failure.get("time"))) + API_FAILURE_BACKOFF
    except ValueError:
        return None
    return until if (now or datetime.now()) < until else None


def background_update_due(update_config: dict[str, Any]) -> bool:
    """``should_update``, held off for an hour after the API was unreachable.

    For the server's own background syncs only. ``should_update`` itself is
    unchanged, so status still reports whether an update is due.
    """
    return should_update(update_config) and api_failure_backoff_until(update_config) is None


def describe_api_failure(update_config: dict[str, Any]) -> str | None:
    """One status line for the stored embedding API failure, or None."""
    failure = update_config.get("last_api_failure")
    if not isinstance(failure, dict):
        return None
    line = (
        f"{failure.get('time', 'unknown time')}: could not reach "
        f"{failure.get('host', 'the embedding API')} ({failure.get('reason', 'unknown')})"
    )
    until = api_failure_backoff_until(update_config)
    if until is not None:
        line += f"; background updates paused until {until.isoformat(timespec='seconds')}"
    return line


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


def semantic_search_configured(config_path: str | None) -> bool:
    """Whether this install has semantic search set up at all.

    ``zotero-mcp setup --skip-semantic-search`` writes no ``semantic_search``
    block, and those installs should never pay for ChromaDB — which is the
    whole point of the gates above. Used to decide whether the Windows
    main-thread pre-import in ``cli.serve`` applies (#485).
    """
    if not config_path or not os.path.exists(config_path):
        return False
    try:
        with open(config_path) as f:
            return bool(json.load(f).get("semantic_search"))
    except Exception:
        return False
