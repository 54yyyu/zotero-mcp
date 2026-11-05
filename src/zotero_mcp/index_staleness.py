"""Is the semantic index behind the library, and is a batch run unfinished?

Both answers come cheaply and without the embedding stack: the library's
current version is one SQLite read (local mode) or one API call (web mode),
and an unfinished batch run is read from its manifest on disk. Nothing here
loads an embedding model or a vector index, and nothing takes the Zotero API
lock, so the status tools can show it on every call.

The saved side is the ``last_sync_versions`` map ``update_database`` writes
into the config, one entry per library. A check that cannot be made says so
(state ``unknown``); it never reports "up to date" by default.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PERSONAL_GROUP_ID = 0
BATCH_PROVIDERS = ("openai", "gemini")


@dataclass
class Staleness:
    """Outcome of one freshness check.

    ``state`` is ``up_to_date``, ``stale``, ``no_record`` (the config holds no
    saved version for this library) or ``unknown`` (the check failed; ``detail``
    says why).
    """

    state: str
    saved: int | None = None
    current: int | None = None
    detail: str = ""


def _read_semantic_section(config_path: str | None) -> dict[str, Any]:
    if not config_path or not os.path.exists(config_path):
        return {}
    try:
        with open(config_path) as f:
            return (json.load(f).get("semantic_search") or {}) or {}
    except Exception as e:
        logger.warning(f"Could not read {config_path} for the freshness check: {e}")
        return {}


def saved_sync_version(config_path: str | None, group_id: int) -> int | None:
    """Library version recorded at the last completed update, or None.

    Mirrors ``_load_last_sync_version``: once the per-library map exists it is
    authoritative; before that, the pre-per-library scalar belongs to the
    personal library only.
    """
    section = _read_semantic_section(config_path)
    versions = section.get("last_sync_versions")
    raw: Any
    if isinstance(versions, dict):
        raw = versions.get(str(group_id))
    elif group_id == PERSONAL_GROUP_ID:
        raw = section.get("last_sync_version")
    else:
        raw = None
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _configured_db_path(config_path: str | None) -> str | None:
    try:
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                data = json.load(f)
            return data.get("zotero_db_path") or (data.get("semantic_search") or {}).get("zotero_db_path")
    except Exception:
        pass
    return None


def _local_library_version(group_id: int, db_path: str | None) -> int:
    from zotero_mcp.local_db import LocalZoteroReader

    with LocalZoteroReader(db_path=db_path) as reader:
        version = reader.get_library_version(group_id)
    if version is None:
        raise LookupError(f"library {group_id} is not in the local Zotero database")
    return version


def _web_library_version() -> int:
    from zotero_mcp.client import get_zotero_client

    return int(get_zotero_client().last_modified_version())


def check_index_staleness(
    config_path: str | None,
    *,
    group_id: int,
    local: bool,
    db_path: str | None = None,
) -> Staleness:
    """Compare the saved sync version for ``group_id`` with the library's now."""
    saved = saved_sync_version(config_path, group_id)
    if saved is None:
        return Staleness("no_record", detail="no sync version is saved for this library")
    try:
        if local:
            current = _local_library_version(group_id, db_path or _configured_db_path(config_path))
        else:
            current = _web_library_version()
    except Exception as e:
        return Staleness("unknown", saved=saved, detail=f"{type(e).__name__}: {e}")

    if current <= 0:
        return Staleness(
            "unknown",
            saved=saved,
            current=current,
            detail="the library has no version yet (it has not synced with zotero.org)",
        )
    if current == saved:
        return Staleness("up_to_date", saved=saved, current=current)
    return Staleness("stale", saved=saved, current=current)


def check_active_library(config_path: str | None) -> Staleness:
    """:func:`check_index_staleness` for the active library and mode; any
    failure, including reading which library is active, becomes ``unknown``."""
    try:
        from zotero_mcp.client import get_active_group_id
        from zotero_mcp.utils import is_local_mode

        return check_index_staleness(
            config_path, group_id=get_active_group_id(), local=is_local_mode()
        )
    except Exception as e:
        return Staleness("unknown", detail=f"{type(e).__name__}: {e}")


def status_lines(config_path: str | None, update_command: str, import_command: str) -> list[str]:
    """Plain-text lines for the CLI status commands."""
    lines = [
        "Index freshness: "
        + format_staleness(check_active_library(config_path), update_command)
    ]
    lines += [
        f"Unfinished batch: {line}"
        for line in format_unfinished_batches(unfinished_batch_runs(config_path), import_command)
    ]
    return lines


def format_staleness(result: Staleness, update_command: str) -> str:
    """One paragraph for the status output. ``update_command`` is what to run."""
    if result.state == "up_to_date":
        return f"up to date (library version {result.current}, same as at the last update)."
    if result.state == "stale":
        return (
            f"STALE. The library has changed since the index was last updated "
            f"(indexed at library version {result.saved}, now {result.current}). "
            f"Semantic search misses what changed. Run `{update_command}` to bring it up to date."
        )
    if result.state == "no_record":
        return (
            "no update has been recorded for this library, so the index may be empty or "
            f"cover a different library. Run `{update_command}`."
        )
    return (
        f"couldn't check ({result.detail}). Do not assume the index is current; "
        f"`{update_command}` refreshes it."
    )


def _batch_dirs(config_path: str | None) -> Path:
    if config_path:
        return Path(config_path).expanduser().parent
    return Path.home() / ".config" / "zotero-mcp"


def unfinished_batch_runs(config_path: str | None) -> list[dict[str, Any]]:
    """Newest batch run per provider that still has chunks not imported.

    Reads manifests only: no provider call, and no directory is created.
    """
    from zotero_mcp.batch_common import find_manifest

    found: list[dict[str, Any]] = []
    base = _batch_dirs(config_path)
    for provider in BATCH_PROVIDERS:
        root = base / f"{provider}_batches"
        if not root.is_dir():
            continue
        try:
            manifest = find_manifest(root, provider_label=provider)
        except FileNotFoundError:
            continue
        entries = manifest.get("batches", [])
        imported = sum(1 for b in entries if b.get("imported_at"))
        if entries and imported < len(entries):
            found.append(
                {
                    "provider": provider,
                    "run_id": manifest.get("run_id"),
                    "total": len(entries),
                    "imported": imported,
                }
            )
    return found


def format_unfinished_batches(
    found: list[dict[str, Any]], import_command: str = "zotero-mcp batch-import"
) -> list[str]:
    return [
        f"{b['provider']} batch run {b['run_id']}: {b['total'] - b['imported']} of {b['total']} "
        f"chunks not yet imported into the index (from the local manifest; provider state not "
        f"checked). Run `{import_command}` to import finished chunks."
        for b in found
    ]
