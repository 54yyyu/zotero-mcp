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

    ``state`` is ``up_to_date``, ``stale``, ``ahead`` (the saved version is
    ahead of the library's), ``no_record`` (the config holds no
    saved version for this library) or ``unknown`` (the check failed; ``detail``
    says why).
    """

    state: str
    saved: int | None = None
    current: int | None = None
    detail: str = ""
    group_id: int | None = None


def _read_semantic_section(config_path: str | None) -> dict[str, Any]:
    if not config_path or not os.path.exists(config_path):
        return {}
    try:
        with open(config_path) as f:
            return (json.load(f).get("semantic_search") or {}) or {}
    except Exception as e:
        logger.warning(f"Could not read {config_path} for the freshness check: {e}")
        return {}


def _env_default_group_id() -> int:
    if os.getenv("ZOTERO_LIBRARY_TYPE", "user") == "group":
        try:
            return int(os.getenv("ZOTERO_LIBRARY_ID") or 0)
        except (TypeError, ValueError):
            return PERSONAL_GROUP_ID
    return PERSONAL_GROUP_ID


def saved_sync_version(config_path: str | None, group_id: int) -> int | None:
    """Library version recorded at the last completed update, or None.

    Mirrors ``_load_last_sync_version``: once the per-library map exists it is
    authoritative; before that, the pre-per-library scalar belongs to the
    env-configured default library only.
    """
    section = _read_semantic_section(config_path)
    versions = section.get("last_sync_versions")
    raw: Any
    if isinstance(versions, dict):
        raw = versions.get(str(group_id))
    elif group_id == _env_default_group_id():
        # Same provenance rule as update's _migrate_legacy_sync_version.
        raw = section.get("last_sync_version")
    else:
        raw = None
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _configured_db_path(config_path: str | None) -> str | None:
    """The Zotero database path ``update_database`` uses: the config's
    ``semantic_search.zotero_db_path`` (None means auto-detect)."""
    return _read_semantic_section(config_path).get("zotero_db_path") or None


#: Seconds the version request may take; a hung Zotero must not block status.
API_TIMEOUT = 5.0


def _api_library_version() -> int:
    """The library's current version: the ``Last-Modified-Version`` header of
    ``items?limit=1``, which is what ``last_modified_version()`` reads in
    ``update_database`` (the local API in local mode, the web API otherwise).

    Sent through the client's own HTTP session rather than through pyzotero,
    whose ``_retrieve_data`` passes its own 30 s timeout on every request and
    retries a 429 up to three times with backoff: neither fits a status call.
    One request, ``API_TIMEOUT`` seconds, no retry.
    """
    from zotero_mcp.client import get_zotero_client

    zot = get_zotero_client()
    url = f"{zot.endpoint}/{zot.library_type}/{zot.library_id}/items"
    response = zot.client.get(url, params={"limit": 1, "format": "json"}, timeout=API_TIMEOUT)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code} from {zot.endpoint}")
    return int(response.headers["last-modified-version"])


def _local_file_version(group_id: int, db_path: str | None) -> tuple[int, bool]:
    """(version, complete) read from the Zotero database file. ``complete`` is
    False when the read cannot see changes still in Zotero's write-ahead log."""
    from zotero_mcp.local_db import LocalZoteroReader

    with LocalZoteroReader(db_path=db_path) as reader:
        version = reader.get_library_version(group_id)
        complete = reader.read_reflects_wal()
    if version is None:
        raise LookupError(f"library {group_id} is not in the local Zotero database")
    return version, complete


def check_index_staleness(
    config_path: str | None,
    *,
    group_id: int,
    local: bool,
    db_path: str | None = None,
) -> Staleness:
    """Compare the saved sync version for ``group_id`` with the library's now.

    The current version comes from the API call ``update_database`` makes. In
    local mode, if the local API is unreachable, the Zotero database file is
    read instead; that read is trusted to say "same version" only when it saw
    the write-ahead log too, because a version committed there is invisible to
    a read in place.
    """
    result = _check(config_path, group_id=group_id, local=local, db_path=db_path)
    result.group_id = group_id
    return result


def _check(config_path: str | None, *, group_id: int, local: bool, db_path: str | None) -> Staleness:
    saved = saved_sync_version(config_path, group_id)
    if saved is None:
        return Staleness("no_record", detail="no sync version is saved for this library")
    complete = True
    try:
        try:
            current = _api_library_version()
        except Exception as api_error:
            if not local:
                raise
            try:
                current, complete = _local_file_version(
                    group_id, db_path or _configured_db_path(config_path)
                )
            except Exception as file_error:
                raise RuntimeError(
                    f"the local API did not answer ({type(api_error).__name__}: {api_error}) "
                    f"and the database file could not be read ({type(file_error).__name__}: {file_error})"
                ) from None
    except Exception as e:
        return Staleness("unknown", saved=saved, detail=f"{type(e).__name__}: {e}")

    if current <= 0:
        return Staleness(
            "unknown",
            saved=saved,
            current=current,
            detail="the library has no version yet (it has not synced with zotero.org)",
        )
    if current > saved:
        return Staleness("stale", saved=saved, current=current)
    if not complete:
        return Staleness(
            "unknown",
            saved=saved,
            current=current,
            detail=(
                "the local API did not answer, and the database file may hold newer changes "
                "in its write-ahead log that a direct read cannot see"
            ),
        )
    if current == saved:
        return Staleness("up_to_date", saved=saved, current=current)
    return Staleness("ahead", saved=saved, current=current)


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


def active_group_id() -> int | None:
    try:
        from zotero_mcp.client import get_active_group_id

        return get_active_group_id()
    except Exception:
        return None


def index_problem(collection_info: dict[str, Any]) -> str | None:
    """Why the saved library version says nothing about this ChromaDB
    collection (empty, unreadable, not initialized), or None if it is usable."""
    if collection_info.get("unreadable"):
        return "the ChromaDB collection is unreadable"
    if collection_info.get("error"):
        return "the ChromaDB collection could not be read"
    if collection_info.get("initialized") is False:
        return "the ChromaDB collection is not initialized"
    if not collection_info.get("count"):
        return "the ChromaDB collection is empty"
    return None


def _library_label(group_id: int | None) -> str:
    if group_id is None:
        return "library unknown"
    return "personal library" if group_id == PERSONAL_GROUP_ID else f"group {group_id}"


def format_staleness(
    result: Staleness, update_command: str, index_problem: str | None = None
) -> str:
    """One paragraph for the status output. ``update_command`` is what to run;
    ``index_problem`` (from :func:`index_problem`) overrides the verdict."""
    where = f" Checked: the active library ({_library_label(result.group_id)}); other libraries are not checked."
    if index_problem:
        return (
            f"the saved library version is not meaningful because {index_problem}. "
            f"Run `{update_command}`." + where
        )
    if result.state == "up_to_date":
        body = f"up to date (library version {result.current}, same as at the last update)."
    elif result.state == "stale":
        body = (
            f"STALE. The library has changed since the index was last updated "
            f"(indexed at library version {result.saved}, now {result.current}). "
            f"Semantic search misses what changed. Run `{update_command}` to bring it up to date."
        )
    elif result.state == "ahead":
        body = (
            f"STALE. The saved version ({result.saved}) is ahead of the library's "
            f"({result.current}), so it did not come from this library; the next update does "
            f"a full scan. Run `{update_command}`."
        )
    elif result.state == "no_record":
        body = (
            "no update has been recorded for this library, so the index may be empty or "
            f"cover a different library. Run `{update_command}`."
        )
    else:
        body = (
            f"couldn't check ({result.detail}). Do not assume the index is current; "
            f"`{update_command}` refreshes it."
        )
    return body + where


def status_lines(
    config_path: str | None,
    update_command: str,
    import_command: str,
    status_command: str,
    collection_info: dict[str, Any] | None = None,
) -> list[str]:
    """Plain-text lines for the CLI status commands."""
    lines = [
        "Index freshness: "
        + format_staleness(
            check_active_library(config_path),
            update_command,
            index_problem(collection_info) if collection_info is not None else None,
        )
    ]
    lines += [
        f"Unfinished batch: {line}"
        for line in format_unfinished_batches(
            unfinished_batch_runs(config_path, active_group_id()), import_command, status_command
        )
    ]
    return lines


def _batch_dirs(config_path: str | None) -> Path:
    if config_path:
        return Path(config_path).expanduser().parent
    return Path.home() / ".config" / "zotero-mcp"


_DEAD_STATES = frozenset({"failed", "expired", "cancelled"})
_DEAD_RAW_STATUSES = {
    "failed": "failed",
    "expired": "expired",
    "cancelled": "cancelled",
    "cancelling": "cancelled",
    "JOB_STATE_FAILED": "failed",
    "JOB_STATE_EXPIRED": "expired",
    "JOB_STATE_CANCELLED": "cancelled",
}


def _recorded_state(provider: str, entry: dict[str, Any]) -> str:
    """The chunk's state as the manifest recorded it; no provider call."""
    state = entry.get("state")
    if not state and entry.get("status"):
        # The raw statuses the two provider modules map to a dead state. Kept
        # here because importing those modules loads chromadb (~0.3 s).
        state = _DEAD_RAW_STATUSES.get(str(entry["status"]), "")
    return str(state or "")


def unfinished_batch_runs(config_path: str | None, group_id: int | None = None) -> list[dict[str, Any]]:
    """Newest batch run of ``group_id`` per provider that still has chunks not
    imported (any library's newest run when ``group_id`` is None).

    Reads manifests only: no provider call, and no directory is created.
    ``dead`` counts the unimported chunks the manifest records as failed,
    expired or cancelled, which ``batch-import`` cannot import.
    """
    from zotero_mcp.batch_common import find_manifest, newest_manifest_for_group

    found: list[dict[str, Any]] = []
    base = _batch_dirs(config_path)
    for provider in BATCH_PROVIDERS:
        root = base / f"{provider}_batches"
        if not root.is_dir():
            continue
        if group_id is None:
            try:
                manifest = find_manifest(root, provider_label=provider)
            except FileNotFoundError:
                continue
        else:
            manifest = newest_manifest_for_group(root, group_id)
            if manifest is None:
                continue
        entries = manifest.get("batches", [])
        pending = [b for b in entries if not b.get("imported_at")]
        if entries and pending:
            found.append(
                {
                    "provider": provider,
                    "run_id": manifest.get("run_id"),
                    "total": len(entries),
                    "imported": len(entries) - len(pending),
                    "dead": sum(1 for b in pending if _recorded_state(provider, b) in _DEAD_STATES),
                }
            )
    return found


def format_unfinished_batches(
    found: list[dict[str, Any]],
    import_command: str = "zotero-mcp batch-import",
    status_command: str = "zotero-mcp batch-status",
) -> list[str]:
    lines = []
    for b in found:
        left = b["total"] - b["imported"]
        live = left - b["dead"]
        head = f"{b['provider']} batch run {b['run_id']}: "
        if live:
            lines.append(
                f"{head}{left} of {b['total']} chunks not yet imported into the index "
                f"(from the local manifest; provider state not checked). "
                f"Run `{import_command}` to import the finished ones."
            )
        if b["dead"]:
            lines.append(
                f"{head}{b['dead']} chunk(s) failed, expired or were cancelled and cannot be "
                f"imported. See `{status_command}`."
            )
    return lines
