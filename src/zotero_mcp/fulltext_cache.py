"""Transient plain-text cache for extracted fulltext.

Fulltext retrieval is the slow half of ``update-db --fulltext`` (a pdfminer
subprocess per PDF, or a web-API round trip per item). When the later
embedding step fails — rate limits, token limits, network — the extracted
text would otherwise be lost and re-extracted on the next run. This module
persists the plain text to disk keyed by attachment identity, so a rerun
skips straight to embedding. Entries are evicted once their item has been
successfully embedded (realtime mode) or imported (batch mode).

Layout under ``~/.config/zotero-mcp/fulltext_cache/``:
- ``index.json`` — maps cache keys to entry metadata (source, item_key, ...)
- ``<hash>.txt`` — the raw extracted UTF-8 text, one file per entry

Local extraction entries are keyed by attachment key + file mtime/size so a
replaced PDF misses the cache; web-API entries are keyed by item key + Zotero
item version. All index mutations happen under a module lock with atomic
replace, so concurrent extraction workers (see multicore extraction in
semantic_search) can write safely.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_INDEX_LOCK = threading.Lock()

INDEX_VERSION = 1


def _private_chmod(path: Path) -> None:
    try:
        os.chmod(path, 0o600 if path.is_file() else 0o700)
    except OSError:
        pass


def get_fulltext_cache_root(config_path: str | None = None) -> Path:
    """Return the directory used to store cached fulltext."""
    if config_path:
        root = Path(config_path).expanduser().parent / "fulltext_cache"
    else:
        root = Path.home() / ".config" / "zotero-mcp" / "fulltext_cache"
    root.mkdir(parents=True, exist_ok=True)
    _private_chmod(root)
    return root


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _index_path(root: Path) -> Path:
    return root / "index.json"


def _load_index_unlocked(root: Path) -> dict[str, Any]:
    path = _index_path(root)
    if not path.exists():
        return {"version": INDEX_VERSION, "entries": {}}
    try:
        with open(path, encoding="utf-8") as f:
            index = json.load(f)
        if not isinstance(index.get("entries"), dict):
            index["entries"] = {}
        return index
    except Exception as e:
        logger.warning(f"Could not read fulltext cache index ({e}); starting fresh")
        return {"version": INDEX_VERSION, "entries": {}}


def _save_index_unlocked(root: Path, index: dict[str, Any]) -> None:
    path = _index_path(root)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    _private_chmod(tmp)
    os.replace(tmp, path)


def _text_path(root: Path, entry_hash: str) -> Path:
    return root / f"{entry_hash}.txt"


def _get_entry_text(root: Path, entry: dict[str, Any]) -> str | None:
    try:
        return _text_path(root, entry["hash"]).read_text(encoding="utf-8")
    except Exception:
        return None


def _put_entry(root: Path, index_key: str, entry: dict[str, Any], text: str) -> None:
    with _INDEX_LOCK:
        index = _load_index_unlocked(root)
        old = index["entries"].get(index_key)
        text_path = _text_path(root, entry["hash"])
        try:
            text_path.write_text(text, encoding="utf-8")
            _private_chmod(text_path)
        except OSError as e:
            logger.warning(f"Could not write fulltext cache entry for {index_key}: {e}")
            return
        index["entries"][index_key] = entry
        _save_index_unlocked(root, index)
        # Drop a superseded text file (e.g. the PDF changed on disk).
        if old and old.get("hash") != entry["hash"]:
            _text_path(root, old["hash"]).unlink(missing_ok=True)


def get_cached_text(
    attachment_key: str,
    mtime_ns: int,
    size: int,
    profile: str | None = None,
    config_path: str | None = None,
) -> tuple[str, str] | None:
    """Return ``(text, source)`` for an unchanged attachment, or None on miss.

    ``profile`` captures extraction settings that change the output for the
    same file (currently the PDF page cap); a mismatch is a cache miss.
    """
    root = get_fulltext_cache_root(config_path)
    with _INDEX_LOCK:
        entry = _load_index_unlocked(root)["entries"].get(attachment_key)
        if (
            not entry
            or entry.get("mtime_ns") != mtime_ns
            or entry.get("size") != size
            or entry.get("profile") != profile
        ):
            return None
        text = _get_entry_text(root, entry)
    if text is None:
        return None
    return text, entry.get("source", "pdf")


def put_cached_text(
    attachment_key: str,
    mtime_ns: int,
    size: int,
    source: str,
    item_key: str | None,
    text: str,
    path: str | None = None,
    profile: str | None = None,
    config_path: str | None = None,
) -> None:
    """Cache extracted text for a local attachment file."""
    root = get_fulltext_cache_root(config_path)
    entry = {
        "hash": _hash_key(f"{attachment_key}:{mtime_ns}:{size}"),
        "mtime_ns": mtime_ns,
        "size": size,
        "source": source,
        "item_key": item_key,
        "path": path,
        "profile": profile,
        "cached_at": _utc_now(),
    }
    _put_entry(root, attachment_key, entry, text)


def get_cached_web_text(
    item_key: str,
    version: int,
    config_path: str | None = None,
) -> tuple[str, str] | None:
    """Return ``(text, source)`` cached from the web API, or None on miss."""
    root = get_fulltext_cache_root(config_path)
    with _INDEX_LOCK:
        entry = _load_index_unlocked(root)["entries"].get(f"web:{item_key}")
        if not entry or entry.get("version") != version:
            return None
        text = _get_entry_text(root, entry)
    if text is None:
        return None
    return text, entry.get("source", "web-api")


def put_cached_web_text(
    item_key: str,
    version: int,
    source: str,
    text: str,
    config_path: str | None = None,
) -> None:
    """Cache fulltext fetched via the Zotero web API."""
    root = get_fulltext_cache_root(config_path)
    entry = {
        "hash": _hash_key(f"{item_key}:web:{version}"),
        "version": version,
        "source": source,
        "item_key": item_key,
        "cached_at": _utc_now(),
    }
    _put_entry(root, f"web:{item_key}", entry, text)


def evict_many(item_keys: Iterable[str], config_path: str | None = None) -> int:
    """Remove all cache entries belonging to the given item keys.

    Called after the items' embeddings landed in ChromaDB — the cache exists
    only to survive embedding failures. Returns the number of entries removed.
    """
    keys = {k for k in item_keys if k}
    if not keys:
        return 0
    root = get_fulltext_cache_root(config_path)
    removed = 0
    with _INDEX_LOCK:
        index = _load_index_unlocked(root)
        entries = index["entries"]
        for index_key in [k for k, e in entries.items() if e.get("item_key") in keys]:
            entry = entries.pop(index_key)
            _text_path(root, entry.get("hash", "")).unlink(missing_ok=True)
            removed += 1
        if removed:
            _save_index_unlocked(root, index)
    return removed


def evict(item_key: str, config_path: str | None = None) -> int:
    """Remove all cache entries for one item key."""
    return evict_many([item_key], config_path=config_path)


def purge_stale(max_age_days: int = 30, config_path: str | None = None) -> int:
    """Best-effort cleanup guarding against unbounded growth.

    Removes entries older than ``max_age_days`` (an abandoned run) and local
    entries whose backing attachment file no longer exists. Also removes
    orphaned ``.txt`` files that no index entry references.
    """
    root = get_fulltext_cache_root(config_path)
    removed = 0
    now = datetime.now(timezone.utc)
    with _INDEX_LOCK:
        index = _load_index_unlocked(root)
        entries = index["entries"]
        for index_key in list(entries):
            entry = entries[index_key]
            stale = False
            try:
                cached_at = datetime.fromisoformat(entry.get("cached_at", "").replace("Z", "+00:00"))
                if (now - cached_at).days >= max_age_days:
                    stale = True
            except ValueError:
                stale = True
            backing = entry.get("path")
            if backing and not Path(backing).exists():
                stale = True
            if stale:
                entries.pop(index_key)
                _text_path(root, entry.get("hash", "")).unlink(missing_ok=True)
                removed += 1
        referenced = {e.get("hash") for e in entries.values()}
        for orphan in root.glob("*.txt"):
            if orphan.stem not in referenced:
                orphan.unlink(missing_ok=True)
        if removed:
            _save_index_unlocked(root, index)
    return removed


def clear_all(config_path: str | None = None) -> int:
    """Delete every cache entry. Returns the number of entries removed."""
    root = get_fulltext_cache_root(config_path)
    with _INDEX_LOCK:
        index = _load_index_unlocked(root)
        removed = len(index["entries"])
        for orphan in root.glob("*.txt"):
            orphan.unlink(missing_ok=True)
        _index_path(root).unlink(missing_ok=True)
    return removed
