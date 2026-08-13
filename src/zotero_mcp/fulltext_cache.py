"""Transient on-disk cache for extracted attachment fulltext.

``update-db --fulltext`` runs in two phases: extract text from every changed
attachment, then embed it. Extraction is cheap per file but not free across a
library — roughly 100 ms per PDF, so a 16,000-item library spends around half
an hour in phase one. Embedding is the phase that fails: rate limits, token
limits, network errors, or the user pressing Ctrl-C. When it does, the
extracted text is lost and the next run pays for phase one all over again.

This module persists that text to disk so a rerun skips straight to embedding.
Entries are evicted once their item's embedding has actually landed in
ChromaDB — the cache exists only to survive the gap between the two phases,
not as a long-lived store.

Layout under ``~/.config/zotero-mcp/fulltext_cache/``:

- ``index.json`` — maps cache keys to entry metadata (source, item_key, ...)
- ``<hash>.txt`` — the extracted UTF-8 text, one file per entry

Concurrency
-----------
Extraction runs in a :class:`~concurrent.futures.ProcessPoolExecutor` (see
``extraction_workers``), but **only the parent process touches this cache**.
Workers receive a file path and return text; the parent writes each entry as
its future completes. Nothing here needs to be process-safe, and a lock in
this module would not have provided that anyway — each worker is a separate
interpreter.

Writing incrementally rather than once at the end is what makes the cache
useful: a run killed part-way through still leaves every completed extraction
on disk for the next one. :data:`_INDEX_LOCK` guards against the parent
mutating the index from more than one thread.
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
    """Keep cached text readable only by its owner — it is library content."""
    try:
        os.chmod(path, 0o600 if path.is_file() else 0o700)
    except OSError:
        pass


def get_fulltext_cache_root(config_path: str | None = None) -> Path:
    """Return the directory used to store cached fulltext, creating it."""
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


def _text_path(root: Path, entry_hash: str) -> Path:
    return root / f"{entry_hash}.txt"


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
    """Cache one attachment's extracted text. Parent process only.

    Identity is the attachment key plus the file's mtime and size, so a PDF
    replaced on disk lands on a different hash and can never serve stale text.
    A failure here is not fatal — it just means the next run re-extracts.
    """
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
    with _INDEX_LOCK:
        index = _load_index_unlocked(root)
        old = index["entries"].get(attachment_key)
        try:
            _text_path(root, entry["hash"]).write_text(text, encoding="utf-8")
            _private_chmod(_text_path(root, entry["hash"]))
        except OSError as e:
            logger.debug(f"Could not cache fulltext for {attachment_key}: {e}")
            return
        index["entries"][attachment_key] = entry
        _save_index_unlocked(root, index)
        # Drop superseded text (e.g. the attachment changed on disk).
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
    same input file — currently the PDF page cap. A mismatch is a miss, so
    lowering ``pdf_max_pages`` can never serve text extracted under a higher
    cap. Which *attachment* was chosen does not need to be in the profile:
    the entry is keyed by attachment key, so a changed
    ``attachment_priority`` resolves to a different key and misses naturally.
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
        try:
            text = _text_path(root, entry["hash"]).read_text(encoding="utf-8")
        except Exception:
            return None
    return text, entry.get("source", "pdf")


def evict_many(item_keys: Iterable[str], config_path: str | None = None) -> int:
    """Remove every cache entry belonging to the given item keys.

    Called once an item's embedding has landed in ChromaDB. Returns the
    number of entries removed.
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

    Removes entries older than ``max_age_days`` (an abandoned run) and entries
    whose backing attachment file is gone. Also sweeps orphaned ``.txt`` files
    that no index entry references — the residue of a run that died inside
    :func:`put_cached_text`, after it wrote the text file but before it saved
    the index that points at it.
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
