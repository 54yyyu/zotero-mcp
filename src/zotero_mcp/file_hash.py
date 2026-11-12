"""sha256 of attachment files, cached on disk by attachment key + file stamp.

The item-keyed index stores a content hash per attachment record so that a
file that was merely touched or moved is recognised as unchanged. Hashing a
large PDF library is slow, so the digest is remembered next to the fulltext
cache and only recomputed when the ``"<mtime_ns>:<size>"`` stamp changes.

Only the parent process calls this (same rule as :mod:`fulltext_cache`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from contextlib import contextmanager
from pathlib import Path

from zotero_mcp.fulltext_cache import get_fulltext_cache_root
from zotero_mcp.utils import write_json_atomic

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CHUNK = 1024 * 1024

# Cache files with a ``batched`` block open: ``[entries, dirty, depth]``. A
# digest then lands in memory and the file is written once, when the outermost
# block ends, instead of being rewritten whole on every miss (quadratic over a
# first full run of a large library).
_OPEN: dict[Path, list] = {}


def _cache_file(config_path: str | None) -> Path:
    return get_fulltext_cache_root(config_path) / "file_hashes.json"


def _load(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"Could not read file hash cache ({e}); starting fresh")
        return {}


@contextmanager
def batched(config_path: str | None = None):
    """Hold new digests in memory and write the cache file once, on exit.

    Reentrant. The file is written even when the block raises, so a stopped
    run keeps the hashing it paid for.
    """
    cache = _cache_file(config_path)
    with _LOCK:
        state = _OPEN.get(cache)
        if state is None:
            state = _OPEN[cache] = [_load(cache), False, 0]
        state[2] += 1
    try:
        yield
    finally:
        with _LOCK:
            state[2] -= 1
            if state[2] == 0:
                del _OPEN[cache]
                if state[1]:
                    _save(cache, state[0])


def _save(cache: Path, data: dict) -> None:
    try:
        write_json_atomic(cache, data)
    except OSError as e:  # a missed cache write only costs a re-hash
        logger.debug(f"Could not save file hash cache: {e}")


def known_stamps(config_path: str | None = None) -> dict[str, str]:
    """``{attachment key: file stamp}`` of every digest the cache holds, read once.

    The stamp is the one the file had when it was last hashed, so a file whose
    live stamp differs has changed since (the migration hashes what it re-keys).
    """
    cache = _cache_file(config_path)
    with _LOCK:
        state = _OPEN.get(cache)
        entries = state[0] if state else _load(cache)
    return {
        key: entry["stamp"] for key, entry in entries.items()
        if isinstance(entry, dict) and entry.get("stamp") and entry.get("sha256")
    }


def get_file_hash(
    attachment_key: str,
    path: str | Path,
    file_stamp: str,
    config_path: str | None = None,
) -> str | None:
    """sha256 hex of the file's bytes, or None if it cannot be read.

    Streams the file; a cached digest for the same ``file_stamp`` is returned
    without opening it.
    """
    cache = _cache_file(config_path)
    with _LOCK:
        state = _OPEN.get(cache)
        entry = (state[0] if state else _load(cache)).get(attachment_key)
    if isinstance(entry, dict) and entry.get("stamp") == file_stamp and entry.get("sha256"):
        return entry["sha256"]

    digest = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(_CHUNK):
                digest.update(chunk)
    except OSError as e:
        logger.debug(f"Could not hash {path}: {e}")
        return None
    value = digest.hexdigest()

    with _LOCK:
        state = _OPEN.get(cache)
        if state is not None:
            state[0][attachment_key] = {"stamp": file_stamp, "sha256": value}
            state[1] = True
        else:
            data = _load(cache)
            data[attachment_key] = {"stamp": file_stamp, "sha256": value}
            _save(cache, data)
    return value
