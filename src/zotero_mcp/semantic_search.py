"""
Semantic search functionality for Zotero MCP.

This module provides semantic search capabilities by integrating ChromaDB
with the existing Zotero client to enable vector-based similarity search
over research libraries.
"""

import contextlib
import functools
import json
import logging
import os
import queue
import re
import sys
import threading
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pyzotero.zotero_errors import ResourceNotFoundError

try:
    import tiktoken
except Exception:
    tiktoken = None


from . import batch_common, chroma_worker, fulltext_cache, gemini_batch, openai_batch
from .chroma_client import (
    ChromaClient,
    create_chroma_client,
    record_key,
    unreadable_index_facts,
    unreadable_index_message,
)
from .client import get_active_group_id, get_zotero_client
from .client import read_config_for_update as _read_config_for_update

# Re-exported so callers keep importing them from here, while the
# ChromaDB-free definitions stay importable without this module (#485).
from .config_light import (  # noqa: F401
    _DEFAULT_RERANKER_CONFIG,
    _DEFAULT_UPDATE_CONFIG,
    load_reranker_config,
    load_update_config,
    reranker_enabled,
    should_update,
)
from .embeddings.base import EmbeddingAPIUnreachable
from .embeddings.registry import batch_capable_providers
from .extract import PAGE_SEPARATOR
from .identifiers import metadata_match_keys
from .local_db import PERSONAL_LIBRARY_GROUP_ID, LocalZoteroReader
from .semantic_filters import OWNER_FIELDS, combine, live_fields, matches, referenced_fields, split_filters
from .utils import (
    _paginate,
    ensure_private_dir,
    format_creators,
    is_local_mode,
    suppress_stdout,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

# Batch-capable providers, by name. Everything provider-specific beyond the
# module itself lives on the module's ``ADAPTER`` (see batch_common.
# BatchAdapter), so this table stays a lookup rather than a second, parallel
# description of each provider that could drift from the adapter.
_BATCH_MODULES = {"openai": openai_batch, "gemini": gemini_batch}

SEARCH_ARTICLES = "articles"
SEARCH_PASSAGES = "passages"

#: Above this many record keys the local pre-filter's ``item_key $in`` clause
#: is left out and the filter is applied to the hits instead (#826).
PREFILTER_MAX_KEYS = 5000
#: A post-filtered search queries at most this many times, ×4 candidates each.
FILTER_QUERY_ROUNDS = 3
#: Zotero's cap on keys per ``itemKey`` parameter.
_ITEMKEY_BATCH = 50
#: Keys per zotero.sqlite lookup, well under SQLite's bound-parameter limit.
_LOCAL_KEY_BATCH = 500
# Items whose stored state a local scan reads ahead in one batched lookup.
_SCAN_SLICE = 500
_UNSET: Any = object()


def _count_items(count: int) -> str:
    return f"{count:,} item" if count == 1 else f"{count:,} items"

# How each provider spells "ready to import", for error text only. OpenAI
# reports "completed", Gemini "succeeded"; both normalize to
# batch_common.STATE_SUCCEEDED, which is what decision logic actually uses.
_IMPORTABLE_DESC = {"openai": "completed", "gemini": "succeeded"}


def _batch_module(provider: str):
    """Module implementing ``provider``'s Batch API flows."""
    try:
        return _BATCH_MODULES[provider]
    except KeyError:
        raise ValueError(
            f"Unknown batch provider {provider!r}; expected one of {sorted(_BATCH_MODULES)}"
        ) from None


def _batch_adapter(provider: str):
    """``provider``'s BatchAdapter, read off its module so that monkeypatched
    module attributes are still honored (adapter methods call by bare name)."""
    return _batch_module(provider).ADAPTER


def _is_superseded(current: dict[str, Any], newest: dict[str, Any] | None) -> bool:
    """Has ``newest`` taken over the items ``current`` was submitted for?

    Only when it has may the older run's parked chunks be abandoned and its
    sync watermark withheld. Three conditions, all necessary:

    * **A different run.** Compared by ``run_id``, never by manifest path: the
      same run reached through a differently spelled config path must not
      supersede itself.
    * **The same library.** A newer run for another ``group_id`` re-embeds none
      of this run's items, so it supersedes nothing here.
    * **Coverage.** A newer force-rebuild run re-embeds the whole library and
      therefore covers anything older. A newer incremental run covers an older
      incremental one (both were cut from the same sync watermark, which only a
      complete import advances) but *not* an older force-rebuild run, whose
      items it never touched.
    """
    if not newest or newest.get("run_id") == current.get("run_id"):
        return False
    if newest.get("group_id") != current.get("group_id"):
        return False
    return bool(newest.get("force_full_rebuild")) or not current.get("force_full_rebuild")


def _waiting_for_text_hint(include_fulltext: bool) -> str:
    """Say how the items waiting for full text get filled, in the mode this run is in.

    ``update-db --fulltext`` reads the local database, so outside local mode it
    only errors; there the items are filled from the text Zotero has extracted.
    """
    if is_local_mode():
        return "run 'zotero-mcp update-db --fulltext' to add them."
    if include_fulltext:
        return (
            "an update adds each one once Zotero has extracted its text "
            "('zotero-mcp update-db --fulltext' does this in local mode)."
        )
    return (
        "updates run without full text, so they stay out until "
        "semantic_search.include_fulltext is turned on."
    )


def _report(message: str) -> None:
    """Write a progress message to stderr, never failing the caller.

    Progress output is a courtesy, so a closed or broken stderr must not take
    down a multi-hour indexing run with it.
    """
    try:
        sys.stderr.write(message)
        sys.stderr.flush()
    except Exception:
        pass


def _term_width(default: int = 80) -> int:
    """Terminal width in columns, or ``default`` when stderr isn't a tty."""
    try:
        return os.get_terminal_size().columns
    except (OSError, ValueError):
        return default


def _display_cols(text: str) -> int:
    """Rendered width of ``text``: East-Asian wide characters count as 2."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _clip_to_width(line: str, max_cols: int) -> str:
    """Truncate a single-line progress string to ``max_cols`` display columns.

    A len()-based slice lets a CJK title wrap the terminal (57 chars ≈ 114
    columns), and once a ``\r`` line wraps, every later redraw repaints only
    the last physical line — the residue above it is the garbled progress
    output users see.
    """
    if max_cols <= 0:
        return ""
    cols = 0
    for idx, ch in enumerate(line):
        cols += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if cols > max_cols:
            return line[:idx]
    return line


def _realtime_slice_size(max_parallel: int) -> int:
    """How many items one preparation pass hands to the embedding workers.

    Scales with parallelism so a single pass yields enough payloads to keep
    every worker busy, and is capped so the classify step — which holds
    ``_chroma_call_lock`` — stays short. The floor of 25 is the historical
    sequential batch size, so an unparallelized run slices exactly as before.
    """
    return min(25 * max(1, max_parallel), 200)


_WIN_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_WIN_STILL_ACTIVE = 259
_WIN_ERROR_ACCESS_DENIED = 5


@functools.lru_cache(maxsize=1)
def _kernel32():
    """kernel32 with the three entry points below given explicit signatures.

    HANDLE must be declared: ctypes defaults a return type to C ``int``, which
    truncates a 64-bit handle and leaks it.
    """
    import ctypes
    from ctypes import wintypes

    lib = ctypes.WinDLL("kernel32", use_last_error=True)
    lib.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    lib.OpenProcess.restype = wintypes.HANDLE
    lib.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    lib.GetExitCodeProcess.restype = wintypes.BOOL
    lib.CloseHandle.argtypes = (wintypes.HANDLE,)
    lib.CloseHandle.restype = wintypes.BOOL
    return lib


def _pid_is_alive_windows(pid: int) -> bool:
    """Liveness check for Windows, where ``os.kill(pid, 0)`` is not a probe.

    ``signal.CTRL_C_EVENT`` is 0, so ``os.kill(pid, 0)`` never asks whether
    *pid* exists: it calls ``GenerateConsoleCtrlEvent`` and delivers a real
    Ctrl+C to every process sharing this console — the test run itself, and
    whatever shell launched it. ``OpenProcess`` answers the actual question
    and sends nothing.
    """
    import ctypes
    from ctypes import wintypes

    lib = _kernel32()
    handle = lib.OpenProcess(_WIN_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # Access denied means it exists but is owned by another user; that is
        # the PermissionError branch below, so report it alive.
        return ctypes.get_last_error() == _WIN_ERROR_ACCESS_DENIED
    try:
        code = wintypes.DWORD()
        if not lib.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True  # the handle opened, so the process object exists
        return code.value == _WIN_STILL_ACTIVE
    finally:
        lib.CloseHandle(handle)


def _pid_is_alive(pid: int) -> bool:
    """Best-effort liveness check for a process id."""
    if sys.platform == "win32":
        try:
            return _pid_is_alive_windows(pid)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    except Exception:
        return False
    return True


def read_lock_holder(lock_path: Path) -> tuple[int | None, bool]:
    """Return ``(holder_pid, alive)`` for the update lock, for diagnostics.

    ``holder_pid`` is None when the file is missing/unparseable. ``alive`` says
    whether that pid still exists — a held lock whose holder is dead would be a
    genuinely stale lock (flock releases those automatically on POSIX, so this
    is purely to make the user-facing "skipped" message precise).
    """
    try:
        raw = lock_path.read_text().strip()
        pid = int(raw)
    except Exception:
        return None, False
    return pid, _pid_is_alive(pid)


def _force_update_requested() -> bool:
    """Whether the user asked to bypass the cross-process update lock."""
    return os.getenv("ZOTERO_MCP_FORCE_UPDATE", "").strip().lower() in {"1", "true", "yes"}


def _update_lock_path() -> Path:
    """Where the cross-process update lock lives.

    One function so the test suite can point it at a temp dir: tests that
    run update_database() used to take the user's real lock, so two test
    runs at once, or a test run during a real update, made each other skip.
    """
    return Path.home() / ".config" / "zotero-mcp" / "update.lock"


# msvcrt.locking is a mandatory byte-range lock: the pid text at offset 0 must
# stay readable for the process that loses the race, so the lock sits past it.
_WIN_LOCK_OFFSET = 4096


def _try_lock(fd) -> bool:
    """Take a non-blocking exclusive lock on the open file; False if held."""
    try:
        import fcntl
    except ImportError:
        import msvcrt  # Windows

        fd.seek(_WIN_LOCK_OFFSET)
        try:
            msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
        finally:
            fd.seek(0)
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


@contextlib.contextmanager
def _acquire_update_lock(lock_path: Path):
    """Non-blocking exclusive lock over an update-database run.

    Yields True if the lock was acquired (caller should proceed), False if
    another process already holds it (caller should skip). This prevents the
    MCP server's auto-update in ``server_lifespan`` from racing a manual
    ``zotero-mcp update-db`` invocation on the same ChromaDB collection.

    Setting ``ZOTERO_MCP_FORCE_UPDATE=1`` bypasses the lock entirely — an
    escape hatch for the rare case where a lock appears stuck (e.g. a crashed
    holder on a filesystem with quirky flock semantics) and the user knowingly
    accepts the small double-work risk.

    ``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows (#267).

    While it is held, this process's ChromaDB worker does not exit when idle.
    """
    if _force_update_requested():
        with chroma_worker.update_running():
            yield True
        return

    ensure_private_dir(lock_path.parent)
    fd = None
    try:
        # "a" not "w": a process that loses the race must not truncate the
        # holder's pid before the lock fails. We truncate after acquiring.
        fd = open(lock_path, "a")
        if not _try_lock(fd):
            yield False
            return
        # Record our pid so a concurrent invocation can report the holder.
        try:
            fd.seek(0)
            fd.truncate()
            fd.write(str(os.getpid()))
            fd.flush()
        except Exception:
            pass
        with chroma_worker.update_running():
            yield True
    finally:
        if fd is not None:
            fd.close()


# Loaded on first use, not at import. get_encoding() parses ~100k BPE ranks
# (0.1-0.25 s), or downloads them when they are not cached yet, and this
# module is imported on the main thread before the server answers its first
# request on Windows (#485) — for a tokenizer that only _truncate_to_tokens
# uses. False records a failed load so it is not retried on every call.
_tokenizer = None


def _get_tokenizer():
    """The cl100k_base encoding, or None when tiktoken cannot provide it."""
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.get_encoding("cl100k_base") if tiktoken else False
        except Exception:
            _tokenizer = False
    return _tokenizer or None


def _truncate_to_tokens(text: str, max_tokens: int = 8000) -> str:
    """Truncate text to fit within embedding model token limit.

    Uses tiktoken for accurate token counting when available,
    falls back to conservative character-based estimation.
    """
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        tokens = tokenizer.encode(text, disallowed_special=())
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = tokenizer.decode(tokens)
    else:
        # Fallback: conservative char limit (~1.5 chars/token for non-Latin scripts)
        max_chars = max_tokens * 2
        if len(text) > max_chars:
            text = text[:max_chars]
    return text


# Bumped when the ChromaDB metadata shape changes in a way that requires a
# one-time migration of existing documents. Version 2 (#163) added the
# `group_id` field. A persisted collection whose config.json records a lower
# (or absent, i.e. 0) version gets migrated via `_backfill_group_ids()`.
# Version 3: the backfill became evidence-based (guessed attribution never
# feeds the group_id-scoped deletion pass). Bumped past 2 even though no
# released build ever completed a version-2 backfill — the sole save site
# sat behind a call that always raised — so that any index migrated by
# intermediate development code re-runs the corrected migration; a saved
# version number is not proof the migration that saved it was correct.
_INDEX_SCHEMA_VERSION = 3

# The incremental deletion pass refuses (without an explicit opt-in) to
# remove at least this many docs AND at least this fraction of the syncing
# library's indexed docs in one run: that fingerprint is far more likely a
# truncated item_versions() response or a scoping regression than a real
# purge, and deletions from a derived index are only cheap to undo until
# the next re-embed.
_MASS_DELETION_MIN_DOCS = 25
_MASS_DELETION_MIN_FRACTION = 0.25

# Item types that are never indexed as standalone documents. Both API
# fetch paths use this; the local SQLite scan in local_db.py excludes the
# same three types in SQL (#604).
_NON_INDEXED_ITEM_TYPES = frozenset({"attachment", "note", "annotation"})


def _extract_fulltext_batch(reader, items):
    """Yield ``(item_id, (text, source) | None)`` for every item in ``items``.

    Prefers the reader's batch API, which parallelises across a process pool
    when configured. Falls back to one call per item for readers that do not
    provide it — the minimal doubles used in tests implement only
    ``extract_fulltext_for_item(item_id)``, and they should not have to grow
    a new method just because the real reader gained a faster path.
    """
    batch = getattr(reader, "extract_fulltext_for_items", None)
    if batch is not None:
        yield from batch(items)
        return
    for item_id, _item_key in items:
        yield item_id, reader.extract_fulltext_for_item(item_id)


#: End-of-stream marker for the streaming index pipeline's queues. A unique
#: object so it can never collide with a real payload.
_STREAM_SENTINEL = object()

#: Vectors buffered before the streaming committer writes to ChromaDB. Large
#: enough that commits are not the bottleneck, small enough that a crash loses
#: little work and the fulltext cache is evicted steadily rather than at the end.
_STREAM_COMMIT_THRESHOLD = 200


#: Work-unit kinds (#826). Indexing works on two kinds of unit, both carried as
#: tagged dicts so they flow through the same queues and passes:
#:
#: * ``"item"``: a regular item, the API-shaped item dict, which becomes one
#:   bibliographic record (``KEY``) built from its metadata alone.
#: * ``"attachment"``: one file of an item (or a standalone file), which
#:   becomes one attachment record (``ATT``) and, with chunking, passages
#:   (``ATT#n``) built from the file's text alone.
UNIT_ITEM = "item"
UNIT_ATTACHMENT = "attachment"

#: The prefix ``_attach_web_fulltext`` puts before the attachment key.
_WEB_ATTACHMENT_SOURCE = "web-api:attachment:"

#: file_stamp of an attachment read through the API whose file Zotero
#: describes with neither md5 nor version (see _api_file_stamp).
_API_UNKNOWN_STAMP = "api:unknown"


def _item_unit(item: dict[str, Any]) -> dict[str, Any]:
    """Tag an API-shaped regular item, in place, as a bibliographic unit."""
    item["unit"] = UNIT_ITEM
    return item


def _attachment_unit(
    key: str,
    text: str,
    *,
    file_stamp: str,
    file_hash: str | None = None,
    date_modified: str | None = "",
    group_id: int | None = None,
    owner_key: str | None = None,
    label: str = "",
) -> dict[str, Any]:
    """An attachment work unit: one file's text and the record's own state.

    ``owner_key`` (the parent, or None for a standalone file) and ``label``
    are for progress and bookkeeping in this run only; neither is stored.
    """
    return {
        "key": key,
        "unit": UNIT_ATTACHMENT,
        "text": text or "",
        "file_stamp": file_stamp,
        "file_hash": file_hash,
        "date_modified": date_modified or "",
        "group_id": group_id,
        "owner_key": owner_key,
        "label": label,
    }


def _api_file_stamp(row: Any) -> str:
    """``file_stamp`` of an attachment read through the Zotero API.

    The API gives no file mtime_ns/size, so the stamp is built from what it
    has: the stored file's ``md5`` with its ``mtime`` when Zotero has them
    (they change exactly when the file is replaced), else the attachment
    item's ``version`` (which also changes on a rename, a harmless re-embed).
    Prefixed so it can never equal a local ``"<mtime_ns>:<size>"`` stamp.
    """
    data = (row.get("data") if isinstance(row, dict) else None) or {}
    if data.get("md5"):
        return f"api:{data['md5']}:{data.get('mtime', '')}"
    version = row.get("version", data.get("version")) if isinstance(row, dict) else None
    if version is not None:
        return f"api-version:{version}"
    return _API_UNKNOWN_STAMP


def _is_attachment_unit(unit: dict[str, Any]) -> bool:
    return unit.get("unit") == UNIT_ATTACHMENT


#: Metadata that records written before #826 carried and new records never do.
_LEGACY_METADATA_KEYS = (
    "item_type", "title", "date", "date_added", "creators", "publication", "url", "doi",
    "tags", "citation_key", "has_fulltext", "fulltext_source", "attachment_keys",
    "attachment_priority", "parent_item_key",
)


def _is_legacy_record(metadata: dict[str, Any] | None) -> bool:
    """A record written before #826: every one of them stored ``title``, no new one does."""
    return bool(metadata) and "title" in metadata


def _text_record_keys(metadatas: list[dict[str, Any]]) -> set[str]:
    """Record keys among ``metadatas`` built from a file's text: the keys text is cached under.

    A bibliographic record is built from metadata, so it has nothing in the
    fulltext cache to evict.
    """
    return {
        m["item_key"] for m in metadatas
        if m.get("item_key") and ("file_stamp" in m or "chunk_index" in m)
    }


def _indexed_with_text(metadata: dict[str, Any] | None) -> bool:
    """Whether a stored record was built from a file's text.

    An attachment record always is; a legacy record says so in ``has_fulltext``.
    """
    metadata = metadata or {}
    if "file_stamp" in metadata:
        return True
    return bool(metadata.get("has_fulltext")) and metadata.get("has_fulltext") != "failed"


def _own_metadata(key: str, group_id: Any, date_modified: Any) -> dict[str, Any]:
    """``item_key``, ``group_id`` and ``date_modified`` of a bare record.

    Library attribution (#163): 0 = personal, else groupID. When it is
    unknown the key is OMITTED, not defaulted: positive attribution is
    deletion authority under the group_id-scoped deletion pass, so unknown
    must stay unknown.
    """
    metadata: dict[str, Any] = {"item_key": key}
    if group_id is not None:
        metadata["group_id"] = int(group_id)
    metadata["date_modified"] = date_modified or ""
    return metadata


class _HeldText:
    """Attachments' text kept on disk between the article pass and the passage pass.

    A run writing both levels embeds every article record before any passage.
    Holding each item's text in memory until its passages are written would
    keep the whole library's text in memory for the run, so the article pass
    moves it to a private temporary directory and the passage pass reads it
    back, one slice at a time. No file is parsed twice either way. Text that
    can't be written to disk stays in memory.
    """

    _KEY = "_held_fulltext"

    def __init__(self) -> None:
        self._dir: Any = None
        self._count = 0
        self._closed = False

    def hold(self, item: dict[str, Any]) -> None:
        text = item.get("text") or ""
        if not text or self._closed:
            # After close, a pipeline thread still finishing must not leave
            # a new directory behind: the text stays in memory.
            return
        try:
            if self._dir is None:
                import tempfile

                self._dir = tempfile.TemporaryDirectory(prefix="zotero-mcp-text-")
            self._count += 1
            path = Path(self._dir.name) / f"{self._count}.txt"
            path.write_text(text, encoding="utf-8")
        except OSError as e:
            logger.debug(f"Keeping the text of {item.get('key')} in memory: {e}")
            return
        item[self._KEY] = str(path)
        item["text"] = ""

    def restore(self, item: dict[str, Any]) -> None:
        path = item.pop(self._KEY, None)
        if path:
            item["text"] = Path(path).read_text(encoding="utf-8")
            Path(path).unlink(missing_ok=True)

    @staticmethod
    def release(item: dict[str, Any]) -> None:
        """Drop the text again once the attachment's passages are built."""
        if item.get("text"):
            item["text"] = ""

    def close(self) -> None:
        self._closed = True
        if self._dir is not None:
            self._dir.cleanup()
            self._dir = None


def _passage_digest(document: str) -> str:
    """A short fingerprint of a passage's text."""
    import hashlib

    return hashlib.sha256(document.encode("utf-8")).hexdigest()[:16]


def _split_prepared_into_requests(prepared: dict[str, Any], request_batch_size: int):
    """Yield ``(documents, metadatas, ids, item_keys)`` request-sized payloads.

    ``item_keys`` is a list of ``(item_key, already_existed)`` pairs, so the
    committer can keep added-vs-updated accounting item-granular no matter how
    many chunks an item produced.

    Splits only on item boundaries. A payload may therefore exceed
    ``request_batch_size`` when a single item contributed more chunks than
    that, which is deliberate: an item whose chunks were spread across two
    independently committed requests could end up half-indexed if one of them
    failed, and stale passages are pruned per item after each successful write.
    """
    documents = prepared["documents"]
    metadatas = prepared["metadatas"]
    ids = prepared["ids"]
    existing = prepared["existing_item_keys"]

    buffer_docs: list[str] = []
    buffer_metas: list[dict[str, Any]] = []
    buffer_ids: list[str] = []
    buffer_keys: list[tuple[str, bool]] = []
    offset = 0

    for item_key, doc_count in zip(
        prepared["item_keys_order"], prepared["item_doc_counts"]
    ):
        if doc_count <= 0:
            continue
        end = offset + doc_count
        buffer_docs.extend(documents[offset:end])
        buffer_metas.extend(metadatas[offset:end])
        buffer_ids.extend(ids[offset:end])
        buffer_keys.append((item_key, item_key in existing))
        offset = end

        if len(buffer_docs) >= request_batch_size:
            yield buffer_docs, buffer_metas, buffer_ids, buffer_keys
            buffer_docs, buffer_metas, buffer_ids, buffer_keys = [], [], [], []

    if buffer_docs:
        yield buffer_docs, buffer_metas, buffer_ids, buffer_keys


def warmup_reranker(config_path: str | None = None) -> bool:
    """Preload the configured reranker into the process-wide cache.

    Lets the server pay the cross-encoder load cost once at startup (off the
    request path) so the first real ``zotero_semantic_search`` is fast too
    (issue #283). Returns ``True`` if a model was warmed, ``False`` if the
    reranker is disabled. Never raises — a failed warmup must not crash startup.
    """
    cfg = load_reranker_config(config_path)
    if not cfg.get("enabled", False):
        return False
    model = cfg.get("model", _DEFAULT_RERANKER_CONFIG["model"])
    try:
        get_cached_reranker(model, cfg)
        return True
    except Exception as e:
        logger.warning(f"Reranker warmup failed for '{model}': {e}")
        return False


# ---------------------------------------------------------------------------
# Passage-level chunking (Tier-1 grounded retrieval)
# ---------------------------------------------------------------------------

# Separator between PDF pages in extracted fulltext. ``extract`` always emits
# it; text that reached us another way (Zotero's own full-text cache) may not
# carry it, in which case only char offsets are reported and ``page`` is
# omitted from passage metadata.
_PAGE_SEPARATOR = PAGE_SEPARATOR


def split_into_passages(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 200,
    max_chunks: int = 20,
) -> list[tuple[str, int, int]]:
    """Split *text* into overlapping passages on natural boundaries.

    Pure function (no I/O, no model load) so it is unit-testable in isolation.
    Returns a list of ``(passage_text, char_start, char_end)`` tuples with
    character offsets into the original string. Each window targets
    ``chunk_size`` characters but is snapped back to the nearest paragraph or
    sentence boundary in its second half so passages read as coherent quotes.
    Consecutive windows overlap by ``overlap`` characters so a relevant span
    straddling a boundary is still captured whole in one of them. At most
    ``max_chunks`` passages are produced (a guard against pathologically long
    documents inflating the index).
    """
    text = (text or "").strip()
    if not text:
        return []
    if overlap >= chunk_size:
        overlap = chunk_size // 4

    passages: list[tuple[str, int, int]] = []
    start = 0
    n = len(text)
    while start < n and len(passages) < max_chunks:
        end = min(n, start + chunk_size)
        if end < n:
            window = text[start:end]
            for sep in ("\n\n", ". ", ".\n", "\n", " "):
                idx = window.rfind(sep)
                if idx != -1 and idx >= int(chunk_size * 0.5):
                    end = start + idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            passages.append((chunk, start, min(end, n)))
        if end >= n:
            break
        new_start = end - overlap
        # Guarantee forward progress even when overlap is large.
        start = new_start if new_start > start else end
    return passages


def _same_moment(first: Any, second: Any) -> bool:
    """Whether two Zotero timestamps are the same time, whichever format each is in.

    The local database gives ``2026-07-02 01:01:48`` and the API gives
    ``2026-07-02T01:01:48Z``, both UTC, so comparing the text would make every
    item indexed one way look changed to the other.
    """
    if first == second:
        return True

    def parse(value: Any) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(str(value).strip().replace(" ", "T").replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    a, b = parse(first), parse(second)
    return a is not None and a == b


def _attachment_priority_changed(existing_metadata: dict, current_tag: str) -> bool:
    """True when a document was extracted under a different attachment priority.

    A document indexed before this field existed carries no tag. That is
    treated as "unchanged", not as a mismatch: every pre-existing index would
    otherwise re-extract in full on the first run after upgrading, which is a
    lot of work to impose on someone who never touched the setting. Those
    documents pick the tag up the next time they are re-indexed for any other
    reason, and converge from there.
    """
    stored = existing_metadata.get("attachment_priority")
    if stored is None:
        return False
    return stored != current_tag


def _page_for_offset(text: str, offset: int) -> int | None:
    """Return the 1-indexed page containing *offset*, or None if unknowable.

    Only meaningful when *text* carries ``_PAGE_SEPARATOR`` form-feed page
    breaks (page-aware extraction). Returns None otherwise so callers can omit
    a page field rather than report a misleading one.
    """
    if _PAGE_SEPARATOR not in text:
        return None
    return text.count(_PAGE_SEPARATOR, 0, max(0, offset)) + 1


# Canonical section labels, keyed by the heading that introduces them. Longer
# names precede the shorter ones they contain ("materials and methods" before
# "methods") so the alternation below cannot match a prefix of a longer
# heading. Deliberately conservative: an unrecognised heading leaves the
# passage unlabelled rather than mislabelled, since a wrong section is worse
# than none for a reader deciding whether a hit is the paper's own claim.
_SECTION_ALIASES: tuple[tuple[str, str], ...] = (
    (r"abstract", "Abstract"),
    (r"introduction", "Introduction"),
    (r"background", "Introduction"),
    (r"related work", "Introduction"),
    (r"literature review", "Introduction"),
    (r"theoretical framework", "Introduction"),
    (r"(?:the\s+)?present study", "Introduction"),
    (r"materials?\s+and\s+methods?", "Methods"),
    (r"methodology", "Methods"),
    (r"methods?", "Methods"),
    (r"participants?", "Methods"),
    (r"measures?", "Methods"),
    (r"procedure", "Methods"),
    (r"data\s+analys[ei]s", "Methods"),
    (r"statistical\s+analys[ei]s", "Methods"),
    (r"results?\s+and\s+discussion", "Results"),
    (r"results?", "Results"),
    (r"findings?", "Results"),
    (r"general\s+discussion", "Discussion"),
    (r"discussion", "Discussion"),
    (r"limitations?", "Discussion"),
    (r"implications?", "Discussion"),
    (r"conclusions?", "Conclusion"),
    (r"acknowledge?ments?", "Back matter"),
    (r"funding", "Back matter"),
    (r"conflicts? of interest", "Back matter"),
    (r"references?", "References"),
    (r"bibliography", "References"),
    (r"appendix[\w \t.:-]*", "Appendix"),
    (r"supplementary[\w \t.:-]*", "Appendix"),
)

# A heading is recognised by shape as well as name: its own line, optionally
# numbered ("3.", "3.1", "IV."), nothing after it but an optional colon.
# Extracted PDF text keeps line breaks but loses styling, so shape is the only
# signal left that a line is a heading rather than prose mentioning the word.
_SECTION_HEADING_RE = re.compile(
    r"^[ \t]*(?:\d+(?:\.\d+)*[.)]?[ \t]+|[IVXivx]{1,5}[.)][ \t]+)?"
    r"(" + "|".join(pattern for pattern, _ in _SECTION_ALIASES) + r")"
    r"[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


# A chapter or part heading ends whatever section preceded it. Without this a
# label runs forward until the next *recognised* heading, and since chapter
# titles are not recognised, every chapter after the first in an edited volume
# or monograph inherits the tail of the one before it, usually "References",
# because chapter bibliographies are the last thing this parser recognises.
# That is the mislabelling the module set out to avoid.
#
# A false positive here only clears a label, never invents one, so the pattern
# can afford to be generous: any line that opens with Chapter/Part/Book and a
# number, numeral or spelled-out ordinal, with or without a title after it.
_SECTION_RESET_RE = re.compile(
    r"""
    ^[ \t]*(?:
        # A chapter, part or book heading, numbered any of the usual ways.
        (?:chapter|part|book)[ \t]+
        (?:\d{1,3}|[ivxlc]{1,7}|one|two|three|four|five|six|seven|eight|nine|
           ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|
           eighteen|nineteen|twenty)
        \b[^\n]{0,80}
      |
        # An all-capitals line short enough to be a heading rather than
        # prose. Extracted PDFs lose styling, so case and length are what is
        # left to go on; requiring two letters and no lowercase keeps ordinary
        # sentences, wrapped fragments and bare numbers out. Case-sensitive
        # even though the pattern as a whole is not, since case is the signal.
        (?-i:
            (?=[^\na-z]{5,70}[ \t]*$)
            (?=[^\n]*[A-Z][^\n]*[A-Z])
            [A-Z0-9][^\na-z]{4,69}
        )
    )
    [ \t]*$
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)


def _section_offsets(text: str) -> list[tuple[int, str | None]]:
    """Offsets of recognised section headings in *text*, in document order.

    A ``None`` label marks a chapter or part boundary, which ends the previous
    section without starting a named one.

    Computed once per document and shared by all of its passages: scanning the
    whole text per chunk would be quadratic in a long paper.
    """
    text = text or ""
    marks: list[tuple[int, str | None]] = []
    for match in _SECTION_HEADING_RE.finditer(text):
        heading = match.group(1).strip()
        for pattern, label in _SECTION_ALIASES:
            if re.fullmatch(pattern, heading, re.IGNORECASE):
                marks.append((match.start(), label))
                break
    # A recognised heading may also look like an unrecognised one (a line
    # reading "METHODS" matches both), and the label must win over the reset.
    named = {offset for offset, _ in marks}
    marks.extend(
        (match.start(), None)
        for match in _SECTION_RESET_RE.finditer(text)
        if match.start() not in named
    )
    marks.sort(key=lambda mark: mark[0])
    return marks


def _section_at(marks: list[tuple[int, str | None]], offset: int) -> str | None:
    """The section label in force at *offset*, or None when unknown.

    None is returned before the first recognised heading (front matter, and
    every document whose headings this parser did not recognise) and after a
    chapter boundary that has not yet been followed by one.
    """
    label: str | None = None
    for start, name in marks:
        if start > offset:
            break
        label = name
    return label


def best_snippet(query: str, text: str, width: int = 320) -> tuple[str, int]:
    """Return the ``width``-char window of *text* richest in query terms.

    Used to surface a *grounded* quote — the part of a matched document that
    actually overlaps the query — instead of a blind head-truncation. Returns
    ``(snippet, char_start)``. Falls back to the head of the text when no query
    term appears. Pure and dependency-free (lexical overlap only).
    """
    text = text or ""
    if not text.strip():
        return "", 0
    if len(text) <= width:
        return text.strip(), 0
    terms = [t for t in re.findall(r"\w+", (query or "").lower()) if len(t) > 2]
    if not terms:
        return text[:width].strip(), 0
    lowered = text.lower()
    # Score each candidate window anchored at a query-term hit; keep the best.
    best_start = 0
    best_score = -1
    for m in re.finditer(r"\w+", lowered):
        if m.group(0) not in terms:
            continue
        start = max(0, m.start() - width // 3)
        window = lowered[start : start + width]
        score = sum(window.count(t) for t in terms)
        if score > best_score:
            best_score = score
            best_start = start
    snippet = text[best_start : best_start + width].strip()
    return snippet, best_start


# A search slower than this looks like a hang, so the result says why.
SLOW_PASSAGE_LOAD_SECONDS = 5.0


def _passage_load_note(load: dict | None) -> str | None:
    """Why a passage search was slow, from the worker's ``passage_load`` report, or None."""
    if not load or load.get("seconds", 0) < SLOW_PASSAGE_LOAD_SECONDS:
        return None
    seconds = f"{load['seconds']:.0f}"
    minutes = f"{load.get('idle_minutes', 0):g}"
    setting = (
        "set semantic_search.idle_unload_minutes in config.json "
        "(0 keeps it loaded until the server exits)"
    )
    if load.get("reloaded"):
        return (
            f"This search took {seconds} s because the passage ChromaDB collection had to be "
            f"loaded into memory again: it is freed after {minutes} minutes without a passage "
            f"search or index update. Passage searches are fast while it stays loaded. To keep "
            f"it loaded longer, {setting}."
        )
    return (
        f"This search took {seconds} s because the passage ChromaDB collection had to be "
        f"loaded into memory first. Passage searches are fast while it stays loaded; it is "
        f"freed after {minutes} minutes without one. To change that, {setting}."
    )


def _drop_missing_documents(results: dict) -> int:
    """Remove hits whose document text is gone, keeping the parallel lists aligned.

    A server that stays up while documents are deleted from the collection by
    another process (a CLI update, or the deletion pass) can get ids back from
    its open handle whose documents are ``None``. The cross-encoder accepts
    only strings, so one such hit failed every search until restart (#545).
    A search should come back with fewer results instead. Returns how many
    hits were dropped.
    """
    documents = (results.get("documents") or [[]])[0]
    if not documents:
        return 0
    keep = [i for i, doc in enumerate(documents) if isinstance(doc, str)]
    dropped = len(documents) - len(keep)
    if dropped:
        for key in ("ids", "distances", "documents", "metadatas"):
            column = results.get(key)
            if column and column[0]:
                column[0] = [column[0][i] for i in keep]
        logger.warning(
            "Dropped %d semantic search hit(s) whose documents were removed from "
            "the index while this server was running.", dropped,
        )
    return dropped


def _collect_hits(chroma_results: dict) -> list[dict[str, Any]]:
    """One dict per ChromaDB result: its record key, library, text, scores."""
    ids = (chroma_results.get("ids") or [[]])[0] or []
    distances = (chroma_results.get("distances") or [[]])[0] or []
    documents = (chroma_results.get("documents") or [[]])[0] or []
    metadatas = (chroma_results.get("metadatas") or [[]])[0] or []
    rerank_scores = (chroma_results.get("rerank_scores") or [[]])[0] or []
    hits: list[dict[str, Any]] = []
    for i, raw_id in enumerate(ids):
        meta = metadatas[i] if i < len(metadatas) else {}
        meta = meta if isinstance(meta, dict) else {}
        group_id = meta.get("group_id")
        hit = {
            "record_key": record_key(raw_id),
            "meta": meta,
            "library": int(group_id) if group_id is not None else None,
            "document": documents[i] if i < len(documents) else "",
            "distance": distances[i] if i < len(distances) else None,
        }
        if i < len(rerank_scores):
            hit["rerank_score"] = rerank_scores[i]
        hits.append(hit)
    return hits


def _is_trashed(item: dict) -> bool:
    return bool((item.get("data") or {}).get("deleted"))


def _fetch_items_via_client(client, keys: list[str]) -> tuple[dict[str, dict], dict[str, Exception]]:
    """Fetch ``keys`` through a pyzotero client: ``(found, failed)``.

    Batches of 50 via ``itemKey``. That listing leaves trashed items out, so
    a key absent from a batch's answer is trashed or deleted, and either way
    not a result. Where the batch call itself fails, keys are asked for one
    at a time, so that one failure is reported per key and a not-found key
    is simply missing.
    """
    found: dict[str, dict] = {}
    failed: dict[str, Exception] = {}
    for start in range(0, len(keys), _ITEMKEY_BATCH):
        batch = keys[start:start + _ITEMKEY_BATCH]
        wanted = set(batch)
        try:
            # Paged, not one call: the local API can list the requested
            # items' children first, past the end of a capped page (#499).
            for item in _paginate(client.items, itemKey=",".join(batch)) or []:
                # An itemKey query also returns the items' child notes.
                if isinstance(item, dict) and item.get("key") in wanted:
                    found[item["key"]] = item
            continue
        except Exception as e:
            logger.debug(f"Batched item lookup failed, asking per key: {e}")
        for key in batch:
            try:
                found[key] = client.item(key)
            except ResourceNotFoundError:
                continue
            except Exception as e:
                failed[key] = e
    return found, failed


class CrossEncoderReranker:
    """Optional cross-encoder re-ranker for semantic search results."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.label = f"local/{model_name}"

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[int]:
        """Re-rank documents by relevance to query.

        Returns indices of top_k documents in descending relevance order.
        """
        return [idx for idx, _ in self.rerank_with_scores(query, documents, top_k)]

    def rerank_with_scores(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        """Re-rank documents, returning ``(index, score)`` pairs.

        Scores are the raw cross-encoder relevance logits, surfaced so search
        results can report *why* an item ranked where it did, not just an
        opaque order.
        """
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in ranked[:top_k]]

    def rerank_detailed(
        self, query: str, documents: list[str], top_k: int
    ) -> tuple[list[tuple[int, float]], str | None]:
        """``(pairs, error)``, matching :class:`APIReranker`; a local model has no soft failure."""
        return self.rerank_with_scores(query, documents, top_k), None


# Hosted rerank endpoints. All of them take a query, a list of documents and a
# cut-off, and answer with ``{index, relevance_score}`` records, so one client
# covers them and switching vendors is a config edit. What differs:
#
# * the spelling of the cut-off (``top_k`` vs ``top_n``);
# * the key the records are listed under -- Voyage answers
#   ``{"object": "list", "data": [...]}`` while Cohere, OpenRouter and
#   Contextual answer ``{"results": [...]}``. Both are read, so a provider that
#   changes its envelope does not silently turn re-ranking off;
# * how an instruction is passed. Contextual takes an ``instruction`` field.
#   Voyage's instruction-following models (rerank-2.5 and later) read the
#   instruction from the query itself and have no such field, so for them,
#   and for any provider without a documented field, it is prepended.
_RERANK_PROVIDERS: dict[str, dict[str, str]] = {
    "voyage": {
        "url": "https://api.voyageai.com/v1/rerank",
        "key_env": "VOYAGE_API_KEY",
        "top_field": "top_k",
        "instruction": "query",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/rerank",
        "key_env": "OPENROUTER_API_KEY",
        "top_field": "top_n",
        "instruction": "query",
    },
    "cohere": {
        "url": "https://api.cohere.com/v2/rerank",
        "key_env": "COHERE_API_KEY",
        "top_field": "top_n",
        "instruction": "query",
    },
    "contextual": {
        "url": "https://api.contextual.ai/v1/rerank",
        "key_env": "CONTEXTUAL_API_KEY",
        "top_field": "top_n",
        "instruction": "field",
    },
}

# Statuses worth one more attempt: throttling and transient server errors.
_RERANK_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
# Ceiling on a Retry-After wait. A search is interactive; past this it is
# better to answer in retrieval order than to keep the caller waiting.
_RERANK_MAX_RETRY_WAIT = 5.0


class APIReranker:
    """Re-rank through a hosted cross-encoder instead of a local model.

    Interface-compatible with :class:`CrossEncoderReranker`, so the search path
    does not care which one it holds. Exists because the strong local
    cross-encoders need a GPU to be interactive — on CPU they add seconds to
    every query, while these endpoints answer in 200-600ms.

    A failed call degrades to the retriever's own order rather than raising:
    losing the re-ranking is a worse search, but losing the search entirely
    because a network call timed out is a broken tool. The failure is not
    silent, though: :meth:`rerank_detailed` returns the reason, and the search
    result carries it, so a misconfigured model or key shows up in the very
    next search instead of only in a log nobody reads.
    """

    def __init__(
        self,
        model_name: str,
        provider: str,
        base_url: str | None = None,
        api_key_env: str | None = None,
        instruction: str | None = None,
        timeout: float = 30.0,
    ):
        preset = _RERANK_PROVIDERS.get(provider, {})
        if not preset and not (base_url and api_key_env):
            raise ValueError(
                f"Unknown rerank provider '{provider}'. Use one of "
                f"{', '.join(sorted(_RERANK_PROVIDERS))}, or set both "
                "reranker.base_url and reranker.api_key_env."
            )
        self.provider = provider
        self.model_name = model_name
        self.url = base_url or preset["url"]
        self.top_field = preset.get("top_field", "top_n")
        self.instruction_mode = preset.get("instruction", "query")
        self.key_env = api_key_env or preset.get("key_env", "")
        self.instruction = instruction
        self.timeout = timeout

        self.api_key = os.getenv(self.key_env) if self.key_env else None
        if not self.api_key:
            raise ValueError(
                f"Rerank provider '{provider}' needs {self.key_env} in the "
                "environment; it is unset."
            )

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model_name}"

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[int]:
        """Indices of the top_k documents, most relevant first."""
        return [idx for idx, _ in self.rerank_with_scores(query, documents, top_k)]

    def rerank_with_scores(
        self, query: str, documents: list[str], top_k: int
    ) -> list[tuple[int, float]]:
        """``(index, relevance_score)`` pairs from the hosted reranker."""
        return self.rerank_detailed(query, documents, top_k)[0]

    def _payload(self, query: str, documents: list[str], cutoff: int) -> dict[str, Any]:
        if self.instruction and self.instruction_mode == "query":
            query = f"{self.instruction.strip()}\n\nQuery: {query}"
        payload: dict[str, Any] = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            self.top_field: cutoff,
        }
        if self.instruction and self.instruction_mode == "field":
            payload["instruction"] = self.instruction
        return payload

    @staticmethod
    def _parse(body: Any, n_documents: int) -> list[tuple[int, float]]:
        """Read ``{index, relevance_score}`` records from either envelope."""
        if not isinstance(body, dict):
            return []
        records = body.get("results")
        if not isinstance(records, list):
            records = body.get("data")
        if not isinstance(records, list):
            return []
        ranked: list[tuple[int, float]] = []
        seen: set[int] = set()
        for r in records:
            if not isinstance(r, dict) or "index" not in r:
                continue
            try:
                idx = int(r["index"])
                score = float(r.get("relevance_score", r.get("score", 0.0)))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < n_documents and idx not in seen:
                seen.add(idx)
                ranked.append((idx, score))
        # Providers return records sorted, but nothing in their contracts says
        # so for every vendor; sort rather than trust.
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked

    def rerank_detailed(
        self, query: str, documents: list[str], top_k: int
    ) -> tuple[list[tuple[int, float]], str | None]:
        """``(pairs, error)``: the ranking, and why it fell back if it did.

        On failure the pairs are the first ``top_k`` documents in their
        original (retrieval) order with score 0.0, and ``error`` says why.
        """
        import requests

        if not documents:
            return [], None
        cutoff = max(1, min(top_k, len(documents)))
        fallback = [(i, 0.0) for i in range(cutoff)]
        payload = self._payload(query, documents, cutoff)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        error: str | None = None
        for attempt in range(2):
            try:
                response = requests.post(
                    self.url, json=payload, headers=headers, timeout=self.timeout
                )
            except Exception as e:  # network error, timeout
                error = f"request failed: {e}"
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                break

            if response.status_code in _RERANK_RETRY_STATUSES and attempt == 0:
                wait = 1.0
                try:
                    wait = float(response.headers.get("Retry-After", wait))
                except (TypeError, ValueError):
                    pass
                time.sleep(max(0.0, min(wait, _RERANK_MAX_RETRY_WAIT)))
                continue
            if response.status_code >= 400:
                detail = (response.text or "").strip().replace("\n", " ")[:200]
                error = f"HTTP {response.status_code}: {detail}" if detail else f"HTTP {response.status_code}"
                break
            try:
                ranked = self._parse(response.json(), len(documents))
            except ValueError as e:
                error = f"unreadable response: {e}"
                break
            if ranked:
                return ranked[:top_k], None
            error = "response listed no ranked documents"
            break

        logger.warning(f"Rerank via {self.label} failed ({error}); keeping retrieval order.")
        return fallback, error


# Process-wide reranker cache (issue #283).
#
# The MCP search path builds a fresh ``ZoteroSemanticSearch`` per request, so a
# reranker held on the instance (``self._reranker``) was reloaded from disk on
# *every* call — the cross-encoder load dominates at ~tens of seconds and blew
# past client timeouts. The weights are immutable for a given ``model_name``, so
# caching the loaded reranker at module scope keeps it warm across requests and
# instances. The lock prevents two concurrent first-calls from double-loading.
_RERANKER_CACHE: dict[str, CrossEncoderReranker] = {}
_RERANKER_CACHE_LOCK = threading.Lock()


def get_cached_reranker(
    model_name: str, config: dict[str, Any] | None = None
) -> "CrossEncoderReranker | APIReranker":
    """Return a process-wide cached reranker, built once per configuration.

    ``config`` is the ``reranker`` block. When it names a ``provider`` the
    reranker is a hosted :class:`APIReranker`; otherwise it is the local
    cross-encoder, as before. The cache key includes the provider so switching
    vendors mid-process cannot hand back the previous one.
    """
    provider = (config or {}).get("provider") or ""
    cache_key = f"{provider}:{model_name}" if provider else model_name
    cached = _RERANKER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with _RERANKER_CACHE_LOCK:
        # Re-check under the lock: another thread may have loaded it while we
        # waited, and the model load is far too expensive to repeat.
        cached = _RERANKER_CACHE.get(cache_key)
        if cached is None:
            if provider:
                cfg = config or {}
                cached = APIReranker(
                    model_name=model_name,
                    provider=provider,
                    base_url=cfg.get("base_url"),
                    api_key_env=cfg.get("api_key_env"),
                    instruction=cfg.get("instruction"),
                    timeout=float(cfg.get("timeout", 30.0)),
                )
            else:
                cached = CrossEncoderReranker(model_name=model_name)
            _RERANKER_CACHE[cache_key] = cached
        return cached


def _new_scoped_client(library_id: str, library_type: str, api_key: str | None, local: bool):
    """Construct a pyzotero client scoped to one library — seam for tests."""
    from pyzotero import zotero

    from zotero_mcp.client import _with_api_url

    zot = zotero.Zotero(
        library_id=library_id,
        library_type=library_type,
        api_key=api_key,
        local=local,
    )
    return zot if local else _with_api_url(zot)


class ZoteroSemanticSearch:
    """Semantic search interface for Zotero libraries using ChromaDB."""

    # Class-level fallback so instances built without __init__ (test doubles
    # do this) still resolve the attribute — None means "use the config".
    extraction_workers: int | None = None

    # Serializes every ChromaDB call made from the streaming index path, where
    # a producer thread classifies one slice while the main thread commits the
    # previous one. ChromaDB gives no concurrency guarantee for a single
    # PersistentClient, and the calls it guards are short local I/O — the
    # embedding round-trips this pipeline exists to overlap all happen outside
    # it. Class-level for the same reason as extraction_workers above; real
    # instances get their own in __init__, and sharing this one would only
    # over-serialize, never corrupt.
    _chroma_call_lock = threading.Lock()

    def __init__(
        self,
        chroma_client: ChromaClient | None = None,
        config_path: str | None = None,
        db_path: str | None = None,
        extraction_workers: int | None = None,
    ):
        """
        Initialize semantic search.

        Args:
            chroma_client: Optional ChromaClient instance
            config_path: Path to configuration file
            db_path: Optional path to Zotero database (overrides config file)
            extraction_workers: Optional parallel-extraction worker count
                (overrides ``semantic_search.extraction.workers`` in config)
        """
        self.chroma_client = chroma_client or create_chroma_client(config_path)
        self.zotero_client = get_zotero_client()
        self._chroma_call_lock = threading.Lock()
        self.config_path = config_path
        self.db_path = db_path  # CLI override for Zotero database path
        self.extraction_workers = extraction_workers  # CLI override, None = use config
        # Item keys seen by the most recent local sqlite scan (set by
        # _get_items_from_local_db); used to verify watermark promotion.
        self._last_scan_snapshot_keys: set[str] | None = None
        # Per-library clients for cross-library result enrichment (#492).
        # Keyed by group_id; None records that a library was found
        # unreachable so a batch of hits from it fails once, not per item.
        self._scoped_clients: dict[int, Any] = {}

        # Load update configuration
        self.update_config = self._load_update_config()

        # Reranker (lazy-initialized on first search)
        self._reranker: CrossEncoderReranker | None = None
        self._reranker_config = self._load_reranker_config()

        # Passage-level chunking (opt-in; default off preserves item-level
        # indexing and existing collections byte-for-byte).
        self._chunking_config = self._load_chunking_config()

    def _load_chunking_config(self) -> dict[str, Any]:
        """Load passage-chunking configuration from file or use defaults.

        When ``enabled`` is true, each item is indexed as several overlapping
        passages (id ``<item_key>#<n>``) instead of one item-level vector, so
        semantic search returns grounded passage quotes and long PDFs are
        searchable past the single-vector truncation limit. Off by default.
        """
        config: dict[str, Any] = {
            "enabled": False,
            "chunk_size": 1500,
            "overlap": 200,
            "max_chunks_per_item": 20,
        }
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    file_config = json.load(f)
                    config.update(file_config.get("semantic_search", {}).get("chunking", {}))
            except Exception as e:
                logger.warning(f"Error loading chunking config: {e}")
        return config

    @property
    def _chunking_enabled(self) -> bool:
        return bool(self._chunking_config.get("enabled", False))

    # Set per run by update_database; see _prune_stale_chunks.
    _drop_passages_on_write = False
    # The gaps a local scan should keep items for, set per run by update_database.
    _scan_gaps: dict[str, set[str]] = {}
    # Items a run only adds a missing level to, set per run by update_database.
    _level_fill_keys: set[str] = frozenset()
    # Text kept between the article and passage passes, set per run by update_database.
    _held_text: _HeldText | None = None
    # Items whose old passages could not be deleted in the article pass, set per run.
    _undropped_passages: set[str] = frozenset()
    # Legacy items this run rewrites in the item-keyed shape (#826): their
    # parent-keyed passages go once the new bibliographic record is in.
    _legacy_rewrites: set[str] = frozenset()
    # Record keys a full local scan wants indexed, for the deletion pass;
    # None when the scan did not produce a complete set.
    _scan_desired_keys: set[str] | None = None
    # A gap fill waiting for full text Zotero did not have is asked for it again
    # after this many days, doubling after each answer without text, up to the
    # cap. Zotero indexes PDFs some time after they are added.
    GAP_TEXT_RECHECK_FIRST_DAYS = 1
    GAP_TEXT_RECHECK_MAX_DAYS = 30

    # Share of the model's input limit a passage may fill. Characters per token
    # is an estimate, and the head of a passage is cut off if it overshoots.
    CHUNK_FIT_MARGIN = 0.85

    # Below this many tokens a single vector cannot hold a whole paper.
    SMALL_CONTEXT_TOKENS = 2000

    def _effective_chunk_size(self) -> int:
        """``chunk_size`` capped so a passage fits the embedding model's limit.

        The default chunk_size (1500 chars, ~350 tokens) is longer than
        all-MiniLM-L6-v2 reads (256 tokens), which would silently cut the tail
        off every passage. Models whose limit already fits are unchanged.
        """
        configured = int(self._chunking_config.get("chunk_size", 1500))
        max_tokens = getattr(self.chroma_client, "embedding_max_tokens", None)
        if not isinstance(max_tokens, (int, float)) or max_tokens <= 0:
            return configured
        ef = getattr(self.chroma_client, "embedding_function", None)
        chars_per_token = getattr(ef, "chars_per_token", 4)
        fit = max(1, int(max_tokens * chars_per_token * self.CHUNK_FIT_MARGIN))
        if fit >= configured:
            return configured
        if not getattr(self, "_chunk_fit_logged", False):
            self._chunk_fit_logged = True
            logger.info(
                f"Passage chunk_size reduced from {configured} to {fit} characters to fit "
                f"the embedding model's {int(max_tokens)}-token input limit."
            )
        return fit

    def _fulltext_fit_warning(self) -> str | None:
        """Warning for item-level fulltext indexing on a small-context model.

        Without chunking, an item's text is one vector truncated at the model's
        input limit, so a PDF's body is dropped and only the start (usually
        the abstract) stays searchable.
        """
        max_tokens = getattr(self.chroma_client, "embedding_max_tokens", None)
        if not isinstance(max_tokens, (int, float)) or max_tokens >= self.SMALL_CONTEXT_TOKENS:
            return None
        if self._chunking_enabled:
            return None
        model = getattr(self.chroma_client, "embedding_model", None) or "The embedding model"
        return (
            f"Full text is being indexed as one vector per item, and {model} takes only "
            f"{int(max_tokens)} tokens, so only the first ~{int(max_tokens)} tokens of each "
            "item (usually the abstract) are searchable. Enable semantic_search.chunking in "
            "config.json and run update-db again to index whole papers."
        )

    def _load_reranker_config(self) -> dict[str, Any]:
        """Load reranker configuration from file or use defaults."""
        return load_reranker_config(self.config_path)

    def _get_reranker(self) -> CrossEncoderReranker | None:
        """Get the reranker, reusing the process-wide cache if enabled.

        Each MCP request builds a new ``ZoteroSemanticSearch``, so the model is
        fetched from :func:`get_cached_reranker` (loaded once per process) rather
        than reloaded per instance (issue #283).
        """
        if not self._reranker_config.get("enabled", False):
            return None
        if self._reranker is None:
            model = self._reranker_config.get("model", _DEFAULT_RERANKER_CONFIG["model"])
            try:
                self._reranker = get_cached_reranker(model, self._reranker_config)
            except Exception as e:
                # A missing API key or unknown provider must not take the whole
                # search down — degrade to no re-ranking and say why once.
                logger.warning(f"Reranker unavailable ({e}); searching without it.")
                return None
        return self._reranker

    def _load_update_config(self) -> dict[str, Any]:
        """Load update configuration from file or use defaults."""
        return load_update_config(self.config_path)

    def _load_include_fulltext_setting(self) -> bool:
        """Whether to fetch fulltext via the Zotero web API during indexing.

        Defaults to True so existing users auto-upgrade to fulltext indexing on
        their next sync. Users can opt out by setting
        `semantic_search.include_fulltext: false` in the config file.
        It governs the API-based path, which runs in web and local mode
        alike; `extract_fulltext` (`update-db --fulltext`) replaces it with
        extraction from the local sqlite DB.
        """
        if not self.config_path or not os.path.exists(self.config_path):
            return True
        try:
            with open(self.config_path) as f:
                file_config = json.load(f)
                value = file_config.get("semantic_search", {}).get("include_fulltext", True)
                return bool(value)
        except Exception as e:
            logger.warning(f"Error loading include_fulltext setting: {e}")
            return True

    def _load_batch_enabled(self, provider: str) -> bool:
        """Whether Batch API indexing is enabled by config for ``provider``.

        Reads ``semantic_search.<provider>_batch.enabled`` — sibling keys
        (``openai_batch``, ``gemini_batch``), so an existing config keeps
        working and no migration is needed.
        """
        if not self.config_path or not os.path.exists(self.config_path):
            return False
        try:
            with open(self.config_path) as f:
                file_config = json.load(f)
                value = (
                    file_config
                    .get("semantic_search", {})
                    .get(f"{provider}_batch", {})
                    .get("enabled", False)
                )
                return bool(value)
        except Exception as e:
            logger.warning(f"Error loading {provider} batch setting: {e}")
            return False

    def _load_openai_batch_enabled(self) -> bool:
        """Whether OpenAI Batch API indexing is enabled by semantic config."""
        return self._load_batch_enabled("openai")

    def _load_batch_throttle_config(self, provider: str) -> dict[str, Any]:
        """Throttling limits for ``provider``'s Batch API submissions.

        ``semantic_search.<provider>_batch.batch_max_enqueued_tokens`` caps how
        many estimated tokens may sit queued with the provider at once — the
        quota whose violation surfaces as a 429 on a large library.
        ``batch_max_requests`` caps requests per uploaded JSONL file. Defaults
        are the providers' Tier 1 limits; raise them in config on higher tiers.
        """
        config: dict[str, Any] = {
            "batch_max_enqueued_tokens": _batch_adapter(provider).default_max_enqueued_tokens,
            "batch_max_requests": _batch_adapter(provider).max_requests,
        }
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    file_config = json.load(f)
                block = file_config.get("semantic_search", {}).get(f"{provider}_batch", {})
                for key in config:
                    if block.get(key) is not None:
                        config[key] = int(block[key])
            except Exception as e:
                logger.warning(f"Error loading {provider} batch throttle config: {e}")
        return config

    def _resolve_batch_enabled(self, provider: str, override: bool | None) -> bool:
        """Resolve CLI override + config default for ``provider``'s batch indexing.

        Batch mode is only active when the configured embedding model matches
        ``provider``: submission builds requests from
        ``self.chroma_client.embedding_config``, so running a provider whose
        embedding space is not the configured one would write vectors that no
        query could ever match.
        """
        requested = self._load_batch_enabled(provider) if override is None else override
        return bool(requested and self.chroma_client.embedding_model == provider)

    def _resolve_openai_batch_enabled(self, use_openai_batch: bool | None) -> bool:
        """Resolve CLI override + config default for OpenAI batch indexing."""
        return self._resolve_batch_enabled("openai", use_openai_batch)

    def _resolve_gemini_batch_enabled(self, use_gemini_batch: bool | None) -> bool:
        """Resolve CLI override + config default for Gemini batch indexing."""
        return self._resolve_batch_enabled("gemini", use_gemini_batch)

    def _resolve_batch_mode(
        self,
        use_batch: bool | None = None,
        batch_provider: str | None = None,
        use_openai_batch: bool | None = None,
        use_gemini_batch: bool | None = None,
    ) -> tuple[bool, str]:
        """Resolve which provider (if any) runs the Batch API this run.

        Returns ``(enabled, provider)``. Priority: explicit ``batch_provider``
        > explicit ``use_batch`` > the deprecated per-provider flags > config.

        An explicit ``batch_provider`` that disagrees with the configured
        embedding model raises rather than silently falling back to realtime:
        the caller asked for something that cannot be honored, and quietly
        doing something else is how a multi-hour run ends up in the wrong
        embedding space. ``use_batch=False`` still forces realtime without
        discarding an explicit provider choice.
        """
        providers_with_batch = batch_capable_providers()
        if batch_provider is not None and batch_provider not in providers_with_batch:
            raise ValueError(
                f"Unknown batch_provider {batch_provider!r}; must be one of "
                f"{providers_with_batch} (providers with Batch API support)."
            )
        if batch_provider is not None:
            if use_batch is False:
                return False, batch_provider
            if self.chroma_client.embedding_model != batch_provider:
                raise ValueError(
                    f"batch_provider={batch_provider!r} requires embedding_model "
                    f"{batch_provider!r}, but '{self.chroma_client.embedding_model}' "
                    "is configured."
                )
            return True, batch_provider
        if use_batch is not None:
            model = self.chroma_client.embedding_model
            if use_batch and model not in providers_with_batch:
                raise ValueError(
                    f"use_batch=True requires a batch-capable embedding_model; "
                    f"'{model}' has no Batch API support (supported: {providers_with_batch})."
                )
            provider = model if model in providers_with_batch else providers_with_batch[0]
            return self._resolve_batch_enabled(provider, use_batch), provider
        # Nothing explicit: per-provider config-driven resolution, which is
        # exactly the pre-existing behaviour for an OpenAI-only config.
        resolved_openai = self._resolve_batch_enabled("openai", use_openai_batch)
        resolved_gemini = self._resolve_batch_enabled("gemini", use_gemini_batch)
        provider = "openai" if resolved_openai else "gemini"
        return resolved_openai or resolved_gemini, provider

    def _client_group_id(self) -> int:
        """group_id of the library ``self.zotero_client`` is actually scoped to.

        Read off the client object itself (pyzotero stores its scope as
        ``library_type``/``library_id``, with ``library_type`` normalized to
        the plural URL form), NOT from the module-level active-library
        override: the override is mutable shared state that a
        ``zotero_switch_library`` tool call can change while a background
        update run is in flight, and an identity read at call time would
        attach the wrong library to this run's tagging, watermark and
        deletion scope. (``get_zotero_client()`` constructs a fresh pyzotero
        instance per call and switching mutates only the override, so a
        bound client's scope attributes cannot change under us.)

        A client that CLAIMS group scope but has an unparseable library_id
        raises: identity is deletion authority under the scoped deletion
        pass, and importing it from the mutable override instead would
        attach another library's identity to this client's data. Client
        doubles that carry no scope attributes at all fall back to
        ``get_active_group_id()``.
        """
        library_type = getattr(self.zotero_client, "library_type", None)
        if library_type in ("group", "groups"):
            try:
                return int(getattr(self.zotero_client, "library_id", None))
            except (TypeError, ValueError):
                raise ValueError(
                    "Cannot determine the Zotero client's group id "
                    f"(library_type={library_type!r}, library_id="
                    f"{getattr(self.zotero_client, 'library_id', None)!r}); "
                    "refusing to run a scope-sensitive update with unprovable "
                    "library identity."
                ) from None
        if library_type in ("user", "users"):
            return PERSONAL_LIBRARY_GROUP_ID
        return get_active_group_id()

    def _pinned_group_id(self) -> int:
        """The run's pinned library identity, or the live active-library
        lookup outside an update run. Every scope-sensitive read inside a
        run must go through this (or ``_active_library_key``), never through
        ``get_active_group_id()`` directly — the module-level override can
        change mid-run."""
        run_group_id = getattr(self, "_run_group_id", None)
        if run_group_id is not None:
            return run_group_id
        return get_active_group_id()

    def _active_library_key(self) -> str:
        """Config key for the library ``self.zotero_client`` is scoped to.

        Same identity as the ``group_id`` stamped on indexed documents:
        ``"0"`` is the personal library, anything else a Zotero groupID.
        Inside an update run this is the run's pinned library identity
        (see ``_client_group_id``).
        """
        return str(self._pinned_group_id())

    def _migrate_legacy_sync_version(self, legacy: Any, library_key: str) -> int:
        """Interpret a pre-#393 scalar ``last_sync_version`` for one library.

        The scalar carries no record of which library produced it, so it can
        only be reused where provenance is unambiguous: when no runtime
        library override is active, the client is scoped to the
        env-configured default library, which is the only library a config
        could have been tracking across restarts (``zotero_switch_library``
        overrides live in memory and are never persisted). That covers every
        existing single-library user, who keeps their watermark and avoids a
        needless full re-scan on upgrade.

        When the library at hand is any other, the scalar is discarded: a
        redundant full scan is cheap next to trusting a foreign library's
        counter, which makes ``item_versions(since=...)`` return nothing and
        silently skips the entire library. Provenance is judged against
        ``library_key`` — the run's pinned identity — rather than a live
        read of the mutable override, which can be cleared or changed
        mid-run by a concurrent ``zotero_switch_library``.
        """
        if legacy is None:
            return 0
        env_default_group_id = PERSONAL_LIBRARY_GROUP_ID
        if os.getenv("ZOTERO_LIBRARY_TYPE", "user") == "group":
            try:
                env_default_group_id = int(os.getenv("ZOTERO_LIBRARY_ID") or 0)
            except (TypeError, ValueError):
                env_default_group_id = PERSONAL_LIBRARY_GROUP_ID
        if library_key != str(env_default_group_id):
            logger.info(
                f"Ignoring legacy last_sync_version for library {library_key}: "
                "the scalar's provenance is the env-configured default library; "
                "bootstrapping this library's own sync watermark instead."
            )
            return 0
        try:
            return int(legacy)
        except (TypeError, ValueError):
            return 0

    def _load_last_sync_version(self) -> int:
        """Last Zotero library version fully indexed into ChromaDB for the
        library the Zotero client is currently scoped to.

        Zero means "no prior successful sync for this library; bootstrap
        required". Used to drive since-based incremental ingest via
        pyzotero's `item_versions(since=V)` and `new_fulltext(since=V)`.

        Watermarks are stored per library under `last_sync_versions`, keyed
        by group_id ("0" = personal). Every Zotero library has its own
        independent, monotonically increasing version counter, so the single
        shared scalar this replaces corrupted sync state for both libraries
        after `zotero_switch_library` (#393).
        """
        if not self.config_path or not os.path.exists(self.config_path):
            return 0
        try:
            with open(self.config_path) as f:
                section = json.load(f).get("semantic_search", {}) or {}
        except Exception as e:
            logger.warning(f"Error loading last_sync_version: {e}")
            return 0

        library_key = self._active_library_key()
        versions = section.get("last_sync_versions")
        if isinstance(versions, dict):
            # The map is authoritative once written: a library absent from it
            # has never been synced, so it must bootstrap rather than inherit
            # another library's counter.
            try:
                return int(versions.get(library_key) or 0)
            except (TypeError, ValueError):
                return 0

        return self._migrate_legacy_sync_version(
            section.get("last_sync_version"), library_key
        )

    def _config_for_update(self) -> dict | None:
        """The config file for a read-modify-write, or None to skip the write.

        A file that exists but cannot be parsed (a typo from hand-editing, or
        another process caught mid-write) also holds the API key, the local
        write key and the embedding settings; rewriting it from ``{}`` would
        drop all of them, so the save is skipped and logged instead.
        """
        try:
            return _read_config_for_update(self.config_path)
        except OSError as e:
            logger.error(f"Not saving index state: {e}")
            return None

    def _save_update_config(
        self,
        last_sync_version: int | None = None,
        library_key: str | None = None,
    ) -> None:
        """Save update configuration and optionally update the sync watermark
        of ``library_key`` (defaults to the currently active library)."""
        if not self.config_path:
            return

        full_config = self._config_for_update()
        if full_config is None:
            return

        # Update semantic search config
        if "semantic_search" not in full_config:
            full_config["semantic_search"] = {}

        full_config["semantic_search"]["update_config"] = self.update_config
        if last_sync_version is not None:
            key = str(library_key) if library_key is not None else self._active_library_key()
            versions = full_config["semantic_search"].get("last_sync_versions")
            if not isinstance(versions, dict):
                versions = {}
            versions[key] = int(last_sync_version)
            full_config["semantic_search"]["last_sync_versions"] = versions
            # Back-compat mirror of the pre-#393 scalar, personal library
            # only: an older zotero-mcp (or a downgrade) reads that key and
            # applies it to whatever library it is pointed at, so a group's
            # counter must never leak into it.
            if key == str(PERSONAL_LIBRARY_GROUP_ID):
                full_config["semantic_search"]["last_sync_version"] = int(last_sync_version)

        try:
            write_json_atomic(self.config_path, full_config)
        except Exception as e:
            logger.error(f"Error saving update config: {e}")

    def _load_index_schema_version(self) -> int:
        """Schema version of the persisted ChromaDB collection's metadata shape."""
        if not self.config_path or not os.path.exists(self.config_path):
            return 0
        try:
            with open(self.config_path) as f:
                value = json.load(f).get("semantic_search", {}).get("index_schema_version", 0)
                return int(value) if value is not None else 0
        except Exception as e:
            logger.warning(f"Error loading index_schema_version: {e}")
            return 0

    def _save_index_schema_version(self, version: int) -> None:
        """Record that the collection's metadata now matches ``version``."""
        if not self.config_path:
            return
        full_config = self._config_for_update()
        if full_config is None:
            return
        full_config.setdefault("semantic_search", {})["index_schema_version"] = int(version)
        try:
            write_json_atomic(self.config_path, full_config)
        except Exception as e:
            logger.error(f"Error saving index_schema_version: {e}")

    def _load_backfill_unattributed(self) -> int:
        """Count of docs the last backfill could not attribute to a library."""
        if not self.config_path or not os.path.exists(self.config_path):
            return 0
        try:
            with open(self.config_path) as f:
                value = json.load(f).get("semantic_search", {}).get("backfill_unattributed", 0)
                return int(value) if value else 0
        except Exception:
            return 0

    def _save_backfill_unattributed(self, count: int) -> None:
        """Persist the unattributed-doc count so later updates keep warning."""
        if not self.config_path:
            return
        full_config = self._config_for_update()
        if full_config is None:
            return
        section = full_config.setdefault("semantic_search", {})
        if count:
            section["backfill_unattributed"] = int(count)
        else:
            section.pop("backfill_unattributed", None)
        try:
            write_json_atomic(self.config_path, full_config)
        except Exception as e:
            logger.error(f"Error saving backfill_unattributed: {e}")

    def _backfill_group_ids(self) -> dict[str, int]:
        """One-time metadata-only migration: tag pre-#163 docs with ``group_id``.

        Docs indexed before #163 carry no ``group_id`` metadata key. This
        pages through the collection and attaches one via
        ``ChromaClient.update_metadatas()`` — metadata-only, so no
        re-embedding and no doc-id change.

        Attribution is strictly evidence-based, because the deletion pass is
        scoped by ``group_id`` and a guessed tag is a future deletion
        warrant. Evidence, in order:

        1. ``LocalZoteroReader.get_key_group_map()`` (local mode): ground
           truth from the live database, trashed items included so trash is
           attributed to its true library.
        2. Membership in the active library's ``item_versions()`` — fetched
           lazily, at most once per run. In local mode this also covers
           WAL-fresh items the immutable sqlite read cannot see (#292); a
           fetch failure propagates so the caller retries next update rather
           than tagging by guesswork.
        3. No evidence → left untagged and counted. Untagged docs are
           excluded from library-filtered search and can never match the
           deletion pass's ``group_id`` filter.

        Idempotent: docs that already carry ``group_id`` are left untouched,
        so a partial/interrupted run just resumes on the next call.

        Returns ``{"scanned": N, "migrated": N, "unattributed": N}``.
        """
        stats = {"scanned": 0, "migrated": 0, "unattributed": 0}

        key_group_map: dict[str, int] | None = None
        if is_local_mode():
            try:
                zotero_db_path = self.db_path
                if not zotero_db_path and self.config_path and os.path.exists(self.config_path):
                    with open(self.config_path) as f:
                        zotero_db_path = (
                            json.load(f).get("semantic_search", {}).get("zotero_db_path")
                        )
                with LocalZoteroReader(db_path=zotero_db_path) as reader:
                    key_group_map, _ = reader.get_key_group_map()
            except Exception as e:
                logger.warning(
                    f"group_id backfill: could not read local database, "
                    f"falling back to active-library-membership attribution: {e}"
                )
                key_group_map = None

        active_group_id = self._pinned_group_id()
        active_library_keys: set[str] | None = None

        for ids, metadatas in self.chroma_client.iter_metadatas():
            stats["scanned"] += len(ids)
            update_ids: list[str] = []
            update_metas: list[dict[str, Any]] = []
            for doc_id, meta in zip(ids, metadatas):
                meta = dict(meta or {})
                if "group_id" in meta:
                    continue
                item_key = meta.get("item_key") or record_key(doc_id)
                if key_group_map is not None and item_key in key_group_map:
                    group_id = key_group_map[item_key]
                else:
                    if active_library_keys is None:
                        active_library_keys = set(
                            (self.zotero_client.item_versions() or {}).keys()
                        )
                        if not active_library_keys:
                            # The same HTTP-200-but-empty response shape the
                            # deletion pass treats as an API fault. Accepting
                            # it as negative evidence would mark the one-time
                            # migration complete with these docs unattributed
                            # — permanently, once the schema gate closes.
                            raise Exception(
                                "item_versions() returned no items while indexed "
                                "documents need library-membership evidence; "
                                "treating this as an API fault. The backfill "
                                "will retry on the next update."
                            )
                    if item_key in active_library_keys:
                        group_id = active_group_id
                    else:
                        stats["unattributed"] += 1
                        continue
                meta["group_id"] = int(group_id)
                update_ids.append(doc_id)
                update_metas.append(meta)
            if update_ids:
                self.chroma_client.update_metadatas(update_ids, update_metas)
                stats["migrated"] += len(update_ids)

        return stats

    def _create_document_text(self, item: dict[str, Any]) -> str:
        """
        Create searchable text from a Zotero item.

        Args:
            item: Zotero item dictionary

        Returns:
            Combined text for embedding
        """
        data = item.get("data", {})
        item_type = data.get("itemType", "")

        # Annotations have no title / creators / abstract — they have
        # ``annotationText`` (the highlighted passage) and an optional
        # ``annotationComment``. The previous "title + creators + abstract"
        # template fell back to ``format_creators([])`` → "No authors
        # listed", which then embedded identically for every annotation in
        # the library, collapsing them all to a single vector and
        # dominating every semantic-search result (#287).
        if item_type == "annotation":
            return self._create_annotation_document_text(data)

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

            note_text = re.sub(r"<[^>]+>", "", note)
            extra_fields.append(note_text)

        # Combine all text fields
        text_parts = [title, creators_text, abstract] + extra_fields
        return " ".join(filter(None, text_parts))

    def _create_annotation_document_text(self, data: dict[str, Any]) -> str:
        """Build the embedding text for an annotation item.

        Combines ``annotationText`` (highlighted passage) and
        ``annotationComment`` (user's commentary), plus any tags. Returns
        the empty string when nothing meaningful is present so the caller
        can decide to skip the item rather than embedding noise.
        """
        parts: list[str] = []
        if highlighted := (data.get("annotationText") or "").strip():
            parts.append(highlighted)
        if comment := (data.get("annotationComment") or "").strip():
            parts.append(comment)
        if tags := data.get("tags"):
            tag_text = " ".join(t.get("tag", "") for t in tags if t.get("tag"))
            if tag_text:
                parts.append(tag_text)
        return " ".join(parts)

    def _create_metadata(self, item: dict[str, Any]) -> dict[str, Any]:
        """Metadata of a regular item's bibliographic record: its own state only.

        ``item_key``, ``group_id`` and ``date_modified`` (#826). Everything
        bibliographic is resolved live from Zotero at search time, so nothing
        stored here can go stale; ``date_modified`` is the record's version
        stamp for change detection.
        """
        data = item.get("data", {})
        return _own_metadata(item.get("key", ""), data.get("group_id"), data.get("dateModified"))

    def should_update_database(self) -> bool:
        """Check if the database should be updated based on configuration."""
        return should_update(self.update_config)

    def _get_items_from_source(
        self,
        limit: int | None = None,
        extract_fulltext: bool = False,
        chroma_client: ChromaClient | None = None,
        force_rebuild: bool = False,
        include_fulltext_via_api: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Get items from either local database or API.

        When extract_fulltext=True, requires local mode (ZOTERO_LOCAL=true);
        raises RuntimeError if local mode is not enabled. This path reads the
        local Zotero sqlite database and extracts PDF text on-disk.

        When include_fulltext_via_api=True (web or local API), fetches the
        server-side extracted fulltext that Zotero cloud has already built
        for each PDF — no local files required.

        Otherwise uses API metadata only (fastest, title/abstract/tags).

        Args:
            limit: Optional limit on number of items
            extract_fulltext: Whether to extract fulltext from the local sqlite DB
            chroma_client: ChromaDB client to check for existing documents (None to skip checks)
            force_rebuild: Whether to force extraction even if item exists
            include_fulltext_via_api: Fetch fulltext via the Zotero web API

        Returns:
            List of items in API-compatible format
        """
        if extract_fulltext:
            if not is_local_mode():
                raise RuntimeError(
                    "Fulltext extraction requires local mode but ZOTERO_LOCAL is not enabled. "
                    "Set ZOTERO_LOCAL=true or run 'zotero-mcp setup' to enable local mode."
                )
            return self._get_items_from_local_db(
                limit, extract_fulltext=extract_fulltext, chroma_client=chroma_client, force_rebuild=force_rebuild
            )
        else:
            return self._get_items_from_api(limit, include_fulltext=include_fulltext_via_api)

    def _get_items_from_local_db(
        self,
        limit: int | None = None,
        extract_fulltext: bool = False,
        chroma_client: ChromaClient | None = None,
        force_rebuild: bool = False,
    ) -> list[dict[str, Any]]:
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
            attachment_priority = None
            zotero_db_path = self.db_path  # CLI override takes precedence
            collection_keys = None
            config_workers = None
            # If semantic_search config file exists, prefer its setting
            try:
                if self.config_path and os.path.exists(self.config_path):
                    with open(self.config_path) as _f:
                        _cfg = json.load(_f)
                        semantic_cfg = _cfg.get("semantic_search", {})
                        extraction_cfg = semantic_cfg.get("extraction", {})
                        pdf_max_pages = extraction_cfg.get("pdf_max_pages")
                        attachment_priority = extraction_cfg.get("attachment_priority")
                        config_workers = extraction_cfg.get("workers")
                        collection_keys = semantic_cfg.get("collection_keys")
                        # Use config db_path only if no CLI override
                        if not zotero_db_path:
                            zotero_db_path = semantic_cfg.get("zotero_db_path")
            except Exception:
                pass

            # CLI flag beats config; 1 (fully sequential) when neither is set.
            # Never exceed the core count — extraction is CPU-bound, so extra
            # workers only add process-spawn and scheduling overhead.
            workers = self.extraction_workers or config_workers or 1
            workers = max(1, min(int(workers), os.cpu_count() or 1))

            with (
                suppress_stdout(),
                LocalZoteroReader(
                    db_path=zotero_db_path,
                    pdf_max_pages=pdf_max_pages,
                    attachment_priority=attachment_priority,
                    extraction_workers=workers,
                    # The indexing path is the one place the transient cache
                    # is safe to write: it extracts under the *indexing* page
                    # cap, which is what a later run will want to reuse.
                    fulltext_cache_enabled=True,
                    config_path=self.config_path,
                ) as reader,
            ):
                # Capture the snapshot's full key set on the SAME connection
                # this scan uses. The staleness check after the (potentially
                # long) extraction must compare against what this scan could
                # actually see — a fresh read taken later could already
                # include rows from a WAL checkpoint that landed mid-scan.
                self._last_scan_snapshot_keys = reader.get_all_item_keys()
                # Library attribution (#163): map every item key to its
                # group_id (0 = personal, else Zotero groupID) via direct SQL
                # on the same connection this scan uses. Feed/publications
                # items have no group_id equivalent and are dropped below
                # rather than mis-tagged as personal.
                key_group_map, excluded_keys = reader.get_key_group_map()
                # Phase 1: fetch metadata only (fast)
                sys.stderr.write("Scanning local Zotero database for items...\n")
                if collection_keys:
                    sys.stderr.write(f"Filtering to collections: {collection_keys}\n")
                local_items = reader.get_items_with_text(limit=limit, include_fulltext=False, collection_keys=collection_keys)
                if excluded_keys:
                    local_items = [it for it in local_items if it.key not in excluded_keys]
                candidate_count = len(local_items)
                sys.stderr.write(f"Found {candidate_count} candidate items.\n")

                # Optional deduplication: if preprint and journalArticle share a DOI/title, keep journalArticle
                # Build index by the shared doi/title match keys (#496): same
                # rule the duplicate detector uses, so the two agree on what
                # counts as the same work.
                def _work_keys(it):
                    return metadata_match_keys(it, kinds=("doi", "title"))

                key_to_best = {}
                for it in local_items:

                    def consider(k):
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

                    for k in _work_keys(it):
                        consider(k)

                # If a preprint loses against a journal article for same DOI/title, drop it
                filtered_items = []
                for it in local_items:
                    # If there is a journalArticle alternative for same DOI or title, and this is preprint, drop
                    if getattr(it, "item_type", None) == "preprint":
                        drop = False
                        for k in _work_keys(it):
                            best = key_to_best.get(k)
                            if (
                                best is not None
                                and best is not it
                                and getattr(best, "item_type", None) == "journalArticle"
                            ):
                                drop = True
                                break
                        if drop:
                            continue
                    filtered_items.append(it)

                local_items = filtered_items
                total_to_extract = len(local_items)
                if total_to_extract != candidate_count:
                    try:
                        sys.stderr.write(
                            f"After filtering/dedup: {total_to_extract} items to process. Extracting content...\n"
                        )
                    except Exception:
                        pass
                else:
                    try:
                        sys.stderr.write("Extracting content...\n")
                    except Exception:
                        pass

                if extract_fulltext:
                    units = self._scan_local_units(
                        reader,
                        local_items,
                        key_group_map,
                        excluded_keys,
                        chroma_client=chroma_client,
                        force_rebuild=force_rebuild,
                        collection_keys=collection_keys,
                        limit=limit,
                    )
                else:
                    units = [self._local_item_unit(it, key_group_map.get(it.key)) for it in local_items]

                logger.info(f"Retrieved {len(units)} work units from local database")
                return units

        except Exception as e:
            logger.error(f"Error reading from local database: {e}")
            logger.info("Falling back to API...")
            return self._get_items_from_api(limit)

    def _local_item_unit(self, item: Any, group_id: int | None) -> dict[str, Any]:
        """A regular item from the local database as a bibliographic unit."""
        unit = _item_unit({
            "key": item.key,
            "version": 0,  # Local items don't have versions
            "data": {
                "key": item.key,
                "itemType": getattr(item, "item_type", None) or "journalArticle",
                "title": item.title or "",
                "abstractNote": item.abstract or "",
                "extra": item.extra or "",
                "dateAdded": item.date_added,
                "dateModified": item.date_modified,
                "creators": self._parse_creators_string(item.creators) if item.creators else [],
                # Library attribution (#163): 0 = personal, else groupID.
                # Ground truth from get_key_group_map(); an item missing from
                # the map (e.g. added mid-scan) stays UNATTRIBUTED — a guessed
                # "personal" would be positive attribution minted from
                # nothing, i.e. deletion authority. The next sync re-tags it.
                "group_id": group_id,
            },
        })
        if item.notes:
            unit["data"]["notes"] = item.notes
        return unit

    @staticmethod
    def _legacy_changed(legacy: dict[str, Any], item: Any, att_keys: str, priority_tag: str) -> bool:
        """Whether a legacy item changed since its records were written.

        The criteria those records were written under: library attribution,
        the item's ``date_modified``, the attachment-key fingerprint (attaching
        a file does not bump the parent's date), the "failed" marker, and the
        attachment priority (#378, #428, #446). An unchanged legacy item keeps
        its records until ``--force-rebuild`` (#826, decision 1).
        """
        if "group_id" not in legacy:
            return True
        if not _same_moment(legacy.get("date_modified"), getattr(item, "date_modified", "")):
            return True
        has_fulltext = legacy.get("has_fulltext", False)
        stored_keys = legacy.get("attachment_keys")
        if has_fulltext == "failed":
            # Records without attachment_keys retry once, then converge.
            return stored_keys != att_keys
        if has_fulltext and not att_keys:
            # Its text-bearing attachments are gone (#428).
            return True
        if has_fulltext and stored_keys is not None and stored_keys != att_keys:
            return True
        if not has_fulltext and att_keys:
            return True
        return _attachment_priority_changed(legacy, priority_tag)

    def _scan_local_units(
        self,
        reader: Any,
        local_items: list[Any],
        key_group_map: dict[str, int],
        excluded_keys: set[str],
        *,
        chroma_client: Any,
        force_rebuild: bool,
        collection_keys: list[str] | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        """The work units a local ``--fulltext`` scan needs written (#826).

        Every indexable regular item is a bibliographic unit, and every file
        ``attachment_priority`` selects for it an attachment unit; so is every
        standalone file in a readable category. A unit is written when it has
        no record, or its record is stale: a bibliographic record by
        ``date_modified``, an attachment record by ``file_stamp``. A file whose
        extraction failed at this stamp is not tried again until it changes.
        A legacy item is left alone until it changes; then it is rewritten.

        Also sets ``_scan_desired_keys``, every record key this library should
        have, for the deletion pass, when the scan covered the whole library.
        """
        from . import file_hash

        check = chroma_client is not None and not force_rebuild
        priority_tag = ",".join(reader.attachment_priority)
        dates = reader.get_date_modified_by_key()
        desired: set[str] = set()
        # Units in write order, a regular item's files right after it; files
        # still to extract hold their target here until then.
        planned: list[tuple[dict[str, Any], Any]] = []
        up_to_date = 0
        skipped_failed: list[str] = []
        # Targets whose record is current and only lacks a level: a failed
        # extraction of one must not drop what is already paid for.
        current_keys: set[str] = set()
        # Stored metadata read ahead for a slice of items in few batched gets
        # (#368), instead of two or three single gets per item.
        prefetched: dict[str, dict[str, Any]] = {}
        passage_keys: set[str] | None = None
        if check:
            try:
                passage_keys = chroma_client.item_keys_by_level()[1]
            except Exception:
                passage_keys = None  # unknown: probe every key's first passage

        def prefetch(keys: list[str]) -> None:
            """Load the stored state of ``keys`` (and first passages of those that have any)."""
            ids = list(keys)
            ids += [f"{k}#0" for k in keys if passage_keys is None or k in passage_keys]
            batched = getattr(chroma_client, "get_metadatas", None)
            if batched is not None:
                prefetched.update(batched(ids))
            else:
                for doc_id in ids:
                    meta = chroma_client.get_document_metadata(doc_id)
                    if meta is not None:
                        prefetched[doc_id] = meta

        def stored_of(key: str) -> dict[str, Any] | None:
            """The record of ``key``, else its first passage, as ``get_document_metadata`` finds it."""
            return prefetched.get(key) or prefetched.get(f"{key}#0")

        def queue_file(target: Any, *, group_id: int | None, owner: str | None, label: str,
                       date_modified: str | None, rewrite: bool) -> None:
            nonlocal up_to_date
            unit = _attachment_unit(
                target.key, "", file_stamp=target.file_stamp, date_modified=date_modified,
                group_id=group_id, owner_key=owner, label=label,
            )
            gap = self._scan_gaps.get(target.key) if check else None
            stored = stored_of(target.key) if check and not rewrite else None
            current = self._unit_is_current(unit, stored)
            if current:
                if not gap:
                    up_to_date += 1
                    return
                unit["_index_levels"] = gap
                current_keys.add(target.key)
            elif stored and "chunk_index" in stored and gap == {SEARCH_ARTICLES}:
                # Its passages are there; only the attachment record is missing.
                unit["_index_levels"] = gap
            # A forced rebuild tries every file again.
            if check and fulltext_cache.failed(target.key, target.file_stamp, config_path=self.config_path):
                skipped_failed.append(label or target.key)
                if not current:
                    # A record of an older version of the file is stale.
                    desired.discard(target.key)
                return
            planned.append((unit, target))

        _extract_logger = logging.getLogger("zotero_mcp.extract")
        _prev_level = _extract_logger.level
        # A warning about one unreadable attachment would otherwise land in
        # the middle of the \r progress line.
        _extract_logger.setLevel(logging.CRITICAL)
        try:
            total = len(local_items)
            resolved: dict[str, list[Any]] = {}
            for item_idx, it in enumerate(local_items, 1):
                display = self._scan_display(it)
                self._scan_progress(item_idx, total, up_to_date, len(planned), display)

                if (item_idx - 1) % _SCAN_SLICE == 0:
                    resolved = {
                        other.key: reader._resolve_extraction_targets(other.item_id)
                        for other in local_items[item_idx - 1:item_idx - 1 + _SCAN_SLICE]
                    }
                    if check:
                        prefetch([
                            k for key, found in resolved.items()
                            for k in (key, *(t.key for t in found))
                        ])

                group_id = key_group_map.get(it.key)
                bib = self._local_item_unit(it, group_id)
                targets = resolved[it.key]
                desired.add(it.key)
                desired.update(target.key for target in targets)

                existing = stored_of(it.key) if check else None
                legacy = (
                    self._legacy_metadata(it.key, existing, passage=prefetched.get(f"{it.key}#0"))
                    if existing else None
                )
                rewrite = False
                if legacy is not None:
                    gap = self._scan_gaps.get(it.key)
                    att_keys = ",".join(
                        sorted(k for k, _p, _c in reader.get_fulltext_meta_for_item(it.item_id))
                    )
                    # Only text has passages: a legacy item indexed without
                    # any (none found, or extraction failed) lacks none.
                    lacks_passages = gap == {SEARCH_PASSAGES} and legacy.get("has_fulltext") is True
                    if self._legacy_changed(legacy, it, att_keys, priority_tag) or lacks_passages:
                        bib["legacy_rewrite"] = rewrite = True
                    elif gap == {SEARCH_ARTICLES}:
                        # Only its article record is missing: write that, and
                        # leave its passages, which cost money, as they are.
                        bib["_index_levels"] = gap
                        planned.append((bib, None))
                        continue
                    else:
                        up_to_date += 1
                        continue
                if rewrite or not self._unit_is_current(bib, existing):
                    planned.append((bib, None))
                else:
                    up_to_date += 1
                for target in targets:
                    queue_file(
                        target, group_id=group_id, owner=it.key, label=display,
                        date_modified=dates.get(target.key), rewrite=rewrite,
                    )

            if limit is None:
                # Standalone files, in the configured collections if any.
                members = reader.get_collection_item_keys(collection_keys) if collection_keys else None
                standalone = []
                for key, path, ctype, date_modified, _library_id in reader._iter_standalone_attachments():
                    if key in excluded_keys or (members is not None and key not in members):
                        continue
                    target = reader._resolve_standalone_target(key, path, ctype)
                    if target is not None:
                        standalone.append((key, target, date_modified))
                for start in range(0, len(standalone), _SCAN_SLICE):
                    part = standalone[start:start + _SCAN_SLICE]
                    if check:
                        prefetch([key for key, _t, _d in part])
                    for key, target, date_modified in part:
                        desired.add(key)
                        queue_file(
                            target, group_id=key_group_map.get(key), owner=None,
                            label=Path(target.path).name, date_modified=date_modified, rewrite=False,
                        )

            # Parse the files. With extraction_workers == 1 a sequential
            # walk; above 1 a process pool, results arriving out of order.
            pending = {unit["key"]: (unit, target) for unit, target in planned if target is not None}
            failed_now = 0
            done = 0
            for att_key, result in reader.extract_targets([target for _u, target in pending.values()]):
                unit, target = pending[att_key]
                if result and (result[0] or "").strip():
                    unit["text"] = result[0]
                    unit["file_hash"] = file_hash.get_file_hash(
                        att_key, target.path, target.file_stamp, config_path=self.config_path
                    )
                elif att_key in current_keys:
                    # A gap fill of a record that is current: the file read at
                    # this stamp before, so the failure is probably transient.
                    # Keep the paid-for record and leave no failure marker, or
                    # the gap would stay open until the file changes.
                    pass
                else:
                    # Not tried again until the file changes; a stale record
                    # of it is outside the desired set and goes.
                    fulltext_cache.record_failure(att_key, target.file_stamp, config_path=self.config_path)
                    desired.discard(att_key)
                    failed_now += 1
                done += 1
                if done % 10 == 0 or done == len(pending):
                    _report(f"\r  Extracting text: {done}/{len(pending)}{' ' * 40}")
        finally:
            _extract_logger.setLevel(_prev_level)

        units = [unit for unit, target in planned if target is None or unit["text"]]
        try:
            sys.stderr.write(f"\r{' ' * _term_width(120)}\r")
            parts = [f"  Extraction complete: {len(units)} records to write"]
            if up_to_date:
                parts.append(f"{up_to_date} already up to date")
            sys.stderr.write(", ".join(parts) + "\n")
            if failed_now:
                sys.stderr.write(
                    f"  {failed_now} attachment(s) produced no text "
                    "(not retried until the file changes)\n"
                )
            if skipped_failed:
                sys.stderr.write(
                    f"  {len(skipped_failed)} attachment(s) skipped (extraction previously failed):\n"
                )
                for name in skipped_failed[:5]:
                    sys.stderr.write(f"    - {name}\n")
                if len(skipped_failed) > 5:
                    sys.stderr.write(f"    ... and {len(skipped_failed) - 5} more\n")
                sys.stderr.write("  (To retry these, replace the file, or run with --force-rebuild)\n")
        except Exception:
            pass

        # A limited or collection-filtered scan sees part of the library: its
        # set would delete the rest. Those runs keep item_versions().
        if limit is None and not collection_keys:
            self._scan_desired_keys = desired
        return units

    @staticmethod
    def _scan_display(item: Any) -> str:
        """``Author (Year) — Title`` for the scan's progress line."""
        title = getattr(item, "title", "") or ""
        creators = getattr(item, "creators", "") or ""
        date = getattr(item, "date_added", "") or ""
        first_author = ""
        if creators:
            first_author = creators.split(";")[0].split(",")[0].strip()
            if first_author:
                first_author += " et al." if ";" in creators else ""
        year = date[:4] if date and len(date) >= 4 else ""
        if first_author and year:
            citation = f"{first_author} ({year}) — "
        elif first_author:
            citation = f"{first_author} — "
        else:
            citation = ""
        display = f"{citation}{title}"
        return display[:57] + "..." if len(display) > 60 else display

    @staticmethod
    def _scan_progress(index: int, total: int, up_to_date: int, to_write: int, display: str) -> None:
        """One \r-overwritten progress line, clipped to the terminal width."""
        try:
            max_len = _term_width() - 1
            status_parts = []
            if up_to_date:
                status_parts.append(f"{up_to_date} up to date")
            if to_write:
                status_parts.append(f"{to_write} to write")
            status = f" ({', '.join(status_parts)})" if status_parts else ""
            line = _clip_to_width(f"  Processing {index}/{total}{status} — {display or 'working...'}", max_len)
            sys.stderr.write(f"\r{line}{' ' * max(0, max_len - _display_cols(line))}")
            sys.stderr.flush()
        except Exception:
            pass

    def _find_index_gaps(
        self, *, limit: int | None, force_rebuild: bool, fill_passages: bool = True
    ) -> dict[str, set[str]]:
        """Items that lack a level this index should have: ``{key: {levels}}``.

        Enabling chunking leaves every item without passages, and a migration
        leaves every item without an article record. Neither shows up as a
        library change, so the sync watermark cannot find them.
        """
        self._passage_gaps_held = 0
        if limit is not None or force_rebuild or not self._separate_passages():
            return {}
        articles, passages = self.chroma_client.item_keys_by_level()
        gaps: dict[str, set[str]] = {key: {SEARCH_ARTICLES} for key in passages - articles}
        # An automatic run (startup, pre-search) leaves passage gaps alone:
        # after chunking is switched on that is the whole library, which a paid
        # embedding API bills; it waits for an update the user starts.
        # Only attachment and legacy records have passages to lack (#826); a
        # bibliographic record never has any.
        bearing = getattr(self.chroma_client, "passage_bearing_keys", None)
        eligible = articles & bearing() if bearing else articles
        missing = eligible - passages if self._chunking_enabled else set()
        self._passage_gaps_held = 0 if fill_passages else len(missing)
        if fill_passages:
            for key in missing:
                gaps[key] = {SEARCH_PASSAGES}
        return gaps

    def _get_gap_items_from_api(
        self,
        gaps: dict[str, set[str]],
        *,
        already_fetched: set[str],
        library_keys: set[str] | None,
        include_fulltext: bool,
    ) -> list[dict[str, Any]]:
        """Fetch the items in ``gaps`` that belong to this library, marked with what they lack.

        Items of other libraries wait for a run against their own library, and
        with the library's keys unknown there is nothing to go on this run.
        """
        if library_keys is None:
            return []
        waiting = set(self.update_config.get("gaps_waiting_for_text") or [])
        skipped = self._still_waiting_for_text(include_fulltext)
        items: list[dict[str, Any]] = []
        for key in sorted(gaps):
            if key in already_fetched or key in skipped:
                continue
            if key not in library_keys:
                # Another library's item: its own library's run asks about it.
                continue
            if not include_fulltext:
                # No run without text can fill this; don't fetch the item to find out.
                existing = self.chroma_client.get_document_metadata(key) or {}
                if _indexed_with_text(existing):
                    waiting.add(key)
                    continue
            try:
                item = self.zotero_client.item(key)
            except Exception as e:
                logger.debug(f"item({key}) failed while filling index gaps: {e}")
                self._note_unanswered_recheck(key, waiting)
                continue
            if (item or {}).get("data", {}).get("itemType") == "attachment" and include_fulltext:
                # An attachment record's missing level comes from its own text.
                items.append(self._attachment_gap_unit(key, item, gaps[key]))
                continue
            if not item or item.get("data", {}).get("itemType") in _NON_INDEXED_ITEM_TYPES:
                self._note_unanswered_recheck(key, waiting)
                continue
            item["_index_levels"] = gaps[key]
            items.append(item)
        api_items = [item for item in items if not item.get("unit")]
        if include_fulltext and api_items:
            self._attach_web_fulltext(api_items)
        self._tag_group_id(api_items)
        self.update_config["gaps_waiting_for_text"] = sorted(waiting)
        return self._without_fills_lacking_text(items)

    def _attachment_gap_unit(
        self, key: str, item: dict[str, Any], levels: set[str]
    ) -> dict[str, Any]:
        """An indexed attachment's unit for a gap fill, with the text Zotero has for it.

        Transitional (#826): the API paths learn to select attachments in a
        later change. Only the missing level is written while the file is the
        one the stored record describes; a changed file is written in full.
        """
        data = item.get("data") or {}
        text, failed = "", False
        try:
            resp = self.zotero_client.fulltext_item(key)
            text = str(resp.get("content", "") or "") if isinstance(resp, dict) else (
                resp if isinstance(resp, str) else ""
            )
        except ResourceNotFoundError:
            pass
        except Exception as e:
            logger.debug(f"fulltext_item({key}) failed while filling index gaps: {e}")
            failed = True
        unit = _attachment_unit(
            key,
            text,
            file_stamp=_api_file_stamp(item),
            date_modified=data.get("dateModified"),
            group_id=self._pinned_group_id(),
            owner_key=data.get("parentItem"),
            label=data.get("title", ""),
        )
        existing = self.chroma_client.get_document_metadata(key) or {}
        if existing.get("file_stamp") == unit["file_stamp"]:
            unit["_index_levels"] = levels
        elif "chunk_index" in existing and levels == {SEARCH_ARTICLES}:
            # Its passages are there, and carry no stamp: only the record is missing.
            unit["_index_levels"] = levels
        unit["fulltext_attempted"] = not text.strip()
        unit["fulltext_fetch_failed"] = failed
        return unit

    @staticmethod
    def _local_time(value: Any) -> datetime | None:
        """A stored ISO time as naive local time, or None when it can't be read."""
        try:
            moment = datetime.fromisoformat(str(value))
            if moment.tzinfo is not None:
                moment = moment.astimezone().replace(tzinfo=None)
        except (TypeError, ValueError, OverflowError, OSError):
            return None
        return moment

    @staticmethod
    def _check_count(check: Any) -> int:
        """How many times Zotero answered a waiting item without text; 0 if unreadable."""
        try:
            return max(int(check.get("checks") or 0), 0) if isinstance(check, dict) else 0
        except (TypeError, ValueError, OverflowError):
            return 0

    def _text_checks(self) -> dict[str, Any]:
        checks = self.update_config.get("gaps_text_checked")
        return dict(checks) if isinstance(checks, dict) else {}

    @classmethod
    def _text_recheck_due(cls, check: Any, now: datetime) -> bool:
        """Whether a waiting item with the stored ``check`` may be asked for its text again.

        ``at`` and ``checks`` record Zotero's answers without text, and set the
        backoff. ``tried_at`` records a try that got no answer (a failed
        request): it doesn't lengthen the backoff, but the item isn't tried
        again for a day. A record that can't be read is due, and so is one
        stamped in the future, which no run could have written.
        """
        if not isinstance(check, dict):
            return True
        tried = cls._local_time(check.get("tried_at"))
        if tried is not None and timedelta(0) <= now - tried < timedelta(days=cls.GAP_TEXT_RECHECK_FIRST_DAYS):
            return False
        at = cls._local_time(check.get("at"))
        if at is None or at > now:
            return True
        checks = max(cls._check_count(check), 1)
        wait = min(cls.GAP_TEXT_RECHECK_FIRST_DAYS * 2 ** min(checks - 1, 16), cls.GAP_TEXT_RECHECK_MAX_DAYS)
        return now - at >= timedelta(days=wait)

    def _note_unanswered_recheck(self, key: str, waiting: set[str]) -> None:
        """Record that a waiting item was due but this run got no answer about its text."""
        if key not in waiting:
            return
        checks = self._text_checks()
        previous = checks.get(key) if isinstance(checks.get(key), dict) else {}
        checks[key] = {**previous, "tried_at": datetime.now().isoformat()}
        self.update_config["gaps_text_checked"] = checks

    def _still_waiting_for_text(self, include_fulltext: bool) -> set[str]:
        """Waiting items this run leaves alone.

        A run without full text can't fill any of them. A run with it asks
        Zotero again for each item whose backoff has run out.
        """
        waiting = set(self.update_config.get("gaps_waiting_for_text") or [])
        if not include_fulltext:
            return waiting
        checks = self._text_checks()
        now = datetime.now()
        not_due = {key for key in waiting if not self._text_recheck_due(checks.get(key), now)}
        return not_due | self._other_library_keys(waiting - not_due)

    def _other_library_keys(self, keys: set[str]) -> set[str]:
        """Those of ``keys`` the index records as another library's items.

        The waiting list is shared by every library in the index. This run
        can't ask Zotero about another library's item, so it neither counts
        it as work nor touches its record: that library's own run does.
        """
        if not keys:
            return set()
        lookup = getattr(self.chroma_client, "item_group_ids", None)
        groups = lookup(sorted(keys)) if lookup else {}
        if not isinstance(groups, dict):
            return set()
        own = self._pinned_group_id()
        return {key for key, group_id in groups.items() if group_id != own}

    def _without_fills_lacking_text(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop fills that would write metadata-only records for an item with full text.

        The item's other level was built from its full text, which this run
        does not have, say because the text came from a local ``--fulltext``
        run. Filling from metadata would close the gap for good with a record
        that cannot match the body of the paper. The gap stays, and the item
        goes on the waiting list: runs without full text leave it alone, and
        runs with full text ask Zotero again once its backoff runs out
        (``_text_recheck_due``). A failed request parks nothing.
        """
        kept: list[dict[str, Any]] = []
        waiting = set(self.update_config.get("gaps_waiting_for_text") or [])
        for item in items:
            # An attachment unit carries its text and fetch flags itself.
            data = item if _is_attachment_unit(item) else item.get("data", {})
            text = item.get("text") if _is_attachment_unit(item) else data.get("fulltext")
            key = item.get("key")
            # A regular item's missing article record is its bibliographic
            # record, built from metadata alone (#826): only passages and an
            # attachment's records need its text.
            levels = item.get("_index_levels") or set()
            needs_text = _is_attachment_unit(item) or SEARCH_PASSAGES in levels
            if levels and needs_text and not (text or "").strip():
                existing = self.chroma_client.get_document_metadata(key) or {}
                if _indexed_with_text(existing):
                    if data.get("fulltext_fetch_failed"):
                        # A failed fetch says nothing about the text: the
                        # item is left for the next run, not parked.
                        self._note_unanswered_recheck(key, waiting)
                        continue
                    waiting.add(key)
                    if data.get("fulltext_attempted"):
                        # Zotero answered, without text: wait longer next time.
                        checks = self._text_checks()
                        checks[key] = {
                            "at": datetime.now().isoformat(),
                            "checks": self._check_count(checks.get(key)) + 1,
                        }
                        self.update_config["gaps_text_checked"] = checks
                    continue
            if _is_attachment_unit(item) and not (text or "").strip():
                # Nothing to write from: an attachment unit is its text.
                continue
            if key in waiting and (text or "").strip():
                waiting.discard(key)
                checks = self._text_checks()
                checks.pop(key, None)
                self.update_config["gaps_text_checked"] = checks
            kept.append(item)
        self.update_config["gaps_waiting_for_text"] = sorted(waiting)
        if not self._text_checks():
            self.update_config.pop("gaps_text_checked", None)
        return kept

    def _narrow_full_scan_to_changes(
        self, items: list[dict[str, Any]], gaps: dict[str, set[str]]
    ) -> list[dict[str, Any]]:
        """Keep only what a full scan of a split index actually has to embed.

        With no sync watermark, the API scan fetches everything and would embed
        every level of every item, though most are indexed and unchanged. An
        unchanged item keeps just its gap, if it has one.
        """
        kept: list[dict[str, Any]] = []
        for item in items:
            data = item.get("data", {})
            existing = self.chroma_client.get_document_metadata(item.get("key", ""))
            if not existing or not _same_moment(existing.get("date_modified"), data.get("dateModified")):
                kept.append(item)
                continue
            # New text for an item indexed without it: never indexed with
            # text, or "failed" (asked, none then). Attaching a PDF doesn't
            # change the item, so this is where its text comes in.
            if (data.get("fulltext") or "").strip() and existing.get("has_fulltext") is not True:
                kept.append(item)
                continue
            if item.get("key") in gaps:
                item["_index_levels"] = gaps[item["key"]]
                kept.append(item)
        return self._without_fills_lacking_text(kept)

    @staticmethod
    def _fetched_keys(items: list[dict[str, Any]]) -> set[str]:
        """Keys ``items`` already cover: each item, and the attachment its text came from."""
        keys = {item.get("key") for item in items}
        for item in items:
            source = (item.get("data") or {}).get("fulltextSource") or ""
            if source.startswith(_WEB_ATTACHMENT_SOURCE):
                keys.add(source[len(_WEB_ATTACHMENT_SOURCE):])
        return keys

    def _keep_current_articles(
        self, items: list[dict[str, Any]], gaps: dict[str, set[str]]
    ) -> None:
        """Changed items whose article record is already current get only their passages.

        A run stopped between the article and passage passes leaves the sync
        version where it was, so the next run fetches the same changes again.
        Its article records were written; an item that has one for its
        current version, written from the same text, and no passages needs
        only those. Anything else is rewritten in full, as before.
        """
        for item in items:
            key = item.get("key", "")
            if gaps.get(key) != {SEARCH_PASSAGES} or item.get("_index_levels"):
                continue
            data = item.get("data", {})
            existing = self.chroma_client.get_document_metadata(key)
            if not existing or "chunk_index" in existing:
                continue
            if not _same_moment(existing.get("date_modified"), data.get("dateModified")):
                continue
            # Written from the same text: with full text both times, or without.
            if bool((data.get("fulltext") or "").strip()) != (existing.get("has_fulltext") is True):
                continue
            item["_index_levels"] = gaps[key]

    def _stored_metadata(self, key: str) -> dict[str, Any] | None:
        """``get_document_metadata``, or None from a client that can't look records up."""
        lookup = getattr(self.chroma_client, "get_document_metadata", None)
        return lookup(key) if lookup else None

    def _legacy_metadata(
        self,
        key: str,
        existing: dict[str, Any] | None = None,
        *,
        passage: dict[str, Any] | None | Any = _UNSET,
    ) -> dict[str, Any] | None:
        """The metadata of a record of ``key`` written before #826, or None.

        A regular item is legacy while any of its records carries ``title``:
        its bare record, or, beside a new bibliographic record written by an
        article-level fill, its parent-keyed passages. ``existing`` is the
        item's ``get_document_metadata`` result when the caller has it, and
        ``passage`` the stored metadata of ``KEY#0`` (None when absent), so
        a caller that read ahead spares the extra lookup.
        """
        if existing is None:
            existing = self._stored_metadata(key)
        if _is_legacy_record(existing):
            return existing
        if existing and "chunk_index" not in existing:
            if passage is _UNSET:
                passage = self._stored_metadata(f"{key}#0")
            if _is_legacy_record(passage):
                return passage
        return None

    @staticmethod
    def _unit_is_current(unit: dict[str, Any], existing: dict[str, Any] | None) -> bool:
        """Whether ``existing``, a record's stored metadata, already describes ``unit``.

        A bibliographic record is current while its ``date_modified`` is the
        item's, an attachment record while its ``file_stamp`` is the file's.
        A record without library attribution, a legacy record, or only a
        passage of the key is never current.
        """
        if not existing or "group_id" not in existing or "chunk_index" in existing:
            return False
        if _is_legacy_record(existing):
            return False
        if _is_attachment_unit(unit):
            return existing.get("file_stamp") == unit.get("file_stamp")
        return _same_moment(existing.get("date_modified"), unit.get("data", {}).get("dateModified"))

    def _plan_units(self, items: list[dict[str, Any]], *, force_rebuild: bool) -> list[dict[str, Any]]:
        """The work units an API-fetched list of items needs written this run.

        Units the local scan made pass through: it already decided. For an
        API item (#826, transitional until the API paths select attachments
        themselves): a legacy item missing only its article record gets a new
        bibliographic record and keeps its passages; any other legacy item
        here has changed and is rewritten in full, its legacy passages going
        once its new record is in. Records already current are left alone.
        """
        units: list[dict[str, Any]] = []
        for item in items:
            if item.get("unit"):
                units.append(item)
                continue
            bib, *attachments = self._as_units([item])
            levels = bib.pop("_index_levels", None)
            if force_rebuild:
                units.append(bib)
                units.extend(attachments)
                continue
            existing = self._stored_metadata(bib["key"])
            if self._legacy_metadata(bib["key"], existing) is not None:
                if levels == {SEARCH_ARTICLES}:
                    bib["_index_levels"] = levels
                    units.append(bib)
                    continue
                bib["legacy_rewrite"] = True
                units.append(bib)
                units.extend(attachments)
                continue
            if not self._unit_is_current(bib, existing):
                units.append(bib)
            for attachment in attachments:
                stored = self._stored_metadata(attachment["key"])
                if not self._unit_is_current(attachment, stored):
                    units.append(attachment)
                elif gap := self._scan_gaps.get(attachment["key"]):
                    attachment["_index_levels"] = gap
                    units.append(attachment)
        return units

    def _unfinished_batch_run(self) -> str | None:
        """Run id of a batch run for this library that has not finished landing.

        Its results include records a gap check would see as missing, and
        paying for them again would also supersede the run. The same rule
        that holds back new submissions (:func:`batch_common.unfinished_run`).
        """
        group_id = self._pinned_group_id()
        for name in _BATCH_MODULES:
            run = batch_common.unfinished_run(
                batch_common.get_batch_root(name, self.config_path), group_id, _batch_adapter(name)
            )
            if run:
                return str(run.get("run_id"))
        return None

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
        for creator in creators_str.split(";"):
            creator = creator.strip()
            if not creator:
                continue

            if "," in creator:
                last, first = creator.split(",", 1)
                creators.append({"creatorType": "author", "firstName": first.strip(), "lastName": last.strip()})
            else:
                creators.append({"creatorType": "author", "name": creator})

        return creators

    def _fetch_fulltext_via_web_api(self, item_key: str) -> tuple[str, str]:
        """Fetch the fulltext Zotero already extracted for a top-level item.

        Zotero keeps extracted text per PDF attachment, never under the parent
        item's own key, so this walks the item's PDF attachment children and
        returns the first that has any. It works against the web API and the
        local API alike, without needing the PDF file.

        Returns:
            (text, source) where source is ``web-api:attachment:<key>``. Empty
            strings mean no fulltext is available for this item.
        """
        text, source, _failed = self._fetch_fulltext_checked(item_key)
        return text, source

    def _fetch_fulltext_checked(self, item_key: str) -> tuple[str, str, bool]:
        """``_fetch_fulltext_via_web_api``, plus whether a request failed on the way.

        No text after a failed request says nothing about the item: the text
        may well be there. Zotero's 404 for an attachment without text is an
        answer, not a failure.
        """
        try:
            children = _paginate(self.zotero_client.children, item_key) or []
        except ResourceNotFoundError:
            children = []
        except Exception as e:
            logger.debug(f"children({item_key}) failed: {e}")
            return "", "", True
        return self._fulltext_from_attachments_checked(
            [self._note_api_attachment(child) for child in children]
        )

    # What the API run has seen of each PDF attachment, by key: the record
    # state an attachment unit needs (see _api_file_stamp).
    _api_attachments: dict[str, dict[str, Any]] | None = None

    def _note_api_attachment(self, row: Any) -> str | None:
        """``_pdf_attachment_key(row)``, remembering the attachment's file state."""
        key = self._pdf_attachment_key(row)
        if key:
            if self._api_attachments is None:
                self._api_attachments = {}
            self._api_attachments[key] = {
                "file_stamp": _api_file_stamp(row),
                "date_modified": (row.get("data") or {}).get("dateModified", ""),
            }
        return key

    @staticmethod
    def _pdf_attachment_key(child: Any) -> str | None:
        """The key of ``child`` when it is a PDF attachment, else None."""
        data = child.get("data", {}) if isinstance(child, dict) else {}
        if data.get("itemType") != "attachment":
            return None
        if data.get("contentType") != "application/pdf":
            return None
        return child.get("key") or data.get("key") or None

    def _fulltext_from_attachments(self, attachment_keys: list[str | None]) -> tuple[str, str]:
        """Zotero's extracted text for the first of ``attachment_keys`` that has any."""
        text, source, _failed = self._fulltext_from_attachments_checked(attachment_keys)
        return text, source

    def _fulltext_from_attachments_checked(
        self, attachment_keys: list[str | None]
    ) -> tuple[str, str, bool]:
        """``_fulltext_from_attachments``, plus whether a request failed on the way."""
        failed = False
        for child_key in attachment_keys:
            if not child_key:
                continue
            try:
                resp = self.zotero_client.fulltext_item(child_key)
            except ResourceNotFoundError:
                continue
            except Exception as e:
                logger.debug(f"fulltext_item({child_key}) failed: {e}")
                failed = True
                continue
            text = str(resp.get("content", "") or "") if isinstance(resp, dict) else (
                resp if isinstance(resp, str) else ""
            )
            if text.strip():
                return text, f"web-api:attachment:{child_key}", False
        return "", "", failed

    # For a whole-library pass over at least this many items, one paged listing
    # of the attachments is cheaper than asking for each item's children.
    _BULK_ATTACHMENT_WALK_MIN_ITEMS = 25

    def _pdf_attachment_keys_by_parent(self) -> dict[str, list[str]] | None:
        """PDF attachment keys grouped by parent key, from one paged listing.

        None when the listing fails, so the caller can fall back to asking
        per item.
        """
        try:
            rows = _paginate(self.zotero_client.items, itemType="attachment") or []
        except Exception as e:
            logger.debug(f"attachment listing failed, asking per item: {e}")
            return None
        by_parent: dict[str, list[str]] = {}
        for row in rows:
            key = self._note_api_attachment(row)
            parent = (row.get("data") or {}).get("parentItem") if isinstance(row, dict) else None
            if key and parent:
                by_parent.setdefault(parent, []).append(key)
        return by_parent

    def _attach_web_fulltext(
        self, items: list[dict[str, Any]], *, whole_library: bool = False
    ) -> None:
        """Populate `data.fulltext` on each item in place using Zotero's API.

        Works against the web API and the local API alike: both serve the text
        Zotero has already extracted for each PDF attachment.

        ``whole_library`` says ``items`` is the entire library, which is when
        listing every attachment once beats asking per item. For a handful of
        changed items in a large library it would page through attachments
        nobody needs, so the incremental path leaves it off.
        """
        total = len(items)
        if not total:
            return
        try:
            sys.stderr.write(f"\nFetching fulltext for {total} items via web API...\n")
            sys.stderr.flush()
        except Exception:
            pass
        fetched = 0
        pdf_keys = (
            self._pdf_attachment_keys_by_parent()
            if whole_library and total >= self._BULK_ATTACHMENT_WALK_MIN_ITEMS
            else None
        )
        for idx, item in enumerate(items, 1):
            key = item.get("key", "")
            data = item.setdefault("data", {})
            # Skip items that obviously can't have fulltext
            if data.get("itemType") in {"note", "annotation"}:
                data["fulltext_attempted"] = True
                continue
            if not key:
                continue
            # These are top-level items, never attachments themselves, so
            # Zotero holds no text under their own key: ask for their PDF
            # attachments only (the parent probe was a request per item that
            # always came back empty).
            if pdf_keys is not None:
                text, source, failed = self._fulltext_from_attachments_checked(pdf_keys.get(key, []))
            else:
                text, source, failed = self._fetch_fulltext_checked(key)
            if text:
                data["fulltext"] = text
                data["fulltextSource"] = source
                # The attachment the text came from, for its own record (#826).
                attachment_key = source[len(_WEB_ATTACHMENT_SOURCE):]
                data["fulltextAttachment"] = dict((self._api_attachments or {}).get(attachment_key) or {})
                fetched += 1
            else:
                data["fulltext_attempted"] = True
                if failed:
                    # A gap fill must not take this as "Zotero has no text".
                    data["fulltext_fetch_failed"] = True
            if idx % 25 == 0 or idx == total:
                try:
                    sys.stderr.write(f"\r  Fulltext: {idx}/{total} items checked, {fetched} with text")
                    sys.stderr.flush()
                except Exception:
                    pass
        try:
            sys.stderr.write("\n")
        except Exception:
            pass

    def _tag_group_id(self, items: list[dict[str, Any]]) -> None:
        """Stamp every item's ``data.group_id`` with the active library, in place.

        Web-API item/version fetches always cover exactly one library — the
        one ``self.zotero_client`` is scoped to — so every item an API-mode
        scan or incremental fetch returns can be tagged with that library.
        Inside an update run the identity is pinned once per run; outside
        one it falls back to the active-library lookup.
        """
        group_id = self._pinned_group_id()
        for item in items:
            item.setdefault("data", {})["group_id"] = group_id

    def _get_items_from_api(self, limit: int | None = None, include_fulltext: bool = False) -> list[dict[str, Any]]:
        """
        Get items from Zotero API (original implementation).

        Args:
            limit: Optional limit on number of items
            include_fulltext: If True, fetch server-side extracted PDF text
                via pyzotero's fulltext_item endpoint for each returned
                top-level item. Enables full-text semantic indexing without
                requiring local Zotero mode.

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

            # Filter out attachments, notes and annotations
            filtered_items = [
                item for item in items if item.get("data", {}).get("itemType") not in _NON_INDEXED_ITEM_TYPES
            ]

            all_items.extend(filtered_items)
            start += batch_size

            if len(items) < batch_size:
                break

        if limit:
            all_items = all_items[:limit]

        if include_fulltext:
            self._attach_web_fulltext(all_items, whole_library=not limit)

        self._tag_group_id(all_items)

        logger.info(f"Retrieved {len(all_items)} items from API")
        return all_items

    def _get_changed_items_from_api(
        self, since_version: int, include_fulltext: bool = False
    ) -> tuple[list[dict[str, Any]], set[str] | None]:
        """Fetch only items changed in the Zotero library since a given version.

        Uses pyzotero's `item_versions(since=V)` to discover changed top-level
        item keys, then fetches their full payloads one at a time. When
        `include_fulltext` is True, also fetches server-side extracted text
        for each changed item.

        Returns:
            (changed_items, current_library_keys). The second element powers
            deletion detection: any doc attributed to this library but absent
            from it has been removed from the library. It is ``None`` — NOT
            an empty set — when the fetch fails: "unknown" must make the
            deletion pass skip, never look like "the library is empty" (which
            once turned a transient API failure into a full index wipe).
        """
        logger.info(f"Fetching changed items since library version {since_version}...")
        try:
            changed_versions = self.zotero_client.item_versions(since=since_version) or {}
        except Exception as e:
            raise Exception(f"Failed to fetch item_versions(since={since_version}): {e}") from e

        current_keys: set[str] | None
        try:
            current_keys = set((self.zotero_client.item_versions() or {}).keys())
        except Exception as e:
            logger.warning(
                f"Failed to fetch current item_versions for deletion check: {e}; "
                "this run will skip deletion detection."
            )
            current_keys = None

        if not changed_versions:
            return [], current_keys

        changed_items: list[dict[str, Any]] = []
        for key in changed_versions.keys():
            try:
                item = self.zotero_client.item(key)
            except Exception as e:
                logger.debug(f"item({key}) failed during incremental fetch: {e}")
                continue
            if not item:
                continue
            item_type = item.get("data", {}).get("itemType")
            # Don't index attachments/notes as standalone entries; only
            # top-level research items participate in semantic search.
            if item_type in _NON_INDEXED_ITEM_TYPES:
                continue
            changed_items.append(item)

        if include_fulltext and changed_items:
            self._attach_web_fulltext(changed_items)

        self._tag_group_id(changed_items)

        return changed_items, current_keys

    def _verify_local_snapshot_version(self, target_sync_version: int) -> int | None:
        """Decide whether the local sqlite snapshot supports promoting the
        API-derived sync watermark.

        The local-extraction scan reads zotero.sqlite with `immutable=1`,
        which cannot see rows still sitting in an un-checkpointed WAL file.
        The API (served by the running Zotero) *does* see them, so its
        library version may cover items the scan never returned. Promoting
        `last_sync_version` in that state makes every later incremental
        update skip those items forever (issue #292).

        Returns:
            `target_sync_version` if every item key known to the API is
            present in the sqlite snapshot, otherwise None (keep the
            previous watermark so the next update re-covers the gap).
        """
        try:
            api_keys = set((self.zotero_client.item_versions() or {}).keys())

            # Prefer the key set captured by the scan's own connection: a
            # fresh read here could already see rows from a WAL checkpoint
            # that landed mid-scan, masking the very staleness we check for.
            snapshot_keys = getattr(self, "_last_scan_snapshot_keys", None)
            if snapshot_keys is None:
                zotero_db_path = self.db_path  # CLI override takes precedence
                if not zotero_db_path and self.config_path and os.path.exists(self.config_path):
                    try:
                        with open(self.config_path) as f:
                            zotero_db_path = (
                                json.load(f).get("semantic_search", {}).get("zotero_db_path")
                            )
                    except Exception:
                        pass
                with LocalZoteroReader(db_path=zotero_db_path) as reader:
                    snapshot_keys = reader.get_all_item_keys()
        except Exception as e:
            logger.warning(
                f"Could not verify local snapshot completeness ({e}); "
                "keeping previous sync watermark."
            )
            return None

        missing = api_keys - snapshot_keys
        if missing:
            logger.warning(
                f"{len(missing)} item(s) are visible via the Zotero API but "
                "missing from the local sqlite snapshot (immutable reads "
                "cannot see un-checkpointed WAL data); keeping previous sync "
                "watermark so the next update can pick them up."
            )
            return None
        return target_sync_version

    def _separate_passages(self) -> bool:
        """Whether this index keeps passages in their own collection."""
        check = getattr(self.chroma_client, "separate_passages", None)
        return bool(check) and check() is True

    def _levels_to_write(self, item: dict[str, Any]) -> tuple[bool, bool]:
        """``(article record, passage records)`` a work unit should get in this run.

        A regular item's bibliographic record is article-level only and never
        has passages. An attachment always gets its article record, plus
        passages with chunking on, wherever its passages live. A unit in the
        index's gap list gets only what it lacks.
        """
        levels = item.get("_index_levels")
        if not _is_attachment_unit(item):
            return (not levels or SEARCH_ARTICLES in levels), False
        if levels and self._separate_passages():
            return SEARCH_ARTICLES in levels, SEARCH_PASSAGES in levels
        return True, self._chunking_enabled

    def _build_item_records(
        self, item: dict[str, Any], *, chunking: bool, article: bool = True
    ) -> list[dict[str, Any]]:
        """Build the index records of one work unit, without embedding or writing them.

        The one place a unit becomes ``{"id", "document", "metadata"}`` records,
        for the realtime and the Batch API paths alike (#826):

        * a regular item: one bibliographic record ``KEY`` from its metadata
          text alone (``chunking`` does not apply; it has no passages);
        * an attachment: one record ``ATT`` from the file's text alone and,
          with ``chunking``, one record per overlapping passage ``ATT#n``, with
          offsets, page and section computed on that text. ``max_chunks_per_item``
          caps the passages of each attachment.

        Every record stores its own state only. Returns ``[]`` for a unit with
        no key or no text. A failure raises with nothing returned, so a unit
        contributes all of its records or none.
        """
        key = item.get("key", "")
        if not key:
            return []
        if not _is_attachment_unit(item):
            if not article:
                return []
            text = self._create_document_text(item)
            if not text.strip():
                return []
            metadata = self._create_metadata(item)
            if item.get("legacy_rewrite"):
                # ChromaDB's upsert merges metadata into the stored record's:
                # a None removes a key, or the legacy fields would stay.
                metadata.update(dict.fromkeys(_LEGACY_METADATA_KEYS))
            return [{
                "id": key,
                "document": self.chroma_client.truncate_text(text),
                "metadata": metadata,
            }]

        text = item.get("text") or ""
        if not text.strip():
            return []
        group_id = item.get("group_id")
        records: list[dict[str, Any]] = []
        if article:
            metadata = _own_metadata(key, group_id, item.get("date_modified"))
            metadata["file_stamp"] = item["file_stamp"]
            # Always present, None when unknown: ChromaDB's upsert merges
            # metadata, so an omitted key would keep the previous file's hash.
            metadata["file_hash"] = item.get("file_hash") or None
            records.append({
                "id": key,
                "document": self.chroma_client.truncate_text(text),
                "metadata": metadata,
            })
        if not chunking:
            return records

        passages = split_into_passages(
            text,
            self._effective_chunk_size(),
            int(self._chunking_config.get("overlap", 200)),
            int(self._chunking_config.get("max_chunks_per_item", 20)),
        )
        section_marks = _section_offsets(text) if passages else []
        for ci, (chunk_text, c0, c1) in enumerate(passages):
            cmeta: dict[str, Any] = {"item_key": key}
            if group_id is not None:
                cmeta["group_id"] = int(group_id)
            cmeta["chunk_index"] = ci
            cmeta["n_chunks"] = len(passages)
            cmeta["char_start"] = c0
            cmeta["char_end"] = c1
            # None removes a stale value left by an earlier version (upsert merges).
            cmeta["page"] = _page_for_offset(text, c0)
            cmeta["section"] = _section_at(section_marks, c0)
            records.append({
                "id": f"{key}#{ci}",
                "document": self.chroma_client.truncate_text(chunk_text),
                "metadata": cmeta,
            })
        return records

    def _as_units(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Work units for ``items``; units pass through unchanged.

        An API-shaped item becomes a bibliographic unit, plus an attachment
        unit for the text ``_attach_web_fulltext`` found, keyed by the
        attachment it came from. Text from no known attachment can't be
        keyed, so it is not indexed.
        """
        units: list[dict[str, Any]] = []
        for item in items:
            if item.get("unit"):
                units.append(item)
                continue
            raw = item.get("data") or {}
            text = raw.get("fulltext") or ""
            source = raw.get("fulltextSource") or ""
            attachment = raw.get("fulltextAttachment") or {}
            # A copy without the text: the attachment unit carries it.
            data = {k: v for k, v in raw.items() if k not in ("fulltext", "fulltextAttachment")}
            units.append(_item_unit({**item, "data": data}))
            if text.strip() and source.startswith(_WEB_ATTACHMENT_SOURCE):
                units.append(_attachment_unit(
                    source[len(_WEB_ATTACHMENT_SOURCE):],
                    text,
                    file_stamp=attachment.get("file_stamp") or _API_UNKNOWN_STAMP,
                    date_modified=attachment.get("date_modified"),
                    group_id=data.get("group_id"),
                    owner_key=item.get("key"),
                    label=data.get("title", ""),
                ))
        return units

    def _prepare_index_records(self, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Prepare ChromaDB records without embedding or writing them."""
        stats = {"processed": 0, "skipped": 0, "errors": 0}
        records: list[dict[str, Any]] = []

        for item in self._as_units(items):
            try:
                article, passages = self._levels_to_write(item)
                item_records = self._build_item_records(item, chunking=passages, article=article)
                if not item_records:
                    stats["skipped"] += 1
                    continue

                if item.get("_index_levels"):
                    # The import must not treat this as a rewrite of the item.
                    for record in item_records:
                        record["level_fill"] = True
                if item.get("legacy_rewrite"):
                    # The import drops the item's legacy passages once this lands.
                    item_records[0]["legacy_rewrite"] = True
                records.extend(item_records)
                stats["processed"] += 1

            except Exception as e:
                logger.error(f"Error processing item {item.get('key', 'unknown')}: {e}")
                stats["errors"] += 1

        return records, stats

    def _existing_item_keys(self, item_keys: list[str]) -> set[str]:
        """Which of these items have a record at either level."""
        found = self.chroma_client.get_existing_ids(
            [*item_keys, *(f"{key}#0" for key in item_keys)]
        )
        return {record_key(doc_id) for doc_id in found}

    def _submit_batch_index(
        self,
        provider: str,
        items: list[dict[str, Any]],
        force_full_rebuild: bool,
        target_sync_version: int | None,
        stats: dict[str, Any],
        max_enqueued_tokens: int | None = None,
        max_requests: int | None = None,
    ) -> dict[str, Any]:
        """Prepare records and submit asynchronous embedding batches."""
        module = _batch_module(provider)
        adapter = _batch_adapter(provider)
        label = adapter.label
        records, prepare_stats = self._prepare_index_records(items)
        stats["processed_items"] += prepare_stats["processed"]
        stats["skipped_items"] += prepare_stats["skipped"]
        stats["errors"] += prepare_stats["errors"]

        if not records:
            stats["batch_submitted"] = False
            if prepare_stats["errors"]:
                stats["batch_error"] = f"No documents were prepared for {label} Batch API submission"
            else:
                # Nothing changed that needs a record: no batch to wait for.
                stats["batch_nothing_to_embed"] = True
            return stats

        # Every article record goes in the files before any passage, so
        # article-level search is complete once the first files land. The sort
        # is stable: an item's passages stay in order.
        records.sort(key=lambda record: "#" in record["id"])
        # Its article and passages now land in different files: the import
        # counts the item where its article lands, not again at its first passage.
        # An article record also carries a digest of each of its item's new
        # passages, which follow in later files (see _drop_unlike_passages).
        with_article = {record["id"] for record in records if "#" not in record["id"]}
        follow: dict[str, dict[str, str]] = {}
        for record in records:
            key = record_key(record["id"])
            if "#" in record["id"] and key in with_article:
                follow.setdefault(key, {})[record["id"]] = _passage_digest(record["document"])
                if record["id"].endswith("#0"):
                    record["counted_with_article"] = True
        for record in records:
            if record["id"] in follow:
                record["passages_follow"] = follow[record["id"]]
        item_keys = sorted({record_key(record["id"]) for record in records})
        existing_items = (
            self._existing_item_keys(item_keys) if not force_full_rebuild else set()
        )
        model_name = self.chroma_client.embedding_config.get("model_name", adapter.default_model)
        submit_kwargs: dict[str, Any] = {}
        if max_enqueued_tokens is not None:
            submit_kwargs["max_enqueued_tokens"] = max_enqueued_tokens
        if max_requests is not None:
            submit_kwargs["max_requests"] = max_requests
        manifest = module.submit_embedding_batches(
            records=records,
            model_name=model_name,
            embedding_config=self.chroma_client.embedding_config,
            config_path=self.config_path,
            force_full_rebuild=force_full_rebuild,
            target_sync_version=target_sync_version,
            # The manifest's group_id keys the watermark save at import time;
            # it must carry the run's pinned identity, not the live override.
            group_id=self._pinned_group_id(),
            **submit_kwargs,
        )
        stats["batch_provider"] = provider
        stats["batch_submitted"] = True
        stats["batch_run_id"] = manifest["run_id"]
        stats["batch_manifest"] = manifest["manifest_path"]
        # Pending (throttled, not yet submitted) chunks have no batch_id yet.
        stats["batch_ids"] = [b["batch_id"] for b in manifest.get("batches", []) if b.get("batch_id")]
        stats["batch_pending"] = sum(
            1 for b in manifest.get("batches", []) if b.get("status") == batch_common.STATE_PENDING
        )
        stats["submitted_items"] = len(item_keys)
        stats["submitted_records"] = len(records)
        stats["estimated_updated_items"] = len(existing_items)
        stats["estimated_added_items"] = len(item_keys) - len(existing_items)
        return stats

    def _submit_openai_batch_index(
        self,
        items: list[dict[str, Any]],
        force_full_rebuild: bool,
        target_sync_version: int | None,
        stats: dict[str, Any],
        max_enqueued_tokens: int | None = None,
        max_requests: int | None = None,
    ) -> dict[str, Any]:
        """Prepare records and submit asynchronous OpenAI embedding batches."""
        return self._submit_batch_index(
            "openai", items, force_full_rebuild, target_sync_version, stats,
            max_enqueued_tokens=max_enqueued_tokens, max_requests=max_requests,
        )

    def _submit_gemini_batch_index(
        self,
        items: list[dict[str, Any]],
        force_full_rebuild: bool,
        target_sync_version: int | None,
        stats: dict[str, Any],
        max_enqueued_tokens: int | None = None,
        max_requests: int | None = None,
    ) -> dict[str, Any]:
        """Prepare records and submit asynchronous Gemini embedding batches."""
        return self._submit_batch_index(
            "gemini", items, force_full_rebuild, target_sync_version, stats,
            max_enqueued_tokens=max_enqueued_tokens, max_requests=max_requests,
        )

    def _run_deletion_pass(self, stats, current_library_keys, allow_mass_deletion):
        """Delete indexed docs of the syncing library that Zotero no longer has.

        Shared by the incremental and full-scan paths. It used to run only on
        the incremental one, so a full scan (every local-mode update from the
        MCP tool, which extracts fulltext) never removed deleted items, and
        still promoted the watermark, which then gave the incremental path
        nothing left to reconcile (#457). ``current_library_keys`` is the
        record keys that may stay: a full local scan's desired set (regular
        items and the files selected for indexing, #826), else every key
        ``item_versions()`` lists. Stored records are compared by their own
        record key. None means the key set could not be fetched; that records
        a skip reason, and a skipped pass keeps the watermark where it was.
        """
        # Delete docs of THIS library that are no longer present in
        # it. Scope is the run's group_id, applied DB-side: only docs
        # positively attributed to the syncing library are deletion
        # candidates, so another library's docs — and docs with no
        # attribution at all — can never be deleted by this pass
        # (#404 wiped every other library from the index). Chunk ids
        # (``<key>#<n>``) map back to item keys so deletion works
        # identically whether or not chunking is on.
        try:
            stored_ids = self.chroma_client.get_all_ids(
                where={"group_id": int(self._run_group_id)}
            )
            stored_item_keys = {record_key(i) for i in stored_ids}
            # Which kind of delete a key needs comes from what is stored, not
            # from the chunking setting, which may have changed since.
            keys_with_passages = {record_key(i) for i in stored_ids if "#" in i}
            if current_library_keys is None:
                # item_versions() failed: unknown is not "empty".
                stats["deletion_skipped_reason"] = "item_versions_unavailable"
            elif (
                not current_library_keys
                and stored_item_keys
                and not allow_mass_deletion
            ):
                # HTTP-200-but-empty against a non-empty store is
                # indistinguishable from an API fault, and wiping a
                # small library this way would slip under any
                # count-based guard. A user who really emptied the
                # library opts in with --allow-mass-deletion.
                logger.warning(
                    f"item_versions() returned no items while {len(stored_item_keys)} "
                    f"document(s) are indexed for library {self._active_library_key()}; "
                    "treating this as an API fault and skipping deletion "
                    "detection. If the library really is empty, rerun with "
                    "--allow-mass-deletion."
                )
                stats["deletion_skipped_reason"] = "empty_item_versions"
            else:
                to_delete_keys = sorted(
                    k for k in (stored_item_keys - current_library_keys) if k
                )
                if (
                    to_delete_keys
                    and not allow_mass_deletion
                    and len(to_delete_keys) >= _MASS_DELETION_MIN_DOCS
                    and len(to_delete_keys)
                    >= _MASS_DELETION_MIN_FRACTION * len(stored_item_keys)
                ):
                    sample = ", ".join(to_delete_keys[:5])
                    logger.warning(
                        f"Deletion pass wants to remove {len(to_delete_keys)} of "
                        f"{len(stored_item_keys)} indexed document(s) for library "
                        f"{self._active_library_key()} ({sample}, ...). That volume "
                        "usually means a truncated item_versions() response or a "
                        "sync-scoping bug, not a real purge — skipping. If the "
                        "deletions are intentional, rerun once with "
                        "--allow-mass-deletion."
                    )
                    stats["deletion_skipped_reason"] = "mass_deletion_guard"
                elif to_delete_keys:
                    for k in to_delete_keys:
                        if k in keys_with_passages:
                            # Scoped: a chunk set can carry mixed
                            # group_ids (partial rewrite, key
                            # collision); a bare parent-key delete
                            # would broaden this scoped candidate
                            # into an unscoped delete.
                            self.chroma_client.delete_item_chunks(
                                k, group_id=int(self._run_group_id)
                            )
                    bare_keys = [k for k in to_delete_keys if k in stored_ids]
                    if bare_keys:
                        self.chroma_client.delete_documents(bare_keys)
                    stats["deleted_items"] = len(to_delete_keys)
                    try:
                        sys.stderr.write(f"\nDeleted {len(to_delete_keys)} items no longer present in Zotero.\n")
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Deletion pass failed: {e}")

    def update_database(
        self,
        force_full_rebuild: bool = False,
        limit: int | None = None,
        extract_fulltext: bool = False,
        include_fulltext: bool | None = None,
        use_openai_batch: bool | None = None,
        use_gemini_batch: bool | None = None,
        use_batch: bool | None = None,
        batch_provider: str | None = None,
        batch_max_tokens: int | None = None,
        batch_max_requests: int | None = None,
        auto_loop: bool = False,
        batch_poll_interval: int = 60,
        allow_mass_deletion: bool = False,
        automatic: bool = False,
    ) -> dict[str, Any]:
        """
        Update the semantic search database with Zotero items.

        ``automatic`` marks a run nobody asked for (the startup update, the
        pre-search sync): it does not fill passages missing for items that
        have not changed, which after chunking is switched on is the whole
        library; ``update-db`` does.

        Args:
            force_full_rebuild: Whether to rebuild the entire database
            limit: Limit number of items to process (for testing)
            extract_fulltext: Whether to extract fulltext content from the
                local Zotero sqlite database (requires ZOTERO_LOCAL=true)
            include_fulltext: Whether to fetch the fulltext Zotero has
                already extracted for each PDF, through the API (web or
                local; no PDF files are read). Defaults to the
                `semantic_search.include_fulltext` config setting (True
                unless explicitly disabled). Ignored only when
                `extract_fulltext` is set, which extracts richer text from
                the local database instead.
            use_openai_batch: Deprecated in favour of `use_batch` /
                `batch_provider`. Override for OpenAI Batch API indexing.
                None uses `semantic_search.openai_batch.enabled`. Ignored
                whenever `use_batch` or `batch_provider` is given.
            use_gemini_batch: Deprecated in favour of `use_batch` /
                `batch_provider`. Override for Gemini Batch API indexing.
                None uses `semantic_search.gemini_batch.enabled`.
            use_batch: Provider-neutral on/off for Batch API indexing; the
                provider is inferred from the configured embedding model
                unless `batch_provider` names one.
            batch_provider: Which batch provider to use. Must match the
                configured embedding model.
            batch_max_tokens: Override for the enqueued-token throttle
                (`semantic_search.<provider>_batch.batch_max_enqueued_tokens`).
            batch_max_requests: Override for the per-file request cap
                (`semantic_search.<provider>_batch.batch_max_requests`).
            auto_loop: After submitting, keep polling, importing completed
                batches and submitting parked ones until the run finishes.
            batch_poll_interval: Seconds between auto-loop polls.
            allow_mass_deletion: One-run opt-in for a deletion pass that
                would remove a large share of the library's indexed docs
                (or all of them, when item_versions() reports the library
                empty). Deliberately a parameter, not an env var: nothing
                persistent can disable the guard, and the server's
                unattended background sync can never mass-delete.

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
            "recovered_items": 0,
            "skipped_items": 0,
            "deleted_items": 0,
            "errors": 0,
            "start_time": start_time.isoformat(),
            "duration": None,
        }

        # Guard against concurrent rebuilds: the MCP server auto-launches
        # update_database on startup while the user may also run
        # `zotero-mcp update-db` manually. A cross-process flock avoids
        # double work and potential ChromaDB corruption.
        lock_path = _update_lock_path()
        lock_cm = _acquire_update_lock(lock_path)
        acquired = lock_cm.__enter__()
        if not acquired:
            lock_cm.__exit__(None, None, None)
            holder_pid, holder_alive = read_lock_holder(lock_path)
            if holder_pid and not holder_alive:
                logger.warning(
                    "Update lock at %s is held by dead pid %s (stale). "
                    "flock should have released it; set ZOTERO_MCP_FORCE_UPDATE=1 "
                    "to bypass if this persists.",
                    lock_path,
                    holder_pid,
                )
            else:
                logger.warning(
                    "Another semantic-search update is already running "
                    "(lock held at %s by pid %s); skipping this invocation. "
                    "This is expected when the MCP server's background sync is "
                    "active. Set ZOTERO_MCP_FORCE_UPDATE=1 to override.",
                    lock_path,
                    holder_pid if holder_pid else "unknown",
                )
            stats["duration"] = "0:00:00"
            stats["skipped_reason"] = "another_update_in_progress"
            return stats

        try:
            # Pin this run's library identity once, from the client the run
            # will read items/versions from. Everything scope-sensitive in
            # the run (tagging, watermark key, backfill attribution, the
            # deletion pass) uses this snapshot, so a concurrent
            # zotero_switch_library cannot re-point half a run at another
            # library.
            self._run_group_id = self._client_group_id()
            # Set by this run's local scan, if it covers the whole library.
            self._scan_desired_keys = None

            # An index ChromaDB cannot read fails every later step with an
            # error that does not say why (#747). Stop with one that does; a
            # forced rebuild moves it aside instead of deleting it through
            # ChromaDB, which would hit the same error.
            if read_error := self.chroma_client.read_error():
                info = {
                    "error": read_error,
                    "persist_directory": self.chroma_client.persist_directory,
                    **unreadable_index_facts(self.chroma_client.persist_directory),
                }
                if not force_full_rebuild:
                    stats["error"] = (
                        unreadable_index_message(info, self.update_config.get("last_update"))
                        if info.get("unreadable") else f"Cannot read the semantic index: {read_error}"
                    )
                    return stats
                moved = self.chroma_client.move_aside()
                logger.warning("Moved the unreadable index to %s for --force-rebuild", moved)
                _report(f"The index could not be read; moved it to {moved} and building a new one.\n")

            # --force-rebuild resets the ENTIRE collection but repopulates
            # only the active library. Combinations that would silently drop
            # indexed data need the same explicit opt-in as any other mass
            # deletion.
            if force_full_rebuild and not allow_mass_deletion:
                error = None
                if limit is not None:
                    error = (
                        f"--force-rebuild with --limit would reset the whole "
                        f"collection and repopulate only {limit} item(s); rerun "
                        "with --allow-mass-deletion to confirm."
                    )
                else:
                    foreign = self.chroma_client.get_all_ids(
                        where={"group_id": {"$ne": int(self._run_group_id)}}
                    )
                    if foreign:
                        error = (
                            f"--force-rebuild would reset the whole collection, but "
                            f"{len(foreign)} document(s) are not attributed to the "
                            f"active library (library {self._active_library_key()}) — "
                            "other libraries' documents and unattributed documents "
                            "would be dropped permanently, since a rebuild "
                            "repopulates only the active library. Rerun with "
                            "--allow-mass-deletion to confirm."
                        )
                if error:
                    logger.error(error)
                    stats["error"] = error
                    end_time = datetime.now()
                    stats["duration"] = str(end_time - start_time)
                    stats["end_time"] = end_time.isoformat()
                    return stats

            # One-time metadata migration (#163): tag any pre-existing docs
            # that lack group_id. Skipped on a force rebuild — the reset
            # below wipes the collection anyway, so every doc gets tagged
            # fresh via the normal indexing path.
            if not force_full_rebuild and self._load_index_schema_version() < _INDEX_SCHEMA_VERSION:
                try:
                    backfill_stats = self._backfill_group_ids()
                    if backfill_stats["migrated"]:
                        try:
                            sys.stderr.write(
                                f"Migrated {backfill_stats['migrated']} existing document(s) "
                                "to the multi-library index format.\n"
                            )
                        except Exception:
                            pass
                    self._save_backfill_unattributed(backfill_stats.get("unattributed", 0))
                    self._save_index_schema_version(_INDEX_SCHEMA_VERSION)
                except Exception as e:
                    logger.error(
                        f"group_id metadata backfill failed ({e}); existing documents "
                        "may be missing library attribution, so library-filtered "
                        "search will not cover them and deletion cleanup will skip "
                        "them. The backfill retries on the next update."
                    )

            # Unattributed docs are excluded from library-filtered search and
            # from deletion cleanup; keep that visible on every update, not
            # just the one that discovered it.
            unattributed = self._load_backfill_unattributed()
            if unattributed:
                logger.warning(
                    f"Up to {unattributed} indexed document(s) have no library "
                    "attribution (count from the last group_id backfill; documents "
                    "re-indexed since then may have gained attribution). Unattributed "
                    "documents are excluded from library-filtered search and from "
                    "deletion cleanup."
                )

            # Resolve include_fulltext default from config if not specified
            if include_fulltext is None:
                include_fulltext = self._load_include_fulltext_setting()

            # Web-API fulltext only applies when not using the local sqlite
            # extractor (extract_fulltext=True takes precedence in local mode)
            include_fulltext_via_api = include_fulltext and not extract_fulltext
            batch_enabled, active_batch_provider = self._resolve_batch_mode(
                use_batch=use_batch,
                batch_provider=batch_provider,
                use_openai_batch=use_openai_batch,
                use_gemini_batch=use_gemini_batch,
            )
            throttle = self._load_batch_throttle_config(active_batch_provider)
            if batch_max_tokens is not None:
                throttle["batch_max_enqueued_tokens"] = batch_max_tokens
            if batch_max_requests is not None:
                throttle["batch_max_requests"] = batch_max_requests

            # A new run would be cut from the same sync version as an
            # unfinished one, so it pays again for the same items, and the
            # older run's parked chunks are never sent. Background syncs hit
            # this every startup or once a day; wait for the import instead.
            # A force rebuild, asked for by name, still goes ahead.
            if batch_enabled and not force_full_rebuild:
                waiting = batch_common.unfinished_run(
                    batch_common.get_batch_root(active_batch_provider, self.config_path),
                    self._pinned_group_id(),
                    _batch_adapter(active_batch_provider),
                )
                if waiting:
                    stats["skipped_reason"] = "batch_run_unfinished"
                    stats["skipped_message"] = (
                        f"Batch run {waiting.get('run_id')} for this library is unfinished, "
                        "so no new batch was submitted. Run 'zotero-mcp batch-import' to "
                        "collect it; the changes since then go out on the next update."
                    )
                    _report(f"\n{stats['skipped_message']}\n")
                    stats["duration"] = str(datetime.now() - start_time)
                    return stats

            # A silent loss without chunking: --fulltext extracts whole PDFs but
            # a small-context model embeds only the first few hundred tokens.
            fit_warning = self._fulltext_fit_warning() if extract_fulltext else None
            if fit_warning:
                stats["fulltext_truncated"] = True
                logger.warning(fit_warning)
                _report(f"\nWarning: {fit_warning}\n")

            # In batch mode, defer destructive rebuilds until import so the
            # existing search index remains usable while the batch runs.
            # A rebuild empties the index first, so the embedding API is
            # checked before that, not after.
            api_checked = False
            if force_full_rebuild and not batch_enabled:
                self._check_embedding_api()
                api_checked = True
                logger.info("Force rebuilding database...")
                self.chroma_client.reset_collection()

            # Decide whether to use since-based incremental ingest.
            # Incremental requires: not a forced rebuild, not a local-extraction
            # run (incremental path covers web-API metadata and optionally
            # fulltext only), not a test limit, and a known prior sync version.
            last_sync_version = self._load_last_sync_version() if not force_full_rebuild else 0
            use_incremental = (
                not force_full_rebuild and not extract_fulltext and limit is None and last_sync_version > 0
            )

            # When a collection filter is configured, skip the API-based
            # incremental path: it fetches changed items from the WHOLE
            # library and its deletion pass compares against all library
            # keys, both of which would bypass the filter. The local
            # full-scan path applies collection_keys and skips
            # already-indexed items, so filtered updates stay cheap.
            configured_collection_keys = None
            try:
                if self.config_path and os.path.exists(self.config_path):
                    with open(self.config_path) as _f:
                        configured_collection_keys = (
                            json.load(_f).get("semantic_search", {}).get("collection_keys")
                        )
            except Exception:
                pass
            if configured_collection_keys and use_incremental:
                use_incremental = False
                try:
                    sys.stderr.write(
                        f"Collection filter active ({configured_collection_keys}); "
                        "using local full scan instead of API incremental update.\n"
                    )
                except Exception:
                    pass

            # Pick up a layout changed by another process (a migration) before
            # deciding where this run writes.
            layout_of = getattr(self.chroma_client, "index_layout", None)
            layout = layout_of() if layout_of else None
            if not isinstance(layout, dict):
                layout = {}
            self._drop_passages_on_write = (
                self._separate_passages()
                and not self._chunking_enabled
                and bool(layout.get("passage_records"))
            )
            index_gaps = self._find_index_gaps(
                limit=limit, force_rebuild=force_full_rebuild,
                fill_passages=not automatic,
            )
            if getattr(self, "_passage_gaps_held", 0):
                stats["passage_gaps_held"] = self._passage_gaps_held
                logger.info(
                    "%d item(s) have no passages yet; run `zotero-mcp update-db` to "
                    "embed them (automatic updates leave this to you)",
                    self._passage_gaps_held,
                )
            # Items that only a run with their full text can fill. What is no
            # longer a gap is forgotten, but only by a run that looked for
            # every gap: a limited run and a rebuild don't, and would forget
            # items still waiting.
            if limit is None and not force_full_rebuild and self._separate_passages():
                waiting_for_text = [
                    key for key in self.update_config.get("gaps_waiting_for_text") or [] if key in index_gaps
                ]
                if "gaps_waiting_for_text" in self.update_config:
                    self.update_config["gaps_waiting_for_text"] = waiting_for_text
                if "gaps_text_checked" in self.update_config:
                    self.update_config["gaps_text_checked"] = {
                        key: check for key, check in self._text_checks().items() if key in waiting_for_text
                    }
            if index_gaps and (unfinished_run := self._unfinished_batch_run()):
                _report(
                    f"Batch run {unfinished_run} has not finished, so missing records are left "
                    "for now. Run 'zotero-mcp batch-import' to carry it forward.\n"
                )
                index_gaps = {}
            self._scan_gaps = index_gaps
            # Waiting items count as work only when this run may ask for their text.
            left_alone = self._still_waiting_for_text(include_fulltext_via_api)
            has_fillable_gaps = len(index_gaps) > len(left_alone & set(index_gaps))

            target_sync_version: int | None = None
            all_items: list[dict[str, Any]] = []
            if use_incremental:
                try:
                    target_sync_version = self.zotero_client.last_modified_version()
                except Exception as e:
                    logger.warning(f"last_modified_version() failed, falling back to full scan: {e}")
                    use_incremental = False

            if use_incremental and last_sync_version > (target_sync_version or 0):
                # A library's version counter never decreases, so a watermark
                # ahead of it cannot have come from this library (e.g. a
                # legacy scalar migrated from a differently-scoped install).
                # Trusting it would make item_versions(since=...) return an
                # empty dict and silently skip the whole library (#393).
                logger.warning(
                    f"Stored sync watermark ({last_sync_version}) is ahead of "
                    f"library {self._active_library_key()}'s current version "
                    f"({target_sync_version}); falling back to a full scan."
                )
                use_incremental = False
                last_sync_version = 0

            if use_incremental and target_sync_version == last_sync_version and not has_fillable_gaps:
                # No changes since last sync; skip ingest but still touch last_update
                try:
                    sys.stderr.write(
                        f"\nLibrary unchanged since last sync (version {last_sync_version}); no items to reindex.\n"
                    )
                except Exception:
                    pass
                self.update_config["last_update"] = datetime.now().isoformat()
                self._save_update_config(
                    last_sync_version=target_sync_version,
                    library_key=str(self._run_group_id),
                )
                end_time = datetime.now()
                stats["duration"] = str(end_time - start_time)
                stats["end_time"] = end_time.isoformat()
                return stats

            if use_incremental:
                all_items, current_library_keys = self._get_changed_items_from_api(
                    since_version=last_sync_version,
                    include_fulltext=include_fulltext_via_api,
                )
                self._run_deletion_pass(stats, current_library_keys, allow_mass_deletion)
                self._keep_current_articles(all_items, index_gaps)
                gap_items = self._get_gap_items_from_api(
                    index_gaps,
                    already_fetched=self._fetched_keys(all_items),
                    library_keys=current_library_keys,
                    include_fulltext=include_fulltext_via_api,
                )
                if gap_items:
                    _report(f"Adding the records missing from {len(gap_items)} indexed items.\n")
                    all_items.extend(gap_items)
            else:
                # Full scan: bootstrap or forced rebuild.
                # Capture the library version BEFORE scanning so any changes
                # made during the scan will be picked up by the next
                # incremental run. Skipping this after a force_full_rebuild
                # would leave last_sync_version stale and the next
                # incremental run would miss items that haven't changed
                # since the old watermark (because they were just deleted
                # along with the collection).
                try:
                    target_sync_version = self.zotero_client.last_modified_version()
                except Exception as e:
                    logger.warning(f"last_modified_version() failed: {e}")
                    target_sync_version = None
                all_items = self._get_items_from_source(
                    limit=limit,
                    extract_fulltext=extract_fulltext,
                    chroma_client=self.chroma_client if not force_full_rebuild else None,
                    force_rebuild=force_full_rebuild,
                    include_fulltext_via_api=include_fulltext_via_api,
                )
                # A full scan of a split index that already holds records
                # embeds only changes and gaps. Without a sync version (a
                # stopped rebuild, a migrated index, a reset watermark) it
                # would otherwise pay for every record of the library again.
                if (
                    not extract_fulltext
                    and not force_full_rebuild
                    and limit is None
                    and self._separate_passages()
                    and (layout.get("article_records", 0) or layout.get("passage_records", 0))
                ):
                    all_items = self._narrow_full_scan_to_changes(all_items, index_gaps)
                # The local-extraction scan may lag behind the API version
                # captured above (immutable sqlite reads skip WAL contents);
                # only promote the watermark if the snapshot was complete.
                if extract_fulltext and target_sync_version is not None:
                    target_sync_version = self._verify_local_snapshot_version(
                        target_sync_version
                    )
                # A forced rebuild starts from an empty collection and a
                # limited run is a test run; everything else reconciles
                # deletions here too (#457).
                if not force_full_rebuild and limit is None:
                    if self._scan_desired_keys is not None:
                        # Every record key the library should have (#826):
                        # what the scan left out (trashed, deleted, outranked
                        # or unreadable files) goes. item_versions() would keep
                        # an outranked file, which Zotero still has.
                        current_library_keys = self._scan_desired_keys
                    else:
                        try:
                            current_library_keys = set(
                                (self.zotero_client.item_versions() or {}).keys()
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to fetch current item_versions for deletion check: {e}; "
                                "skipping deletion detection this run."
                            )
                            current_library_keys = None
                    self._run_deletion_pass(stats, current_library_keys, allow_mass_deletion)

            # From here on the run works on units: a bibliographic record per
            # regular item, an attachment record (and passages) per file (#826).
            all_items = self._plan_units(all_items, force_rebuild=force_full_rebuild)
            self._legacy_rewrites = {unit["key"] for unit in all_items if unit.get("legacy_rewrite")}

            if limit is not None:
                # A limited run may index only a subset; promoting would strand
                # the rest (cf. #292). After a forced rebuild the old watermark is
                # just as wrong, because the collection it described is gone:
                # zero sends the next run back through a full scan.
                target_sync_version = 0 if force_full_rebuild else None

            stats["total_items"] = len(all_items)
            logger.info(f"Found {stats['total_items']} items to process")
            if waiting := self.update_config.get("gaps_waiting_for_text"):
                _report(
                    f"{len(waiting)} indexed items need their full text to add the missing records; "
                    f"{_waiting_for_text_hint(include_fulltext)}\n"
                )
            self._level_fill_keys = {item["key"] for item in all_items if item.get("_index_levels")}

            if batch_enabled:
                stats["batch_mode"] = True
                stats["batch_provider"] = active_batch_provider
                batch_label = _batch_adapter(active_batch_provider).label
                if stats.get("deletion_skipped_reason"):
                    # The manifest's target_sync_version is promoted at import
                    # time; a skipped deletion pass must stay retryable there
                    # exactly as on the realtime path.
                    target_sync_version = None
                _report(f"\nSubmitting {len(all_items)} items to {batch_label} Batch API...\n")
                stats = self._submit_batch_index(
                    active_batch_provider,
                    all_items,
                    force_full_rebuild=force_full_rebuild,
                    target_sync_version=target_sync_version,
                    stats=stats,
                    max_enqueued_tokens=throttle["batch_max_enqueued_tokens"],
                    max_requests=throttle["batch_max_requests"],
                )
                # This run did its work, so an auto-update is not due again
                # until its next turn. Without this, should_update stayed true
                # until an import and every search could pay for the same
                # changes again. The sync version waits for the import, unless
                # there is nothing to import: then it moves now, or every later
                # run would fetch the same changes again.
                self.update_config["last_update"] = datetime.now().isoformat()
                if (
                    stats.get("batch_nothing_to_embed")
                    and target_sync_version is not None
                    and not force_full_rebuild
                ):
                    self._save_update_config(
                        last_sync_version=target_sync_version,
                        library_key=str(self._run_group_id),
                    )
                else:
                    self._save_update_config()
                if stats.get("batch_nothing_to_embed"):
                    _report("  Nothing needs embedding; no batch was submitted.\n")
                    end_time = datetime.now()
                    stats["duration"] = str(end_time - start_time)
                    stats["end_time"] = end_time.isoformat()
                    return stats
                batch_ids = ", ".join(stats.get("batch_ids", []))
                _report(
                    f"  Submitted {batch_label} embedding batch"
                    f"{'es' if len(stats.get('batch_ids', [])) != 1 else ''}: {batch_ids}\n"
                )
                if stats.get("batch_pending"):
                    _report(
                        f"  {stats['batch_pending']} chunk(s) held back by the enqueued-token "
                        "budget; the next 'zotero-mcp batch-import' (or --auto-loop) submits them "
                        "as running batches finish.\n"
                    )
                if auto_loop and stats.get("batch_submitted"):
                    self.auto_loop_batch_pipeline(
                        active_batch_provider,
                        poll_interval=batch_poll_interval,
                        max_enqueued_tokens=throttle["batch_max_enqueued_tokens"],
                        stats=stats,
                    )
                else:
                    _report("  Run 'zotero-mcp batch-status' to check progress.\n")
                    _report("  Run 'zotero-mcp batch-import' after the batch completes.\n")
                end_time = datetime.now()
                stats["duration"] = str(end_time - start_time)
                stats["end_time"] = end_time.isoformat()
                return stats

            # One short request before any real work: off the network an API
            # accepts, every batch of the run would fail, and the run would
            # still save the update time as if it had worked.
            if all_items and not api_checked:
                self._check_embedding_api()

            # User-friendly progress reporting
            total = stats["total_items"] = len(all_items)
            try:
                sys.stderr.write(f"\nIndexing {total} items...\n\n")
                sys.stderr.flush()
            except Exception:
                pass

            # Process items in batches. This counts ITEMS, not documents: with
            # chunking enabled one item yields up to max_chunks_per_item
            # documents, so 25 items can be thousands of embedding inputs — the
            # old "25 × 8000 tokens = 200k, within OpenAI's limit" arithmetic
            # here only held when chunking was off (#423).
            # Request size is therefore bounded one layer down, by each
            # embedding function's own request_batch_size, which is where the
            # provider's real per-request limit belongs.
            batch_size = 25
            seen_items = 0
            _failed_docs = []  # Collect failures for end-of-run retry

            def _report_item_progress(item: dict[str, Any]) -> None:
                """Advance the single-line progress display by one item.

                Called from exactly one thread on both paths — the loop below,
                or the streaming producer — so the `\\r` line never interleaves
                and items are still announced in input order.
                """
                nonlocal seen_items
                seen_items += 1
                title = item.get("label") or item.get("data", {}).get("title", "")
                if title and len(title) > 60:
                    title = title[:57] + "..."
                pct = int(seen_items / total * 100) if total else 0
                # -1 keeps the cursor from wrapping onto the next line. Clip
                # by display columns, not characters: a CJK title must not be
                # allowed to wrap, or every later \r redraw garbles the screen.
                max_cols = _term_width() - 1
                line = f"  [{pct:3d}%] {seen_items}/{total} — {title or 'processing...'}"
                line = _clip_to_width(line, max_cols)
                _report(f"\r{line}{' ' * max(0, max_cols - _display_cols(line))}")

            # Overlap preparation, embedding and commits when the embedding
            # function is configured for concurrent requests. Off unless
            # embedding_config.max_parallel_requests says otherwise, so the
            # default run is byte-for-byte the historical sequential path. The
            # upsert_embeddings check also keeps the minimal ChromaDB doubles
            # used in tests — which implement only upsert_documents — on it.
            embedding_function = getattr(self.chroma_client, "embedding_function", None)
            max_parallel = getattr(embedding_function, "max_parallel_requests", 1) or 1
            use_streaming = (
                embedding_function is not None
                and max_parallel > 1
                and hasattr(self.chroma_client, "upsert_embeddings")
            )

            # On a split index every article record is embedded before any
            # passage, so article-level search is complete early in a long
            # run. A run stopped between the passes leaves only passages to
            # write; the sync version stays where it was, and the gap fill
            # finds items with an article record and no passages.
            passage_items = (
                [item for item in all_items if self._levels_to_write(item)[1]]
                if self._separate_passages()
                else []
            )
            if passage_items:
                passes = [
                    (SEARCH_ARTICLES, all_items, frozenset(item["key"] for item in passage_items)),
                    (SEARCH_PASSAGES, passage_items, frozenset()),
                ]
                self._held_text = _HeldText()
                self._undropped_passages = set()
            else:
                passes = [(None, all_items, frozenset())]
            accounted: set[str] = set()

            for level, pass_items, keep_cached in passes:
                if level == SEARCH_PASSAGES:
                    seen_items = 0
                    total = len(pass_items)
                    _report(
                        f"\r{' ' * _term_width(120)}\r"
                        f"  Article records written. Writing passages for {total} items...\n"
                    )
                if use_streaming:
                    logger.info(
                        f"Streaming index: {max_parallel} parallel embedding requests"
                    )
                    self._stream_index_items(
                        pass_items,
                        force_full_rebuild,
                        stats,
                        _failed_docs,
                        embedding_function,
                        max_parallel,
                        _report_item_progress,
                        level=level,
                        keep_cached=keep_cached,
                        accounted_keys=accounted,
                    )
                    continue
                for i in range(0, len(pass_items), batch_size):
                    batch = pass_items[i : i + batch_size]

                    # Show per-item progress within this batch
                    for item in batch:
                        _report_item_progress(item)

                    if level is None:
                        batch_stats = self._process_item_batch(batch, force_full_rebuild, _failed_docs)
                    else:
                        batch_stats = self._process_item_batch(
                            batch, force_full_rebuild, _failed_docs,
                            level=level, keep_cached=keep_cached, accounted=accounted,
                        )

                    stats["processed_items"] += batch_stats["processed"]
                    stats["added_items"] += batch_stats["added"]
                    stats["updated_items"] += batch_stats["updated"]
                    stats["skipped_items"] += batch_stats["skipped"]
                    stats["errors"] += batch_stats["errors"]

                    logger.info(
                        f"Processed {seen_items}/{total} items (added: {stats['added_items']}, skipped: {stats['skipped_items']})"
                    )

            # Retry any documents that failed during the main run
            retry_fail = 0
            if _failed_docs:
                try:
                    sys.stderr.write(f"\r{' ' * _term_width(120)}\r")
                    sys.stderr.write(f"\n  Retrying {len(_failed_docs)} failed items...\n")
                except Exception:
                    pass

                import time as _retry_time

                _retry_time.sleep(1)  # Brief pause before retry
                # Articles first here too.
                _failed_docs.sort(key=lambda failed: "chunk_index" in failed[1])

                retry_ok = 0
                # Items whose passages all got written, and items with any
                # passage still failing. Pruning waits until the item's
                # passages are done, then runs once per item, not per passage.
                retried_metas: dict[str, dict[str, Any]] = {}
                retried_articles: list[dict[str, Any]] = []
                still_failing: set[str] = set()
                for doc, meta, doc_id in _failed_docs:
                    # A passage's own item_key names the record it belongs to.
                    parent = meta.get("item_key") if "chunk_index" in meta else None
                    try:
                        self.chroma_client.upsert_documents([doc], [meta], [doc_id])
                        if parent is not None:
                            retried_metas[parent] = meta
                        else:
                            retried_articles.append(meta)
                        retry_ok += 1
                        stats["errors"] -= 1  # Remove from error count
                        # Don't classify as added vs updated — when the
                        # original batch failed, the add/update lookup never
                        # ran, so we don't know which category it belongs in.
                        # Track recovered items in their own bucket.
                        stats["recovered_items"] += 1
                    except Exception as e2:
                        retry_fail += 1
                        if parent is not None:
                            still_failing.add(parent)
                        logger.error(f"Retry failed for {doc_id}: {e2}")

                # No workers are running here, so _chroma_call_lock is not
                # needed for this prune.
                self._prune_stale_chunks(
                    retried_articles
                    + [m for k, m in retried_metas.items() if k not in still_failing]
                )

                try:
                    sys.stderr.write(f"  Retry: {retry_ok} recovered, {retry_fail} still failed\n")
                except Exception:
                    pass

            # Clear the progress line and show summary
            try:
                sys.stderr.write(f"\r{' ' * _term_width(120)}\r")  # Clear line
                summary = (
                    f"  Done: {stats['processed_items']} indexed, "
                    f"{stats['skipped_items']} skipped, "
                    f"{stats['errors']} errors"
                )
                if stats["recovered_items"]:
                    summary += f", {stats['recovered_items']} recovered"
                sys.stderr.write(summary + "\n")
            except Exception:
                pass

            # Update last update time, and promote last_sync_version on success.
            # A run whose deletion pass was SKIPPED must not promote: the next
            # run would take the unchanged-version early return and never
            # re-enter deletion detection, so the documented rerun with
            # --allow-mass-deletion would silently do nothing.
            # Likewise when items still failed after the retry (#610): in
            # incremental mode they would never appear in a later
            # item_versions(since=...) result, so they would stay unindexed.
            # Trade-off: an item that always fails keeps the watermark back
            # on every run. Persisting failed keys instead is a follow-up.
            self.update_config["last_update"] = datetime.now().isoformat()
            if stats.get("deletion_skipped_reason") or retry_fail:
                self._save_update_config()
            else:
                self._save_update_config(
                    last_sync_version=target_sync_version,
                    library_key=str(self._run_group_id),
                )

            end_time = datetime.now()
            stats["duration"] = str(end_time - start_time)
            stats["end_time"] = end_time.isoformat()

            logger.info(f"Database update completed in {stats['duration']}")
            return stats

        except EmbeddingAPIUnreachable as e:
            # Raised by the check before any document is embedded: nothing
            # changed, and last_update and the sync version stay where they were.
            logger.error(f"Stopping the update: {e}")
            stats["error"] = f"{e} Run the update again once the API can be reached."
            stats["duration"] = str(datetime.now() - start_time)
            return stats
        except Exception as e:
            logger.error(f"Error updating database: {e}")
            stats["error"] = str(e)
            end_time = datetime.now()
            stats["duration"] = str(end_time - start_time)
            return stats
        finally:
            self._run_group_id = None
            self._drop_passages_on_write = False
            self._scan_gaps = {}
            self._level_fill_keys = frozenset()
            self._legacy_rewrites = frozenset()
            self._scan_desired_keys = None
            # The server keeps one instance: the next run starts without these.
            self._undropped_passages = frozenset()
            if self._held_text is not None:
                self._held_text.close()
                self._held_text = None
            # Release the update flock on every exit path. Paired with the
            # __enter__ call above; the "not acquired" branch releases
            # separately before its early return, so this finally only runs
            # for the path where we actually hold the lock.
            lock_cm.__exit__(None, None, None)

    def _check_embedding_api(self) -> None:
        """Send one short request to a remote embedding API; raises
        :class:`EmbeddingAPIUnreachable` if it cannot be reached. Local
        models have nothing to check."""
        check = getattr(type(getattr(self.chroma_client, "embedding_function", None)), "check_reachable", None)
        if check is not None:
            check(self.chroma_client.embedding_function)

    def _prepare_and_classify_slice(
        self,
        items: list[dict[str, Any]],
        force_rebuild: bool = False,
        level: str | None = None,
    ) -> dict[str, Any]:
        """Build the documents for ``items`` and classify them existing-vs-new.

        Everything :meth:`_process_item_batch` does up to, but not including,
        handing documents to ChromaDB: assembling each item's text and
        metadata, splitting it into passages when chunking is on, truncating to
        the embedding model's limit, probing which items are already indexed,
        and clearing an item's stale passages before its new ones are written.

        Safe to call from a worker thread. The assembly loop touches no shared
        state; the two ChromaDB calls at the end are guarded by
        ``_chroma_call_lock`` so they can never interleave with a commit
        running on another thread.

        Returns the prepared parallel lists plus ``item_doc_counts`` — how many
        documents each entry in ``item_keys_order`` contributed — so a caller
        can split the batch on item boundaries without re-deriving them from
        the ids.

        ``level`` limits the slice to one level, for a run that writes every
        article record before any passage: ``SEARCH_ARTICLES`` skips items that
        get no article record (the passage pass counts them) and moves the text
        of items still to get passages to disk; ``SEARCH_PASSAGES`` reads it
        back and counts only items the article pass did not.
        """
        stats = {"processed": 0, "skipped": 0, "errors": 0}

        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []
        # One entry per *item* successfully prepared (not per chunk) so add/
        # update accounting stays item-granular regardless of chunking.
        item_keys_order: list[str] = []
        # Documents contributed by each entry of item_keys_order, so callers
        # can slice the flat lists back into whole items.
        item_doc_counts: list[int] = []

        held = self._held_text
        for item in self._as_units(items):
            try:
                article, passages = self._levels_to_write(item)
                counts = True
                if level == SEARCH_ARTICLES:
                    if not article:
                        continue
                    hold = passages
                    passages = False
                    records = self._build_item_records(item, chunking=False, article=True)
                    if hold and held is not None:
                        held.hold(item)
                elif level == SEARCH_PASSAGES:
                    counts = not article
                    if held is not None:
                        held.restore(item)
                    try:
                        records = self._build_item_records(item, chunking=passages, article=False)
                    finally:
                        _HeldText.release(item)
                else:
                    records = self._build_item_records(item, chunking=passages, article=article)
                if not records:
                    if counts:
                        stats["skipped"] += 1
                    continue

                documents.extend(record["document"] for record in records)
                metadatas.extend(record["metadata"] for record in records)
                ids.extend(record["id"] for record in records)
                item_keys_order.append(item["key"])
                item_doc_counts.append(len(records))
                if counts:
                    stats["processed"] += 1

            except Exception as e:
                logger.error(f"Error processing item {item.get('key', 'unknown')}: {e}")
                stats["errors"] += 1

        # Which items already existed (drives added-vs-updated). Stale
        # passages are pruned only after the new ones are written
        # (_prune_stale_chunks), so a failed embed never loses an item (#610).
        existing_item_keys: set[str] = set()
        if documents and not force_rebuild:
            with self._chroma_call_lock:
                # A unit is indexed if it has a record at either level.
                existing_item_keys = self._existing_item_keys(item_keys_order)

        return {
            "documents": documents,
            "metadatas": metadatas,
            "ids": ids,
            "item_keys_order": item_keys_order,
            "item_doc_counts": item_doc_counts,
            "existing_item_keys": existing_item_keys,
            "prep_stats": stats,
        }

    def _stream_index_items(
        self,
        all_items: list[dict[str, Any]],
        force_rebuild: bool,
        stats: dict[str, Any],
        failed_docs: list,
        embedding_function: Any,
        max_parallel: int,
        report_progress: Any,
        level: str | None = None,
        keep_cached: frozenset[str] = frozenset(),
        accounted_keys: set[str] | None = None,
    ) -> None:
        """Index ``all_items`` with preparation, embedding and commits overlapped.

        One producer thread prepares slices and splits them into request-sized
        payloads; ``max_parallel`` worker threads embed those payloads; the
        calling thread is the sole committer, buffering vectors and writing
        them with ``upsert_embeddings``. Only the workers block on the network,
        so several embedding requests are in flight while the next slice is
        being prepared and the previous one committed.

        Mutates ``stats`` and ``failed_docs`` in place, exactly as the
        synchronous path does, so the caller's end-of-run retry pass is shared
        between both paths. ``level``, ``keep_cached`` and ``accounted_keys``
        are as in :meth:`_process_item_batch`.
        """
        slice_size = _realtime_slice_size(max_parallel)
        request_batch_size = (
            getattr(embedding_function, "request_batch_size", None) or 64
        )

        # Bounded in payloads, not items: the producer must never materialize
        # the whole library's prepared documents or computed vectors at once.
        queue_size = max(2 * max_parallel, 4)
        chunk_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        vector_queue: queue.Queue = queue.Queue(maxsize=queue_size)

        stop_event = threading.Event()
        stats_lock = threading.Lock()
        failed_lock = threading.Lock()
        errors_lock = threading.Lock()
        thread_errors: list[BaseException] = []

        def bump(key: str, amount: int = 1) -> None:
            # The producer and the committer both write to `stats`, and
            # `stats[key] += n` is several bytecodes with a GIL release
            # possible in between, so every update goes through this lock.
            if not amount:
                return
            with stats_lock:
                stats[key] += amount

        def record_failures(documents, metadatas, ids) -> None:
            with failed_lock:
                failed_docs.extend(zip(documents, metadatas, ids))

        def record_thread_error(exc: BaseException) -> None:
            with errors_lock:
                thread_errors.append(exc)

        def producer() -> None:
            try:
                for start in range(0, len(all_items), slice_size):
                    if stop_event.is_set():
                        break
                    slice_items = all_items[start : start + slice_size]
                    for item in slice_items:
                        report_progress(item)

                    if level is None:
                        prepared = self._prepare_and_classify_slice(
                            slice_items, force_rebuild
                        )
                    else:
                        prepared = self._prepare_and_classify_slice(
                            slice_items, force_rebuild, level=level
                        )
                    prep = prepared["prep_stats"]
                    bump("processed_items", prep["processed"])
                    bump("skipped_items", prep["skipped"])
                    bump("errors", prep["errors"])

                    for payload in _split_prepared_into_requests(
                        prepared, request_batch_size
                    ):
                        if stop_event.is_set():
                            break
                        chunk_queue.put(payload)
            except BaseException as exc:  # noqa: BLE001 - re-raised by the caller
                record_thread_error(exc)
            finally:
                # Unconditional. A producer that died early must still release
                # every worker, or they block on get() forever and the commit
                # loop never sees the sentinels that end it.
                for _ in range(max_parallel):
                    chunk_queue.put(_STREAM_SENTINEL)

        def worker() -> None:
            try:
                while True:
                    payload = chunk_queue.get()
                    if payload is _STREAM_SENTINEL:
                        break
                    documents, metadatas, ids, item_keys = payload
                    try:
                        vectors = embedding_function(documents)
                    except Exception as exc:
                        # One sub-batch failing is not fatal: hand it to the
                        # end-of-run retry pass and keep the worker alive, so a
                        # single bad request cannot end the whole run.
                        logger.warning(
                            f"Embedding request failed ({exc}), saving for retry"
                        )
                        record_failures(documents, metadatas, ids)
                        bump("errors", len(documents))
                        continue
                    vector_queue.put((documents, metadatas, ids, item_keys, vectors))
            except BaseException as exc:  # noqa: BLE001 - re-raised by the caller
                record_thread_error(exc)
            finally:
                # Also unconditional, and exactly one per worker, so the commit
                # loop's countdown always reaches zero.
                vector_queue.put(_STREAM_SENTINEL)

        write_docs: list[str] = []
        write_metas: list[dict[str, Any]] = []
        write_ids: list[str] = []
        write_vectors: list[Any] = []
        write_keys: dict[str, bool] = {}
        if accounted_keys is None:
            accounted_keys = set()

        def flush() -> None:
            nonlocal write_docs, write_metas, write_ids, write_vectors, write_keys
            if not write_vectors:
                return
            try:
                with self._chroma_call_lock:
                    self.chroma_client.upsert_embeddings(
                        write_docs, write_metas, write_ids, write_vectors
                    )
                    self._prune_stale_chunks(write_metas)
                    if level == SEARCH_ARTICLES:
                        self._drop_replaced_passages(
                            [k for k, existed in write_keys.items() if existed and k in keep_cached]
                        )
            except Exception as exc:
                logger.warning(f"Batch upsert failed ({exc}), saving for retry")
                record_failures(write_docs, write_metas, write_ids)
                bump("errors", len(write_docs))
            else:
                for item_key, already_existed in write_keys.items():
                    if item_key in accounted_keys:
                        continue
                    accounted_keys.add(item_key)
                    bump("updated_items" if already_existed else "added_items")
                # Same contract as the synchronous path: the transient copy of
                # an item's extracted text has done its job once its embedding
                # is persisted, so evict per commit rather than at the end.
                try:
                    text_keys = _text_record_keys(write_metas)
                    if done := [k for k in write_keys if k not in keep_cached and k in text_keys]:
                        fulltext_cache.evict_many(done, config_path=self.config_path)
                except Exception as exc:
                    logger.debug(f"Fulltext cache eviction failed: {exc}")
            finally:
                write_docs, write_metas, write_ids, write_vectors = [], [], [], []
                write_keys = {}

        producer_thread = threading.Thread(
            target=producer, name="zmcp-index-producer", daemon=True
        )
        worker_threads = [
            threading.Thread(target=worker, name=f"zmcp-index-worker-{i}", daemon=True)
            for i in range(max_parallel)
        ]
        producer_thread.start()
        for thread in worker_threads:
            thread.start()

        active_workers = max_parallel
        try:
            while active_workers > 0:
                try:
                    payload = vector_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if payload is _STREAM_SENTINEL:
                    # Each worker emits its sentinel only after its last
                    # result, so once all of them have arrived the queue is
                    # drained by construction.
                    active_workers -= 1
                    continue
                documents, metadatas, ids, item_keys, vectors = payload
                write_docs.extend(documents)
                write_metas.extend(metadatas)
                write_ids.extend(ids)
                write_vectors.extend(vectors)
                for item_key, already_existed in item_keys:
                    write_keys[item_key] = already_existed
                if len(write_vectors) >= _STREAM_COMMIT_THRESHOLD:
                    flush()
            flush()
        except BaseException:
            # Ctrl-C lands here: it is delivered to the main thread, which is
            # this commit loop. Commit what has already been paid for, then
            # unblock any thread stalled on a full queue so it can reach its
            # sentinel-pushing finally block.
            stop_event.set()
            try:
                flush()
            except Exception:
                pass
            self._drain_stream_queues(
                (chunk_queue, vector_queue), [producer_thread, *worker_threads]
            )
            raise
        finally:
            producer_thread.join(timeout=5)
            for thread in worker_threads:
                thread.join(timeout=5)

        if thread_errors:
            raise thread_errors[0]

    @staticmethod
    def _drain_stream_queues(queues, threads, timeout: float = 5.0) -> None:
        """Discard queued work until every thread has exited, or ``timeout``.

        A thread blocked in ``put()`` on a full queue cannot reach its
        ``finally`` block, so on the abort path the queues have to be emptied
        for the pipeline to unwind. Whatever is discarded is safe to lose:
        indexing is idempotent per item key and Zotero remains the source of
        truth, so the next run picks those items up again.

        Thread cancellation is cooperative — a worker inside an HTTP request
        cannot be interrupted — so this bounds how long the caller waits, not
        how long the thread runs.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and any(t.is_alive() for t in threads):
            for q in queues:
                try:
                    while True:
                        q.get_nowait()
                except queue.Empty:
                    pass
            time.sleep(0.05)

    def _drop_replaced_passages(self, item_keys: list[str]) -> None:
        """Delete the old passages of items whose new article record was just written.

        Their new passages follow in the passage pass. A run stopped before
        then leaves each item with an article and no passages, which the gap
        fill finds; old passages left in place would describe the item before
        its change, and hide it from the gap fill for good.
        """
        if not hasattr(self.chroma_client, "prune_item_chunks") or not item_keys:
            return
        for item_key in item_keys:
            self.chroma_client.prune_item_chunks(item_key, 0)
        # The delete logs its errors and goes on, so check that it happened.
        try:
            left = {record_key(doc_id) for doc_id in self.chroma_client.get_existing_ids(
                [f"{item_key}#0" for item_key in item_keys]
            )}
        except Exception as e:
            logger.debug(f"Could not check the deleted passages: {e}")
            left = set(item_keys)
        for item_key in sorted(left):
            logger.warning(
                f"Could not delete the old passages of {item_key} from the ChromaDB collection; "
                "they stay until its new passages are written."
            )
            self._note_undropped_passages(item_key)

    def _note_undropped_passages(self, item_key: str) -> None:
        """Remember an item whose old passages are still stored after its new article."""
        if isinstance(self._undropped_passages, set):
            self._undropped_passages.add(item_key)

    def _drop_unlike_passages(self, item_key: str, digests: dict[str, str]) -> None:
        """Delete an item's stored passages that are not the ones its new article came with.

        A batch run sends article records before passages, so an item's new
        article can land with its passages still to come. Its old passages
        would describe the item before its change and hide it from the gap
        fill. Passages of this run that landed first match their digest, and
        stay.
        """
        passages = getattr(self.chroma_client, "passage_documents", None)
        if passages is None:
            return
        try:
            stale = [
                doc_id
                for doc_id, document in passages(item_key)
                if digests.get(doc_id) != _passage_digest(document or "")
            ]
            if stale:
                # The delete logs its errors and goes on, so check that it happened.
                self.chroma_client.delete_documents(stale)
                left = self.chroma_client.get_existing_ids(stale)
            else:
                left = set()
        except Exception as e:
            logger.debug(f"Could not read or delete the passages of {item_key}: {e}")
            left = {item_key}
        if left:
            logger.warning(
                f"Could not delete the old passages of {item_key} from the ChromaDB collection; "
                "they stay until its new passages are written."
            )
            self._note_undropped_passages(item_key)

    def _prune_stale_chunks(self, metadatas: list[dict[str, Any]]) -> None:
        """Drop each just-written item's passages past its new ``n_chunks``.

        Runs after a successful upsert, never before: write first, then
        prune, so a failed embed leaves the old passages in place (#610).
        Worker paths call this under ``_chroma_call_lock``. The end-of-run
        retry calls it without the lock, which is safe because no workers
        are running by then. It prunes once per item in ``metadatas``.
        """
        if not hasattr(self.chroma_client, "prune_item_chunks"):
            return
        if self._drop_passages_on_write:
            # Chunking is off, so an item written now has no current passages.
            # Left in place they would stay searchable by level, and hide the
            # item from the gap fill when chunking comes back on.
            # An item only getting its missing article record has not changed,
            # and those passages have been paid for.
            scopes = {
                m["item_key"]: m.get("group_id")
                for m in metadatas
                if m.get("item_key")
                and "chunk_index" not in m
                and m["item_key"] not in self._level_fill_keys
            }
            for item_key, group_id in scopes.items():
                self.chroma_client.delete_item_chunks(item_key, group_id=group_id)
        counts = {
            m["item_key"]: m["n_chunks"]
            for m in metadatas
            if "chunk_index" in m and "n_chunks" in m and m.get("item_key")
        }
        for item_key, n_chunks in counts.items():
            self.chroma_client.prune_item_chunks(item_key, n_chunks)
        # A legacy item's passages were keyed by the item itself; its new
        # bibliographic record has none, and its text now lives under its
        # attachments. Dropped once that record is written, as above (#610).
        legacy = {
            m["item_key"]: m.get("group_id")
            for m in metadatas
            if "chunk_index" not in m and m.get("item_key") in self._legacy_rewrites
        }
        for item_key, group_id in legacy.items():
            self.chroma_client.delete_item_chunks(item_key, group_id=group_id)

    def _process_item_batch(
        self,
        items: list[dict[str, Any]],
        force_rebuild: bool = False,
        _failed_docs: list | None = None,
        level: str | None = None,
        keep_cached: frozenset[str] = frozenset(),
        accounted: set[str] | None = None,
    ) -> dict[str, int]:
        """Prepare a batch of items and hand it to ChromaDB for embedding.

        The synchronous index path: preparation and classification happen in
        :meth:`_prepare_and_classify_slice`, then ``upsert_documents`` blocks
        while ChromaDB embeds the batch. The streaming path in
        ``update_database`` replaces only this second half.

        _failed_docs: optional list (passed by reference from update_database)
        that collects (doc_text, metadata, doc_id) tuples for batches that fail
        mid-run. Without this, the retry path at update_database:839-865 is
        dead code — a NameError raised here would crash the whole reindex,
        making every transient ChromaDB error fatal instead of recoverable.

        ``level``, ``keep_cached`` and ``accounted`` serve a run written in two
        passes: the level this pass writes, the items whose cached text the
        passage pass still needs, and the items already counted as added or
        updated.
        """
        if level is None:
            prepared = self._prepare_and_classify_slice(items, force_rebuild)
        else:
            prepared = self._prepare_and_classify_slice(items, force_rebuild, level=level)
        prep_stats = prepared["prep_stats"]
        stats = {
            "processed": prep_stats["processed"],
            "added": 0,
            "updated": 0,
            "skipped": prep_stats["skipped"],
            "errors": prep_stats["errors"],
        }

        documents = prepared["documents"]
        metadatas = prepared["metadatas"]
        ids = prepared["ids"]
        item_keys_order = prepared["item_keys_order"]
        existing_item_keys = prepared["existing_item_keys"]

        # Add documents to ChromaDB if any
        if documents:
            try:
                with self._chroma_call_lock:
                    self.chroma_client.upsert_documents(documents, metadatas, ids)
                    self._prune_stale_chunks(metadatas)
                    if level == SEARCH_ARTICLES:
                        self._drop_replaced_passages(
                            [k for k in item_keys_order if k in existing_item_keys and k in keep_cached]
                        )
                for k in item_keys_order:
                    if accounted is not None:
                        if k in accounted:
                            continue
                        accounted.add(k)
                    if k in existing_item_keys:
                        stats["updated"] += 1
                    else:
                        stats["added"] += 1
                # These items are embedded and persisted, so the transient
                # copy of their extracted text has done its job. Evicting per
                # batch rather than at the end of the run keeps the cache
                # roughly proportional to what is still un-embedded, instead
                # of growing to the size of the whole library.
                try:
                    text_keys = _text_record_keys(metadatas)
                    if done := [k for k in item_keys_order if k not in keep_cached and k in text_keys]:
                        fulltext_cache.evict_many(done, config_path=self.config_path)
                except Exception as e:
                    logger.debug(f"Fulltext cache eviction failed: {e}")
            except Exception as e:
                # Batch failed — collect failures for end-of-run retry.
                # ChromaDB's ONNX tokenizer can fail intermittently in bursts;
                # retrying immediately usually fails too. Collecting failures
                # and retrying after all batches are done is more effective.
                logger.warning(f"Batch upsert failed ({e}), saving for retry")
                if _failed_docs is not None:
                    for j in range(len(documents)):
                        _failed_docs.append((documents[j], metadatas[j], ids[j]))
                    # Count them as errors so stats are accurate
                    stats["errors"] += len(documents)
                else:
                    # No retry list — this is the legacy crash path; re-raise
                    # so caller sees the real error instead of hiding it.
                    raise

        return stats

    def _get_batch_status(self, provider: str, batch_ids: list[str] | None = None) -> dict[str, Any]:
        """Refresh and return Batch API status for the latest run or selected batches."""
        module = _batch_module(provider)
        label = _batch_adapter(provider).label
        selected_ids = set(batch_ids or [])
        manifest = module.find_manifest(
            config_path=self.config_path,
            batch_id=next(iter(selected_ids), None),
        )
        manifest = module.refresh_manifest_status(
            manifest,
            embedding_config=self.chroma_client.embedding_config,
            batch_ids=selected_ids or None,
        )
        batches = [
            batch for batch in manifest.get("batches", [])
            if not selected_ids or batch.get("batch_id") in selected_ids
        ]
        missing_ids = selected_ids - {batch.get("batch_id") for batch in batches}
        if missing_ids:
            raise FileNotFoundError(f"No {label} batch manifest entries found for: {', '.join(sorted(missing_ids))}")
        return {
            "provider": provider,
            "run_id": manifest.get("run_id"),
            "manifest_path": manifest.get("manifest_path"),
            "model": manifest.get("model"),
            "force_full_rebuild": manifest.get("force_full_rebuild", False),
            "batches": batches,
        }

    def get_openai_batch_status(self, batch_ids: list[str] | None = None) -> dict[str, Any]:
        """Refresh and return OpenAI Batch API status for the latest run or selected batches."""
        return self._get_batch_status("openai", batch_ids)

    def get_gemini_batch_status(self, batch_ids: list[str] | None = None) -> dict[str, Any]:
        """Refresh and return Gemini Batch API status for the latest run or selected batches."""
        return self._get_batch_status("gemini", batch_ids)

    def _superseded_by(self, provider: str, manifest: dict[str, Any]) -> str | None:
        """Run id of the newer run that has taken over this run's items, if any.

        "Newer" is the newest run *for the same library*, ranked by the
        immutable ``created_at``/``run_id`` order rather than by manifest mtime
        (which a status refresh or an import would bump, letting a superseded
        run pass itself off as current).
        """
        module = _batch_module(provider)
        newest = module.newest_manifest_for_group(
            config_path=self.config_path, group_id=manifest.get("group_id")
        )
        return str(newest.get("run_id")) if _is_superseded(manifest, newest) else None

    def _submit_pending_chunks(
        self,
        provider: str,
        manifest: dict[str, Any],
        client: Any,
        superseded_by: str | None,
        max_enqueued_tokens: int | None = None,
    ) -> int:
        """Submit this run's throttle-parked chunks; returns how many went out.

        The enqueued-token throttle parks overflow chunks as ``pending`` with no
        batch id. Only the auto-loop used to submit them, so a plain
        ``batch-import`` left them parked forever. Every submission of a parked
        chunk goes through here, so the one rule that must not be broken holds
        everywhere: a run that a newer one has superseded (``superseded_by``,
        from :meth:`_superseded_by`) submits nothing, because the newer run has
        already re-submitted those items and paying twice is real money.

        ``max_enqueued_tokens`` defaults to the budget recorded in the manifest,
        so a resumed run keeps the budget it was submitted with; a manifest
        predating that field submits nothing, which is safe. Submitted entries
        are updated in place, so every list the caller derived from
        ``manifest["batches"]`` sees the new batch ids.
        """
        if superseded_by:
            return 0
        module = _batch_module(provider)
        return module.submit_pending_batches(
            manifest,
            embedding_config=self.chroma_client.embedding_config,
            max_enqueued_tokens=max_enqueued_tokens,
            client=client,
        )

    def _import_batch(
        self,
        provider: str,
        batch_ids: list[str] | None = None,
        _skip_lock: bool = False,
    ) -> dict[str, Any]:
        """Import completed Batch API embeddings into ChromaDB.

        ``_skip_lock`` is for the auto-loop, which already holds the update
        lock via ``update_database``; re-acquiring it would self-deadlock.
        """
        module = _batch_module(provider)
        adapter = _batch_adapter(provider)
        label = adapter.label
        selected_ids = set(batch_ids or [])
        manifest = module.find_manifest(
            config_path=self.config_path,
            batch_id=next(iter(selected_ids), None),
        )
        # Whether a newer run has taken this one's items over decides two
        # things below: no more chunks may be submitted for it, and its sync
        # watermark must stay where it is. Settled once here, before the
        # refresh, so every decision in this import agrees.
        superseded_by = self._superseded_by(provider, manifest)
        manifest = module.refresh_manifest_status(
            manifest,
            embedding_config=self.chroma_client.embedding_config,
            batch_ids=selected_ids or None,
        )

        all_batches = manifest.get("batches", [])
        batches = [
            batch for batch in all_batches
            if not selected_ids or batch.get("batch_id") in selected_ids
        ]
        missing_ids = selected_ids - {batch.get("batch_id") for batch in batches}
        if missing_ids:
            raise FileNotFoundError(f"No {label} batch manifest entries found for: {', '.join(sorted(missing_ids))}")
        if not batches:
            raise ValueError(f"No matching {label} batches found in the local manifest")
        if manifest.get("force_full_rebuild") and selected_ids and len(batches) != len(all_batches):
            raise RuntimeError(f"Force-rebuild {label} batch runs must be imported as a complete run")

        stats = {
            "provider": provider,
            "run_id": manifest.get("run_id"),
            "manifest_path": manifest.get("manifest_path"),
            "batches_seen": len(batches),
            "batches_imported": 0,
            "batches_skipped": 0,
            "batches_submitted": 0,
            "imported_items": 0,
            "added_items": 0,
            "updated_items": 0,
            "failed_items": 0,
            "missing_items": 0,
            "errors": [],
        }
        if superseded_by:
            # One entry, recorded once, covering both consequences.
            stats["errors"].append({"error": (
                f"run {manifest.get('run_id')} is superseded by run {superseded_by}: its pending "
                "chunks are not submitted and its sync watermark is not promoted, because the "
                "newer run covers the same items"
            )})

        lock_path = _update_lock_path()
        lock_cm = contextlib.nullcontext(True) if _skip_lock else _acquire_update_lock(lock_path)
        acquired = lock_cm.__enter__()
        if not acquired:
            lock_cm.__exit__(None, None, None)
            raise RuntimeError(f"Another semantic-search update is already running (lock held at {lock_path})")

        try:
            client = adapter.create_client(self.chroma_client.embedding_config)
            # A separate-collection index imported with chunking off drops the
            # passages of items it rewrites, as a realtime run would.
            layout_of = getattr(self.chroma_client, "index_layout", None)
            layout = layout_of() if layout_of else None
            self._drop_passages_on_write = (
                self._separate_passages()
                and not self._chunking_enabled
                and isinstance(layout, dict)
                and bool(layout.get("passage_records"))
            )

            if manifest.get("force_full_rebuild"):
                def _incomplete() -> list[str]:
                    return [
                        batch.get("batch_id") or "(pending)"
                        for batch in all_batches
                        if not batch.get("imported_at")
                        and batch_common._entry_state(adapter, batch) not in batch_common.IMPORTABLE_STATES
                    ]

                if _incomplete():
                    # A force-rebuild run imports all-or-nothing. Before
                    # refusing, submit whatever the throttle parked: that is
                    # the progress the user is waiting for, not an error. The
                    # submission runs under the update lock (unlike the refusal
                    # it replaces), since it writes the manifest.
                    submitted = 0
                    if not selected_ids:
                        submitted = self._submit_pending_chunks(provider, manifest, client, superseded_by)
                        stats["batches_submitted"] += submitted
                    still_incomplete = _incomplete()
                    if still_incomplete and submitted:
                        stats["deferred"] = (
                            f"{submitted} pending chunk(s) submitted; the force-rebuild run imports once "
                            f"all batches complete (waiting on: {', '.join(still_incomplete)})"
                        )
                        return stats
                    if still_incomplete:
                        message = (
                            f"Force-rebuild {label} batch runs can only be imported after all batches complete: "
                            + ", ".join(still_incomplete)
                        )
                        if superseded_by:
                            message += f" (run {superseded_by} has since superseded this one)"
                        raise RuntimeError(message)

            already_imported = any(batch.get("imported_at") for batch in all_batches)
            if (
                manifest.get("force_full_rebuild")
                and not already_imported
                and any(not batch.get("imported_at") for batch in batches)
            ):
                self.chroma_client.reset_collection()

            for batch in batches:
                if batch.get("imported_at"):
                    stats["batches_skipped"] += 1
                    continue
                if not batch.get("batch_id"):
                    # Parked by the enqueued-token throttle; submitted below,
                    # after this pass imports what has completed.
                    stats["batches_skipped"] += 1
                    continue
                if batch_common._entry_state(adapter, batch) not in batch_common.IMPORTABLE_STATES:
                    stats["batches_skipped"] += 1
                    stats["errors"].append({
                        "batch_id": batch.get("batch_id"),
                        "error": f"Batch status is {batch.get('status')}, not {_IMPORTABLE_DESC[provider]}",
                    })
                    continue
                if adapter.uses_error_file and not batch.get("output_file_id"):
                    stats["batches_skipped"] += 1
                    stats["errors"].append({"batch_id": batch.get("batch_id"), "error": "Missing output_file_id"})
                    continue

                records_path = Path(batch["records_path"])
                output_path = records_path.with_name(records_path.stem + "-output.jsonl")
                if output_path.exists():
                    output_text = output_path.read_text(encoding="utf-8")
                else:
                    output_text = adapter.download_output(client, batch, output_path)

                # Rows without a correlation key are matched positionally
                # against the exact submitted order, so the records file is
                # read before parsing rather than after.
                chunk_records = module.read_jsonl(records_path)
                records = {record["id"]: record for record in chunk_records}
                id_order = [record["id"] for record in chunk_records]
                embeddings_by_id, row_failures = adapter.parse_output(output_text, id_order)

                error_path = records_path.with_name(records_path.stem + "-errors.jsonl")
                row_failures.extend(adapter.download_errors(client, batch, error_path))
                ids = [doc_id for doc_id in embeddings_by_id if doc_id in records]
                if superseded_by:
                    # The newer run has these items' current passages. Writing
                    # these over them could bring back text from before the
                    # change, and passages past the item's new length.
                    ids = [doc_id for doc_id in ids if "#" not in doc_id]
                unexpected_output_ids = [doc_id for doc_id in embeddings_by_id if doc_id not in records]
                failure_ids = {
                    failure.get("custom_id")
                    for failure in row_failures
                    if failure.get("custom_id")
                }
                missing_result_ids = [
                    doc_id
                    for doc_id in records
                    if doc_id not in embeddings_by_id and doc_id not in failure_ids
                ]
                missing_errors = [
                    {"custom_id": doc_id, "error": "Batch output returned an embedding for an unknown record"}
                    for doc_id in unexpected_output_ids
                ] + [
                    {"custom_id": doc_id, "error": "No embedding or error row returned for batch record"}
                    for doc_id in missing_result_ids
                ]
                stats["missing_items"] += len(unexpected_output_ids) + len(missing_result_ids)
                stats["failed_items"] += len(row_failures)
                stats["errors"].extend(row_failures)
                stats["errors"].extend(missing_errors)

                if ids:
                    # Item-level counts, each item counted where it first appears:
                    # its article record, or its first passage when the run had
                    # no article for it.
                    first_records = {
                        record_key(doc_id)
                        for doc_id in ids
                        if "#" not in doc_id
                        or (doc_id.endswith("#0") and not records[doc_id].get("counted_with_article"))
                    }
                    existing_items = self._existing_item_keys(sorted(first_records))
                    metadatas = [records[doc_id]["metadata"] for doc_id in ids]
                    self.chroma_client.upsert_embeddings(
                        documents=[records[doc_id]["document"] for doc_id in ids],
                        metadatas=metadatas,
                        ids=ids,
                        embeddings=[embeddings_by_id[doc_id] for doc_id in ids],
                    )
                    # After the write, as on the realtime path (#610). Each record
                    # carries its item's new length, so this is safe with the
                    # item's other passages still in a later file.
                    self._level_fill_keys = {
                        record_key(doc_id) for doc_id in ids if records[doc_id].get("level_fill")
                    }
                    self._legacy_rewrites = {
                        record_key(doc_id) for doc_id in ids if records[doc_id].get("legacy_rewrite")
                    }
                    self._prune_stale_chunks(metadatas)
                    followed = [
                        doc_id for doc_id in ids if records[doc_id].get("passages_follow")
                    ]
                    if not superseded_by:
                        for doc_id in followed:
                            self._drop_unlike_passages(doc_id, records[doc_id]["passages_follow"])
                    stats["imported_items"] += len(ids)
                    stats["updated_items"] += len(existing_items)
                    stats["added_items"] += len(first_records) - len(existing_items)
                    # Same contract as the realtime path: the embeddings are
                    # in ChromaDB, so the transient copy of their extracted
                    # text has done its job. Without this the cache grows to
                    # hold the whole library on the batch flow, since nothing
                    # else on it ever evicts. Text is cached under the record
                    # key of the attachment it came from (#826).
                    # An article whose passages are in a later file keeps it:
                    # they may still need to be extracted or rebuilt.
                    try:
                        text_keys = _text_record_keys(metadatas)
                        fulltext_cache.evict_many(
                            {
                                record_key(doc_id) for doc_id in ids
                                if doc_id not in followed and record_key(doc_id) in text_keys
                            },
                            config_path=self.config_path,
                        )
                    except Exception as e:
                        logger.debug(f"Fulltext cache eviction failed: {e}")

                batch["imported_at"] = datetime.now().isoformat()
                batch["imported_count"] = len(ids)
                stats["batches_imported"] += 1

            module.save_manifest(manifest)

            if not selected_ids:
                # Importing the completed batches freed enqueued-token headroom,
                # so this is where the chunks the throttle parked get their turn.
                # An import of named batch ids never does this: the user asked
                # for those batches, not for the run to be carried forward.
                stats["batches_submitted"] += self._submit_pending_chunks(
                    provider, manifest, client, superseded_by
                )

            # Newly submitted chunks have no ``imported_at``, so the watermark
            # stays where it is until the whole run has landed.
            if all(batch.get("imported_at") for batch in all_batches):
                if superseded_by:
                    # The newer run was cut from this same watermark and will
                    # advance it when it lands; doing it here would skip
                    # whatever changed between the two runs.
                    logger.debug(
                        "Run %s imported completely but is superseded by %s; "
                        "leaving the sync watermark to the newer run",
                        manifest.get("run_id"),
                        superseded_by,
                    )
                else:
                    self.update_config["last_update"] = datetime.now().isoformat()
                    # Promote the watermark of the library the batch was submitted
                    # against, not whichever library happens to be active now.
                    manifest_group_id = manifest.get("group_id")
                    self._save_update_config(
                        last_sync_version=manifest.get("target_sync_version"),
                        library_key=None if manifest_group_id is None else str(manifest_group_id),
                    )
            return stats
        finally:
            self._drop_passages_on_write = False
            self._level_fill_keys = frozenset()
            self._legacy_rewrites = frozenset()
            lock_cm.__exit__(None, None, None)

    def import_openai_batch(self, batch_ids: list[str] | None = None) -> dict[str, Any]:
        """Import completed OpenAI Batch API embeddings into ChromaDB."""
        return self._import_batch("openai", batch_ids)

    def import_gemini_batch(self, batch_ids: list[str] | None = None) -> dict[str, Any]:
        """Import completed Gemini Batch API embeddings into ChromaDB."""
        return self._import_batch("gemini", batch_ids)

    def auto_loop_batch_pipeline(
        self,
        provider: str,
        poll_interval: int = 60,
        max_enqueued_tokens: int | None = None,
        stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Drive a throttled batch run to completion: poll, import, submit pending.

        Loops until every entry in the latest run's manifest is imported, or
        until no further progress is possible (everything left is terminal and
        nothing can be submitted). The on-disk manifest is consistent at every
        step, so after Ctrl-C or a crash a later ``batch-import`` imports what
        completed and submits what is still pending; a new ``update-db --batch``
        starts a fresh run instead.

        Must be called with the update lock already held (``update_database``
        holds it), hence ``_skip_lock`` on the imports below.
        """
        import time

        module = _batch_module(provider)
        adapter = _batch_adapter(provider)
        label = adapter.label
        aggregate = {"provider": provider, "polls": 0, "imported_items": 0, "submitted_chunks": 0}

        while True:
            imported_submitted = 0
            try:
                import_stats = self._import_batch(provider, _skip_lock=True)
                aggregate["imported_items"] += import_stats.get("imported_items", 0)
                # The import submits this run's parked chunks itself, so those
                # count here too; otherwise the tally below reports 0 for
                # chunks that were in fact submitted this poll.
                imported_submitted = import_stats.get("batches_submitted", 0)
            except RuntimeError as e:
                # Force-rebuild manifests are all-or-nothing, so _import_batch
                # refuses until every chunk is importable. Expected mid-run.
                logger.debug(f"auto-loop import deferred: {e}")
            aggregate["polls"] += 1

            manifest = module.find_manifest(config_path=self.config_path)
            client = adapter.create_client(self.chroma_client.embedding_config)
            # Through _submit_pending_chunks, not straight to the provider
            # module: the loop must obey the same supersession rule as every
            # other submission path.
            superseded_by = self._superseded_by(provider, manifest)
            if superseded_by:
                logger.debug(
                    "auto-loop: run %s is superseded by %s; its pending chunks stay parked",
                    manifest.get("run_id"),
                    superseded_by,
                )
            submitted = imported_submitted + self._submit_pending_chunks(
                provider, manifest, client, superseded_by, max_enqueued_tokens=max_enqueued_tokens
            )
            aggregate["submitted_chunks"] += submitted

            entries = manifest.get("batches", [])
            remaining = [b for b in entries if not b.get("imported_at")]
            if not remaining:
                break
            active = [
                b for b in remaining
                if b.get("batch_id")
                and batch_common._entry_state(adapter, b) not in batch_common.TERMINAL_STATES
            ]
            if not active and submitted == 0:
                failed = [b.get("batch_id") or "(pending)" for b in remaining]
                _report(
                    f"  [{label} auto-loop] no progress possible - {len(remaining)} chunk(s) "
                    f"failed or are stuck ({', '.join(failed)}). Inspect with 'zotero-mcp batch-status'.\n"
                )
                aggregate["stalled"] = failed
                break

            n_pending = sum(1 for b in remaining if b.get("status") == batch_common.STATE_PENDING)
            _report(
                f"  [{label} auto-loop] {len(entries) - len(remaining)}/{len(entries)} chunks imported, "
                f"{submitted} newly submitted, {n_pending} pending; next poll in {poll_interval}s.\n"
            )
            time.sleep(poll_interval)

        if "stalled" not in aggregate:
            _report(f"  [{label} auto-loop] run complete: {aggregate['imported_items']} embeddings imported.\n")
        if stats is not None:
            stats["auto_loop"] = aggregate
        return aggregate


    def search(self,
               query: str,
               limit: int = 10,
               filters: dict[str, Any] | None = None,
               group_id: int | None = None,
               level: str | None = None) -> dict[str, Any]:
        """
        Perform semantic search over the Zotero library.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            filters: Optional filters. Stored fields (library, passage
                location) are filtered by ChromaDB; bibliographic fields are
                evaluated against each hit's owner as Zotero has it now (#826).
            group_id: Restrict results to one library (0 = personal, else
                groupID). ``None`` (default) searches every indexed library —
                DB-side filtering via a ChromaDB ``where`` clause, never a
                Python post-filter.
            level: ``"articles"`` (one record per item, for finding works on a
                topic) or ``"passages"`` (for finding where something is done
                or stated). ``None`` (default) follows the chunking setting.

        Returns:
            Search results with Zotero item details
        """
        try:
            search_level, _chunked, level_note, level_error = self._resolve_search_level(level)
            if level_error:
                return {
                    "query": query,
                    "limit": limit,
                    "filters": filters,
                    "group_id": group_id,
                    "results": [],
                    "total_found": 0,
                    "error": level_error,
                }
            reranker = self._get_reranker()
            multiplier = max(1, int(self._reranker_config.get("candidate_multiplier", 3) or 3))
            # Hits collapse to their owners and trashed ones are dropped, at
            # either level: an attachment's record and its parent's
            # bibliographic record are two hits of one owner, and passages
            # several more. So fetch several candidates per wanted result.
            fetch_limit = limit * 4
            if reranker:
                fetch_limit = max(fetch_limit, limit * multiplier)

            chroma_filters, live_where = split_filters(filters)
            where_parts = [chroma_filters]
            if group_id is not None:
                where_parts.append({"group_id": int(group_id)})
            prefilter_keys = self._prefilter_keys(live_where, group_id) if live_where else None
            if prefilter_keys is not None:
                where_parts.append({"item_key": {"$in": sorted(prefilter_keys)}})
            where = combine(where_parts)
            # A post-filter can discard most candidates, so it may ask again
            # for more; a pre-filtered query already holds only matches.
            post_filter = bool(live_where) and prefilter_keys is None
            rounds = FILTER_QUERY_ROUNDS if post_filter else 1
            # Post-filtering re-ranks only the hits that passed the filter, at
            # most this many: the candidate set grows ×4 a round, and hosted
            # re-rankers cap and bill per document.
            rerank_bound = limit * max(4, multiplier)

            enriched_results: list[dict[str, Any]] = []
            rerank_info: dict[str, Any] | None = None
            load_note = None
            candidates = 0
            if prefilter_keys is None or prefilter_keys:
                for _ in range(rounds):
                    if not post_filter:
                        results, rerank_info, load_note, candidates = self._query_candidates(
                            query, fetch_limit, where, search_level, reranker
                        )
                        # Enrich results with full Zotero item data, collapsing
                        # hits to their owners and capping at `limit` owners.
                        enriched_results = self._enrich_search_results(results, query, limit)
                        break
                    results, _, load_note, candidates = self._query_candidates(
                        query, fetch_limit, where, search_level, None
                    )
                    survivors = self._filter_hits(self._resolve_hits(_collect_hits(results)), live_where)
                    enough = len({h["owner_key"] for h in survivors}) >= limit
                    if reranker and survivors:
                        survivors = survivors[:rerank_bound]
                        order, scores, rerank_info = self._rerank(
                            query, [h["document"] for h in survivors], reranker
                        )
                        survivors = [survivors[i] for i in order]
                        if scores is not None:
                            for hit, score in zip(survivors, scores):
                                hit["rerank_score"] = score
                    enriched_results = self._build_results(survivors, query, limit)
                    if enough or candidates < fetch_limit:
                        break
                    fetch_limit *= 4

            filter_note = None
            if rounds > 1 and len(enriched_results) < limit:
                found = len(enriched_results)
                filter_note = (
                    f"Only {found} item{'' if found == 1 else 's'} matched the filters among the "
                    f"{candidates} closest records searched."
                )

            return {
                "query": query,
                "limit": limit,
                "filters": filters,
                "group_id": group_id,
                "results": enriched_results,
                "total_found": len(enriched_results),
                "rerank": rerank_info,
                "level": search_level,
                "level_note": level_note,
                "load_note": load_note,
                "filter_note": filter_note,
            }

        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            error = str(e)
            try:
                # A damaged index fails here with ChromaDB's own words; say what it means (#747).
                persist_directory = self.chroma_client.persist_directory
                facts = unreadable_index_facts(persist_directory)
                if facts and self.chroma_client.read_error():
                    error = unreadable_index_message(
                        {"error": error, "persist_directory": persist_directory, **facts},
                        self.update_config.get("last_update"),
                    )
            except Exception:
                pass
            return {
                "query": query,
                "limit": limit,
                "filters": filters,
                "group_id": group_id,
                "results": [],
                "total_found": 0,
                "error": error,
                # The index is local, but each query is embedded by the API.
                "embedding_api_unreachable": isinstance(e, EmbeddingAPIUnreachable),
            }

    def _query_candidates(
        self,
        query: str,
        fetch_limit: int,
        where: dict[str, Any] | None,
        search_level: str | None,
        reranker: Any,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, str | None, int]:
        """Query ChromaDB and re-rank: ``(results, rerank info, load note, candidates)``.

        ``candidates`` is how many records ChromaDB returned, so a caller can
        tell a short answer from an exhausted index.
        """
        search_kwargs: dict[str, Any] = {}
        # On an unmigrated index everything is in the article collection,
        # which is what the client searches by default.
        if search_level == SEARCH_PASSAGES and self.chroma_client.separate_passages():
            search_kwargs["level"] = SEARCH_PASSAGES
        results = self.chroma_client.search(
            query_texts=[query], n_results=fetch_limit, where=where, **search_kwargs
        )
        load_note = None
        if isinstance(results, dict):
            results.pop("embedding_seconds", None)  # in-process; the worker drops it itself
            load_note = _passage_load_note(results.pop("passage_load", None))
        candidates = len((results.get("ids") or [[]])[0] or [])

        _drop_missing_documents(results)

        rerank_info: dict[str, Any] | None = None
        if reranker and results.get("documents") and results["documents"][0]:
            # Rank every candidate at either level: collapsing to `limit`
            # owners happens in enrichment, after trashed hits are dropped.
            ranked_indices, scores, rerank_info = self._rerank(query, results["documents"][0], reranker)
            for key in ["ids", "distances", "documents", "metadatas"]:
                if results.get(key) and results[key][0]:
                    results[key][0] = [results[key][0][i] for i in ranked_indices]
            if scores is not None:
                results["rerank_scores"] = [scores]
        return results, rerank_info, load_note, candidates

    @staticmethod
    def _rerank(
        query: str, documents: list[str], reranker: Any
    ) -> tuple[list[int], list[float] | None, dict[str, Any]]:
        """Re-rank ``documents``: ``(order, scores or None, rerank info)``."""
        top_k = len(documents)
        if isinstance(reranker, (APIReranker, CrossEncoderReranker)):
            pairs, rerank_error = reranker.rerank_detailed(query, documents, top_k)
        else:  # any other object honouring the plain rerank() contract
            pairs = [(i, 0.0) for i in reranker.rerank(query, documents, top_k=top_k)]
            rerank_error = None
        scores = None
        if rerank_error is None and isinstance(reranker, (APIReranker, CrossEncoderReranker)):
            scores = [score for _, score in pairs]
        info = {
            "model": getattr(reranker, "label", type(reranker).__name__),
            "applied": rerank_error is None,
            "error": rerank_error,
            "candidates": len(documents),
        }
        return [i for i, _ in pairs], scores, info

    def _prefilter_keys(self, live_where: dict[str, Any] | None, group_id: int | None) -> set[str] | None:
        """Record keys a live filter can match, from zotero.sqlite; None if unknown.

        Local mode only: the filter is evaluated over every owner in scope,
        and the matches expand to the owners' keys plus their child
        attachments' keys, which covers every record an owner can have (a
        legacy record's ``item_key`` is its parent's, so it matches too).
        None, so that the caller post-filters instead, outside local mode,
        when the filter needs hit-time data (``has_fulltext``, or passage
        fields under a mixed ``$or``), or above ``PREFILTER_MAX_KEYS`` keys.
        """
        if not live_where or referenced_fields(live_where) - OWNER_FIELDS:
            return None
        try:
            reader = self._open_local_reader()
        except Exception as e:
            logger.debug(f"Filter pre-check: no local database ({e})")
            return None
        if reader is None:
            return None
        try:
            owners = reader.get_filter_owners(group_id=group_id)
        except Exception as e:
            logger.warning(f"Filter pre-check against zotero.sqlite failed: {e}")
            return None
        finally:
            reader.close()
        keys: set[str] = set()
        for owner in owners:
            if matches(live_where, live_fields(owner)):
                keys.add(owner["key"])
                keys.update(owner.get("attachment_keys") or ())
                if len(keys) > PREFILTER_MAX_KEYS:
                    return None
        return keys

    def _resolve_search_level(
        self, requested: str | None
    ) -> tuple[str | None, bool, str | None, str | None]:
        """Decide what to search: ``(level, whether hits are passages, note, error)``.

        Reads the layout from SQLite, so choosing costs no vector index load.
        A client double with no layout gets the behaviour from before levels.
        """
        layout_of = getattr(self.chroma_client, "index_layout", None)
        layout = layout_of() if layout_of else None
        if not isinstance(layout, dict):
            return requested, self._chunking_enabled, None, None
        if requested not in (None, SEARCH_ARTICLES, SEARCH_PASSAGES):
            return None, False, None, (
                f"Unknown search level {requested!r}: use '{SEARCH_ARTICLES}' or '{SEARCH_PASSAGES}'."
            )

        if layout["unmigrated_passages"]:
            # Passages are in the article collection, which is searched either
            # way, as it always was until this index is migrated.
            if requested == SEARCH_ARTICLES and layout["article_records"]:
                # Articles are there too, so this index is mixed, not passage-only.
                return None, False, None, (
                    "This index keeps article and passage records together in one ChromaDB collection, "
                    "so articles can't be searched on their own. To search them separately, run "
                    "'zotero-mcp migrate-index', then 'zotero-mcp update-db'."
                )
            if requested == SEARCH_ARTICLES:
                return None, False, None, (
                    "This index holds passages only, in the article-level ChromaDB collection. To search "
                    "articles too, run 'zotero-mcp migrate-index', then 'zotero-mcp update-db'."
                )
            return SEARCH_PASSAGES, self._chunking_enabled or requested == SEARCH_PASSAGES, None, None

        if requested == SEARCH_PASSAGES and not layout["passage_records"]:
            return None, False, None, (
                "This index has no passages. Enable semantic_search.chunking in config.json "
                "and run 'zotero-mcp update-db' to add them."
            )
        if requested == SEARCH_ARTICLES and not layout["article_records"] and layout["passage_records"]:
            return None, False, None, (
                "This index has no article records yet. Run 'zotero-mcp update-db' to embed them."
            )
        level = requested or self._default_search_level(layout)
        return level, level == SEARCH_PASSAGES, self._search_level_note(level, layout), None

    def _default_search_level(self, layout: dict[str, Any]) -> str:
        """The level to search when the caller did not choose one.

        The chunking setting decides, so results never switch level because an
        index run reached some point. The other level is used only while the
        chosen one has no records at all, and the note says so.
        """
        articles, passages = layout["article_records"], layout["passage_records"]
        if self._chunking_enabled:
            return SEARCH_PASSAGES if passages else SEARCH_ARTICLES
        return SEARCH_ARTICLES if articles or not passages else SEARCH_PASSAGES

    def _search_level_note(self, level: str, layout: dict[str, Any]) -> str | None:
        """Tell the caller when the other level holds more, or a level is missing.

        A passage fill can take days, and until it ends passages cover only part
        of the library. An agent can't see that from the results, so it gets the
        counts and can decide to search the other level. The counts are for the
        whole index, whichever library was searched. An index whose levels cover
        the same items, one with articles alone and chunking off, or one still
        unmigrated gets no note; an index with passages alone is told no article
        records are indexed yet.
        """
        if layout["unmigrated_passages"] or layout["passage_items"] is None:
            # No passage count could be read: give no counts rather than wrong ones.
            return None
        items = {SEARCH_ARTICLES: layout["article_records"], SEARCH_PASSAGES: layout["passage_items"]}
        other = SEARCH_PASSAGES if level == SEARCH_ARTICLES else SEARCH_ARTICLES
        searched, elsewhere = items[level], items[other]
        if not searched:
            return None

        note = f"Searched {level} ({_count_items(searched)} in the index)."
        if elsewhere:
            # Said only when it could change what the agent does next: every
            # search would otherwise pay for a line that tells it nothing.
            if elsewhere <= searched:
                return None
            return f'{note} The {other[:-1]} level has {_count_items(elsewhere)}, so level="{other}" may find more.'
        if level == SEARCH_PASSAGES:
            return f"{note} No article records are indexed yet."
        return f"{note} No passages are indexed yet." if self._chunking_enabled else None

    def _enrich_search_results(
        self,
        chroma_results: dict[str, Any],
        query: str,
        limit: int | None = None,
        live_where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve hits to their owners live, filter, collapse, cap (#826).

        A hit's record key is ``record_key(id)``: an attachment's own key
        for a new record, the parent's for a legacy one. It is resolved
        against Zotero now, not from anything stored: its owner is its parent
        if it has one, else the item itself, so a re-parented attachment shows
        under its new parent. Hits whose item or owner is trashed or gone are
        dropped. ``live_where`` is then evaluated per hit, against the owner's
        current data with passage fields read from the hit. Results collapse
        by owner, the best-ranked hit winning, and are capped at ``limit``.
        """
        hits = _collect_hits(chroma_results)
        if not hits:
            return []
        return self._build_results(self._filter_hits(self._resolve_hits(hits), live_where), query, limit)

    def _filter_hits(
        self, resolved: list[dict[str, Any]], live_where: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """The resolved hits that pass ``live_where``, in order (all of them without one)."""
        if not live_where:
            return resolved
        fulltext_owners = {
            h["owner_key"] for h in resolved
            if "owner" in h and self._hit_has_fulltext(h["meta"], h["record"])
        }
        kept = []
        for hit in resolved:
            if "owner" not in hit:
                continue  # unresolvable, so it cannot be shown to match
            fields = {
                **hit["meta"],
                **live_fields(hit["owner"]),
                "has_fulltext": hit["owner_key"] in fulltext_owners,
            }
            if matches(live_where, fields):
                kept.append(hit)
        return kept

    def _build_results(
        self, resolved: list[dict[str, Any]], query: str, limit: int | None
    ) -> list[dict[str, Any]]:
        """Collapse resolved hits by owner, best first, and cap at ``limit``.

        Each result describes its owner (``item_key``, ``zotero_item``) and
        the hit that found it: the matched text, a grounded
        ``matched_passage``, passage provenance from the hit's metadata, and
        ``matched_attachment`` when the hit's record is an attachment of the
        owner (not the owner itself).
        """
        enriched: list[dict[str, Any]] = []
        seen_owners: set[str] = set()
        for hit in resolved:
            if hit["owner_key"] in seen_owners:
                continue
            seen_owners.add(hit["owner_key"])

            meta, document, distance = hit["meta"], hit["document"], hit["distance"]
            passage, passage_offset = best_snippet(query, document)

            enriched_result: dict[str, Any] = {
                "item_key": hit["owner_key"],
                "similarity_score": (1 - distance) if distance is not None else 0,
                "matched_text": document,
                "matched_passage": passage,
                "metadata": meta,
                "query": query,
            }
            if "rerank_score" in hit:
                enriched_result["rerank_score"] = hit["rerank_score"]
            # Passage provenance — present only on a chunk-indexed collection.
            for mk in ("chunk_index", "n_chunks", "char_start", "char_end", "page", "section"):
                if mk in meta:
                    enriched_result[mk] = meta[mk]
            if "char_start" not in enriched_result and passage_offset:
                enriched_result["passage_offset"] = passage_offset

            if "owner" in hit:
                enriched_result["zotero_item"] = hit["owner"]
                record_data = hit["record"].get("data") or {}
                if hit["record_key"] != hit["owner_key"] and record_data.get("itemType") == "attachment":
                    enriched_result["matched_attachment"] = {
                        "key": hit["record_key"],
                        "title": record_data.get("title", ""),
                        "filename": record_data.get("filename", ""),
                    }
            else:
                logger.error(f"Error enriching result for item {hit['owner_key']}: {hit['error']}")
                enriched_result["error"] = f"Could not fetch full item data: {hit['error']}"

            enriched.append(enriched_result)
            if limit and len(enriched) >= limit:
                break
        return enriched

    @staticmethod
    def _hit_has_fulltext(meta: dict[str, Any], record: dict[str, Any]) -> bool:
        """Whether one hit shows its owner has indexed full text.

        Defined from what a hit carries, since nothing else is known at search
        time: an attachment's record is file text; a legacy record says so in
        its own ``has_fulltext``; a bibliographic record alone does not. An
        owner has full text when any of its candidate hits does, so one whose
        attachment records were not among the candidates reads as False.
        """
        return meta.get("has_fulltext") is True or (record.get("data") or {}).get("itemType") == "attachment"

    def _resolve_hits(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach ``record``, ``owner`` and ``owner_key`` to each hit, in place.

        Two batched lookups per library: the hits' own keys, then the parents
        not already fetched. Trashed or missing hits and owners are left out
        of the returned list. A hit that could not be looked up at all, such
        as one from a library this configuration cannot reach, stays with an
        ``error`` and its record key as owner, so the failure is reported
        rather than silently dropped (#492).
        """
        wanted: dict[int | None, set[str]] = {}
        for hit in hits:
            wanted.setdefault(hit["library"], set()).add(hit["record_key"])
        items, errors = self._resolve_items(wanted)

        parents: dict[int | None, set[str]] = {}
        for (library, _key), item in items.items():
            parent = (item.get("data") or {}).get("parentItem")
            if parent and not _is_trashed(item) and (library, parent) not in items and (library, parent) not in errors:
                parents.setdefault(library, set()).add(parent)
        if parents:
            more_items, more_errors = self._resolve_items(parents)
            items.update(more_items)
            errors.update(more_errors)

        resolved: list[dict[str, Any]] = []
        for hit in hits:
            library, key = hit["library"], hit["record_key"]
            record = items.get((library, key))
            if record is None:
                if (library, key) in errors:
                    resolved.append({**hit, "owner_key": key, "error": errors[(library, key)]})
                continue  # gone from Zotero since it was indexed
            owner_key = (record.get("data") or {}).get("parentItem") or key
            owner = items.get((library, owner_key))
            if owner is None:
                if (library, owner_key) in errors:
                    resolved.append({**hit, "owner_key": owner_key, "error": errors[(library, owner_key)]})
                continue
            if _is_trashed(record) or _is_trashed(owner):
                continue
            resolved.append({**hit, "record": record, "owner": owner, "owner_key": owner_key})
        return resolved

    def _resolve_items(
        self, wanted: dict[int | None, set[str]]
    ) -> tuple[dict[tuple[int | None, str], dict], dict[tuple[int | None, str], str]]:
        """Look up ``{library: keys}``: ``(items, errors)``, both keyed ``(library, key)``.

        Trashed items may be returned, flagged ``data.deleted``; missing ones
        are in neither mapping. ``library`` None is a hit whose record names no
        library (indexes from before #396).

        Local mode reads zotero.sqlite first, every library in one place. What
        it cannot serve (no database, or a snapshot older than the item) goes
        to the API in batches of 50: through the bound client for its own
        library and untagged hits, and through a client scoped to the hit's
        library for any other (#492), whose failures name that library.
        """
        items: dict[tuple[int | None, str], dict] = {}
        errors: dict[tuple[int | None, str], str] = {}
        wanted = {library: keys for library, keys in wanted.items() if keys}
        if not wanted:
            return items, errors
        items.update(self._read_local_items(wanted))

        try:
            client_group_id = self._client_group_id()
        except ValueError:
            client_group_id = None
        for library, keys in wanted.items():
            rest = sorted(k for k in keys if (library, k) not in items)
            if not rest:
                continue
            foreign = library is not None and client_group_id is not None and library != client_group_id
            client = self._client_for_group(library) if foreign else self.zotero_client
            if client is None:
                for key in rest:
                    errors[(library, key)] = (
                        f"item belongs to library {library}, which cannot be "
                        "reached from this configuration — no local Zotero "
                        "database, and the web API needs ZOTERO_API_KEY plus, for "
                        "the personal library, a user-scoped ZOTERO_LIBRARY_ID"
                    )
                continue
            found, failed = _fetch_items_via_client(client, rest)
            items.update(((library, key), item) for key, item in found.items())
            for key, error in failed.items():
                errors[(library, key)] = (
                    f"item belongs to library {library}, and the configured "
                    f"credentials could not read it from there: {error}"
                    if foreign else str(error)
                )
        return items, errors

    def _read_local_items(
        self, wanted: dict[int | None, set[str]]
    ) -> dict[tuple[int | None, str], dict]:
        """``wanted`` from zotero.sqlite, trashed items included; {} outside local mode."""
        try:
            reader = self._open_local_reader()
        except Exception as e:
            logger.debug(f"Resolving search hits: no local database ({e})")
            return {}
        if reader is None:
            return {}
        found: dict[tuple[int | None, str], dict] = {}
        try:
            for library, keys in wanted.items():
                ordered = sorted(keys)
                for start in range(0, len(ordered), _LOCAL_KEY_BATCH):
                    batch = ordered[start:start + _LOCAL_KEY_BATCH]
                    # group_id=None reads every library, for untagged hits.
                    for key, item in reader.get_full_items(batch, group_id=library).items():
                        found[(library, key)] = item
        except Exception as e:
            logger.warning(f"Resolving search hits from zotero.sqlite failed: {e}")
        finally:
            reader.close()
        return found

    def _client_for_group(self, group_id: int):
        """A pyzotero client scoped to the library ``group_id`` names, or None.

        The per-library complement of ``_read_local_items`` (#492): the bound
        client can only serve its own library, and outside local mode there
        is no ``zotero.sqlite`` to fall back on, so a foreign hit needs a
        client of its own. 0 is the personal library; anything else is a
        Zotero groupID. Cached per id for this instance's lifetime — a search
        returning ten hits from one group builds one client, and a library
        found unreachable is recorded as None so it fails once, not per hit.

        In local mode the local API serves group libraries too, so the
        resolver also acts as a last resort when ``zotero.sqlite`` did not
        hold a key (e.g. the snapshot predates the item).
        """
        if group_id in self._scoped_clients:
            return self._scoped_clients[group_id]

        local = is_local_mode()
        api_key = os.getenv("ZOTERO_API_KEY")
        if group_id == PERSONAL_LIBRARY_GROUP_ID:
            if not local and os.getenv("ZOTERO_LIBRARY_TYPE", "user") == "group":
                # Group-primary configuration: ZOTERO_LIBRARY_ID names a
                # group, so the personal user ID is unknowable here. A
                # client built from it would ask the wrong library and 404
                # misleadingly; better no client and the explicit message.
                library_id = None
            else:
                library_id = os.getenv("ZOTERO_LIBRARY_ID") or ("0" if local else None)
            library_type = "user"
        else:
            library_id, library_type = str(group_id), "group"

        client = None
        if library_id and (local or api_key):
            try:
                client = _new_scoped_client(library_id, library_type, api_key, local)
            except Exception as e:
                logger.warning(
                    f"Cross-library enrichment: could not build a client for "
                    f"library {group_id}: {e}"
                )
        self._scoped_clients[group_id] = client
        return client

    def _open_local_reader(self) -> LocalZoteroReader | None:
        """A reader over this install's ``zotero.sqlite``, or None outside
        local mode. Resolves the database path the same way the group_id
        backfill does: an explicit ``db_path``, else the one recorded in the
        semantic-search config, else auto-detection."""
        if not is_local_mode():
            return None
        zotero_db_path = self.db_path
        if not zotero_db_path and self.config_path and os.path.exists(self.config_path):
            with open(self.config_path) as f:
                zotero_db_path = json.load(f).get("semantic_search", {}).get("zotero_db_path")
        return LocalZoteroReader(db_path=zotero_db_path)

    def get_database_status(self) -> dict[str, Any]:
        """Get status information about the semantic search database."""
        collection_info = self.chroma_client.get_collection_info()
        batch_active = self._resolve_openai_batch_enabled(None)

        return {
            "collection_info": collection_info,
            "update_config": self.update_config,
            "openai_batch": {
                "enabled": self._load_openai_batch_enabled(),
                "active": batch_active,
            },
            "chunking": {
                "enabled": self._chunking_enabled,
                "max_tokens": getattr(self.chroma_client, "embedding_max_tokens", None),
            },
            "should_update": self.should_update_database(),
            "last_update": self.update_config.get("last_update"),
        }

    def delete_item(self, item_key: str) -> bool:
        """Delete an item from the semantic search database."""
        try:
            self.chroma_client.delete_documents([item_key])
            self.chroma_client.delete_item_chunks(item_key)
            return True
        except Exception as e:
            logger.error(f"Error deleting item {item_key}: {e}")
            return False


def create_semantic_search(
    config_path: str | None = None,
    db_path: str | None = None,
    extraction_workers: int | None = None,
) -> ZoteroSemanticSearch:
    """
    Create a ZoteroSemanticSearch instance.

    Args:
        config_path: Path to configuration file
        db_path: Optional path to Zotero database (overrides config file)
        extraction_workers: Optional parallel-extraction worker count
            (overrides the value in the config file)

    Returns:
        Configured ZoteroSemanticSearch instance
    """
    return ZoteroSemanticSearch(
        config_path=config_path, db_path=db_path, extraction_workers=extraction_workers
    )
