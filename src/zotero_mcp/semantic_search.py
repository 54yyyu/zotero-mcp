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

    _tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    tiktoken = None
    _tokenizer = None


from . import batch_common, chroma_worker, fulltext_cache, gemini_batch, openai_batch
from .chroma_client import (
    ChromaClient,
    create_chroma_client,
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
    auto_update_enabled,
    batch_notice_path,
    load_batch_notice,
    load_reranker_config,
    load_update_config,
    reranker_enabled,
    should_update,
)
from .embeddings.base import EmbeddingAPIUnreachable, RemoteEmbeddingFunction
from .embeddings.registry import batch_capable_providers
from .extract import PAGE_SEPARATOR
from .identifiers import metadata_match_keys
from .local_db import PERSONAL_LIBRARY_GROUP_ID, LocalZoteroReader
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


def _run_replaced_index(manifest: dict[str, Any]) -> bool:
    """Did this run empty the ChromaDB collection and replace its contents?

    A force rebuild that gave a file up writes what arrived over the existing
    index instead, so the sync version that index had still holds. ``reset_at``
    is written just before the reset; a manifest from before it (with no
    ``since_version`` either) counts as a reset unless it gave a file up.
    """
    if not manifest.get("force_full_rebuild"):
        return False
    if manifest.get("reset_at"):
        return True
    if "since_version" in manifest:
        return False
    return not any(batch.get("abandoned") for batch in manifest.get("batches", []))


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


def _is_superseded(current: dict[str, Any], newest: dict[str, Any] | None, adapter: Any = None) -> bool:
    """Has ``newest`` taken over the items ``current`` was submitted for?

    Only when it has may the older run's parked chunks be abandoned and its
    sync watermark withheld. Compared by ``run_id``, never by manifest path: the
    same run reached through a differently spelled config path must not
    supersede itself. See :func:`batch_common.run_covers` for the coverage rule;
    with ``adapter`` given, a dead ``newest`` takes nothing over.
    """
    return bool(newest) and batch_common.run_covers(newest, current, adapter)


def _deferred_message(
    run: dict[str, Any], run_count: int, batch: bool, window_hours: int, fill_days: int | None = None
) -> str:
    """What a user is told when an update has to wait behind an unfinished run.

    ``fill_days`` is given when ``run`` adds missing records and is past the
    provider's window: it holds updates up that many days from its last
    activity, and only the user can move it on.
    """
    others = f" (and {run_count - 1} older one(s))" if run_count > 1 else ""
    what = "submitting a new batch" if batch else "updating the index"
    run_id = run.get("run_id")
    if fill_days is not None:
        return (
            f"Not {what}: batch run {run_id}{others} for this library is still adding records "
            "missing from the index. A new run would not include them, so they would be left out. "
            "The sync version stays where it is, so these changes go in after that run has "
            f"finished, or {fill_days} days after its last activity, when it stops holding up "
            "updates. It is past the provider's window, so nothing more happens to it on its own: "
            "'zotero-mcp batch-import' sends its waiting chunks and imports what is ready, "
            "'zotero-mcp update-db --auto-loop' waits for it to finish and then updates, and "
            f"'zotero-mcp batch-abandon --run-id {run_id}' gives it up."
        )
    return (
        f"Not {what}: batch run {run_id}{others} for this library is still unfinished, and going "
        "ahead would pay for its items again or leave it stranded. The sync version stays where "
        "it is, so these changes go in after that run has finished, or once it is past the "
        f"provider's window, {window_hours} hours after its last chunk went out, when it stops "
        "holding up updates. Check it with "
        "'zotero-mcp batch-status'; 'zotero-mcp update-db' or 'zotero-mcp batch-import' collects "
        "it, 'zotero-mcp update-db --auto-loop' waits for it to finish and then updates, and "
        f"'zotero-mcp batch-abandon --run-id {run_id}' gives it up."
    )


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


def _truncate_to_tokens(text: str, max_tokens: int = 8000) -> str:
    """Truncate text to fit within embedding model token limit.

    Uses tiktoken for accurate token counting when available,
    falls back to conservative character-based estimation.
    """
    if _tokenizer is not None:
        tokens = _tokenizer.encode(text, disallowed_special=())
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = _tokenizer.decode(tokens)
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


class _HeldText:
    """Items' full text kept on disk between the article pass and the passage pass.

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
        data = item.get("data") or {}
        text = data.get("fulltext") or ""
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
        data["fulltext"] = ""

    def restore(self, item: dict[str, Any]) -> None:
        path = item.pop(self._KEY, None)
        if path:
            item.setdefault("data", {})["fulltext"] = Path(path).read_text(encoding="utf-8")
            Path(path).unlink(missing_ok=True)

    @staticmethod
    def release(item: dict[str, Any]) -> None:
        """Drop the text again once the item's passages are built."""
        data = item.get("data") or {}
        if data.get("fulltext"):
            data["fulltext"] = ""

    def close(self) -> None:
        self._closed = True
        if self._dir is not None:
            self._dir.cleanup()
            self._dir = None


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
    # Items this run writes that were kept for a retry or given up on, set per
    # run by update_database, and items whose records could not be built.
    _retry_keys_in_run: set[str] = frozenset()
    _unprepared_keys: set[str] = frozenset()
    # Runs in a row an item's records may fail to embed before it is given up
    # on until it next changes in Zotero.
    RETRY_ITEM_MAX_ATTEMPTS = 3
    # How long an unfinished batch run stops gap fills; a run left this long has
    # been forgotten, and its results may be gone.
    BATCH_RUN_BLOCKS_FILL_DAYS = 14
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

    def _load_last_sync_version(self, library_key: str | None = None) -> int:
        """Last Zotero library version fully indexed into ChromaDB for
        ``library_key``, by default the library the Zotero client is currently
        scoped to.

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

        library_key = library_key if library_key is not None else self._active_library_key()
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
                item_key = meta.get("item_key") or doc_id.split("#", 1)[0]
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
        """
        Create metadata for a Zotero item.

        Args:
            item: Zotero item dictionary

        Returns:
            Metadata dictionary for ChromaDB
        """
        data = item.get("data", {})

        metadata = {
            "item_key": item.get("key", ""),
            "item_type": data.get("itemType", ""),
            "title": data.get("title", ""),
            "date": data.get("date", ""),
            "date_added": data.get("dateAdded", ""),
            "date_modified": data.get("dateModified", ""),
            "creators": format_creators(data.get("creators", [])),
            "publication": data.get("publicationTitle", ""),
            "url": data.get("url", ""),
            "doi": data.get("DOI", ""),
        }
        # Library attribution (#163): 0 = personal, else groupID. Every
        # item-producing path (local scan, API scan, incremental API fetch)
        # stamps data["group_id"] before this runs. When attribution is
        # unknown (key missing from the local map), the key is OMITTED, not
        # defaulted: positive attribution is deletion authority under the
        # group_id-scoped deletion pass, so unknown must stay unknown.
        if (group_id := data.get("group_id")) is not None:
            metadata["group_id"] = int(group_id)
        # If fulltext was extracted (or attempted), mark it so incremental
        # updates don't keep re-trying items that failed extraction
        if data.get("fulltext"):
            metadata["has_fulltext"] = True
            if data.get("fulltextSource"):
                metadata["fulltext_source"] = data.get("fulltextSource")
        elif data.get("fulltext_attempted"):
            # Extraction was attempted but failed (timeout, empty, etc.)
            # Mark so we don't retry on every incremental update
            metadata["has_fulltext"] = "failed"

        # Record the attachment-key set (local mode only) so update runs can
        # retry a "failed" item once its attachments change — attaching a file
        # does not bump the parent's dateModified.
        if (att_keys := data.get("attachmentKeys")) is not None:
            metadata["attachment_keys"] = att_keys

        # Which attachment kind won when this document was extracted. A later
        # run compares it to the current setting: change the priority and the
        # stored text may now come from the wrong file (#378).
        if (att_priority := data.get("attachmentPriority")) is not None:
            metadata["attachment_priority"] = att_priority

        # Add tags as a single string
        if tags := data.get("tags"):
            metadata["tags"] = " ".join([tag.get("tag", "") for tag in tags])
        else:
            metadata["tags"] = ""

        # Add citation key if available
        extra = data.get("extra", "")
        citation_key = ""
        for line in extra.split("\n"):
            if line.lower().startswith(("citation key:", "citationkey:")):
                citation_key = line.split(":", 1)[1].strip()
                break
        metadata["citation_key"] = citation_key

        return metadata

    def auto_update_enabled(self) -> bool:
        """Did the user ask for automatic updates, due or not?"""
        return auto_update_enabled(self.update_config)

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
                # Stamped on every document so a later run can tell that the
                # chosen attachment kind may have changed under it.
                priority_tag = ",".join(reader.attachment_priority)
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

                # Phase 2: selectively extract fulltext only when requested
                if extract_fulltext:
                    extracted = 0
                    skipped_existing = 0
                    updated_existing = 0
                    items_to_process = []
                    # Items whose attachment still needs parsing. Collected
                    # here rather than parsed inline so the expensive half can
                    # run as one batch — see the phase 2b loop below.
                    pending_extraction = []

                    total_local = len(local_items)
                    _skipped_failed = []  # Items skipped because extraction previously failed
                    # Items skipped that never had anything to extract — kept
                    # out of the "extraction previously failed" report, which
                    # alarmed users into force-rebuilds that cannot help (#446).
                    _skipped_no_source = 0

                    # Temporarily suppress the extractor's logger: a warning
                    # about one unreadable attachment would otherwise land in
                    # the middle of the \r progress line.
                    _extract_logger = logging.getLogger("zotero_mcp.extract")
                    _prev_level = _extract_logger.level
                    _extract_logger.setLevel(logging.CRITICAL)

                    for item_idx, it in enumerate(local_items, 1):
                        # Build display string: Author (Year) — Title
                        title = getattr(it, "title", "") or ""
                        creators = getattr(it, "creators", "") or ""
                        date = getattr(it, "date_added", "") or ""
                        first_author = ""
                        if creators:
                            first_author = creators.split(";")[0].split(",")[0].strip()
                            if first_author:
                                first_author += " et al." if ";" in creators else ""
                        year = ""
                        if date and len(date) >= 4:
                            year = date[:4]
                        citation = ""
                        if first_author and year:
                            citation = f"{first_author} ({year}) — "
                        elif first_author:
                            citation = f"{first_author} — "
                        display = f"{citation}{title}"
                        if len(display) > 60:
                            display = display[:57] + "..."

                        # Single-line progress with \r overwrite
                        # MUST fit within terminal width to prevent wrapping
                        try:
                            try:
                                term_width = os.get_terminal_size().columns
                            except (OSError, ValueError):
                                term_width = 80
                            # Build the line and truncate to terminal width - 1
                            # (- 1 to prevent the cursor from wrapping to next line)
                            max_len = term_width - 1
                            status_parts = []
                            if skipped_existing > 0:
                                status_parts.append(f"{skipped_existing} up to date")
                            if extracted > 0:
                                status_parts.append(f"{extracted} extracted")
                            status = f" ({', '.join(status_parts)})" if status_parts else ""
                            prefix = f"  Processing {item_idx}/{total_local}{status} — "
                            # Truncate display to fit remaining space
                            remaining = max_len - len(prefix) - 3  # -3 for "..."
                            if remaining > 0 and display and len(display) > remaining:
                                display = display[:remaining] + "..."
                            line = f"{prefix}{display or 'working...'}"
                            # Clip by display columns so a CJK title can't
                            # wrap the terminal and garble later \r redraws.
                            line = _clip_to_width(line, max_len)
                            sys.stderr.write(f"\r{line}{' ' * max(0, max_len - _display_cols(line))}")
                            sys.stderr.flush()
                        except Exception:
                            pass

                        should_extract = True

                        # Current attachment-key set, stored in metadata so a
                        # later run can detect attachment changes. Attaching a
                        # file does NOT bump the parent's dateModified, so the
                        # date check alone never clears a "failed" marker.
                        att_keys = ",".join(
                            sorted(k for k, _p, _c in reader.get_fulltext_meta_for_item(it.item_id))
                        )
                        it._attachment_keys = att_keys
                        it._attachment_priority = priority_tag

                        # CHECK IF ITEM ALREADY EXISTS (unless force_rebuild or no client)
                        if chroma_client and not force_rebuild:
                            # With passage-chunking the stored ids are
                            # "<key>#<n>"; get_document_metadata falls back to
                            # chunk 0 so chunked items are still recognized.
                            existing_metadata = chroma_client.get_document_metadata(it.key)
                            if existing_metadata and "group_id" not in existing_metadata:
                                # Indexed before multi-library attribution and
                                # not (yet) covered by the backfill: re-upsert
                                # so the doc gains its group_id — otherwise an
                                # unchanged untagged doc is skipped as "up to
                                # date" forever and stays excluded from
                                # library-filtered search and cleanup.
                                updated_existing += 1
                            elif existing_metadata:
                                chroma_has_fulltext = existing_metadata.get("has_fulltext", False)
                                local_has_fulltext = bool(att_keys)

                                # Skip if extraction previously failed AND neither the item
                                # nor its attachment set has changed since (handles both a
                                # replaced bad PDF and a PDF newly attached to an item that
                                # was indexed metadata-only)
                                if chroma_has_fulltext == "failed":
                                    chroma_date = existing_metadata.get("date_modified", "")
                                    item_date = getattr(it, "date_modified", "") or ""
                                    stored_att_keys = existing_metadata.get("attachment_keys")
                                    if chroma_date == item_date and stored_att_keys == att_keys:
                                        # Nothing changed since the failure — don't retry
                                        should_extract = False
                                        skipped_existing += 1
                                        if att_keys:
                                            _skipped_failed.append(display or f"item {it.key}")
                                        else:
                                            # Legacy "failed" marker on an item
                                            # with no text-bearing attachments:
                                            # nothing ever failed, there was
                                            # nothing to try (#446).
                                            _skipped_no_source += 1
                                    else:
                                        # Item or its attachments changed since last
                                        # failure (legacy records without attachment_keys
                                        # retry once, then converge) — retry
                                        updated_existing += 1
                                elif chroma_has_fulltext and not local_has_fulltext:
                                    # Indexed with text, but every text-bearing
                                    # attachment is gone. Re-index metadata-only
                                    # so the passages from the deleted file stop
                                    # matching searches (#428).
                                    updated_existing += 1
                                elif (
                                    chroma_has_fulltext
                                    and existing_metadata.get("attachment_keys") is not None
                                    and existing_metadata.get("attachment_keys") != att_keys
                                ):
                                    # A different attachment set backs the text
                                    # now, e.g. a replaced PDF (#428). Records
                                    # written before attachment_keys existed are
                                    # not compared, or every such index would
                                    # re-extract in full on upgrade.
                                    updated_existing += 1
                                elif not chroma_has_fulltext and local_has_fulltext:
                                    # Document exists but lacks fulltext - we need to update it
                                    updated_existing += 1
                                elif _attachment_priority_changed(
                                    existing_metadata, priority_tag
                                ):
                                    # The stored text may have come from an
                                    # attachment the user has since deprioritized
                                    # — re-extract rather than serve a stale
                                    # PDF-derived embedding (#378).
                                    updated_existing += 1
                                else:
                                    should_extract = False
                                    skipped_existing += 1

                        if not should_extract and it.key in self._scan_gaps:
                            action = self._scan_gap_action(
                                existing_metadata or {}, self._scan_gaps[it.key]
                            )
                            it._index_levels = self._scan_gaps[it.key]
                            skipped_existing -= 1
                            if action == "extract":
                                should_extract = True
                            else:
                                # Extraction already failed and nothing changed:
                                # index what metadata there is, as the first run
                                # did, instead of parsing that file again.
                                it._fulltext_attempted = True
                                extracted += 1
                                items_to_process.append(it)

                        if should_extract:
                            # Defer the parse itself — see phase 2b.
                            if not getattr(it, "fulltext", None):
                                pending_extraction.append(it)
                            extracted += 1
                            items_to_process.append(it)

                            # (progress shown inline above via \r)

                    # Phase 2b: parse the chosen attachments. With
                    # extraction_workers == 1 this is the same sequential walk
                    # as before; above 1 it fans out over a process pool, and
                    # results arrive out of order — hence the lookup by id.
                    if pending_extraction:
                        by_id = {it.item_id: it for it in pending_extraction}
                        done = 0
                        for item_id, result in _extract_fulltext_batch(
                            reader, [(it.item_id, it.key) for it in pending_extraction]
                        ):
                            target = by_id.get(item_id)
                            if target is None:
                                continue
                            if result:
                                target.fulltext, target.fulltext_source = result
                            elif getattr(target, "_attachment_keys", ""):
                                # An attachment existed but produced no text —
                                # record the attempt so incremental runs don't
                                # re-parse it until something changes.
                                target._fulltext_attempted = True
                            # Items with no text-bearing attachments get no
                            # marker at all (#446): there was nothing to try,
                            # "failed" would mislead the skip report, and the
                            # has_fulltext-absent path already re-indexes them
                            # the moment a first attachment appears.
                            done += 1
                            if done % 10 == 0 or done == len(pending_extraction):
                                try:
                                    line = f"  Extracting text: {done}/{len(pending_extraction)}"
                                    sys.stderr.write(f"\r{line}{' ' * 40}")
                                    sys.stderr.flush()
                                except Exception:
                                    pass

                    _extract_logger.setLevel(_prev_level)

                    # Clear progress line and show extraction summary
                    try:
                        sys.stderr.write(f"\r{' ' * _term_width(120)}\r")  # Clear progress line
                        parts = [f"  Extraction complete: {extracted} items to index"]
                        if skipped_existing > 0:
                            parts.append(f"{skipped_existing} already up to date")
                        sys.stderr.write(", ".join(parts) + "\n")
                        if updated_existing > 0:
                            sys.stderr.write(f"  ({updated_existing} items updated with new fulltext)\n")
                        # Honest accounting for the run's own extraction pass
                        # (#446): "N indexed" used to cover items whose
                        # attachments produced no text, so a --force-rebuild
                        # reported "0 errors" and only the NEXT run's skip
                        # report revealed how many extractions failed.
                        _attempt_failed = sum(
                            1 for _it in pending_extraction
                            if getattr(_it, "_fulltext_attempted", False)
                        )
                        _no_source = sum(
                            1 for _it in pending_extraction
                            if not getattr(_it, "fulltext", None)
                            and not getattr(_it, "_fulltext_attempted", False)
                        )
                        if _attempt_failed:
                            sys.stderr.write(
                                f"  {_attempt_failed} item(s) had attachments that produced no text "
                                "(indexed metadata-only; not retried until the item or its attachments change)\n"
                            )
                        if _no_source:
                            sys.stderr.write(
                                f"  {_no_source} item(s) have no text-bearing attachments (indexed metadata-only)\n"
                            )
                        if _skipped_no_source:
                            sys.stderr.write(
                                f"  {_skipped_no_source} item(s) remain metadata-only (no text-bearing attachments)\n"
                            )
                        if _skipped_failed:
                            sys.stderr.write(
                                f"  {len(_skipped_failed)} item(s) skipped (PDF extraction previously failed):\n"
                            )
                            for name in _skipped_failed[:5]:  # Show first 5
                                sys.stderr.write(f"    - {name}\n")
                            if len(_skipped_failed) > 5:
                                sys.stderr.write(f"    ... and {len(_skipped_failed) - 5} more\n")
                            sys.stderr.write(
                                "  (To retry these, attach or replace the PDF, or run with --force-rebuild)\n"
                            )
                    except Exception:
                        pass

                    # Replace local_items with filtered list
                    local_items = items_to_process
                else:
                    # Skip fulltext extraction for faster processing
                    for it in local_items:
                        it.fulltext = None
                        it.fulltext_source = None

                # Convert to API-compatible format
                api_items = []
                for item in local_items:
                    # Create API-compatible item structure
                    api_item = {
                        "key": item.key,
                        "version": 0,  # Local items don't have versions
                        "data": {
                            "key": item.key,
                            "itemType": getattr(item, "item_type", None) or "journalArticle",
                            "title": item.title or "",
                            "abstractNote": item.abstract or "",
                            "extra": item.extra or "",
                            # Include fulltext only when extracted
                            "fulltext": getattr(item, "fulltext", None) or "" if extract_fulltext else "",
                            "fulltextSource": getattr(item, "fulltext_source", None) or "" if extract_fulltext else "",
                            # Flag if extraction was attempted but failed (timeout, empty)
                            "fulltext_attempted": getattr(item, "_fulltext_attempted", False),
                            "dateAdded": item.date_added,
                            "dateModified": item.date_modified,
                            "creators": self._parse_creators_string(item.creators) if item.creators else [],
                            # Library attribution (#163): 0 = personal, else
                            # groupID. Ground truth from get_key_group_map();
                            # an item missing from the map (e.g. added
                            # mid-scan) stays UNATTRIBUTED — a guessed
                            # "personal" would be positive attribution minted
                            # from nothing, i.e. deletion authority. The next
                            # incremental sync re-tags it with evidence.
                            "group_id": key_group_map.get(item.key),
                        },
                    }
                    # Attachment-key set (computed during the extraction scan);
                    # persisted to metadata so incremental runs can detect
                    # newly attached files on previously-failed items.
                    if (att := getattr(item, "_attachment_keys", None)) is not None:
                        api_item["data"]["attachmentKeys"] = att
                    if (prio := getattr(item, "_attachment_priority", None)) is not None:
                        api_item["data"]["attachmentPriority"] = prio
                    if (levels := getattr(item, "_index_levels", None)) is not None:
                        api_item["_index_levels"] = levels

                    # Add notes if available
                    if item.notes:
                        api_item["data"]["notes"] = item.notes

                    api_items.append(api_item)

                logger.info(f"Retrieved {len(api_items)} items from local database")
                return api_items

        except Exception as e:
            logger.error(f"Error reading from local database: {e}")
            logger.info("Falling back to API...")
            return self._get_items_from_api(limit)

    @staticmethod
    def _scan_gap_action(existing_metadata: dict[str, Any], gap: set[str] | None) -> str | None:
        """What a scan does with an up-to-date item that lacks a level, or None."""
        if not gap:
            return None
        return "metadata_only" if existing_metadata.get("has_fulltext") == "failed" else "extract"

    def _find_index_gaps(self, *, limit: int | None, force_rebuild: bool) -> dict[str, set[str]]:
        """Items that lack a level this index should have: ``{key: {levels}}``.

        Enabling chunking leaves every item without passages, and a migration
        leaves every item without an article record. Neither shows up as a
        library change, so the sync watermark cannot find them. Nor does an
        item kept for a retry (``retry_item_keys``), on any index.
        """
        if limit is not None or force_rebuild:
            return {}
        retry_keys = list(self.update_config.get("retry_item_keys") or [])
        if not self._separate_passages():
            # One level, chosen by the setting; only a kept item is due.
            return {key: {SEARCH_ARTICLES} for key in retry_keys}
        articles, passages = self.chroma_client.item_keys_by_level()
        gaps: dict[str, set[str]] = {key: {SEARCH_ARTICLES} for key in passages - articles}
        if self._chunking_enabled:
            for key in articles - passages:
                gaps[key] = {SEARCH_PASSAGES}
        # Items whose records failed to embed. They get every level again,
        # whatever they have: a changed item whose write failed still has its
        # old records. The key goes once the write succeeds (_settle_retry_keys).
        levels = {SEARCH_ARTICLES, SEARCH_PASSAGES} if self._chunking_enabled else {SEARCH_ARTICLES}
        for key in retry_keys:
            gaps[key] = set(levels)
        # An item given up on waits for its next change in Zotero, or the gap
        # fill would pay for the records that keep failing on every run.
        for key in self.update_config.get("retry_item_keys_given_up") or []:
            gaps.pop(key, None)
        return gaps

    def _settle_retry_keys(self, *, written: set[str], failed: set[str], keep_failed: bool) -> list[str]:
        """Update the items kept for a retry once a write has finished.

        ``written`` items got all their records: they are no longer kept, nor
        given up on. A ``failed`` item counts one more failed run; with
        ``keep_failed`` it is kept even if it was not before (a split index, or
        a rebuild), otherwise only an item already kept is counted. After
        ``RETRY_ITEM_MAX_ATTEMPTS`` failed runs it is given up on until it next
        changes in Zotero, so a record that can never be embedded is not paid
        for on every run. Returns the items given up on now.
        """
        config = self.update_config
        kept = set(config.get("retry_item_keys") or [])
        attempts = {key: int(n) for key, n in (config.get("retry_item_attempts") or {}).items()}
        given_up = list(config.get("retry_item_keys_given_up") or [])
        newly_given_up: list[str] = []
        for key in sorted(failed):
            if key not in kept and not keep_failed:
                continue
            # A key kept before attempts were counted has failed once already.
            count = attempts.get(key, 1 if key in kept else 0) + 1
            if count >= self.RETRY_ITEM_MAX_ATTEMPTS:
                kept.discard(key)
                attempts.pop(key, None)
                if key not in given_up:
                    given_up.append(key)
                newly_given_up.append(key)
            else:
                kept.add(key)
                attempts[key] = count
        for key in written - failed:
            kept.discard(key)
            attempts.pop(key, None)
        given_up = [key for key in given_up if key not in kept and key not in written - failed]
        if kept or "retry_item_keys" in config:
            config["retry_item_keys"] = sorted(kept)
        if attempts or "retry_item_attempts" in config:
            config["retry_item_attempts"] = {key: attempts[key] for key in sorted(attempts) if key in kept}
        if given_up or "retry_item_keys_given_up" in config:
            config["retry_item_keys_given_up"] = given_up
        return newly_given_up

    def _failed_records_note(self, manifest: dict[str, Any], failed_ids: list[str], covered: bool) -> str:
        """What an import says about a landed run's records that have no embedding."""
        count = len(failed_ids)
        keys = {doc_id.split("#", 1)[0] for doc_id in failed_ids}
        kept = set(self.update_config.get("retry_item_keys") or [])
        if covered and keys & kept:
            return (
                f"{count} record(s) have no embedding; the next update-db rewrites the items kept "
                "for a retry, and a later run covers the rest."
            )
        if covered:
            return f"{count} record(s) have no embedding; a later run covers their items."
        if manifest.get("reset_at") or self._separate_passages() or keys <= kept:
            return f"{count} record(s) still have no embedding after retries; the next update-db embeds them again."
        return (
            f"{count} record(s) still have no embedding after retries. The sync version stays put, so the "
            "next update submits every change since that version again, including chunks already imported."
        )

    @classmethod
    def _given_up_message(cls, keys: list[str]) -> str:
        """What the user is told about items given up on."""
        shown = ", ".join(keys[:10]) + (f" and {len(keys) - 10} more" if len(keys) > 10 else "")
        return (
            f"Gave up on {len(keys)} item(s) whose records failed to embed in "
            f"{cls.RETRY_ITEM_MAX_ATTEMPTS} runs: {shown}. They are not tried again until they "
            "change in Zotero; the log shows the embedding error."
        )

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
                if existing.get("has_fulltext") and existing.get("has_fulltext") != "failed":
                    waiting.add(key)
                    continue
            try:
                item = self.zotero_client.item(key)
            except Exception as e:
                logger.debug(f"item({key}) failed while filling index gaps: {e}")
                self._note_unanswered_recheck(key, waiting)
                continue
            if not item or item.get("data", {}).get("itemType") in _NON_INDEXED_ITEM_TYPES:
                self._note_unanswered_recheck(key, waiting)
                continue
            item["_index_levels"] = gaps[key]
            items.append(item)
        if include_fulltext and items:
            self._attach_web_fulltext(items)
        self._tag_group_id(items)
        self.update_config["gaps_waiting_for_text"] = sorted(waiting)
        return self._without_fills_lacking_text(items)

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
            data = item.get("data", {})
            key = item.get("key")
            if item.get("_index_levels") and not (data.get("fulltext") or "").strip():
                existing = self.chroma_client.get_document_metadata(key) or {}
                if existing.get("has_fulltext") and existing.get("has_fulltext") != "failed":
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
            if key in waiting and (data.get("fulltext") or "").strip():
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
            if not existing or "parent_item_key" in existing:
                continue
            if not _same_moment(existing.get("date_modified"), data.get("dateModified")):
                continue
            # Written from the same text: with full text both times, or without.
            if bool((data.get("fulltext") or "").strip()) != (existing.get("has_fulltext") is True):
                continue
            item["_index_levels"] = gaps[key]

    def _unfinished_batch_run(self) -> str | None:
        """Run id of a recent batch run for this library that has not finished landing.

        Its results include records a gap check would see as missing, and
        paying for them again would also supersede the run.
        """
        manifest = self._unfinished_batch_manifest()
        return str(manifest.get("run_id")) if manifest else None

    def _unfinished_batch_manifest(self) -> dict[str, Any] | None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.BATCH_RUN_BLOCKS_FILL_DAYS)
        group_id = self._pinned_group_id()
        provider = self._collect_provider()
        if provider:
            # The runs collect and batch-import carry forward: not one a newer
            # run or sync took over, or another model made, which nothing will
            # ever import. Newest first.
            candidates = list(reversed(self._unfinished_batch_runs(provider, group_id)))
        else:
            candidates = [
                run
                for name, module in _BATCH_MODULES.items()
                for run in module.manifests_for_group(config_path=self.config_path, group_id=group_id)
                if batch_common.run_is_unfinished(run, _batch_adapter(name))
            ]
        for run in candidates:
            # From the last thing it did: a throttled run sends its chunks
            # over days, and an import or a resend shows it is still moving.
            active = batch_common.last_activity(run)
            if active is None or active < cutoff:
                continue
            return run
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
            [self._pdf_attachment_key(child) for child in children]
        )

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
            key = self._pdf_attachment_key(row)
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
        """``(article record, passage records)`` an item should get in this run.

        An item in the index's gap list needs only what it lacks; anything else
        gets every level the settings call for. An unmigrated index has one
        level, chosen by the setting, as it always did.
        """
        chunking = self._chunking_enabled
        if not self._separate_passages():
            return not chunking, chunking
        levels = item.get("_index_levels")
        if levels:
            return SEARCH_ARTICLES in levels, SEARCH_PASSAGES in levels
        return True, chunking

    def _build_item_records(
        self, item: dict[str, Any], *, chunking: bool, article: bool | None = None
    ) -> list[dict[str, Any]]:
        """Build the index records of one item, without embedding or writing them.

        The one place an item becomes ``{"id", "document", "metadata"}`` records,
        for the realtime and the Batch API paths alike. With ``chunking``, one
        record per overlapping passage (``<item_key>#<n>``), so search can return
        a grounded quote and long PDFs stay searchable past the single-vector
        truncation limit. With ``article``, which defaults to the opposite of
        ``chunking``, one record under the item key, first.

        Returns ``[]`` for an item with no key or no text. A failure raises with
        nothing returned, so an item contributes all of its records or none.
        """
        if article is None:
            article = not chunking
        item_key = item.get("key", "")
        if not item_key:
            return []

        # Always include structured fields; append fulltext when available
        fulltext = item.get("data", {}).get("fulltext", "")
        structured_text = self._create_document_text(item)
        if fulltext.strip():
            doc_text = (structured_text + "\n\n" + fulltext) if structured_text.strip() else fulltext
        else:
            doc_text = structured_text
        metadata = self._create_metadata(item)

        if not doc_text.strip():
            return []

        records: list[dict[str, Any]] = []
        if article:
            # Truncate to fit the configured embedding model's token limit
            records.append({
                "id": item_key,
                "document": self.chroma_client.truncate_text(doc_text),
                "metadata": metadata,
            })
        if not chunking:
            return records

        passages = split_into_passages(
            doc_text,
            self._effective_chunk_size(),
            int(self._chunking_config.get("overlap", 200)),
            int(self._chunking_config.get("max_chunks_per_item", 20)),
        )
        section_marks = _section_offsets(doc_text) if passages else []
        for ci, (chunk_text, c0, c1) in enumerate(passages):
            cmeta = dict(metadata)
            cmeta["parent_item_key"] = item_key
            cmeta["chunk_index"] = ci
            cmeta["n_chunks"] = len(passages)
            cmeta["char_start"] = c0
            cmeta["char_end"] = c1
            page = _page_for_offset(doc_text, c0)
            if page is not None:
                cmeta["page"] = page
            section = _section_at(section_marks, c0)
            if section is not None:
                cmeta["section"] = section
            records.append({
                "id": f"{item_key}#{ci}",
                "document": self.chroma_client.truncate_text(chunk_text),
                "metadata": cmeta,
            })
        return records

    def _prepare_index_records(self, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Prepare ChromaDB records without embedding or writing them."""
        stats = {"processed": 0, "skipped": 0, "errors": 0}
        records: list[dict[str, Any]] = []

        for item in items:
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
                records.extend(item_records)
                stats["processed"] += 1

            except Exception as e:
                logger.error(f"Error processing item {item.get('key', 'unknown')}: {e}")
                stats["errors"] += 1
                if isinstance(self._unprepared_keys, set) and item.get("key"):
                    self._unprepared_keys.add(item["key"])

        return records, stats

    def _existing_item_keys(self, item_keys: list[str]) -> set[str]:
        """Which of these items have a record at either level."""
        found = self.chroma_client.get_existing_ids(
            [*item_keys, *(f"{key}#0" for key in item_keys)]
        )
        return {doc_id.split("#", 1)[0] for doc_id in found}

    def _submit_batch_index(
        self,
        provider: str,
        items: list[dict[str, Any]],
        force_full_rebuild: bool,
        target_sync_version: int | None,
        stats: dict[str, Any],
        max_enqueued_tokens: int | None = None,
        max_requests: int | None = None,
        scope: dict[str, Any] | None = None,
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

        item_keys = sorted({record["id"].split("#", 1)[0] for record in records})
        existing_items = (
            self._existing_item_keys(item_keys) if not force_full_rebuild else set()
        )
        model_name = self.chroma_client.embedding_config.get("model_name", adapter.default_model)
        submit_kwargs: dict[str, Any] = {}
        if max_enqueued_tokens is not None:
            submit_kwargs["max_enqueued_tokens"] = max_enqueued_tokens
        if max_requests is not None:
            submit_kwargs["max_requests"] = max_requests
        if scope is not None:
            submit_kwargs["scope"] = scope
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
        # Items kept for a retry, or given up on, that this run writes: the
        # import drops them once the run lands (_promote_run).
        # A kept item with no text to embed has nothing left to write, so it
        # counts as written with the rest, not just the items in a file.
        retry_keys = sorted(set(self._retry_keys_in_run) - self._unprepared_keys)
        if retry_keys:
            manifest["retry_keys"] = retry_keys
            module.save_manifest(manifest)
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
        nothing left to reconcile (#457). ``current_library_keys`` of None
        means the key set could not be fetched; that records a skip reason,
        and a skipped pass keeps the watermark where it was.
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
            stored_item_keys = {i.split("#", 1)[0] for i in stored_ids}
            # Which kind of delete a key needs comes from what is stored, not
            # from the chunking setting, which may have changed since.
            keys_with_passages = {i.split("#", 1)[0] for i in stored_ids if "#" in i}
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
                    # A deleted item can't change in Zotero again, so it must
                    # not stay kept for a retry or listed as given up on.
                    self._settle_retry_keys(written=set(to_delete_keys), failed=set(), keep_failed=False)
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
        allow_rebuild_import: bool = True,
    ) -> dict[str, Any]:
        """
        Update the semantic search database with Zotero items.

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
                An unfinished run that would defer this update is driven to
                the end first, and the update then goes ahead.
            batch_poll_interval: Seconds between auto-loop polls.
            allow_mass_deletion: One-run opt-in for a deletion pass that
                would remove a large share of the library's indexed docs
                (or all of them, when item_versions() reports the library
                empty). Deliberately a parameter, not an env var: nothing
                persistent can disable the guard, and the server's
                unattended background sync can never mass-delete.
            allow_rebuild_import: False keeps the collect at the start of this
                run from replacing the index with a finished rebuild. The
                server's background thread passes it: a reset there would
                leave searches on a part-filled index, and the thread dies with
                the process.

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

        # Set once a realtime rebuild has emptied the index, so a stop after
        # that point can say the old index is gone rather than "kept".
        rebuild_reset = False
        try:
            # Pin this run's library identity once, from the client the run
            # will read items/versions from. Everything scope-sensitive in
            # the run (tagging, watermark key, backfill attribution, the
            # deletion pass) uses this snapshot, so a concurrent
            # zotero_switch_library cannot re-point half a run at another
            # library.
            self._run_group_id = self._client_group_id()

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

            # Results already paid for come first, whatever this run goes on to
            # do. A rebuild re-embeds everything itself, so waiting chunks stay
            # unsent, which would pay twice for the same items, and an older
            # rebuild isn't imported, which would wipe the index the user is
            # replacing because it has gone stale.
            self._collect_unfinished_batches(
                stats,
                submit_waiting=not force_full_rebuild,
                skip_rebuilds=force_full_rebuild,
                allow_rebuild_import=allow_rebuild_import,
            )

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

            # A silent loss without chunking: --fulltext extracts whole PDFs but
            # a small-context model embeds only the first few hundred tokens.
            fit_warning = self._fulltext_fit_warning() if extract_fulltext else None
            if fit_warning:
                stats["fulltext_truncated"] = True
                logger.warning(fit_warning)
                _report(f"\nWarning: {fit_warning}\n")

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

            def _this_run() -> dict[str, Any]:
                """What batch_common.run_covers reads about this run."""
                if use_incremental:
                    since_version = last_sync_version
                elif (
                    last_sync_version == 0
                    and not extract_fulltext
                    and not configured_collection_keys
                    and limit is None
                ):
                    since_version = 0  # a first build reads the whole library
                else:
                    since_version = None
                return {
                    "limited": limit is not None,
                    "since_version": since_version,
                    "allow_mass_deletion": bool(allow_mass_deletion),
                }

            # Before reading a single item: a deferred run does no work, and
            # leaves the deletion pass and the sync version to the run that
            # eventually goes ahead. Realtime updates wait too, or they would
            # buy the in-flight items again and then drop the batch; only an
            # explicit --no-batch goes ahead, told what it costs. Runs dead or
            # past their window hold up nothing (_blocking_batch_runs).
            collect_provider = self._collect_provider()
            if collect_provider:
                new_run = {
                    **_this_run(),
                    "group_id": self._run_group_id,
                    "force_full_rebuild": force_full_rebuild,
                }
                in_flight = self._blocking_batch_runs(collect_provider, self._run_group_id)
                explicit_realtime = not batch_enabled and False in (use_batch, use_openai_batch, use_gemini_batch)
                blocking = [run for run in in_flight if not batch_common.may_supersede(new_run, run)]
                if blocking and auto_loop and not explicit_realtime:
                    # The user asked to wait for batch runs: carry the ones in
                    # the way to the end, as 'batch-import' would, then go on.
                    blocking = self._wait_for_blocking_runs(
                        collect_provider, new_run, batch_poll_interval, stats
                    )
                    if not blocking:
                        # Those runs have landed and moved the sync version
                        # this update starts from.
                        last_sync_version = self._load_last_sync_version() if not force_full_rebuild else 0
                        use_incremental = (
                            not force_full_rebuild
                            and not extract_fulltext
                            and limit is None
                            and last_sync_version > 0
                            and not configured_collection_keys
                        )
                if in_flight and explicit_realtime:
                    stats["batch_discard_warning"] = (
                        f"Batch run {in_flight[-1].get('run_id')} is still unfinished. This realtime "
                        "update embeds its items again, and its results will be discarded."
                    )
                    logger.warning(stats["batch_discard_warning"])
                    _report(f"\nWarning: {stats['batch_discard_warning']}\n")
                elif blocking:
                    stats["batch_provider"] = collect_provider
                    stats["batch_mode"] = batch_enabled
                    stats["batch_deferred"] = self._deferral(collect_provider, blocking, batch_enabled)
                    self._save_batch_notice(deferred=stats["batch_deferred"])
                    logger.warning(stats["batch_deferred"])
                    _report(f"\n{stats['batch_deferred']}\n")
                    end_time = datetime.now()
                    stats["duration"] = str(end_time - start_time)
                    stats["end_time"] = end_time.isoformat()
                    return stats
                elif force_full_rebuild:
                    # Named by the user, so it goes ahead, but a run in flight
                    # or ready is paid for and is about to be thrown away.
                    replaced = [
                        run for run in self._unfinished_batch_runs(collect_provider, self._run_group_id)
                        if batch_common.may_supersede(new_run, run)
                    ]
                    if replaced:
                        more = f" and {len(replaced) - 1} more" if len(replaced) > 1 else ""
                        price = "" if batch_enabled else ", at realtime prices"
                        stats["batch_discard_warning"] = (
                            f"This rebuild replaces batch run {replaced[-1].get('run_id')}{more}, which is "
                            "unfinished or ready to import: its results will be discarded, and every item "
                            f"is embedded again{price}."
                        )
                        logger.warning(stats["batch_discard_warning"])
                        _report(f"\nWarning: {stats['batch_discard_warning']}\n")
                self._save_batch_notice(deferred=None)

            # In batch mode, defer destructive rebuilds until import so the
            # existing search index remains usable while the batch runs.
            # A rebuild empties the index first, so check the embedding API
            # before that: off the network it accepts, a rebuild would leave
            # an empty index behind and nothing to say so.
            embedding_function = getattr(self.chroma_client, "embedding_function", None)
            api_checked = False
            # Whether a provider call succeeded in this run; only then may
            # the stored API failure be cleared.
            api_answered = False
            if force_full_rebuild and not batch_enabled:
                if isinstance(embedding_function, RemoteEmbeddingFunction):
                    api_answered = embedding_function.check_reachable()
                    api_checked = True
                logger.info("Force rebuilding database...")
                # On record first, so that a crash in between can't leave an
                # emptied index that older batch runs think is still theirs.
                self._save_last_rebuild(str(self._run_group_id))
                # The watermark describes the index about to be emptied. Zero
                # it first, so a rebuild that stops part way (or a crash
                # right after the reset) sends the next plain update-db
                # through a full scan instead of "library unchanged".
                self._save_update_config(
                    last_sync_version=0, library_key=str(self._run_group_id)
                )
                self.chroma_client.reset_collection()
                rebuild_reset = True

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
            index_gaps = self._find_index_gaps(limit=limit, force_rebuild=force_full_rebuild)
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
                # last_api_failure stays: this path made no provider call, so
                # it cannot say the API is reachable again.
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
                    already_fetched={item.get("key") for item in all_items},
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
            # Items kept for a retry, or given up on, that this run writes. A
            # kept item is rewritten in full, so it is no level fill.
            due = {
                *(self.update_config.get("retry_item_keys") or []),
                *(self.update_config.get("retry_item_keys_given_up") or []),
            }
            self._retry_keys_in_run = {item["key"] for item in all_items if item.get("key") in due}
            self._unprepared_keys = set()
            self._level_fill_keys = {
                item["key"]
                for item in all_items
                if item.get("_index_levels") and item["key"] not in self._retry_keys_in_run
            }

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
                    scope=_this_run(),
                )
                # This run did its work, so an auto-update is not due again
                # until its next turn. Without this, should_update stayed true
                # until an import and every search could pay for the same
                # changes again. The sync version waits for the import, unless
                # there is nothing to import: then it moves now, or every later
                # run would fetch the same changes again.
                self.update_config["last_update"] = datetime.now().isoformat()
                if stats.get("batch_nothing_to_embed"):
                    # Kept items with no text to embed have nothing left to write.
                    self._settle_retry_keys(
                        written=set(self._retry_keys_in_run) - self._unprepared_keys,
                        failed=set(),
                        keep_failed=False,
                    )
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
                        "budget; the next 'zotero-mcp update-db' or 'zotero-mcp batch-import' "
                        "(or --auto-loop) submits them as running batches finish.\n"
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

            # One short request before any real work: an API that only answers
            # on some networks would otherwise fail every batch of the run.
            if (
                all_items
                and not api_checked
                and isinstance(embedding_function, RemoteEmbeddingFunction)
            ):
                api_answered = embedding_function.check_reachable()

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
                title = item.get("data", {}).get("title", "")
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
            failed_keys: set[str] = set()
            if _failed_docs:
                try:
                    sys.stderr.write(f"\r{' ' * _term_width(120)}\r")
                    sys.stderr.write(f"\n  Retrying {len(_failed_docs)} failed items...\n")
                except Exception:
                    pass

                import time as _retry_time

                _retry_time.sleep(1)  # Brief pause before retry
                # Articles first here too.
                _failed_docs.sort(key=lambda failed: "parent_item_key" in failed[1])

                retry_ok = 0
                # Items whose passages all got written, and items with any
                # passage still failing. Pruning waits until the item's
                # passages are done, then runs once per item, not per passage.
                retried_metas: dict[str, dict[str, Any]] = {}
                still_failing: set[str] = set()
                for doc, meta, doc_id in _failed_docs:
                    parent = meta.get("parent_item_key")
                    try:
                        self.chroma_client.upsert_documents([doc], [meta], [doc_id])
                        if parent is not None:
                            retried_metas[parent] = meta
                        retry_ok += 1
                        stats["errors"] -= 1  # Remove from error count
                        # Don't classify as added vs updated — when the
                        # original batch failed, the add/update lookup never
                        # ran, so we don't know which category it belongs in.
                        # Track recovered items in their own bucket.
                        stats["recovered_items"] += 1
                    except EmbeddingAPIUnreachable:
                        raise
                    except Exception as e2:
                        retry_fail += 1
                        failed_keys.add(doc_id.split("#", 1)[0])
                        if parent is not None:
                            still_failing.add(parent)
                        logger.error(f"Retry failed for {doc_id}: {e2}")

                # No workers are running here, so _chroma_call_lock is not
                # needed for this prune.
                self._prune_stale_chunks(
                    [m for k, m in retried_metas.items() if k not in still_failing]
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
            # A split index keeps those items for a retry instead, so the
            # changes the run did write are not bought again on every run.
            # An unmigrated index keeps holding the watermark.
            self.update_config["last_update"] = datetime.now().isoformat()
            written = stats["added_items"] + stats["updated_items"] + stats["recovered_items"]
            if api_answered or written:
                self.update_config.pop("last_api_failure", None)
            keep_failed = self._separate_passages()
            failed_keys |= self._unprepared_keys & set(self._retry_keys_in_run)
            # A kept item that fails again is tracked by its retry key, so it
            # needs no held watermark; holding it would buy every change again.
            kept_before = set(self.update_config.get("retry_item_keys") or [])
            given_up = self._settle_retry_keys(
                written=(set(self._retry_keys_in_run) | set(self._undropped_passages)) - failed_keys,
                failed=failed_keys,
                keep_failed=keep_failed,
            )
            if given_up:
                stats["retry_given_up"] = self._given_up_message(given_up)
                _report(f"  {stats['retry_given_up']}\n")
            # The server's background update throws its stats away.
            self._note_given_up(announce=bool(given_up))
            if stats.get("deletion_skipped_reason") or (
                retry_fail and not keep_failed and failed_keys - kept_before
            ):
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
            # Stop here: every later request would fail the same way. What
            # was already written stays; last_update and the sync watermark
            # are left alone, so the next run picks up the rest.
            logger.error(f"Stopping the update: {e}")
            self._record_api_failure(e)
            if rebuild_reset:
                # The rebuild started from an empty index, so everything in it
                # now was written by this run.
                indexed = stats["added_items"] + stats["updated_items"]
                stats["error"] = (
                    f"{e} The rebuild stopped part way: {indexed} item(s) are in "
                    "the index now. Run `zotero-mcp update-db` once the API can "
                    "be reached to add the rest."
                )
            else:
                stats["error"] = (
                    f"{e} Items already indexed were kept; run `zotero-mcp update-db` "
                    "again once the API can be reached."
                )
            stats["embedding_api_unreachable"] = e.host
            end_time = datetime.now()
            stats["duration"] = str(end_time - start_time)
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
            # The server keeps one instance: the next run starts without these.
            self._undropped_passages = frozenset()
            if self._held_text is not None:
                self._held_text.close()
                self._held_text = None
            self._retry_keys_in_run = frozenset()
            self._unprepared_keys = frozenset()
            # Release the update flock on every exit path. Paired with the
            # __enter__ call above; the "not acquired" branch releases
            # separately before its early return, so this finally only runs
            # for the path where we actually hold the lock.
            lock_cm.__exit__(None, None, None)

    def _record_api_failure(self, error: EmbeddingAPIUnreachable) -> None:
        """Store when the embedding API was last unreachable, and its host.

        Kept in ``update_config`` so the status tools can show it and the
        server's background syncs can hold off for an hour (see
        ``config_light.background_update_due``). Only the time, host and a
        fixed reason phrase are stored: never a URL or a key. The next
        successful update removes it.
        """
        self.update_config["last_api_failure"] = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "host": error.host,
            "reason": error.reason,
        }
        self._save_update_config()

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

        chunking = self._chunking_enabled
        separate = self._separate_passages()

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
        for item in items:
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
                if isinstance(self._unprepared_keys, set) and item.get("key"):
                    self._unprepared_keys.add(item["key"])

        # Which items already existed (drives added-vs-updated). Stale
        # passages are pruned only after the new ones are written
        # (_prune_stale_chunks), so a failed embed never loses an item (#610).
        existing_item_keys: set[str] = set()
        if documents and not force_rebuild:
            with self._chroma_call_lock:
                if separate:
                    # An item is indexed if it has a record at either level.
                    probe_ids = item_keys_order + [f"{k}#0" for k in item_keys_order]
                    found = self.chroma_client.get_existing_ids(probe_ids)
                    existing_item_keys = {doc_id.split("#", 1)[0] for doc_id in found}
                elif chunking:
                    probe_ids = [f"{k}#0" for k in item_keys_order]
                    existing_chunk0 = self.chroma_client.get_existing_ids(probe_ids)
                    existing_item_keys = {cid.split("#", 1)[0] for cid in existing_chunk0}
                else:
                    existing_item_keys = self.chroma_client.get_existing_ids(ids)

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
                    if stop_event.is_set():
                        # Drain without embedding: the run is stopping.
                        continue
                    documents, metadatas, ids, item_keys = payload
                    try:
                        vectors = embedding_function(documents)
                    except EmbeddingAPIUnreachable as exc:
                        # Every later request would fail the same way: stop
                        # the producer and the other workers, and let the
                        # committer write what is already embedded.
                        record_thread_error(exc)
                        stop_event.set()
                        continue
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
                    if done := [k for k in write_keys if k not in keep_cached]:
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
            left = {doc_id.split("#", 1)[0] for doc_id in self.chroma_client.get_existing_ids(
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
        """Remember an item whose old passages are still stored after its new article.

        It is kept for a retry at once, and saved, so a run stopped before its
        new passages are written can't leave the old ones for good: the next
        run rewrites the item. A run that writes them lets it go.
        """
        if isinstance(self._undropped_passages, set):
            self._undropped_passages.add(item_key)
        self._settle_retry_keys(written=set(), failed={item_key}, keep_failed=True)
        self._save_update_config()

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
                and "parent_item_key" not in m
                and m["item_key"] not in self._level_fill_keys
            }
            for item_key, group_id in scopes.items():
                self.chroma_client.delete_item_chunks(item_key, group_id=group_id)
        counts = {
            m["parent_item_key"]: m["n_chunks"]
            for m in metadatas
            if "parent_item_key" in m and "n_chunks" in m
        }
        for item_key, n_chunks in counts.items():
            self.chroma_client.prune_item_chunks(item_key, n_chunks)

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
                    if done := [k for k in item_keys_order if k not in keep_cached]:
                        fulltext_cache.evict_many(done, config_path=self.config_path)
                except Exception as e:
                    logger.debug(f"Fulltext cache eviction failed: {e}")
            except EmbeddingAPIUnreachable:
                raise
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
        # Saving the refreshed status needs the update lock: a background
        # collect may be writing this manifest. While someone holds it, this
        # still reports the live status, it just doesn't save it.
        with _acquire_update_lock(_update_lock_path()) as acquired:
            manifest = self._find_run(provider, next(iter(selected_ids), None))
            manifest = module.refresh_manifest_status(
                manifest,
                embedding_config=self.chroma_client.embedding_config,
                batch_ids=selected_ids or None,
                persist=bool(acquired),
                # A status check never marks a chunk lost: a 404 from it may
                # only mean the key or base URL is wrong.
                mark_lost=False,
                recheck_lost=True,
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

        Any newer run *for the same library* counts, not just the newest: a
        ``--limit`` run on top of a rebuild doesn't undo what the rebuild took
        over. Runs are ranked by the immutable ``created_at``/``run_id`` order
        rather than by manifest mtime (which a status refresh or an import would
        bump, letting a superseded run pass itself off as current).
        """
        module = _batch_module(provider)
        for newer in module.manifests_for_group(
            config_path=self.config_path, group_id=manifest.get("group_id")
        ):
            if newer.get("run_id") == manifest.get("run_id"):
                break
            if _is_superseded(manifest, newer, _batch_adapter(provider)):
                return str(newer.get("run_id"))
        return None

    def _overtaken_by_sync(self, manifest: dict[str, Any]) -> int | None:
        """The sync version that already includes this run's items, if any.

        A sync version only moves when a whole update has landed, so one past
        the version this run was cut from and at least the version it targets
        means another update, such as a realtime one, indexed these items and
        anything newer. The run's results would write older text over newer, and
        its waiting chunks would be paid for twice. A rebuild is never overtaken:
        it exists to replace the index, which an incremental sync doesn't do
        (:meth:`_stale_rebuild_note` decides when one has gone stale).

        A manifest from before ``since_version`` existed can't say where it was
        cut from, so a target behind the sync version has to be enough.
        """
        since, target = manifest.get("since_version"), manifest.get("target_sync_version")
        if manifest.get("force_full_rebuild") or target is None:
            return None
        if all(batch.get("imported_at") for batch in manifest.get("batches", [])):
            return None  # all in: nothing left to write over anything
        group_id = manifest.get("group_id")
        current = self._load_last_sync_version(None if group_id is None else str(group_id))
        if "since_version" not in manifest:
            return current if target < current else None
        if since is None:
            return None
        return current if since < current and target <= current else None

    def _run_is_usable(self, provider: str, run: dict[str, Any]) -> bool:
        """May this run go into this index without the user naming it?

        Not one a later sync overtook, one cut before the index was last reset,
        one made with another embedding model, whose vectors no query here could
        match, or one the user gave up on.
        """
        adapter = _batch_adapter(provider)
        model_name = self.chroma_client.embedding_config.get("model_name", adapter.default_model)
        return (
            not run.get("abandoned_at")
            and run.get("model") in (None, model_name)
            and self._overtaken_by_sync(run) is None
            and not self._predates_reset(run)
        )

    def _predates_reset(self, run: dict[str, Any]) -> bool:
        """Was this run cut against an index that has since been replaced?

        A run submitted before the library's last reset (``last_rebuild_at``, set
        by a batch or a realtime rebuild) was cut from a sync version that
        described the index before it was emptied. Promoting its target would
        claim that index's contents for the new one: after a rebuild given up
        part way, that moves the sync version back up from 0 and the missing
        items never return. Such a run is never imported unattended and never
        promotes; an import by batch id still lands its records. A rebuild is
        self-contained and has its own rule (:meth:`_stale_rebuild_note`).
        """
        if run.get("force_full_rebuild"):
            return False
        group_id = run.get("group_id")
        library_key = str(group_id) if group_id is not None else self._active_library_key()
        last_rebuild = batch_common._parse_stamp(self._load_last_rebuild(library_key))
        created = batch_common._parse_stamp(run.get("created_at"))
        return bool(last_rebuild and created and created < last_rebuild)

    def _find_run(self, provider: str, batch_id: str | None = None) -> dict[str, Any]:
        """The run with this batch id, or else the one to work on for this library."""
        return _batch_module(provider).find_manifest(
            config_path=self.config_path,
            batch_id=batch_id,
            group_id=self._pinned_group_id(),
            is_current=lambda run: self._run_is_usable(provider, run),
        )

    def _load_last_rebuild(self, library_key: str) -> str | None:
        """When an index for this library was last replaced, if on record."""
        try:
            with open(self.config_path) as f:
                stamps = json.load(f).get("semantic_search", {}).get("last_rebuild_at") or {}
            return stamps.get(library_key)
        except (OSError, ValueError, TypeError, AttributeError):
            return None

    def _save_last_rebuild(self, library_key: str, stamp: str | None = None) -> None:
        """Record that the index is being replaced, by whichever path."""
        if not self.config_path:
            return
        full_config = self._config_for_update()
        if full_config is None:
            return
        section = full_config.setdefault("semantic_search", {})
        if not isinstance(section.get("last_rebuild_at"), dict):
            section["last_rebuild_at"] = {}
        section["last_rebuild_at"][library_key] = stamp or batch_common._utc_now()
        try:
            write_json_atomic(self.config_path, full_config)
        except Exception as e:
            logger.error(f"Error saving the last rebuild time: {e}")

    def _stale_rebuild_note(self, run: dict[str, Any]) -> str | None:
        """Why this rebuild must not replace the index unattended, if it must not.

        Importing one empties the ChromaDB collection, so it has to be
        demonstrably what the user still wants: nothing newer has synced past its
        target, no other rebuild has landed since it was submitted, and its
        chunking matches the index. A rebuild already part way in is exempt: the
        reset has happened and finishing is all that's left.
        """
        if not run.get("force_full_rebuild") or any(b.get("imported_at") for b in run.get("batches", [])):
            return None
        group_id = run.get("group_id")
        library_key = str(group_id) if group_id is not None else self._active_library_key()
        own_reset = batch_common._parse_stamp(run.get("reset_at"))
        last_reset = batch_common._parse_stamp(self._load_last_rebuild(library_key))
        if own_reset and (last_reset is None or own_reset >= last_reset):
            return None  # it did the latest reset itself and was cut off before a chunk landed
        run_id = run.get("run_id")
        choice = (
            f"Run 'zotero-mcp batch-import' to replace the index with this older rebuild, or "
            f"'zotero-mcp batch-abandon --run-id {run_id}' to drop it."
        )
        target = run.get("target_sync_version")
        if target is None or self._load_last_sync_version(library_key) > target:
            return f"Rebuild run {run_id} is older than the current index. {choice}"
        last_rebuild = self._load_last_rebuild(library_key)
        if last_rebuild and last_rebuild > str(run.get("created_at") or ""):
            return f"Rebuild run {run_id} was submitted before the index was last rebuilt. {choice}"
        try:
            index_chunked = self.chroma_client.has_passage_documents()
        except Exception as e:
            logger.debug("Could not read the index's chunking: %s", e)
            return f"Rebuild run {run_id} is waiting: the index's chunking could not be read ({type(e).__name__}). {choice}"
        if index_chunked != bool(run.get("chunked")):
            return f"Rebuild run {run_id} has different passage chunking from the index. {choice}"
        return None

    def _collect_provider(self) -> str | None:
        """The provider whose runs can go into this index, if there is one.

        Only the configured embedding model's: vectors from any other would land
        in a ChromaDB collection where no query could match them. Without a
        config file there is no batch directory to look at, either.
        """
        model = getattr(self.chroma_client, "embedding_model", None)
        return model if self.config_path and model in _BATCH_MODULES else None

    def _unfinished_batch_runs(self, provider: str, group_id: int) -> list[dict[str, Any]]:
        """Runs for ``group_id`` still worth working on, oldest first.

        Reads local manifests only, so it is safe to call on every update. Leaves
        out runs that a newer run or a later sync has taken over, runs made with
        another model, which this index can't use, and rebuilds too stale to
        replace the index unattended (:meth:`_stale_rebuild_note`). Those last
        take a decision from the user, and :meth:`_batch_notes` asks for it.
        """
        module = _batch_module(provider)
        adapter = _batch_adapter(provider)
        runs = module.manifests_for_group(config_path=self.config_path, group_id=group_id)
        unfinished = [
            run
            for position, run in enumerate(runs)
            if batch_common.run_is_unfinished(run, adapter)
            and self._run_is_usable(provider, run)
            and not any(_is_superseded(run, newer, adapter) for newer in runs[:position])
            and self._stale_rebuild_note(run) is None
        ]
        unfinished.reverse()
        return unfinished

    def _blocking_batch_runs(self, provider: str, group_id: int) -> list[dict[str, Any]]:
        """Unfinished runs that make an update wait, oldest first.

        Not one past the provider's window: nothing more is coming from it unless
        someone asks, and it can't be allowed to hold updates up for good. A run
        that fills missing records holds them up longer, for
        ``BATCH_RUN_BLOCKS_FILL_DAYS`` after its last activity: no new run would
        cover its fill (:func:`batch_common.run_covers`), and with the fill held
        back meanwhile (:meth:`_unfinished_batch_run`), it would be left behind.
        """
        adapter = _batch_adapter(provider)
        return [
            run for run in self._unfinished_batch_runs(provider, group_id)
            if not batch_common.run_past_window(run, adapter) or self._fill_still_holds(run)
        ]

    def _deferral(self, provider: str, blocking: list[dict[str, Any]], batch: bool) -> str:
        """The deferral message for runs from :meth:`_blocking_batch_runs`, naming the newest."""
        adapter = _batch_adapter(provider)
        run = blocking[-1]
        held_for_fill = batch_common.run_past_window(run, adapter)
        return _deferred_message(
            run,
            len(blocking),
            batch,
            adapter.completion_window_hours + batch_common.WINDOW_MARGIN_HOURS,
            fill_days=self.BATCH_RUN_BLOCKS_FILL_DAYS if held_for_fill else None,
        )

    def _fill_still_holds(self, run: dict[str, Any]) -> bool:
        """Is this a fill run whose last activity is recent enough to hold up a new run?"""
        if not batch_common.run_has_level_fill(run):
            return False
        active = batch_common.last_activity(run)
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.BATCH_RUN_BLOCKS_FILL_DAYS)
        return active is not None and active >= cutoff

    def _batch_notes(self, provider: str, group_id: int, skip_rebuilds: bool = False) -> list[str]:
        """What needs a decision from the user, whatever this pass could do.

        Neither a stale rebuild nor one that can't finish is collected, so nothing
        else would ever mention them.
        """
        adapter = _batch_adapter(provider)
        runs = _batch_module(provider).manifests_for_group(config_path=self.config_path, group_id=group_id)
        notes = []
        if skip_rebuilds:
            return notes
        for position, run in enumerate(runs):
            if (
                batch_common.run_is_unfinished(run, adapter)
                and self._run_is_usable(provider, run)
                and not any(_is_superseded(run, newer, adapter) for newer in runs[:position])
                and (note := self._stale_rebuild_note(run))
            ):
                notes.append(note)
        # Only the latest run: any newer run has moved on from an older rebuild.
        newest = runs[0] if runs else None
        if newest and newest.get("force_full_rebuild") and self._run_is_usable(provider, newest):
            library_key = str(group_id)
            last_rebuild = self._load_last_rebuild(library_key)
            dead = [
                batch.get("batch_id") or "(pending)"
                for batch in newest.get("batches", [])
                if not batch_common.is_settled(batch) and batch_common._entry_is_dead(adapter, batch)
            ]
            if dead and not (last_rebuild and last_rebuild > str(newest.get("created_at") or "")):
                run_id = newest.get("run_id")
                notes.append(
                    f"Rebuild run {run_id} can't finish, because these chunks failed: {', '.join(dead)}. "
                    "The index is unchanged. Run 'zotero-mcp update-db --force-rebuild' to rebuild "
                    f"again, or 'zotero-mcp batch-abandon --run-id {run_id}' to drop it."
                )
        return notes

    def _finish_promotions(self, provider: str, group_id: int) -> None:
        """Finish a sync version save that a crash cut off. No network call.

        The last chunk's save and the sync version's are two writes. A crash in
        between leaves a run that looks finished, and the next update would buy
        its items again.
        """
        adapter = _batch_adapter(provider)
        runs = _batch_module(provider).manifests_for_group(config_path=self.config_path, group_id=group_id)
        for position, run in enumerate(runs):
            batches = run.get("batches", [])
            if (
                "since_version" in run  # older manifests have no promoted_at to read
                and batches
                and all(batch_common.is_settled(batch) for batch in batches)
                and not run.get("promoted_at")
                and self._run_is_usable(provider, run)
                and not any(_is_superseded(run, newer, adapter) for newer in runs[:position])
            ):
                self._promote_run(provider, run, _run_replaced_index(run))

    def _promote_run(self, provider: str, manifest: dict[str, Any], rebuilt_now: bool) -> list[str]:
        """A whole run has landed: move the sync version of its library.

        Of the library the batch was submitted against, not whichever library
        happens to be active now. An import never lowers it: running batch-import
        again on a settled run would make the next update re-embed what is
        already indexed. The exception is a rebuild landing right now, which just
        replaced the whole ChromaDB collection: the sync version has to say what
        the collection now holds, or the changes between the two versions would
        be missing for good, and 0 when that is unknown.

        Returns the items given up on now (:meth:`_settle_retry_keys`).
        """
        self.update_config["last_update"] = datetime.now().isoformat()
        group_id = manifest.get("group_id")
        library_key = None if group_id is None else str(group_id)
        target = manifest.get("target_sync_version")
        failed_ids = manifest.get("failed_ids") or []
        failed_keys = {doc_id.split("#", 1)[0] for doc_id in failed_ids}
        keep_failed = bool(failed_ids) and bool(manifest.get("reset_at") or self._separate_passages())
        kept_before = set(self.update_config.get("retry_item_keys") or [])
        given_up = self._settle_retry_keys(
            written=set(manifest.get("retry_keys") or []) - failed_keys,
            failed=failed_keys,
            keep_failed=keep_failed,
        )
        if keep_failed:
            # Records that never got an embedding. Holding the watermark would
            # pay for this run's other items again every run, and after a
            # forced rebuild the old one described a collection that is gone.
            # Move on: the items are kept for a retry (_settle_retry_keys).
            rebuilt_now = rebuilt_now or bool(manifest.get("reset_at"))
        elif failed_keys - kept_before:
            # An unmigrated index has no gap fill, so the watermark stays, as
            # on the realtime path (#610). An item kept for a retry already
            # has its key, and needs no held watermark.
            target = None
        if rebuilt_now:
            target = 0 if target is None else target
        elif target is not None and target < self._load_last_sync_version(library_key):
            target = None
        self._save_update_config(last_sync_version=target, library_key=library_key)
        manifest["promoted_at"] = datetime.now().isoformat()
        _batch_module(provider).save_manifest(manifest)
        return given_up

    def _collect_unfinished_batches(
        self,
        stats: dict[str, Any],
        submit_waiting: bool = True,
        skip_rebuilds: bool = False,
        allow_rebuild_import: bool = True,
    ) -> None:
        """Import finished files and submit waiting files of unfinished runs.

        Call with the update lock held and the run's library pinned. A failure
        is reported in ``stats["batch_collect"]`` and in the batch notice, never
        raised, so the update goes on to report its own outcome. Nothing happens
        here without an unfinished run, which means no network call. Each provider
        call is one short request: this doesn't wait for a batch, so the lock
        isn't held across a provider's batch window.
        """
        provider = self._collect_provider()
        if provider is None:
            # The configured model can't import any batch run, so none of the
            # notes can be acted on: batch-import skips other-model runs, and
            # nothing would ever clear them. They go, with the deferral, which
            # nothing can cause without a batch provider either.
            self._save_batch_notice(notes=[], deferred=None)
            return
        group_id = self._pinned_group_id()
        adapter = _batch_adapter(provider)
        errors: list[str] = []
        given_up_notes: list[str] = []
        gave_up = False
        try:
            self._finish_promotions(provider, group_id)
            runs = self._unfinished_batch_runs(provider, group_id)
        except Exception as e:
            logger.warning(f"Could not read the batch runs: {e}")
            runs = []
            errors.append(f"could not read the batch runs ({type(e).__name__})")

        summary: dict[str, Any] = {
            "provider": provider,
            "runs": [],
            "imported_items": 0,
            "submitted_chunks": 0,
            "errors": errors,
        }
        # Oldest first, so that newer text lands last and a sync version can
        # only move forward through the runs.
        for run in runs:
            run_id = run.get("run_id")
            if skip_rebuilds and run.get("force_full_rebuild"):
                continue
            past_window = batch_common.run_past_window(run, adapter)
            try:
                if self._superseded_by(provider, run) or self._overtaken_by_sync(run) is not None:
                    continue  # an earlier run in this pass took it over
                result = self._import_batch(
                    provider,
                    _skip_lock=True,
                    _collect_run=run,
                    # Nothing past the window spends more unless someone asks.
                    _submit_waiting=submit_waiting and not past_window,
                    _allow_reset=allow_rebuild_import,
                )
            except Exception as e:
                # Whatever was imported is saved, and the sync version only
                # moves when a whole run has landed, so this run just stays
                # unfinished for the next attempt. The SDK's own text goes to the
                # debug log only: it can carry request details.
                logger.debug("Collecting run %s failed: %s", run_id, e)
                errors.append(
                    f"run {run_id}: could not check or import it ({type(e).__name__}); "
                    "the next update tries again"
                )
                result = None
            if result is not None:
                summary["runs"].append(run_id)
                summary["imported_items"] += result.get("imported_items", 0)
                summary["submitted_chunks"] += result.get("batches_submitted", 0)
                if result.get("needs_user"):
                    errors.append(result["deferred"])
                elif result.get("deferred"):
                    summary.setdefault("waiting", []).append(result["deferred"])
                failed_rows = result.get("failed_items", 0) + result.get("missing_items", 0)
                if failed_rows:
                    errors.append(f"run {run_id}: {failed_rows} item(s) came back without an embedding")
                if result.get("retry_given_up"):
                    gave_up = True
                    given_up_notes.append(f"run {run_id}: {result['retry_given_up']}")
                    errors.append(given_up_notes[-1])
            batches = run.get("batches", [])
            if batch_common.run_is_unfinished(run, adapter):
                if past_window:
                    errors.append(
                        f"run {run_id} is past the provider's {adapter.completion_window_hours}-hour window "
                        "and no longer holds up updates. 'zotero-mcp batch-import' retries it and sends any "
                        f"waiting chunks; 'zotero-mcp batch-abandon --run-id {run_id}' drops it."
                    )
            elif not run.get("force_full_rebuild"):  # _batch_notes speaks for a dead rebuild
                lost = sum(1 for batch in batches if batch.get("lost"))
                # Rows of a chunk that ended early go round again on their own,
                # so only a chunk nothing more can come from counts here.
                short = lost + sum(
                    1 for batch in batches
                    if not batch.get("lost")
                    and not batch_common.is_settled(batch)
                    and batch_common._entry_is_dead(adapter, batch)
                )
                if short:
                    what = f"the provider no longer has {lost} of its chunks" if lost else (
                        f"{short} chunk(s) failed or expired at the provider"
                    )
                    errors.append(
                        f"run {run_id}: {what}. The sync version stays put, so the next update submits "
                        "every change since that version again, including chunks already imported."
                    )

        try:
            errors.extend(self._batch_notes(provider, group_id, skip_rebuilds))
        except Exception as e:
            logger.warning(f"Could not read the batch runs: {e}")
            errors.append(f"could not read the batch runs ({type(e).__name__})")

        try:
            still_blocking = bool(self._blocking_batch_runs(provider, group_id))
        except Exception:
            still_blocking = True
        keep_deferred = self.batch_notice(group_id).get("deferred") if still_blocking else None
        # Items given up on get a notice of their own, which outlives this collect.
        self._save_batch_notice(notes=[e for e in errors if e not in given_up_notes], deferred=keep_deferred)
        self._note_given_up(announce=gave_up)

        if not summary["runs"] and not errors:
            return

        stats["batch_collect"] = summary
        for error in errors:
            logger.warning(f"Collecting batch results: {error}")
            _report(f"Collecting batch results: {error}\n")
        if summary["runs"]:
            if summary["imported_items"] or summary["submitted_chunks"]:
                _report(
                    f"Collected {len(summary['runs'])} unfinished batch run(s): "
                    f"{summary['imported_items']} embeddings imported, "
                    f"{summary['submitted_chunks']} waiting chunk(s) submitted.\n"
                )
            else:
                _report(f"Checked {len(summary['runs'])} unfinished batch run(s): nothing new is ready yet.\n")

    def batch_notice(self, group_id: int | None = None) -> dict[str, Any]:
        """The last collect problems and deferral for this library, if any."""
        if group_id is None:
            group_id = self._pinned_group_id() if getattr(self, "_run_group_id", None) is not None \
                else self._client_group_id()
        return load_batch_notice(self.config_path, str(group_id))

    _KEEP = object()

    def _refresh_batch_notice(self, provider: str, group_id: int, run_id: str) -> None:
        """Bring the notice up to date after the user acted on ``run_id``.

        Notes about that run go, unless they still apply (:meth:`_batch_notes`
        says so); notes about other runs stay. A deferral stays only while
        something still holds updates up. Called with the update lock held.
        """
        if not self.config_path:
            return  # no notice file, and no batch directory to read
        try:
            old = self.batch_notice(group_id)
            current = [note for note in self._batch_notes(provider, group_id) if run_id in note]
            notes = [note for note in old.get("notes", []) if run_id not in note] + current
            deferred = old.get("deferred") if self._blocking_batch_runs(provider, group_id) else None
        except Exception as e:
            logger.debug("Could not refresh the batch notice: %s", e)
            return
        self._save_batch_notice(notes=notes, deferred=deferred, library_key=str(group_id))

    def _note_given_up(self, announce: bool) -> None:
        """Keep the notice about items given up on in step with ``retry_item_keys_given_up``.

        ``announce`` says this run gave items up: the notice then names every
        item given up on. An existing notice follows the list as it shrinks,
        and goes once the items have changed in Zotero and been written. Items
        given up on before this notice existed are left to the status line.
        """
        keys = list(self.update_config.get("retry_item_keys_given_up") or [])
        if not keys:
            self._save_batch_notice(given_up=None)
        elif announce or self.batch_notice(self._pinned_group_id()).get("given_up"):
            self._save_batch_notice(given_up=self._given_up_message(keys))

    def _save_batch_notice(
        self,
        notes: Any = _KEEP,
        deferred: Any = _KEEP,
        library_key: str | None = None,
        given_up: Any = _KEEP,
    ) -> None:
        """Keep what a background run has to tell, for whoever looks next.

        A background run's stats go nowhere. Written only when something changed,
        and only with the update lock held, like the manifests. Never fatal: this
        is a courtesy on top of a run that has its own outcome.
        """
        if not self.config_path:
            return
        path = batch_notice_path(self.config_path)
        if library_key is None:
            library_key = str(self._pinned_group_id())
        try:
            try:
                with open(path, encoding="utf-8") as f:
                    notices = json.load(f)
                if not isinstance(notices, dict):
                    notices = {}
            except (OSError, ValueError):
                notices = {}
            old = notices.get(library_key) or {}
            new = {
                "notes": old.get("notes", []) if notes is self._KEEP else notes,
                "deferred": old.get("deferred") if deferred is self._KEEP else deferred,
            }
            # Items given up on, from either path; see _note_given_up.
            kept_given_up = old.get("given_up") if given_up is self._KEEP else given_up
            if kept_given_up:
                new["given_up"] = kept_given_up
            comparable = {key: value for key, value in old.items() if key != "updated_at"}
            if not new["notes"] and not new["deferred"] and not new.get("given_up"):
                if library_key not in notices:
                    return
                notices.pop(library_key)
            elif comparable == new:
                return
            else:
                notices[library_key] = {**new, "updated_at": datetime.now().isoformat()}
            write_json_atomic(path, notices)
        except Exception as e:
            logger.debug("Could not save the batch notice: %s", e)

    def abandon_batch_run(self, provider: str, run_id: str | None = None, confirm: bool = False) -> dict[str, Any]:
        """Give up on a run: it stops being collected, reported and holding anything up.

        The sync version doesn't move, so the next update covers its items again.
        The exception is a rebuild that already emptied the ChromaDB collection:
        the index then holds only some of the library, and an update from the
        old sync version would see nothing to do. The sync version goes to 0
        instead, so the next update reads the whole library
        (``sync_version_reset`` in the result). That can mean paying to embed the
        whole library again, so it takes ``confirm``; without it nothing changes
        and the result says what it would cost (``needs_confirmation``).

        Its files stay, and ``batch-import --batch-id`` can still import from them.
        With no ``run_id`` this works only if there's exactly one run it could mean.
        """
        with _acquire_update_lock(_update_lock_path()) as acquired:
            if not acquired:
                raise RuntimeError(
                    f"Another semantic-search update is already running (lock held at {_update_lock_path()})"
                )
            module = _batch_module(provider)
            adapter = _batch_adapter(provider)
            group_id = self._client_group_id()
            runs = module.manifests_for_group(config_path=self.config_path, group_id=group_id)
            if run_id is None:
                candidates = [
                    run for position, run in enumerate(runs)
                    if not run.get("abandoned_at") and (
                        batch_common.run_is_unfinished(run, adapter)
                        or (position == 0 and run.get("force_full_rebuild") and any(
                            not batch.get("imported_at") for batch in run.get("batches", [])
                        ))
                    )
                ]
                if len(candidates) != 1:
                    return {"candidates": [run.get("run_id") for run in candidates]}
                run = candidates[0]
            else:
                matching = [run for run in runs if run.get("run_id") == run_id]
                if not matching:
                    raise FileNotFoundError(f"No {adapter.label} batch run {run_id} for this library")
                run = matching[0]
            # reset_at is written just before the reset. Manifests from before
            # it (no since_version either) have only their imported chunks to
            # show for it; a rebuild that gave a file up writes over the index
            # without a reset.
            if "since_version" in run:
                did_reset = bool(run.get("reset_at"))
            else:
                did_reset = bool(run.get("reset_at")) or (
                    any(batch.get("imported_at") for batch in run.get("batches", []))
                    and not any(batch.get("abandoned") for batch in run.get("batches", []))
                )
            reset_happened = bool(run.get("force_full_rebuild") and not run.get("promoted_at") and did_reset)
            if reset_happened and not confirm:
                found_id = run.get("run_id")
                return {
                    "needs_confirmation": found_id,
                    "message": (
                        f"Rebuild run {found_id} has already replaced part of the index. Giving it up "
                        "sets the sync version to 0, so the next update reads the whole library and "
                        "puts back what is missing. Without --fulltext that embeds the whole library "
                        "again, at your provider's price, and with auto-update on the next search may "
                        "start it. In local mode, 'zotero-mcp update-db --fulltext' skips items already "
                        "indexed and costs only what is missing. If the provider still has this run's "
                        "results, 'zotero-mcp batch-import' finishes the rebuild instead. To give it "
                        f"up anyway: 'zotero-mcp batch-abandon --run-id {found_id} --yes'."
                    ),
                }
            run["abandoned_at"] = datetime.now().isoformat()
            module.save_manifest(run)
            if reset_happened:
                self._save_update_config(last_sync_version=0, library_key=str(group_id))
            self._refresh_batch_notice(provider, group_id, str(run.get("run_id")))
            return {"abandoned": run.get("run_id"), "sync_version_reset": reset_happened}

    def has_unfinished_batch_run(self) -> bool:
        """Is a run holding updates back for this library? Reads local files only."""
        provider = self._collect_provider()
        return bool(provider and self._blocking_batch_runs(provider, self._client_group_id()))

    def collect_batch_results(self, allow_rebuild_import: bool = True, update_due: bool = False) -> dict[str, Any]:
        """Collect unfinished batch runs and do nothing else; a new run never starts here.

        ``update_due`` says an automatic update was due and this collect ran in
        its place. If a run still holds it up, the deferral is recorded in the
        notice, as the update itself would have done, so the user can see why
        nothing is being updated.
        """
        stats: dict[str, Any] = {}
        with _acquire_update_lock(_update_lock_path()) as acquired:
            if not acquired:
                stats["skipped_reason"] = "another_update_in_progress"
                return stats
            try:
                self._run_group_id = self._client_group_id()
                if read_error := self.chroma_client.read_error():
                    stats["error"] = f"Cannot read the semantic index: {read_error}"
                    return stats
                self._collect_unfinished_batches(stats, allow_rebuild_import=allow_rebuild_import)
                provider = self._collect_provider()
                blocking = self._blocking_batch_runs(provider, self._run_group_id) if update_due and provider else []
                if blocking:
                    try:
                        batch_enabled = self._resolve_batch_mode()[0]
                    except Exception:
                        batch_enabled = False
                    stats["batch_deferred"] = self._deferral(provider, blocking, batch_enabled)
                    self._save_batch_notice(deferred=stats["batch_deferred"])
            finally:
                self._run_group_id = None
        return stats

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
        _collect_run: dict[str, Any] | None = None,
        _submit_waiting: bool = True,
        _allow_reset: bool = True,
    ) -> dict[str, Any]:
        """Import completed Batch API embeddings into ChromaDB.

        ``_skip_lock`` is for callers that already hold the update lock via
        ``update_database``; re-acquiring it would self-deadlock.

        The lock covers the whole import, status refresh included. A manifest
        read before the lock and saved after could undo what another process
        did in between, down to losing a submitted chunk's batch id, which would
        then be paid for again.

        ``_collect_run`` is the unattended path, from
        :meth:`_collect_unfinished_batches`: it names the run to work on, and a
        rebuild that isn't ready is reported as ``deferred``, not raised, with
        ``needs_user`` set when only the user can move it on.
        ``_submit_waiting=False`` leaves throttle-parked chunks unsent.
        ``_allow_reset=False`` stops at a rebuild's reset and asks for 'batch-import'.

        With no batch ids, a run that a later sync overtook, that another model
        made or that the user gave up on is not imported. Its text would be older
        than the index, or its vectors unmatchable, so that takes naming it.
        """
        lock_path = _update_lock_path()
        lock_cm = contextlib.nullcontext(True) if _skip_lock else _acquire_update_lock(lock_path)
        acquired = lock_cm.__enter__()
        if not acquired:
            lock_cm.__exit__(None, None, None)
            raise RuntimeError(f"Another semantic-search update is already running (lock held at {lock_path})")
        manifest = None
        try:
            module = _batch_module(provider)
            adapter = _batch_adapter(provider)
            label = adapter.label
            selected_ids = set(batch_ids or [])
            manifest = _collect_run or self._find_run(provider, next(iter(selected_ids), None))
            if not selected_ids and not self._run_is_usable(provider, manifest):
                return {
                    "provider": provider,
                    "run_id": manifest.get("run_id"),
                    "manifest_path": manifest.get("manifest_path"),
                    "imported_items": 0,
                    "batches_submitted": 0,
                    "errors": [{"error": (
                        f"Nothing to import: run {manifest.get('run_id')} is older than the index, was cut "
                        "before the index was last rebuilt, was made with another embedding model, or was "
                        "given up on. To import it anyway, name its batches with --batch-id; that never "
                        "moves the sync version."
                    )}],
                }
            # Whether something newer has taken this run's items over decides two
            # things below: no more chunks may be submitted for it, and its sync
            # watermark must stay where it is. Settled once here, before the
            # refresh, so every decision in this import agrees.
            superseded_by = self._superseded_by(provider, manifest)
            overtaken_at = self._overtaken_by_sync(manifest)
            taken_over = superseded_by or (None if overtaken_at is None else f"library version {overtaken_at}")
            manifest = module.refresh_manifest_status(
                manifest,
                embedding_config=self.chroma_client.embedding_config,
                batch_ids=selected_ids or None,
                # Asking for an import is asking again about chunks a 404 took
                # out of the unattended path.
                recheck_lost=_collect_run is None,
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
            # Items whose records this import wrote, and items with a record left
            # unwritten: a run taken over never promotes, so it settles its
            # kept items here.
            landed_keys: set[str] = set()
            unlanded_keys: set[str] = set()
            if superseded_by:
                # One entry, recorded once, covering both consequences.
                stats["errors"].append({"error": (
                    f"run {manifest.get('run_id')} is superseded by run {superseded_by}: its pending "
                    "chunks are not submitted and its sync watermark is not promoted, because the "
                    "newer run covers the same items"
                )})
            elif overtaken_at is not None:
                stats["errors"].append({"error": (
                    f"run {manifest.get('run_id')} is superseded by an update that reached library "
                    f"version {overtaken_at}: its pending chunks are not submitted and its sync "
                    "watermark is not promoted"
                )})

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

            # A failed or expired file goes round again, or is given up on so
            # the rest of the run can land. Not for named batches, which don't
            # carry the run forward, nor for a run taken over by a newer one.
            if not selected_ids and not taken_over:
                _requeued, abandoned = batch_common.requeue_failed_batches(adapter, manifest)
                stats["batches_abandoned"] = abandoned

            if manifest.get("force_full_rebuild"):
                def _incomplete() -> list[str]:
                    return [
                        batch.get("batch_id") or "(pending)"
                        for batch in all_batches
                        if not batch_common.is_settled(batch)
                        and not batch_common.is_importable(adapter, batch)
                    ]

                if _incomplete():
                    # This run imports all at once, later than a throttled
                    # run's first results may stay available, so fetch them now.
                    self._download_finished_batches(adapter, client, all_batches)
                    # A force-rebuild run imports all-or-nothing. Before
                    # refusing, submit whatever the throttle parked: that is
                    # the progress the user is waiting for, not an error. The
                    # submission runs under the update lock (unlike the refusal
                    # it replaces), since it writes the manifest.
                    submitted = 0
                    if not selected_ids and _submit_waiting:
                        submitted = self._submit_pending_chunks(provider, manifest, client, taken_over)
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
                        if _collect_run is not None:
                            stats["deferred"] = message
                            return stats
                        raise RuntimeError(message)

            def _output_path(batch: dict[str, Any]) -> Path:
                records_path = Path(batch["records_path"])
                return records_path.with_name(records_path.stem + "-output.jsonl")

            # A forced rebuild replaces the index only when the whole run is here.
            # With a file given up on, or nothing at all to import, what arrived
            # is written over the existing index instead, which keeps the rest.
            if (
                manifest.get("force_full_rebuild")
                and not superseded_by
                and not any(batch.get("imported_at") or batch.get("abandoned") for batch in all_batches)
                and any(not batch.get("imported_at") for batch in batches)
            ):
                run_id = manifest.get("run_id")
                if not _allow_reset:
                    stats["deferred"] = (
                        f"Rebuild run {run_id} is ready. Run 'zotero-mcp batch-import' to replace the "
                        "index with it."
                    )
                    stats["needs_user"] = True
                    return stats
                if _collect_run is not None and not manifest.get("allow_mass_deletion"):
                    # The reset drops the whole ChromaDB collection. The user agreed
                    # to what it held when they asked for the rebuild, not to what
                    # has been indexed from elsewhere since; an unattended import
                    # won't guess, and not knowing counts as not agreed.
                    try:
                        foreign = self.chroma_client.count_outside_library(int(manifest["group_id"]))
                    except Exception as e:
                        logger.debug("Could not count documents outside the library: %s", e)
                        stats["deferred"] = (
                            f"Rebuild run {run_id} is ready, but it could not be checked for other "
                            f"libraries' documents ({type(e).__name__}). The next update tries again; "
                            "'zotero-mcp batch-import' imports it anyway."
                        )
                        stats["needs_user"] = True
                        return stats
                    if foreign:
                        stats["deferred"] = (
                            f"Rebuild run {run_id} is ready, but importing it replaces the "
                            f"whole ChromaDB collection, which now holds {foreign} document(s) from "
                            "other libraries or with no library. Run 'zotero-mcp batch-import' to "
                            "import it anyway."
                        )
                        stats["needs_user"] = True
                        return stats
                # Every file in hand before the index is emptied: a download that
                # fails now costs nothing, and after the reset it would leave an
                # empty index behind.
                for batch in batches:
                    if not batch_common.is_settled(batch) and not _output_path(batch).exists():
                        try:
                            adapter.download_output(client, batch, _output_path(batch))
                        except Exception as e:
                            if not batch_common.is_not_found(e):
                                raise
                            batch["lost"] = True
                            module.save_manifest(manifest)
                            message = (
                                f"Rebuild run {run_id} can't finish: the provider no longer has the "
                                f"output of {batch.get('batch_id')}. The index is unchanged."
                            )
                            if _collect_run is not None:
                                stats["deferred"] = message
                                return stats
                            raise RuntimeError(message) from e
                # On record before the collection is emptied, in the run and for
                # the library: a crash right after the reset leaves an empty
                # index whose run has no chunk in yet, and giving that run up
                # must still know the index is gone.
                manifest["reset_at"] = batch_common._utc_now()
                module.save_manifest(manifest)
                self._save_last_rebuild(
                    str(manifest["group_id"]) if manifest.get("group_id") is not None else self._active_library_key(),
                    manifest["reset_at"],
                )
                self.chroma_client.reset_collection()

            for batch in batches:
                if batch.get("imported_at"):
                    stats["batches_skipped"] += 1
                    continue
                if not batch.get("batch_id") or batch.get("abandoned"):
                    # Parked by the enqueued-token throttle; submitted below,
                    # after this pass imports what has completed. Or given up on.
                    stats["batches_skipped"] += 1
                    continue
                if batch.get("lost"):
                    stats["batches_skipped"] += 1
                    stats["errors"].append({
                        "batch_id": batch.get("batch_id"), "error": "The provider no longer has this batch",
                    })
                    continue
                # An expired, cancelled or failed batch can still leave a file of
                # rows that are paid for. Those go in, but not as a whole chunk.
                complete = batch_common._entry_state(adapter, batch) in batch_common.IMPORTABLE_STATES
                if not complete and not batch_common.entry_has_results(adapter, batch):
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
                output_path = _output_path(batch)
                if output_path.exists():
                    output_text = output_path.read_text(encoding="utf-8")
                else:
                    try:
                        output_text = adapter.download_output(client, batch, output_path)
                    except Exception as e:
                        if not batch_common.is_not_found(e):
                            raise
                        batch["lost"] = True
                        module.save_manifest(manifest)
                        stats["batches_skipped"] += 1
                        stats["errors"].append({
                            "batch_id": batch.get("batch_id"), "error": "The provider no longer has this file",
                        })
                        continue

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
                    # its article record or first passage. An item's passages
                    # can be spread over several files.
                    first_records = {
                        doc_id.split("#", 1)[0]
                        for doc_id in ids
                        if "#" not in doc_id or doc_id.endswith("#0")
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
                        doc_id.split("#", 1)[0] for doc_id in ids if records[doc_id].get("level_fill")
                    }
                    self._prune_stale_chunks(metadatas)
                    stats["imported_items"] += len(ids)
                    stats["updated_items"] += len(existing_items)
                    stats["added_items"] += len(first_records) - len(existing_items)
                    # Same contract as the realtime path: the embeddings are
                    # in ChromaDB, so the transient copy of their extracted
                    # text has done its job. Without this the cache grows to
                    # hold the whole library on the batch flow, since nothing
                    # else on it ever evicts. Ids are ``<key>`` or
                    # ``<key>#<n>``, so strip any chunk suffix first.
                    try:
                        fulltext_cache.evict_many(
                            {doc_id.split("#", 1)[0] for doc_id in ids},
                            config_path=self.config_path,
                        )
                    except Exception as e:
                        logger.debug(f"Fulltext cache eviction failed: {e}")

                # Rows with no vector go round again on their own, which also
                # covers an expired batch's unfinished rows. A named-batch import
                # does not carry the run forward, and a superseded run's items
                # belong to the newer one.
                unfinished = [
                    records[doc_id]
                    for doc_id in records
                    if doc_id not in embeddings_by_id
                ]
                landed_keys.update(doc_id.split("#", 1)[0] for doc_id in ids)
                unlanded_keys.update(record["id"].split("#", 1)[0] for record in unfinished)
                if selected_ids or superseded_by:
                    batch_common.add_failed_ids(manifest, [record["id"] for record in unfinished])
                else:
                    batch_common.add_retry_batch(
                        adapter, manifest, unfinished, int(batch.get("attempts") or 1)
                    )

                batch["imported_at"] = datetime.now().isoformat()
                batch["imported_count"] = len(ids)
                if not complete:
                    batch["imported_partial"] = True
                stats["batches_imported"] += 1
                # Saved per file: at this scale an interrupted import must not
                # start again (and a rebuild cut off part way must resume, not
                # empty the index again), and the files below are about to go.
                module.save_manifest(manifest)
                batch_common.remove_batch_files(batch)

            module.save_manifest(manifest)

            if taken_over and (written := (landed_keys - unlanded_keys) & set(manifest.get("retry_keys") or [])):
                # In the index now, though the run never moves the sync version.
                self._settle_retry_keys(written=written, failed=set(), keep_failed=False)
                self._save_update_config()

            if not selected_ids and _submit_waiting:
                # Importing the completed batches freed enqueued-token headroom,
                # so this is where the chunks the throttle parked get their turn.
                # An import of named batch ids never does this: the user asked
                # for those batches, not for the run to be carried forward.
                stats["batches_submitted"] += self._submit_pending_chunks(provider, manifest, client, taken_over)

            # Newly submitted chunks have no ``imported_at``, so the watermark
            # stays where it is until the whole run has landed.
            if all(batch_common.is_settled(batch) for batch in all_batches):
                failed_ids = manifest.get("failed_ids") or []
                given_up: list[str] = []
                covered = bool(superseded_by) or overtaken_at is not None or self._predates_reset(manifest)
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
                elif (
                    overtaken_at is None
                    and not manifest.get("promoted_at")
                    # Cut against an index that has since been replaced: its
                    # target says nothing about what the index holds now.
                    and not self._predates_reset(manifest)
                ):
                    given_up = self._promote_run(
                        provider,
                        manifest,
                        _run_replaced_index(manifest) and stats["batches_imported"] > 0,
                    )
                if failed_ids and not {doc_id.split("#", 1)[0] for doc_id in failed_ids} <= set(given_up):
                    stats["errors"].append({"error": self._failed_records_note(manifest, failed_ids, covered)})
                if given_up:
                    stats["retry_given_up"] = self._given_up_message(given_up)
                    stats["errors"].append({"error": stats["retry_given_up"]})
            return stats
        finally:
            # The user acted on this run, so a note asking them to (from a
            # background collect) mustn't stay on every search. A collect
            # writes the whole notice itself when its pass ends.
            if _collect_run is None and manifest is not None:
                group_id = manifest.get("group_id")
                self._refresh_batch_notice(
                    provider,
                    int(group_id) if group_id is not None else self._pinned_group_id(),
                    str(manifest.get("run_id")),
                )
            self._drop_passages_on_write = False
            self._level_fill_keys = frozenset()
            lock_cm.__exit__(None, None, None)

    @staticmethod
    def _download_finished_batches(adapter: Any, client: Any, batches: list[dict[str, Any]]) -> None:
        """Save finished batches' results locally ahead of their import."""
        for batch in batches:
            if batch.get("imported_at") or not batch.get("batch_id"):
                continue
            if not batch_common.is_importable(adapter, batch):
                continue
            if adapter.uses_error_file and not batch.get("output_file_id"):
                continue
            output_path = batch_common.output_path_for(batch)
            if output_path.exists():
                continue
            try:
                adapter.download_output(client, batch, output_path)
            except Exception as e:
                logger.warning(f"Could not download batch {batch.get('batch_id')} yet: {e}")

    def import_openai_batch(self, batch_ids: list[str] | None = None) -> dict[str, Any]:
        """Import completed OpenAI Batch API embeddings into ChromaDB."""
        return self._import_batch("openai", batch_ids)

    def import_gemini_batch(self, batch_ids: list[str] | None = None) -> dict[str, Any]:
        """Import completed Gemini Batch API embeddings into ChromaDB."""
        return self._import_batch("gemini", batch_ids)

    def _wait_for_blocking_runs(
        self, provider: str, new_run: dict[str, Any], poll_interval: int, stats: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Drive the runs that hold this update up to the end, one at a time.

        For ``--auto-loop``: the run worked on is the one 'batch-import' would
        pick. Stops when nothing holds the update up any more, or when a run
        can make no more progress. Returns the runs still in the way, oldest
        first. Called with the update lock held.
        """
        def in_the_way() -> list[dict[str, Any]]:
            return [
                run for run in self._blocking_batch_runs(provider, self._run_group_id)
                if not batch_common.may_supersede(new_run, run)
            ]

        waited = {"runs": [], "polls": 0, "imported_items": 0, "submitted_chunks": 0}
        stats["batch_waited"] = waited
        blocking = in_the_way()
        # Each pass finishes a run or stops at one that can't go on, so this
        # many passes is enough for every run there is.
        for _ in range(len(self._unfinished_batch_runs(provider, self._run_group_id))):
            if not blocking:
                break
            _report(
                f"\nWaiting for batch run {blocking[-1].get('run_id')} to finish before updating "
                "(--auto-loop).\n"
            )
            loop = self.auto_loop_batch_pipeline(provider, poll_interval=poll_interval)
            waited["runs"].append(loop.get("run_id"))
            for key in ("polls", "imported_items", "submitted_chunks"):
                waited[key] += loop.get(key, 0)
            blocking = in_the_way()
            if loop.get("stalled"):
                waited["stalled"] = loop["stalled"]
                break
        return blocking

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
        step, so after Ctrl-C or a crash a later ``batch-import`` or ``update-db``
        imports what completed and submits what is still pending.

        Must be called with the update lock already held (``update_database``
        holds it), hence ``_skip_lock`` on the imports below.
        """
        import time

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

            manifest = self._find_run(provider)
            aggregate["run_id"] = manifest.get("run_id")
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
            aggregate["abandoned"] = sum(1 for b in entries if b.get("abandoned"))
            remaining = [b for b in entries if not batch_common.is_settled(b)]
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
            filters: Optional metadata filters
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
            search_level, chunked, level_note, level_error = self._resolve_search_level(level)
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
            fetch_limit = limit
            if chunked:
                # Passages are grouped back to items downstream, so fetch
                # several chunks per desired item to still surface ~limit
                # distinct papers.
                fetch_limit = max(fetch_limit, limit * 4)
            if reranker:
                fetch_limit = max(fetch_limit, limit * multiplier)

            where = filters
            if group_id is not None:
                group_clause = {"group_id": int(group_id)}
                where = {"$and": [filters, group_clause]} if filters else group_clause

            # Perform semantic search
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

            _drop_missing_documents(results)

            rerank_info: dict[str, Any] | None = None
            if reranker and results.get("documents") and results["documents"][0]:
                documents = results["documents"][0]
                # Chunked: rank every surviving passage (grouping to `limit`
                # papers happens in enrichment). Item-level: the historical
                # top-k = limit.
                top_k = len(documents) if chunked else limit
                if isinstance(reranker, (APIReranker, CrossEncoderReranker)):
                    pairs, rerank_error = reranker.rerank_detailed(query, documents, top_k)
                else:  # any other object honouring the plain rerank() contract
                    pairs = [(i, 0.0) for i in reranker.rerank(query, documents, top_k=top_k)]
                    rerank_error = None
                ranked_indices = [i for i, _ in pairs]
                for key in ["ids", "distances", "documents", "metadatas"]:
                    if results.get(key) and results[key][0]:
                        results[key][0] = [results[key][0][i] for i in ranked_indices]
                if rerank_error is None and isinstance(reranker, (APIReranker, CrossEncoderReranker)):
                    results["rerank_scores"] = [[score for _, score in pairs]]
                rerank_info = {
                    "model": getattr(reranker, "label", type(reranker).__name__),
                    "applied": rerank_error is None,
                    "error": rerank_error,
                    "candidates": len(documents),
                }

            # Enrich results with full Zotero item data, grouping passages back
            # to their parent items and capping at `limit` distinct papers.
            enriched_results = self._enrich_search_results(results, query, limit)

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
            }

        except EmbeddingAPIUnreachable as e:
            # The index is local, but the query still has to be embedded by
            # the API. Never fall back to another provider: its vectors would
            # not match the index.
            logger.error(f"Error performing semantic search: {e}")
            return {
                "query": query,
                "limit": limit,
                "filters": filters,
                "group_id": group_id,
                "results": [],
                "total_found": 0,
                "error": str(e),
                "embedding_api_unreachable": True,
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
            }

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
        """Tell the caller what was searched and what the other level holds.

        A passage fill can take days, and until it ends passages cover only part
        of the library. An agent can't see that from the results, so it gets the
        counts and can decide to search the other level. The counts are for the
        whole index, whichever library was searched. Only an index with articles
        alone and chunking off, or one still unmigrated, gets no note; an index
        with passages alone is told no article records are indexed yet.
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
            note += f" The {other[:-1]} level has {_count_items(elsewhere)}"
            return f"{note}, so searching it may find more." if elsewhere > searched else f"{note}."
        if level == SEARCH_PASSAGES:
            return f"{note} No article records are indexed yet."
        return f"{note} No passages are indexed yet." if self._chunking_enabled else None

    def _enrich_search_results(
        self, chroma_results: dict[str, Any], query: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Enrich ChromaDB results with full Zotero item data.

        Chunk-aware: when the collection is indexed as passages, ids look like
        ``<item_key>#<n>``. Results are grouped back to their parent item — the
        first (best-ranked) passage per item wins — and capped at ``limit``
        distinct items. For every hit a grounded ``matched_passage`` quote and,
        when available, the passage's character offset and page are attached so
        callers can cite precisely. Item-level collections (ids without ``#``)
        flow through unchanged.
        """
        enriched: list[dict[str, Any]] = []

        if not chroma_results.get("ids") or not chroma_results["ids"][0]:
            return enriched

        ids = chroma_results["ids"][0]
        distances = chroma_results.get("distances", [[]])[0]
        documents = chroma_results.get("documents", [[]])[0]
        metadatas = chroma_results.get("metadatas", [[]])[0]
        rerank_scores = (chroma_results.get("rerank_scores") or [[]])[0] or []

        seen_items: set[str] = set()
        for i, raw_id in enumerate(ids):
            item_key = raw_id.split("#", 1)[0]
            if item_key in seen_items:
                continue
            seen_items.add(item_key)

            distance = distances[i] if i < len(distances) else None
            document = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}

            passage, passage_offset = best_snippet(query, document)

            enriched_result: dict[str, Any] = {
                "item_key": item_key,
                "similarity_score": (1 - distance) if distance is not None else 0,
                "matched_text": document,
                "matched_passage": passage,
                "metadata": meta if isinstance(meta, dict) else {},
                "query": query,
            }
            if i < len(rerank_scores):
                enriched_result["rerank_score"] = rerank_scores[i]
            # Passage provenance — present only on a chunk-indexed collection.
            if isinstance(meta, dict):
                for mk in ("chunk_index", "n_chunks", "char_start", "char_end", "page", "section"):
                    if mk in meta:
                        enriched_result[mk] = meta[mk]
            if "char_start" not in enriched_result and passage_offset:
                enriched_result["passage_offset"] = passage_offset

            enriched.append(enriched_result)
            if limit and len(enriched) >= limit:
                break

        self._attach_zotero_items(enriched)
        return enriched

    def _attach_zotero_items(self, enriched: list[dict[str, Any]]) -> None:
        """Fill in each result's ``zotero_item``, in place.

        Hits from the library ``self.zotero_client`` is scoped to are fetched
        through it, as before. Hits from any *other* library cannot be: the
        client is bound to one library and a foreign key simply 404s, which is
        why a group-library paper was found by semantic search and then
        reported as an error rather than a result (#163). Those are hydrated
        from ``zotero.sqlite`` instead, in one batched query, and arrive
        already carrying their ``library`` attribution.

        Where the local database cannot serve a foreign hit — every web-API
        install, since there is no ``zotero.sqlite`` there — a client scoped
        to the hit's *own* library is tried instead (#492). Failures on that
        path name the library and the credential requirement, because the
        alternative is the silent 404 this exists to remove.

        A document whose ``group_id`` is missing cannot be recognised as
        foreign up front — every index built before #396 is untagged, which
        is most of them. Those go to the client first and fall back to the
        local database when it cannot serve them, so the fix does not depend
        on having re-indexed.
        """
        try:
            client_group_id = self._client_group_id()
        except ValueError:
            client_group_id = None

        def _is_foreign(result: dict[str, Any]) -> bool:
            group_id = (result.get("metadata") or {}).get("group_id")
            if group_id is None or client_group_id is None:
                return False
            return int(group_id) != int(client_group_id)

        already_tried = {r["item_key"] for r in enriched if _is_foreign(r)}
        local_items = self._hydrate_locally(sorted(already_tried))

        unresolved: list[dict[str, Any]] = []
        for result in enriched:
            item_key = result["item_key"]
            if item_key in local_items:
                result["zotero_item"] = local_items[item_key]
                continue
            if _is_foreign(result):
                # Local hydration came up empty for a hit known to be
                # foreign. The bound client cannot serve it — that 404 is
                # the #492 symptom — so ask a client scoped to the hit's own
                # library instead. Outside local mode this is the only path
                # that can serve the hit at all.
                self._fetch_via_scoped_client(result)
                continue
            try:
                result["zotero_item"] = self.zotero_client.item(item_key)
            except Exception as e:
                result["_enrich_error"] = e
                unresolved.append(result)

        # Second chance for anything the client could not serve: on an
        # untagged index that is exactly how a foreign hit presents itself.
        # Keys the pass above already looked up are skipped — they are known
        # to be absent, and asking twice cannot change that.
        retry = [r for r in unresolved if r["item_key"] not in already_tried]
        if retry:
            recovered = self._hydrate_locally([r["item_key"] for r in retry])
            for result in retry:
                item = recovered.get(result["item_key"])
                if item is not None:
                    result["zotero_item"] = item
                    result.pop("_enrich_error", None)

        for result in enriched:
            error = result.pop("_enrich_error", None)
            if error is not None:
                logger.error(
                    f"Error enriching result for item {result['item_key']}: {error}"
                )
                result["error"] = f"Could not fetch full item data: {error}"

    def _fetch_via_scoped_client(self, result: dict[str, Any]) -> None:
        """Hydrate one foreign hit through a client scoped to its library.

        Reached only when ``_hydrate_locally`` could not serve a hit whose
        ``group_id`` names a library other than the bound client's. Fills in
        ``zotero_item`` on success; on failure records an ``_enrich_error``
        that names the library and, where relevant, the credential
        requirement — a foreign hit must never surface as a bare 404 from a
        library that was never going to have it.
        """
        group_id = int((result.get("metadata") or {})["group_id"])
        client = self._client_for_group(group_id)
        if client is None:
            result["_enrich_error"] = RuntimeError(
                f"item belongs to library {group_id}, which cannot be "
                "reached from this configuration — no local Zotero "
                "database, and the web API needs ZOTERO_API_KEY plus, for "
                "the personal library, a user-scoped ZOTERO_LIBRARY_ID"
            )
            return
        try:
            result["zotero_item"] = client.item(result["item_key"])
        except Exception as e:
            result["_enrich_error"] = RuntimeError(
                f"item belongs to library {group_id}, and the configured "
                f"credentials could not read it from there: {e}"
            )

    def _client_for_group(self, group_id: int):
        """A pyzotero client scoped to the library ``group_id`` names, or None.

        The per-library complement of ``_hydrate_locally`` (#492): the bound
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

    def _hydrate_locally(self, keys: list[str]) -> dict[str, dict]:
        """Hydrate `keys` from zotero.sqlite, or {} if that is not possible."""
        if not keys:
            return {}
        try:
            reader = self._open_local_reader()
        except Exception as e:
            logger.debug(f"Cross-library enrichment: no local database ({e})")
            return {}
        if reader is None:
            return {}
        try:
            return reader.get_items_by_keys(keys)
        except Exception as e:
            logger.warning(f"Cross-library enrichment failed: {e}")
            return {}
        finally:
            reader.close()

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
            "batch_notice": self.batch_notice(),
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
