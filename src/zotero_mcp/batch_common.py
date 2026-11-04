"""Provider-neutral helpers shared by the OpenAI and Gemini Batch API modules.

Everything here is independent of the provider SDK: JSONL I/O, local run
manifests (which tie asynchronous batch jobs back to the document text and
metadata needed for ChromaDB), and the record-splitting algorithm that packs
embedding requests into Batch-API-sized chunk files.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from zotero_mcp.utils import write_json_atomic

# ---------------------------------------------------------------------------
# Normalized batch-job state vocabulary
#
# Providers expose incompatible raw status vocabularies (OpenAI's lowercase
# strings vs Gemini's ``JOB_STATE_*`` enum names). Manifests keep the raw
# ``status`` verbatim (on-disk format/CLI output unchanged) and additionally
# carry a normalized ``state`` so decision logic (import-eligibility,
# terminal-ness, enqueued-token accounting) is provider-agnostic.
# ---------------------------------------------------------------------------

STATE_PENDING = "pending"
STATE_SUBMITTED = "submitted"
STATE_IN_PROGRESS = "in_progress"
STATE_SUCCEEDED = "succeeded"
STATE_PARTIAL = "partial"
STATE_FAILED = "failed"
STATE_EXPIRED = "expired"
STATE_CANCELLED = "cancelled"

IMPORTABLE_STATES = frozenset({STATE_SUCCEEDED, STATE_PARTIAL})
TERMINAL_STATES = frozenset({STATE_SUCCEEDED, STATE_PARTIAL, STATE_FAILED, STATE_EXPIRED, STATE_CANCELLED})

# How long past a provider's completion window a run may hold up other work.
WINDOW_MARGIN_HOURS = 24

# Times a record may be sent before it is given up on. A large run spans many
# files over days, so one failure must not stop it, and one that keeps failing
# must not hold it up for ever.
MAX_BATCH_ATTEMPTS = 3


@runtime_checkable
class BatchAdapter(Protocol):
    """Provider-specific surface the generic batch flows below are driven by.

    Everything a new Batch-API-capable provider must implement: request
    shaping, submission/status SDK calls, status-string normalization, and
    output parsing. Manifest bookkeeping, chunk-loop budgeting, pending-chunk
    parking, and token accounting are provider-agnostic and live once in this
    module's ``submit_embedding_batches``/``submit_pending_batches``/
    ``refresh_manifest_status``, parameterized by an adapter instance.
    """

    provider: str
    label: str
    default_model: str
    max_requests: int
    max_file_bytes: int
    default_max_enqueued_tokens: int
    chars_per_token: float
    uses_error_file: bool
    completion_window_hours: int

    def create_client(self, embedding_config: dict[str, Any] | None) -> Any:
        """Build an SDK client from semantic embedding config + environment."""
        ...

    def build_request(self, record: dict[str, Any], model_name: str) -> dict[str, Any]:
        """Build one JSONL request line/object for this provider's batch endpoint."""
        ...

    def submit_chunk(
        self, client: Any, input_path: Path, run_id: str, index: int, model_name: str
    ) -> dict[str, Any]:
        """Upload one chunk file and create its batch job; returns manifest fields
        (including the raw ``status``)."""
        ...

    def retrieve_status(self, client: Any, batch_entry: dict[str, Any]) -> dict[str, Any]:
        """Fetch current status for one submitted batch; returns updated raw
        fields (``status`` plus provider-specific output/destination fields)."""
        ...

    def normalize_status(self, raw_status: str | None) -> str:
        """Map a raw provider status string to the normalized vocabulary above."""
        ...

    def download_output(self, client: Any, batch_entry: dict[str, Any], output_path: Path) -> str:
        """Download a completed batch's output and persist it locally, returning the text."""
        ...

    def parse_output(
        self, text: str, id_order: list[str]
    ) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
        """Parse a downloaded output file into ``(embeddings_by_id, failures)``."""
        ...

    def download_errors(
        self, client: Any, batch_entry: dict[str, Any], output_path: Path
    ) -> list[dict[str, Any]]:
        """Download/parse this batch's error rows; ``[]`` for providers with no
        separate error file (``uses_error_file is False``)."""
        ...


def _private_chmod(path: Path) -> None:
    try:
        os.chmod(path, 0o600 if path.is_file() else 0o700)
    except OSError:
        pass


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _object_attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _utc_now() -> str:
    """Fixed-width ISO-8601 UTC stamp, microseconds included.

    The microseconds are load-bearing: ``created_at`` is what orders runs
    (:func:`_run_order_key`), and two runs minted in the same second must not
    compare equal.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def estimate_tokens(text: str, chars_per_token: float = 3.0) -> int:
    """Deterministic char-based token estimate for batch quota accounting.

    Deliberately conservative (the default overestimates typical English by
    ~15-25%) so enqueued-token headroom is never overspent. Used for quota
    slicing only — never for truncation, which stays tokenizer-accurate.
    """
    return max(1, math.ceil(len(text) / chars_per_token))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(_json_dumps(row) + "\n")
    _private_chmod(path)


def write_text_atomic(path: Path, text: str) -> None:
    """Write a downloaded file all at once. The import trusts that a file which
    exists is whole, so a cut-off download must leave no file behind."""
    path = Path(path)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)
        raise
    _private_chmod(path)


def output_path_for(batch: dict[str, Any]) -> Path:
    """Where a chunk's downloaded output is kept, next to its records file."""
    records_path = Path(batch["records_path"])
    return records_path.with_name(records_path.stem + "-output.jsonl")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_manifest(manifest: dict[str, Any]) -> None:
    # Rewritten on every status refresh and import, and it is the only local
    # record of a paid provider batch. write_json_atomic writes a temp file and
    # renames it, so a crash or a full disk mid-write leaves the previous
    # manifest instead of a truncated one that _sorted_manifests then skips as
    # unreadable. The file and its directory are owner-only.
    write_json_atomic(Path(manifest["manifest_path"]), manifest)


def load_manifest(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["manifest_path"] = str(path)
    return manifest


def _run_order_key(path: Path, manifest: dict[str, Any]) -> tuple[str, str]:
    """Sort key deciding which run is newer: submission time, then run id.

    Both parts are written once, when the run is created, and never rewritten,
    so no amount of re-saving can reorder runs — and re-saving is routine:
    every status refresh and every import rewrites the manifest it was handed.
    File mtime, the obvious alternative, would therefore make "newest" mean
    "looked at most recently", which is how a superseded run used to talk its
    way back into being current. ``created_at`` is fixed-width ISO-8601 UTC, so
    it orders lexicographically; ``run_id`` (itself timestamp-prefixed) settles
    ties between runs stamped in the same microsecond.

    One caveat, stated plainly: a manifest from a release that stamped whole
    seconds (``…T10:00:00Z``) sorts *after* a microsecond stamp of that same
    second (``…T10:00:00.000000Z``), because ``"Z" > "."`` — so a pre-upgrade
    run would win a same-second race against a run submitted after the upgrade,
    and ``run_id`` never gets to settle it, the two strings being unequal. This
    is deterministic rather than arbitrary, and reaching it takes an upgrade
    plus a resubmission within the same second; runs minutes apart — every real
    pair — order correctly either way.

    A manifest missing either field falls back to the empty string / its run
    directory name, which ranks it below any run that carries them.
    """
    return (
        str(manifest.get("created_at") or ""),
        str(manifest.get("run_id") or path.parent.name),
    )


def _sorted_manifests(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Every readable run manifest under ``root``, newest first."""
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in root.glob("*/manifest.json"):
        try:
            loaded.append((path, load_manifest(path)))
        except (OSError, ValueError):
            continue  # unreadable or half-written manifest: not a candidate
    loaded.sort(key=lambda item: _run_order_key(*item), reverse=True)
    return loaded


def iter_manifests(root: Path) -> list[Path]:
    """Run manifest paths under ``root``, newest first (:func:`_run_order_key`)."""
    return [path for path, _ in _sorted_manifests(root)]


def newest_run_path(root: Path) -> str | None:
    """Manifest path of the newest run, or ``None`` when there are no runs."""
    manifests = _sorted_manifests(root)
    return str(manifests[0][0]) if manifests else None


def manifests_for_group(root: Path, group_id: int | None) -> list[dict[str, Any]]:
    """Every run submitted against ``group_id``, newest first.

    Runs are per library, and so is superseding one: a newer run for a
    different library re-embeds none of this one's items.
    """
    return [manifest for _, manifest in _sorted_manifests(root) if manifest.get("group_id") == group_id]


def run_will_never_land(manifest: dict[str, Any], adapter: BatchAdapter | None = None) -> bool:
    """A run the user gave up on, or (given ``adapter``) one that can't finish:
    nothing it was submitted for will ever reach the index through it."""
    if manifest.get("abandoned_at"):
        return True
    batches = manifest.get("batches", [])
    return bool(
        adapter is not None
        and batches
        and not run_is_unfinished(manifest, adapter)
        and not all(batch.get("imported_at") for batch in batches)
    )


def run_covers(newer: dict[str, Any], older: dict[str, Any], adapter: BatchAdapter | None = None) -> bool:
    """Does ``newer`` embed everything ``older`` was submitted for?

    The one place this is decided. ``newer`` is a manifest, or the same fields
    of a run about to be submitted. Runs carry ``limited`` (stopped at
    ``--limit`` items) and ``since_version``: the sync version an incremental
    run was cut from, 0 for a first build, ``None`` for a scan that skips what
    is already indexed and so says nothing about changed items.

    A force rebuild re-embeds the whole library, so it covers anything, unless
    it stopped at ``--limit``. Otherwise a run covers an older one only if it
    was cut from the same sync version or an earlier one. Manifests written
    before these fields existed were all assumed to be cut from one sync
    version; between one of those and a newer manifest nothing can be proved.

    A run that will never land covers nothing (:func:`run_will_never_land`):
    abandoned, or with ``adapter`` given, dead. Otherwise an older run's paid
    results would be dropped for a newer run that never delivers them.
    """
    if newer.get("run_id") == older.get("run_id") or newer.get("group_id") != older.get("group_id"):
        return False
    if run_will_never_land(newer, adapter):
        return False
    if newer.get("limited"):
        return False
    if newer.get("force_full_rebuild"):
        return True
    if older.get("force_full_rebuild") or older.get("limited"):
        return False
    if "since_version" not in newer and "since_version" not in older:
        return True
    newer_since, older_since = newer.get("since_version"), older.get("since_version")
    if newer_since is None or older_since is None:
        return False
    return newer_since <= older_since


def may_supersede(new_run: dict[str, Any], unfinished: dict[str, Any]) -> bool:
    """May ``new_run`` be submitted while ``unfinished`` is still running?

    Only a full rebuild, which the user asks for by name. An incremental run
    can cover an unfinished one, but it would pay again for every item in it
    and throw away results already being paid for, when waiting gains the same
    index at no cost: its changes are still there after the import. On a
    schedule, submitting each time would also restart the clock on a run that
    takes up to a day, so that with steady changes it never finishes.
    """
    return bool(new_run.get("force_full_rebuild")) and run_covers(new_run, unfinished)


def is_not_found(error: Exception) -> bool:
    """Is this the provider saying it doesn't have the batch or file (HTTP 404)?

    OpenAI's SDK carries the code as ``status_code``, Gemini's as ``code``.
    """
    return 404 in (getattr(error, "status_code", None), getattr(error, "code", None))


def _entry_is_complete(adapter: BatchAdapter, batch: dict[str, Any]) -> bool:
    """The provider finished every row of this chunk, and there's a file to read."""
    if batch.get("lost") or _entry_state(adapter, batch) not in IMPORTABLE_STATES:
        return False
    # A batch whose every row failed completes with only an error file.
    return not (adapter.uses_error_file and batch.get("batch_id") and not batch.get("output_file_id"))


def entry_has_results(adapter: BatchAdapter, batch: dict[str, Any]) -> bool:
    """There are paid rows to import: a complete chunk, or the file an expired,
    cancelled or failed one left behind."""
    if _entry_is_complete(adapter, batch):
        return True
    return bool(
        not batch.get("lost")
        and _entry_state(adapter, batch) in TERMINAL_STATES
        and adapter.uses_error_file
        and batch.get("output_file_id")
    )


def entry_will_requeue(adapter: BatchAdapter, batch: dict[str, Any]) -> bool:
    """A failed or expired chunk with nothing to import: the next import sends
    it again, or gives it up after ``MAX_BATCH_ATTEMPTS``
    (:func:`requeue_failed_batches`)."""
    return bool(
        batch.get("batch_id")
        and not batch.get("lost")
        and not is_settled(batch)
        and _entry_state(adapter, batch) in (STATE_FAILED, STATE_EXPIRED)
        and not entry_has_results(adapter, batch)
    )


def _entry_is_dead(adapter: BatchAdapter, batch: dict[str, Any]) -> bool:
    """Nothing can ever be imported from this chunk, and it won't be sent again."""
    if batch.get("lost"):
        return True
    if entry_will_requeue(adapter, batch):
        return False
    return _entry_state(adapter, batch) in TERMINAL_STATES and not entry_has_results(adapter, batch)


def run_is_unfinished(manifest: dict[str, Any], adapter: BatchAdapter) -> bool:
    """Can this run still make progress: an import, a chunk to submit, or a
    failed chunk to send again or give up on?

    A run left with only dead chunks is finished: its sync version was never
    promoted, so the next update covers those items again. So is a rebuild with
    any dead chunk, because it can't replace the index without it. A run the
    user gave up on is finished too.
    """
    if manifest.get("abandoned_at"):
        return False
    remaining = [batch for batch in manifest.get("batches", []) if not is_settled(batch)]
    if manifest.get("force_full_rebuild") and any(_entry_is_dead(adapter, batch) for batch in remaining):
        return False
    return any(not _entry_is_dead(adapter, batch) for batch in remaining)


def _parse_stamp(stamp: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def run_past_window(manifest: dict[str, Any], adapter: BatchAdapter) -> bool:
    """Has the provider had its full window for everything this run sent?

    Counted from the last chunk sent, not from the start of the run: a throttled
    run sends its chunks one after another, for days if it's large. Past this,
    the provider has settled or expired them all, so a run still unfinished is
    one that nothing is coming from unless someone asks.
    """
    sent = [batch["submitted_at"] for batch in manifest.get("batches", []) if batch.get("submitted_at")]
    stamps = [_parse_stamp(stamp) for stamp in [*sent, manifest.get("created_at")]]
    stamps = [stamp for stamp in stamps if stamp]
    if not stamps:
        return True
    limit = timedelta(hours=adapter.completion_window_hours + WINDOW_MARGIN_HOURS)
    return datetime.now(timezone.utc) - max(stamps) > limit


def find_manifest(
    root: Path,
    batch_id: str | None = None,
    provider_label: str = "batch",
    group_id: int | None = None,
    adapter: BatchAdapter | None = None,
    is_current: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """Find the manifest that contains a batch ID, or else the run to work on.

    Given ``group_id`` and ``adapter`` that is the newest unfinished run of
    that library no newer run covers, so that neither another library's run nor
    a small newer run can hide it; then the library's newest run. ``is_current``
    rules out runs the caller knows are of no use, which then count only when
    nothing else is left. Otherwise, and when the library has no runs, it is the
    newest run overall.
    """
    manifests = [manifest for _, manifest in _sorted_manifests(root)]
    if batch_id is not None:
        for manifest in manifests:
            if any(batch.get("batch_id") == batch_id for batch in manifest.get("batches", [])):
                return manifest
        raise FileNotFoundError(f"No {provider_label} manifest found for batch {batch_id}")
    if group_id is not None and adapter is not None:
        ours = [manifest for manifest in manifests if manifest.get("group_id") == group_id]
        current = [manifest for manifest in ours if is_current is None or is_current(manifest)]
        for manifest in current:
            newer = ours[:ours.index(manifest)]
            if run_is_unfinished(manifest, adapter) and not any(run_covers(run, manifest, adapter) for run in newer):
                return manifest
        if current:
            return current[0]
        if ours:
            return ours[0]
    if manifests:
        return manifests[0]
    raise FileNotFoundError(f"No {provider_label} manifests found")


def split_embedding_records(
    records: list[dict[str, Any]],
    build_request: Callable[[dict[str, Any]], dict[str, Any]],
    max_requests: int,
    max_file_bytes: int,
    provider_label: str = "batch",
    max_tokens: int | None = None,
    chars_per_token: float = 3.0,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Split records into JSONL-sized chunks accepted by the Batch API.

    ``max_tokens`` optionally caps the *estimated* enqueued tokens per chunk
    file (quota throttling); ``None`` disables token-based splitting. Callers
    that need a chunk's token total can recompute it from the returned records
    with :func:`estimate_tokens`.
    """
    chunks: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    current_records: list[dict[str, Any]] = []
    current_requests: list[dict[str, Any]] = []
    current_bytes = 0
    current_tokens = 0

    for record in records:
        request = build_request(record)
        line_bytes = len((_json_dumps(request) + "\n").encode("utf-8"))
        if line_bytes > max_file_bytes:
            raise ValueError(f"{provider_label} batch request for {record['id']} exceeds the file size limit")
        record_tokens = estimate_tokens(record["document"], chars_per_token)

        would_exceed_count = len(current_requests) >= max_requests
        would_exceed_bytes = current_requests and current_bytes + line_bytes > max_file_bytes
        would_exceed_tokens = (
            max_tokens is not None and current_requests and current_tokens + record_tokens > max_tokens
        )
        if would_exceed_count or would_exceed_bytes or would_exceed_tokens:
            chunks.append((current_records, current_requests))
            current_records = []
            current_requests = []
            current_bytes = 0
            current_tokens = 0

        current_records.append(record)
        current_requests.append(request)
        current_bytes += line_bytes
        current_tokens += record_tokens

    if current_requests:
        chunks.append((current_records, current_requests))

    return chunks


def ensure_states(manifest: dict[str, Any], adapter: BatchAdapter) -> dict[str, Any]:
    """Backfill normalized ``state`` for manifest entries that predate it.

    Pre-refactor manifests on disk carry only the raw provider ``status``.
    Called by the provider modules' ``load_manifest``/``find_manifest``
    wrappers so every entry reaching decision logic has a ``state``, without
    changing ``load_manifest``'s ``(path) -> dict`` signature (still used
    directly by tests/callers with no adapter available).
    """
    for batch in manifest.get("batches", []):
        if "state" not in batch and "status" in batch:
            batch["state"] = adapter.normalize_status(batch["status"])
    return manifest


def _entry_state(adapter: BatchAdapter, batch: dict[str, Any]) -> str:
    """Normalized state for one manifest entry.

    Recomputed from the raw ``status`` whenever one is present, rather than
    trusting a cached ``state`` — ``status`` is the field every code path
    (including tests that poke it directly to simulate an SDK status change)
    keeps fresh, whereas a cached ``state`` can go stale relative to it.
    Falls back to a cached ``state`` only for the degenerate case of an entry
    with no ``status`` at all.
    """
    if "status" in batch:
        return adapter.normalize_status(batch["status"])
    return batch.get("state") or adapter.normalize_status(None)


def is_importable(adapter: BatchAdapter, batch: dict[str, Any]) -> bool:
    """Whether this entry has results to import now.

    An expired OpenAI batch still returns what finished before the window
    closed, and those rows have been charged for (:func:`entry_has_results`).
    """
    return entry_has_results(adapter, batch)


def is_settled(batch: dict[str, Any]) -> bool:
    """Whether this entry needs nothing more: imported, or given up on."""
    return bool(batch.get("imported_at") or batch.get("abandoned"))


def add_failed_ids(manifest: dict[str, Any], ids: list[str]) -> None:
    """Record ids given up on; a run with any does not advance the sync version."""
    failed = manifest.setdefault("failed_ids", [])
    failed.extend(doc_id for doc_id in ids if doc_id not in failed)


def requeue_failed_batches(adapter: BatchAdapter, manifest: dict[str, Any]) -> tuple[int, int]:
    """Park failed or expired files to be submitted again, or give up on them.

    A file with results to import is left for the import, which retries only
    the rows that failed. Returns ``(requeued, abandoned)``. Cancelled files
    stay as they are, since someone cancelled them.
    """
    requeued = abandoned = 0
    for batch in manifest.get("batches", []):
        if not entry_will_requeue(adapter, batch):
            continue
        attempts = int(batch.get("attempts") or 1)
        if attempts >= MAX_BATCH_ATTEMPTS:
            batch["abandoned"] = True
            add_failed_ids(manifest, [record["id"] for record in read_jsonl(Path(batch["records_path"]))])
            abandoned += 1
            continue
        batch["previous_batch_ids"] = [*batch.get("previous_batch_ids", []), batch["batch_id"]]
        batch.update({
            "batch_id": None,
            "status": STATE_PENDING,
            "state": STATE_PENDING,
            "attempts": attempts + 1,
        })
        requeued += 1
    if requeued or abandoned:
        save_manifest(manifest)
    return requeued, abandoned


def add_retry_batch(
    adapter: BatchAdapter,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    previous_attempts: int,
) -> bool:
    """Queue rows that failed as a new file of their own, or give up on them.

    Returns whether a retry was queued. A manifest with no enqueued-token budget
    predates throttling and could never submit the file, so its rows count as
    given up on.
    """
    if not records:
        return False
    attempts = previous_attempts + 1
    if attempts > MAX_BATCH_ATTEMPTS or manifest.get("max_enqueued_tokens") is None:
        add_failed_ids(manifest, [record["id"] for record in records])
        return False
    run_dir = Path(manifest["manifest_path"]).parent
    model_name = manifest.get("model") or adapter.default_model
    stem = f"batch-{len(manifest.get('batches', [])) + 1:03d}"
    input_path = run_dir / f"{stem}-input.jsonl"
    records_path = run_dir / f"{stem}-records.jsonl"
    write_jsonl(input_path, [adapter.build_request(record, model_name) for record in records])
    write_jsonl(records_path, records)
    manifest["batches"].append({
        "input_path": str(input_path),
        "records_path": str(records_path),
        "request_count": len(records),
        "request_tokens": sum(estimate_tokens(r["document"], adapter.chars_per_token) for r in records),
        "imported_at": None,
        "imported_count": 0,
        "attempts": attempts,
        "batch_id": None,
        "status": STATE_PENDING,
        "state": STATE_PENDING,
    })
    return True


def remove_batch_files(batch: dict[str, Any]) -> None:
    """Delete an imported file's local copies, which embeddings make very large.

    At 3072 dimensions an output row is about 65 KB, so a million imported
    passages would otherwise leave tens of gigabytes behind. The manifest stays.
    """
    records_path = Path(batch["records_path"])
    paths = [
        *([Path(batch["input_path"])] if batch.get("input_path") else []),
        records_path,
        records_path.with_name(records_path.stem + "-output.jsonl"),
        records_path.with_name(records_path.stem + "-errors.jsonl"),
    ]
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def get_batch_root(provider: str, config_path: str | None = None) -> Path:
    """Return (creating if needed) the directory used to store a provider's
    batch manifests, e.g. ``<config dir>/openai_batches``."""
    if config_path:
        root = Path(config_path).expanduser().parent / f"{provider}_batches"
    else:
        root = Path.home() / ".config" / "zotero-mcp" / f"{provider}_batches"
    root.mkdir(parents=True, exist_ok=True)
    _private_chmod(root)
    return root


def submit_embedding_batches(
    adapter: BatchAdapter,
    records: list[dict[str, Any]],
    model_name: str,
    embedding_config: dict[str, Any] | None,
    config_path: str | None = None,
    force_full_rebuild: bool = False,
    target_sync_version: int | None = None,
    group_id: int | None = None,
    client: Any | None = None,
    max_enqueued_tokens: int | None = None,
    max_requests: int | None = None,
    scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Upload JSONL files and create one or more embedding batches for ``adapter``.

    Shared body for every ``BatchAdapter`` (near-verbatim duplicate between
    OpenAI/Gemini before this refactor). When ``max_enqueued_tokens`` is set,
    only chunks that fit the budget are submitted now; the rest are written
    to disk and recorded as ``status``/``state`` ``"pending"`` manifest
    entries for :func:`submit_pending_batches`. ``scope`` is what
    :func:`run_covers` reads about this run, saved in the manifest.

    Every chunk is on record as pending before the first upload. A submission
    cut off part way then leaves a run with chunks still to send, which the next
    collect picks up. Recording them as they went left a shorter run that looked
    complete, and whose import moved the sync version past the unsent items.
    """
    if not records:
        raise ValueError(f"No documents were prepared for {adapter.label} Batch API submission")

    if max_requests is None:
        max_requests = adapter.max_requests

    client = client or adapter.create_client(embedding_config)
    root = get_batch_root(adapter.provider, config_path)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _private_chmod(run_dir)

    manifest: dict[str, Any] = {
        "version": 2,
        "provider": adapter.provider,
        "run_id": run_id,
        "created_at": _utc_now(),
        "model": model_name,
        "force_full_rebuild": bool(force_full_rebuild),
        "target_sync_version": target_sync_version,
        # Library the batch was submitted against, so the import can promote
        # that library's sync watermark even if the active library changed
        # while the batch was running (#393).
        "group_id": group_id,
        "max_enqueued_tokens": max_enqueued_tokens,
        "manifest_path": str(run_dir / "manifest.json"),
        "batches": [],
        # Passage ids look like "<key>#<n>"; a rebuild must match its index.
        "chunked": any("#" in record["id"] for record in records),
        **(scope or {}),
    }

    chunks = split_embedding_records(
        records,
        lambda record: adapter.build_request(record, model_name),
        max_requests=max_requests,
        max_file_bytes=adapter.max_file_bytes,
        provider_label=adapter.label,
        max_tokens=max_enqueued_tokens,
        chars_per_token=adapter.chars_per_token,
    )
    for index, (chunk_records, requests) in enumerate(chunks, start=1):
        stem = f"batch-{index:03d}"
        input_path = run_dir / f"{stem}-input.jsonl"
        records_path = run_dir / f"{stem}-records.jsonl"
        write_jsonl(input_path, requests)
        write_jsonl(records_path, chunk_records)
        manifest["batches"].append({
            "input_path": str(input_path),
            "records_path": str(records_path),
            "request_count": len(requests),
            "request_tokens": sum(estimate_tokens(r["document"], adapter.chars_per_token) for r in chunk_records),
            "imported_at": None,
            "imported_count": 0,
            "batch_id": None,
            "status": STATE_PENDING,
            "state": STATE_PENDING,
        })
    save_manifest(manifest)

    enqueued_tokens = 0
    for index, entry in enumerate(manifest["batches"], start=1):
        chunk_tokens = entry["request_tokens"]
        fits_budget = (
            max_enqueued_tokens is None
            or enqueued_tokens == 0  # always submit at least one chunk
            or enqueued_tokens + chunk_tokens <= max_enqueued_tokens
        )
        if fits_budget:
            entry.update(adapter.submit_chunk(client, Path(entry["input_path"]), run_id, index, model_name))
            entry["state"] = adapter.normalize_status(entry.get("status"))
            entry["submitted_at"] = _utc_now()
            enqueued_tokens += chunk_tokens
            save_manifest(manifest)

    return manifest


def submit_pending_batches(
    adapter: BatchAdapter,
    manifest: dict[str, Any],
    embedding_config: dict[str, Any] | None,
    max_enqueued_tokens: int | None = None,
    client: Any | None = None,
) -> int:
    """Submit ``adapter``'s pending chunks while they fit the enqueued-token budget.

    Returns the number of newly submitted chunks. Refresh the manifest status
    first so terminal batches no longer count against the budget.
    """
    if max_enqueued_tokens is None:
        max_enqueued_tokens = manifest.get("max_enqueued_tokens")
    if max_enqueued_tokens is None:
        return 0
    client = client or adapter.create_client(embedding_config)

    enqueued = sum(
        int(batch.get("request_tokens") or 0)
        for batch in manifest.get("batches", [])
        if batch.get("batch_id") and _entry_state(adapter, batch) not in TERMINAL_STATES
    )

    submitted = 0
    run_id = manifest.get("run_id", "run")
    model_name = manifest.get("model") or adapter.default_model
    for index, batch in enumerate(manifest.get("batches", []), start=1):
        if batch.get("batch_id") or batch.get("status") != STATE_PENDING:
            continue
        chunk_tokens = int(batch.get("request_tokens") or 0)
        if enqueued and enqueued + chunk_tokens > max_enqueued_tokens:
            continue
        batch.update(adapter.submit_chunk(client, Path(batch["input_path"]), run_id, index, model_name))
        batch["state"] = adapter.normalize_status(batch.get("status"))
        batch["submitted_at"] = _utc_now()
        enqueued += chunk_tokens
        submitted += 1
        save_manifest(manifest)
    return submitted


def refresh_manifest_status(
    adapter: BatchAdapter,
    manifest: dict[str, Any],
    embedding_config: dict[str, Any] | None,
    batch_ids: set[str] | None = None,
    client: Any | None = None,
    persist: bool = True,
    mark_lost: bool = True,
    recheck_lost: bool = False,
) -> dict[str, Any]:
    """Retrieve current status for ``adapter``'s selected batches and persist it.

    ``persist=False`` is for a reader without the update lock: saving could
    overwrite what the lock holder wrote in the meantime, such as the batch id
    of a chunk it submitted, and that chunk would then be paid for again.

    A provider 404 marks a chunk ``lost`` (with ``mark_lost``), so that it stops
    holding updates up. It isn't proof the batch is gone: a wrong key or base
    URL gets the same answer. So ``recheck_lost`` asks again and clears the mark
    on an answer, a status check never sets it, and a chunk whose output is
    already on disk is never marked.
    """
    client = client or adapter.create_client(embedding_config)
    for batch in manifest.get("batches", []):
        if not batch.get("batch_id"):
            continue  # pending chunk, not yet submitted
        if batch.get("imported_at"):
            continue  # settled; its status can no longer change anything
        if batch.get("lost") and not recheck_lost:
            continue
        if batch_ids and batch.get("batch_id") not in batch_ids:
            continue
        try:
            batch.update(adapter.retrieve_status(client, batch))
        except Exception as e:
            if not is_not_found(e):
                raise
            if mark_lost and not output_path_for(batch).exists():
                batch["lost"] = True
            continue
        batch.pop("lost", None)
        batch["state"] = adapter.normalize_status(batch.get("status"))
    if persist:
        save_manifest(manifest)
    return manifest
