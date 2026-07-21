"""Provider-neutral helpers shared by the OpenAI and Gemini Batch API modules.

Everything here is independent of the provider SDK: JSONL I/O, local run
manifests (which tie asynchronous batch jobs back to the document text and
metadata needed for ChromaDB), and the record-splitting algorithm that packs
embedding requests into Batch-API-sized chunk files.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_manifest(manifest: dict[str, Any]) -> None:
    manifest_path = Path(manifest["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _private_chmod(manifest_path)


def load_manifest(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["manifest_path"] = str(path)
    return manifest


def iter_manifests(root: Path) -> list[Path]:
    return sorted(root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def find_manifest(root: Path, batch_id: str | None = None, provider_label: str = "batch") -> dict[str, Any]:
    """Find the newest manifest, or the manifest that contains a batch ID."""
    for path in iter_manifests(root):
        manifest = load_manifest(path)
        if batch_id is None:
            return manifest
        if any(batch.get("batch_id") == batch_id for batch in manifest.get("batches", [])):
            return manifest
    if batch_id:
        raise FileNotFoundError(f"No {provider_label} manifest found for batch {batch_id}")
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
    client: Any | None = None,
    max_enqueued_tokens: int | None = None,
    max_requests: int | None = None,
) -> dict[str, Any]:
    """Upload JSONL files and create one or more embedding batches for ``adapter``.

    Shared body for every ``BatchAdapter`` (near-verbatim duplicate between
    OpenAI/Gemini before this refactor). When ``max_enqueued_tokens`` is set,
    only chunks that fit the budget are submitted now; the rest are written
    to disk and recorded as ``status``/``state`` ``"pending"`` manifest
    entries for :func:`submit_pending_batches`.
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
        "max_enqueued_tokens": max_enqueued_tokens,
        "manifest_path": str(run_dir / "manifest.json"),
        "batches": [],
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
    enqueued_tokens = 0
    for index, (chunk_records, requests) in enumerate(chunks, start=1):
        stem = f"batch-{index:03d}"
        input_path = run_dir / f"{stem}-input.jsonl"
        records_path = run_dir / f"{stem}-records.jsonl"
        write_jsonl(input_path, requests)
        write_jsonl(records_path, chunk_records)
        chunk_tokens = sum(estimate_tokens(r["document"], adapter.chars_per_token) for r in chunk_records)

        entry: dict[str, Any] = {
            "input_path": str(input_path),
            "records_path": str(records_path),
            "request_count": len(requests),
            "request_tokens": chunk_tokens,
            "imported_at": None,
            "imported_count": 0,
        }

        fits_budget = (
            max_enqueued_tokens is None
            or enqueued_tokens == 0  # always submit at least one chunk
            or enqueued_tokens + chunk_tokens <= max_enqueued_tokens
        )
        if fits_budget:
            entry.update(adapter.submit_chunk(client, input_path, run_id, index, model_name))
            entry["state"] = adapter.normalize_status(entry.get("status"))
            enqueued_tokens += chunk_tokens
        else:
            entry.update({"batch_id": None, "status": STATE_PENDING, "state": STATE_PENDING})

        manifest["batches"].append(entry)
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
) -> dict[str, Any]:
    """Retrieve current status for ``adapter``'s selected batches and persist it."""
    client = client or adapter.create_client(embedding_config)
    for batch in manifest.get("batches", []):
        if not batch.get("batch_id"):
            continue  # pending chunk, not yet submitted
        if batch_ids and batch.get("batch_id") not in batch_ids:
            continue
        batch.update(adapter.retrieve_status(client, batch))
        batch["state"] = adapter.normalize_status(batch.get("status"))
    save_manifest(manifest)
    return manifest
