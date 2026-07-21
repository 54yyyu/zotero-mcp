"""OpenAI Batch API helpers for semantic-search embeddings.

The Batch API is asynchronous, so this module owns the local run manifests that
tie OpenAI batch IDs back to the document text and metadata needed for ChromaDB.
Provider-neutral machinery (JSONL I/O, manifests, record splitting) lives in
``batch_common``; this module keeps only OpenAI-specific request building,
submission, status mapping, and output parsing. Shared helpers are re-exported
so existing callers and tests keep working unchanged.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import batch_common
from .batch_common import (
    _json_dumps,  # noqa: F401 — re-exported for provider-symmetric callers/tests
    _jsonable,
    _object_attr,
    _private_chmod,
    _utc_now,
    estimate_tokens,  # noqa: F401 — re-export
    read_jsonl,  # noqa: F401 — re-exported so callers/tests can stay provider-symmetric
    save_manifest,
    write_jsonl,
)

OPENAI_BATCH_ENDPOINT = "/v1/embeddings"
OPENAI_BATCH_COMPLETION_WINDOW = "24h"
OPENAI_BATCH_MAX_REQUESTS = 50_000
OPENAI_BATCH_MAX_FILE_BYTES = 200 * 1024 * 1024

# Safe Tier 1 default for throttled submissions (OpenAI Tier 1 caps enqueued
# batch tokens at 3M); users on higher tiers can raise it in config.
OPENAI_BATCH_MAX_ENQUEUED_TOKENS = 2_500_000

# Char-based token estimate ratio for quota accounting (see batch_common).
OPENAI_CHARS_PER_TOKEN = 3.0


def get_openai_batch_root(config_path: str | None = None) -> Path:
    """Return the directory used to store OpenAI batch manifests."""
    if config_path:
        root = Path(config_path).expanduser().parent / "openai_batches"
    else:
        root = Path.home() / ".config" / "zotero-mcp" / "openai_batches"
    root.mkdir(parents=True, exist_ok=True)
    _private_chmod(root)
    return root


def create_openai_client(embedding_config: dict[str, Any] | None = None) -> Any:
    """Build an OpenAI client from semantic embedding config and environment."""
    embedding_config = embedding_config or {}
    api_key = embedding_config.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required for Batch API embeddings")

    try:
        import openai
    except ImportError as exc:
        raise ImportError("openai package is required for OpenAI Batch API embeddings") from exc

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = embedding_config.get("base_url") or os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    return openai.OpenAI(**client_kwargs)


def build_embedding_request(record: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Build one JSONL request object for the OpenAI embeddings endpoint."""
    return {
        "custom_id": record["id"],
        "method": "POST",
        "url": OPENAI_BATCH_ENDPOINT,
        "body": {
            "model": model_name,
            "input": record["document"],
            "encoding_format": "float",
        },
    }


def split_embedding_records(
    records: list[dict[str, Any]],
    model_name: str,
    max_requests: int = OPENAI_BATCH_MAX_REQUESTS,
    max_file_bytes: int = OPENAI_BATCH_MAX_FILE_BYTES,
    max_tokens: int | None = None,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Split records into JSONL-sized chunks accepted by the Batch API."""
    return batch_common.split_embedding_records(
        records,
        lambda record: build_embedding_request(record, model_name),
        max_requests=max_requests,
        max_file_bytes=max_file_bytes,
        provider_label="OpenAI",
        max_tokens=max_tokens,
        chars_per_token=OPENAI_CHARS_PER_TOKEN,
    )


def load_manifest(path: Path) -> dict[str, Any]:
    return batch_common.load_manifest(path)


def iter_manifests(config_path: str | None = None) -> list[Path]:
    return batch_common.iter_manifests(get_openai_batch_root(config_path))


def find_manifest(config_path: str | None = None, batch_id: str | None = None) -> dict[str, Any]:
    """Find the newest manifest, or the manifest that contains a batch ID."""
    return batch_common.find_manifest(
        get_openai_batch_root(config_path), batch_id=batch_id, provider_label="OpenAI batch"
    )


def _submit_chunk(client: Any, input_path: Path, run_id: str, index: int) -> dict[str, Any]:
    """Upload one chunk file and create its OpenAI batch; returns manifest fields."""
    with open(input_path, "rb") as input_file:
        input_file_obj = client.files.create(file=input_file, purpose="batch")
    input_file_id = _object_attr(input_file_obj, "id")

    batch_obj = client.batches.create(
        input_file_id=input_file_id,
        endpoint=OPENAI_BATCH_ENDPOINT,
        completion_window=OPENAI_BATCH_COMPLETION_WINDOW,
        metadata={
            "zotero_mcp_run_id": run_id,
            "zotero_mcp_chunk": str(index),
        },
    )
    return {
        "batch_id": _object_attr(batch_obj, "id"),
        "input_file_id": input_file_id,
        "status": _object_attr(batch_obj, "status", "validating"),
        "output_file_id": _object_attr(batch_obj, "output_file_id"),
        "error_file_id": _object_attr(batch_obj, "error_file_id"),
        "request_counts": _jsonable(_object_attr(batch_obj, "request_counts")),
    }


def submit_embedding_batches(
    records: list[dict[str, Any]],
    model_name: str,
    embedding_config: dict[str, Any] | None,
    config_path: str | None = None,
    force_full_rebuild: bool = False,
    target_sync_version: int | None = None,
    client: Any | None = None,
    max_enqueued_tokens: int | None = None,
    max_requests: int = OPENAI_BATCH_MAX_REQUESTS,
) -> dict[str, Any]:
    """Upload JSONL files and create one or more OpenAI embedding batches.

    When ``max_enqueued_tokens`` is set, only chunks that fit the budget are
    submitted now; the rest are written to disk and recorded as
    ``status: "pending"`` manifest entries for :func:`submit_pending_batches`.
    """
    if not records:
        raise ValueError("No documents were prepared for OpenAI Batch API submission")

    client = client or create_openai_client(embedding_config)
    root = get_openai_batch_root(config_path)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _private_chmod(run_dir)

    manifest: dict[str, Any] = {
        "version": 2,
        "provider": "openai",
        "run_id": run_id,
        "created_at": _utc_now(),
        "endpoint": OPENAI_BATCH_ENDPOINT,
        "completion_window": OPENAI_BATCH_COMPLETION_WINDOW,
        "model": model_name,
        "force_full_rebuild": bool(force_full_rebuild),
        "target_sync_version": target_sync_version,
        "max_enqueued_tokens": max_enqueued_tokens,
        "manifest_path": str(run_dir / "manifest.json"),
        "batches": [],
    }

    chunks = split_embedding_records(
        records, model_name, max_requests=max_requests, max_tokens=max_enqueued_tokens
    )
    enqueued_tokens = 0
    for index, (chunk_records, requests) in enumerate(chunks, start=1):
        stem = f"batch-{index:03d}"
        input_path = run_dir / f"{stem}-input.jsonl"
        records_path = run_dir / f"{stem}-records.jsonl"
        write_jsonl(input_path, requests)
        write_jsonl(records_path, chunk_records)
        chunk_tokens = sum(estimate_tokens(r["document"], OPENAI_CHARS_PER_TOKEN) for r in chunk_records)

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
            entry.update(_submit_chunk(client, input_path, run_id, index))
            enqueued_tokens += chunk_tokens
        else:
            entry.update({"batch_id": None, "status": "pending"})

        manifest["batches"].append(entry)
        save_manifest(manifest)

    return manifest


def refresh_manifest_status(
    manifest: dict[str, Any],
    embedding_config: dict[str, Any] | None,
    batch_ids: set[str] | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Retrieve current OpenAI status for selected batches and persist it."""
    client = client or create_openai_client(embedding_config)
    for batch in manifest.get("batches", []):
        if not batch.get("batch_id"):
            continue  # pending chunk, not yet submitted
        if batch_ids and batch.get("batch_id") not in batch_ids:
            continue
        batch_obj = client.batches.retrieve(batch["batch_id"])
        batch["status"] = _object_attr(batch_obj, "status", batch.get("status"))
        batch["output_file_id"] = _object_attr(batch_obj, "output_file_id", batch.get("output_file_id"))
        batch["error_file_id"] = _object_attr(batch_obj, "error_file_id", batch.get("error_file_id"))
        batch["request_counts"] = _jsonable(_object_attr(batch_obj, "request_counts", batch.get("request_counts")))
    save_manifest(manifest)
    return manifest


def submit_pending_batches(
    manifest: dict[str, Any],
    embedding_config: dict[str, Any] | None,
    max_enqueued_tokens: int | None = None,
    client: Any | None = None,
) -> int:
    """Submit pending chunks while they fit the enqueued-token budget.

    Returns the number of newly submitted chunks. Refresh the manifest status
    first so terminal batches no longer count against the budget.
    """
    if max_enqueued_tokens is None:
        max_enqueued_tokens = manifest.get("max_enqueued_tokens")
    if max_enqueued_tokens is None:
        return 0
    client = client or create_openai_client(embedding_config)

    terminal = {"completed", "failed", "expired", "cancelled"}
    enqueued = sum(
        int(batch.get("request_tokens") or 0)
        for batch in manifest.get("batches", [])
        if batch.get("batch_id") and batch.get("status") not in terminal
    )

    submitted = 0
    run_id = manifest.get("run_id", "run")
    for index, batch in enumerate(manifest.get("batches", []), start=1):
        if batch.get("batch_id") or batch.get("status") != "pending":
            continue
        chunk_tokens = int(batch.get("request_tokens") or 0)
        if enqueued and enqueued + chunk_tokens > max_enqueued_tokens:
            continue
        batch.update(_submit_chunk(client, Path(batch["input_path"]), run_id, index))
        enqueued += chunk_tokens
        submitted += 1
        save_manifest(manifest)
    return submitted


def content_to_text(content: Any) -> str:
    text = getattr(content, "text", None)
    if callable(text):
        text = text()
    if isinstance(text, str):
        return text
    if hasattr(content, "read"):
        content = content.read()
    if isinstance(content, bytes | bytearray):
        return bytes(content).decode("utf-8")
    return str(content)


def download_file_text(client: Any, file_id: str, output_path: Path) -> str:
    content = client.files.content(file_id)
    text = content_to_text(content)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    _private_chmod(output_path)
    return text


def parse_embedding_output(text: str) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    """Parse a Batch API output file into embeddings keyed by custom_id."""
    embeddings: dict[str, list[float]] = {}
    failures: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        custom_id = row.get("custom_id")
        response = row.get("response") or {}
        error = row.get("error")
        status_code = response.get("status_code")
        if error or status_code != 200:
            failures.append({"custom_id": custom_id, "error": error, "status_code": status_code})
            continue
        try:
            embedding = response["body"]["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            failures.append({"custom_id": custom_id, "error": f"Could not parse embedding: {exc}"})
            continue
        if custom_id:
            embeddings[custom_id] = embedding
    return embeddings, failures


def parse_error_output(text: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        if raw_line.strip():
            row = json.loads(raw_line)
            failures.append({"custom_id": row.get("custom_id"), "error": row.get("error")})
    return failures
