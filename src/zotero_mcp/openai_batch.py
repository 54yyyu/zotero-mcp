"""OpenAI Batch API helpers for semantic-search embeddings.

The Batch API is asynchronous, so this module owns the local run manifests that
tie OpenAI batch IDs back to the document text and metadata needed for ChromaDB.
Provider-neutral machinery (JSONL I/O, manifests, record splitting, the
submit/refresh/pending-promotion flows) lives in ``batch_common``, driven by
``OpenAIBatchAdapter`` below; this module keeps only OpenAI-specific request
building, submission, status mapping, and output parsing. Every existing
module-level name is kept as a thin wrapper (delegating to the shared
``ADAPTER`` instance) with its exact prior signature, so callers/tests that
monkeypatch these attributes keep working unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from zotero_mcp.embeddings.registry import attach_batch_adapter

from . import batch_common
from .batch_common import (
    _json_dumps,  # noqa: F401 — re-exported for provider-symmetric callers/tests
    _jsonable,
    _object_attr,
    _private_chmod,
    _utc_now,  # noqa: F401 — re-export
    estimate_tokens,  # noqa: F401 — re-export
    read_jsonl,  # noqa: F401 — re-exported so callers/tests can stay provider-symmetric
    save_manifest,  # noqa: F401 — re-exported; generic flows now call batch_common's directly
    write_jsonl,  # noqa: F401 — re-exported; generic flows now call batch_common's directly
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

# Raw ``Batch.status`` values -> normalized batch_common vocabulary. Values
# not listed here (unexpected/future SDK statuses) are treated as still
# in-flight so a stalled auto-loop keeps polling rather than treating them as
# terminal.
_STATUS_MAP = {
    "validating": batch_common.STATE_SUBMITTED,
    "in_progress": batch_common.STATE_IN_PROGRESS,
    "finalizing": batch_common.STATE_IN_PROGRESS,
    "completed": batch_common.STATE_SUCCEEDED,
    "failed": batch_common.STATE_FAILED,
    "expired": batch_common.STATE_EXPIRED,
    "cancelled": batch_common.STATE_CANCELLED,
    "cancelling": batch_common.STATE_CANCELLED,
    "pending": batch_common.STATE_PENDING,
}


def _normalize_openai_status(raw_status: str | None) -> str:
    return _STATUS_MAP.get(raw_status or "", batch_common.STATE_IN_PROGRESS)


def get_openai_batch_root(config_path: str | None = None) -> Path:
    """Return the directory used to store OpenAI batch manifests."""
    return batch_common.get_batch_root("openai", config_path)


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
    manifest = batch_common.load_manifest(path)
    return batch_common.ensure_states(manifest, ADAPTER)


def iter_manifests(config_path: str | None = None) -> list[Path]:
    return batch_common.iter_manifests(get_openai_batch_root(config_path))


def find_manifest(config_path: str | None = None, batch_id: str | None = None) -> dict[str, Any]:
    """Find the newest manifest, or the manifest that contains a batch ID."""
    manifest = batch_common.find_manifest(
        get_openai_batch_root(config_path), batch_id=batch_id, provider_label="OpenAI batch"
    )
    return batch_common.ensure_states(manifest, ADAPTER)


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


class OpenAIBatchAdapter:
    """``batch_common.BatchAdapter`` implementation for OpenAI's Batch API.

    Methods call the module-level functions above by bare name so that
    monkeypatching this module's attributes (e.g. ``openai_batch.create_openai_client``)
    is honored — Python resolves those names against this module's ``__dict__``
    at call time, not at class-definition time.
    """

    provider = "openai"
    label = "OpenAI"
    default_model = "text-embedding-3-small"
    max_requests = OPENAI_BATCH_MAX_REQUESTS
    max_file_bytes = OPENAI_BATCH_MAX_FILE_BYTES
    default_max_enqueued_tokens = OPENAI_BATCH_MAX_ENQUEUED_TOKENS
    chars_per_token = OPENAI_CHARS_PER_TOKEN
    uses_error_file = True

    def create_client(self, embedding_config: dict[str, Any] | None) -> Any:
        return create_openai_client(embedding_config)

    def build_request(self, record: dict[str, Any], model_name: str) -> dict[str, Any]:
        return build_embedding_request(record, model_name)

    def submit_chunk(
        self, client: Any, input_path: Path, run_id: str, index: int, model_name: str | None = None
    ) -> dict[str, Any]:
        return _submit_chunk(client, input_path, run_id, index)

    def retrieve_status(self, client: Any, batch_entry: dict[str, Any]) -> dict[str, Any]:
        batch_obj = client.batches.retrieve(batch_entry["batch_id"])
        return {
            "status": _object_attr(batch_obj, "status", batch_entry.get("status")),
            "output_file_id": _object_attr(batch_obj, "output_file_id", batch_entry.get("output_file_id")),
            "error_file_id": _object_attr(batch_obj, "error_file_id", batch_entry.get("error_file_id")),
            "request_counts": _jsonable(
                _object_attr(batch_obj, "request_counts", batch_entry.get("request_counts"))
            ),
        }

    def normalize_status(self, raw_status: str | None) -> str:
        return _normalize_openai_status(raw_status)

    def download_output(self, client: Any, batch_entry: dict[str, Any], output_path: Path) -> str:
        return download_file_text(client, batch_entry["output_file_id"], output_path)

    def parse_output(
        self, text: str, id_order: list[str]
    ) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
        return parse_embedding_output(text)

    def download_errors(
        self, client: Any, batch_entry: dict[str, Any], output_path: Path
    ) -> list[dict[str, Any]]:
        if not batch_entry.get("error_file_id"):
            return []
        error_text = download_file_text(client, batch_entry["error_file_id"], output_path)
        return parse_error_output(error_text)


ADAPTER = OpenAIBatchAdapter()

# Register this module's adapter with the provider registry so
# ``batch_capable_providers()`` (registry.py) reports "openai" and the CLI's
# ``--batch-provider`` choices include it. Import-time attachment is what
# makes "has this module been imported?" and "is this provider batch-capable?"
# the same question, which is the property cli.py's lazy import relies on.
attach_batch_adapter("openai", ADAPTER)


def submit_embedding_batches(
    records: list[dict[str, Any]],
    model_name: str,
    embedding_config: dict[str, Any] | None,
    config_path: str | None = None,
    force_full_rebuild: bool = False,
    target_sync_version: int | None = None,
    group_id: int | None = None,
    client: Any | None = None,
    max_enqueued_tokens: int | None = None,
    max_requests: int = OPENAI_BATCH_MAX_REQUESTS,
) -> dict[str, Any]:
    """Upload JSONL files and create one or more OpenAI embedding batches.

    When ``max_enqueued_tokens`` is set, only chunks that fit the budget are
    submitted now; the rest are written to disk and recorded as
    ``status: "pending"`` manifest entries for :func:`submit_pending_batches`.
    """
    return batch_common.submit_embedding_batches(
        ADAPTER,
        records=records,
        model_name=model_name,
        embedding_config=embedding_config,
        config_path=config_path,
        force_full_rebuild=force_full_rebuild,
        target_sync_version=target_sync_version,
        group_id=group_id,
        client=client,
        max_enqueued_tokens=max_enqueued_tokens,
        max_requests=max_requests,
    )


def refresh_manifest_status(
    manifest: dict[str, Any],
    embedding_config: dict[str, Any] | None,
    batch_ids: set[str] | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Retrieve current OpenAI status for selected batches and persist it."""
    return batch_common.refresh_manifest_status(
        ADAPTER, manifest, embedding_config, batch_ids=batch_ids, client=client
    )


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
    return batch_common.submit_pending_batches(
        ADAPTER, manifest, embedding_config, max_enqueued_tokens=max_enqueued_tokens, client=client
    )
