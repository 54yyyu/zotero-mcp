"""Gemini Batch API helpers for semantic-search embeddings.

Mirrors ``openai_batch``: the Batch API is asynchronous, so this module owns
the local run manifests that tie Gemini batch job names back to the document
text and metadata needed for ChromaDB.

Uses ``client.batches.create_embeddings`` from google-genai, which is marked
experimental upstream (it targets the ``{model}:asyncBatchEmbedContent``
endpoint of the Gemini Developer API; Vertex AI is not supported). Requests
are uploaded as a JSONL file via the Files API. Each request line carries a
``key`` for correlation; the output additionally preserves input order
(guaranteed by the API), so parsing falls back to positional matching against
the submitted records when a response row has no key.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .openai_batch import (
    _json_dumps,
    _private_chmod,
    _utc_now,
    read_jsonl,  # noqa: F401 — re-exported so callers/tests can stay provider-symmetric
    write_jsonl,
)

logger = logging.getLogger(__name__)

# No request-count cap is documented for embedding batch jobs; mirror the
# OpenAI cap and additionally split on file size (Files API uploads are
# capped at 2 GB; stay far below that).
GEMINI_BATCH_MAX_REQUESTS = 50_000
GEMINI_BATCH_MAX_FILE_BYTES = 100 * 1024 * 1024
GEMINI_BATCH_DISPLAY_NAME_PREFIX = "zotero-mcp"

# JobState values that mean the job is finished (successfully or not).
GEMINI_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}
GEMINI_IMPORTABLE_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}


def get_gemini_batch_root(config_path: str | None = None) -> Path:
    """Return the directory used to store Gemini batch manifests."""
    if config_path:
        root = Path(config_path).expanduser().parent / "gemini_batches"
    else:
        root = Path.home() / ".config" / "zotero-mcp" / "gemini_batches"
    root.mkdir(parents=True, exist_ok=True)
    _private_chmod(root)
    return root


def create_gemini_client(embedding_config: dict[str, Any] | None = None) -> Any:
    """Build a google-genai client from semantic embedding config and environment."""
    embedding_config = embedding_config or {}
    api_key = (
        embedding_config.get("api_key")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not api_key:
        raise ValueError("Gemini API key is required for Batch API embeddings")

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError("google-genai package is required for Gemini Batch API embeddings") from exc

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = embedding_config.get("base_url") or os.getenv("GEMINI_BASE_URL")
    if base_url:
        client_kwargs["http_options"] = types.HttpOptions(baseUrl=base_url)
    return genai.Client(**client_kwargs)


def _is_v2_model(model_name: str) -> bool:
    return "gemini-embedding-2" in model_name


def apply_text_shaping(text: str, model_name: str) -> str:
    """Reproduce GeminiEmbeddingFunction's document text shaping.

    v2 models take the task instruction as an in-prompt prefix; batch-computed
    vectors must live in the same embedding space as realtime ones, so the
    exact same prefix is prepended here. v1 models use the ``task_type``
    request field instead (see build_embedding_request).
    """
    if _is_v2_model(model_name):
        from .chroma_client import GeminiEmbeddingFunction

        return f"{GeminiEmbeddingFunction.V2_DOC_PREFIX}{text}"
    return text


def build_embedding_request(record: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Build one JSONL request line for the asyncBatchEmbedContent endpoint."""
    request: dict[str, Any] = {
        "content": {"parts": [{"text": apply_text_shaping(record["document"], model_name)}]},
    }
    if not _is_v2_model(model_name):
        # Matches GeminiEmbeddingFunction's realtime EmbedContentConfig so
        # batch vectors are identical to realtime vectors for v1 models.
        request["task_type"] = "RETRIEVAL_DOCUMENT"
        request["title"] = "Zotero library document"
    return {"key": record["id"], "request": request}


def split_embedding_records(
    records: list[dict[str, Any]],
    model_name: str,
    max_requests: int = GEMINI_BATCH_MAX_REQUESTS,
    max_file_bytes: int = GEMINI_BATCH_MAX_FILE_BYTES,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Split records into JSONL-sized chunks accepted by the Batch API."""
    chunks: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    current_records: list[dict[str, Any]] = []
    current_requests: list[dict[str, Any]] = []
    current_bytes = 0

    for record in records:
        request = build_embedding_request(record, model_name)
        line_bytes = len((_json_dumps(request) + "\n").encode("utf-8"))
        if line_bytes > max_file_bytes:
            raise ValueError(f"Gemini batch request for {record['id']} exceeds the file size limit")

        would_exceed_count = len(current_requests) >= max_requests
        would_exceed_bytes = current_requests and current_bytes + line_bytes > max_file_bytes
        if would_exceed_count or would_exceed_bytes:
            chunks.append((current_records, current_requests))
            current_records = []
            current_requests = []
            current_bytes = 0

        current_records.append(record)
        current_requests.append(request)
        current_bytes += line_bytes

    if current_requests:
        chunks.append((current_records, current_requests))

    return chunks


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


def iter_manifests(config_path: str | None = None) -> list[Path]:
    root = get_gemini_batch_root(config_path)
    return sorted(root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def find_manifest(config_path: str | None = None, batch_id: str | None = None) -> dict[str, Any]:
    """Find the newest manifest, or the manifest that contains a batch job name."""
    for path in iter_manifests(config_path):
        manifest = load_manifest(path)
        if batch_id is None:
            return manifest
        if any(batch.get("batch_id") == batch_id for batch in manifest.get("batches", [])):
            return manifest
    if batch_id:
        raise FileNotFoundError(f"No Gemini batch manifest found for batch {batch_id}")
    raise FileNotFoundError("No Gemini batch manifests found")


def _warn_experimental_once() -> None:
    if not getattr(_warn_experimental_once, "_done", False):
        logger.warning(
            "The Gemini Batch API for embeddings is marked experimental by Google "
            "and may change without notice."
        )
        _warn_experimental_once._done = True


def _call_quietly(func, /, *args, **kwargs):
    """Call an SDK method with its ExperimentalWarning suppressed (we surface
    our own one-time note instead of one warning per API call)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


def submit_embedding_batches(
    records: list[dict[str, Any]],
    model_name: str,
    embedding_config: dict[str, Any] | None,
    config_path: str | None = None,
    force_full_rebuild: bool = False,
    target_sync_version: int | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Upload JSONL files and create one or more Gemini embedding batch jobs."""
    if not records:
        raise ValueError("No documents were prepared for Gemini Batch API submission")

    _warn_experimental_once()
    client = client or create_gemini_client(embedding_config)
    root = get_gemini_batch_root(config_path)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _private_chmod(run_dir)

    manifest: dict[str, Any] = {
        "version": 1,
        "provider": "gemini",
        "run_id": run_id,
        "created_at": _utc_now(),
        "model": model_name,
        "force_full_rebuild": bool(force_full_rebuild),
        "target_sync_version": target_sync_version,
        "manifest_path": str(run_dir / "manifest.json"),
        "batches": [],
    }

    chunks = split_embedding_records(records, model_name)
    for index, (chunk_records, requests) in enumerate(chunks, start=1):
        stem = f"batch-{index:03d}"
        input_path = run_dir / f"{stem}-input.jsonl"
        records_path = run_dir / f"{stem}-records.jsonl"
        write_jsonl(input_path, requests)
        write_jsonl(records_path, chunk_records)

        uploaded = _call_quietly(
            client.files.upload,
            file=str(input_path),
            config={"mime_type": "application/jsonl", "display_name": f"{run_id}-{stem}"},
        )
        input_file_name = getattr(uploaded, "name", None) or (
            uploaded.get("name") if isinstance(uploaded, dict) else None
        )

        batch_job = _call_quietly(
            client.batches.create_embeddings,
            model=model_name,
            src={"file_name": input_file_name},
            config={"display_name": f"{GEMINI_BATCH_DISPLAY_NAME_PREFIX}-{run_id}-{index}"},
        )

        manifest["batches"].append(
            {
                "batch_id": getattr(batch_job, "name", None),
                "input_file_name": input_file_name,
                "status": _job_state_name(batch_job),
                "dest_file_name": _dest_file_name(batch_job),
                "input_path": str(input_path),
                "records_path": str(records_path),
                "request_count": len(requests),
                "imported_at": None,
                "imported_count": 0,
            }
        )
        save_manifest(manifest)

    return manifest


def _job_state_name(batch_job: Any) -> str:
    state = getattr(batch_job, "state", None)
    if state is None and isinstance(batch_job, dict):
        state = batch_job.get("state")
    name = getattr(state, "name", None)
    if name:
        return name
    return str(state) if state else "JOB_STATE_UNSPECIFIED"


def _dest_file_name(batch_job: Any) -> str | None:
    dest = getattr(batch_job, "dest", None)
    if dest is None and isinstance(batch_job, dict):
        dest = batch_job.get("dest")
    if dest is None:
        return None
    if isinstance(dest, dict):
        return dest.get("file_name")
    return getattr(dest, "file_name", None)


def _inlined_responses(batch_job: Any) -> list[Any] | None:
    dest = getattr(batch_job, "dest", None)
    if dest is None and isinstance(batch_job, dict):
        dest = batch_job.get("dest")
    if dest is None:
        return None
    if isinstance(dest, dict):
        return dest.get("inlined_embed_content_responses")
    return getattr(dest, "inlined_embed_content_responses", None)


def refresh_manifest_status(
    manifest: dict[str, Any],
    embedding_config: dict[str, Any] | None,
    batch_ids: set[str] | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Retrieve current Gemini job state for selected batches and persist it."""
    client = client or create_gemini_client(embedding_config)
    for batch in manifest.get("batches", []):
        if batch_ids and batch.get("batch_id") not in batch_ids:
            continue
        batch_job = _call_quietly(client.batches.get, name=batch["batch_id"])
        batch["status"] = _job_state_name(batch_job)
        dest_file = _dest_file_name(batch_job)
        if dest_file:
            batch["dest_file_name"] = dest_file
    save_manifest(manifest)
    return manifest


def download_results(client: Any, batch: dict[str, Any], output_path: Path) -> str:
    """Download a completed job's output JSONL and persist it locally.

    Handles both file-based results (``dest.file_name``) and, defensively,
    inlined responses (serialized to the same JSONL shape so parsing has one
    code path).
    """
    batch_job = _call_quietly(client.batches.get, name=batch["batch_id"])
    dest_file = _dest_file_name(batch_job)
    if dest_file:
        content = _call_quietly(client.files.download, file=dest_file)
        if isinstance(content, bytes | bytearray):
            text = bytes(content).decode("utf-8")
        else:
            text = str(content)
    else:
        inlined = _inlined_responses(batch_job)
        if inlined is None:
            raise ValueError(f"Gemini batch {batch.get('batch_id')} has no result destination")
        rows = []
        for item in inlined:
            if hasattr(item, "model_dump"):
                rows.append(item.model_dump(exclude_none=True))
            elif isinstance(item, dict):
                rows.append(item)
            else:
                rows.append({"response": getattr(item, "response", None)})
        text = "\n".join(_json_dumps(row) for row in rows)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    _private_chmod(output_path)
    return text


def _extract_embedding_values(row: dict[str, Any]) -> list[float] | None:
    """Pull the embedding vector out of one output row, tolerating the
    response shapes the endpoint has been observed to use."""
    response = row.get("response") or {}
    candidates = []
    embedding = response.get("embedding")
    if isinstance(embedding, dict):
        candidates.append(embedding)
    embeddings = response.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        candidates.append(embeddings[0])
    # Some rows nest the payload one level down.
    if isinstance(row.get("embedding"), dict):
        candidates.append(row["embedding"])
    for candidate in candidates:
        values = candidate.get("values") if isinstance(candidate, dict) else None
        if isinstance(values, list) and values:
            return values
    return None


def parse_embedding_output(
    text: str, id_order: list[str]
) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    """Parse a batch output file into embeddings keyed by record id.

    Rows carrying a ``key`` are matched by key; rows without one are matched
    positionally against ``id_order`` (the exact submitted order recorded in
    the run's records file — the API guarantees output order matches input
    order).
    """
    embeddings: dict[str, list[float]] = {}
    failures: list[dict[str, Any]] = []
    position = 0
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        row_key = row.get("key") or row.get("custom_id")
        if not row_key and position < len(id_order):
            row_key = id_order[position]
        position += 1

        error = row.get("error")
        if error:
            failures.append({"custom_id": row_key, "error": error})
            continue
        values = _extract_embedding_values(row)
        if values is None:
            failures.append({"custom_id": row_key, "error": "Could not parse embedding from batch output row"})
            continue
        if row_key:
            embeddings[row_key] = values
    return embeddings, failures
