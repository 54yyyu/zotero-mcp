"""OpenAI Batch API helpers for semantic-search embeddings.

The Batch API is asynchronous, so this module owns the local run manifests that
tie OpenAI batch IDs back to the document text and metadata needed for ChromaDB.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OPENAI_BATCH_ENDPOINT = "/v1/embeddings"
OPENAI_BATCH_COMPLETION_WINDOW = "24h"
OPENAI_BATCH_MAX_REQUESTS = 50_000
OPENAI_BATCH_MAX_FILE_BYTES = 200 * 1024 * 1024


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
            raise ValueError(f"OpenAI batch request for {record['id']} exceeds the 200 MB file limit")

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


def iter_manifests(config_path: str | None = None) -> list[Path]:
    root = get_openai_batch_root(config_path)
    return sorted(root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def find_manifest(config_path: str | None = None, batch_id: str | None = None) -> dict[str, Any]:
    """Find the newest manifest, or the manifest that contains a batch ID."""
    for path in iter_manifests(config_path):
        manifest = load_manifest(path)
        if batch_id is None:
            return manifest
        if any(batch.get("batch_id") == batch_id for batch in manifest.get("batches", [])):
            return manifest
    if batch_id:
        raise FileNotFoundError(f"No OpenAI batch manifest found for batch ID {batch_id}")
    raise FileNotFoundError("No OpenAI batch manifests found")


def submit_embedding_batches(
    records: list[dict[str, Any]],
    model_name: str,
    embedding_config: dict[str, Any] | None,
    config_path: str | None = None,
    force_full_rebuild: bool = False,
    target_sync_version: int | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Upload JSONL files and create one or more OpenAI embedding batches."""
    if not records:
        raise ValueError("No documents were prepared for OpenAI Batch API submission")

    client = client or create_openai_client(embedding_config)
    root = get_openai_batch_root(config_path)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _private_chmod(run_dir)

    manifest: dict[str, Any] = {
        "version": 1,
        "run_id": run_id,
        "created_at": _utc_now(),
        "endpoint": OPENAI_BATCH_ENDPOINT,
        "completion_window": OPENAI_BATCH_COMPLETION_WINDOW,
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

        manifest["batches"].append(
            {
                "batch_id": _object_attr(batch_obj, "id"),
                "input_file_id": input_file_id,
                "status": _object_attr(batch_obj, "status", "validating"),
                "output_file_id": _object_attr(batch_obj, "output_file_id"),
                "error_file_id": _object_attr(batch_obj, "error_file_id"),
                "request_counts": _jsonable(_object_attr(batch_obj, "request_counts")),
                "input_path": str(input_path),
                "records_path": str(records_path),
                "request_count": len(requests),
                "imported_at": None,
                "imported_count": 0,
            }
        )
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
        if batch_ids and batch.get("batch_id") not in batch_ids:
            continue
        batch_obj = client.batches.retrieve(batch["batch_id"])
        batch["status"] = _object_attr(batch_obj, "status", batch.get("status"))
        batch["output_file_id"] = _object_attr(batch_obj, "output_file_id", batch.get("output_file_id"))
        batch["error_file_id"] = _object_attr(batch_obj, "error_file_id", batch.get("error_file_id"))
        batch["request_counts"] = _jsonable(_object_attr(batch_obj, "request_counts", batch.get("request_counts")))
    save_manifest(manifest)
    return manifest


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
