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
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
