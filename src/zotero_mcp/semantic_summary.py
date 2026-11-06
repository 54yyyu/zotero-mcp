"""One description of the semantic-search configuration, shared by every command.

``setup``, ``db-status`` and ``status`` all show the same settings. They used
to each hand-roll their own list, so a setting added to one (chunking, say)
silently never appeared in the others. They now all render the rows built here.

Deliberately imports nothing heavy (no chromadb): ``cli`` must stay cheap to
import (#445).
"""

from __future__ import annotations

from typing import Any

Rows = list[tuple[str, str]]

_BATCH_LABELS = {"openai": "OpenAI", "gemini": "Gemini"}


def format_chunking_status(status: dict) -> str:
    """Describe what the next indexing run will actually do about chunking.

    Reports the effective behaviour, not the requested setting: the Batch API
    path has no chunking step, so ``chunking.enabled: true`` there still yields
    one truncated vector per item (#416).
    """
    chunking = status.get("chunking", {})
    if not chunking.get("enabled"):
        max_tokens = chunking.get("max_tokens")
        if isinstance(max_tokens, (int, float)) and max_tokens < 2000:
            return f"disabled (item-level indexing; only the first ~{int(max_tokens)} tokens of each item are searchable)"
        return "disabled (item-level indexing)"
    if chunking.get("effective"):
        return "enabled"
    return "requested but NOT applied (Batch API path indexes item-level)"


def _enabled(value: Any) -> str:
    return "enabled" if value else "disabled"


def batch_enabled(config: dict) -> bool:
    """Whether the configured provider's Batch API is switched on."""
    provider = config.get("embedding_model")
    return bool((config.get(f"{provider}_batch") or {}).get("enabled", False))


def semantic_config_rows(
    config: dict,
    zotero_db_path: str | None = None,
    max_tokens: int | float | None = None,
    batch_active: bool | None = None,
) -> Rows:
    """Label/value rows describing a ``semantic_search`` config section.

    ``batch_active`` is what the next run will actually do, when the caller
    knows it; without it the config's own setting is reported.
    """
    provider = config.get("embedding_model") or "unknown"
    embedding = config.get("embedding_config") or {}
    update = config.get("update_config") or {}
    extraction = config.get("extraction") or {}
    chunking = config.get("chunking") or {}
    batch_on = batch_enabled(config) if batch_active is None else bool(batch_active)

    rows: Rows = [
        ("Embedding model", provider),
        ("Embedding model name", str(embedding.get("model_name") or "default")),
    ]
    if embedding.get("base_url"):
        rows.append(("Embedding base URL", str(embedding["base_url"])))
    if provider in _BATCH_LABELS:
        rows.append((f"{_BATCH_LABELS[provider]} Batch API", _enabled(batch_on)))

    frequency = update.get("update_frequency") or "manual"
    auto_off = frequency != "manual" and not update.get("auto_update", True)
    rows.append(("Update frequency", frequency + (" (auto update off)" if auto_off else "")))

    pages = extraction.get("pdf_max_pages")
    rows.append((
        "Full-text indexing",
        _enabled(config.get("include_fulltext", True))
        + (f" (PDF max pages: {pages})" if pages else ""),
    ))

    chunk_status = {"chunking": {
        "enabled": bool(chunking.get("enabled")),
        "effective": bool(chunking.get("enabled")) and not batch_on,
        "max_tokens": max_tokens,
    }}
    chunk_text = format_chunking_status(chunk_status)
    if chunk_text == "enabled":
        chunk_text += (
            f" ({chunking.get('chunk_size', 1500)} chars, overlap {chunking.get('overlap', 200)},"
            f" up to {chunking.get('max_chunks_per_item', 20)} passages per item)"
        )
    rows.append(("Passage chunking", chunk_text))

    rows.append(("Reranker", _enabled((config.get("reranker") or {}).get("enabled", False))))
    rows.append(("Zotero database path", zotero_db_path or config.get("zotero_db_path") or "auto-detect"))
    return rows


def print_rows(rows: Rows, prefix: str = "- ") -> None:
    """Print rows as ``<prefix>Label: value`` lines."""
    for label, value in rows:
        print(f"{prefix}{label}: {value}")
