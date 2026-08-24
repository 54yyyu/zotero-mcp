"""Regenerate ``tests/live/sentinels/*.json`` against real embedding providers.

Not a pytest module (no ``test_`` functions), so it is never collected by
``pytest tests/`` -- it is a standalone script, run manually:

    uv run python tests/live/generate_sentinels.py

Each sentinel is a small JSON file capturing one provider's embedding of a
fixed sentinel text: ``{provider, model_name, text, dim, vector,
generated_at}``. ``tests/live/test_embedding_sentinels.py`` re-embeds the same
text against a live provider and asserts the cosine similarity to the
recorded vector is >= 0.999 -- a drift check that catches accidental changes
to text-shaping, provider defaults, or model behavior that unit tests (which
never touch the network) cannot see.

Sentinels are written compactly (``json.dumps(..., separators=(",", ":"))``,
one line per file) rather than pretty-printed -- a 1536-float vector
pretty-printed one value per line is ~1,500 lines of diff noise for a value
that is, in practice, opaque binary data to a human reviewer.

Providers this script attempts, each independently and non-fatally:

- **Ollama**: if a local server is reachable at ``OLLAMA_BASE_URL`` (or
  ``http://localhost:11434``) and has ``nomic-embed-text`` pulled, embeds the
  sentinel text and writes ``sentinels/ollama-nomic-embed-text.json``.
- **OpenAI**: if an API key is available (``OPENAI_API_KEY`` env var, or
  ``~/.config/zotero-mcp/config.json`` -> ``semantic_search.embedding_config``
  when ``semantic_search.embedding_model == "openai"``), embeds with
  ``text-embedding-3-small`` and writes
  ``sentinels/openai-text-embedding-3-small.json``.
- **Gemini**: only if a Gemini key is available (``GEMINI_API_KEY`` /
  ``GOOGLE_API_KEY`` env var, or config has
  ``embedding_model == "gemini"`` with an ``api_key``), embeds with the
  provider's default model (``gemini-embedding-001``) and writes
  ``sentinels/gemini-gemini-embedding-001.json``.

A provider that isn't reachable/configured is skipped with a printed message
-- this script never fails just because a given provider is unavailable on
the machine it's run on. The api_key itself is never printed or logged.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SENTINEL_TEXT = "The quick brown fox jumps over the lazy dog."

SENTINELS_DIR = Path(__file__).parent / "sentinels"
CONFIG_PATH = Path.home() / ".config" / "zotero-mcp" / "config.json"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def _load_config() -> dict[str, Any] | None:
    if not CONFIG_PATH.exists():
        return None
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def _config_embedding_config(provider: str) -> dict[str, Any] | None:
    """Return the production embedding_config dict when the machine's config
    has semantic_search.embedding_model == provider; else None. Never logs
    the contents (caller must be equally careful with api_key)."""
    config = _load_config()
    if config is None:
        return None
    semantic_search = config.get("semantic_search", {}) or {}
    if semantic_search.get("embedding_model") != provider:
        return None
    return dict(semantic_search.get("embedding_config", {}) or {})


def _write_sentinel(path: Path, provider: str, model_name: str, text: str, vector: list[float]) -> None:
    payload = {
        "provider": provider,
        "model_name": model_name,
        "text": text,
        "dim": len(vector),
        "vector": [float(x) for x in vector],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    SENTINELS_DIR.mkdir(parents=True, exist_ok=True)
    # Compact separators, no indent: keeps the vector on one line so a
    # diff shows "the vector changed", not thousands of changed lines.
    with open(path, "w") as f:
        f.write(json.dumps(payload, separators=(",", ":")))
        f.write("\n")
    print(f"wrote {path} (provider={provider}, model={model_name}, dim={len(vector)})")


def generate_ollama() -> None:
    import requests

    base_url = (os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        print(f"skip ollama: not reachable at {base_url}: {exc}")
        return

    models = [m.get("name", "") for m in payload.get("models", [])]
    pulled = {name.split(":", 1)[0] for name in models}
    if "nomic-embed-text" not in pulled:
        print(f"skip ollama: 'nomic-embed-text' not pulled (pulled models: {sorted(pulled)})")
        return

    from zotero_mcp.embeddings.providers.ollama import OllamaEmbeddingFunction

    ef = OllamaEmbeddingFunction(model_name="nomic-embed-text", base_url=base_url)
    vector = ef.embed_query(SENTINEL_TEXT)
    _write_sentinel(
        SENTINELS_DIR / "ollama-nomic-embed-text.json",
        provider="ollama",
        model_name="nomic-embed-text",
        text=SENTINEL_TEXT,
        vector=vector,
    )


def generate_openai() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        cfg = _config_embedding_config("openai")
        if cfg:
            api_key = cfg.get("api_key")
    if not api_key:
        print("skip openai: no OPENAI_API_KEY env var and no configured openai api_key in config.json")
        return

    from zotero_mcp.embeddings.providers.openai import OpenAIEmbeddingFunction

    model_name = "text-embedding-3-small"
    ef = OpenAIEmbeddingFunction(model_name=model_name, api_key=api_key)
    vector = ef.embed_query(SENTINEL_TEXT)
    _write_sentinel(
        SENTINELS_DIR / "openai-text-embedding-3-small.json",
        provider="openai",
        model_name=model_name,
        text=SENTINEL_TEXT,
        vector=vector,
    )


def generate_gemini() -> None:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        cfg = _config_embedding_config("gemini")
        if cfg:
            api_key = cfg.get("api_key")
    if not api_key:
        print("skip gemini: no GEMINI_API_KEY/GOOGLE_API_KEY env var and no configured gemini api_key in config.json")
        return

    from zotero_mcp.embeddings.providers.gemini import GeminiEmbeddingFunction

    model_name = "gemini-embedding-001"  # GeminiEmbeddingFunction's default
    ef = GeminiEmbeddingFunction(model_name=model_name, api_key=api_key)
    vector = ef.embed_query(SENTINEL_TEXT)
    _write_sentinel(
        SENTINELS_DIR / f"gemini-{model_name}.json",
        provider="gemini",
        model_name=model_name,
        text=SENTINEL_TEXT,
        vector=vector,
    )


def main() -> None:
    generate_ollama()
    generate_openai()
    generate_gemini()


if __name__ == "__main__":
    main()
