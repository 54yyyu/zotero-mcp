"""Embedding-space drift checks against recorded sentinel vectors.

Each ``sentinels/*.json`` file (see ``generate_sentinels.py``) pins one
provider's embedding of a fixed sentinel text at a point in time. These
tests re-embed the same text live and assert the cosine similarity to the
recorded vector is still >= 0.999 -- catching accidental drift in
text-shaping, provider defaults, or model behavior that the network-free
unit suite cannot see.

Ollama runs behind the ``ollama_available`` fixture (same as
``test_ollama_live.py``); OpenAI/Gemini each run behind the config-match
gate (``configured_provider`` in conftest.py) -- a test for provider X only
actually calls the API when this machine's
``~/.config/zotero-mcp/config.json`` has
``semantic_search.embedding_model == X``. On this machine that's "openai",
so the Gemini sentinel test skips cleanly via the gate even when a Gemini
sentinel file happens to exist.

Collected but skipped by default; run with
``ZOTERO_MCP_LIVE_TESTS=1 uv run pytest tests/live/test_embedding_sentinels.py -v``.
If a sentinel file is missing, regenerate it with
``uv run python tests/live/generate_sentinels.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")

from zotero_mcp.embeddings.providers.gemini import GeminiEmbeddingFunction  # noqa: E402
from zotero_mcp.embeddings.providers.ollama import OllamaEmbeddingFunction  # noqa: E402
from zotero_mcp.embeddings.registry import create_embedding_function  # noqa: E402

SENTINELS_DIR = Path(__file__).parent / "sentinels"


def _load_sentinel(filename: str) -> dict[str, Any]:
    path = SENTINELS_DIR / filename
    if not path.exists():
        pytest.skip(
            f"sentinel file missing ({path}) — run "
            "`uv run python tests/live/generate_sentinels.py`"
        )
    with open(path) as f:
        return json.load(f)


@pytest.mark.timeout(60)
def test_ollama_sentinel(ollama_available, cosine_similarity):
    sentinel = _load_sentinel("ollama-nomic-embed-text.json")

    ef = OllamaEmbeddingFunction(model_name=sentinel["model_name"], base_url=ollama_available)
    vector = ef.embed_query(sentinel["text"])

    assert len(vector) == sentinel["dim"]
    similarity = cosine_similarity(vector, sentinel["vector"])
    assert similarity >= 0.999, f"ollama sentinel drift detected: cosine={similarity}"


@pytest.mark.timeout(60)
def test_openai_sentinel(configured_provider, cosine_similarity):
    production_config = configured_provider("openai")
    sentinel = _load_sentinel("openai-text-embedding-3-small.json")

    cfg = {**production_config, "model_name": sentinel["model_name"]}
    ef = create_embedding_function("openai", cfg)
    vector = ef.embed_query(sentinel["text"])

    assert len(vector) == sentinel["dim"]
    similarity = cosine_similarity(vector, sentinel["vector"])
    assert similarity >= 0.999, f"openai sentinel drift detected: cosine={similarity}"


@pytest.mark.timeout(60)
def test_gemini_sentinel(configured_provider, cosine_similarity):
    production_config = configured_provider("gemini")
    sentinel = _load_sentinel("gemini-gemini-embedding-001.json")

    cfg = {**production_config, "model_name": sentinel["model_name"]}
    ef = create_embedding_function("gemini", cfg)
    assert isinstance(ef, GeminiEmbeddingFunction)
    vector = ef.embed_query(sentinel["text"])

    assert len(vector) == sentinel["dim"]
    similarity = cosine_similarity(vector, sentinel["vector"])
    assert similarity >= 0.999, f"gemini sentinel drift detected: cosine={similarity}"
