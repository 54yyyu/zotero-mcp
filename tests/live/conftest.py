"""Shared fixtures for the live-testing layer (``tests/live/``).

Everything under ``tests/live/`` hits a real network service (a local Ollama
server, or a paid provider API using the machine's production config) rather
than a mock. ``pytest_collection_modifyitems`` below marks every item in this
directory with ``pytest.mark.live`` and, unless ``ZOTERO_MCP_LIVE_TESTS=1`` is
set, also skips it — so a bare ``uv run pytest tests/`` always collects these
tests (they show up as skipped) but never talks to the network, and
``ZOTERO_MCP_LIVE_TESTS=1 uv run pytest tests/live/ -v`` runs them for real.

Two independent gates exist on top of that:

- ``ollama_available`` (session fixture): skips if a local Ollama server
  isn't reachable or doesn't have the required model pulled.
- ``configured_provider`` (fixture factory) / ``load_live_config``: the
  "config-match gate" — a test asking for provider X only runs if the
  machine's actual ``~/.config/zotero-mcp/config.json`` has
  ``semantic_search.embedding_model == X``, in which case it gets that
  provider's real production ``embedding_config`` (including api_key). This
  is intentional: whoever runs live tests is exercising the exact
  configuration their own deployment uses, so a real paid API call is
  justified. The api_key is never printed or logged by anything in this file.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import requests

LIVE_DIR = Path(__file__).parent
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
CONFIG_PATH = Path.home() / ".config" / "zotero-mcp" / "config.json"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark every test under tests/live/ as `live`, and skip it by default.

    ``ZOTERO_MCP_LIVE_TESTS=1`` is the opt-in that lets these tests actually
    run (see module docstring). This hook is defined in tests/live/conftest.py
    but, once pytest loads it, applies to the whole collected session — the
    explicit path check below is what limits its effect to this directory.
    """
    live_enabled = os.environ.get("ZOTERO_MCP_LIVE_TESTS") == "1"
    skip_reason = "live tests disabled; set ZOTERO_MCP_LIVE_TESTS=1 to run tests/live/"
    for item in items:
        try:
            item.path.relative_to(LIVE_DIR)
        except ValueError:
            continue
        item.add_marker(pytest.mark.live)
        if not live_enabled:
            item.add_marker(pytest.mark.skip(reason=skip_reason))


@pytest.fixture(scope="session")
def ollama_available() -> str:
    """Skip unless a local Ollama server is reachable and has nomic-embed-text.

    Returns the resolved base_url on success so tests can reuse it.
    """
    base_url = (os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        pytest.skip(f"Ollama not reachable at {base_url}: {exc}")

    try:
        payload = resp.json()
    except Exception as exc:
        pytest.skip(f"Ollama at {base_url} returned an unparsable /api/tags response: {exc}")

    models = [m.get("name", "") for m in payload.get("models", [])]
    # Model names come back as "nomic-embed-text:latest"; compare on the
    # part before the tag so any pulled tag counts.
    pulled = {name.split(":", 1)[0] for name in models}
    if "nomic-embed-text" not in pulled:
        pytest.skip(
            f"Ollama is up at {base_url} but 'nomic-embed-text' is not pulled "
            f"(pulled models: {sorted(pulled)}). Run: ollama pull nomic-embed-text"
        )
    return base_url


@pytest.fixture
def count_requests_post(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple, dict]]:
    """Count calls to the global ``requests.post`` while passing them through.

    ``OllamaEmbeddingFunction._embed_batch`` does ``import requests`` *inside*
    the method body, so there is no module attribute on
    ``zotero_mcp.embeddings.providers.ollama`` to monkeypatch — the global
    ``requests.post`` is the only interception point.
    """
    calls: list[tuple[tuple, dict]] = []
    original_post = requests.post

    def counting_post(*args: Any, **kwargs: Any):
        calls.append((args, kwargs))
        return original_post(*args, **kwargs)

    monkeypatch.setattr(requests, "post", counting_post)
    return calls


@pytest.fixture
def wrap_embed_batch():
    """Factory fixture: wrap an embedding-function INSTANCE's ``_embed_batch``
    with a counting passthrough.

    Provider-agnostic (works for SDK-based providers like OpenAI/Gemini,
    where there is no single global function to patch the way there is for
    Ollama's ``requests.post``). Usage::

        calls = wrap_embed_batch(ef)
        ef(["a", "b", "c"])
        assert len(calls) == 2
    """

    def _wrap(ef: Any) -> list[tuple[tuple, dict]]:
        calls: list[tuple[tuple, dict]] = []
        original = ef._embed_batch

        def counting(*args: Any, **kwargs: Any):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        ef._embed_batch = counting
        return calls

    return _wrap


def load_live_config() -> dict[str, Any] | None:
    """Load ``~/.config/zotero-mcp/config.json``, or ``None`` if missing/unreadable.

    Same default path ``create_chroma_client`` reads. Never logs or prints
    the contents (the caller must be equally careful with ``api_key``).
    """
    if not CONFIG_PATH.exists():
        return None
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return None


@pytest.fixture
def configured_provider():
    """Factory fixture implementing the config-match gate.

    ``configured_provider("openai")`` returns the production
    ``semantic_search.embedding_config`` dict when the machine's config has
    ``semantic_search.embedding_model == "openai"``; otherwise it
    ``pytest.skip``s with a message naming the actual configured provider.
    Missing config file skips cleanly too.
    """

    def _get(provider: str) -> dict[str, Any]:
        config = load_live_config()
        if config is None:
            pytest.skip(
                f"no {CONFIG_PATH} found; cannot run the config-matched '{provider}' live test"
            )
        semantic_search = config.get("semantic_search", {}) or {}
        actual = semantic_search.get("embedding_model")
        if actual != provider:
            pytest.skip(f"live config embedding_model is '{actual}', not '{provider}'")
        return dict(semantic_search.get("embedding_config", {}) or {})

    return _get


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-python cosine similarity (no numpy dependency required)."""
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
    norm_b = math.sqrt(sum(float(y) * float(y) for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@pytest.fixture
def cosine_similarity() -> Callable[[list[float], list[float]], float]:
    """Returns the pure-python cosine-similarity helper.

    Exposed as a fixture (returning the function) rather than something test
    modules import directly: with ``consider_namespace_packages`` enabled
    (required so this directory's conftest.py doesn't collide with
    tests/conftest.py — see pyproject.toml), tests/live/ is not reliably on
    sys.path as a plain import root, so a bare/dotted cross-module import
    from a test file is fragile. Fixture injection goes through pytest's own
    plugin registry instead, which is unaffected by that.
    """
    return _cosine_similarity
