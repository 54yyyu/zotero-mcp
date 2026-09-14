"""Gating and backend-probe fixtures for the live cross-backend parity suite.

Everything under tests/live/ is skipped unless ZOTERO_MCP_LIVE_TESTS=1 is set
— zero network calls happen otherwise, including at collection time. When
enabled, these fixtures probe whichever real Zotero backends are actually
reachable from this machine (local desktop API, web API, local zotero.sqlite)
and hand back None for anything unavailable, so tests can skip gracefully
rather than fail when e.g. only one backend is configured.
"""

import os

import pytest

LIVE_TESTS_ENV_VAR = "ZOTERO_MCP_LIVE_TESTS"


def pytest_collection_modifyitems(config, items):
    if os.environ.get(LIVE_TESTS_ENV_VAR, "").strip() == "1":
        return
    skip = pytest.mark.skip(
        reason=f"set {LIVE_TESTS_ENV_VAR}=1 to run live cross-backend parity tests"
    )
    for item in items:
        if "tests/live" in str(item.fspath).replace(os.sep, "/"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def local_zot():
    """Real pyzotero client against local Zotero desktop, or None if
    unreachable (desktop not running, local server disabled, etc.)."""
    from zotero_mcp.client import get_local_zotero_client

    return get_local_zotero_client()


@pytest.fixture(scope="session")
def web_zot():
    """Real pyzotero client against the Zotero web API, or None if
    ZOTERO_LIBRARY_ID/ZOTERO_API_KEY aren't set."""
    from zotero_mcp.client import get_web_zotero_client

    return get_web_zotero_client()


@pytest.fixture(scope="session")
def sql_reader(local_zot):
    """LocalZoteroReader against this machine's zotero.sqlite, or None.

    Only meaningful when local Zotero is set up here (same machine, same
    data directory as `local_zot` just probed) — get_local_zotero_reader()
    additionally requires ZOTERO_LOCAL to be truthy even though it only
    touches the DB file, never the HTTP API, so it's set for the session
    if the caller hasn't already set it.
    """
    if local_zot is None:
        # `yield`, not `return`: a bare return in a generator fixture raises
        # StopIteration, which pytest reports as "sql_reader did not yield a
        # value" — every dependent test ERRORs instead of skipping, exactly
        # when Zotero desktop is closed.
        yield None
        return
    os.environ.setdefault("ZOTERO_LOCAL", "true")
    from zotero_mcp.local_db import get_local_zotero_reader

    reader = get_local_zotero_reader()
    yield reader
    if reader is not None:
        reader.close()


@pytest.fixture(scope="session")
def available_backends(local_zot, web_zot, sql_reader):
    """{'local_api': zot|None, 'web_api': zot|None, 'sqlite': reader|None}."""
    backends = {"local_api": local_zot, "web_api": web_zot, "sqlite": sql_reader}
    present = [name for name, client in backends.items() if client is not None]
    missing = [name for name, client in backends.items() if client is None]
    print(f"\n[live parity] available backends: {present or 'none'}"
          f"{f' (skipped: {missing})' if missing else ''}")
    return backends


@pytest.fixture(scope="session")
def personal_library_item_count(local_zot, web_zot) -> int | None:
    """Fast item count (zot.num_items(), a single lightweight request) for
    whichever pyzotero client is available, or None if neither is.

    advanced_search's pyzotero fallback has no server-side query support at
    all — it pages the ENTIRE library client-side, 100 items at a time, and
    filters in Python (this is exactly the slowness the SQL backend exists
    to fix). Against a real library of any size that makes it impractical
    for a routine live-test run, so tests use this count to skip that
    specific comparison rather than hang for minutes.
    """
    zot = local_zot or web_zot
    if zot is None:
        return None
    try:
        return zot.num_items()
    except Exception:
        return None


@pytest.fixture(scope="session")
def discovered_values(local_zot, web_zot):
    """Real, non-hardcoded query values pulled from whichever backend is
    reachable — used to exercise each condition field against real data
    from the connected library instead of a fixed name/title/collection."""
    from ._discovery import discover

    zot = local_zot or web_zot
    if zot is None:
        return {}
    return discover(zot)


# -- embedding-provider fixtures -------------------------------------------
#
# Everything below this line is the *embedding* half of this file (sentinel /
# provider / fault-injection / Chroma-roundtrip live tests). The half above is
# the search/Zotero-library half (local_zot / web_zot / sql_reader). The two
# are disjoint: they share only the module-wide gate at the top.
#
# Everything under tests/live/ hits a real network service (a local Ollama
# server, or a paid provider API using the machine's production config)
# rather than a mock. A bare ``uv run pytest tests/`` always collects these
# tests (they show up as skipped, via the hook above) but never talks to the
# network; ``ZOTERO_MCP_LIVE_TESTS=1 uv run pytest tests/live/ -v`` runs them
# for real.
#
# Two independent gates exist on top of the module-wide skip:
#
# - ``ollama_available`` (session fixture): skips if a local Ollama server
#   isn't reachable or doesn't have the required model pulled.
# - ``configured_provider`` (fixture factory) / ``load_live_config``: the
#   "config-match gate" -- a test asking for provider X only runs if the
#   machine's actual ``~/.config/zotero-mcp/config.json`` has
#   ``semantic_search.embedding_model == X``, in which case it gets that
#   provider's real production ``embedding_config`` (including api_key). This
#   is intentional: whoever runs live tests is exercising the exact
#   configuration their own deployment uses, so a real paid API call is
#   justified. The api_key is never printed or logged by anything in this
#   file.

import json  # noqa: E402
import math  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import requests  # noqa: E402

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
CONFIG_PATH = Path.home() / ".config" / "zotero-mcp" / "config.json"


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
    ``zotero_mcp.embeddings.providers.ollama`` to monkeypatch -- the global
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
    modules import directly, so it is reachable from any test module in this
    package via plain fixture injection without relying on a particular
    import-root layout.
    """
    return _cosine_similarity
