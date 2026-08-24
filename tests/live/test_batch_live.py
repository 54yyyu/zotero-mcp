"""Live tests for the Batch API path, behind a gate of their own.

These are separated from the rest of tests/live/ by a second environment
variable, ``ZOTERO_MCP_LIVE_BATCH=1``, because they are not like the other
live tests. A batch job is asynchronous: this submits real work to the
provider and then *waits* for it, which took roughly three minutes against
Gemini on the run that validated this file and carries no upper bound the
provider guarantees. Folding that into the ordinary live suite would make a
quick check into a coffee break, so it is opt-in twice:

    ZOTERO_MCP_LIVE_TESTS=1 ZOTERO_MCP_LIVE_BATCH=1 \\
        uv run pytest tests/live/test_batch_live.py -v

The provider is whichever one ``~/.config/zotero-mcp/config.json`` names, via
the same config-match gate the rest of the live suite uses, so this exercises
a real configuration rather than a synthetic one.

What is worth proving live, and cannot be proved otherwise:

**Batch vectors must land in the same embedding space as realtime ones.**
Nothing enforces this. The batch path builds raw JSONL request bodies by
hand while the realtime path goes through the provider SDK, so the two can
drift apart in ways that still produce perfectly valid-looking vectors — and
a collection holding both would then return quietly wrong neighbours forever.
Gemini makes this sharp: v2 models take the task instruction as a prompt
prefix and v1 models take a ``task_type`` field, and the field is spelled
``retrieval_document`` through the SDK but ``RETRIEVAL_DOCUMENT`` in raw
REST, because the SDK normalizes the enum and raw JSON does not. That
asymmetry looks like a bug and is not, which is exactly the kind of thing
that gets "fixed" into a real bug later. This test is the guard.

Measured on the validating run (gemini-embedding-001, 3072 dims): cosine
0.999999, 0.999999, 1.000000 against realtime, with zero failed rows.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")

LIVE_BATCH_ENV_VAR = "ZOTERO_MCP_LIVE_BATCH"

if os.environ.get(LIVE_BATCH_ENV_VAR, "").strip() != "1":
    pytest.skip(
        f"set {LIVE_BATCH_ENV_VAR}=1 to submit real Batch API jobs "
        "(asynchronous; minutes to hours)",
        allow_module_level=True,
    )

from zotero_mcp import batch_common  # noqa: E402
from zotero_mcp.embeddings.registry import create_embedding_function  # noqa: E402

CONFIG_PATH = Path.home() / ".config" / "zotero-mcp" / "config.json"

TEXTS = [
    "Structural equation modeling with latent variables in the social sciences.",
    "A transient disk cache avoids re-parsing PDFs after an interrupted run.",
    "Rate limiting against a token budget rather than a request count.",
]

# How long to wait for the provider before giving up. Generous, because a
# timeout here means "the provider was slow", not "the code is wrong" — the
# test reports that distinction rather than failing ambiguously.
POLL_TIMEOUT_S = float(os.environ.get("ZOTERO_MCP_LIVE_BATCH_TIMEOUT", "1800"))
POLL_INTERVAL_S = 20.0

# The project sets a global 30s pytest-timeout, which is right for a unit
# suite and fatal here: the module fixture below blocks on a real batch job
# for minutes. pytest-timeout applies the *test's* timeout to its fixture
# setup too, so this has to be raised for every test in the module, not just
# on the fixture.
pytestmark = pytest.mark.timeout(POLL_TIMEOUT_S + 300)


def _cosine(a, b) -> float:
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    return dot / (na * nb) if na and nb else 0.0


@pytest.fixture(scope="module")
def live_batch_run():
    """Submit ONE real batch job and share it across every test in this module.

    Module-scoped deliberately: each test below asserts on a different facet
    of the same run, and submitting a job per test would multiply both the
    cost and the waiting for no extra coverage.
    """
    if not CONFIG_PATH.exists():
        pytest.skip(f"no {CONFIG_PATH}; cannot run a config-matched live test")
    semantic = json.load(open(CONFIG_PATH)).get("semantic_search", {}) or {}
    provider = semantic.get("embedding_model")
    if provider not in ("openai", "gemini"):
        pytest.skip(f"configured embedding_model {provider!r} has no Batch API support")
    config = dict(semantic.get("embedding_config", {}) or {})
    if not config.get("api_key"):
        pytest.skip("configured embedding_config has no api_key")

    module = __import__(f"zotero_mcp.{provider}_batch", fromlist=["_"])
    adapter = module.ADAPTER
    model_name = config.get("model_name", adapter.default_model)
    records = [
        {"id": f"LIVEBATCH{i}", "document": text, "metadata": {}}
        for i, text in enumerate(TEXTS)
    ]

    realtime = create_embedding_function(provider, config)(TEXTS)

    manifest = module.submit_embedding_batches(
        records=records,
        model_name=model_name,
        embedding_config=config,
        config_path=str(CONFIG_PATH),
    )

    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        manifest = module.refresh_manifest_status(manifest, embedding_config=config)
        if all(
            batch_common._entry_state(adapter, b) in batch_common.TERMINAL_STATES
            for b in manifest["batches"]
        ):
            break
        time.sleep(POLL_INTERVAL_S)
    else:
        pytest.skip(
            f"{adapter.label} batch job did not reach a terminal state within "
            f"{POLL_TIMEOUT_S:.0f}s — a slow provider, not a code failure"
        )

    return {
        "provider": provider,
        "adapter": adapter,
        "module": module,
        "config": config,
        "manifest": manifest,
        "realtime": realtime,
        "model_name": model_name,
    }


def _parsed_embeddings(run):
    adapter, module, batch = run["adapter"], run["module"], run["manifest"]["batches"][0]
    client = adapter.create_client(run["config"])
    records_path = Path(batch["records_path"])
    output_path = records_path.with_name(records_path.stem + "-output.jsonl")
    text = adapter.download_output(client, batch, output_path)
    id_order = [r["id"] for r in module.read_jsonl(records_path)]
    return adapter.parse_output(text, id_order)


def test_batch_job_reaches_a_successful_terminal_state(live_batch_run):
    """The submit/poll cycle works end to end against the real provider."""
    batch = live_batch_run["manifest"]["batches"][0]
    state = batch_common._entry_state(live_batch_run["adapter"], batch)
    assert state in batch_common.IMPORTABLE_STATES, (
        f"batch finished in state {state!r} (raw status {batch.get('status')!r})"
    )


def test_manifest_carries_both_raw_status_and_normalized_state(live_batch_run):
    """The normalization is exercised by a real provider status string.

    Unit tests feed the mapping strings chosen by hand. This proves the
    provider's actual vocabulary is covered — an unmapped status silently
    normalizes to in_progress, which would leave an auto-loop polling a job
    that had already finished.
    """
    for batch in live_batch_run["manifest"]["batches"]:
        assert batch.get("status"), "raw provider status missing from the manifest"
        assert batch.get("state") in batch_common.TERMINAL_STATES
        # The raw status is kept verbatim, not overwritten by the normalized one.
        assert batch["state"] != batch["status"] or batch["status"] in batch_common.TERMINAL_STATES


def test_every_submitted_record_comes_back(live_batch_run):
    embeddings, failures = _parsed_embeddings(live_batch_run)
    assert not failures, f"batch reported failed rows: {failures}"
    assert set(embeddings) == {f"LIVEBATCH{i}" for i in range(len(TEXTS))}


def test_batch_vectors_match_realtime_vectors(live_batch_run):
    """The one that matters: batch and realtime share an embedding space.

    Asserted per document by cosine against its own realtime vector, and
    additionally by margin against the other documents — so a systematic
    shaping difference (a missing v2 prefix, a dropped task_type) fails here
    rather than degrading retrieval quality invisibly in production.
    """
    embeddings, _ = _parsed_embeddings(live_batch_run)
    realtime = live_batch_run["realtime"]

    matched = []
    for i in range(len(TEXTS)):
        vector = embeddings[f"LIVEBATCH{i}"]
        assert len(vector) == len(realtime[i]), (
            f"document {i}: batch dim {len(vector)} != realtime dim {len(realtime[i])}"
        )
        matched.append(_cosine(vector, realtime[i]))

    worst_matched = min(matched)
    best_mismatched = max(
        _cosine(embeddings[f"LIVEBATCH{i}"], realtime[j])
        for i in range(len(TEXTS))
        for j in range(len(TEXTS))
        if i != j
    )

    assert worst_matched > 0.99, (
        f"batch vectors diverge from realtime (worst cosine {worst_matched:.6f}); "
        "the two paths are no longer producing the same embedding space — check "
        "text shaping (v2 prefix) and task_type spelling in build_embedding_request"
    )
    assert worst_matched > best_mismatched, (
        f"matched pairs ({worst_matched:.6f}) do not beat mismatched pairs "
        f"({best_mismatched:.6f}); batch output is misaligned with its records"
    )
