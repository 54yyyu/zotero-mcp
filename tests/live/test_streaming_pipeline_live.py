"""Live tests for the parallel/paced embedding path, behind the config-match
gate in tests/live/conftest.py (``configured_provider``).

Two claims this branch makes are verified against mocks everywhere else, and
a mock cannot falsify either of them:

1. **Parallelism does not change which vector belongs to which document.**
   ``max_parallel_requests > 1`` splits a call into concurrent sub-batches and
   reassembles them. The unit tests pin that reassembly bit-for-bit against a
   deterministic fake, which is right for a fake — but a real provider is not
   bit-reproducible, so the live test has to assert the property that actually
   matters instead. Measured against OpenAI: two *sequential* runs of the same
   input agree bit-for-bit on only 9 of 12 vectors, the same rate as
   sequential-vs-parallel, at a minimum cosine of 0.99999956 either way. So
   the ordering property is asserted by margin — matched pairs (~1.0) against
   mismatched pairs (~0.96) — rather than by equality.

2. **OpenAI really does send ``x-ratelimit-*-tokens`` on the embeddings
   endpoint.** The limiter reads those headers as a refill hint. An earlier
   revision of this work asserted the opposite, based on a probe that used a
   bare ``requests.post`` and missed them; the design survived being wrong
   because the headers are only ever a hint on top of a configured floor.
   This test is what keeps that corrected claim honest — if the provider
   stops sending them, or the SDK stops surfacing them, this fails rather
   than the limiter silently losing its refill signal.

Inputs are a handful of short strings; total live-API cost is a small
fraction of a cent.
"""

import json
import math
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

from zotero_mcp.embeddings.registry import create_embedding_function  # noqa: E402

# Enough documents to span several sub-batches once request_batch_size is
# forced low, so the parallel path genuinely has work to interleave.
DOCS = [f"live streaming pipeline document number {i}" for i in range(12)]


@pytest.fixture
def production_config(configured_provider):
    """The machine's real embedding_config, whatever provider it names.

    Unlike test_provider_live.py these tests are provider-agnostic: the
    parallel path lives in RemoteEmbeddingFunction, shared by every remote
    provider, so whichever one is configured exercises it.
    """
    from zotero_mcp.embeddings.registry import PROVIDERS

    config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
    if not config_path.exists():
        pytest.skip(f"no {config_path}; cannot run a config-matched live test")
    with open(config_path) as f:
        semantic = (json.load(f).get("semantic_search", {}) or {})
    provider = semantic.get("embedding_model")
    if provider not in PROVIDERS or provider in ("default", "huggingface"):
        pytest.skip(f"configured embedding_model {provider!r} is not a remote provider")
    return provider, dict(semantic.get("embedding_config", {}) or {})


def _cosine(a, b) -> float:
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    return dot / (na * nb) if na and nb else 0.0


@pytest.mark.timeout(180)
def test_parallel_keeps_each_vector_with_its_own_document(production_config):
    """The reassembly-order property, which is the failure mode worth catching.

    A parallel path that returned results in completion order rather than
    submission order would still hand back valid-looking vectors, just
    attached to the wrong documents — silently, and permanently, into the
    index. So this asserts two things: every vector still matches its own
    document's sequential twin (cosine ~1), and matched pairs beat every
    mismatched pair by a wide margin. The second half is what a shuffle
    fails; the first half alone would pass on a shuffle of near-identical
    documents.

    Not asserted: bit-identity. See the module docstring — the provider does
    not offer it even between two sequential runs.
    """
    provider, config = production_config

    sequential = create_embedding_function(
        provider, {**config, "request_batch_size": 4, "max_parallel_requests": None}
    )(DOCS)
    parallel = create_embedding_function(
        provider, {**config, "request_batch_size": 4, "max_parallel_requests": 4}
    )(DOCS)

    assert len(parallel) == len(sequential) == len(DOCS)

    matched = [_cosine(parallel[i], sequential[i]) for i in range(len(DOCS))]
    worst_matched = min(matched)
    best_mismatched = max(
        _cosine(parallel[i], sequential[j])
        for i in range(len(DOCS))
        for j in range(len(DOCS))
        if i != j
    )

    # Loose floor, well clear of the ~0.96 mismatched ceiling measured below.
    # Not tighter: the provider's run-to-run agreement is usually ~0.9999995
    # but was observed dipping to 0.9983 on text-embedding-3-large, and a
    # threshold that tight would fail on the provider's noise, not on a bug.
    assert worst_matched > 0.99, (
        f"a parallel vector diverged from its sequential twin (cosine {worst_matched:.9f}); "
        "that is far beyond the provider's own run-to-run variation"
    )
    assert worst_matched > best_mismatched, (
        f"matched pairs ({worst_matched:.9f}) do not beat mismatched pairs "
        f"({best_mismatched:.9f}) — the parallel path has reassembled sub-batch "
        "results in the wrong order"
    )


@pytest.mark.timeout(180)
def test_parallel_run_is_not_slower_than_sequential(production_config):
    """A weak but non-vacuous timing check on the concurrency actually happening.

    Deliberately asserts only "not dramatically slower" rather than a
    speedup: wall time against a live API is not reproducible enough to pin a
    ratio, and a flaky performance assertion in CI is worse than none. What
    this does catch is the parallel path accidentally serializing *and*
    paying coordination overhead on top.
    """
    provider, config = production_config
    base = {**config, "request_batch_size": 2}

    t0 = time.monotonic()
    create_embedding_function(provider, {**base, "max_parallel_requests": None})(DOCS)
    sequential_s = time.monotonic() - t0

    t0 = time.monotonic()
    create_embedding_function(provider, {**base, "max_parallel_requests": 4})(DOCS)
    parallel_s = time.monotonic() - t0

    assert parallel_s < sequential_s * 2.0, (
        f"parallel path took {parallel_s:.2f}s against {sequential_s:.2f}s sequential; "
        "expected concurrency to at least not cost time"
    )


@pytest.mark.timeout(60)
def test_openai_embeddings_endpoint_still_sends_ratelimit_headers(configured_provider):
    """Pins the corrected claim in this branch's commit 184c8c7.

    The limiter treats these headers as a refill hint layered on the
    configured ``tokens_per_minute``. If they disappear the run stays correct
    but loses its live signal, and nothing else in the suite would notice.
    """
    config = configured_provider("openai")

    import openai

    client = openai.OpenAI(api_key=config["api_key"], base_url=config.get("base_url"))
    raw = client.embeddings.with_raw_response.create(
        model=config.get("model_name", "text-embedding-3-small"),
        input="live rate limit header probe",
        encoding_format="float",
    )
    headers = {k.lower(): v for k, v in raw.headers.items()}

    assert "x-ratelimit-remaining-tokens" in headers, (
        "OpenAI stopped sending x-ratelimit-remaining-tokens on /v1/embeddings; "
        f"headers seen: {sorted(headers)}"
    )
    assert "x-ratelimit-limit-tokens" in headers
    assert int(headers["x-ratelimit-limit-tokens"]) > 0
    assert int(headers["x-ratelimit-remaining-tokens"]) >= 0


@pytest.mark.timeout(120)
def test_limiter_records_wait_time_under_a_tight_token_budget(production_config):
    """A budget far below the payload must make the limiter actually block.

    Asserts the limiter *waited*, not how long: the refill rate makes the
    exact duration a function of the budget, and pinning it would be pinning
    arithmetic the unit tests already cover. What is worth proving live is
    that the budget is consulted on the real request path at all, rather than
    being computed and then bypassed.
    """
    provider, config = production_config
    ef = create_embedding_function(
        provider,
        {**config, "request_batch_size": 2, "max_parallel_requests": 2, "tokens_per_minute": 200},
    )

    t0 = time.monotonic()
    vectors = ef(DOCS[:6])
    elapsed = time.monotonic() - t0

    assert len(vectors) == 6
    assert elapsed > 0.5, (
        f"embedding 6 documents under a 200 TPM budget finished in {elapsed:.2f}s; "
        "the token budget does not appear to be enforced on the live request path"
    )
