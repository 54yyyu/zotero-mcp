"""Tests for RemoteEmbeddingFunction (src/zotero_mcp/embeddings/base.py).

Covers only the provider-agnostic base class -- not the concrete openai/gemini/ollama
providers, which each have their own test file. Every retry/backoff-relevant test drives
an AdaptiveRateLimiter with a fake clock/sleep (same style as tests/test_rate_limiter.py's
FakeClock) so nothing here actually waits. Coverage:

1. Sub-batching: splitting, a falsy request_batch_size, and the empty-input guard.
2. Parallelism: multi-threaded execution, index-addressed output ordering regardless of
   completion order, the sequential fallback, and sequential/parallel output parity.
3. Retries: recovery, exhaustion, non-retryable fast-fail, and retry_after handling.
4. _embed_batch's two return shapes (bare list vs. (vectors, headers)).
5. embed_query's routing, and truncate-before-prepare ordering.
6. _prepare_document application.
7. _common_config()'s round-trip, and the __new__-without-__init__ design property.
8. _init_common's tokens_per_minute precedence: argument > env var > class default.

Note on empty input: RemoteEmbeddingFunction.__call__ is wrapped by ChromaDB's own
EmbeddingFunction.__init_subclass__ (applied to every subclass, this file's fakes
included), which calls validate_embeddings(normalize_embeddings(...)) on whatever
__call__ returns and raises ValueError for an empty list before the caller ever sees it.
So "returns []" is not observable through the public __call__ protocol; what IS
observable and is pinned below is that _embed_batch is never invoked for an empty input.
tests/test_ollama_embedding.py::test_call_empty_input_issues_no_http_request relies on
the same reasoning for a concrete provider.
"""

import threading
import time

import pytest

pytest.importorskip("chromadb")

from zotero_mcp.embeddings.base import COMMON_CONFIG_KEYS, RemoteEmbeddingFunction  # noqa: E402
from zotero_mcp.embeddings.ratelimit import AdaptiveRateLimiter  # noqa: E402


class FakeClock:
    """Manually advanced monotonic clock plus a sleep that advances it.

    Copied in style from tests/test_rate_limiter.py::FakeClock rather than imported, since
    that module is a test file, not a shared library -- every wait recorded here goes
    through the same fake-clock seam AdaptiveRateLimiter already exposes for exactly this
    purpose.
    """

    def __init__(self, start=1000.0):
        self.now = start
        self.slept = []  # every duration passed to sleep()

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


class _RetryableError(Exception):
    """Marker exception a test's `_classify_error_impl` classifies as retryable."""


class _FatalError(Exception):
    """Marker exception that must never be retried (exercises the base class's default
    `_classify_error`, which always returns (False, None))."""


class _FlakyBatch:
    """Callable `_embed_batch_impl` that raises `error` on its first `fail_times`
    invocations, then returns `result` -- drives `_embed_with_retry`'s retry loop
    deterministically."""

    def __init__(self, error, fail_times, result):
        self.error = error
        self.fail_times = fail_times
        self.result = result
        self.attempts = 0

    def __call__(self, texts, is_query=False):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise self.error
        return self.result


class _RecordingLimiter(AdaptiveRateLimiter):
    """AdaptiveRateLimiter that records every `headers` value passed to `on_success`, so a
    test can assert that a `_embed_batch` returning `(vectors, headers)` actually reaches
    the limiter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_success_calls = []

    def on_success(self, headers=None):
        self.on_success_calls.append(headers)
        super().on_success(headers)


class _ProgrammableEF(RemoteEmbeddingFunction):
    """RemoteEmbeddingFunction subclass whose hooks delegate to instance-level callables set
    by `_make_ef`, so one class covers every scenario in this file (sub-batching,
    parallelism, retries, header propagation, query preparation) without a new subclass per
    test.

    Every `_embed_batch` call is recorded into `self.calls` -- texts, the `is_query` flag,
    and the calling thread's name -- so tests can assert on batching, ordering and
    threading without needing to decode return values.
    """

    def _embed_batch(self, texts, is_query=False):
        with self._lock:
            self.calls.append(
                {
                    "texts": list(texts),
                    "is_query": is_query,
                    "thread": threading.current_thread().name,
                }
            )
        return self._embed_batch_impl(list(texts), is_query)

    def _classify_error(self, exc):
        impl = getattr(self, "_classify_error_impl", None)
        if impl is not None:
            return impl(exc)
        return super()._classify_error(exc)

    def _prepare_document(self, text):
        impl = getattr(self, "_prepare_document_impl", None)
        if impl is not None:
            return impl(text)
        return super()._prepare_document(text)

    def _prepare_query(self, text):
        impl = getattr(self, "_prepare_query_impl", None)
        if impl is not None:
            return impl(text)
        return super()._prepare_query(text)


def _make_ef(
    *,
    request_batch_size=None,
    rate_limit_rps=None,
    max_parallel_requests=1,
    max_retries=0,
    tokens_per_minute=None,
    truncate_queries=False,
    max_input_tokens=None,
    embed_batch_impl=None,
    classify_error_impl=None,
    prepare_document_impl=None,
    prepare_query_impl=None,
    limiter=None,
):
    """Build a `_ProgrammableEF` via `__new__`, bypassing the real `__init__` entirely.

    Mirrors the pattern already used elsewhere in this repo (e.g.
    tests/test_openai_embedding_batching.py::_make) for constructing a provider without its
    real constructor (API keys, client setup, ...).
    """
    ef = _ProgrammableEF.__new__(_ProgrammableEF)
    ef.calls = []
    ef._lock = threading.Lock()
    ef.request_batch_size = request_batch_size
    ef.rate_limit_rps = rate_limit_rps
    ef.max_parallel_requests = max_parallel_requests
    ef.max_retries = max_retries
    ef.tokens_per_minute = tokens_per_minute
    ef.truncate_queries = truncate_queries
    if max_input_tokens is not None:
        ef.max_input_tokens = max_input_tokens
    ef._embed_batch_impl = embed_batch_impl or (lambda texts, is_query: [[0.0] for _ in texts])
    if classify_error_impl is not None:
        ef._classify_error_impl = classify_error_impl
    if prepare_document_impl is not None:
        ef._prepare_document_impl = prepare_document_impl
    if prepare_query_impl is not None:
        ef._prepare_query_impl = prepare_query_impl
    if limiter is not None:
        ef.limiter = limiter
    return ef


# -- 1. Sub-batching ----------------------------------------------------------


def test_call_splits_large_input_into_subbatches_preserving_order():
    """With request_batch_size=2 and a 5-document input, __call__ issues exactly 3 requests
    of sizes 2, 2, 1, and the concatenated output is in input order."""
    ef = _make_ef(
        request_batch_size=2,
        embed_batch_impl=lambda texts, is_query: [[float(v)] for v in texts],
    )
    out = ef([0, 1, 2, 3, 4])
    assert [len(call["texts"]) for call in ef.calls] == [2, 2, 1]
    assert out == [[0.0], [1.0], [2.0], [3.0], [4.0]]


def test_falsy_request_batch_size_sends_whole_input_in_one_request():
    """request_batch_size=None (falsy) means the whole input goes in ONE request."""
    ef = _make_ef(
        request_batch_size=None,
        embed_batch_impl=lambda texts, is_query: [[float(v)] for v in texts],
    )
    out = ef([0, 1, 2, 3, 4])
    assert len(ef.calls) == 1
    assert len(ef.calls[0]["texts"]) == 5
    assert out == [[0.0], [1.0], [2.0], [3.0], [4.0]]


def test_empty_input_issues_zero_requests():
    """An empty input issues ZERO requests, not one empty one -- __call__'s own guard runs
    before any _embed_batch call. (See the module docstring above for why the guard's
    `return []` itself is not independently observable through __call__.)"""
    ef = _make_ef(embed_batch_impl=lambda texts, is_query: [[float(v)] for v in texts])
    with pytest.raises(ValueError):
        ef([])
    assert ef.calls == []


# -- 2. Parallelism and ordering ----------------------------------------------


def test_parallel_requests_use_multiple_threads():
    """max_parallel_requests > 1 with several sub-batches runs them concurrently: more than
    one distinct thread services the sub-batches.

    A barrier forces all three sub-batch calls to be in flight simultaneously, so this
    proves genuine concurrency rather than merely observing >1 thread by scheduling luck.
    """
    barrier = threading.Barrier(3, timeout=5)

    def embed_batch_impl(texts, is_query):
        barrier.wait()
        return [[float(v)] for v in texts]

    ef = _make_ef(request_batch_size=2, max_parallel_requests=3, embed_batch_impl=embed_batch_impl)
    ef([0, 1, 2, 3, 4, 5])

    thread_names = {call["thread"] for call in ef.calls}
    assert len(thread_names) > 1


def test_output_order_matches_input_order_regardless_of_completion_order():
    """Sub-batches that finish out of order still land in their index-addressed slot.

    Earlier sub-batches are made to sleep LONGER than later ones (a real, tiny sleep), so
    later requests genuinely finish first -- yet the concatenated result must still be in
    input order. This pins that __call__ writes each sub-batch's result into a slot
    addressed by its original index, not by completion order.
    """
    docs = list(range(6))  # 3 sub-batches of size 2: [0,1] [2,3] [4,5]
    num_subbatches = 3

    def embed_batch_impl(texts, is_query):
        sub_index = texts[0] // 2
        time.sleep((num_subbatches - 1 - sub_index) * 0.02)
        return [[float(v)] for v in texts]

    ef = _make_ef(request_batch_size=2, max_parallel_requests=3, embed_batch_impl=embed_batch_impl)
    out = ef(docs)

    assert out == [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]


def test_sequential_path_when_max_parallel_requests_is_one():
    """max_parallel_requests <= 1 stays on the sequential path: every sub-batch runs on the
    calling thread itself, never a thread-pool worker."""
    calling_thread = threading.current_thread().name
    ef = _make_ef(
        request_batch_size=2,
        max_parallel_requests=1,
        embed_batch_impl=lambda texts, is_query: [[float(v)] for v in texts],
    )
    ef([0, 1, 2, 3, 4, 5])

    assert len(ef.calls) == 3
    assert {call["thread"] for call in ef.calls} == {calling_thread}


def test_single_subbatch_stays_sequential_even_with_parallelism_enabled():
    """A single sub-batch is not routed through the thread pool even when
    max_parallel_requests > 1."""
    calling_thread = threading.current_thread().name
    ef = _make_ef(
        request_batch_size=None,  # whole input as one sub-batch
        max_parallel_requests=4,
        embed_batch_impl=lambda texts, is_query: [[float(v)] for v in texts],
    )
    ef([0, 1, 2])

    assert len(ef.calls) == 1
    assert ef.calls[0]["thread"] == calling_thread


def test_parallel_output_is_bit_identical_to_sequential_output():
    """The same deterministic fake, run once sequentially and once with parallelism 4,
    produces the exact same output list."""
    docs = list(range(12))

    def embed_batch_impl(texts, is_query):
        return [[float(v)] for v in texts]

    ef_seq = _make_ef(request_batch_size=3, max_parallel_requests=1, embed_batch_impl=embed_batch_impl)
    ef_par = _make_ef(request_batch_size=3, max_parallel_requests=4, embed_batch_impl=embed_batch_impl)

    out_seq = ef_seq(docs)
    out_par = ef_par(docs)

    assert out_seq == out_par
    assert [float(v[0]) for v in out_seq] == [float(v[0]) for v in out_par] == [float(d) for d in docs]


# -- 3. Retries -----------------------------------------------------------------


def test_retryable_error_recovers_after_two_failures():
    """A _embed_batch that raises a retryable error twice then succeeds returns the
    successful result, having made exactly 3 total attempts."""
    clock = FakeClock()
    flaky = _FlakyBatch(error=_RetryableError("throttled"), fail_times=2, result=[[9.0]])
    ef = _make_ef(
        max_retries=5,
        embed_batch_impl=flaky,
        classify_error_impl=lambda exc: (True, None),
        limiter=AdaptiveRateLimiter(clock=clock.time, sleep=clock.sleep),
    )
    result = ef._embed_with_retry(["x"])
    assert result == [[9.0]]
    assert len(ef.calls) == 3


def test_retryable_error_that_never_clears_raises_after_max_retries():
    """An error that stays retryable forever raises after exactly max_retries retries: the
    attempt count is max_retries + 1."""
    clock = FakeClock()
    flaky = _FlakyBatch(error=_RetryableError("still throttled"), fail_times=10_000, result=[[0.0]])
    ef = _make_ef(
        max_retries=3,
        embed_batch_impl=flaky,
        classify_error_impl=lambda exc: (True, None),
        limiter=AdaptiveRateLimiter(clock=clock.time, sleep=clock.sleep),
    )
    with pytest.raises(_RetryableError):
        ef._embed_with_retry(["x"])
    assert len(ef.calls) == 3 + 1


def test_non_retryable_error_propagates_after_one_attempt():
    """The base _classify_error default of (False, None) means an unrecognized error
    propagates immediately -- no retries."""
    clock = FakeClock()

    def always_fails(texts, is_query):
        raise _FatalError("boom")

    ef = _make_ef(
        max_retries=5,
        embed_batch_impl=always_fails,
        # no classify_error_impl -> falls through to RemoteEmbeddingFunction's own default
        limiter=AdaptiveRateLimiter(clock=clock.time, sleep=clock.sleep),
    )
    with pytest.raises(_FatalError):
        ef._embed_with_retry(["x"])
    assert len(ef.calls) == 1


def test_retry_after_is_passed_to_limiter_and_honored():
    """retry_after returned from _classify_error is handed to the limiter's
    on_throttle/wait, which sleeps for exactly that long -- and nothing else."""
    clock = FakeClock()
    flaky = _FlakyBatch(error=_RetryableError("rate limited"), fail_times=1, result=[[1.0]])
    ef = _make_ef(
        max_retries=3,
        embed_batch_impl=flaky,
        classify_error_impl=lambda exc: (True, 7.5),
        limiter=AdaptiveRateLimiter(clock=clock.time, sleep=clock.sleep),
    )
    result = ef._embed_with_retry(["x"])
    assert result == [[1.0]]
    assert clock.slept == [7.5]


# -- 4. _embed_batch's two return shapes -----------------------------------------


def test_embed_batch_returning_bare_list_works():
    """A _embed_batch returning a plain list of vectors works unchanged."""
    clock = FakeClock()
    ef = _make_ef(
        embed_batch_impl=lambda texts, is_query: [[1.0], [2.0]],
        limiter=AdaptiveRateLimiter(clock=clock.time, sleep=clock.sleep),
    )
    result = ef._embed_with_retry(["a", "b"])
    assert result == [[1.0], [2.0]]


def test_embed_batch_returning_tuple_reaches_limiter_on_success():
    """A _embed_batch returning (vectors, headers) yields just the vectors, and the headers
    reach limiter.on_success unchanged."""
    clock = FakeClock()
    headers = {"x-ratelimit-remaining-tokens": "42"}
    limiter = _RecordingLimiter(clock=clock.time, sleep=clock.sleep)
    ef = _make_ef(
        embed_batch_impl=lambda texts, is_query: ([[1.0], [2.0]], headers),
        limiter=limiter,
    )
    result = ef._embed_with_retry(["a", "b"])
    assert result == [[1.0], [2.0]]
    assert limiter.on_success_calls == [headers]


# -- 5. embed_query ---------------------------------------------------------------


def test_embed_query_uses_prepare_query_not_prepare_document_and_unwraps_result():
    """embed_query routes through _prepare_query (never _prepare_document) and
    _embed_batch(..., is_query=True), returning a single vector rather than a list of
    one."""
    ef = _make_ef(
        prepare_document_impl=lambda text: text + "-DOC",
        prepare_query_impl=lambda text: text + "-QUERY",
        embed_batch_impl=lambda texts, is_query: [[42.0]],
    )
    result = ef.embed_query("hello")

    assert ef.calls[-1]["texts"] == ["hello-QUERY"]
    assert ef.calls[-1]["is_query"] is True
    assert result == [42.0]
    assert result != [[42.0]]


def test_embed_query_truncates_before_prepare_query_when_enabled():
    """With truncate_queries=True, the text is truncated BEFORE _prepare_query runs: the
    marker _prepare_query prepends survives intact at the front, and the body behind it is
    the part that gets cut."""
    body = "x" * 30
    ef = _make_ef(
        truncate_queries=True,
        max_input_tokens=5,  # chars_per_token=4 -> max_chars=20
        prepare_query_impl=lambda text: "MARK:" + text,
        embed_batch_impl=lambda texts, is_query: [[0.0]],
    )
    ef.embed_query(body)

    sent = ef.calls[-1]["texts"][0]
    assert sent == "MARK:" + "x" * 20


def test_embed_query_does_not_truncate_when_disabled():
    """truncate_queries=False (the default) means no truncation happens before
    _prepare_query."""
    body = "x" * 30
    ef = _make_ef(
        truncate_queries=False,
        max_input_tokens=5,
        prepare_query_impl=lambda text: "MARK:" + text,
        embed_batch_impl=lambda texts, is_query: [[0.0]],
    )
    ef.embed_query(body)

    sent = ef.calls[-1]["texts"][0]
    assert sent == "MARK:" + body


# -- 6. _prepare_document ---------------------------------------------------------


def test_call_applies_prepare_document_to_every_input():
    """__call__ applies _prepare_document to every input before batching/sending."""
    ef = _make_ef(
        request_batch_size=2,
        prepare_document_impl=lambda text: text.upper(),
        embed_batch_impl=lambda texts, is_query: [[0.0] for _ in texts],
    )
    ef(["ab", "cd", "ef"])

    all_texts = [t for call in ef.calls for t in call["texts"]]
    assert all_texts == ["AB", "CD", "EF"]


# -- 7. Config round-trip and the __new__-without-__init__ design property --------


def test_common_config_returns_all_five_keys():
    """_common_config() returns exactly the five COMMON_CONFIG_KEYS, with the values that
    were set."""
    ef = _make_ef(
        request_batch_size=64,
        rate_limit_rps=2.5,
        max_parallel_requests=3,
        max_retries=4,
        tokens_per_minute=1000.0,
    )
    cfg = ef._common_config()

    assert set(cfg.keys()) == set(COMMON_CONFIG_KEYS)
    assert cfg == {
        "request_batch_size": 64,
        "rate_limit_rps": 2.5,
        "max_parallel_requests": 3,
        "max_retries": 4,
        "tokens_per_minute": 1000.0,
    }


def test_new_without_init_survives_common_config_and_call():
    """An instance built via Cls.__new__(Cls), skipping __init__ entirely, can still call
    _common_config() and __call__ without raising AttributeError.

    This is an explicit design property, not an accident: several existing tests in this
    repo construct providers this way (see base.py's own module docstring), so every
    attribute RemoteEmbeddingFunction itself reads goes through getattr(..., default) and
    its rate limiter is built lazily on first use rather than in __init__.
    """
    ef = _ProgrammableEF.__new__(_ProgrammableEF)
    ef.calls = []
    ef._lock = threading.Lock()
    ef._embed_batch_impl = lambda texts, is_query: [[float(v)] for v in texts]
    # Deliberately nothing else set: no request_batch_size, max_parallel_requests,
    # max_retries, rate_limit_rps, tokens_per_minute, or limiter.

    cfg = ef._common_config()
    assert cfg["max_parallel_requests"] == RemoteEmbeddingFunction.max_parallel_requests_default
    assert cfg["max_retries"] == RemoteEmbeddingFunction.max_retries_default
    assert cfg["request_batch_size"] is None
    assert cfg["rate_limit_rps"] is None
    assert cfg["tokens_per_minute"] is None

    out = ef([0, 1, 2, 3])
    assert out == [[0.0], [1.0], [2.0], [3.0]]


# -- 8. _init_common's tokens_per_minute precedence --------------------------------


class _TpmEF(RemoteEmbeddingFunction):
    """Minimal RemoteEmbeddingFunction subclass with a known default_tokens_per_minute, for
    pinning _init_common's tokens_per_minute resolution precedence."""

    default_tokens_per_minute = 4242.0

    def _embed_batch(self, texts, is_query=False):
        return [[0.0] for _ in texts]


def _init_tpm(tokens_per_minute=None):
    ef = _TpmEF.__new__(_TpmEF)
    ef._init_common(
        model_name="fake-model",
        base_url=None,
        request_batch_size=None,
        rate_limit_rps=None,
        max_parallel_requests=None,
        max_retries=None,
        tokens_per_minute=tokens_per_minute,
    )
    return ef


def test_init_common_explicit_tokens_per_minute_wins(monkeypatch):
    """An explicit tokens_per_minute argument wins over both the env var and the class
    default."""
    monkeypatch.setenv("ZOTERO_TOKENS_PER_MINUTE", "999")
    ef = _init_tpm(tokens_per_minute=555.0)
    assert ef.tokens_per_minute == 555.0


def test_init_common_env_var_used_when_no_explicit_arg(monkeypatch):
    """With no explicit argument, ZOTERO_TOKENS_PER_MINUTE wins over the class default."""
    monkeypatch.setenv("ZOTERO_TOKENS_PER_MINUTE", "12345")
    ef = _init_tpm()
    assert ef.tokens_per_minute == 12345.0


def test_init_common_falls_back_to_class_default(monkeypatch):
    """With neither an explicit argument nor the env var set, tokens_per_minute falls back
    to the subclass's own default_tokens_per_minute."""
    monkeypatch.delenv("ZOTERO_TOKENS_PER_MINUTE", raising=False)
    ef = _init_tpm()
    assert ef.tokens_per_minute == _TpmEF.default_tokens_per_minute


def test_init_common_garbage_env_value_falls_back_to_class_default(monkeypatch):
    """An unparseable env value (e.g. "abc") is treated as absent, falling back to the class
    default rather than raising."""
    monkeypatch.setenv("ZOTERO_TOKENS_PER_MINUTE", "abc")
    ef = _init_tpm()
    assert ef.tokens_per_minute == _TpmEF.default_tokens_per_minute
