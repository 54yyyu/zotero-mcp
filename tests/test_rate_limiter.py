"""Tests for AdaptiveRateLimiter (src/zotero_mcp/embeddings/ratelimit.py).

Every test drives the limiter with a fake, manually-advanced clock so the
suite never sleeps for real and never depends on wall-clock timing. Coverage:

1. Unarmed-by-default no-op behavior.
2. Token bucket pacing arithmetic (hand-computed expected waits).
3. Oversized single request does not hang (the min(n, capacity) behavior).
4. on_throttle halving + retry_after / exponential-backoff / cap semantics.
5. on_throttle floors at min_tpm.
6. on_success additive increase clamped at the configured ceiling.
7. Anti-regression: on_throttle never arms an unconfigured RPS bucket.
8. An explicitly configured RPS bucket does pace.
9. Header-headroom refill is best-effort and never raises on bad input.
10. Thread safety under concurrent acquire() calls.
"""

import threading

import pytest

from zotero_mcp.embeddings.ratelimit import AdaptiveRateLimiter


class FakeClock:
    """Manually advanced monotonic clock plus a sleep that advances it.

    Internally locked so it can be shared safely across threads in the
    concurrency test: acquire() calls the injected sleep *outside* the
    limiter's own lock, so multiple threads can call FakeClock.sleep()
    concurrently with each other and with FakeClock.time() reads happening
    inside another thread's locked section.
    """

    def __init__(self, start=1000.0):
        self.start = start
        self.now = start
        self.slept = []  # every duration passed to sleep()
        self._guard = threading.Lock()

    def time(self):
        with self._guard:
            return self.now

    def sleep(self, seconds):
        with self._guard:
            self.slept.append(seconds)
            self.now += seconds

    def advance(self, seconds):
        with self._guard:
            self.now += seconds


# -- 1. Unarmed by default ---------------------------------------------------


def test_unarmed_limiter_is_a_complete_noop():
    """No tpm and no initial_rps: acquire() never waits, both rate properties are None."""
    clock = FakeClock()
    limiter = AdaptiveRateLimiter(clock=clock.time, sleep=clock.sleep)

    assert limiter.tokens_per_minute is None
    assert limiter.requests_per_second is None

    assert limiter.acquire() == 0.0
    assert limiter.acquire(estimated_tokens=50_000) == 0.0
    assert clock.slept == []


# -- 2. Token bucket paces to the configured TPM -----------------------------


def test_token_bucket_paces_to_configured_tpm():
    """acquire() waits deficit/tokens_per_second once the burst is drained, then refills over time."""
    clock = FakeClock()
    # tpm=60_000 -> 1000 tokens/sec; token_burst gives an exact capacity to reason about.
    limiter = AdaptiveRateLimiter(
        tpm=60_000.0, token_burst=1000.0, clock=clock.time, sleep=clock.sleep
    )

    # Bucket starts full: a request inside the burst is free.
    assert limiter.acquire(estimated_tokens=600) == 0.0
    # Drain the remainder of the burst.
    assert limiter.acquire(estimated_tokens=400) == 0.0
    assert clock.slept == []

    # Bucket is now empty; the next request must wait for its deficit at
    # 1000 tokens/sec: deficit=500 tokens -> wait = 500 / 1000 = 0.5s.
    wait = limiter.acquire(estimated_tokens=500)
    expected_wait = 500.0 / 1000.0
    assert wait == pytest.approx(expected_wait)
    assert clock.slept[-1] == pytest.approx(expected_wait)

    # Advancing the clock further (beyond the sleep already accounted for)
    # refills the bucket, so a small subsequent request is free again.
    clock.advance(1.0)
    assert limiter.acquire(estimated_tokens=1) == 0.0


# -- 3. Oversized request does not hang --------------------------------------


def test_oversized_request_waits_for_full_bucket_not_forever():
    """estimated_tokens far larger than token_burst waits only capacity/tps, never hangs on an unreachable balance."""
    clock = FakeClock()
    limiter = AdaptiveRateLimiter(
        tpm=60_000.0, token_burst=100.0, clock=clock.time, sleep=clock.sleep
    )
    huge = 100_000

    # First call: bucket starts full (== capacity, which is >= needed), so it
    # is granted immediately -- and drives the bucket into debt.
    assert limiter.acquire(estimated_tokens=huge) == 0.0

    # Second call: bucket is empty. Because needed = min(n, capacity), the
    # wait is capacity/tps, not (huge - bucket)/tps -- finite and small
    # regardless of how large `huge` is.
    wait = limiter.acquire(estimated_tokens=huge)
    expected = 100.0 / 1000.0  # capacity / tokens_per_second
    assert wait == pytest.approx(expected)
    assert wait < 1.0


# -- 4. on_throttle halving, retry_after, exponential backoff, cap ----------


def test_on_throttle_halves_both_armed_dimensions():
    """A single on_throttle halves TPM and RPS together when both are armed."""
    clock = FakeClock()
    limiter = AdaptiveRateLimiter(
        tpm=1000.0, initial_rps=10.0, clock=clock.time, sleep=clock.sleep
    )
    limiter.on_throttle(None)
    assert limiter.tokens_per_minute == pytest.approx(500.0)
    assert limiter.requests_per_second == pytest.approx(5.0)


def test_on_throttle_returns_retry_after_when_given():
    """retry_after, when provided, always wins over the exponential fallback."""
    clock = FakeClock()
    limiter = AdaptiveRateLimiter(tpm=1000.0, clock=clock.time, sleep=clock.sleep)
    delay = limiter.on_throttle(retry_after=12.5)
    assert delay == 12.5
    assert limiter.tokens_per_minute == pytest.approx(500.0)


def test_on_throttle_exponential_backoff_without_retry_after():
    """With no retry_after, consecutive throttles back off as 1, 2, 4, 8 seconds."""
    clock = FakeClock()
    limiter = AdaptiveRateLimiter(tpm=1_000_000.0, clock=clock.time, sleep=clock.sleep)
    delays = [limiter.on_throttle(None) for _ in range(4)]
    assert delays == [1.0, 2.0, 4.0, 8.0]


def test_on_throttle_backoff_capped_at_sixty_seconds():
    """The exponential fallback never exceeds 60 seconds, however many consecutive throttles occur."""
    clock = FakeClock()
    limiter = AdaptiveRateLimiter(
        tpm=1_000_000_000.0, clock=clock.time, sleep=clock.sleep
    )
    delays = [limiter.on_throttle(None) for _ in range(10)]
    assert delays[-1] == 60.0
    assert max(delays) == 60.0


def test_on_success_resets_consecutive_throttle_counter():
    """on_success() resets the backoff counter so the next throttle starts over at 1.0s."""
    clock = FakeClock()
    limiter = AdaptiveRateLimiter(tpm=1_000_000.0, clock=clock.time, sleep=clock.sleep)
    assert limiter.on_throttle(None) == 1.0
    assert limiter.on_throttle(None) == 2.0
    limiter.on_success()
    assert limiter.on_throttle(None) == 1.0


# -- 5. on_throttle floors at min_tpm ----------------------------------------


def test_on_throttle_floors_at_min_tpm():
    """Repeated halving never drives tokens_per_minute below the configured min_tpm."""
    clock = FakeClock()
    limiter = AdaptiveRateLimiter(
        tpm=1000.0, min_tpm=100.0, clock=clock.time, sleep=clock.sleep
    )
    for _ in range(30):
        limiter.on_throttle(None)
    assert limiter.tokens_per_minute >= 100.0
    assert limiter.tokens_per_minute == pytest.approx(100.0)


# -- 6. on_success additive increase, clamped at ceiling ---------------------


def test_on_success_additive_increase_clamped_to_ceiling():
    """on_success() grows the rate by 5% per call but never exceeds max_tpm."""
    clock = FakeClock()
    limiter = AdaptiveRateLimiter(
        tpm=1000.0, max_tpm=1000.0, clock=clock.time, sleep=clock.sleep
    )
    limiter.on_throttle(None)  # halve to 500
    assert limiter.tokens_per_minute == pytest.approx(500.0)

    previous = 500.0
    for _ in range(50):
        limiter.on_success()
        current = limiter.tokens_per_minute
        assert current >= previous - 1e-9
        assert current <= 1000.0 + 1e-9
        previous = current

    assert limiter.tokens_per_minute == pytest.approx(1000.0)


# -- 7. Anti-regression: throttle never arms an unconfigured RPS bucket -----


def test_on_throttle_never_arms_an_unconfigured_rps_bucket():
    """Anti-regression for arm-on-first-429.

    An earlier implementation left the RPS bucket unarmed until the first 429
    and then seeded it at half the observed request cadence, which let one
    early 429 pin throughput far below the true ceiling for a whole run. This
    pins the fix: initial_rps=None must stay None through on_throttle, and a
    subsequent acquire() must remain a complete no-op for the request
    dimension. The complementary case -- an armed TPM dimension DOES halve on
    the same throttle -- is asserted too, to show throttling still works for
    the dimension that was actually configured.
    """
    clock = FakeClock()
    limiter = AdaptiveRateLimiter(
        tpm=60_000.0, initial_rps=None, clock=clock.time, sleep=clock.sleep
    )
    assert limiter.requests_per_second is None

    limiter.on_throttle(5.0)

    assert limiter.requests_per_second is None
    wait = limiter.acquire()  # estimated_tokens=0 -> token dimension is also a no-op here
    assert wait == 0.0
    assert clock.slept == []

    # Complementary: the dimension that *was* configured does get halved.
    assert limiter.tokens_per_minute == pytest.approx(30_000.0)


# -- 8. An explicitly configured RPS bucket does pace ------------------------


def test_configured_rps_bucket_paces_requests():
    """With initial_rps armed, draining the burst forces the next acquire to wait ~1/rps."""
    clock = FakeClock()
    limiter = AdaptiveRateLimiter(
        initial_rps=10.0, max_rps=10.0, burst=2, clock=clock.time, sleep=clock.sleep
    )
    assert limiter.acquire() == 0.0
    assert limiter.acquire() == 0.0

    wait = limiter.acquire()
    assert wait == pytest.approx(0.1)
    assert clock.slept[-1] == pytest.approx(0.1)


# -- 9. Header headroom refill is best-effort --------------------------------


def test_on_success_header_headroom_tops_up_the_bucket():
    """on_success(headers) with ample reported headroom refills the bucket early."""
    clock = FakeClock()
    limiter = AdaptiveRateLimiter(
        tpm=60_000.0, token_burst=1000.0, clock=clock.time, sleep=clock.sleep
    )
    # Drain the bucket completely.
    assert limiter.acquire(estimated_tokens=1000) == 0.0
    wait_before = limiter.acquire(estimated_tokens=500)
    assert wait_before > 0.0

    headers = {
        "x-ratelimit-remaining-tokens": "900000",
        "x-ratelimit-limit-tokens": "1000000",
    }
    limiter.on_success(headers)

    # Bucket was topped up early; a modest request is now free again.
    wait_after = limiter.acquire(estimated_tokens=100)
    assert wait_after == 0.0


def test_on_success_headroom_headers_case_insensitive_lookup():
    """Title-Case header keys are still recognized (get_header tries name, lower, and title)."""
    clock = FakeClock()
    limiter = AdaptiveRateLimiter(
        tpm=60_000.0, token_burst=1000.0, clock=clock.time, sleep=clock.sleep
    )
    assert limiter.acquire(estimated_tokens=1000) == 0.0  # drain to empty

    headers = {
        "X-Ratelimit-Remaining-Tokens": "900000",
        "X-Ratelimit-Limit-Tokens": "1000000",
    }
    limiter.on_success(headers)

    assert limiter.acquire(estimated_tokens=100) == 0.0


@pytest.mark.parametrize(
    "headers",
    [
        None,
        {},
        {"x-ratelimit-remaining-tokens": "not-a-number", "x-ratelimit-limit-tokens": "1000"},
        {"x-ratelimit-remaining-tokens": "10", "x-ratelimit-limit-tokens": "0"},
        {"x-ratelimit-remaining-tokens": None, "x-ratelimit-limit-tokens": None},
    ],
)
def test_on_success_headroom_refill_never_raises_on_bad_input(headers):
    """on_success must be robust to missing, empty, non-numeric, or zero-limit headers."""
    clock = FakeClock()
    limiter = AdaptiveRateLimiter(tpm=60_000.0, clock=clock.time, sleep=clock.sleep)
    limiter.on_success(headers)  # must not raise


# -- 10. Thread safety --------------------------------------------------------


def test_concurrent_acquire_is_thread_safe_and_bounded():
    """8 threads hammering acquire() concurrently must not raise, and must never
    let more tokens through without waiting than the bucket (plus any elapsed
    refill) could actually supply."""
    clock = FakeClock()
    capacity = 200.0
    tpm = 60_000.0  # tps = 1000 tokens/sec
    limiter = AdaptiveRateLimiter(
        tpm=tpm, token_burst=capacity, clock=clock.time, sleep=clock.sleep
    )

    result_lock = threading.Lock()
    errors = []
    granted_immediately = []  # tokens let through with wait == 0.0

    def worker():
        try:
            for _ in range(5):
                n = 20
                wait = limiter.acquire(estimated_tokens=n)
                if wait == 0.0:
                    with result_lock:
                        granted_immediately.append(n)
        except Exception as exc:  # pragma: no cover - failure path
            with result_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors
    assert all(not t.is_alive() for t in threads)

    elapsed = clock.now - clock.start
    tokens_per_second = tpm / 60.0
    max_supply = capacity + tokens_per_second * elapsed + 1e-6
    assert sum(granted_immediately) <= max_supply
