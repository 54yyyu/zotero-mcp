"""Unit tests for AdaptiveRateLimiter (token bucket + AIMD adaptation).

All tests drive the limiter with a fake clock/sleep pair so nothing here
actually waits in real time or touches the network.
"""

from zotero_mcp.embeddings.ratelimit import AdaptiveRateLimiter


class _FakeClock:
    """Monotonic clock that only advances when `advance()` is called."""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeSleeper:
    """Records sleep durations and advances the paired clock instead of
    actually blocking, so acquire() calls that would wait resolve instantly."""

    def __init__(self, clock: _FakeClock):
        self.clock = clock
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.advance(seconds)


def _make_limiter(**kwargs):
    clock = _FakeClock()
    sleeper = _FakeSleeper(clock)
    limiter = AdaptiveRateLimiter(clock=clock, sleep=sleeper, **kwargs)
    return limiter, clock, sleeper


def test_token_bucket_paces_requests_at_configured_rps():
    """Once burst tokens are exhausted, acquire() waits ~1/rps between calls."""
    limiter, clock, sleeper = _make_limiter(initial_rps=2.0, burst=1)

    limiter.acquire()  # consumes the single burst token, no wait
    assert sleeper.calls == []

    limiter.acquire()  # bucket empty: must wait ~0.5s for the next token
    assert len(sleeper.calls) == 1
    assert abs(sleeper.calls[0] - 0.5) < 1e-9

    limiter.acquire()
    assert len(sleeper.calls) == 2
    assert abs(sleeper.calls[1] - 0.5) < 1e-9


def test_burst_allows_n_immediate_acquires():
    """The first `burst` acquires must not wait even with the clock frozen."""
    limiter, clock, sleeper = _make_limiter(initial_rps=1.0, burst=4)

    for _ in range(4):
        limiter.acquire()
    assert sleeper.calls == [], "burst capacity should absorb the first 4 acquires"

    # 5th acquire exhausts the bucket and must wait for a token to accrue.
    limiter.acquire()
    assert len(sleeper.calls) == 1
    assert sleeper.calls[0] > 0


def test_initial_rps_none_is_unlimited_until_first_throttle():
    """acquire() must be a true no-op while unarmed."""
    limiter, clock, sleeper = _make_limiter(initial_rps=None, burst=1)

    for _ in range(20):
        limiter.acquire()
    assert sleeper.calls == [], "unarmed limiter must never sleep"


def test_first_throttle_arms_from_observed_rate():
    """After several fast acquires, on_throttle() with no prior rate arms the
    limiter using the observed request cadence rather than a fixed constant."""
    limiter, clock, sleeper = _make_limiter(initial_rps=None, burst=100)

    # 11 acquires spaced 0.1s apart -> an observed rate of ~10 rps.
    for _ in range(11):
        limiter.acquire()
        clock.advance(0.1)

    delay = limiter.on_throttle(retry_after=None)
    # The very first throttle just arms the limiter; it does not itself
    # imply a wait beyond the backoff/no-retry-after fallback.
    assert delay >= 0
    # Confirm it actually armed: a fresh acquire() sequence should now be
    # rate-limited around the observed rate rather than unlimited.
    sleeper.calls.clear()
    for _ in range(200):
        limiter.acquire()
    assert sleeper.calls, "limiter should be armed (paced) after its first throttle"


def test_first_throttle_falls_back_to_1rps_with_no_observations():
    """With no prior acquire() calls to observe, arming falls back to a sane
    default rather than raising or staying unarmed forever. Arming also
    drains the token bucket — a throttle means "slow down now", so it must
    not immediately permit a fresh burst of requests."""
    limiter, clock, sleeper = _make_limiter(initial_rps=None, burst=4)
    limiter.on_throttle(retry_after=None)
    limiter.acquire()
    assert len(sleeper.calls) == 1
    assert abs(sleeper.calls[0] - 1.0) < 1e-6


def test_on_throttle_halves_rate_floored_at_min_rps():
    limiter, clock, sleeper = _make_limiter(initial_rps=1.0, min_rps=0.2, burst=1)
    limiter.on_throttle(retry_after=None)
    assert limiter._rps == 0.5
    limiter.on_throttle(retry_after=None)
    assert limiter._rps == 0.25
    limiter.on_throttle(retry_after=None)
    # 0.25 * 0.5 = 0.125, floored at min_rps=0.2
    assert limiter._rps == 0.2
    limiter.on_throttle(retry_after=None)
    assert limiter._rps == 0.2


def test_on_throttle_returns_retry_after_verbatim_when_provided():
    limiter, clock, sleeper = _make_limiter(initial_rps=5.0, burst=1)
    delay = limiter.on_throttle(retry_after=12.5)
    assert delay == 12.5


def test_on_throttle_uses_exponential_backoff_without_retry_after():
    limiter, clock, sleeper = _make_limiter(initial_rps=5.0, burst=1)
    d1 = limiter.on_throttle(retry_after=None)
    d2 = limiter.on_throttle(retry_after=None)
    d3 = limiter.on_throttle(retry_after=None)
    assert [d1, d2, d3] == [1.0, 2.0, 4.0]


def test_on_success_creeps_up_but_never_exceeds_max_rps():
    limiter, clock, sleeper = _make_limiter(initial_rps=1.0, max_rps=1.2, burst=1)
    for _ in range(100):
        limiter.on_success()
    assert limiter._rps <= 1.2
    assert limiter._rps == 1.2  # should have saturated at the cap


def test_on_success_noop_while_unarmed():
    limiter, clock, sleeper = _make_limiter(initial_rps=None, burst=1)
    limiter.on_success()
    assert limiter._rps is None


def test_max_rps_is_a_hard_cap_even_after_recovery_from_throttle():
    """Simulates the OpenAI wiring: user-configured rate_limit_rps is both
    initial_rps and max_rps, so recovery must never exceed the user's cap."""
    limiter, clock, sleeper = _make_limiter(initial_rps=4.0, max_rps=4.0, burst=1)
    limiter.on_throttle(retry_after=None)
    assert limiter._rps == 2.0
    for _ in range(1000):
        limiter.on_success()
    assert limiter._rps == 4.0


# -- TPM bucket: proactive, tokens-per-minute pacing ------------------------
#
# Unlike RPS, this dimension is armed immediately from a configured `tpm`
# rather than learned from a throttle — see ratelimit.py's module docstring
# for why (a burst of concurrent large requests can exhaust a whole minute's
# token budget before RPS-only pacing would ever notice).


def test_tpm_unset_makes_estimated_tokens_a_noop():
    """No tpm configured: acquire() must ignore estimated_tokens entirely,
    matching pre-existing RPS-only behavior exactly."""
    limiter, clock, sleeper = _make_limiter(initial_rps=None)
    for _ in range(20):
        limiter.acquire(estimated_tokens=1_000_000)
    assert sleeper.calls == []


def test_tpm_armed_from_construction_no_throttle_needed():
    """Unlike RPS, a configured tpm paces from the very first acquire()."""
    limiter, clock, sleeper = _make_limiter(initial_rps=None, tpm=60.0, token_burst=10.0)
    limiter.acquire(estimated_tokens=10)  # exactly drains the 10-token burst
    assert sleeper.calls == []
    limiter.acquire(estimated_tokens=10)  # bucket empty: tps = 1.0, so 10s wait
    assert len(sleeper.calls) == 1
    assert abs(sleeper.calls[0] - 10.0) < 1e-9


def test_tpm_default_burst_is_quarter_of_budget():
    limiter, clock, sleeper = _make_limiter(initial_rps=None, tpm=400.0)
    assert limiter._token_capacity == 100.0


def test_tpm_and_rps_wait_independently_max_wins():
    """Both dimensions armed: acquire() must wait for whichever is more
    restrictive, not sum or ignore either one."""
    limiter, clock, sleeper = _make_limiter(
        initial_rps=100.0, burst=1, tpm=60.0, token_burst=1.0,
    )
    limiter.acquire(estimated_tokens=1)  # drains both burst=1 buckets
    assert sleeper.calls == []
    limiter.acquire(estimated_tokens=1)
    # RPS wait ~= 1/100 = 0.01s; TPM wait = 1/(60/60) = 1.0s. Must take the max.
    assert len(sleeper.calls) == 1
    assert abs(sleeper.calls[0] - 1.0) < 1e-6


def test_tpm_oversized_single_request_does_not_deadlock():
    """A request bigger than the whole burst capacity must wait for the
    bucket to fill (not forever) and then be let through rather than hang."""
    limiter, clock, sleeper = _make_limiter(initial_rps=None, tpm=60.0, token_burst=5.0)
    limiter.acquire(estimated_tokens=50)  # bucket starts full (5.0 >= needed=5.0): no wait
    assert sleeper.calls == []
    limiter.acquire(estimated_tokens=50)  # bucket now drained: must wait for it to refill to capacity
    assert len(sleeper.calls) == 1
    # tps = 1.0 token/sec; waiting for the bucket to reach its 5.0 cap takes 5s.
    assert abs(sleeper.calls[0] - 5.0) < 1e-9


def test_on_throttle_halves_tpm_only_when_armed():
    limiter, clock, sleeper = _make_limiter(initial_rps=None, tpm=600.0, min_tpm=1.0)
    assert limiter._tps == 10.0
    limiter.on_throttle(retry_after=None)
    assert limiter._tps == 5.0


def test_on_throttle_is_noop_for_tpm_when_unarmed():
    limiter, clock, sleeper = _make_limiter(initial_rps=1.0, burst=1)
    limiter.on_throttle(retry_after=None)
    assert limiter._tps is None


def test_on_success_grows_tpm_toward_max_tpm():
    limiter, clock, sleeper = _make_limiter(
        initial_rps=None, tpm=60.0, max_tpm=120.0, min_tpm=0.1,
    )
    limiter.on_throttle(retry_after=None)  # halve to 0.5 tps
    for _ in range(200):
        limiter.on_success()
    assert limiter._tps == 2.0  # max_tpm=120 -> cap of 2.0 tokens/sec
