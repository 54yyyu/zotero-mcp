"""The token wait must stay bounded however far the rate has decayed (#548).

Capacity was fixed from the initial tokens-per-minute while each 429 halved
the rate, and the wait is deficit / rate, so a run against Gemini's free tier
slept about 34 hours for one request and looked hung.
"""

import inspect

from zotero_mcp.embeddings.ratelimit import AdaptiveRateLimiter


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def _limiter(clock, **kw):
    return AdaptiveRateLimiter(tpm=950_000, clock=clock, sleep=clock.sleep, **kw)


def _throttle(limiter):
    # on_throttle's signature may take the response headers; pass nothing.
    params = inspect.signature(limiter.on_throttle).parameters
    limiter.on_throttle(*([None] * sum(1 for p in params.values()
                                       if p.default is inspect.Parameter.empty
                                       and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))))


def _succeed(limiter):
    params = inspect.signature(limiter.on_success).parameters
    limiter.on_success(*([None] * sum(1 for p in params.values()
                                      if p.default is inspect.Parameter.empty
                                      and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))))


def test_wait_stays_bounded_after_the_rate_collapses_to_its_floor():
    clock = _Clock()
    limiter = _limiter(clock)
    for _ in range(25):
        _throttle(limiter)
    assert limiter.tokens_per_minute == 100.0
    limiter.acquire(estimated_tokens=200_000)
    limiter.acquire(estimated_tokens=200_000)
    # A quarter-minute of budget at the current rate, not hours.
    assert clock.slept and max(clock.slept) <= 16.0


def test_explicit_token_burst_is_left_alone():
    clock = _Clock()
    limiter = _limiter(clock, token_burst=50_000)
    for _ in range(5):
        _throttle(limiter)
    assert limiter._token_capacity == 50_000


def test_capacity_grows_back_with_the_rate():
    clock = _Clock()
    limiter = _limiter(clock)
    for _ in range(10):
        _throttle(limiter)
    low = limiter._token_capacity
    for _ in range(50):
        _succeed(limiter)
    assert limiter._token_capacity > low
