"""Adaptive AIMD rate limiter shared by realtime embedding providers.

A token-bucket limiter paces outgoing HTTP requests at a target rate while
still allowing a small burst so parallel workers are not serialized waiting
on a single token. On top of the bucket sits an AIMD (additive-increase /
multiplicative-decrease) controller: a successful request nudges the rate up
toward ``max_rps``; a throttled request (429/5xx) halves it, floored at
``min_rps``.

When no rate is configured (``initial_rps=None``), the limiter starts
*unarmed*: ``acquire()`` is a no-op and requests run at native speed. The
first throttle arms it, seeded at half the request rate actually observed up
to that point (or a conservative 1.0 rps fallback) rather than an arbitrary
constant — half, because the observed rate is exactly what the provider just
rejected.

``clock``/``sleep`` are injectable so tests can drive the limiter without
real waiting.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

# How many recent request timestamps to keep for estimating the pre-throttle
# request rate. Small and fixed — this is a rough seed for the AIMD
# controller, not a precise measurement.
_RATE_ESTIMATE_WINDOW = 50

# Ceiling for the exponential backoff used when a throttled request carries
# no Retry-After hint.
_MAX_BACKOFF_SECONDS = 60.0


class AdaptiveRateLimiter:
    """Thread-safe token-bucket limiter with AIMD rate adaptation.

    One instance is shared by every worker (thread) embedding through a
    single ``RemoteEmbeddingFunction`` instance, so ``burst`` should be at
    least as large as the embedding function's ``max_parallel_requests`` —
    otherwise parallel workers would just queue up behind a bucket that only
    ever holds one token.
    """

    def __init__(
        self,
        initial_rps: float | None,
        min_rps: float = 0.1,
        max_rps: float | None = None,
        burst: int = 4,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._min_rps = min_rps
        self._max_rps = max_rps
        self._burst = max(1, burst)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()

        # None means "unarmed": acquire() is a no-op until the first throttle.
        self._rps: float | None = initial_rps
        self._tokens: float = float(self._burst)
        self._last_refill = self._clock()
        self._consecutive_throttles = 0
        self._recent_request_ts: list[float] = []

    def acquire(self) -> None:
        """Block until a request slot is available (no-op while unarmed).

        The wait duration is computed under the lock; the actual sleep
        happens outside it so one thread waiting doesn't block another
        thread's bookkeeping.
        """
        with self._lock:
            self._record_request_ts()
            if self._rps is None:
                return
            wait = self._locked_take_token()

        if wait > 0:
            self._sleep(wait)

    def on_success(self) -> None:
        """Additive increase: current rate creeps up toward ``max_rps``.

        No-op while unarmed — an unthrottled provider has no rate to creep,
        it just keeps running at native speed.
        """
        with self._lock:
            if self._rps is None:
                return
            increment = max(self._rps * 0.05, 0.05)
            self._rps = self._rps + increment
            if self._max_rps is not None:
                self._rps = min(self._rps, self._max_rps)
            self._consecutive_throttles = 0

    def on_throttle(self, retry_after: float | None) -> float:
        """Multiplicative decrease; returns the delay the caller should sleep.

        Arms the limiter (if unarmed) or halves the current rate, floored at
        ``min_rps`` and capped at ``max_rps``. ``retry_after`` (parsed from
        the provider's response, when available) always wins over the
        exponential fallback — the provider is telling us exactly how long
        to wait.
        """
        with self._lock:
            if self._rps is None:
                self._rps = self._estimate_initial_rps()
            else:
                self._rps = max(self._min_rps, self._rps * 0.5)
            if self._max_rps is not None:
                self._rps = min(self._rps, self._max_rps)
            # Discard accrued tokens: a throttle means we were just told to
            # slow down, so the next acquire() should not spend a token that
            # accrued at the (too fast) old rate.
            self._tokens = 0.0
            self._last_refill = self._clock()
            self._consecutive_throttles += 1
            consecutive = self._consecutive_throttles

        if retry_after is not None:
            return retry_after
        return min(_MAX_BACKOFF_SECONDS, 2.0 ** (consecutive - 1))

    def wait(self, seconds: float) -> None:
        """Sleep via the injected ``sleep`` fn (used for retry delays that
        happen outside the token-bucket path, e.g. honoring Retry-After), so
        every wait in this limiter goes through the same fake-clock seam.
        """
        if seconds and seconds > 0:
            self._sleep(seconds)

    # -- internals, all assume `self._lock` is held by the caller ----------

    def _record_request_ts(self) -> None:
        ts = self._recent_request_ts
        ts.append(self._clock())
        if len(ts) > _RATE_ESTIMATE_WINDOW:
            del ts[: len(ts) - _RATE_ESTIMATE_WINDOW]

    def _locked_take_token(self) -> float:
        now = self._clock()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._tokens = min(float(self._burst), self._tokens + elapsed * self._rps)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return 0.0
        deficit = 1.0 - self._tokens
        wait = deficit / self._rps
        self._tokens = 0.0
        # Pre-account for the wait we're about to ask the caller to sleep so
        # the next acquire() doesn't double-count that elapsed time.
        self._last_refill += wait
        return wait

    def _estimate_initial_rps(self) -> float:
        # Seed at HALF the observed rate: the observed cadence is precisely
        # what the provider just throttled, so continuing at it would only
        # buy a second 429 before the first multiplicative decrease kicks in.
        ts = self._recent_request_ts
        if len(ts) >= 2:
            span = ts[-1] - ts[0]
            if span > 0:
                observed = (len(ts) - 1) / span
                if observed > 0:
                    return max(self._min_rps, observed * 0.5)
        return 1.0
