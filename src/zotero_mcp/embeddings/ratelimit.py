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

A second, independent bucket paces *tokens per minute* (``tpm``) rather than
requests per second. Unlike the RPS bucket, it is armed proactively from a
configured ceiling instead of learned reactively from a 429 — for embeddings,
the TPM ceiling is usually known up front (it's published per model/tier),
whereas a safe *request* rate depends on how large each request happens to
be. Pacing only by RPS is blind to request size: a handful of concurrent
large batch requests can exhaust an entire minute's token quota in seconds
even though the request rate looks tiny, producing a burst of silent 429s
that ``on_throttle`` then has to claw back from. Arming TPM pacing up front
prevents that burst instead of reacting to it after the fact. Passing no
``tpm`` leaves this dimension unarmed (``acquire()``'s token check is then a
no-op), exactly like the RPS dimension's ``initial_rps=None``.

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
        tpm: float | None = None,
        min_tpm: float = 100.0,
        max_tpm: float | None = None,
        token_burst: float | None = None,
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

        # Token-per-minute bucket: independent of the RPS one above, and
        # (unlike it) armed immediately when `tpm` is given rather than
        # waiting for a throttle — see module docstring. None means unarmed:
        # acquire()'s token check is a no-op, matching every existing caller
        # that never passes `estimated_tokens`.
        self._min_tps = min_tpm / 60.0
        self._max_tps = (max_tpm / 60.0) if max_tpm is not None else None
        self._tps: float | None = (tpm / 60.0) if tpm is not None else None
        # Default burst is a quarter of the per-minute budget: enough for
        # several concurrent requests to proceed without serializing to one
        # at a time, but far short of the full minute's budget so a
        # thundering herd of workers can't exhaust it in one instant (the
        # exact failure mode this bucket exists to prevent).
        default_token_burst = (tpm / 4.0) if tpm is not None else 0.0
        self._token_capacity = float(token_burst if token_burst is not None else default_token_burst)
        self._token_bucket: float = self._token_capacity
        self._token_last_refill = self._clock()

    def acquire(self, estimated_tokens: int = 0) -> None:
        """Block until a request slot is available (no-op while unarmed).

        ``estimated_tokens`` is only consulted when the TPM bucket is armed
        (``tpm`` was passed at construction); callers that never estimate
        token cost can omit it and get the pre-existing RPS-only behavior.

        The wait duration is computed under the lock; the actual sleep
        happens outside it so one thread waiting doesn't block another
        thread's bookkeeping.
        """
        with self._lock:
            self._record_request_ts()
            wait = 0.0
            if self._rps is not None:
                wait = max(wait, self._locked_take_token())
            if self._tps is not None and estimated_tokens > 0:
                wait = max(wait, self._locked_take_tokens(float(estimated_tokens)))

        if wait > 0:
            self._sleep(wait)

    def on_success(self) -> None:
        """Additive increase: current rate(s) creep up toward their max.

        No-op per dimension while that dimension is unarmed — an unthrottled
        provider has no rate to creep, it just keeps running at native speed.
        """
        with self._lock:
            if self._rps is not None:
                increment = max(self._rps * 0.05, 0.05)
                self._rps = self._rps + increment
                if self._max_rps is not None:
                    self._rps = min(self._rps, self._max_rps)
            if self._tps is not None:
                increment = max(self._tps * 0.05, 0.05)
                self._tps = self._tps + increment
                if self._max_tps is not None:
                    self._tps = min(self._tps, self._max_tps)
            self._consecutive_throttles = 0

    def on_throttle(self, retry_after: float | None) -> float:
        """Multiplicative decrease; returns the delay the caller should sleep.

        Arms the RPS rate (if unarmed) or halves it, floored at ``min_rps``
        and capped at ``max_rps``. Also halves the TPM rate the same way,
        but only if TPM pacing is armed — a throttle could be either kind of
        limit, and backing off both is cheap insurance either way.
        ``retry_after`` (parsed from the provider's response, when
        available) always wins over the exponential fallback — the provider
        is telling us exactly how long to wait.
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

            if self._tps is not None:
                self._tps = max(self._min_tps, self._tps * 0.5)
                if self._max_tps is not None:
                    self._tps = min(self._tps, self._max_tps)
                self._token_bucket = 0.0
                self._token_last_refill = self._clock()

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

    def _locked_take_tokens(self, n: float) -> float:
        now = self._clock()
        elapsed = now - self._token_last_refill
        self._token_last_refill = now
        self._token_bucket = min(self._token_capacity, self._token_bucket + elapsed * self._tps)
        # A single request larger than our own burst capacity can never be
        # fully "affordable" — cap what we wait for at the bucket's max so
        # such a request waits for a full bucket and then proceeds (going
        # into debt) instead of waiting forever for an unreachable balance.
        needed = min(n, self._token_capacity)
        if self._token_bucket >= needed:
            self._token_bucket = max(0.0, self._token_bucket - n)
            return 0.0
        deficit = needed - self._token_bucket
        wait = deficit / self._tps
        self._token_bucket = 0.0
        self._token_last_refill += wait
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
