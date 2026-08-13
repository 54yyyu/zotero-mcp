"""Adaptive AIMD rate limiter shared by the remote embedding providers.

Two independent token buckets live in one object, because the two ceilings a
provider enforces are not interchangeable:

- **Tokens per minute** (``tpm``) is the one that actually binds for
  embeddings. At a realistic 64 x ~500-token payload a single request costs
  ~32K tokens, so OpenAI's Tier 1 allowance of 1,000,000 TPM caps throughput
  near 31 requests/minute against a 3,000 RPM request allowance — a ~100x
  gap. Pacing by request rate alone is blind to request *size*: a few
  concurrent large batches can burn a minute's token quota in seconds while
  the request rate still looks trivial.
- **Requests per second** (``initial_rps``) matters only for providers or
  proxies that meter requests rather than tokens.

Each bucket is armed only when its ceiling is passed in, and an unarmed
bucket makes :meth:`acquire` a no-op for that dimension. Deliberately, a
throttle never *arms* an unarmed bucket: the TPM ceiling is published per
model and tier, so it is seeded from configuration rather than discovered by
taking 429s. Inferring a rate from the request cadence that just got
throttled means one early 429 can pin throughput far below the true ceiling
for the rest of a multi-hour run, with nothing to distinguish that from
correct behaviour.

On top of each armed bucket sits an AIMD controller: a successful request
nudges the rate up by 5%, a throttled one halves it. Both are clamped to the
configured ceiling, so the rate recovers *to* what the operator asked for and
never past it.

``clock``/``sleep`` are injectable so tests drive the limiter without waiting.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

# Ceiling for the exponential backoff used when a throttled request carries
# no Retry-After hint.
_MAX_BACKOFF_SECONDS = 60.0

# Fraction of the token bucket refilled early when a provider's response
# headers report ample headroom.
_HEADROOM_REFILL_FRACTION = 0.5

# Remaining-token ratio above which those headers count as "ample headroom".
_HEADROOM_THRESHOLD = 0.3


class AdaptiveRateLimiter:
    """Thread-safe token-bucket limiter with AIMD rate adaptation.

    One instance is shared by every worker thread embedding through a single
    :class:`~zotero_mcp.embeddings.base.RemoteEmbeddingFunction`, so ``burst``
    should be at least as large as that function's ``max_parallel_requests`` —
    otherwise parallel workers serialize behind a bucket that only ever holds
    one token.
    """

    def __init__(
        self,
        *,
        tpm: float | None = None,
        initial_rps: float | None = None,
        min_rps: float = 0.1,
        max_rps: float | None = None,
        burst: int = 4,
        min_tpm: float = 100.0,
        max_tpm: float | None = None,
        token_burst: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._consecutive_throttles = 0

        # Request bucket. None means unarmed, and stays unarmed: only an
        # explicitly configured rate_limit_rps ever turns this dimension on.
        self._min_rps = min_rps
        self._max_rps = max_rps
        self._burst = max(1, burst)
        self._rps: float | None = initial_rps
        self._tokens: float = float(self._burst)
        self._last_refill = self._clock()

        # Token bucket, in tokens per second. Armed from the configured TPM.
        self._min_tps = min_tpm / 60.0
        self._max_tps = (max_tpm / 60.0) if max_tpm is not None else None
        self._tps: float | None = (tpm / 60.0) if tpm is not None else None
        # Capacity defaults to a quarter of the per-minute budget: enough for
        # several concurrent requests to proceed without serializing to one at
        # a time, but far short of a full minute's budget, so a thundering
        # herd of workers cannot exhaust it in a single instant — the exact
        # failure this bucket exists to prevent.
        default_token_burst = (tpm / 4.0) if tpm is not None else 0.0
        self._token_capacity = float(
            token_burst if token_burst is not None else default_token_burst
        )
        self._token_bucket: float = self._token_capacity
        self._token_last_refill = self._clock()

    @property
    def tokens_per_minute(self) -> float | None:
        """Current TPM budget, or None while the token bucket is unarmed."""
        with self._lock:
            return None if self._tps is None else self._tps * 60.0

    @property
    def requests_per_second(self) -> float | None:
        """Current RPS budget, or None while the request bucket is unarmed."""
        with self._lock:
            return self._rps

    def acquire(self, estimated_tokens: int = 0) -> float:
        """Block until a request slot is available; return seconds waited.

        ``estimated_tokens`` is consulted only when the TPM bucket is armed.
        The wait is computed under the lock and slept outside it, so one
        thread waiting never blocks another thread's bookkeeping.
        """
        with self._lock:
            wait = 0.0
            if self._rps is not None:
                wait = max(wait, self._locked_take_request_token())
            if self._tps is not None and estimated_tokens > 0:
                wait = max(wait, self._locked_take_tokens(float(estimated_tokens)))

        if wait > 0:
            self._sleep(wait)
            return wait
        return 0.0

    def on_success(self, headers: Any | None = None) -> None:
        """Additive increase: each armed rate creeps 5% toward its ceiling.

        A no-op per dimension while that dimension is unarmed. When the
        provider's response headers report ample token headroom, the local
        token bucket is refilled early so parallel workers are not held back
        by a bucket that is more conservative than the provider itself.
        """
        with self._lock:
            if self._rps is not None:
                self._rps += max(self._rps * 0.05, 0.05)
                if self._max_rps is not None:
                    self._rps = min(self._rps, self._max_rps)

            if self._tps is not None:
                self._tps += max(self._tps * 0.05, 0.05)
                if self._max_tps is not None:
                    self._tps = min(self._tps, self._max_tps)
                self._locked_apply_header_headroom(headers)

            self._consecutive_throttles = 0

    def on_throttle(self, retry_after: float | None) -> float:
        """Multiplicative decrease; returns the delay the caller should sleep.

        Halves each *armed* rate, floored at its minimum and clamped to its
        ceiling, and discards accrued credit — a throttle means the budget
        that credit accrued under was too generous. An unarmed dimension stays
        unarmed: see the module docstring on why a 429 is not used to
        discover a rate.

        ``retry_after``, when the provider sent one, always wins over the
        exponential fallback: it is the provider stating exactly how long to
        wait.
        """
        with self._lock:
            if self._rps is not None:
                self._rps = max(self._min_rps, self._rps * 0.5)
                if self._max_rps is not None:
                    self._rps = min(self._rps, self._max_rps)
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
        """Sleep via the injected ``sleep``.

        Used for retry delays that happen outside the token-bucket path (e.g.
        honoring Retry-After) so every wait in this limiter goes through the
        same fake-clock seam.
        """
        if seconds and seconds > 0:
            self._sleep(seconds)

    # -- internals; all assume ``self._lock`` is held by the caller --------

    def _locked_take_request_token(self) -> float:
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
        # Pre-account for the wait we are about to ask the caller to sleep, so
        # the next acquire() does not also credit that elapsed time.
        self._last_refill += wait
        return wait

    def _locked_take_tokens(self, n: float) -> float:
        now = self._clock()
        elapsed = now - self._token_last_refill
        self._token_last_refill = now
        self._token_bucket = min(
            self._token_capacity, self._token_bucket + elapsed * self._tps
        )
        # A single request larger than the bucket's own capacity can never be
        # fully affordable, so cap what we wait for at capacity: such a request
        # waits for a full bucket and then proceeds into debt, rather than
        # waiting forever on a balance it can never reach.
        needed = min(n, self._token_capacity)
        if self._token_bucket >= needed:
            self._token_bucket = max(0.0, self._token_bucket - n)
            return 0.0
        deficit = needed - self._token_bucket
        wait = deficit / self._tps
        self._token_bucket = 0.0
        self._token_last_refill += wait
        return wait

    def _locked_apply_header_headroom(self, headers: Any | None) -> None:
        """Refill the token bucket early when headers report ample headroom.

        Best-effort and entirely optional. OpenAI's embeddings endpoint does
        not return ``x-ratelimit-*-tokens`` at all — a direct probe came back
        with only ``x-ratelimit-limit-requests`` — which is precisely why the
        bucket is seeded from configuration instead. This path exists so that
        providers which *do* send the headers are not paced more conservatively
        than they need to be; nothing depends on it.
        """
        if not headers or not hasattr(headers, "get"):
            return

        def get_header(name: str) -> Any:
            # Explicitly "first non-None", not an `or` chain: a legitimate
            # value of 0 is exactly the interesting case here (no headroom
            # left), and `or` would discard it as though the header were
            # absent.
            for key in (name, name.lower(), name.title()):
                value = headers.get(key)
                if value is not None:
                    return value
            return None

        try:
            remaining = get_header("x-ratelimit-remaining-tokens")
            limit = get_header("x-ratelimit-limit-tokens")
            if remaining is None or limit is None:
                return
            remaining_tokens = float(remaining)
            limit_tokens = float(limit)
        except (ValueError, TypeError):
            return

        if limit_tokens > 0 and (remaining_tokens / limit_tokens) > _HEADROOM_THRESHOLD:
            self._token_bucket = min(
                self._token_capacity,
                self._token_bucket + self._token_capacity * _HEADROOM_REFILL_FRACTION,
            )
