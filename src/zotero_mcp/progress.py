"""Thread-safe single-line progress reporting for long-running scans.

The fulltext extraction phase historically rendered progress with ad-hoc
``\\r`` writes inside a sequential loop. With multicore extraction several
worker threads finish at unpredictable times, so both the counters and the
terminal rendering must be guarded by one lock — otherwise counter updates
are lost and two threads interleave partial ``\\r`` lines.
"""

from __future__ import annotations

import os
import sys
import threading


class ExtractionProgress:
    """Counters plus a single-line stderr progress renderer.

    ``record(outcome=..., display=...)`` may be called from any thread; each
    call increments ``completed`` and the named outcome counter, then redraws
    the progress line. Outcome names are free-form — the summary labels below
    control what shows up in the rendered line.
    """

    # Rendering labels for the parenthesized status, in display order.
    # Outcomes not listed here still count, but are not shown inline.
    _STATUS_LABELS = [
        ("cache_hit", "cached"),
        ("extracted", "extracted"),
        ("skipped_existing", "up to date"),
        ("skipped_failed", "skipped"),
        ("timeout", "timed out"),
        ("failed", "failed"),
    ]

    def __init__(self, total: int, stream=None, prefix: str = "  Processing"):
        self.total = total
        self.completed = 0
        self.counters: dict[str, int] = {}
        self._lock = threading.Lock()
        self._stream = stream if stream is not None else sys.stderr
        self._prefix = prefix

    def record(self, outcome: str, display: str = "") -> None:
        """Count one completed item and redraw the progress line."""
        with self._lock:
            self.completed += 1
            self.counters[outcome] = self.counters.get(outcome, 0) + 1
            self._render(display)

    def count(self, outcome: str) -> int:
        with self._lock:
            return self.counters.get(outcome, 0)

    def _term_width(self) -> int:
        try:
            return os.get_terminal_size().columns
        except (OSError, ValueError):
            return 80

    def _render(self, display: str) -> None:
        # Caller holds the lock; rendering inside it keeps concurrent \r
        # writes from interleaving on the same terminal line.
        try:
            max_len = self._term_width() - 1
            status_parts = [
                f"{self.counters[key]} {label}"
                for key, label in self._STATUS_LABELS
                if self.counters.get(key)
            ]
            status = f" ({', '.join(status_parts)})" if status_parts else ""
            prefix = f"{self._prefix} {self.completed}/{self.total}{status} — "
            remaining = max_len - len(prefix) - 3
            if remaining > 0 and display and len(display) > remaining:
                display = display[:remaining] + "..."
            line = f"{prefix}{display or 'working...'}"
            if len(line) > max_len:
                line = line[:max_len]
            self._stream.write(f"\r{line}{' ' * max(0, max_len - len(line))}")
            self._stream.flush()
        except Exception:
            pass

    def clear_line(self) -> None:
        with self._lock:
            try:
                self._stream.write(f"\r{' ' * (self._term_width() - 1)}\r")
                self._stream.flush()
            except Exception:
                pass

    def finish(self) -> dict[str, int]:
        """Clear the progress line and return a snapshot of all counters."""
        self.clear_line()
        with self._lock:
            snapshot = dict(self.counters)
            snapshot["completed"] = self.completed
            return snapshot


class ConsecutiveFailureBreaker:
    """Thread-safe circuit breaker over consecutive failures.

    Mirrors the sequential "stop after N consecutive timeouts" behavior:
    ``record(True)`` increments, ``record(False)`` resets, and the breaker
    trips once the count reaches ``threshold``. Under concurrency the counter
    advances in completion order, which is the natural analogue of
    "consecutive" when several extractions are in flight at once. Once
    tripped, the breaker stays tripped.
    """

    def __init__(self, threshold: int):
        self.threshold = threshold
        self._count = 0
        self._lock = threading.Lock()
        self._tripped = threading.Event()

    def record(self, is_failure: bool) -> bool:
        """Record one outcome; returns True when this call trips the breaker."""
        if self._tripped.is_set():
            return False
        with self._lock:
            if is_failure:
                self._count += 1
                if self._count >= self.threshold:
                    self._tripped.set()
                    return True
            else:
                self._count = 0
        return False

    @property
    def tripped(self) -> bool:
        return self._tripped.is_set()
