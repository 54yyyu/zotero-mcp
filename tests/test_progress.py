"""Tests for the thread-safe progress reporter (zotero_mcp.progress)."""

import io
import random
import threading

from zotero_mcp.progress import ConsecutiveFailureBreaker, ExtractionProgress


def test_sequential_counters_and_finish():
    stream = io.StringIO()
    progress = ExtractionProgress(total=4, stream=stream)
    progress.record("extracted", "Paper A")
    progress.record("cache_hit", "Paper B")
    progress.record("skipped_existing", "Paper C")
    progress.record("timeout", "Paper D")

    counts = progress.finish()
    assert counts["completed"] == 4
    assert counts["extracted"] == 1
    assert counts["cache_hit"] == 1
    assert counts["skipped_existing"] == 1
    assert counts["timeout"] == 1

    out = stream.getvalue()
    assert "1 cached" in out
    assert "1 extracted" in out


def test_concurrent_records_lose_no_updates():
    outcomes = ["extracted", "cache_hit", "skipped_existing", "failed"]
    n_threads, per_thread = 8, 200
    progress = ExtractionProgress(total=n_threads * per_thread, stream=io.StringIO())

    def worker(seed):
        rng = random.Random(seed)
        for _ in range(per_thread):
            progress.record(rng.choice(outcomes), "item")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    counts = progress.finish()
    assert counts["completed"] == n_threads * per_thread
    assert sum(counts.get(o, 0) for o in outcomes) == n_threads * per_thread


def test_render_never_exceeds_terminal_width(monkeypatch):
    stream = io.StringIO()
    progress = ExtractionProgress(total=1, stream=stream)
    monkeypatch.setattr(progress, "_term_width", lambda: 40)
    progress.record("extracted", "An extremely long paper title that must be truncated somewhere")
    line = stream.getvalue().split("\r")[-1]
    assert len(line) <= 39


def test_breaker_trips_after_threshold():
    breaker = ConsecutiveFailureBreaker(3)
    assert not breaker.record(True)
    assert not breaker.record(True)
    assert breaker.record(True)  # trips exactly on the threshold call
    assert breaker.tripped
    # Further records never "trip again"
    assert not breaker.record(True)


def test_breaker_resets_on_success():
    breaker = ConsecutiveFailureBreaker(3)
    breaker.record(True)
    breaker.record(True)
    breaker.record(False)  # reset
    assert not breaker.record(True)
    assert not breaker.record(True)
    assert not breaker.tripped
    assert breaker.record(True)
    assert breaker.tripped


def test_breaker_thread_safety():
    breaker = ConsecutiveFailureBreaker(50)
    trip_count = [0]
    lock = threading.Lock()

    def worker():
        for _ in range(100):
            if breaker.record(True):
                with lock:
                    trip_count[0] += 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert breaker.tripped
    assert trip_count[0] == 1  # exactly one caller observes the trip
