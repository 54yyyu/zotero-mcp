"""Zotero API requests run up to four at a time, one client per thread (#826).

Zotero asks clients to make no more than 4 concurrent requests and to honour
Backoff / Retry-After. A pyzotero client keeps per-request state, so one
client object never serves two threads; and its backoff is per object, so the
pool shares the longest one with every client it runs next.

Fakes only: nothing here opens a connection or reads the user's configuration.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from zotero_mcp import semantic_search


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(semantic_search, "_api_backoff_until", 0.0)


class Tracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.per_client: dict[int, int] = {}
        self.client_peak = 0

    def job(self, client, value, pause=0.05):
        def run():
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
                n = self.per_client.get(id(client), 0) + 1
                self.per_client[id(client)] = n
                self.client_peak = max(self.client_peak, n)
            time.sleep(pause)
            with self.lock:
                self.active -= 1
                self.per_client[id(client)] -= 1
            return value

        return client, run


def _client():
    return SimpleNamespace(backoff_until=0.0)


def test_at_most_four_requests_run_at_once_and_more_than_one_does():
    tracker = Tracker()
    jobs = [tracker.job(_client(), n) for n in range(10)]

    results = semantic_search._run_per_client(jobs)

    assert results == list(range(10)), "results come back in the jobs' order"
    assert tracker.peak == 4


def test_one_client_never_serves_two_threads():
    tracker = Tracker()
    shared = _client()
    jobs = [tracker.job(shared, n) for n in range(3)] + [tracker.job(_client(), n) for n in range(3, 6)]

    assert semantic_search._run_per_client(jobs) == list(range(6))
    assert tracker.client_peak == 1


def test_a_backoff_one_client_got_holds_back_the_others():
    first, second = _client(), _client()
    until = time.time() + 30

    def told_to_back_off():
        first.backoff_until = until  # as pyzotero records a Backoff header

    semantic_search._run_per_client([(first, told_to_back_off)])
    seen = []
    semantic_search._run_per_client([(second, lambda: seen.append(second.backoff_until))])

    assert seen == [until], "pyzotero waits out backoff_until before its next request"


def test_a_failing_job_raises_after_the_others_finish():
    tracker = Tracker()

    def boom():
        raise RuntimeError("503")

    jobs = [tracker.job(_client(), 1), (_client(), boom), tracker.job(_client(), 3)]

    with pytest.raises(RuntimeError, match="503"):
        semantic_search._run_per_client(jobs)
    assert tracker.active == 0


def test_a_client_without_backoff_state_is_fine():
    assert semantic_search._run_per_client([(object(), lambda: "ok")]) == ["ok"]
