"""When the semantic search worker exits, once it holds passages.

A passage ChromaDB collection is large and slow to load again, an article
collection small and quick. So once a worker has passages loaded, only
passage activity (a search at that level, or anything that writes or deletes
passages) keeps it alive; article searches just start a new worker cheaply
after it exits. It never exits while this process runs an index update or
reads through an iterator. And the first passage search in a worker reports
how long it took, so a slow load can be explained to the user.

These tests drive the worker client with a fake process and a fake clock, so
no real minutes pass and no ChromaDB is opened.
"""

import itertools
import logging
import sqlite3

import pytest

from zotero_mcp import chroma_worker
from zotero_mcp.chroma_worker import WorkerChromaClient

IDLE = 600.0  # seconds: the default 10 minutes


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class _Pipe:
    def close(self):
        pass


class _Proc:
    _pids = itertools.count(50000)

    def __init__(self, *args, **kwargs):
        self.pid = next(self._pids)
        self.stdin = _Pipe()
        self.stdout = _Pipe()
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -9


@pytest.fixture()
def worker(tmp_path, monkeypatch):
    """A worker-backed client whose worker is a fake answering ``replies``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    clock = _Clock()
    monkeypatch.setattr(chroma_worker, "_clock", clock)
    # Updates earlier tests ran ended on the real clock, far past this one's.
    monkeypatch.setattr(chroma_worker, "_updates_running", 0)
    monkeypatch.setattr(chroma_worker, "_last_update_end", 0.0)
    monkeypatch.setattr(chroma_worker.subprocess, "Popen", _Proc)
    monkeypatch.setattr(WorkerChromaClient, "_reap_when_idle", lambda self, proc: None)
    sent = []
    iterators = {}

    def reply(message):
        kind, name = message[0], message[1]
        if kind == "getattr":
            return ("callable", None)
        if kind == "call" and name == "iter_metadatas":
            iterators[7] = iter([(["A"], [{}]), (["B"], [{}])])
            return ("iter", 7)
        if kind == "next":
            try:
                return ("ok", next(iterators[name]))
            except StopIteration:
                return ("done", None)
        if kind == "call" and name == "separate_passages":
            if isinstance(client.separate, Exception):
                return ("error", client.separate)
            return ("ok", client.separate)
        if kind == "call" and name == "search":
            client.search_hook(message)
            return ("ok", {"ids": [["A"]], "documents": [["text"]], **client.search_extra})
        return ("ok", None)

    monkeypatch.setattr(chroma_worker, "_send", lambda stream, message: sent.append(message))
    monkeypatch.setattr(chroma_worker, "_receive", lambda stream: reply(sent[-1]))
    client = WorkerChromaClient(str(tmp_path / "config.json"), idle_seconds=IDLE)
    client.search_hook = lambda message: None
    client.search_extra = {}  # more keys in the worker's search reply
    client.separate = True  # passages in their own ChromaDB collection (a migrated index)
    client.clock = clock
    return client


def _idle_check(client) -> bool:
    """One pass of the idle check the reaper thread runs; True if it stopped the worker."""
    proc = client._proc
    return client._stop_if_idle(proc)


def _article_search(client):
    return client.search(query_texts=["q"], n_results=5)


def _passage_search(client):
    return client.search(query_texts=["q"], n_results=5, level="passages")


def test_article_activity_keeps_a_worker_without_passages(worker):
    _article_search(worker)
    worker.clock.now += 500
    _article_search(worker)
    worker.clock.now += 500

    assert not _idle_check(worker)
    worker.clock.now += 101
    assert _idle_check(worker)
    assert worker.worker_pid is None


def test_once_passages_are_loaded_article_searches_do_not_keep_the_worker(worker):
    _passage_search(worker)
    first = worker.worker_pid
    for _ in range(3):
        worker.clock.now += 190
        _article_search(worker)
    worker.clock.now += IDLE - 570

    assert _idle_check(worker)
    # The next article search starts a new worker, which has no passages
    # loaded, so article searches keep that one as before.
    _article_search(worker)
    assert worker.worker_pid not in (None, first)
    worker.clock.now += 500
    _article_search(worker)
    worker.clock.now += 500
    assert not _idle_check(worker)


def test_a_passage_search_keeps_a_worker_with_passages(worker):
    _passage_search(worker)
    worker.clock.now += 500
    _passage_search(worker)
    worker.clock.now += 500

    assert not _idle_check(worker)


PASSAGE_CALLS = {
    "search by keyword": ("search", (), {"query_texts": ["q"], "level": "passages"}),
    "search by position": ("search", (["q"], 5, None, None, "passages"), {}),
    "add": ("add_documents", (["t"], [{}], ["K#0"]), {}),
    "upsert": ("upsert_documents", (), {"documents": ["t"], "metadatas": [{}], "ids": ["A", "K#3"]}),
    "upsert vectors": ("upsert_embeddings", (["t"], [{}], ["K#0"], [[0.0]]), {}),
    "delete": ("delete_documents", (["K#1"],), {}),
    "metadata update": ("update_metadatas", (["K#1"], [{}]), {}),
    "delete chunks": ("delete_item_chunks", ("K",), {}),
    "prune chunks": ("prune_item_chunks", ("K", 3), {}),
    "reset": ("reset_collection", (), {}),
}


@pytest.mark.parametrize("call", PASSAGE_CALLS.values(), ids=PASSAGE_CALLS.keys())
def test_passage_activity_resets_the_timer(worker, call):
    name, args, kwargs = call
    _passage_search(worker)
    worker.clock.now += 500
    getattr(worker, name)(*args, **kwargs)
    worker.clock.now += 500

    assert not _idle_check(worker)
    worker.clock.now += 100
    assert _idle_check(worker)


PASSAGE_WRITES = {k: v for k, v in PASSAGE_CALLS.items() if v[0] != "search"}


@pytest.mark.parametrize("call", PASSAGE_WRITES.values(), ids=PASSAGE_WRITES.keys())
def test_on_an_unmigrated_index_a_passage_write_leaves_every_call_counting(worker, call):
    # Passages still in the article collection: searches carry no level, and
    # the one collection holds everything, so any call keeps the worker.
    worker.separate = False
    name, args, kwargs = call
    getattr(worker, name)(*args, **kwargs)
    for _ in range(4):
        worker.clock.now += 500
        _article_search(worker)
        assert not _idle_check(worker)

    worker.clock.now += IDLE
    assert _idle_check(worker)


@pytest.mark.parametrize("call", PASSAGE_WRITES.values(), ids=PASSAGE_WRITES.keys())
def test_a_failed_layout_check_never_fails_the_write(worker, call, caplog, monkeypatch):
    # With routing on (the default from branch 6) separate_passages() reads
    # SQLite, which can fail or time out. The write, a reset_collection for
    # recovery included, must still run; it then counts as ordinary activity,
    # which keeps the worker (the safe side).
    sent = []
    real_send = chroma_worker._send
    worker.separate = sqlite3.OperationalError("database is locked")
    name, args, kwargs = call
    monkeypatch.setattr(
        chroma_worker, "_send", lambda stream, message: (sent.append(message), real_send(stream, message))
    )
    with caplog.at_level(logging.WARNING, logger="zotero_mcp.chroma_worker"):
        getattr(worker, name)(*args, **kwargs)
        getattr(worker, name)(*args, **kwargs)

    assert [m[1] for m in sent if m[0] == "call"].count(name) == 2  # both writes went through
    assert len([r for r in caplog.records if "database is locked" in r.getMessage()]) == 1
    for _ in range(4):
        worker.clock.now += 500
        _article_search(worker)
        assert not _idle_check(worker)


ARTICLE_CALLS = {
    "article search": ("search", (), {"query_texts": ["q"], "level": "articles"}),
    "article upsert": ("upsert_embeddings", (["t"], [{}], ["K"], [[0.0]]), {}),
    "article delete": ("delete_documents", (["K"],), {}),
    "read": ("get_existing_ids", (["K#0"],), {}),
    "info": ("get_collection_info", (), {}),
}


@pytest.mark.parametrize("call", ARTICLE_CALLS.values(), ids=ARTICLE_CALLS.keys())
def test_other_activity_does_not_reset_the_timer_once_passages_are_loaded(worker, call):
    name, args, kwargs = call
    _passage_search(worker)
    worker.clock.now += 500
    getattr(worker, name)(*args, **kwargs)
    worker.clock.now += 100

    assert _idle_check(worker)


def test_an_open_iterator_keeps_the_worker(worker):
    _passage_search(worker)
    batches = worker.iter_metadatas()
    assert next(batches) == (["A"], [{}])
    worker.clock.now += 10 * IDLE

    assert not _idle_check(worker)
    assert list(batches) == [(["B"], [{}])]
    worker.clock.now += IDLE - 1
    assert not _idle_check(worker)  # the timer starts when the iterator is done
    worker.clock.now += 1
    assert _idle_check(worker)


def test_an_iterator_closed_early_stops_keeping_the_worker(worker):
    batches = worker.iter_metadatas()
    next(batches)
    batches.close()
    worker.clock.now += IDLE

    assert _idle_check(worker)


def test_an_iterator_keeps_the_worker_before_its_first_item(worker):
    batches = worker.iter_metadatas()
    worker.clock.now += 10 * IDLE

    assert not _idle_check(worker)
    assert len(list(batches)) == 2


def test_an_update_in_this_process_keeps_the_worker(worker, tmp_path):
    from zotero_mcp import semantic_search

    _passage_search(worker)
    with semantic_search._acquire_update_lock(tmp_path / "update.lock") as acquired:
        assert acquired
        worker.clock.now += 10 * IDLE
        assert not _idle_check(worker)
        _article_search(worker)  # say, the update's article pass
        worker.clock.now += 10 * IDLE
        assert not _idle_check(worker)

    worker.clock.now += IDLE - 1
    assert not _idle_check(worker)  # the timer starts when the update ends
    worker.clock.now += 1
    assert _idle_check(worker)


def test_an_update_that_another_process_holds_does_not_keep_the_worker(worker, tmp_path):
    from zotero_mcp import semantic_search

    lock = tmp_path / "update.lock"
    _passage_search(worker)
    with semantic_search._acquire_update_lock(lock) as first:
        assert first
        with semantic_search._acquire_update_lock(lock) as second:
            assert not second  # this one skips its run
    worker.clock.now += IDLE

    assert _idle_check(worker)


def _slow_load(worker, seconds):
    def hook(message):
        if message[3].get("level") == "passages":
            worker.clock.now += seconds

    worker.search_hook = hook


def test_the_first_passage_search_in_a_worker_reports_its_time(worker):
    _slow_load(worker, 40)

    first = _passage_search(worker)
    second = _passage_search(worker)

    assert first["passage_load"] == {"seconds": 40, "reloaded": False, "idle_minutes": 10}
    assert "passage_load" not in second
    assert "passage_load" not in _article_search(worker)


def test_the_reported_time_includes_starting_the_worker(worker, monkeypatch):
    # When the search itself starts the worker (the last one exited between
    # the search's earlier calls and this one), the user waits for that too.
    worker.search  # noqa: B018 - fetched once, as the server has by now
    worker.stop()

    class _SlowStart(_Proc):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            worker.clock.now += 2

    monkeypatch.setattr(chroma_worker.subprocess, "Popen", _SlowStart)
    _slow_load(worker, 40)

    assert _passage_search(worker)["passage_load"]["seconds"] == 42


def test_the_reported_time_leaves_out_embedding_the_query(worker):
    # A slow embedding API is not a slow load: the note would blame the
    # wrong thing and suggest a setting that cannot help.
    _slow_load(worker, 40)
    worker.search_extra = {"embedding_seconds": 37.0}

    first = _passage_search(worker)
    second = _passage_search(worker)

    assert first["passage_load"]["seconds"] == 3
    assert "embedding_seconds" not in first
    assert "embedding_seconds" not in second


def test_a_passage_search_after_an_idle_exit_says_it_reloaded(worker):
    _slow_load(worker, 40)
    _passage_search(worker)
    worker.clock.now += IDLE
    assert _idle_check(worker)

    _article_search(worker)  # a new worker, without passages
    again = _passage_search(worker)

    assert again["passage_load"] == {"seconds": 40, "reloaded": True, "idle_minutes": 10}


def test_a_worker_that_had_no_passages_does_not_make_the_next_load_a_reload(worker):
    _article_search(worker)
    worker.clock.now += IDLE
    assert _idle_check(worker)

    assert _passage_search(worker)["passage_load"]["reloaded"] is False


def test_a_failed_first_passage_search_leaves_the_report_to_the_next(worker, monkeypatch):
    calls = []

    def fail_once(message):
        calls.append(message)
        if len(calls) == 1:
            raise RuntimeError("embedding API down")

    worker.search_hook = fail_once
    real_receive = chroma_worker._receive

    def receive(stream):
        try:
            return real_receive(stream)
        except RuntimeError as e:
            return ("error", e)

    monkeypatch.setattr(chroma_worker, "_receive", receive)
    with pytest.raises(RuntimeError):
        _passage_search(worker)

    assert "passage_load" in _passage_search(worker)
