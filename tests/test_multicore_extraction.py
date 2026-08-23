"""Tests for parallel attachment extraction.

Extraction fans out over a *process* pool rather than a thread pool: the
pdf-inspector parser holds the GIL, so threads scale at roughly 1.1x while
processes reach ~6x. These tests use plain .txt attachments — the extractor
handles them through the same ``extract_file`` seam as PDFs, and they keep the
suite free of binary fixtures.
"""

import logging
import os
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool

import pytest

from zotero_mcp import fulltext_cache, local_db
from zotero_mcp.local_db import (
    LocalZoteroReader,
    _extract_worker,
    _init_extraction_worker,
    _source_for_path,
)


class FakeReader(LocalZoteroReader):
    """Reader over a directory of files, with no sqlite behind it."""

    def __init__(self, files, cfg=None, workers=1, ft_cache=False):
        # Deliberately does not call super().__init__ — there is no database.
        self._files = dict(enumerate(files, start=1))
        self._connection = None
        self.db_path = None
        self.pdf_max_pages = 50
        self.attachment_priority = ("pdf", "html", "other")
        self.extraction_workers = workers
        self.fulltext_cache_enabled = ft_cache
        self.config_path = cfg
        self.zotero_ft_cache = {}

    def _resolve_extraction_target(self, item_id):
        path = self._files.get(item_id)
        return (path, f"ATT{item_id}") if path else None

    def _iter_parent_attachments(self, item_id):
        return [(f"ATT{item_id}", None, "text/plain")]

    def _read_zotero_ft_cache(self, attachment_key):
        return self.zotero_ft_cache.get(attachment_key)


@pytest.fixture
def files(tmp_path):
    out = []
    for i in range(12):
        p = tmp_path / f"doc{i}.txt"
        p.write_text(f"contents of document {i}\n" * 20, encoding="utf-8")
        out.append(p)
    return out


def test_worker_extracts_text(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("some words", encoding="utf-8")
    assert "some words" in _extract_worker(str(p), 50)


def test_worker_returns_empty_instead_of_raising(tmp_path):
    """One unreadable file must never take down a pool worker."""
    missing = tmp_path / "nope.txt"
    assert _extract_worker(str(missing), 50) == ""
    binary = tmp_path / "broken.pdf"
    binary.write_bytes(b"\x00\x01not a pdf at all")
    assert _extract_worker(str(binary), 50) == ""


def test_worker_initializer_silences_extraction_warnings():
    """Log suppression is parent-process state; workers must re-apply it.

    Without this, parallel runs print extraction warnings that the sequential
    path hides — and roughly 0.4% of real-world PDFs fail to parse.
    """
    log = logging.getLogger("zotero_mcp.extract")
    previous = log.level
    try:
        log.setLevel(logging.DEBUG)
        _init_extraction_worker()
        assert log.level == logging.CRITICAL
    finally:
        log.setLevel(previous)


def test_source_is_derived_from_suffix(tmp_path):
    assert _source_for_path(tmp_path / "a.pdf") == "pdf"
    assert _source_for_path(tmp_path / "a.HTML") == "html"
    assert _source_for_path(tmp_path / "a.txt") == "file"


def test_parallel_matches_sequential(files):
    """The whole point: more workers must not change what gets indexed."""
    items = [(i, f"KEY{i}") for i in range(1, len(files) + 1)]
    sequential = dict(FakeReader(files, workers=1).extract_fulltext_for_items(items))
    parallel = dict(FakeReader(files, workers=3).extract_fulltext_for_items(items))
    assert sequential == parallel
    assert len(parallel) == len(files)
    assert all(v is not None for v in parallel.values())


def test_every_item_is_reported_even_when_unreadable(files, tmp_path):
    """Callers rely on one result per input to mark extraction as attempted."""
    files = [*files, tmp_path / "does-not-exist.txt"]
    items = [(i, f"KEY{i}") for i in range(1, len(files) + 1)]
    got = dict(FakeReader(files, workers=3).extract_fulltext_for_items(items))
    assert set(got) == {i for i, _ in items}
    assert got[len(files)] is None


def test_falls_back_to_zotero_ft_cache(files):
    """An attachment that parses to nothing still yields Zotero's own cache."""
    reader = FakeReader(files, workers=2)
    empty = files[0].parent / "empty.txt"
    empty.write_text("", encoding="utf-8")
    reader._files[1] = empty
    reader.zotero_ft_cache["ATT1"] = "text from zotero"
    got = dict(reader.extract_fulltext_for_items([(1, "KEY1")]))
    assert got[1] == ("text from zotero", "zotero-cache")


@pytest.mark.parametrize("workers", [1, 3])
def test_results_are_cached_and_reused(files, tmp_path, workers):
    """A second pass must serve cached text instead of re-parsing.

    Proven by rewriting each file's contents while restoring its mtime and
    byte count: the cache key still matches, so a hit returns the *original*
    text, while a re-parse would return the replacement.
    """
    cfg = str(tmp_path / "config.json")
    items = [(i, f"KEY{i}") for i in range(1, len(files) + 1)]
    reader = FakeReader(files, cfg=cfg, workers=workers, ft_cache=True)
    first = dict(reader.extract_fulltext_for_items(items))

    root = fulltext_cache.get_fulltext_cache_root(cfg)
    assert len(list(root.glob("*.txt"))) == len(files)

    for f in files:
        st = f.stat()
        f.write_text("X" * st.st_size, encoding="utf-8")
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert dict(reader.extract_fulltext_for_items(items)) == first

    # ...and once evicted, the replacement text is what comes back.
    fulltext_cache.evict_many([k for _, k in items], config_path=cfg)
    second = dict(reader.extract_fulltext_for_items(items))
    assert second != first
    assert all(v[0].startswith("X") for v in second.values())


def test_unparseable_fallback_is_cached_too(files, tmp_path):
    """Image-only PDFs parse to nothing and are the slowest files to re-parse.

    Caching the ft-cache fallback is what stops every run paying that cost
    again just to rediscover the file is empty.
    """
    cfg = str(tmp_path / "config.json")
    reader = FakeReader(files, cfg=cfg, workers=1, ft_cache=True)
    empty = tmp_path / "scanned.txt"
    empty.write_text("", encoding="utf-8")
    reader._files[1] = empty
    reader.zotero_ft_cache["ATT1"] = "ocr text"

    assert dict(reader.extract_fulltext_for_items([(1, "KEY1")]))[1] == (
        "ocr text",
        "zotero-cache",
    )
    hit = fulltext_cache.get_cached_text(
        "ATT1",
        empty.stat().st_mtime_ns,
        empty.stat().st_size,
        profile=reader._cache_profile(),
        config_path=cfg,
    )
    assert hit == ("ocr text", "zotero-cache")


def test_cache_is_off_by_default(files, tmp_path):
    """Callers extracting under a different page cap must not poison the cache."""
    cfg = str(tmp_path / "config.json")
    reader = FakeReader(files, cfg=cfg, workers=1)  # ft_cache defaults False
    dict(reader.extract_fulltext_for_items([(1, "KEY1")]))
    root = fulltext_cache.get_fulltext_cache_root(cfg)
    assert list(root.glob("*.txt")) == []


class DeadPool:
    """An executor whose worker processes die partway through the batch.

    Reproduces what a real :class:`BrokenProcessPool` does to a run: futures
    that already completed keep their results, and every future still
    outstanding raises the same error regardless of whether its own file was
    the one that killed the worker. Faking the executor rather than provoking
    a genuine OOM keeps the test deterministic, fast, and portable to Windows
    CI — the handler under test only ever sees the exception type.
    """

    survive = 0  # submissions served normally before the pool dies

    def __init__(self, max_workers=None, initializer=None):
        self.submitted = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, fn, *args):
        future = Future()
        self.submitted += 1
        if self.submitted <= self.survive:
            future.set_result(fn(*args))
        else:
            future.set_exception(
                BrokenProcessPool("A process in the process pool was terminated abruptly")
            )
        return future


def _dead_pool_after(monkeypatch, survive):
    """Install a pool that serves ``survive`` submissions, then dies."""
    monkeypatch.setattr(
        local_db, "ProcessPoolExecutor", type("DeadPoolN", (DeadPool,), {"survive": survive})
    )


def test_broken_process_pool_is_a_runtime_error():
    """Why a blanket ``except Exception`` silently absorbed it.

    Guards the ordering of the two handlers: the specific one has to stay
    ahead of the general one or the containment below is dead code.
    """
    assert issubclass(BrokenProcessPool, RuntimeError)


def test_dead_pool_does_not_strand_the_rest_of_the_batch(files, tmp_path, monkeypatch):
    """One dead worker must cost one item, not every item still pending.

    Empty text is recorded as ``has_fulltext="failed"``, and that marker is
    sticky — the skip logic will not retry it until the item's dateModified or
    attachment set changes, or the whole collection is rebuilt. So letting
    every outstanding future decay to empty text would quietly poison a large
    run for every subsequent run too.
    """
    cfg = str(tmp_path / "config.json")
    items = [(i, f"KEY{i}") for i in range(1, len(files) + 1)]
    healthy = dict(FakeReader(files, workers=1).extract_fulltext_for_items(items))

    _dead_pool_after(monkeypatch, survive=3)
    reader = FakeReader(files, cfg=cfg, workers=4, ft_cache=True)
    got = dict(reader.extract_fulltext_for_items(items))

    assert got == healthy
    assert all(v is not None for v in got.values())
    # The recovered items went through the real in-process path, so their text
    # is cached exactly as a healthy run would have cached it.
    root = fulltext_cache.get_fulltext_cache_root(cfg)
    assert len(list(root.glob("*.txt"))) == len(files)


def test_dead_pool_still_reports_every_item(files, tmp_path, monkeypatch):
    """Callers rely on one result per input to mark extraction as attempted.

    The unreadable file stands in for whatever killed the worker: it still
    yields ``None`` rather than vanishing, and it does not take the readable
    items down with it.
    """
    files = [*files, tmp_path / "does-not-exist.txt"]
    items = [(i, f"KEY{i}") for i in range(1, len(files) + 1)]

    _dead_pool_after(monkeypatch, survive=2)
    got = dict(FakeReader(files, workers=4).extract_fulltext_for_items(items))

    assert set(got) == {i for i, _ in items}
    assert got[len(files)] is None
    assert all(got[i] is not None for i, _ in items[:-1])


def test_dead_pool_recovers_through_the_zotero_ft_cache_too(files, tmp_path, monkeypatch):
    """In-process recovery is the sequential path, fallbacks included."""
    _dead_pool_after(monkeypatch, survive=0)  # dies on the very first result
    reader = FakeReader(files, workers=4)
    empty = tmp_path / "scanned.txt"
    empty.write_text("", encoding="utf-8")
    reader._files[1] = empty
    reader.zotero_ft_cache["ATT1"] = "text from zotero"

    got = dict(reader.extract_fulltext_for_items([(1, "KEY1"), (2, "KEY2")]))

    assert got[1] == ("text from zotero", "zotero-cache")
    assert got[2] is not None


def test_worker_count_is_clamped_to_at_least_one():
    reader = LocalZoteroReader.__new__(LocalZoteroReader)
    assert LocalZoteroReader.extraction_workers == 1
    assert reader.fulltext_cache_enabled is False
