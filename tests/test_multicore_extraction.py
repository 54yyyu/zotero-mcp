"""Tests for parallel attachment extraction.

Extraction fans out over a *process* pool rather than a thread pool: the
pdf-inspector parser holds the GIL, so threads scale at roughly 1.1x while
processes reach ~6x. These tests use plain .txt attachments — the extractor
handles them through the same ``extract_file`` seam as PDFs, and they keep the
suite free of binary fixtures.
"""

import logging
import os

import pytest

from zotero_mcp import fulltext_cache
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


def test_worker_count_is_clamped_to_at_least_one():
    reader = LocalZoteroReader.__new__(LocalZoteroReader)
    assert LocalZoteroReader.extraction_workers == 1
    assert reader.fulltext_cache_enabled is False


# ---------------------------------------------------------------------------
# Broken-pool containment
# ---------------------------------------------------------------------------

class _BrokenPool:
    """A pool whose every future fails with BrokenProcessPool.

    Models one worker dying (OOM, segfault): concurrent.futures then fails
    ALL pending futures with BrokenProcessPool, not just the culprit's.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool

        f = Future()
        f.set_exception(BrokenProcessPool("a process in the pool died"))
        return f


def test_broken_pool_retries_items_in_isolation(files, monkeypatch):
    """One dying worker must not record every in-flight item as failed.

    Before the containment, a BrokenProcessPool was swallowed like an
    ordinary parse failure: every pending item fell through to the
    ft-cache fallback, and callers recorded them all as
    has_fulltext="failed" — which the incremental-sync skip logic then
    never retries. The retry path re-runs each affected item in a fresh
    single-worker pool, so innocents extract normally.
    """
    from zotero_mcp import local_db

    monkeypatch.setattr(local_db, "ProcessPoolExecutor", _BrokenPool)

    reader = FakeReader(files, workers=3)
    isolated_calls = []
    real_worker = _extract_worker

    def fake_isolated(target, max_pages):
        isolated_calls.append(target)
        return real_worker(str(target), max_pages)

    monkeypatch.setattr(reader, "_extract_isolated", fake_isolated)

    items = [(i, f"KEY{i}") for i in range(1, len(files) + 1)]
    got = dict(reader.extract_fulltext_for_items(items))

    assert len(isolated_calls) == len(files)
    assert set(got) == {i for i, _ in items}
    assert all(v is not None for v in got.values())
    for i in range(1, len(files) + 1):
        assert f"contents of document {i - 1}" in got[i][0]


def test_broken_pool_culprit_still_fails_alone(files, monkeypatch):
    """The file that keeps killing workers fails; it takes nobody with it."""
    from zotero_mcp import local_db

    monkeypatch.setattr(local_db, "ProcessPoolExecutor", _BrokenPool)

    reader = FakeReader(files[:3], workers=3)
    real_worker = _extract_worker

    def fake_isolated(target, max_pages):
        if target == reader._files[2]:
            return ""  # its private retry pool broke again
        return real_worker(str(target), max_pages)

    monkeypatch.setattr(reader, "_extract_isolated", fake_isolated)

    got = dict(reader.extract_fulltext_for_items([(1, "K1"), (2, "K2"), (3, "K3")]))

    assert got[1] is not None and got[3] is not None
    assert got[2] is None  # no ft-cache entry either -> genuinely failed


def test_extract_isolated_survives_repeated_pool_breakage(files, monkeypatch):
    """_extract_isolated itself returns '' when its own pool breaks."""
    from zotero_mcp import local_db

    monkeypatch.setattr(local_db, "ProcessPoolExecutor", _BrokenPool)
    reader = FakeReader(files, workers=2)
    assert reader._extract_isolated(files[0], 50) == ""
