"""Tests for embedding throttle (burst/pause) and 429 retry-with-backoff."""

import sys
import time

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search
from zotero_mcp.chroma_client import _is_rate_limit_error, _with_retry


class _RateLimitErr(Exception):
    code = 429


def test_is_rate_limit_error_detects_common_shapes():
    assert _is_rate_limit_error(_RateLimitErr("quota exceeded"))
    assert _is_rate_limit_error(Exception("RESOURCE_EXHAUSTED"))
    assert _is_rate_limit_error(Exception("HTTP 429 too many requests"))
    assert not _is_rate_limit_error(Exception("some other error"))
    # Narrow detector: free-text "rate limit" alone (no 429 / RESOURCE_EXHAUSTED)
    # should NOT trigger a retry, to avoid false positives.
    assert not _is_rate_limit_error(Exception("not a rate limit issue"))


def test_with_retry_recovers_from_rate_limit():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _RateLimitErr("RESOURCE_EXHAUSTED")
        return "ok"

    start = time.monotonic()
    result = _with_retry(flaky, max_attempts=3, base_sleep=0.05)
    elapsed = time.monotonic() - start

    assert result == "ok"
    assert calls["n"] == 2
    assert elapsed >= 0.05


def test_with_retry_reraises_non_rate_limit_immediately():
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("not a rate limit")

    with pytest.raises(ValueError):
        _with_retry(boom, max_attempts=3, base_sleep=10.0)

    assert calls["n"] == 1  # no retry


def test_with_retry_gives_up_after_max_attempts():
    calls = {"n": 0}

    def always_429():
        calls["n"] += 1
        raise _RateLimitErr("RESOURCE_EXHAUSTED")

    with pytest.raises(_RateLimitErr):
        _with_retry(always_429, max_attempts=3, base_sleep=0.01)

    assert calls["n"] == 3


class _FakeChroma:
    embedding_max_tokens = 8000

    def get_existing_ids(self, ids):
        return set()

    def upsert_documents(self, *_args, **_kwargs):
        return None

    def truncate_text(self, text, max_tokens=None):
        return text

    def get_collection_stats(self):
        return {"total_documents": 0}


def _build_search_with_fake_items(monkeypatch, n_items: int):
    """Stub out Zotero + Chroma interaction so update_database runs in <1s."""
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())

    # Bypass the cross-process update lock. In real usage it prevents the MCP
    # server's auto-update from racing a manual `update-db`. Tests don't need
    # it and a host machine where another zotero-mcp process is running would
    # otherwise make these tests bail out before reaching the throttle loop.
    import contextlib as _contextlib

    @_contextlib.contextmanager
    def _always_acquire(_lock_path):
        yield True

    monkeypatch.setattr(semantic_search, "_acquire_update_lock", _always_acquire)

    search = semantic_search.ZoteroSemanticSearch(chroma_client=_FakeChroma())

    fake_items = [
        {"key": f"K{i:04d}", "data": {"title": f"item{i}"}, "version": 1}
        for i in range(n_items)
    ]
    monkeypatch.setattr(search, "_get_items_from_source", lambda **_kw: fake_items)
    monkeypatch.setattr(search, "_load_include_fulltext_setting", lambda: False)
    monkeypatch.setattr(search, "_load_last_sync_version", lambda: 0)
    monkeypatch.setattr(search, "_save_update_config", lambda **_kw: None)
    monkeypatch.setattr(
        search,
        "_process_item_batch",
        lambda batch, force, _failed: {
            "processed": len(batch), "added": len(batch),
            "updated": 0, "skipped": 0, "errors": 0,
        },
    )

    class _ZC:
        def last_modified_version(self):
            return 1

    search.zotero_client = _ZC()
    return search


def test_burst_pause_loop_sleeps_between_bursts(monkeypatch):
    """60 items, burst=25 pause=0.5s → expect 2 pauses (after 25 and 50, not 60)."""
    search = _build_search_with_fake_items(monkeypatch, n_items=60)
    start = time.monotonic()
    search.update_database(burst_items=25, pause_seconds=0.5)
    elapsed = time.monotonic() - start
    # Two 0.5s pauses expected; allow generous upper bound for CI jitter.
    assert 0.9 <= elapsed <= 3.0, f"expected ~1.0s elapsed, got {elapsed:.2f}s"


def test_burst_pause_disabled_when_args_missing(monkeypatch):
    """No burst/pause passed → no sleep, completes instantly."""
    search = _build_search_with_fake_items(monkeypatch, n_items=60)
    start = time.monotonic()
    search.update_database()
    elapsed = time.monotonic() - start
    assert elapsed < 0.5, f"unexpected delay {elapsed:.2f}s without throttle"


def test_burst_pause_skips_final_pause(monkeypatch):
    """Exactly 50 items with burst=25 → 1 pause only (after 25), NOT after 50."""
    search = _build_search_with_fake_items(monkeypatch, n_items=50)
    start = time.monotonic()
    search.update_database(burst_items=25, pause_seconds=0.5)
    elapsed = time.monotonic() - start
    # Only one pause expected.
    assert 0.4 <= elapsed <= 1.5, f"expected ~0.5s elapsed, got {elapsed:.2f}s"
