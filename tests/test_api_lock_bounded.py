"""Tests for the bounded Zotero API lock (with_zotero_api_lock).

Regression coverage for the wedge where one stuck op holding the process-global
RLock made every other tool block until FastMCP's ~60s client timeout, surfacing
as an opaque "-32001 Request timed out" on reads too. The lock now acquires with
a bounded wait and raises ZoteroApiBusyError for waiters instead of hanging.
"""

import threading
import time

import pytest

from zotero_mcp import client as _client
from zotero_mcp.client import (
    ZoteroApiBusyError,
    with_zotero_api_lock,
    _zotero_api_lock,
)


def test_uncontended_call_runs(monkeypatch):
    """With no contention the wrapped function runs and returns normally."""
    monkeypatch.setenv("ZOTERO_MCP_LOCK_TIMEOUT", "5")

    @with_zotero_api_lock
    def f(x):
        return x + 1

    assert f(41) == 42


def test_reentrant_same_thread(monkeypatch):
    """Nested decorated calls on one thread acquire the reentrant lock instantly."""
    monkeypatch.setenv("ZOTERO_MCP_LOCK_TIMEOUT", "5")

    @with_zotero_api_lock
    def inner():
        return "inner-ok"

    @with_zotero_api_lock
    def outer():
        # Simulates add_by_url -> add_by_doi: a decorated call within a decorated call.
        return inner()

    assert outer() == "inner-ok"


def test_waiter_fails_fast_when_lock_held(monkeypatch):
    """A second thread gives up with ZoteroApiBusyError instead of hanging."""
    monkeypatch.setenv("ZOTERO_MCP_LOCK_TIMEOUT", "0.3")

    holder_has_lock = threading.Event()
    release_holder = threading.Event()

    def holder():
        with _zotero_api_lock:
            holder_has_lock.set()
            # Hold past the waiter's bounded acquire window.
            release_holder.wait(timeout=5)

    t = threading.Thread(target=holder)
    t.start()
    assert holder_has_lock.wait(timeout=2), "holder never acquired the lock"

    @with_zotero_api_lock
    def blocked():
        return "should-not-run"

    start = time.monotonic()
    with pytest.raises(ZoteroApiBusyError):
        blocked()
    elapsed = time.monotonic() - start

    # Failed fast around the 0.3s bound, nowhere near a 60s client timeout.
    assert elapsed < 3, f"waiter blocked too long: {elapsed:.1f}s"

    release_holder.set()
    t.join(timeout=5)


def test_lock_released_after_busy_error(monkeypatch):
    """A timed-out acquire must NOT leave the lock held by the waiter."""
    monkeypatch.setenv("ZOTERO_MCP_LOCK_TIMEOUT", "0.3")

    holder_has_lock = threading.Event()
    release_holder = threading.Event()

    def holder():
        with _zotero_api_lock:
            holder_has_lock.set()
            release_holder.wait(timeout=5)

    t = threading.Thread(target=holder)
    t.start()
    assert holder_has_lock.wait(timeout=2)

    @with_zotero_api_lock
    def blocked():
        return "x"

    with pytest.raises(ZoteroApiBusyError):
        blocked()

    # Let the holder release; the lock must now be fully free.
    release_holder.set()
    t.join(timeout=5)

    @with_zotero_api_lock
    def after():
        return "after-ok"

    assert after() == "after-ok"


def test_zero_timeout_opt_out_blocks_until_free(monkeypatch):
    """ZOTERO_MCP_LOCK_TIMEOUT<=0 restores unbounded behaviour (waits, no raise)."""
    monkeypatch.setenv("ZOTERO_MCP_LOCK_TIMEOUT", "0")

    holder_has_lock = threading.Event()
    release_holder = threading.Event()

    def holder():
        with _zotero_api_lock:
            holder_has_lock.set()
            release_holder.wait(timeout=5)

    t = threading.Thread(target=holder)
    t.start()
    assert holder_has_lock.wait(timeout=2)

    result = {}

    @with_zotero_api_lock
    def waiter():
        result["ran"] = True
        return "ran"

    w = threading.Thread(target=lambda: result.setdefault("rv", waiter()))
    w.start()

    # Briefly: waiter should still be blocked (no ZoteroApiBusyError raised).
    time.sleep(0.5)
    assert "ran" not in result, "opt-out should block, not fail fast"

    release_holder.set()
    w.join(timeout=5)
    t.join(timeout=5)
    assert result.get("ran") is True
