"""The suite never takes the user's real update lock.

update_database() locks ~/.config/zotero-mcp/update.lock. When the tests
shared it, a second test run, or a real update-db running at the same time,
made the indexing tests skip their work and fail at random (19 of 45 in
test_streaming_indexing + test_sync_watermark_per_library with the real
lock held).
"""

import fcntl
from pathlib import Path

import pytest

semantic_search = pytest.importorskip("zotero_mcp.semantic_search")


def test_lock_path_points_into_the_tests_temp_dir(tmp_path):
    assert semantic_search._update_lock_path() == tmp_path / "update.lock"


def test_default_lock_path_is_unchanged_for_users(monkeypatch):
    monkeypatch.undo()  # drop the conftest redirect for this one check
    assert semantic_search._update_lock_path() == (
        Path.home() / ".config" / "zotero-mcp" / "update.lock"
    )


def test_a_held_real_lock_does_not_affect_tests(tmp_path, monkeypatch):
    # Hold a lock at the real path's name, but in a fake home, and check
    # update_database's lock acquisition uses the per-test path instead.
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    real = fake_home / ".config" / "zotero-mcp" / "update.lock"
    real.parent.mkdir(parents=True)
    with open(real, "w") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        with semantic_search._acquire_update_lock(semantic_search._update_lock_path()) as acquired:
            assert acquired
