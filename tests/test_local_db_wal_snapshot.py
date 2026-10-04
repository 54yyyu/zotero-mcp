"""Local reads must see what Zotero has written but not yet checkpointed (#536).

Zotero keeps zotero.sqlite in WAL mode under an exclusive lock. The reader used
to open it with ``immutable=1``, which gets past the lock by ignoring the -wal
file, so every item added since the last checkpoint was invisible: PDF tools
said "No PDF attachment found" for a paper added minutes earlier. These tests
reproduce that with a writer holding the same kind of lock Zotero holds.
"""

import os
import sqlite3

import pytest

from zotero_mcp import local_db
from zotero_mcp.local_db import LocalZoteroReader


@pytest.fixture(autouse=True)
def _fresh_snapshot_cache(monkeypatch):
    monkeypatch.setattr(local_db, "_snapshots", {})
    monkeypatch.delenv(local_db.DB_SNAPSHOT_ENV_VAR, raising=False)
    monkeypatch.setenv(local_db.DB_SNAPSHOT_MIN_INTERVAL_ENV_VAR, "0")


@pytest.fixture
def zotero_like_db(tmp_path):
    """A WAL-mode database with checkpointed rows plus rows only in the WAL,
    held open under an exclusive lock the way a running Zotero holds it."""
    path = tmp_path / "zotero.sqlite"
    setup = sqlite3.connect(path)
    setup.execute("PRAGMA journal_mode=WAL")
    setup.execute("CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT)")
    setup.execute("INSERT INTO items (key) VALUES ('OLDITEM1')")
    setup.commit()
    setup.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    setup.close()

    writer = sqlite3.connect(path)
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("PRAGMA locking_mode=EXCLUSIVE")
    writer.execute("INSERT INTO items (key) VALUES ('NEWITEM1')")
    writer.commit()
    assert os.path.getsize(str(path) + "-wal") > 0, "the new row must live only in the WAL"
    yield path, writer
    writer.close()


def _keys(db_path):
    with LocalZoteroReader(db_path=str(db_path)) as reader:
        return reader.get_all_item_keys()


def test_rows_only_in_the_wal_are_visible(zotero_like_db):
    path, _writer = zotero_like_db
    assert _keys(path) == {"OLDITEM1", "NEWITEM1"}


def test_the_old_in_place_read_really_missed_them(zotero_like_db):
    """Guards the test itself: without the snapshot the new row is invisible,
    so the test above is exercising the WAL and not a checkpointed file."""
    path, _writer = zotero_like_db
    conn = sqlite3.connect(f"file:{path}?immutable=1", uri=True)
    try:
        keys = {row[0] for row in conn.execute("SELECT key FROM items")}
    finally:
        conn.close()
    assert keys == {"OLDITEM1"}


def test_snapshot_is_reused_until_zotero_writes_again(zotero_like_db):
    path, writer = zotero_like_db
    first = local_db._wal_snapshot_path(str(path))
    assert first and first == local_db._wal_snapshot_path(str(path))

    writer.execute("INSERT INTO items (key) VALUES ('NEWITEM2')")
    writer.commit()

    second = local_db._wal_snapshot_path(str(path))
    assert second and second != first
    assert _keys(path) == {"OLDITEM1", "NEWITEM1", "NEWITEM2"}


def test_no_wal_reads_in_place(tmp_path):
    path = tmp_path / "zotero.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT)")
    conn.execute("INSERT INTO items (key) VALUES ('ONLYITEM')")
    conn.commit()
    conn.close()

    assert local_db._wal_snapshot_path(str(path)) is None
    assert _keys(path) == {"ONLYITEM"}


def test_opt_out_reads_in_place(zotero_like_db, monkeypatch):
    path, _writer = zotero_like_db
    monkeypatch.setenv(local_db.DB_SNAPSHOT_ENV_VAR, "0")
    assert local_db._wal_snapshot_path(str(path)) is None
    assert _keys(path) == {"OLDITEM1"}


def test_a_failed_copy_falls_back_to_the_in_place_read(zotero_like_db, monkeypatch):
    path, _writer = zotero_like_db

    def _refuse(*_a, **_k):
        raise OSError("sharing violation")

    monkeypatch.setattr(local_db.shutil, "copyfile", _refuse)
    assert local_db._wal_snapshot_path(str(path)) is None
    assert _keys(path) == {"OLDITEM1"}


def test_long_lived_reader_sees_writes_made_after_it_connected(zotero_like_db):
    """The SQLite library backend keeps one reader per thread for the life of
    the process; without a refresh it answered from its first snapshot forever."""
    path, writer = zotero_like_db
    reader = LocalZoteroReader(db_path=str(path))
    try:
        assert reader.get_all_item_keys() == {"OLDITEM1", "NEWITEM1"}
        writer.execute("INSERT INTO items (key) VALUES ('LATEITEM')")
        writer.commit()
        assert reader.refresh_if_stale() is True
        assert "LATEITEM" in reader.get_all_item_keys()
        assert reader.refresh_if_stale() is False
    finally:
        reader.close()


def test_snapshot_is_not_recopied_inside_the_min_interval(zotero_like_db, monkeypatch):
    path, writer = zotero_like_db
    monkeypatch.setenv(local_db.DB_SNAPSHOT_MIN_INTERVAL_ENV_VAR, "30")
    clock = [1000.0]
    monkeypatch.setattr(local_db.time, "monotonic", lambda: clock[0])

    first = local_db._wal_snapshot_path(str(path))
    writer.execute("INSERT INTO items (key) VALUES ('BURST001')")
    writer.commit()

    clock[0] += 10
    assert local_db._wal_snapshot_path(str(path)) == first
    clock[0] += 25
    assert local_db._wal_snapshot_path(str(path)) != first


def _count_copies(monkeypatch):
    copies = [0]
    real = local_db.shutil.copyfile

    def counting(src, dst, *a, **k):
        if str(dst).endswith("zotero.sqlite"):
            copies[0] += 1
        return real(src, dst, *a, **k)

    monkeypatch.setattr(local_db.shutil, "copyfile", counting)
    return copies


def _snapshot_keys(snap):
    conn = sqlite3.connect(snap)
    try:
        return {row[0] for row in conn.execute("SELECT key FROM items")}
    finally:
        conn.close()


def test_read_after_own_write_sees_it_then_throttle_applies_again(zotero_like_db, monkeypatch):
    path, writer = zotero_like_db
    monkeypatch.setenv(local_db.DB_SNAPSHOT_MIN_INTERVAL_ENV_VAR, "30")
    clock = [1000.0]
    monkeypatch.setattr(local_db.time, "monotonic", lambda: clock[0])
    copies = _count_copies(monkeypatch)

    first = local_db._wal_snapshot_path(str(path))
    assert copies[0] == 1

    # Our own write lands inside the throttle window.
    local_db.note_local_write()
    writer.execute("INSERT INTO items (key) VALUES ('OWNWRITE')")
    writer.commit()

    clock[0] += 1
    second = local_db._wal_snapshot_path(str(path))
    assert second != first
    assert "OWNWRITE" in _snapshot_keys(second)
    assert copies[0] == 2

    # (b) one-shot: another change right after, with no new write of ours,
    # is throttled again instead of copying a second time.
    writer.execute("INSERT INTO items (key) VALUES ('ZOTEROEDIT')")
    writer.commit()
    clock[0] += 1
    assert local_db._wal_snapshot_path(str(path)) == second
    assert copies[0] == 2


def test_several_writes_before_a_read_cost_one_copy(zotero_like_db, monkeypatch):
    path, writer = zotero_like_db
    monkeypatch.setenv(local_db.DB_SNAPSHOT_MIN_INTERVAL_ENV_VAR, "30")
    clock = [1000.0]
    monkeypatch.setattr(local_db.time, "monotonic", lambda: clock[0])
    copies = _count_copies(monkeypatch)
    local_db._wal_snapshot_path(str(path))

    for n in range(5):
        local_db.note_local_write()
        writer.execute("INSERT INTO items (key) VALUES (?)", (f"W{n}",))
        writer.commit()
    clock[0] += 1
    snap = local_db._wal_snapshot_path(str(path))
    assert {f"W{n}" for n in range(5)} <= _snapshot_keys(snap)
    assert copies[0] == 2
    clock[0] += 1
    assert local_db._wal_snapshot_path(str(path)) == snap
    assert copies[0] == 2


def test_write_with_no_snapshot_yet_costs_nothing_extra(zotero_like_db, monkeypatch):
    path, _writer = zotero_like_db
    monkeypatch.setenv(local_db.DB_SNAPSHOT_MIN_INTERVAL_ENV_VAR, "30")
    copies = _count_copies(monkeypatch)
    local_db.note_local_write()
    local_db._wal_snapshot_path(str(path))
    local_db._wal_snapshot_path(str(path))
    assert copies[0] == 1


def test_stale_mark_survives_a_read_that_sees_no_change(zotero_like_db, monkeypatch):
    """A read before our write lands finds the files unchanged and keeps the
    copy; the stale mark must still be there for the read after the write."""
    path, writer = zotero_like_db
    monkeypatch.setenv(local_db.DB_SNAPSHOT_MIN_INTERVAL_ENV_VAR, "30")
    clock = [1000.0]
    monkeypatch.setattr(local_db.time, "monotonic", lambda: clock[0])
    first = local_db._wal_snapshot_path(str(path))

    local_db.note_local_write()
    assert local_db._wal_snapshot_path(str(path)) == first  # nothing changed yet
    writer.execute("INSERT INTO items (key) VALUES ('LANDED')")
    writer.commit()
    clock[0] += 1
    assert "LANDED" in _snapshot_keys(local_db._wal_snapshot_path(str(path)))


def test_burst_of_reads_without_writes_copies_at_most_once(zotero_like_db, monkeypatch):
    path, _writer = zotero_like_db
    monkeypatch.setenv(local_db.DB_SNAPSHOT_MIN_INTERVAL_ENV_VAR, "30")
    copies = _count_copies(monkeypatch)
    for _ in range(20):
        local_db._wal_snapshot_path(str(path))
    assert copies[0] == 1


def test_backend_reader_is_refreshed_when_reused(monkeypatch):
    from zotero_mcp import library

    class _Reader:
        refreshed = 0

        def refresh_if_stale(self):
            _Reader.refreshed += 1
            return False

    made = []
    monkeypatch.setattr("zotero_mcp.local_db.get_local_zotero_reader", lambda: made.append(1) or _Reader())
    library.reset_sqlite_reader()
    try:
        first = library._sqlite_reader()
        second = library._sqlite_reader()
        assert first is second and len(made) == 1
        assert _Reader.refreshed == 1
    finally:
        library._thread_state.reader = None

