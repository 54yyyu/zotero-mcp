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
