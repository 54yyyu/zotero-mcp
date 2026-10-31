"""An index ChromaDB cannot read is reported as damaged, with what to run (#747).

The damage is the one users hit: ChromaDB has a write in its log that it has
not applied to the metadata segment, and applying it fails. The fixture builds
that by winding the segment's ``max_seq_id`` back one step and breaking the
``embeddings`` table the write would land in, which reproduces the reported
"Failed to apply logs to the metadata segment" error.
"""

import importlib.util
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

import chromadb
import numpy as np
from chromadb.api.client import SharedSystemClient
from chromadb.config import Settings

from zotero_mcp.chroma_client import (
    ChromaClient,
    read_collection_status,
    unreadable_index_message,
)
from zotero_mcp.semantic_search import ZoteroSemanticSearch


@pytest.fixture()
def damaged_index(tmp_path):
    persist = tmp_path / "chroma_db"
    client = chromadb.PersistentClient(path=str(persist), settings=Settings(anonymized_telemetry=False))
    collection = client.create_collection("zotero_library")
    collection.add(
        ids=[f"ITEM{i:04d}" for i in range(50)],
        embeddings=np.random.default_rng(3).random((50, 8)).astype(np.float32),
        documents=[f"document {i}" for i in range(50)],
    )
    del collection, client
    SharedSystemClient.clear_system_cache()

    db = persist / "chroma.sqlite3"
    con = sqlite3.connect(db)
    (segment,) = con.execute("SELECT id FROM segments WHERE scope = 'METADATA'").fetchone()
    con.execute("UPDATE max_seq_id SET seq_id = seq_id - 1 WHERE segment_id = ?", (segment,))
    con.commit()
    (root,) = con.execute("SELECT rootpage FROM sqlite_master WHERE name = 'embeddings'").fetchone()
    (page_size,) = con.execute("PRAGMA page_size").fetchone()
    con.close()
    with open(db, "r+b") as f:
        f.seek((root - 1) * page_size)
        f.write(b"\xff" * page_size)
    yield persist
    SharedSystemClient.clear_system_cache()


def test_status_reports_a_header_that_undercounts_pages_as_unreadable(damaged_index):
    """The on-disk signature of #747, found from the header without opening ChromaDB."""
    db = damaged_index / "chroma.sqlite3"
    pages = db.stat().st_size // 4096
    with open(db, "r+b") as f:
        f.seek(28)
        f.write((pages - 3).to_bytes(4, "big"))

    info = read_collection_status(None, persist_directory=str(damaged_index))

    assert info["unreadable"] is True
    assert info["count"] == 0
    assert "header counts" in info["error"] and "cut off" in info["error"]
    assert info["size_bytes"] >= db.stat().st_size
    assert info["last_written"] == db.stat().st_mtime


def test_status_reports_writes_chromadb_has_not_applied(damaged_index):
    # The log-apply failure itself only shows when ChromaDB opens the index
    # (search, update-db); the status read, which does not, reports the backlog.
    info = read_collection_status(None, persist_directory=str(damaged_index))

    assert info["initialized"] is True
    assert info["pending_writes"] == 1


def test_status_of_a_new_directory_is_not_initialized_not_unreadable(tmp_path):
    info = read_collection_status(None, persist_directory=str(tmp_path / "chroma_db"))

    assert info["initialized"] is False
    assert "unreadable" not in info and "error" not in info


def _info(written: datetime) -> dict:
    return {
        "error": "Failed to apply logs to the metadata segment",
        "persist_directory": "/x/chroma_db",
        "size_bytes": 26_534_543_976,
        "last_written": written.timestamp(),
    }


def test_message_gives_size_dates_and_both_ways_out():
    message = unreadable_index_message(_info(datetime(2026, 10, 26, 17, 0)), "2026-08-21T16:51:42.747450")

    assert "(24.7 GB)" in message
    assert "Failed to apply logs to the metadata segment" in message
    assert "Last successful update: 2026-08-21 16:51" in message
    assert "Index last written:     2026-10-26 17:00 (after the last successful update" in message
    assert "zotero-mcp repair-db" in message
    assert "zotero-mcp update-db --force-rebuild" in message


def test_message_does_not_suspect_an_update_when_none_ran_since():
    message = unreadable_index_message(_info(datetime(2026, 8, 21, 16, 50)), "2026-08-21T16:51:42")

    assert "Index last written:     2026-08-21 16:50\n" in message


def test_message_without_a_recorded_update():
    message = unreadable_index_message(_info(datetime(2026, 8, 21, 16, 50)), None)

    assert "Last successful update: never recorded" in message


def test_move_aside_keeps_the_damaged_index_and_starts_an_empty_one(damaged_index):
    client = ChromaClient(persist_directory=str(damaged_index), embedding_model="default")
    assert client.read_error()

    moved = Path(client.move_aside())

    assert moved.name.startswith("chroma_db.broken-")
    assert (moved / "chroma.sqlite3").exists()
    assert client.read_error() is None
    assert client.collection.count() == 0
    assert Path(client.persist_directory) == damaged_index


class _Zotero:
    library_type = "users"
    library_id = "1"


class _UnreadableChroma:
    """The ChromaClient surface update_database and search touch before failing."""

    def __init__(self, persist_directory):
        self.persist_directory = str(persist_directory)
        self.moved = False

    def read_error(self):
        return None if self.moved else "Failed to apply logs to the metadata segment"

    def move_aside(self):
        self.moved = True
        return self.persist_directory + ".broken-1"

    def reset_collection(self):
        raise AssertionError("an unreadable index must not be reset through ChromaDB")


def _search(chroma):
    s = object.__new__(ZoteroSemanticSearch)
    s.zotero_client = _Zotero()
    s.chroma_client = chroma
    s.config_path = None
    s.update_config = {"last_update": "2026-08-21T16:51:42"}
    return s


def test_update_stops_on_an_unreadable_index_and_says_what_to_run(tmp_path):
    persist = tmp_path / "chroma_db"
    persist.mkdir()
    (persist / "chroma.sqlite3").write_bytes(b"x" * 4096)
    chroma = _UnreadableChroma(persist)

    stats = _search(chroma).update_database()

    assert "zotero-mcp repair-db" in stats["error"]
    assert "Last successful update: 2026-08-21 16:51" in stats["error"]
    assert not chroma.moved


def test_forced_rebuild_moves_an_unreadable_index_aside_first(tmp_path):
    persist = tmp_path / "chroma_db"
    persist.mkdir()
    (persist / "chroma.sqlite3").write_bytes(b"x" * 4096)
    chroma = _UnreadableChroma(persist)

    # The stub ends the run soon after the guard; what matters is that the
    # index was moved aside before anything tried to reset it through ChromaDB.
    _search(chroma).update_database(force_full_rebuild=True)

    assert chroma.moved


def test_search_explains_an_unreadable_index(tmp_path):
    persist = tmp_path / "chroma_db"
    persist.mkdir()
    (persist / "chroma.sqlite3").write_bytes(b"x" * 4096)
    search = _search(_UnreadableChroma(persist))

    result = search.search("anything")

    assert "zotero-mcp repair-db" in result["error"]
