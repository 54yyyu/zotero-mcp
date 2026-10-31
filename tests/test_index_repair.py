"""``repair-db``: rebuilding an unreadable semantic index from its own files (#747).

Each test builds a real persisted ChromaDB collection shaped like zotero-mcp's
(``KEY#n`` chunk ids, ``item_key`` / ``chunk_index`` / ``n_chunks`` metadata),
damages it the way an interrupted update does or breaks one invariant, and
checks what ``repair_index`` keeps and what it leaves out. Offline-safe: the
vectors are given, so no embedding function is ever called.
"""

import importlib.util
import json
import pickle
import sqlite3
import struct
import sys
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

from zotero_mcp import index_repair
from zotero_mcp.index_repair import RepairError, repair_index

ITEMS = 12
CHUNKS = 3
DIM = 8


def _ids():
    return [f"ITEM{i:04d}#{n}" for i in range(ITEMS) for n in range(CHUNKS)]


def _vectors():
    return np.random.default_rng(7).normal(size=(ITEMS * CHUNKS, DIM)).astype(np.float32)


@pytest.fixture()
def index(tmp_path):
    """A persisted index plus a config.json pointing at it."""
    persist = tmp_path / "chroma_db"
    client = chromadb.PersistentClient(path=str(persist), settings=Settings(anonymized_telemetry=False))
    # A low sync threshold makes ChromaDB write the HNSW files for a handful of vectors.
    collection = client.create_collection(
        "zotero_library", metadata={"hnsw:sync_threshold": 10, "hnsw:batch_size": 10}
    )
    ids = _ids()
    collection.add(
        ids=ids,
        embeddings=_vectors(),
        documents=[f"text of {d}" for d in ids],
        metadatas=[
            {"item_key": d.split("#")[0], "chunk_index": int(d.split("#")[1]), "n_chunks": CHUNKS, "title": d}
            for d in ids
        ],
    )
    del collection, client
    SharedSystemClient.clear_system_cache()

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"semantic_search": {
        "persist_directory": str(persist),
        "embedding_model": "default",
    }}))
    yield persist, str(config)
    SharedSystemClient.clear_system_cache()


def _segment_dir(persist: Path) -> Path:
    (folder,) = [p for p in persist.iterdir() if p.is_dir()]
    return folder


def _read_back(persist: Path) -> dict:
    client = chromadb.PersistentClient(path=str(persist), settings=Settings(anonymized_telemetry=False))
    got = client.get_collection("zotero_library").get(include=["documents", "metadatas", "embeddings"])
    SharedSystemClient.clear_system_cache()
    return {
        doc_id: (doc, meta, np.asarray(emb))
        for doc_id, doc, meta, emb in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"])
    }


@pytest.mark.skipif(sys.platform == "win32", reason="ChromaDB keeps the index files open in this process on Windows, so the folder cannot be renamed")
def test_clean_index_round_trips(index):
    persist, config = index
    report = repair_index(config, progress=lambda _m: None)

    assert report.excluded == {}
    assert (report.items_kept, report.chunks_kept) == (ITEMS, ITEMS * CHUNKS)
    assert Path(report.broken_directory).is_dir()
    got = _read_back(persist)
    assert sorted(got) == sorted(_ids())
    vectors = dict(zip(_ids(), _vectors()))
    for doc_id, (doc, meta, emb) in got.items():
        assert doc == f"text of {doc_id}"
        assert meta["item_key"] == doc_id.split("#")[0] and meta["title"] == doc_id
        np.testing.assert_array_equal(emb, vectors[doc_id])
    assert json.loads(Path(report.report_path).read_text())["chunks_kept"] == ITEMS * CHUNKS


def test_dry_run_changes_nothing(index):
    persist, config = index
    before = sorted(p.name for p in persist.parent.iterdir())
    report = repair_index(config, dry_run=True, progress=lambda _m: None)

    assert report.chunks_kept == ITEMS * CHUNKS and report.broken_directory is None
    after = sorted(p.name for p in persist.parent.iterdir())
    assert [n for n in after if n not in before] == [Path(report.report_path).name]


@pytest.mark.skipif(sys.platform == "win32", reason="ChromaDB keeps the index files open in this process on Windows, so the folder cannot be renamed")
def test_header_that_undercounts_pages_is_corrected_in_the_copy(index):
    """The #747 failure: a cut-off write grew the file but the header kept the old page count."""
    persist, config = index
    db = persist / "chroma.sqlite3"
    original = db.read_bytes()
    pages = len(original) // struct.unpack_from(">H", original, 16)[0]
    with open(db, "r+b") as f:
        f.seek(28)
        f.write(struct.pack(">I", pages - 3))
    con = sqlite3.connect(db)
    with pytest.raises(sqlite3.DatabaseError, match="malformed"):
        con.execute("SELECT count(*) FROM embedding_metadata NOT INDEXED").fetchall()
    con.close()
    damaged = db.read_bytes()

    report = repair_index(config, progress=lambda _m: None)

    assert report.header_page_count_fixed
    assert report.chunks_kept == ITEMS * CHUNKS
    assert (Path(report.broken_directory) / "chroma.sqlite3").read_bytes() == damaged


@pytest.mark.skipif(sys.platform == "win32", reason="ChromaDB keeps the index files open in this process on Windows, so the folder cannot be renamed")
def test_item_with_a_chunk_missing_its_vector_is_left_out(index):
    persist, config = index
    pickle_path = _segment_dir(persist) / "index_metadata.pickle"
    meta = pickle.loads(pickle_path.read_bytes())
    del meta["id_to_label"]["ITEM0003#1"]
    pickle_path.write_bytes(pickle.dumps(meta))

    report = repair_index(config, progress=lambda _m: None)

    assert report.excluded == {"ITEM0003": "chunk without vector"}
    assert report.chunks_kept == (ITEMS - 1) * CHUNKS
    assert not any(d.startswith("ITEM0003") for d in _read_back(persist))


def test_item_with_a_chunk_missing_its_document_is_left_out(index):
    persist, config = index
    con = sqlite3.connect(persist / "chroma.sqlite3")
    con.execute(
        "DELETE FROM embedding_metadata WHERE key = 'chroma:document' AND id = "
        "(SELECT id FROM embeddings WHERE embedding_id = 'ITEM0005#2')"
    )
    con.commit()
    con.close()

    report = repair_index(config, dry_run=True, progress=lambda _m: None)

    assert report.excluded == {"ITEM0005": "chunk without document"}


def test_item_whose_metadata_names_another_chunk_is_left_out(index):
    persist, config = index
    con = sqlite3.connect(persist / "chroma.sqlite3")
    con.execute(
        "UPDATE embedding_metadata SET int_value = 0 WHERE key = 'chunk_index' AND id = "
        "(SELECT id FROM embeddings WHERE embedding_id = 'ITEM0007#2')"
    )
    con.commit()
    con.close()

    report = repair_index(config, dry_run=True, progress=lambda _m: None)

    assert report.excluded == {"ITEM0007": "metadata names another chunk"}


def test_item_with_a_queued_write_is_left_out(index):
    persist, config = index
    con = sqlite3.connect(persist / "chroma.sqlite3")
    (collection_id,) = con.execute("SELECT id FROM collections").fetchone()
    con.execute(
        "INSERT INTO embeddings_queue (seq_id, operation, topic, id) VALUES (?, 3, ?, 'ITEM0009#0')",
        (10_000, f"persistent://default/default/{collection_id}"),
    )
    con.commit()
    con.close()

    report = repair_index(config, dry_run=True, progress=lambda _m: None)

    assert report.excluded == {"ITEM0009": "write still queued"}


@pytest.mark.skipif(sys.platform == "win32", reason="ChromaDB keeps the index files open in this process on Windows, so the folder cannot be renamed")
def test_unknown_vector_format_is_refused(index):
    persist, config = index
    header = _segment_dir(persist) / "header.bin"
    data = bytearray(header.read_bytes())
    data[0:4] = struct.pack("<I", 2)
    header.write_bytes(bytes(data))

    with pytest.raises(RepairError, match="update-db --force-rebuild"):
        repair_index(config, progress=lambda _m: None)
    assert (persist / "chroma.sqlite3").exists()


class _Smuggled:
    pass


@pytest.mark.skipif(sys.platform == "win32", reason="ChromaDB keeps the index files open in this process on Windows, so the folder cannot be renamed")
def test_vector_metadata_pickle_may_only_hold_plain_data(index):
    persist, config = index
    (_segment_dir(persist) / "index_metadata.pickle").write_bytes(
        pickle.dumps({"id_to_label": {"x": 1}, "extra": _Smuggled()})
    )

    with pytest.raises(RepairError, match="index_metadata.pickle"):
        repair_index(config, progress=lambda _m: None)


def test_missing_index_says_to_build_one(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"semantic_search": {"persist_directory": str(tmp_path / "none")}}))

    with pytest.raises(RepairError, match="update-db"):
        repair_index(str(config), progress=lambda _m: None)


class _FlakyConnection:
    """Serves rowids 1..100 but fails any query that touches rowids 40..45."""

    def execute(self, query, params=()):
        if "max(rowid)" in query:
            return _Rows([(100,)])
        lo, hi = params
        if lo <= 45 and hi >= 40:
            raise sqlite3.DatabaseError("database disk image is malformed")
        return _Rows([(r,) for r in range(lo, hi + 1)])


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0]

    def fetchall(self):
        return self.rows


def test_scan_skips_only_the_unreadable_rowid_window(monkeypatch):
    monkeypatch.setattr(index_repair, "_SCAN_WINDOW", 100)
    monkeypatch.setattr(index_repair, "_MIN_SCAN_WINDOW", 4)
    skipped = []

    rows = [r for (r,) in index_repair._scan_rows(_FlakyConnection(), "t", "rowid", skipped.append)]

    assert set(range(1, 101)) - set(rows) <= set(range(37, 49))
    assert set(range(40, 46)).isdisjoint(rows)
    assert sum(skipped) == 100 - len(rows)
