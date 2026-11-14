"""``repair-db`` keeps an item-keyed index (#826) whole: an attachment's own
record ``ATT`` and its passages ``ATT#n`` are one key, not a mixed item."""

import importlib.util
import json
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip("chromadb is incompatible with Python 3.14+", allow_module_level=True)
if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)
if sys.platform == "win32":
    pytest.skip("ChromaDB keeps the index files open in this process on Windows", allow_module_level=True)

import chromadb
import numpy as np
from chromadb.api.client import SharedSystemClient
from chromadb.config import Settings

from zotero_mcp.index_repair import repair_index

DIM = 8
PASSAGES = "zotero_library_passages"
SAVED = {"hnsw:sync_threshold": 10, "hnsw:batch_size": 10}
QUIET = {"progress": lambda _m: None}

BIB = {"item_key": "BIB00001", "group_id": 1, "date_modified": "2026-01-01"}
ATT = {
    "item_key": "ATT00001",
    "group_id": 1,
    "date_modified": "2026-01-01",
    "file_stamp": "1:2",
}


def _passage(n):
    return {"item_key": "ATT00001", "group_id": 1, "chunk_index": n, "n_chunks": 3, "char_start": n, "char_end": n + 1}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    yield
    SharedSystemClient.clear_system_cache()


def _make(tmp_path, collections):
    persist = tmp_path / "chroma_db"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"semantic_search": {"persist_directory": str(persist), "embedding_model": "default"}})
    )
    client = chromadb.PersistentClient(path=str(persist), settings=Settings(anonymized_telemetry=False))
    rng = np.random.default_rng(3)
    for name, rows in collections.items():
        # ChromaDB saves vector files only past the sync threshold; pad every collection past it.
        if name == PASSAGES:
            rows = rows + [
                (f"FILL{i:04d}#0", {**_passage(0), "item_key": f"FILL{i:04d}", "n_chunks": 1}) for i in range(12)
            ]
        else:
            rows = rows + [(f"FILL{i:04d}", {"item_key": f"FILL{i:04d}", "group_id": 1}) for i in range(12)]
        ids = [i for i, _ in rows]
        client.create_collection(name, metadata=SAVED).add(
            ids=ids,
            embeddings=rng.normal(size=(len(ids), DIM)).astype(np.float32),
            documents=[f"text of {i}" for i in ids],
            metadatas=[m for _, m in rows],
        )
    del client
    SharedSystemClient.clear_system_cache()
    return persist, str(config)


def _ids(persist, name):
    client = chromadb.PersistentClient(path=str(persist), settings=Settings(anonymized_telemetry=False))
    got = client.get_collection(name).get(include=[])["ids"]
    SharedSystemClient.clear_system_cache()
    return sorted(got)


def test_a_split_new_shape_index_is_kept_whole(tmp_path):
    persist, config = _make(
        tmp_path,
        {
            "zotero_library": [("BIB00001", BIB), ("ATT00001", ATT)],
            PASSAGES: [(f"ATT00001#{n}", _passage(n)) for n in range(3)],
        },
    )

    report = repair_index(config, **QUIET)

    assert report.excluded == {}
    assert report.chunks_kept == 29
    assert {"ATT00001", "BIB00001"} <= set(_ids(persist, "zotero_library"))
    assert {"ATT00001#0", "ATT00001#1", "ATT00001#2"} <= set(_ids(persist, PASSAGES))


def test_an_attachment_record_beside_its_passages_in_one_collection_is_not_mixed(tmp_path):
    persist, config = _make(
        tmp_path,
        {
            "zotero_library": [("BIB00001", BIB), ("ATT00001", ATT)]
            + [(f"ATT00001#{n}", _passage(n)) for n in range(3)],
        },
    )

    report = repair_index(config, **QUIET)

    assert report.excluded == {}
    assert {"ATT00001", "ATT00001#0", "ATT00001#1", "ATT00001#2", "BIB00001"} <= set(_ids(persist, "zotero_library"))


def test_a_bare_legacy_record_with_chunks_is_still_mixed(tmp_path):
    legacy = {"item_key": "ATT00001", "title": "Old"}
    _, config = _make(
        tmp_path,
        {
            "zotero_library": [("ATT00001", legacy)] + [(f"ATT00001#{n}", _passage(n)) for n in range(3)],
            PASSAGES: [("OTHER001#0", {"item_key": "OTHER001", "chunk_index": 0, "n_chunks": 1})],
        },
    )

    report = repair_index(config, dry_run=True, **QUIET)

    assert report.excluded == {"ATT00001": "mixed chunked and unchunked ids"}


def test_legacy_records_and_a_rekeyed_marker_are_repaired_as_before(tmp_path):
    persist, config = _make(
        tmp_path,
        {
            "zotero_library": [("PARENT01", {"item_key": "PARENT01", "title": "T", "rekeyed_to": "ATT00001"})],
            PASSAGES: [(f"ATT00001#{n}", _passage(n)) for n in range(3)],
        },
    )

    report = repair_index(config, **QUIET)

    assert report.excluded == {}
    assert "PARENT01" in _ids(persist, "zotero_library")
    assert {"ATT00001#0", "ATT00001#1", "ATT00001#2"} <= set(_ids(persist, PASSAGES))
