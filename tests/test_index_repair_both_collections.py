"""``repair-db`` keeps both collections of a split index (#769)."""

import importlib.util
import json
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

if sys.platform == "win32":
    pytest.skip("ChromaDB keeps the index files open in this process on Windows", allow_module_level=True)

import chromadb
import numpy as np
from chromadb.api.client import SharedSystemClient
from chromadb.config import Settings

from zotero_mcp.chroma_client import read_index_layout
from zotero_mcp.index_repair import RepairError, repair_index

ITEMS = 12
CHUNKS = 3
DIM = 8
PASSAGES = "zotero_library_passages"
QUIET = {"progress": lambda _m: None}
# A low sync threshold makes ChromaDB write the HNSW files for a handful of vectors.
SAVED = {"hnsw:sync_threshold": 10, "hnsw:batch_size": 10}


def _article_ids():
    return [f"ITEM{i:04d}" for i in range(ITEMS)]


def _passage_ids():
    return [f"ITEM{i:04d}#{n}" for i in range(ITEMS) for n in range(CHUNKS)]


def _add_articles(client, metadata=SAVED):
    ids = _article_ids()
    client.create_collection("zotero_library", metadata=metadata).add(
        ids=ids,
        embeddings=np.random.default_rng(1).normal(size=(len(ids), DIM)).astype(np.float32),
        documents=[f"text of {d}" for d in ids],
        metadatas=[{"item_key": d, "title": d} for d in ids],
    )


def _add_passages(client, metadata=SAVED):
    ids = _passage_ids()
    client.create_collection(PASSAGES, metadata=metadata).add(
        ids=ids,
        embeddings=np.random.default_rng(2).normal(size=(len(ids), DIM)).astype(np.float32),
        documents=[f"text of {d}" for d in ids],
        metadatas=[
            {
                "item_key": d.split("#")[0],
                "parent_item_key": d.split("#")[0],
                "chunk_index": int(d.split("#")[1]),
                "n_chunks": CHUNKS,
            }
            for d in ids
        ],
    )


@pytest.fixture()
def make_index(tmp_path):
    persist = tmp_path / "chroma_db"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"semantic_search": {
        "persist_directory": str(persist),
        "embedding_model": "default",
    }}))

    def _make(*builders):
        client = chromadb.PersistentClient(path=str(persist), settings=Settings(anonymized_telemetry=False))
        for build in builders:
            build(client)
        del client
        SharedSystemClient.clear_system_cache()
        return persist, str(config)

    yield _make
    SharedSystemClient.clear_system_cache()


def _ids_in(persist: Path, name: str) -> list[str]:
    client = chromadb.PersistentClient(path=str(persist), settings=Settings(anonymized_telemetry=False))
    got = client.get_collection(name).get(include=[])["ids"]
    SharedSystemClient.clear_system_cache()
    return sorted(got)


def test_both_collections_survive_a_repair(make_index):
    persist, config = make_index(_add_articles, _add_passages)

    report = repair_index(config, **QUIET)

    assert report.excluded == {}
    assert report.items_kept == ITEMS
    assert report.chunks_kept == ITEMS + ITEMS * CHUNKS
    assert _ids_in(persist, "zotero_library") == sorted(_article_ids())
    assert _ids_in(persist, PASSAGES) == sorted(_passage_ids())


def test_an_unsplit_index_stays_unsplit(make_index):
    def _legacy(client):
        ids = _passage_ids()
        client.create_collection("zotero_library", metadata=SAVED).add(
            ids=ids,
            embeddings=np.random.default_rng(2).normal(size=(len(ids), DIM)).astype(np.float32),
            documents=[f"text of {d}" for d in ids],
            metadatas=[
                {"item_key": d.split("#")[0], "chunk_index": int(d.split("#")[1]), "n_chunks": CHUNKS}
                for d in ids
            ],
        )

    persist, config = make_index(_legacy)

    repair_index(config, **QUIET)

    layout = read_index_layout(str(persist), "zotero_library")
    assert layout["unmigrated_passages"] == ITEMS * CHUNKS
    assert layout["passage_collection_exists"] is False


def test_a_collection_with_no_saved_vectors_does_not_stop_the_other(make_index):
    """A small article collection beside a large passage one is a normal state."""
    persist, config = make_index(lambda c: _add_articles(c, metadata=None), _add_passages)

    report = repair_index(config, **QUIET)

    assert report.collections_left_out == ["zotero_library"]
    assert report.chunks_kept == ITEMS * CHUNKS
    assert _ids_in(persist, PASSAGES) == sorted(_passage_ids())
    assert json.loads(Path(report.report_path).read_text())["collections_left_out"] == ["zotero_library"]


def test_no_saved_vectors_anywhere_is_refused(make_index):
    _, config = make_index(lambda c: _add_articles(c, metadata=None))

    with pytest.raises(RepairError, match="no saved vector files"):
        repair_index(config, **QUIET)


def test_dry_run_reports_both_collections_and_changes_nothing(make_index):
    persist, config = make_index(_add_articles, _add_passages)

    report = repair_index(config, dry_run=True, **QUIET)

    assert report.chunks_kept == ITEMS + ITEMS * CHUNKS
    assert report.broken_directory is None
    assert _ids_in(persist, "zotero_library") == sorted(_article_ids())
