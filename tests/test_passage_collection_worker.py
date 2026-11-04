"""WorkerChromaClient serves both collections, and notices a migration (#769)."""

import importlib.util
import json
import os
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

from zotero_mcp import chroma_worker
from zotero_mcp.chroma_client import ChromaClient, create_chroma_client
from zotero_mcp.chroma_worker import WorkerChromaClient
from zotero_mcp.index_migration import migrate_index


def _write(client, ids):
    client.upsert_embeddings(
        documents=[f"text of {i}" for i in ids],
        metadatas=[
            {"group_id": 0, "item_key": i.split("#")[0], "parent_item_key": i.split("#")[0],
             "chunk_index": 0, "n_chunks": 1}
            if "#" in i
            else {"group_id": 0, "item_key": i}
            for i in ids
        ],
        ids=ids,
        embeddings=[[1.0, 0.0] for _ in ids],
    )


@pytest.fixture()
def config(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"semantic_search": {
        "persist_directory": str(tmp_path / "chroma_db"),
        "embedding_model": "default",
        "idle_unload_minutes": 10,
    }}))
    src = os.path.dirname(os.path.dirname(chroma_worker.__file__))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")])))
    monkeypatch.setattr(chroma_worker, "_enabled", True)
    monkeypatch.setattr(chroma_worker, "_shared", {})
    yield str(path)
    chroma_worker.stop_all()


def test_both_collections_work_through_the_worker(config):
    client = create_chroma_client(config)
    assert isinstance(client, WorkerChromaClient)

    _write(client, ["AAA", "AAA#0", "BBB#0"])

    assert client.separate_passages() is True
    layout = client.index_layout()
    assert (layout["article_records"], layout["passage_records"]) == (1, 2)
    assert client.get_existing_ids(["AAA", "AAA#0", "ZZZ#0"]) == {"AAA", "AAA#0"}
    assert client.item_keys_by_level() == ({"AAA"}, {"AAA", "BBB"})

    client.prune_item_chunks("AAA", 0)
    assert client.get_all_ids() == {"AAA", "BBB#0"}

    client.reset_collection()
    assert client.get_all_ids() == set()


def test_a_running_worker_picks_up_a_migration(config, tmp_path):
    """Migrating from the command line while a server's worker has the index open."""
    legacy = ChromaClient(
        collection_name="zotero_library", persist_directory=str(tmp_path / "chroma_db")
    )
    legacy.route_passages = False
    _write(legacy, ["AAA#0"])
    del legacy

    worker = create_chroma_client(config)
    assert worker.separate_passages() is False
    assert worker.get_existing_ids(["AAA#0"]) == {"AAA#0"}

    migrate_index(config, progress=lambda _m: None)

    layout = worker.index_layout()
    assert layout["unmigrated_passages"] == 0
    assert worker.separate_passages() is True
    assert worker.get_existing_ids(["AAA#0"]) == {"AAA#0"}

    _write(worker, ["AAA"])
    assert worker.index_layout()["article_records"] == 1
