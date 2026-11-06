"""The guards in front of a rebuild's reset read both ChromaDB collections (#769).

A reset deletes the passage collection too, so the count of other libraries'
documents and the "does the index hold passages" check must see it.
"""

import importlib.util
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

from zotero_mcp.chroma_client import ChromaClient  # noqa: E402

NAME = "guard_test"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    c = ChromaClient(collection_name=NAME, persist_directory=str(tmp_path / "chroma"))
    c.route_passages = True
    return c


def _write(client, ids, metadatas):
    client.upsert_embeddings(
        documents=[f"text of {i}" for i in ids],
        metadatas=metadatas,
        ids=ids,
        embeddings=[[1.0, 0.0] for _ in ids],
    )


def test_another_librarys_passages_count_as_outside_the_library(client):
    """Right after a migration, another library's items may have only passages."""
    _write(
        client,
        ["AAA", "BBB#0", "BBB#1"],
        [
            {"group_id": 0, "item_key": "AAA"},
            {"group_id": 7, "item_key": "BBB", "parent_item_key": "BBB", "chunk_index": 0, "n_chunks": 2},
            {"group_id": 7, "item_key": "BBB", "parent_item_key": "BBB", "chunk_index": 1, "n_chunks": 2},
        ],
    )

    assert client.count_outside_library(0) == 2


def test_passages_in_their_own_collection_count_as_passage_documents(client):
    _write(
        client,
        ["AAA", "AAA#0"],
        [
            {"group_id": 0, "item_key": "AAA"},
            {"group_id": 0, "item_key": "AAA", "parent_item_key": "AAA", "chunk_index": 0, "n_chunks": 1},
        ],
    )

    assert client.has_passage_documents() is True


def test_an_article_only_index_has_no_passage_documents(client):
    _write(client, ["AAA"], [{"group_id": 0, "item_key": "AAA"}])

    assert client.has_passage_documents() is False
    assert client.count_outside_library(0) == 0
