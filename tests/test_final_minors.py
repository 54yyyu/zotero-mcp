"""Smaller fixes of the final review of the item-keyed index (#826).

Real Chroma in temp directories with given vectors; any embedding request fails
the test. HOME and Path.home point into tmp_path.
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

from chromadb.api.models.CollectionCommon import CollectionCommon  # noqa: E402

from zotero_mcp import semantic_search  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient  # noqa: E402


@pytest.fixture()
def search(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: None)

    def refuse(self, *args, **kwargs):
        raise AssertionError("an embedding was requested")

    monkeypatch.setattr(CollectionCommon, "_embed", refuse)
    client = ChromaClient(collection_name="minors", persist_directory=str(tmp_path / "chroma"))
    return semantic_search.ZoteroSemanticSearch(chroma_client=client, config_path=str(tmp_path / "config.json"))


def _put(search, ids_groups):
    ids = list(ids_groups)
    search.chroma_client.upsert_embeddings(
        documents=[f"text {i}" for i in ids],
        metadatas=[
            {"item_key": i, "group_id": g, "date_modified": "", "file_stamp": "1:1"} for i, g in ids_groups.items()
        ],
        ids=ids,
        embeddings=[[1.0, 0.0]] * len(ids),
    )


def test_deleting_records_is_scoped_to_the_runs_library(search):
    _put(search, {"MINE0001": 0, "THEIRS01": 7})
    search._run_group_id = 0

    removed = search._delete_records({"MINE0001", "THEIRS01"})

    stored = set(search.chroma_client.collection.get(include=[])["ids"])
    assert stored == {"THEIRS01"}, "another library's record with a colliding key is left alone"
    assert removed == 1
