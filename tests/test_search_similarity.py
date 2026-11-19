"""The score a semantic search reports for a hit is a similarity.

The score was ``1 - distance`` whatever the distance meant. ChromaDB ranks by
squared L2 unless told otherwise, so a passage with cosine 0.69 to the query
was shown as 0.37 and an unrelated one as about -1.
"""

import importlib.util
import math
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

from test_passage_chunking import _chunking_search

from zotero_mcp.chroma_client import ChromaClient
from zotero_mcp.semantic_search import similarity_from_distance


def _unit(angle_degrees):
    angle = math.radians(angle_degrees)
    return [math.cos(angle), math.sin(angle)]


@pytest.fixture()
def client(tmp_path):
    return ChromaClient(
        collection_name="hits_test",
        persist_directory=str(tmp_path / "chroma"),
        embedding_model="default",
    )


class _Zotero:
    def item(self, key):
        return {"key": key, "data": {"title": f"Title {key}"}}


def test_the_default_space_is_reported_as_a_cosine_similarity(client, monkeypatch):
    """Against a real collection: the score of a hit is its cosine to the query."""
    angles = {"SAME": 0, "NEAR": 30, "ORTHOGONAL": 90, "OPPOSITE": 180}
    client.upsert_embeddings(
        documents=[f"text of {k}" for k in angles],
        metadatas=[{"item_key": k} for k in angles],
        ids=list(angles),
        embeddings=[_unit(a) for a in angles.values()],
    )
    assert client._distance_space(client.collection) == "l2"
    raw = client.collection.query(query_embeddings=[_unit(0)], n_results=4)
    raw["space"] = client._distance_space(client.collection)

    search = _chunking_search(monkeypatch)
    search.zotero_client = _Zotero()
    scores = {r["item_key"]: r["similarity_score"] for r in search._enrich_search_results(raw, "q", limit=4)}

    for key, angle in angles.items():
        assert scores[key] == pytest.approx(math.cos(math.radians(angle)), abs=1e-5)


def test_search_says_which_space_its_distances_are_in(client, monkeypatch):
    class Collection:
        configuration_json = {"hnsw": {"space": "cosine"}}

        def query(self, **kwargs):
            return {"ids": [["A"]], "distances": [[0.25]], "documents": [["a"]], "metadatas": [[{}]]}

    monkeypatch.setattr(client, "_collection_for_level", lambda level: Collection())
    monkeypatch.setattr(client, "embedding_function", object())

    assert client.search(query_texts=["q"], n_results=1)["space"] == "cosine"


def test_a_cosine_or_unknown_space_keeps_one_minus_distance():
    assert similarity_from_distance(0.25, "cosine") == 0.75
    assert similarity_from_distance(0.25, "ip") == 0.75
    assert similarity_from_distance(0.25, None) == 0.75
    assert similarity_from_distance(0.25, "l2") == 0.875
    assert similarity_from_distance(None, "l2") == 0


def test_an_older_collection_reads_its_space_from_its_metadata():
    class Old:
        configuration_json = {}
        metadata = {"hnsw:space": "cosine"}

    class Bare:
        metadata = None

        @property
        def configuration_json(self):
            raise AttributeError("no configuration on this ChromaDB")

    assert ChromaClient._distance_space(Old()) == "cosine"
    assert ChromaClient._distance_space(Bare()) == "l2"
