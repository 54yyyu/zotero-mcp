"""An attachment with only some of its passages has a passage gap (#826, fix round 2b).

A failed row or request can leave ``PDFAAAAA#3`` missing while ``#0..#2`` are
there; the next explicit update writes that attachment's passages and nothing
else. Real Chroma in a temp dir, a recording embedder, a fake reader; HOME,
Path.home and the config path point into tmp_path.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 14):
    pytest.skip("chromadb needs pydantic v1 paths on 3.14+", allow_module_level=True)

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

from _scan_fakes import target  # noqa: E402
from chromadb.api.models.CollectionCommon import CollectionCommon  # noqa: E402
from test_item_keyed_local_scan import NAME, FakeZotero, Item, LibReader  # noqa: E402

from zotero_mcp import semantic_search  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient  # noqa: E402

ITEM = "ITEM0001"
ATT = "PDFAAAAA"
PASSAGES = f"{NAME}_passages"


class Reader(LibReader):
    def get_item_by_key(self, key):
        return next((item for item in self.items if item.key == key), None)


class World:
    def __init__(self, monkeypatch, tmp_path):
        self.monkeypatch = monkeypatch
        # The only embedder: it records what it is asked for, and no provider is reachable.
        self.embedded: list[str] = []

        def embed(collection, input, **kwargs):
            self.embedded.extend(input)
            return [[1.0, 0.0] for _ in input]

        monkeypatch.setattr(CollectionCommon, "_embed", embed)
        monkeypatch.setattr(ChromaClient, "truncate_text", lambda self, text, max_tokens=None: text)
        self.directory = str(tmp_path / "chroma")
        self.config = str(tmp_path / "config.json")
        Path(self.config).write_text(
            json.dumps(
                {
                    "semantic_search": {
                        "persist_directory": self.directory,
                        "collection_name": NAME,
                        "index_schema_version": semantic_search._INDEX_SCHEMA_VERSION,
                        "chunking": {"enabled": True, "chunk_size": 400, "overlap": 50, "max_chunks_per_item": 5},
                    }
                }
            )
        )

    def update(self, reader, **kwargs):
        self.monkeypatch.setattr(semantic_search, "LocalZoteroReader", lambda *a, **kw: reader)
        self.monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: FakeZotero(reader))
        client = ChromaClient(collection_name=NAME, persist_directory=self.directory)
        search = semantic_search.ZoteroSemanticSearch(chroma_client=client, config_path=self.config)
        reader.extracted.clear()
        stats = search.update_database(extract_fulltext=True, **kwargs)
        assert stats.get("error") is None, stats
        return search, stats

    def stored(self, name):
        client = ChromaClient(collection_name=NAME, persist_directory=self.directory)
        got = client.client.get_collection(name).get(include=["embeddings", "metadatas"])
        return {
            doc_id: (meta, [float(x) for x in emb])
            for doc_id, meta, emb in zip(got["ids"], got["metadatas"], got["embeddings"])
        }


@pytest.fixture()
def world(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    return World(monkeypatch, tmp_path)


def _reader():
    return Reader([Item(ITEM, 1)], targets={1: [target(ATT)]})


def test_a_missing_passage_is_refilled_by_the_next_explicit_update(world):
    world.update(_reader())
    passages = sorted(world.stored(PASSAGES))
    assert len(passages) == 5
    ChromaClient(collection_name=NAME, persist_directory=world.directory).client.get_collection(PASSAGES).delete(
        ids=[f"{ATT}#3"]
    )
    world.embedded.clear()

    world.update(_reader())

    assert sorted(world.stored(PASSAGES)) == passages
    assert world.embedded, "the attachment's passages were written again"
    assert all("Body sentence" in text for text in world.embedded), "and nothing else: no record of the item"
    assert ITEM in world.stored(NAME)
