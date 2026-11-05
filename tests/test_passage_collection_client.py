"""ChromaClient with a separate passage collection, on a real persistent index (#769).

Nothing here embeds: vectors are precomputed, as in test_chroma_client_real.py.
"""

import importlib.util
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

from zotero_mcp.chroma_client import (  # noqa: E402
    ChromaClient,
    passage_collection_name,
    read_collection_status,
    read_index_layout,
)

NAME = "layout_test"


def _meta(key, **extra):
    return {"group_id": 0, "item_key": key, "title": f"Title {key}", **extra}


def _passage_meta(key, index, total):
    return _meta(key, parent_item_key=key, chunk_index=index, n_chunks=total)


def _write(client, ids, metadatas, vector=(1.0, 0.0)):
    client.upsert_embeddings(
        documents=[f"text of {i}" for i in ids],
        metadatas=metadatas,
        ids=ids,
        embeddings=[list(vector) for _ in ids],
    )


@pytest.fixture()
def directory(tmp_path):
    return str(tmp_path / "chroma")


@pytest.fixture()
def client(directory):
    """Routing on, as it will be once the write path uses it."""
    c = ChromaClient(collection_name=NAME, persist_directory=directory)
    c.route_passages = True
    return c


def _stored(client, name):
    return set(client.client.get_collection(name).get(include=[])["ids"])


def test_passage_collection_name_follows_the_configured_name():
    assert passage_collection_name("zotero_library") == "zotero_library_passages"
    assert passage_collection_name("custom") == "custom_passages"


def test_opening_a_client_does_not_create_the_passage_collection(client, directory):
    _write(client, ["AAA"], [_meta("AAA")])

    layout = read_index_layout(directory, NAME)
    assert layout["article_records"] == 1
    assert layout["passage_collection_exists"] is False


def test_writes_go_to_the_collection_their_id_belongs_to(client):
    _write(
        client,
        ["AAA", "AAA#0", "AAA#1"],
        [_meta("AAA"), _passage_meta("AAA", 0, 2), _passage_meta("AAA", 1, 2)],
    )

    assert _stored(client, NAME) == {"AAA"}
    assert _stored(client, passage_collection_name(NAME)) == {"AAA#0", "AAA#1"}


def test_with_routing_off_everything_stays_in_one_collection(directory):
    client = ChromaClient(collection_name=NAME, persist_directory=directory)
    assert client.separate_passages() is False

    _write(client, ["AAA", "BBB#0"], [_meta("AAA"), _passage_meta("BBB", 0, 1)])

    assert _stored(client, NAME) == {"AAA", "BBB#0"}
    assert read_index_layout(directory, NAME)["passage_collection_exists"] is False


def test_index_with_passages_in_the_article_collection_keeps_writing_there(directory):
    """An unmigrated chunked index goes on working as it does today."""
    legacy = ChromaClient(collection_name=NAME, persist_directory=directory)
    _write(legacy, ["AAA#0"], [_passage_meta("AAA", 0, 1)])

    client = ChromaClient(collection_name=NAME, persist_directory=directory)
    client.route_passages = True

    assert client.index_layout()["unmigrated_passages"] == 1
    assert client.separate_passages() is False

    _write(client, ["BBB#0"], [_passage_meta("BBB", 0, 1)])

    assert _stored(client, NAME) == {"AAA#0", "BBB#0"}
    assert read_index_layout(directory, NAME)["passage_collection_exists"] is False


def test_layout_counts_each_level(client, directory):
    _write(
        client,
        ["AAA", "BBB", "AAA#0", "AAA#1", "AAA#2"],
        [_meta("AAA"), _meta("BBB")] + [_passage_meta("AAA", i, 3) for i in range(3)],
    )

    layout = read_index_layout(directory, NAME)

    assert layout["article_records"] == 2
    assert layout["passage_records"] == 3
    assert layout["passage_items"] == 1
    assert layout["unmigrated_passages"] == 0
    assert layout["passage_collection_exists"] is True


def test_layout_of_a_directory_with_no_index(tmp_path):
    layout = read_index_layout(str(tmp_path / "nothing"), NAME)

    assert layout["article_records"] == 0
    assert layout["passage_records"] == 0
    assert layout["passage_collection_exists"] is False


def test_searching_by_level_queries_only_that_collection(client):
    _write(client, ["AAA"], [_meta("AAA")])
    _write(client, ["BBB#0"], [_passage_meta("BBB", 0, 1)])

    def query(level):
        collection = client._collection_for_level(level)
        return collection.query(query_embeddings=[[1.0, 0.0]], n_results=5)["ids"][0]

    assert query("articles") == ["AAA"]
    assert query("passages") == ["BBB#0"]


def test_passage_collection_uses_the_same_embedding_function(client):
    _write(client, ["BBB#0"], [_passage_meta("BBB", 0, 1)])

    assert (
        client._collection_for_level("passages")._embedding_function
        is client.embedding_function
    )


def test_status_reports_both_levels_without_opening_chromadb(client, directory, tmp_path):
    _write(
        client,
        ["AAA", "AAA#0", "AAA#1"],
        [_meta("AAA"), _passage_meta("AAA", 0, 2), _passage_meta("AAA", 1, 2)],
    )
    config = tmp_path / "config.json"
    config.write_text(f'{{"semantic_search": {{"collection_name": "{NAME}"}}}}')

    status = read_collection_status(str(config), persist_directory=directory)

    assert status["count"] == 3
    assert status["article_records"] == 1
    assert status["passage_records"] == 2
    assert status["unmigrated_passages"] == 0


def test_collection_info_reports_both_levels(client):
    _write(client, ["AAA", "AAA#0"], [_meta("AAA"), _passage_meta("AAA", 0, 1)])

    info = client.get_collection_info()

    assert info["count"] == 2
    assert info["article_records"] == 1
    assert info["passage_records"] == 1


def test_an_open_client_notices_when_a_collection_is_replaced(client, directory):
    """A long-running worker must not keep a handle to the renamed collection."""
    _write(client, ["AAA#0"], [_passage_meta("AAA", 0, 1)])
    legacy_writer = ChromaClient(collection_name="other", persist_directory=directory)
    _write(legacy_writer, ["ZZZ"], [_meta("ZZZ")])

    client.client.delete_collection(NAME)
    legacy_writer.collection.modify(name=NAME)

    client.index_layout()

    assert set(client.collection.get(include=[])["ids"]) == {"ZZZ"}
