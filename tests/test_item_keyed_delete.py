"""Deleting and pruning passages by the record's own key (#826).

New passages ``ATT#n`` carry ``item_key=ATT``; legacy passages ``KEY#n`` carry
``item_key=KEY`` (and ``parent_item_key=KEY``). Deletes select on ``item_key``
plus ``chunk_index``, so they reach both shapes and never a bare record.
Runs against a real persistent index with precomputed vectors; nothing embeds.
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

from zotero_mcp.chroma_client import ChromaClient, record_key  # noqa: E402

NAME = "item_keyed_delete"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")
    return ChromaClient(collection_name=NAME, persist_directory=str(tmp_path / "chroma"))


def _write(client, records):
    ids = [doc_id for doc_id, _meta in records]
    client.upsert_embeddings(
        documents=[f"text of {i}" for i in ids],
        metadatas=[meta for _id, meta in records],
        ids=ids,
        embeddings=[[1.0, 0.0] for _ in ids],
    )


def _new_passage(att, index, total, group_id=0):
    return (f"{att}#{index}", {"item_key": att, "group_id": group_id, "chunk_index": index, "n_chunks": total})


def _legacy_passage(key, index, total, group_id=0):
    return (
        f"{key}#{index}",
        {
            "item_key": key,
            "parent_item_key": key,
            "title": "Old",
            "group_id": group_id,
            "chunk_index": index,
            "n_chunks": total,
        },
    )


def _ids(client):
    return client.get_all_ids()


def test_record_key_is_the_id_prefix():
    assert record_key("ATTKEY01#3") == "ATTKEY01"
    assert record_key("ITEMKEY1") == "ITEMKEY1"


def test_delete_item_chunks_removes_new_passages_and_keeps_the_bare_record(client):
    _write(
        client,
        [
            ("ATT00001", {"item_key": "ATT00001", "group_id": 0, "file_stamp": "1:2"}),
            _new_passage("ATT00001", 0, 2),
            _new_passage("ATT00001", 1, 2),
            _new_passage("ATT00002", 0, 1),
        ],
    )

    client.delete_item_chunks("ATT00001")

    assert _ids(client) == {"ATT00001", "ATT00002#0"}


def test_delete_item_chunks_reaches_legacy_passages_by_their_item_key(client):
    _write(
        client,
        [
            ("ITEM0001", {"item_key": "ITEM0001", "group_id": 0, "date_modified": "x"}),
            _legacy_passage("ITEM0001", 0, 2),
            _legacy_passage("ITEM0001", 1, 2),
        ],
    )

    client.delete_item_chunks("ITEM0001")

    assert _ids(client) == {"ITEM0001"}


def test_delete_item_chunks_scoped_to_a_library(client):
    _write(client, [_new_passage("ATT00001", 0, 2, group_id=0), _new_passage("ATT00001", 1, 2, group_id=7)])

    client.delete_item_chunks("ATT00001", group_id=0)

    assert _ids(client) == {"ATT00001#1"}


def test_prune_item_chunks_keeps_the_head_and_the_article_record(client):
    _write(
        client,
        [
            ("ATT00001", {"item_key": "ATT00001", "group_id": 0, "file_stamp": "1:2"}),
            *(_new_passage("ATT00001", i, 4) for i in range(4)),
        ],
    )

    client.prune_item_chunks("ATT00001", 2)

    assert _ids(client) == {"ATT00001", "ATT00001#0", "ATT00001#1"}


def test_prune_to_zero_drops_legacy_passages_only(client):
    _write(
        client,
        [
            ("ITEM0001", {"item_key": "ITEM0001", "group_id": 0, "date_modified": "x"}),
            _legacy_passage("ITEM0001", 0, 2),
            _legacy_passage("ITEM0001", 1, 2),
        ],
    )

    client.prune_item_chunks("ITEM0001", 0)

    assert _ids(client) == {"ITEM0001"}


def test_passage_documents_lists_one_records_passages(client):
    _write(
        client,
        [
            ("ATT00001", {"item_key": "ATT00001", "group_id": 0, "file_stamp": "1:2"}),
            _new_passage("ATT00001", 0, 2),
            _new_passage("ATT00001", 1, 2),
            _new_passage("ATT00002", 0, 1),
        ],
    )

    found = dict(client.passage_documents("ATT00001"))

    assert set(found) == {"ATT00001#0", "ATT00001#1"}


def test_item_group_ids_reads_record_keys(client):
    _write(client, [_new_passage("ATT00001", 0, 1, group_id=5)])

    assert client.item_group_ids(["ATT00001"]) == {"ATT00001": 5}
