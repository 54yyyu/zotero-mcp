"""Reads, deletes and resets cover both collections, in any layout (#769)."""

import importlib.util
import sys
from unittest.mock import MagicMock

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

from zotero_mcp import semantic_search  # noqa: E402
from zotero_mcp.chroma_client import (  # noqa: E402
    ChromaClient,
    passage_collection_name,
    read_index_layout,
)

NAME = "cleanup_test"


def _meta(key, **extra):
    return {"group_id": 0, "item_key": key, **extra}


def _passage_meta(key, index, total, **extra):
    return _meta(key, parent_item_key=key, chunk_index=index, n_chunks=total, **extra)


def _write(client, ids, metadatas):
    client.upsert_embeddings(
        documents=[f"text of {i}" for i in ids],
        metadatas=metadatas,
        ids=ids,
        embeddings=[[1.0, 0.0] for _ in ids],
    )


def _stored(client, name):
    return set(client.client.get_collection(name).get(include=[])["ids"])


@pytest.fixture()
def directory(tmp_path):
    return str(tmp_path / "chroma")


@pytest.fixture()
def client(directory):
    c = ChromaClient(collection_name=NAME, persist_directory=directory)
    c.route_passages = True
    return c


@pytest.fixture()
def both(client):
    """Item AAA at both levels, plus BBB as an article only."""
    _write(
        client,
        ["AAA", "BBB", "AAA#0", "AAA#1"],
        [_meta("AAA"), _meta("BBB"), _passage_meta("AAA", 0, 2), _passage_meta("AAA", 1, 2)],
    )
    return client


def test_existing_ids_are_found_in_either_collection(both):
    assert both.get_existing_ids(["AAA", "AAA#1", "CCC", "CCC#0"]) == {"AAA", "AAA#1"}


def test_checking_passage_ids_does_not_create_the_collection(client, directory):
    _write(client, ["AAA"], [_meta("AAA")])

    assert client.get_existing_ids(["AAA#0"]) == set()
    assert read_index_layout(directory, NAME)["passage_collection_exists"] is False


def test_existing_ids_handles_more_ids_than_one_query_may_bind(both):
    many = [f"K{i}#0" for i in range(12_000)] + ["AAA#0"]

    assert both.get_existing_ids(many) == {"AAA#0"}


def test_all_ids_come_from_both_collections(both):
    assert both.get_all_ids() == {"AAA", "BBB", "AAA#0", "AAA#1"}
    assert both.get_all_ids(where={"group_id": 1}) == set()


def test_document_exists_checks_both_collections(both):
    assert both.document_exists("AAA#0") is True
    assert both.document_exists("AAA") is True
    assert both.document_exists("CCC#0") is False


def test_item_metadata_comes_from_the_article_then_the_first_passage(client):
    _write(client, ["AAA"], [_meta("AAA", marker="article")])
    _write(client, ["AAA#0"], [_passage_meta("AAA", 0, 1, marker="passage")])
    _write(client, ["BBB#0"], [_passage_meta("BBB", 0, 1, marker="passage")])

    assert client.get_document_metadata("AAA")["marker"] == "article"
    assert client.get_document_metadata("BBB")["marker"] == "passage"
    assert client.get_document_metadata("CCC") is None


def test_delete_documents_removes_each_id_from_its_collection(both):
    both.delete_documents(["BBB", "AAA#1"])

    assert both.get_all_ids() == {"AAA", "AAA#0"}


def test_deleting_an_items_chunks_keeps_its_article_record(both):
    both.delete_item_chunks("AAA")

    assert both.get_all_ids() == {"AAA", "BBB"}


def test_deleting_chunks_respects_the_library_scope(both):
    both.delete_item_chunks("AAA", group_id=99)

    assert both.get_existing_ids(["AAA#0", "AAA#1"]) == {"AAA#0", "AAA#1"}


def test_pruning_passages_keeps_the_article_record(both):
    both.prune_item_chunks("AAA", keep=1)

    assert both.get_all_ids() == {"AAA", "BBB", "AAA#0"}


def test_pruning_an_unmigrated_index_keeps_the_bare_record(directory):
    """Every layout now writes an article-level record beside passages (#826)."""
    legacy = ChromaClient(collection_name=NAME, persist_directory=directory)
    legacy.route_passages = False
    _write(
        legacy,
        ["AAA", "AAA#0", "AAA#1"],
        [_meta("AAA"), _passage_meta("AAA", 0, 2), _passage_meta("AAA", 1, 2)],
    )

    legacy.prune_item_chunks("AAA", keep=1)

    assert legacy.get_all_ids() == {"AAA", "AAA#0"}


def test_reset_empties_both_collections(both, directory):
    both.reset_collection()

    layout = read_index_layout(directory, NAME)
    assert layout["article_records"] == 0
    assert layout["passage_records"] == 0
    assert layout["passage_collection_exists"] is False
    assert both.get_all_ids() == set()


def test_reset_turns_an_unmigrated_index_into_a_split_one(directory):
    legacy = ChromaClient(collection_name=NAME, persist_directory=directory)
    legacy.route_passages = False
    _write(legacy, ["AAA#0"], [_passage_meta("AAA", 0, 1)])
    client = ChromaClient(collection_name=NAME, persist_directory=directory)
    client.route_passages = True
    assert client.separate_passages() is False

    client.reset_collection()
    _write(client, ["AAA", "AAA#0"], [_meta("AAA"), _passage_meta("AAA", 0, 1)])

    assert _stored(client, NAME) == {"AAA"}
    assert _stored(client, passage_collection_name(NAME)) == {"AAA#0"}


def test_iterating_metadata_covers_both_collections(both):
    seen = {}
    for ids, metadatas in both.iter_metadatas():
        seen.update(zip(ids, metadatas))

    assert set(seen) == {"AAA", "BBB", "AAA#0", "AAA#1"}
    assert seen["AAA#1"]["chunk_index"] == 1


def test_metadata_updates_reach_the_right_collection(both):
    both.update_metadatas(
        ["AAA", "AAA#0"],
        [_meta("AAA", marker="x"), _passage_meta("AAA", 0, 2, marker="y")],
    )

    seen = {}
    for ids, metadatas in both.iter_metadatas():
        seen.update(zip(ids, metadatas))
    assert seen["AAA"]["marker"] == "x"
    assert seen["AAA#0"]["marker"] == "y"


def test_read_error_reports_a_failure_in_either_collection(both):
    assert both.read_error() is None

    real = both._passages()

    class Broken:
        id = real.id

        def count(self):
            raise RuntimeError("segment damaged")

    both._passage_collection = Broken()

    assert "segment damaged" in both.read_error()


# --- the deletion pass -----------------------------------------------------------


@pytest.fixture()
def search(monkeypatch, client):
    monkeypatch.setattr(semantic_search, "get_zotero_client", MagicMock)
    s = semantic_search.ZoteroSemanticSearch(chroma_client=client)
    s._run_group_id = 0
    return s


@pytest.mark.parametrize("chunking", [True, False])
def test_removed_item_leaves_both_collections(search, chunking):
    search._chunking_config = {"enabled": chunking}
    _write(
        search.chroma_client,
        ["GONE", "GONE#0", "GONE#1", "KEPT", "KEPT#0"],
        [
            _meta("GONE"),
            _passage_meta("GONE", 0, 2),
            _passage_meta("GONE", 1, 2),
            _meta("KEPT"),
            _passage_meta("KEPT", 0, 1),
        ],
    )
    stats = {"deleted_items": 0}

    search._run_deletion_pass(stats, {"KEPT"}, allow_mass_deletion=True)

    assert search.chroma_client.get_all_ids() == {"KEPT", "KEPT#0"}
    assert stats["deleted_items"] == 1


def test_removed_item_with_only_passages_leaves_when_chunking_is_off(search):
    """The old pass picked one kind of delete from the setting, and missed these."""
    search._chunking_config = {"enabled": False}
    _write(search.chroma_client, ["GONE#0"], [_passage_meta("GONE", 0, 1)])

    search._run_deletion_pass({"deleted_items": 0}, {"KEPT"}, allow_mass_deletion=True)

    assert search.chroma_client.get_all_ids() == set()


def test_deletion_stays_within_the_library_being_synced(search):
    search._chunking_config = {"enabled": True}
    _write(
        search.chroma_client,
        ["OTHER", "OTHER#0"],
        [
            {"group_id": 5, "item_key": "OTHER"},
            {"group_id": 5, "item_key": "OTHER", "parent_item_key": "OTHER", "chunk_index": 0, "n_chunks": 1},
        ],
    )

    search._run_deletion_pass({"deleted_items": 0}, {"KEPT"}, allow_mass_deletion=True)

    assert search.chroma_client.get_all_ids() == {"OTHER", "OTHER#0"}


def test_delete_item_removes_both_levels(search):
    _write(search.chroma_client, ["GONE", "GONE#0"], [_meta("GONE"), _passage_meta("GONE", 0, 1)])

    assert search.delete_item("GONE") is True

    assert search.chroma_client.get_all_ids() == set()
