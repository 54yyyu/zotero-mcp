"""``migrate-index``: moving an unsplit chunked index to two collections (#769).

Real persisted indexes in temp directories, with given vectors, so nothing embeds.
"""

import importlib.util
import json
import subprocess
import sys
import textwrap

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

import numpy as np

from zotero_mcp import index_migration
from zotero_mcp.chroma_client import ChromaClient, read_index_layout
from zotero_mcp.index_migration import MigrationError, migrate_index

NAME = "zotero_library"
PASSAGES = "zotero_library_passages"
QUIET = {"progress": lambda _m: None}


def _meta(key, **extra):
    return {"group_id": 0, "item_key": key, **extra}


def _passage_meta(key, index, total):
    return _meta(key, parent_item_key=key, chunk_index=index, n_chunks=total)


def _vectors(ids):
    rng = np.random.default_rng(3)
    return {doc_id: rng.normal(size=4).astype(float).tolist() for doc_id in ids}


def _write(client, ids, metadatas, vectors):
    client.upsert_embeddings(
        documents=[f"text of {i}" for i in ids],
        metadatas=metadatas,
        ids=ids,
        embeddings=[vectors[i] for i in ids],
    )


@pytest.fixture()
def index(tmp_path):
    directory = tmp_path / "chroma"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"semantic_search": {
        "persist_directory": str(directory),
        "embedding_model": "default",
    }}))
    return str(directory), str(config)


def _legacy_client(directory):
    client = ChromaClient(collection_name=NAME, persist_directory=directory)
    client.route_passages = False
    return client


def _seed_chunked(directory, items=("AAA", "BBB"), chunks=2, extra_bare=()):
    client = _legacy_client(directory)
    ids = [f"{k}#{n}" for k in items for n in range(chunks)]
    metas = [_passage_meta(k, n, chunks) for k in items for n in range(chunks)]
    ids += list(extra_bare)
    metas += [_meta(k) for k in extra_bare]
    vectors = _vectors(ids)
    _write(client, ids, metas, vectors)
    return vectors


def _read(directory, name):
    client = ChromaClient(collection_name=NAME, persist_directory=directory)
    got = client.client.get_collection(name).get(include=["embeddings", "documents", "metadatas"])
    return {
        doc_id: (doc, meta, [float(x) for x in emb])
        for doc_id, doc, meta, emb in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"])
    }


def test_passages_move_with_their_vectors_and_nothing_embeds(index):
    directory, config = index
    vectors = _seed_chunked(directory)

    report = migrate_index(config, **QUIET)

    assert report.renamed is True
    stored = _read(directory, PASSAGES)
    assert set(stored) == {"AAA#0", "AAA#1", "BBB#0", "BBB#1"}
    for doc_id, (doc, meta, emb) in stored.items():
        assert doc == f"text of {doc_id}"
        assert meta["parent_item_key"] == doc_id.split("#")[0]
        assert emb == pytest.approx(vectors[doc_id], abs=1e-5)
    assert _read(directory, NAME) == {}


def test_report_says_how_many_article_vectors_are_left_to_embed(index):
    directory, config = index
    _seed_chunked(directory)

    report = migrate_index(config, **QUIET)

    assert report.items_needing_articles == 2
    assert report.passage_records == 4


def test_a_migrated_index_is_recognised_as_split(index):
    directory, config = index
    _seed_chunked(directory)

    migrate_index(config, **QUIET)

    layout = read_index_layout(directory, NAME)
    assert layout["unmigrated_passages"] == 0
    assert layout["passage_records"] == 4
    client = ChromaClient(collection_name=NAME, persist_directory=directory)
    assert client.separate_passages() is True


def test_bare_records_in_a_mixed_index_stay_articles(index):
    """Turning chunking on without a rebuild has left indexes with both kinds."""
    directory, config = index
    vectors = _seed_chunked(directory, extra_bare=("OLD",))

    report = migrate_index(config, **QUIET)

    assert report.moved_articles == 1
    stored = _read(directory, NAME)
    assert set(stored) == {"OLD"}
    assert stored["OLD"][2] == pytest.approx(vectors["OLD"], abs=1e-5)
    assert "OLD" not in _read(directory, PASSAGES)
    assert report.items_needing_articles == 2


def test_dry_run_changes_nothing(index):
    directory, config = index
    _seed_chunked(directory, extra_bare=("OLD",))

    report = migrate_index(config, dry_run=True, **QUIET)

    assert report.renamed is False
    assert report.passage_records == 4
    assert report.moved_articles == 1
    assert report.items_needing_articles == 2
    layout = read_index_layout(directory, NAME)
    assert layout["unmigrated_passages"] == 4
    assert layout["passage_collection_exists"] is False


def test_item_level_index_has_nothing_to_migrate(index):
    directory, config = index
    client = ChromaClient(collection_name=NAME, persist_directory=directory)
    ids = ["AAA", "BBB"]
    _write(client, ids, [_meta(k) for k in ids], _vectors(ids))

    report = migrate_index(config, **QUIET)

    assert report.needed is False
    assert set(_read(directory, NAME)) == {"AAA", "BBB"}


def test_running_twice_changes_nothing_more(index):
    directory, config = index
    _seed_chunked(directory, extra_bare=("OLD",))
    migrate_index(config, **QUIET)

    report = migrate_index(config, **QUIET)

    assert report.needed is False
    assert report.renamed is False
    assert set(_read(directory, NAME)) == {"OLD"}


def test_missing_index_is_an_error(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"semantic_search": {"persist_directory": str(tmp_path / "none")}}))

    with pytest.raises(MigrationError, match="no index"):
        migrate_index(str(config), **QUIET)


def test_passages_written_to_the_article_collection_after_a_split_move_over(index):
    """An older version running against a split index writes passages the old way."""
    directory, config = index
    _seed_chunked(directory, items=("AAA",), chunks=1)
    migrate_index(config, **QUIET)
    stray = _vectors(["BBB#0"])
    _write(_legacy_client(directory), ["BBB#0"], [_passage_meta("BBB", 0, 1)], stray)

    report = migrate_index(config, **QUIET)

    assert report.needed is True
    assert report.moved_passages == 1
    assert "BBB#0" in _read(directory, PASSAGES)
    assert _read(directory, NAME) == {}


def test_a_stray_passage_does_not_overwrite_a_newer_one(index):
    directory, config = index
    _seed_chunked(directory, items=("AAA",), chunks=1)
    migrate_index(config, **QUIET)
    legacy = _legacy_client(directory)
    legacy.collection.upsert(
        ids=["AAA#0"], documents=["old text"], metadatas=[_passage_meta("AAA", 0, 1)], embeddings=[[0.0, 0.0, 0.0, 1.0]]
    )

    migrate_index(config, **QUIET)

    assert _read(directory, PASSAGES)["AAA#0"][0] == "text of AAA#0"
    assert _read(directory, NAME) == {}


def test_interrupted_move_of_bare_records_finishes_on_the_next_run(index, monkeypatch):
    directory, config = index
    vectors = _seed_chunked(directory, extra_bare=("OLD1", "OLD2", "OLD3"))
    monkeypatch.setattr(index_migration, "_MOVE_BATCH", 1)
    real = index_migration._delete_moved
    calls = {"n": 0}

    def fail_second(collection, ids):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("interrupted")
        real(collection, ids)

    monkeypatch.setattr(index_migration, "_delete_moved", fail_second)
    with pytest.raises(RuntimeError, match="interrupted"):
        migrate_index(config, **QUIET)
    monkeypatch.setattr(index_migration, "_delete_moved", real)

    report = migrate_index(config, **QUIET)

    assert report.needed is True
    assert set(_read(directory, NAME)) == {"OLD1", "OLD2", "OLD3"}
    stored = _read(directory, NAME)
    assert stored["OLD2"][2] == pytest.approx(vectors["OLD2"], abs=1e-5)
    assert not [i for i in _read(directory, PASSAGES) if "#" not in i]
    assert len(_read(directory, PASSAGES)) == 4


@pytest.mark.skipif(sys.platform == "win32", reason="os._exit leaves the files open for this test's reopening")
def test_process_killed_right_after_the_rename_loses_nothing(index, tmp_path):
    """The rename is the one irreversible step, so test it against a real process death."""
    directory, config = index
    # Enough records for ChromaDB to save its vector files, as a real index would.
    count = 1500
    client = _legacy_client(directory)
    ids = [f"K{i:05d}#0" for i in range(count)]
    vectors = np.random.default_rng(5).normal(size=(count, 8)).astype(float)
    for start in range(0, count, 500):
        part = ids[start:start + 500]
        client.upsert_embeddings(
            documents=[f"text of {i}" for i in part],
            metadatas=[_passage_meta(i.split("#")[0], 0, 1) for i in part],
            ids=part,
            embeddings=vectors[start:start + 500].tolist(),
        )
    del client

    script = tmp_path / "killed.py"
    script.write_text(textwrap.dedent(f"""
        import os
        from zotero_mcp import index_migration

        def die(*args, **kwargs):
            os._exit(9)

        index_migration._after_rename = die
        index_migration.migrate_index({config!r}, progress=lambda _m: None)
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True, timeout=120)
    assert result.returncode == 9, result.stderr.decode()[-500:]

    layout = read_index_layout(directory, NAME)
    assert layout["passage_records"] == count

    # A fresh process finishes it, and search still finds the same record.
    finish = tmp_path / "finish.py"
    finish.write_text(textwrap.dedent(f"""
        import json
        import numpy as np
        from zotero_mcp.chroma_client import ChromaClient
        from zotero_mcp.index_migration import migrate_index

        migrate_index({config!r}, progress=lambda _m: None)
        client = ChromaClient(collection_name={NAME!r}, persist_directory={directory!r})
        probe = {vectors[7].tolist()!r}
        hits = client._collection_for_level("passages").query(query_embeddings=[probe], n_results=1)
        print(json.dumps({{"hit": hits["ids"][0][0], "split": client.separate_passages()}}))
    """))
    result = subprocess.run([sys.executable, str(finish)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-500:]
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome == {"hit": "K00007#0", "split": True}
