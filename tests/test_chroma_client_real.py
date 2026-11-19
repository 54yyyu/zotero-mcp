"""Regression tests against the REAL ChromaClient.

The group_id backfill shipped calling ``ChromaClient.iter_metadatas()`` and
``update_metadatas()`` — methods that existed only on this suite's fakes,
never on the real class — so every production backfill died with
``AttributeError`` while CI stayed green. A fake cannot catch that class of
bug, so these tests run against the real ``ChromaClient`` on a temporary
persistent collection, and a conformance check keeps the suite's fakes from
drifting from the real API again.

Offline-safe: nothing here ever invokes an embedding function. Documents are
seeded with tiny precomputed vectors via ``upsert_embeddings``, and the
default embedding function only downloads its model on first ``__call__``,
which is never reached. This is the suite's first construction of a real
``ChromaClient``; if a chromadb upgrade breaks ``__init__`` itself, that is
an environment failure worth seeing, not a test bug.
"""

import importlib.util
import inspect
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

from zotero_mcp.chroma_client import (
    ChromaClient,
    create_chroma_client,
    read_collection_status,
)

GROUP_ID = 6015547


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    monkeypatch.delenv("ZOTERO_LOCAL", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_TYPE", raising=False)


@pytest.fixture()
def client(tmp_path):
    return ChromaClient(
        collection_name="real_client_test",
        persist_directory=str(tmp_path / "chroma"),
        embedding_model="default",
    )


def _seed(client, n, tagged_group_id=None, prefix="K"):
    """Insert n docs with precomputed vectors; no embedding function runs."""
    ids = [f"{prefix}{i:05d}" for i in range(n)]
    metadatas = []
    for i, doc_id in enumerate(ids):
        meta = {"item_key": doc_id, "title": f"Doc {i}"}
        if tagged_group_id is not None:
            meta["group_id"] = tagged_group_id
        metadatas.append(meta)
    client.upsert_embeddings(
        documents=[f"document body {i}" for i in range(n)],
        metadatas=metadatas,
        ids=ids,
        embeddings=[[float(i % 7), 1.0, 2.0, 3.0] for i in range(n)],
    )
    return ids


# ---------------------------------------------------------------------------
# ids_in_group: the deletion pass's by-id answer must equal the filtered listing
# ---------------------------------------------------------------------------

def test_ids_in_group_agrees_with_the_filtered_listing(client):
    personal = _seed(client, 6, tagged_group_id=0, prefix="P")
    group = _seed(client, 5, tagged_group_id=GROUP_ID, prefix="G")
    untagged = _seed(client, 4, prefix="U")
    # Passages, in their own collection: two libraries share the item key X.
    client.upsert_embeddings(
        documents=["a", "b", "c"],
        metadatas=[
            {"item_key": "X", "parent_item_key": "X", "chunk_index": 0, "group_id": 0},
            {"item_key": "X", "parent_item_key": "X", "chunk_index": 1, "group_id": GROUP_ID},
            {"item_key": "X", "parent_item_key": "X", "chunk_index": 2},
        ],
        ids=["X#0", "X#1", "X#2"],
        embeddings=[[1.0, 0.0, 0.0, 0.0]] * 3,
    )
    everything = [*personal, *group, *untagged, "X#0", "X#1", "X#2", "NOT_STORED"]

    for group_id in (0, GROUP_ID, 424242):
        assert client.ids_in_group(everything, group_id) == client.get_all_ids(
            where={"group_id": group_id}
        )
    assert client.ids_in_group(everything, 0) == {*personal, "X#0"}
    assert client.ids_in_group([], 0) == set()
    # Asked about some records only, it answers about those only.
    assert client.ids_in_group([personal[0], group[0], "X#1"], 0) == {personal[0]}


def test_ids_in_group_raises_on_a_read_error(client, monkeypatch):
    _seed(client, 2, tagged_group_id=0)

    def broken(*args, **kwargs):
        raise RuntimeError("database disk image is malformed")

    monkeypatch.setattr(client.collection, "get", broken)
    with pytest.raises(RuntimeError):
        client.ids_in_group(["K00000"], 0)


# ---------------------------------------------------------------------------
# item_keys_by_level: which items have an article record, and all their passages
# ---------------------------------------------------------------------------

def _passages(client, key, indexes, n_chunks):
    metadatas = []
    for index in indexes:
        meta = {"item_key": key, "parent_item_key": key, "chunk_index": index}
        if n_chunks is not None:
            meta["n_chunks"] = n_chunks
        metadatas.append(meta)
    client.upsert_embeddings(
        documents=[f"{key} passage {index}" for index in indexes],
        metadatas=metadatas,
        ids=[f"{key}#{index}" for index in indexes],
        embeddings=[[1.0, 0.0, 0.0, float(index)] for index in indexes],
    )


def test_item_keys_by_level_counts_only_items_with_all_their_passages(client):
    articles = _seed(client, 3, tagged_group_id=0, prefix="A")
    _passages(client, "WHOLE", range(12), 12)        # past #9: "#10" sorts before "#2"
    _passages(client, "SHORT", [0, 1], 3)            # one passage failed to write
    _passages(client, "HEADLESS", [1, 2], 3)         # its first passage is missing
    _passages(client, "HEADLESS_WHOLE", [1, 2], 2)   # counted by number, as before
    _passages(client, "LEGACY", [0, 1], None)        # written before n_chunks existed

    found_articles, found_passages = client.item_keys_by_level()

    assert found_articles == set(articles)
    assert found_passages == {"WHOLE", "HEADLESS_WHOLE", "LEGACY"}


def test_item_keys_by_level_is_reused_until_the_index_changes(client, monkeypatch):
    from zotero_mcp import chroma_client as chroma_client_module

    _seed(client, 2, tagged_group_id=0, prefix="A")
    _passages(client, "ONE", [0, 1], 2)
    first = client.item_keys_by_level()

    opened = []
    real_connect = chroma_client_module.sqlite3.connect

    def counting_connect(*args, **kwargs):
        opened.append(args)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(chroma_client_module.sqlite3, "connect", counting_connect)
    again = client.item_keys_by_level()
    assert again == first and opened == []
    # A caller changing what it was given must not change the next answer.
    again[0].clear()
    again[1].clear()
    assert client.item_keys_by_level() == first

    _passages(client, "TWO", [0], 1)
    assert client.item_keys_by_level() == (first[0], first[1] | {"TWO"})
    assert opened


# ---------------------------------------------------------------------------
# get_documents_metadata: the single lookup, for many items at once
# ---------------------------------------------------------------------------

def test_get_documents_metadata_answers_as_the_single_lookup_does(client):
    articles = _seed(client, 4, tagged_group_id=0, prefix="A")
    _passages(client, "PASSAGES_ONLY", [0, 1], 2)
    _passages(client, articles[0], [0, 1], 2)   # an item with both levels
    _passages(client, "NO_FIRST", [1], 2)        # no article, no first passage
    keys = [*articles, "PASSAGES_ONLY", "NO_FIRST", "NOT_INDEXED"]

    many = client.get_documents_metadata(keys)

    assert many == {
        key: meta for key in keys if (meta := client.get_document_metadata(key)) is not None
    }
    assert set(many) == {*articles, "PASSAGES_ONLY"}
    assert "parent_item_key" not in many[articles[0]]  # the item's own record wins
    assert many["PASSAGES_ONLY"]["chunk_index"] == 0
    assert client.get_documents_metadata([]) == {}


# ---------------------------------------------------------------------------
# prune_items_chunks: prune_item_chunks for a batch, deleting the same records
# ---------------------------------------------------------------------------

def _prune_scenario(tmp_path, name, *, route_passages=True):
    client = ChromaClient(
        collection_name="real_client_test",
        persist_directory=str(tmp_path / name),
        embedding_model="default",
    )
    client.route_passages = route_passages
    _seed(client, 3, tagged_group_id=0, prefix="A")   # article records, or bare legacy docs
    _passages(client, "A00000", range(6), 6)          # shrinks to 2
    _passages(client, "A00001", range(3), 3)          # same length: nothing to prune
    _passages(client, "HOLES", [0, 1, 5, 6], 7)       # stale tail does not start at the new count
    _passages(client, "GROWN", range(2), 2)           # grew: nothing past the new count
    _passages(client, "UNTOUCHED", range(4), 4)       # not in this write
    return client


PRUNE = {"A00000": 2, "A00001": 3, "HOLES": 3, "GROWN": 5, "NEW_ITEM": 4}


@pytest.mark.parametrize("route_passages", [True, False], ids=["separate", "unmigrated"])
def test_prune_items_chunks_deletes_what_the_per_item_prune_deletes(tmp_path, route_passages):
    one_by_one = _prune_scenario(tmp_path, "one", route_passages=route_passages)
    for key, keep in PRUNE.items():
        one_by_one.prune_item_chunks(key, keep)
    batched = _prune_scenario(tmp_path, "many", route_passages=route_passages)
    batched.prune_items_chunks(PRUNE)

    left = batched.get_all_ids()
    assert left == one_by_one.get_all_ids()
    assert {"A00000#0", "A00000#1", "HOLES#0", "HOLES#1", "UNTOUCHED#3", "GROWN#1"} <= left
    assert not {"A00000#2", "A00000#5", "HOLES#5", "HOLES#6"} & left
    # Where passages share the article collection, the bare record of an item
    # now stored as passages goes too; in its own collection it is the article.
    assert ("A00000" in left) is route_passages
    batched.prune_items_chunks({})


# ---------------------------------------------------------------------------
# iter_metadatas
# ---------------------------------------------------------------------------

def test_iter_metadatas_streams_every_document_exactly_once(client):
    ids = _seed(client, 1200)

    seen = []
    for batch_ids, batch_metas in client.iter_metadatas(batch_size=500):
        assert len(batch_ids) <= 500
        assert len(batch_ids) == len(batch_metas)
        seen.extend(batch_ids)

    assert len(seen) == len(set(seen)), "a document was yielded twice"
    assert set(seen) == set(ids), "a document was skipped"


def test_iter_metadatas_yields_matching_metadata(client):
    _seed(client, 3)

    for batch_ids, batch_metas in client.iter_metadatas(batch_size=500):
        for doc_id, meta in zip(batch_ids, batch_metas):
            assert meta["item_key"] == doc_id


def test_iter_metadatas_survives_updates_between_batches(client):
    """The backfill updates each yielded batch before requesting the next;
    that must never cause skips or duplicates."""
    ids = _seed(client, 1200)

    seen = []
    for batch_ids, batch_metas in client.iter_metadatas(batch_size=500):
        seen.extend(batch_ids)
        client.update_metadatas(
            batch_ids, [dict(m, group_id=0) for m in batch_metas]
        )

    assert set(seen) == set(ids)
    assert len(seen) == len(set(seen))
    tagged = client.get_all_ids(where={"group_id": 0})
    assert tagged == set(ids)


def test_iter_metadatas_empty_collection_yields_nothing(client):
    assert list(client.iter_metadatas()) == []


def test_iter_metadatas_propagates_backend_errors(client, monkeypatch):
    """A backend failure must raise, never masquerade as an empty
    collection: the backfill's one-time schema gate closes permanently on
    'success', so a swallowed error here would silently disable the
    migration with zero documents tagged."""
    _seed(client, 3)

    def boom(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(client.collection, "get", boom)

    with pytest.raises(RuntimeError, match="database is locked"):
        list(client.iter_metadatas())


# ---------------------------------------------------------------------------
# update_metadatas
# ---------------------------------------------------------------------------

def test_update_metadatas_round_trip(client):
    ids = _seed(client, 2)

    client.update_metadatas(
        ids, [{"item_key": ids[0], "group_id": GROUP_ID}, {"item_key": ids[1], "group_id": 0}]
    )

    result = client.collection.get(ids=ids, include=["metadatas"])
    metas = dict(zip(result["ids"], result["metadatas"]))
    assert metas[ids[0]]["group_id"] == GROUP_ID
    assert metas[ids[1]]["group_id"] == 0
    # chromadb 1.5.x merges metadata on update; other versions replace.
    # Either way the keys we sent must be present afterwards.
    assert metas[ids[0]]["item_key"] == ids[0]
    # Pin the installed backend's merge semantics: a partial update must not
    # drop keys it did not send (callers still pass full dicts so the
    # backfill stays correct on replace-semantics versions too).
    assert metas[ids[0]]["title"] == "Doc 0"


def test_iter_metadatas_rejects_invalid_batch_size(client):
    with pytest.raises(ValueError):
        list(client.iter_metadatas(batch_size=0))


def test_update_metadatas_splits_oversized_batches(client, monkeypatch):
    ids = _seed(client, 12)
    monkeypatch.setattr(client.client, "get_max_batch_size", lambda: 5)
    # Spy on the underlying update so the SPLITTING is proven, not just the
    # end state (the real backend accepts 12 at once, so an end-state-only
    # assertion could never fail if the loop were deleted).
    batch_sizes = []
    real_update = client.collection.update

    def spy(**kwargs):
        batch_sizes.append(len(kwargs.get("ids") or []))
        return real_update(**kwargs)

    monkeypatch.setattr(client.collection, "update", spy)

    client.update_metadatas(
        ids, [{"item_key": i, "group_id": 0} for i in ids]
    )

    assert batch_sizes == [5, 5, 2]
    assert client.get_all_ids(where={"group_id": 0}) == set(ids)


def test_update_metadatas_empty_input_is_noop(client):
    client.update_metadatas([], [])  # must not raise


def test_update_metadatas_does_not_touch_documents_or_embeddings(client):
    ids = _seed(client, 1)

    client.update_metadatas(ids, [{"item_key": ids[0], "group_id": 0}])

    result = client.collection.get(ids=ids, include=["documents", "embeddings"])
    assert result["documents"][0] == "document body 0"
    assert list(result["embeddings"][0]) == [0.0, 1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# get_all_ids(where=...)
# ---------------------------------------------------------------------------

def test_get_all_ids_without_filter_returns_everything(client):
    personal = _seed(client, 2, tagged_group_id=0, prefix="P")
    group = _seed(client, 2, tagged_group_id=GROUP_ID, prefix="G")
    untagged = _seed(client, 2, prefix="U")

    assert client.get_all_ids() == set(personal) | set(group) | set(untagged)


def test_get_all_ids_where_scopes_to_one_library(client):
    personal = _seed(client, 2, tagged_group_id=0, prefix="P")
    group = _seed(client, 2, tagged_group_id=GROUP_ID, prefix="G")
    _seed(client, 2, prefix="U")

    assert client.get_all_ids(where={"group_id": GROUP_ID}) == set(group)
    # group_id 0 (personal) is falsy but a valid filter value...
    assert client.get_all_ids(where={"group_id": 0}) == set(personal)


def test_get_all_ids_where_never_matches_untagged_docs(client):
    """Docs with no group_id key are excluded by any group_id filter — the
    invariant that makes unattributed docs structurally undeletable."""
    _seed(client, 3, prefix="U")

    assert client.get_all_ids(where={"group_id": 0}) == set()
    assert client.get_all_ids(where={"group_id": GROUP_ID}) == set()


# ---------------------------------------------------------------------------
# End-to-end against the real client: the exact production composition (real
# iterator + real metadata updates + backfill / deletion logic) that the
# fake-only API drift hid.
# ---------------------------------------------------------------------------

def test_backfill_end_to_end_against_real_chroma(client, monkeypatch):
    from zotero_mcp import semantic_search

    tagged = _seed(client, 10, tagged_group_id=GROUP_ID, prefix="T")
    untagged = _seed(client, 1200, prefix="N")
    evidenced = set(untagged[:700])

    class _Zot:
        def item_versions(self, **kw):
            return {k: 3 for k in evidenced}

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: _Zot())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    search = semantic_search.ZoteroSemanticSearch(chroma_client=client)

    stats = search._backfill_group_ids()

    assert stats == {"scanned": 1210, "migrated": 700, "unattributed": 500}
    assert client.get_all_ids(where={"group_id": 0}) == evidenced
    assert client.get_all_ids(where={"group_id": GROUP_ID}) == set(tagged)

    # Idempotent second pass: nothing newly migrated, nothing guessed.
    stats2 = search._backfill_group_ids()
    assert stats2 == {"scanned": 1210, "migrated": 0, "unattributed": 500}


def test_scoped_deletion_end_to_end_against_real_chroma(client, monkeypatch, tmp_path):
    import json

    from zotero_mcp import semantic_search

    client.upsert_embeddings(
        documents=["live", "dead", "group"],
        metadatas=[
            {"item_key": "PERS_LIVE", "group_id": 0},
            {"item_key": "PERS_DEAD", "group_id": 0},
            {"item_key": "GRP_LIVE", "group_id": GROUP_ID},
        ],
        ids=["PERS_LIVE", "PERS_DEAD", "GRP_LIVE"],
        embeddings=[[1.0, 2.0, 3.0, 4.0]] * 3,
    )
    _seed(client, 1, prefix="UNTAGGED")

    class _Zot:
        def item_versions(self, since=None, **kw):
            return {} if since is not None else {"PERS_LIVE": 9}

        def last_modified_version(self, **kw):
            return 9

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "semantic_search": {
            "embedding_model": "default",
            "update_config": {"auto_update": False, "update_frequency": "manual"},
            "include_fulltext": False,
            "index_schema_version": semantic_search._INDEX_SCHEMA_VERSION,
            "last_sync_versions": {"0": 5},
        }
    }))
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: _Zot())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    search = semantic_search.ZoteroSemanticSearch(
        chroma_client=client, config_path=str(config_path)
    )

    stats = search.update_database()

    assert stats["deleted_items"] == 1
    remaining = client.get_all_ids()
    assert "PERS_DEAD" not in remaining
    assert {"PERS_LIVE", "GRP_LIVE", "UNTAGGED00000"} <= remaining


# ---------------------------------------------------------------------------
# Fake conformance: every public method a suite fake defines must exist on the
# real ChromaClient with a signature that accepts the fake's call shape. This
# is the structural guard against the drift that shipped the dead backfill.
# (Test-only helpers on fakes must be underscore-prefixed to stay exempt.)
#
# Fakes are DISCOVERED, not enumerated: any module-level class in tests/ whose
# name contains "chroma" (other than this file's imports) is pinned, so a new
# fake is covered the day it is written. Modules that fail to import (missing
# optional deps, module-level skips) are skipped; a sanity floor asserts
# discovery still sees the core fakes.
# ---------------------------------------------------------------------------

def _fake_chroma_classes():
    import importlib
    import pathlib

    classes = []
    for path in sorted(pathlib.Path(__file__).parent.glob("test_*.py")):
        if path.stem == pathlib.Path(__file__).stem:
            continue
        try:
            mod = importlib.import_module(path.stem)
        except BaseException:  # module-level pytest.skip or missing optional dep
            continue
        for obj in vars(mod).values():
            if (
                isinstance(obj, type)
                and obj.__module__ == mod.__name__
                and "chroma" in obj.__name__.lower()
                and obj is not ChromaClient
            ):
                classes.append(obj)
    return classes


def test_fake_discovery_sees_the_core_fakes():
    names = {f"{c.__module__}.{c.__qualname__}" for c in _fake_chroma_classes()}
    for expected in (
        "test_semantic_multilibrary._FakeChromaClient",
        "test_sync_watermark_per_library.FakeChromaClient",
        "test_fulltext_web_mode.FakeChromaClient",
        "test_fulltext_sync_watermark.FakeChroma",
        "test_library_scoped_deletion.RecordingChroma",
    ):
        assert expected in names


@pytest.mark.parametrize("fake_cls", _fake_chroma_classes(),
                         ids=lambda c: f"{c.__module__}.{c.__qualname__}")
def test_fakes_conform_to_real_chroma_client_api(fake_cls):
    for name, member in vars(fake_cls).items():
        if name.startswith("_") or not inspect.isfunction(member):
            continue
        real = getattr(ChromaClient, name, None)
        assert real is not None, (
            f"{fake_cls.__module__}.{fake_cls.__qualname__}.{name} does not exist on the "
            "real ChromaClient — a fake-only method is exactly how the dead "
            "group_id backfill shipped"
        )
        fake_params = list(inspect.signature(member).parameters.values())[1:]  # drop self
        pos_args = []
        kw_args = {}
        has_var = False
        for p in fake_params:
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                has_var = True
            elif p.kind == inspect.Parameter.KEYWORD_ONLY:
                kw_args[p.name] = None
            else:
                pos_args.append(None)
        if has_var and not pos_args and not kw_args:
            # A pure *args/**kwargs fake accepts any call the real accepts;
            # method existence is the only checkable contract.
            continue
        try:
            # Positional binding checks arity without requiring the fake's
            # positional parameter NAMES to match (production calls these
            # methods positionally); keyword-only params bind by name.
            inspect.signature(real).bind(None, *pos_args, **kw_args)
        except TypeError as e:
            pytest.fail(
                f"real ChromaClient.{name} signature rejects the fake's call shape "
                f"({fake_cls.__module__}.{fake_cls.__qualname__}): {e}"
            )


# ---------------------------------------------------------------------------
# semantic_search.persist_directory (#617)
# ---------------------------------------------------------------------------

def test_configured_persist_directory_is_written_and_read(monkeypatch, tmp_path):
    """The status reader is handed only the config path, so it has to find a
    moved index on its own, or it reports the index as empty."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.delenv("ZOTERO_EMBEDDING_MODEL", raising=False)
    index_dir = tmp_path / "moved" / "chroma_db"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"semantic_search": {"persist_directory": str(index_dir)}})
    )

    _seed(create_chroma_client(str(config_path)), 3)

    status = read_collection_status(str(config_path))
    assert status["persist_directory"] == str(index_dir)
    assert status["count"] == 3
