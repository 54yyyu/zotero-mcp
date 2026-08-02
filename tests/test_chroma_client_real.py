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
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

from zotero_mcp.chroma_client import ChromaClient

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
