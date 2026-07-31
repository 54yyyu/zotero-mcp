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


def test_update_metadatas_splits_oversized_batches(client, monkeypatch):
    ids = _seed(client, 12)
    monkeypatch.setattr(client.client, "get_max_batch_size", lambda: 5)

    client.update_metadatas(
        ids, [{"item_key": i, "group_id": 0} for i in ids]
    )

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
# Fake conformance: every public method a suite fake defines must exist on the
# real ChromaClient with a signature that accepts the fake's call shape. This
# is the structural guard against the drift that shipped the dead backfill.
# (Test-only helpers on fakes must be underscore-prefixed to stay exempt.)
# ---------------------------------------------------------------------------

def _fake_chroma_classes():
    import test_fulltext_sync_watermark
    import test_fulltext_web_mode
    import test_semantic_multilibrary
    import test_sync_watermark_per_library

    return [
        test_semantic_multilibrary._FakeChromaClient,
        test_sync_watermark_per_library.FakeChromaClient,
        test_fulltext_web_mode.FakeChromaClient,
        test_fulltext_sync_watermark.FakeChroma,
    ]


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
        call_kwargs = {}
        for p in fake_params:
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            call_kwargs[p.name] = None
        try:
            inspect.signature(real).bind(None, **call_kwargs)
        except TypeError as e:
            pytest.fail(
                f"real ChromaClient.{name} signature rejects the fake's call shape "
                f"({fake_cls.__module__}.{fake_cls.__qualname__}): {e}"
            )
