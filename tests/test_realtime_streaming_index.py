"""Tests for the realtime streaming indexer.

`update_database()` keeps its embedding worker pool continuously fed across
outer-loop slice boundaries (via `_dispatch_slice`/`_drain_futures`/
`_finalize_slice`) instead of blocking on one slice's embeddings before
preparing the next, whenever `chroma_client.embedding_function` reports
`max_parallel_requests > 1` and `chroma_client` exposes `upsert_embeddings`.
These tests exercise that path end-to-end through `update_database()`, using
fakes that never touch the network or a real ChromaDB instance.
"""

import json
import sys
import threading
import time

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search


class StreamingFakeChromaClient:
    """ChromaClient double exposing upsert_embeddings, so update_database's
    `use_streaming` gate can evaluate True. `upsert_documents` (used by the
    legacy path and by the unmodified end-of-run retry pass) is implemented
    in terms of the same embedding_function + upsert_embeddings, mirroring
    how real ChromaDB's collection.upsert() invokes the embedding function
    internally when no precomputed vectors are passed.
    """

    def __init__(self, embedding_function, preloaded_ids=None):
        self.embedding_function = embedding_function
        self.embedding_max_tokens = 8000
        self._existing = set(preloaded_ids or [])
        self.upserts: list[tuple[list, list, list, list]] = []  # (documents, metadatas, ids, embeddings)
        self.deleted_parents: list[str] = []
        self._lock = threading.Lock()

    def truncate_text(self, text, max_tokens=None):
        return text

    def get_existing_ids(self, ids):
        with self._lock:
            return {i for i in ids if i in self._existing}

    def delete_item_chunks(self, item_key):
        with self._lock:
            self.deleted_parents.append(item_key)

    def upsert_embeddings(self, documents, metadatas, ids, embeddings):
        with self._lock:
            self.upserts.append((list(documents), list(metadatas), list(ids), list(embeddings)))
            self._existing.update(ids)

    def upsert_documents(self, documents, metadatas, ids):
        embeddings = self.embedding_function(documents)
        self.upsert_embeddings(documents, metadatas, ids, embeddings)


class BlockingEmbeddingFunction:
    """Fake embedding_function: __call__ mimics RemoteEmbeddingFunction's
    contract (one vector per input doc). Stalling/failing is keyed by a
    marker substring so it targets a specific slice's sub-batch by content
    regardless of which thread/order actually executes it.
    """

    def __init__(self, max_parallel_requests=4, request_batch_size=2, per_call_delay=0.0):
        self.max_parallel_requests = max_parallel_requests
        self.request_batch_size = request_batch_size
        self.per_call_delay = per_call_delay
        self.call_log: list[list[str]] = []
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.stall_markers: dict[str, threading.Event] = {}
        self.fail_markers: set[str] = set()

    def __call__(self, input):
        with self._lock:
            self.call_log.append(list(input))
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if self.per_call_delay:
                time.sleep(self.per_call_delay)
            for marker, ev in self.stall_markers.items():
                if any(marker in doc for doc in input):
                    ev.wait(timeout=5)
            for marker in self.fail_markers:
                if any(marker in doc for doc in input):
                    raise RuntimeError(f"simulated embedding failure for marker {marker!r}")
            return [[0.0, 0.0] for _ in input]
        finally:
            with self._lock:
                self._active -= 1


def _item(key, title, fulltext=""):
    data = {
        "title": title,
        "itemType": "journalArticle",
        "abstractNote": f"abstract of {title}",
        "creators": [],
    }
    if fulltext:
        data["fulltext"] = fulltext
    return {"key": key, "data": data}


def _build_search(monkeypatch, chroma, items, config_path=None, slice_size=2):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "_realtime_slice_size", lambda ef: slice_size)
    monkeypatch.setenv("ZOTERO_MCP_FORCE_UPDATE", "1")  # bypass the cross-process lock in tests
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma, config_path=config_path)
    monkeypatch.setattr(search, "_get_items_from_source", lambda **kw: items)
    return search


def test_streaming_keeps_pool_fed_across_slice_boundary(monkeypatch):
    """The core proof of the redesign: slice 1's embeddings stall, but later
    slices must still be written to Chroma before the stall is released."""
    items = [_item(f"K{i}", f"Item {i}" if i != 0 else "STALL_ME item") for i in range(6)]
    ef = BlockingEmbeddingFunction(max_parallel_requests=4, request_batch_size=2)
    stall_event = threading.Event()
    ef.stall_markers["STALL_ME"] = stall_event
    chroma = StreamingFakeChromaClient(ef)
    search = _build_search(monkeypatch, chroma, items, slice_size=2)

    result = {}

    def run():
        result["stats"] = search.update_database()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        # Wait until at least one LATER slice (not containing K0) has landed
        # in Chroma, while the stalled slice has not.
        while time.monotonic() < deadline:
            stalled_landed = any("K0" in ids for *_ , ids, _e in chroma.upserts)
            later_landed = any(
                "K0" not in ids for *_ , ids, _e in chroma.upserts
            )
            if later_landed:
                break
            time.sleep(0.01)
        else:
            pytest.fail("no later slice landed while the first slice was stalled")
        assert not stalled_landed, "stalled slice must not have been written yet"
    finally:
        stall_event.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    stats = result["stats"]
    assert stats["processed_items"] == 6
    assert stats["added_items"] == 6
    assert stats["errors"] == 0
    assert {i for _d, _m, ids, _e in chroma.upserts for i in ids} == {f"K{i}" for i in range(6)}


def test_streaming_respects_max_parallel_requests_concurrency(monkeypatch):
    """The persistent pool must never run more embedding calls concurrently
    than max_parallel_requests, regardless of how many slices are prepared
    ahead of it."""
    items = [_item(f"K{i}", f"Item {i}") for i in range(20)]
    ef = BlockingEmbeddingFunction(max_parallel_requests=3, request_batch_size=2, per_call_delay=0.02)
    chroma = StreamingFakeChromaClient(ef)
    search = _build_search(monkeypatch, chroma, items, slice_size=2)

    stats = search.update_database()

    assert stats["added_items"] == 20
    assert ef.max_active <= 3
    assert ef.max_active > 1, "test is meaningless if calls never overlapped"


def test_streaming_added_vs_updated_matches_legacy_path(monkeypatch):
    """Same items, same pre-existing id, run once through the streaming path
    (max_parallel_requests>1) and once through the legacy synchronous path
    (max_parallel_requests<=1) — added/updated/processed counts must match."""
    items = [_item(f"K{i}", f"Item {i}") for i in range(6)]

    ef_streaming = BlockingEmbeddingFunction(max_parallel_requests=4, request_batch_size=2)
    chroma_streaming = StreamingFakeChromaClient(ef_streaming, preloaded_ids={"K2"})
    search_streaming = _build_search(monkeypatch, chroma_streaming, items, slice_size=2)
    stats_streaming = search_streaming.update_database()

    ef_legacy = BlockingEmbeddingFunction(max_parallel_requests=1, request_batch_size=2)
    chroma_legacy = StreamingFakeChromaClient(ef_legacy, preloaded_ids={"K2"})
    search_legacy = _build_search(monkeypatch, chroma_legacy, items, slice_size=2)
    stats_legacy = search_legacy.update_database()

    for key in ("processed_items", "added_items", "updated_items", "skipped_items", "errors"):
        assert stats_streaming[key] == stats_legacy[key], f"{key} differs: {stats_streaming[key]} != {stats_legacy[key]}"
    assert stats_streaming["updated_items"] == 1
    assert stats_streaming["added_items"] == 5


def test_streaming_defers_whole_slice_on_embedding_failure(monkeypatch):
    """One sub-batch of one slice fails permanently: both its records defer
    to the end-of-run retry, which (matching the pre-existing, unmodified
    retry path) retries one doc at a time — so K1 (no failure marker) is
    recovered individually even though it shared a failed sub-batch with K0
    (the permanently-failing one) during the main streaming pass. Other
    slices must be unaffected throughout."""
    items = [_item(f"K{i}", f"Item {i}" if i != 0 else "FAIL_ME item") for i in range(6)]
    ef = BlockingEmbeddingFunction(max_parallel_requests=4, request_batch_size=2)
    ef.fail_markers.add("FAIL_ME")
    chroma = StreamingFakeChromaClient(ef)
    search = _build_search(monkeypatch, chroma, items, slice_size=2)

    stats = search.update_database()

    assert stats["errors"] == 1  # only K0 (contains FAIL_ME) survives retry as an error
    assert stats["recovered_items"] == 1  # K1 recovered on its individual retry
    assert stats["added_items"] == 4  # the other two slices succeeded in the main pass
    landed_ids = {i for _d, _m, ids, _e in chroma.upserts for i in ids}
    assert landed_ids == {"K1", "K2", "K3", "K4", "K5"}  # K0 never lands


def test_dispatch_slice_respects_in_flight_cap_within_one_slice():
    """Regression test for a real bug caught on a live run: the in-flight
    cap must be checked before EVERY sub-batch submission, not just once per
    slice. A single heavily-chunked slice can hold far more sub-batches than
    the cap on its own (a real 200-item slice under real chunking config
    produced 143 in-flight sub-batches against an intended cap of 32,
    because the cap was only checked between slices)."""
    search = semantic_search.ZoteroSemanticSearch.__new__(semantic_search.ZoteroSemanticSearch)

    class FakeFuture:
        def done(self):
            return False

    class FakePool:
        def __init__(self):
            self.n_submitted = 0

        def submit(self, fn, docs):
            self.n_submitted += 1
            return FakeFuture()

    n_docs = 50
    slice_work = {
        "documents": [f"doc{i}" for i in range(n_docs)],
        "metadatas": [{} for _ in range(n_docs)],
        "ids": [f"id{i}" for i in range(n_docs)],
        "item_keys_order": [f"id{i}" for i in range(n_docs)],
        "existing_item_keys": set(),
        "prep_stats": {"processed": n_docs, "skipped": 0, "errors": 0},
    }
    in_flight: dict = {}
    slices: dict = {}
    max_in_flight = 5
    peak = 0

    def drain_blocking():
        nonlocal peak
        peak = max(peak, len(in_flight))
        # Simulate the oldest in-flight future completing, as real
        # _drain_futures(block=True) would, so dispatch can keep proceeding.
        oldest = next(iter(in_flight))
        del in_flight[oldest]

    pool = FakePool()
    search._dispatch_slice(
        slice_work, lambda docs: [[0.0] for _ in docs], 1, pool,
        in_flight, slices, 0, max_in_flight, drain_blocking,
    )
    peak = max(peak, len(in_flight))

    assert pool.n_submitted == n_docs  # every sub-batch still gets submitted eventually
    assert peak <= max_in_flight
    assert peak > 1, "test is meaningless if the cap was never actually reached"


def test_streaming_chunked_item_written_atomically(monkeypatch, tmp_path):
    """With chunking on and request_batch_size smaller than one item's chunk
    count, its chunks span multiple sub-batches/futures. delete_item_chunks
    must run before any of that item's chunks are embedded, and all its
    chunks must land in exactly one upsert_embeddings call."""
    config = {"semantic_search": {"chunking": {"enabled": True, "chunk_size": 200, "overlap": 0}}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    long_fulltext = "Sentence about entrepreneurship research. " * 200  # many chunk_size=200 passages
    items = [_item("BOOK1", "A Long Paper", fulltext=long_fulltext)]

    ef = BlockingEmbeddingFunction(max_parallel_requests=4, request_batch_size=3)
    chroma = StreamingFakeChromaClient(ef)
    search = _build_search(monkeypatch, chroma, items, config_path=str(config_path), slice_size=10)

    stats = search.update_database()

    assert stats["added_items"] == 1
    assert chroma.deleted_parents == ["BOOK1"]
    assert len(ef.call_log) > 1, "test is meaningless with only one sub-batch"
    assert len(chroma.upserts) == 1
    documents, metadatas, ids, embeddings = chroma.upserts[0]
    assert len(ids) > ef.request_batch_size, "item must have spanned multiple sub-batches"
    assert all(i.startswith("BOOK1#") for i in ids)
    assert len(embeddings) == len(ids)
