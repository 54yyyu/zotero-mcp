"""Streaming index path: overlapped prepare / embed / commit.

`update_database` runs a producer thread, `max_parallel_requests` embedding
workers and a single committer whenever the embedding function is configured
for concurrent requests. These tests pin the two things that matter about it:
it is invisible (same ids, metadata, stats and eviction as the synchronous
path) and it is genuinely concurrent.

Everything here is offline and deterministic. The fake embedding function
records which thread served each request and can be told to fail specific
request numbers, so the failure tests never depend on timing.
"""

import sys
import threading

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search


class FakeEmbeddingFunction:
    """Stand-in for a network-bound embedding function.

    Vectors are derived from the document text, so the same input always
    embeds to the same output and streaming/sequential runs are comparable
    element by element.
    """

    def __init__(self, max_parallel_requests=4, request_batch_size=5, delay=0.02,
                 fail_on_calls=frozenset()):
        self.max_parallel_requests = max_parallel_requests
        self.request_batch_size = request_batch_size
        self._delay = delay
        self._fail_on = set(fail_on_calls)
        self._lock = threading.Lock()
        self._calls = 0
        self.intervals = []  # (thread_name, start, end)

    def __call__(self, documents):
        import time

        with self._lock:
            self._calls += 1
            call_number = self._calls
        start = time.monotonic()
        if self._delay:
            time.sleep(self._delay)
        end = time.monotonic()
        with self._lock:
            self.intervals.append((threading.current_thread().name, start, end))
        if call_number in self._fail_on:
            raise RuntimeError(f"simulated embedding failure on call {call_number}")
        return [[float(len(doc))] for doc in documents]


class FakeStreamingChromaClient:
    """A ChromaDB double that implements `upsert_embeddings`.

    Deliberately a new class rather than an extension of the doubles in other
    test modules: those implement only `upsert_documents`, which is exactly
    what keeps them on the synchronous path, and they must stay that way.
    """

    def __init__(self, embedding_function, existing_keys=frozenset()):
        self.embedding_function = embedding_function
        self._existing = set(existing_keys)
        self.embedding_max_tokens = 8000
        self.embedding_batches = []  # (documents, metadatas, ids, embeddings)
        self.document_batches = []  # (documents, metadatas, ids)
        self.deleted_item_keys = []
        self._lock = threading.Lock()

    def truncate_text(self, text, max_tokens=None):
        return text

    def get_existing_ids(self, ids):
        return {i for i in ids if i.split("#", 1)[0] in self._existing}

    def delete_item_chunks(self, item_key, group_id=None):
        self.deleted_item_keys.append(item_key)

    def upsert_documents(self, documents, metadatas, ids):
        self.document_batches.append((list(documents), list(metadatas), list(ids)))

    def upsert_embeddings(self, documents, metadatas, ids, embeddings):
        with self._lock:
            self.embedding_batches.append(
                (list(documents), list(metadatas), list(ids), list(embeddings))
            )


# Assertion helpers live at module level, not on the fake: a fake-only method
# is how a dead code path once shipped unnoticed, and tests/test_chroma_client_real.py
# now enforces that every double's surface exists on the real ChromaClient.


def committed_triples(chroma):
    """(id, document, first vector element) for everything committed.

    A set, because the streaming committer writes in completion order rather
    than input order and nothing downstream depends on that order.
    """
    triples = set()
    for documents, _metas, ids, embeddings in chroma.embedding_batches:
        for doc, doc_id, vector in zip(documents, ids, embeddings):
            triples.add((doc_id, doc, vector[0]))
    for documents, _metas, ids in chroma.document_batches:
        for doc, doc_id in zip(documents, ids):
            triples.add((doc_id, doc, float(len(doc))))
    return triples


def committed_ids(chroma):
    """Every id handed to ChromaDB, by either commit path."""
    ids = []
    for _docs, _metas, batch_ids, _embeddings in chroma.embedding_batches:
        ids.extend(batch_ids)
    for _docs, _metas, batch_ids in chroma.document_batches:
        ids.extend(batch_ids)
    return ids


def _items(count, prefix="ITEM"):
    return [
        {
            "key": f"{prefix}{i:04d}",
            "data": {
                "title": f"Title {i}",
                "itemType": "journalArticle",
                "abstractNote": f"Abstract number {i} " * (i % 3 + 1),
                "creators": [],
            },
        }
        for i in range(count)
    ]


def _make_search(monkeypatch, chroma, chunking=None):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma)
    monkeypatch.setattr(search, "_save_update_config", lambda *a, **k: None)
    if chunking is not None:
        search._chunking_config = chunking
    return search


def _run_update(monkeypatch, search, items):
    monkeypatch.setattr(search, "_get_items_from_source", lambda **kwargs: list(items))
    return search.update_database()


# -- the gate ----------------------------------------------------------


def test_falls_back_when_client_has_no_upsert_embeddings(monkeypatch):
    """A double implementing only upsert_documents stays on the sync path.

    The minimal ChromaDB doubles across this suite are exactly that shape, so
    the gate must never drag them into the streaming path.
    """

    class DocumentsOnlyClient:
        """Shaped like the minimal doubles elsewhere in the suite."""

        def __init__(self, embedding_function):
            self.embedding_function = embedding_function
            self.embedding_max_tokens = 8000
            self.document_batches = []

        def truncate_text(self, text, max_tokens=None):
            return text

        def get_existing_ids(self, ids):
            return set()

        def upsert_documents(self, documents, metadatas, ids):
            self.document_batches.append((list(documents), list(metadatas), list(ids)))

    chroma = DocumentsOnlyClient(FakeEmbeddingFunction(max_parallel_requests=4))
    search = _make_search(monkeypatch, chroma)

    stats = _run_update(monkeypatch, search, _items(6))

    assert not hasattr(chroma, "upsert_embeddings")
    assert chroma.document_batches, "synchronous path should have upserted documents"
    assert stats["added_items"] == 6


def test_falls_back_when_max_parallel_requests_is_one(monkeypatch):
    """max_parallel_requests == 1 keeps the historical sequential path."""
    embedder = FakeEmbeddingFunction(max_parallel_requests=1)
    chroma = FakeStreamingChromaClient(embedder)
    search = _make_search(monkeypatch, chroma)

    _run_update(monkeypatch, search, _items(6))

    assert chroma.embedding_batches == []
    assert chroma.document_batches, "should have gone through upsert_documents"


def test_falls_back_when_client_has_no_embedding_function(monkeypatch):
    """No embedding_function attribute means no streaming."""
    chroma = FakeStreamingChromaClient(embedding_function=None)
    search = _make_search(monkeypatch, chroma)

    _run_update(monkeypatch, search, _items(4))

    assert chroma.embedding_batches == []
    assert chroma.document_batches


# -- equivalence with the synchronous path -----------------------------


def test_streaming_matches_sequential_results_and_stats(monkeypatch):
    """Streaming and sequential runs agree on ids, documents, vectors, stats.

    The whole feature is meant to be invisible apart from being faster, so
    this is the assertion that matters most.
    """
    items = _items(23)
    existing = {"ITEM0000", "ITEM0005", "ITEM0011"}

    sequential_chroma = FakeStreamingChromaClient(
        FakeEmbeddingFunction(max_parallel_requests=1, delay=0), existing_keys=existing
    )
    sequential_search = _make_search(monkeypatch, sequential_chroma)
    sequential_stats = _run_update(monkeypatch, sequential_search, items)

    streaming_chroma = FakeStreamingChromaClient(
        FakeEmbeddingFunction(max_parallel_requests=4, delay=0), existing_keys=existing
    )
    streaming_search = _make_search(monkeypatch, streaming_chroma)
    streaming_stats = _run_update(monkeypatch, streaming_search, items)

    assert streaming_chroma.embedding_batches, "streaming path did not run"

    for key in ("processed_items", "added_items", "updated_items", "skipped_items", "errors"):
        assert streaming_stats[key] == sequential_stats[key], key

    assert streaming_stats["added_items"] == 20
    assert streaming_stats["updated_items"] == 3
    assert committed_triples(streaming_chroma) == committed_triples(sequential_chroma)
    assert sorted(committed_ids(streaming_chroma)) == sorted(
        committed_ids(sequential_chroma)
    )


def test_streaming_commits_every_item_exactly_once(monkeypatch):
    """No item is dropped or written twice across commit boundaries."""
    items = _items(40)
    chroma = FakeStreamingChromaClient(
        FakeEmbeddingFunction(max_parallel_requests=3, request_batch_size=4, delay=0)
    )
    search = _make_search(monkeypatch, chroma)

    _run_update(monkeypatch, search, items)

    committed = committed_ids(chroma)
    assert sorted(committed) == sorted(item["key"] for item in items)
    assert len(committed) == len(set(committed))


# -- concurrency -------------------------------------------------------


def test_streaming_runs_requests_concurrently(monkeypatch):
    """More than one worker thread embeds, and their requests overlap in time.

    Asserted as an interval overlap rather than a wall-clock threshold, so a
    slow or contended CI runner cannot make it flaky.
    """
    embedder = FakeEmbeddingFunction(
        max_parallel_requests=4, request_batch_size=2, delay=0.05
    )
    chroma = FakeStreamingChromaClient(embedder)
    search = _make_search(monkeypatch, chroma)

    _run_update(monkeypatch, search, _items(24))

    thread_names = {name for name, _start, _end in embedder.intervals}
    assert len(thread_names) > 1, f"expected several worker threads, saw {thread_names}"

    overlapped = any(
        max(a[1], b[1]) < min(a[2], b[2])
        for i, a in enumerate(embedder.intervals)
        for b in embedder.intervals[i + 1 :]
    )
    assert overlapped, "no two embedding requests were ever in flight together"


# -- failure handling --------------------------------------------------


def test_worker_failure_is_retried_and_does_not_stop_the_run(monkeypatch):
    """One failed request routes to the retry pass; the others still commit."""
    embedder = FakeEmbeddingFunction(
        max_parallel_requests=2, request_batch_size=4, delay=0, fail_on_calls={2}
    )
    chroma = FakeStreamingChromaClient(embedder)
    search = _make_search(monkeypatch, chroma)

    retried = []
    original_upsert = chroma.upsert_documents

    def recording_upsert(documents, metadatas, ids):
        retried.append(list(ids))
        original_upsert(documents, metadatas, ids)

    monkeypatch.setattr(chroma, "upsert_documents", recording_upsert)

    stats = _run_update(monkeypatch, search, _items(20))

    assert chroma.embedding_batches, "the surviving requests should still have committed"
    # The failed sub-batch went through the shared end-of-run retry pass,
    # which re-embeds one document at a time via upsert_documents.
    assert retried, "failed documents were not handed to the retry pass"
    assert stats["processed_items"] == 20


def test_commit_failure_routes_documents_to_the_retry_pass(monkeypatch):
    """An upsert_embeddings failure is recoverable, not fatal."""
    embedder = FakeEmbeddingFunction(max_parallel_requests=2, request_batch_size=4, delay=0)
    chroma = FakeStreamingChromaClient(embedder)

    def failing_upsert_embeddings(documents, metadatas, ids, embeddings):
        raise RuntimeError("chroma is unhappy")

    search = _make_search(monkeypatch, chroma)
    monkeypatch.setattr(chroma, "upsert_embeddings", failing_upsert_embeddings)

    stats = _run_update(monkeypatch, search, _items(8))

    # The shared end-of-run retry pass re-attempts each document through
    # upsert_documents, moving it out of `errors` and into `recovered_items`.
    assert chroma.document_batches, "documents were never retried"
    assert stats["recovered_items"] == 8
    assert stats["errors"] == 0


def test_producer_failure_does_not_hang_the_pipeline(monkeypatch):
    """A producer that raises still releases every worker.

    Regression guard for a deadlock shape: the sentinels that end the commit
    loop are pushed from a `finally`, so an exception before the producer's
    normal end cannot leave workers blocked on an empty queue forever. The
    test passing *at all* rather than hanging is the assertion; the error
    surfacing through update_database's usual reporting is the corollary.
    """
    embedder = FakeEmbeddingFunction(max_parallel_requests=3, delay=0)
    chroma = FakeStreamingChromaClient(embedder)
    search = _make_search(monkeypatch, chroma)

    def exploding_prepare(items, force_rebuild=False):
        raise ValueError("preparation blew up")

    monkeypatch.setattr(search, "_prepare_and_classify_slice", exploding_prepare)

    stats = _run_update(monkeypatch, search, _items(10))

    assert "preparation blew up" in stats["error"]
    assert not any(
        thread.name.startswith("zmcp-index-")
        for thread in threading.enumerate()
    ), "pipeline threads outlived the run"


# -- eviction ----------------------------------------------------------


def test_evicts_the_fulltext_cache_once_per_commit(monkeypatch):
    """Every committed item key is evicted, and nothing else is."""
    evicted = []

    def recording_evict(item_keys, config_path=None):
        keys = list(item_keys)
        evicted.append(keys)
        return len(keys)

    monkeypatch.setattr(semantic_search.fulltext_cache, "evict_many", recording_evict)

    embedder = FakeEmbeddingFunction(max_parallel_requests=3, request_batch_size=4, delay=0)
    chroma = FakeStreamingChromaClient(embedder)
    search = _make_search(monkeypatch, chroma)
    items = _items(16)

    _run_update(monkeypatch, search, items)

    assert len(evicted) == len(chroma.embedding_batches)
    assert {key for batch in evicted for key in batch} == {i["key"] for i in items}


def test_does_not_evict_when_the_commit_failed(monkeypatch):
    """A failed commit must leave the cached text in place for the next run."""
    evicted = []
    monkeypatch.setattr(
        semantic_search.fulltext_cache,
        "evict_many",
        lambda item_keys, config_path=None: evicted.append(list(item_keys)),
    )

    embedder = FakeEmbeddingFunction(max_parallel_requests=2, delay=0)
    chroma = FakeStreamingChromaClient(embedder)
    search = _make_search(monkeypatch, chroma)
    monkeypatch.setattr(
        chroma,
        "upsert_embeddings",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no")),
    )
    monkeypatch.setattr(chroma, "upsert_documents", lambda *a, **k: None)

    _run_update(monkeypatch, search, _items(6))

    assert evicted == []


# -- chunking ----------------------------------------------------------


CHUNKING_ON = {
    "enabled": True,
    "chunk_size": 60,
    "overlap": 10,
    "max_chunks_per_item": 20,
}


def test_chunked_streaming_keeps_accounting_per_item(monkeypatch):
    """Added/updated counts items, not chunks, and stale chunks are cleared."""
    items = _items(6)
    # Give one item enough text to chunk into several passages.
    items[0]["data"]["abstractNote"] = "long body text " * 40

    embedder = FakeEmbeddingFunction(max_parallel_requests=3, request_batch_size=4, delay=0)
    chroma = FakeStreamingChromaClient(embedder, existing_keys={"ITEM0000"})
    search = _make_search(monkeypatch, chroma, chunking=CHUNKING_ON)

    stats = _run_update(monkeypatch, search, items)

    assert chroma.embedding_batches, "streaming path did not run"
    assert stats["updated_items"] == 1
    assert stats["added_items"] == 5
    assert "ITEM0000" in chroma.deleted_item_keys

    committed = committed_ids(chroma)
    assert len(committed) > len(items), "chunking should emit more ids than items"
    assert all("#" in doc_id for doc_id in committed)


def test_an_items_chunks_are_never_split_across_requests(monkeypatch):
    """One item's passages always travel in a single embedding request.

    Splitting them would let an item be committed half-indexed when one of the
    two requests failed, while `delete_item_chunks` had already cleared the
    old passages.
    """
    items = _items(4)
    for item in items:
        item["data"]["abstractNote"] = "long body text " * 40

    embedder = FakeEmbeddingFunction(max_parallel_requests=2, request_batch_size=2, delay=0)
    chroma = FakeStreamingChromaClient(embedder)
    search = _make_search(monkeypatch, chroma, chunking=CHUNKING_ON)

    _run_update(monkeypatch, search, items)

    seen_before = set()
    for documents, _metas, ids, _embeddings in chroma.embedding_batches:
        keys_here = {doc_id.split("#", 1)[0] for doc_id in ids}
        assert not (keys_here & seen_before), (
            "an item's chunks were spread across two commits"
        )
        seen_before |= keys_here


# -- the request splitter ----------------------------------------------


def _prepared(item_doc_counts, existing=frozenset()):
    keys = [f"K{i}" for i in range(len(item_doc_counts))]
    documents, ids = [], []
    for key, count in zip(keys, item_doc_counts):
        for chunk_index in range(count):
            documents.append(f"{key}-{chunk_index}")
            ids.append(f"{key}#{chunk_index}")
    return {
        "documents": documents,
        "metadatas": [{} for _ in documents],
        "ids": ids,
        "item_keys_order": keys,
        "item_doc_counts": list(item_doc_counts),
        "existing_item_keys": set(existing),
    }


def test_splitter_groups_up_to_the_request_batch_size():
    """Payloads fill to request_batch_size, flushing on item boundaries."""
    payloads = list(
        semantic_search._split_prepared_into_requests(_prepared([1, 1, 1, 1, 1]), 2)
    )
    assert [len(docs) for docs, _m, _i, _k in payloads] == [2, 2, 1]


def test_splitter_never_splits_one_item():
    """An item bigger than request_batch_size still travels whole."""
    payloads = list(
        semantic_search._split_prepared_into_requests(_prepared([5, 1]), 2)
    )
    assert [len(docs) for docs, _m, _i, _k in payloads] == [5, 1]
    assert [ids for _d, _m, ids, _k in payloads][0] == [f"K0#{i}" for i in range(5)]


def test_splitter_reports_existing_items():
    """Each payload carries (item_key, already_existed) for the committer."""
    payloads = list(
        semantic_search._split_prepared_into_requests(
            _prepared([1, 1], existing={"K1"}), 10
        )
    )
    assert [keys for _d, _m, _i, keys in payloads] == [[("K0", False), ("K1", True)]]


def test_splitter_skips_items_that_produced_no_documents():
    """A zero-document item contributes nothing and no empty payload."""
    payloads = list(
        semantic_search._split_prepared_into_requests(_prepared([1, 0, 1]), 10)
    )
    assert len(payloads) == 1
    assert [keys for _d, _m, _i, keys in payloads] == [[("K0", False), ("K2", False)]]


def test_splitter_yields_nothing_for_an_empty_slice():
    """No documents means no requests at all."""
    assert list(semantic_search._split_prepared_into_requests(_prepared([]), 10)) == []


# ---------------------------------------------------------------------------
# Slice sizing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("max_parallel", "expected"),
    [(1, 25), (2, 50), (4, 100), (8, 200), (16, 200), (64, 200)],
)
def test_realtime_slice_size_scales_then_caps(max_parallel, expected):
    """Slices grow with parallelism so every worker has payloads waiting.

    The 200 cap is not cosmetic: the classify step for a slice holds
    _chroma_call_lock, so an uncapped slice would lengthen the window during
    which nothing else can touch the collection.
    """
    assert semantic_search._realtime_slice_size(max_parallel) == expected


@pytest.mark.parametrize("degenerate", [0, -1])
def test_realtime_slice_size_floors_at_the_sequential_batch(degenerate):
    """A non-positive worker count must still yield a usable slice size.

    Both degenerate inputs break the producer, differently, if the floor is
    removed: the slice size becomes the step of ``range(0, len(all_items),
    slice_size)``, so a negative gives an empty range — the producer emits
    nothing and the run reports success having indexed no items — and a zero
    raises ``ValueError: range() arg 3 must not be zero``.

    Zero is unreachable today because the call site coerces it (``... or 1``),
    but a negative is not: ``-2 or 1`` is ``-2``, so a hand-edited
    ``max_parallel_requests: -2`` reaches this function intact. The silent
    variant is the one worth guarding against.
    """
    assert semantic_search._realtime_slice_size(degenerate) == 25
