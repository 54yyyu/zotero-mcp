"""Tests for the chunking, groupby rerank, throttle, and id-format
migration introduced in the chunked semantic search PR.
"""

import json
import sys
import time

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search


class FakeChromaClient:
    """Richer ChromaClient double covering chunked-era surface."""

    def __init__(self, preloaded_ids=None):
        self.embedding_max_tokens = 8000
        self._ids = set(preloaded_ids or [])
        self.added = []
        self.deleted = []
        self.deleted_by_parent = []
        self.reset_calls = 0
        self._last_query_kwargs = None
        self._search_response = {
            "ids": [[]],
            "distances": [[]],
            "documents": [[]],
            "metadatas": [[]],
        }

    def truncate_text(self, text, max_tokens=None):
        return text

    def get_existing_ids(self, ids):
        return {i for i in ids if i in self._ids}

    def get_all_ids(self):
        return set(self._ids)

    def get_document_metadata(self, doc_id):
        return None

    def upsert_documents(self, documents, metadatas, ids):
        self.added.append((list(documents), list(metadatas), list(ids)))
        for i in ids:
            self._ids.add(i)

    def delete_documents(self, ids):
        self.deleted.extend(list(ids))
        for i in ids:
            self._ids.discard(i)

    def delete_documents_by_parent(self, parent_item_key):
        prefix = f"{parent_item_key}__"
        victims = [i for i in self._ids if i == parent_item_key or i.startswith(prefix)]
        if victims:
            self.deleted.extend(victims)
            for v in victims:
                self._ids.discard(v)
        self.deleted_by_parent.append(parent_item_key)
        return len(victims)

    def reset_collection(self):
        self.reset_calls += 1
        self._ids = set()

    def set_search_response(self, ids, distances, documents, metadatas):
        self._search_response = {
            "ids": [ids],
            "distances": [distances],
            "documents": [documents],
            "metadatas": [metadatas],
        }

    def search(self, query_texts=None, n_results=10, where=None, where_document=None):
        self._last_query_kwargs = {
            "query_texts": query_texts,
            "n_results": n_results,
            "where": where,
        }
        return self._search_response


def _build_search(monkeypatch, chroma=None, config_path=None,
                  zotero_client=None):
    monkeypatch.setattr(semantic_search, "get_zotero_client",
                        lambda: zotero_client or object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    return semantic_search.ZoteroSemanticSearch(
        chroma_client=chroma or FakeChromaClient(),
        config_path=config_path,
    )


def _write_config(tmp_path, extra=None):
    cfg = {
        "semantic_search": {
            "embedding_model": "default",
            "update_config": {"auto_update": False, "update_frequency": "manual"},
            "extraction": {"pdf_max_pages": 10},
            "include_fulltext": True,
        }
    }
    if extra:
        cfg["semantic_search"].update(extra)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return str(p)


# --------- Chunking ---------

def test_chunk_short_text_returns_single_chunk(monkeypatch):
    search = _build_search(monkeypatch)
    chunks = search._chunk_document("A short abstract.", window=1500, overlap=225)
    assert chunks == ["A short abstract."]


def test_chunk_empty_returns_empty(monkeypatch):
    search = _build_search(monkeypatch)
    assert search._chunk_document("", window=1500) == []
    assert search._chunk_document("    ", window=1500) == []


def test_chunk_long_text_produces_multiple_chunks(monkeypatch):
    search = _build_search(monkeypatch)
    # Build a document well above the window size. cl100k_base typically
    # counts ~1.3 tokens per 1-char word; use a very small window so any
    # tokenizer yields at least two chunks.
    text = ("word " * 400).strip()
    chunks = search._chunk_document(text, window=50, overlap=10)
    assert len(chunks) >= 2
    # Chunks are non-empty and meaningfully sized
    for c in chunks:
        assert c.strip()


def test_chunk_overlap_clamped_to_window(monkeypatch):
    """Malformed config with overlap >= window must not loop forever."""
    search = _build_search(monkeypatch)
    text = ("word " * 300).strip()
    # overlap == window would cause step=0 in a naive implementation
    chunks = search._chunk_document(text, window=20, overlap=20)
    assert chunks  # finite, non-empty
    assert len(chunks) < 1000  # didn't explode


def test_chunk_size_respects_window(monkeypatch):
    """Every emitted chunk must fit inside the configured window token limit."""
    search = _build_search(monkeypatch)
    text = ("paragraph text. " * 500).strip()
    window = 100
    chunks = search._chunk_document(text, window=window, overlap=10)
    # Use the same tokenizer the chunker relies on
    from zotero_mcp.semantic_search import _tokenizer
    if _tokenizer is None:
        pytest.skip("tiktoken not available")
    for c in chunks:
        assert len(_tokenizer.encode(c, disallowed_special=())) <= window


# --------- Chunked ingest via _process_item_batch ---------

def _item(key, title="Paper", fulltext=""):
    return {
        "key": key,
        "version": 1,
        "data": {
            "key": key,
            "itemType": "journalArticle",
            "title": title,
            "abstractNote": "abstract",
            "creators": [],
            "fulltext": fulltext,
            "dateAdded": "2024-01-01T00:00:00Z",
            "dateModified": "2024-01-01T00:00:00Z",
        },
    }


def test_process_item_batch_emits_chunked_ids(monkeypatch):
    chroma = FakeChromaClient()
    search = _build_search(monkeypatch, chroma=chroma)
    item = _item("ITEM1", title="T", fulltext="body " * 2000)
    stats = search._process_item_batch([item], force_rebuild=True)
    assert stats["processed"] == 1
    # All added ids are `ITEM1__<i>` and at least 2 chunks emitted
    added_ids = [i for batch in chroma.added for i in batch[2]]
    assert all(i.startswith("ITEM1__") for i in added_ids)
    assert len(added_ids) >= 2
    # Each chunk metadata has parent_item_key and chunk_index
    metas = [m for batch in chroma.added for m in batch[1]]
    assert all(m.get("parent_item_key") == "ITEM1" for m in metas)
    indices = [m["chunk_index"] for m in metas]
    assert indices == list(range(len(indices)))


def test_process_item_batch_cleans_stale_chunks_before_upsert(monkeypatch):
    chroma = FakeChromaClient(preloaded_ids=[
        "ITEM1__0", "ITEM1__1", "ITEM1__2",  # stale chunks from prior run
    ])
    search = _build_search(monkeypatch, chroma=chroma)
    item = _item("ITEM1", title="T", fulltext="body")  # short → 1 chunk
    search._process_item_batch([item], force_rebuild=False)
    # Cleanup called with parent key
    assert "ITEM1" in chroma.deleted_by_parent


# --------- Groupby rerank (_dedupe_by_parent + search) ---------

def test_dedupe_by_parent_keeps_best_chunk(monkeypatch):
    search = _build_search(monkeypatch)
    raw = {
        "ids": [["P1__0", "P1__2", "P2__0", "P1__1"]],
        "distances": [[0.5, 0.3, 0.4, 0.1]],
        "documents": [["d0", "d2", "p2d0", "d1"]],
        "metadatas": [[
            {"parent_item_key": "P1", "chunk_index": 0},
            {"parent_item_key": "P1", "chunk_index": 2},
            {"parent_item_key": "P2", "chunk_index": 0},
            {"parent_item_key": "P1", "chunk_index": 1},
        ]],
    }
    out = search._dedupe_by_parent(raw, keep=10)
    ids = out["ids"][0]
    # Expected: P1 best chunk = P1__1 (distance 0.1), P2 = P2__0 (0.4)
    # Order after dedupe is by distance ascending
    assert ids == ["P1__1", "P2__0"]


def test_dedupe_by_parent_respects_keep_limit(monkeypatch):
    search = _build_search(monkeypatch)
    raw = {
        "ids": [[f"P{i}__0" for i in range(10)]],
        "distances": [[0.1 * i for i in range(10)]],
        "documents": [[f"d{i}" for i in range(10)]],
        "metadatas": [[{"parent_item_key": f"P{i}"} for i in range(10)]],
    }
    out = search._dedupe_by_parent(raw, keep=3)
    assert len(out["ids"][0]) == 3


def test_dedupe_by_parent_falls_back_to_id_prefix(monkeypatch):
    """Legacy entries without parent_item_key metadata should still dedupe
    via the `__` prefix convention."""
    search = _build_search(monkeypatch)
    raw = {
        "ids": [["P1__0", "P1__1"]],
        "distances": [[0.2, 0.1]],
        "documents": [["d0", "d1"]],
        "metadatas": [[{}, {}]],  # missing parent_item_key
    }
    out = search._dedupe_by_parent(raw, keep=10)
    assert out["ids"][0] == ["P1__1"]  # best chunk kept


def test_search_enriches_with_parent_key(monkeypatch):
    """Enrichment must look up the parent item, not the chunk id."""
    class RecordingZoteroClient:
        def __init__(self):
            self.calls = []

        def item(self, key):
            self.calls.append(key)
            return {"key": key, "data": {"title": f"title-{key}"}}

    zot = RecordingZoteroClient()
    chroma = FakeChromaClient()
    chroma.set_search_response(
        ids=["PAPER__3"],
        distances=[0.1],
        documents=["matched paragraph content"],
        metadatas=[{"parent_item_key": "PAPER", "chunk_index": 3, "title": "The Paper"}],
    )
    search = _build_search(monkeypatch, chroma=chroma, zotero_client=zot)
    result = search.search("query", limit=5)
    assert zot.calls == ["PAPER"]  # parent key, not chunk id
    assert result["results"][0]["item_key"] == "PAPER"
    assert result["results"][0]["chunk_id"] == "PAPER__3"
    assert result["results"][0]["chunk_index"] == 3


def test_search_oversamples_for_chunking(monkeypatch):
    chroma = FakeChromaClient()
    chroma.set_search_response(
        ids=["X__0"], distances=[0.1], documents=["d"],
        metadatas=[{"parent_item_key": "X"}],
    )

    class StubZotero:
        def item(self, k):
            return {"key": k, "data": {"title": "x"}}
    search = _build_search(monkeypatch, chroma=chroma, zotero_client=StubZotero())
    search.search("q", limit=5)
    # Chunked oversample: max(limit * 5, 50) = 50 for small limits
    assert chroma._last_query_kwargs["n_results"] == 50


# --------- Throttle ---------

def test_throttle_with_no_config_is_noop(monkeypatch):
    search = _build_search(monkeypatch)
    t0 = time.monotonic()
    for _ in range(5):
        search._throttle_embedding_request()
    # Must complete near-instantly when no rps configured
    assert time.monotonic() - t0 < 0.5


def test_throttle_rate_limits_across_calls(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, extra={"embedding_rate_limit_rps": 10.0})
    search = _build_search(monkeypatch, config_path=config_path)
    # 3 calls at 10 rps should take at least 2 * 0.1s = 0.2s total
    t0 = time.monotonic()
    for _ in range(3):
        search._throttle_embedding_request()
    elapsed = time.monotonic() - t0
    # Lower bound check with some slack for scheduler jitter
    assert elapsed >= 0.18, f"throttle too fast: {elapsed}s for 3 calls at 10 rps"
    # Upper bound: shouldn't sleep much more than necessary
    assert elapsed < 2.0


def test_throttle_respects_invalid_rps(monkeypatch, tmp_path):
    """Negative / zero / non-numeric values must be treated as 'no throttle'."""
    for val in [0, -2, "bogus"]:
        config_path = _write_config(tmp_path, extra={"embedding_rate_limit_rps": val})
        search = _build_search(monkeypatch, config_path=config_path)
        assert search._load_embedding_rate_limit() is None


# --------- Legacy id-format migration ---------

def test_legacy_id_format_triggers_rebuild(monkeypatch, tmp_path):
    """A collection built by the pre-chunking code (ids = raw item keys with
    no `__` delimiter) must be reset on next update_database call so the
    new chunked format takes over cleanly."""
    config_path = _write_config(tmp_path, extra={"last_sync_version": 99})

    class StubZotero:
        def __init__(self):
            self.scenario_items = [_item("X")]
            self.versions = {"X": 99}

        def items(self, start=0, limit=100, **_):
            return self.scenario_items[start:start + limit]

        def item(self, k):
            return self.scenario_items[0]

        def fulltext_item(self, k):
            raise RuntimeError("no fulltext")

        def children(self, k):
            return []

        def last_modified_version(self, **_):
            return 99

        def item_versions(self, since=None, **_):
            if since is None:
                return dict(self.versions)
            return {k: v for k, v in self.versions.items() if v > since}

    # Preload with legacy-format ids (no __ delimiter)
    chroma = FakeChromaClient(preloaded_ids=["LEGACY_A", "LEGACY_B"])
    search = _build_search(monkeypatch, chroma=chroma, zotero_client=StubZotero(),
                           config_path=config_path)

    search.update_database()

    # reset_collection must have been called as part of the legacy migration
    assert chroma.reset_calls >= 1


def test_empty_collection_with_cached_sync_triggers_rebuild(monkeypatch, tmp_path):
    """If the collection was silently reset (e.g. embedding-model change)
    but config still carries last_sync_version > 0, the incremental path
    must NOT fast-path as noop — it must do a full rebuild."""
    config_path = _write_config(tmp_path, extra={"last_sync_version": 7661})

    class StubZotero:
        def __init__(self):
            self.items_data = [_item("ONLY_ONE")]

        def items(self, start=0, limit=100, **_):
            return self.items_data[start:start + limit]

        def item(self, k):
            return self.items_data[0]

        def fulltext_item(self, k):
            raise RuntimeError("no fulltext")

        def children(self, k):
            return []

        def last_modified_version(self, **_):
            return 7661  # same as cached — noop trap

        def item_versions(self, since=None, **_):
            if since is None:
                return {"ONLY_ONE": 7661}
            return {}

    # Collection is empty (e.g. model-change reset)
    chroma = FakeChromaClient(preloaded_ids=[])
    search = _build_search(monkeypatch, chroma=chroma, zotero_client=StubZotero(),
                           config_path=config_path)

    search.update_database()

    # Must have indexed the full library, not fast-pathed as noop
    assert len(chroma.added) > 0, "Full rebuild did not ingest any items"


def test_new_id_format_does_not_trigger_rebuild(monkeypatch, tmp_path):
    """A collection already in chunked format must not be rebuilt."""
    config_path = _write_config(tmp_path, extra={"last_sync_version": 99})

    class StubZotero:
        def items(self, **_):
            return []

        def last_modified_version(self, **_):
            return 99

        def item_versions(self, since=None, **_):
            return {}

    chroma = FakeChromaClient(preloaded_ids=["A__0", "A__1", "B__0"])
    search = _build_search(monkeypatch, chroma=chroma, zotero_client=StubZotero(),
                           config_path=config_path)

    search.update_database()
    assert chroma.reset_calls == 0


# --------- Config loaders ---------

def test_load_chunking_settings_defaults(monkeypatch, tmp_path):
    search = _build_search(monkeypatch, config_path=str(tmp_path / "missing.json"))
    settings = search._load_chunking_settings()
    assert settings == {"window": 1500, "overlap": 225}


def test_load_chunking_settings_overrides(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, extra={"chunking": {"window": 800, "overlap": 80}})
    search = _build_search(monkeypatch, config_path=config_path)
    settings = search._load_chunking_settings()
    assert settings["window"] == 800
    assert settings["overlap"] == 80


def test_load_embedding_rate_limit_defaults_none(monkeypatch, tmp_path):
    search = _build_search(monkeypatch, config_path=str(tmp_path / "missing.json"))
    assert search._load_embedding_rate_limit() is None


# --------- OpenAIEmbeddingFunction sub-batching ---------

def test_openai_embedding_sub_batches_large_input():
    """When input exceeds request_batch_size, __call__ splits the request.

    SiliconFlow caps /v1/embeddings at 64 inputs per POST. Without
    sub-batching, any item that chunks into >64 pieces would 413. This
    test fakes the openai client and confirms request splitting.
    """
    from zotero_mcp.chroma_client import OpenAIEmbeddingFunction
    from unittest.mock import MagicMock

    import threading as _t
    ef = OpenAIEmbeddingFunction.__new__(OpenAIEmbeddingFunction)
    ef.model_name = "BAAI/bge-m3"
    ef.api_key = "test"
    ef.base_url = "https://api.siliconflow.cn/v1"
    ef.request_batch_size = 3
    ef.rate_limit_rps = None
    ef._rate_lock = _t.Lock()
    ef._last_request_ts = 0.0
    ef.client = MagicMock()

    def fake_create(model, input):
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[float(i)]) for i in range(len(input))]
        return resp
    ef.client.embeddings.create.side_effect = fake_create

    # 7 inputs with batch_size=3 -> 3 requests: 3 + 3 + 1
    result = ef(["a", "b", "c", "d", "e", "f", "g"])
    assert len(result) == 7
    assert ef.client.embeddings.create.call_count == 3
    # Verify the split sizes
    call_args = [c.kwargs.get("input", c.args[1] if len(c.args) > 1 else None)
                 for c in ef.client.embeddings.create.call_args_list]
    assert [len(ca) for ca in call_args] == [3, 3, 1]


def test_openai_embedding_single_request_when_under_batch_size():
    """Small inputs still hit the endpoint exactly once."""
    from zotero_mcp.chroma_client import OpenAIEmbeddingFunction
    from unittest.mock import MagicMock

    import threading as _t
    ef = OpenAIEmbeddingFunction.__new__(OpenAIEmbeddingFunction)
    ef.model_name = "text-embedding-3-small"
    ef.api_key = "test"
    ef.base_url = None
    ef.request_batch_size = 64
    ef.rate_limit_rps = None
    ef._rate_lock = _t.Lock()
    ef._last_request_ts = 0.0
    ef.client = MagicMock()
    resp = MagicMock()
    resp.data = [MagicMock(embedding=[0.1, 0.2]) for _ in range(5)]
    ef.client.embeddings.create.return_value = resp

    result = ef(["a", "b", "c", "d", "e"])
    assert len(result) == 5
    assert ef.client.embeddings.create.call_count == 1
