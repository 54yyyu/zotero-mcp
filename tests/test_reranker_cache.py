"""Regression tests for issue #283.

With the reranker enabled, every ``zotero_semantic_search`` MCP call constructs
a fresh ``ZoteroSemanticSearch`` (``create_semantic_search`` is called per
request). The cross-encoder was held on that throwaway instance
(``self._reranker``), so the ~30s model load happened on *every* request and
blew past the client's timeout.

Fix: cache the loaded reranker at module scope, keyed by model name, so it is
loaded once per process and reused across instances. ``warmup_reranker`` lets
the server populate that cache at startup so the first real search is fast too.

These tests use a fake reranker class (patched over ``CrossEncoderReranker``) to
assert the *caching behaviour* without loading the real ~30s model.
"""

import pytest

from zotero_mcp import semantic_search


class _FakeReranker:
    """Counts how many times a reranker is actually constructed (= model load)."""

    load_count = 0

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        type(self).load_count += 1
        self.model_name = model_name


class _FakeChromaClient:
    embedding_max_tokens = 8000

    def search(self, *a, **k):
        return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}


@pytest.fixture(autouse=True)
def _isolate_cache(monkeypatch):
    """Each test gets a clean cache and a fresh, fake, non-loading reranker."""
    semantic_search._RERANKER_CACHE.clear()
    _FakeReranker.load_count = 0
    monkeypatch.setattr(semantic_search, "CrossEncoderReranker", _FakeReranker)
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    yield
    semantic_search._RERANKER_CACHE.clear()


def test_get_cached_reranker_loads_once_per_model():
    r1 = semantic_search.get_cached_reranker("model-a")
    r2 = semantic_search.get_cached_reranker("model-a")
    assert r1 is r2
    assert _FakeReranker.load_count == 1  # second call hits the cache

    r3 = semantic_search.get_cached_reranker("model-b")
    assert r3 is not r1
    assert _FakeReranker.load_count == 2  # distinct model loads separately


def test_reranker_shared_across_instances():
    """The core #283 fix: two separate ZoteroSemanticSearch instances (as the
    per-request MCP path creates) must reuse one loaded model."""
    s1 = semantic_search.ZoteroSemanticSearch(chroma_client=_FakeChromaClient())
    s2 = semantic_search.ZoteroSemanticSearch(chroma_client=_FakeChromaClient())
    s1._reranker_config = {"enabled": True, "model": "shared-model"}
    s2._reranker_config = {"enabled": True, "model": "shared-model"}

    r1 = s1._get_reranker()
    r2 = s2._get_reranker()

    assert r1 is r2
    assert _FakeReranker.load_count == 1  # loaded once despite two instances


def test_get_reranker_returns_none_when_disabled():
    s = semantic_search.ZoteroSemanticSearch(chroma_client=_FakeChromaClient())
    s._reranker_config = {"enabled": False, "model": "shared-model"}
    assert s._get_reranker() is None
    assert _FakeReranker.load_count == 0  # disabled never loads


def test_warmup_reranker_populates_cache(monkeypatch):
    monkeypatch.setattr(semantic_search.os.path, "exists", lambda p: False)
    # Disabled config -> no warmup, no load.
    assert semantic_search.warmup_reranker(config_path=None) is False
    assert _FakeReranker.load_count == 0

    # Enabled config -> warms the cache exactly once.
    monkeypatch.setattr(
        semantic_search,
        "load_reranker_config",
        lambda cp: {"enabled": True, "model": "warm-model"},
    )
    assert semantic_search.warmup_reranker(config_path=None) is True
    assert _FakeReranker.load_count == 1
    # The warmed instance is what a later search would get.
    assert semantic_search.get_cached_reranker("warm-model") is not None
    assert _FakeReranker.load_count == 1  # already warm -> no extra load
