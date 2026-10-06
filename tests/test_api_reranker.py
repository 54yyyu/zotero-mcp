"""Hosted re-ranking through ``APIReranker`` with every HTTP call stubbed.

Covers the cases a hosted endpoint can produce -- success in either response
envelope, HTTP errors, throttling, an empty answer, a network failure -- plus
configuration errors (missing key, unknown provider), and how the result is
reported back through ``search()``.
"""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

import requests

from zotero_mcp import semantic_search
from zotero_mcp.semantic_search import APIReranker, get_cached_reranker

DOCS = ["about cats", "about dogs", "about birds"]


class _Resp:
    def __init__(self, status=200, body=None, text="", headers=None):
        self.status_code = status
        self._body = body
        self.text = text
        self.headers = headers or {}

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


@pytest.fixture
def calls(monkeypatch):
    """Queue of responses for requests.post; records each payload sent."""
    state = {"responses": [], "payloads": [], "urls": []}

    def fake_post(url, json=None, headers=None, timeout=None):
        state["urls"].append(url)
        state["payloads"].append(json)
        nxt = state["responses"].pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(semantic_search.time, "sleep", lambda s: None)
    return state


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    for var in ("VOYAGE_API_KEY", "COHERE_API_KEY", "OPENROUTER_API_KEY", "CONTEXTUAL_API_KEY", "MY_KEY"):
        monkeypatch.setenv(var, "test-key")


def test_voyage_data_envelope_is_read(calls):
    # Voyage answers {"object": "list", "data": [...]}, not {"results": [...]}.
    calls["responses"].append(_Resp(body={
        "object": "list",
        "data": [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.4}],
        "model": "rerank-3",
        "usage": {"total_tokens": 42},
    }))
    r = APIReranker("rerank-3", "voyage")
    pairs, error = r.rerank_detailed("birds", DOCS, top_k=2)
    assert error is None
    assert pairs == [(2, 0.9), (0, 0.4)]
    sent = calls["payloads"][0]
    assert sent["top_k"] == 2 and "top_n" not in sent
    assert sent["model"] == "rerank-3"
    assert calls["urls"][0] == "https://api.voyageai.com/v1/rerank"


def test_results_envelope_is_read(calls):
    calls["responses"].append(_Resp(body={"results": [{"index": 1, "relevance_score": 0.7}]}))
    r = APIReranker("rerank-v3.5", "cohere")
    pairs, error = r.rerank_detailed("dogs", DOCS, top_k=3)
    assert error is None
    assert pairs == [(1, 0.7)]
    assert calls["payloads"][0]["top_n"] == 3


def test_unsorted_records_are_sorted_and_bad_rows_skipped(calls):
    calls["responses"].append(_Resp(body={"results": [
        {"index": 0, "relevance_score": 0.1},
        {"index": 7, "relevance_score": 0.99},  # out of range
        {"relevance_score": 0.5},  # no index
        {"index": 2, "relevance_score": 0.8},
        {"index": 2, "relevance_score": 0.3},  # duplicate
    ]}))
    pairs, error = APIReranker("m", "cohere").rerank_detailed("q", DOCS, top_k=3)
    assert error is None
    assert pairs == [(2, 0.8), (0, 0.1)]


def test_http_error_falls_back_to_retrieval_order_and_says_why(calls):
    calls["responses"].append(_Resp(status=400, text='{"detail": "Model rerank-9 is not supported."}'))
    pairs, error = APIReranker("rerank-9", "voyage").rerank_detailed("q", DOCS, top_k=2)
    assert pairs == [(0, 0.0), (1, 0.0)]
    assert error.startswith("HTTP 400")
    assert "rerank-9" in error


def test_throttled_request_is_retried_once(calls):
    calls["responses"].extend([
        _Resp(status=429, headers={"Retry-After": "2"}),
        _Resp(body={"data": [{"index": 1, "relevance_score": 0.6}]}),
    ])
    pairs, error = APIReranker("rerank-3", "voyage").rerank_detailed("q", DOCS, top_k=1)
    assert error is None and pairs == [(1, 0.6)]
    assert len(calls["payloads"]) == 2


def test_persistent_server_error_gives_up_after_one_retry(calls):
    calls["responses"].extend([_Resp(status=503), _Resp(status=503)])
    pairs, error = APIReranker("m", "cohere").rerank_detailed("q", DOCS, top_k=3)
    assert error.startswith("HTTP 503")
    assert pairs == [(0, 0.0), (1, 0.0), (2, 0.0)]
    assert len(calls["payloads"]) == 2


def test_network_error_falls_back(calls):
    calls["responses"].extend([requests.ConnectionError("down"), requests.ConnectionError("down")])
    pairs, error = APIReranker("m", "cohere").rerank_detailed("q", DOCS, top_k=2)
    assert "request failed" in error
    assert pairs == [(0, 0.0), (1, 0.0)]


def test_empty_result_list_is_an_error(calls):
    calls["responses"].append(_Resp(body={"results": []}))
    pairs, error = APIReranker("m", "cohere").rerank_detailed("q", DOCS, top_k=2)
    assert error == "response listed no ranked documents"
    assert pairs == [(0, 0.0), (1, 0.0)]


def test_unreadable_json_is_an_error(calls):
    calls["responses"].append(_Resp(body=ValueError("not json")))
    _, error = APIReranker("m", "cohere").rerank_detailed("q", DOCS, top_k=2)
    assert error.startswith("unreadable response")


def test_no_documents_makes_no_request(calls):
    assert APIReranker("m", "voyage").rerank_detailed("q", [], top_k=5) == ([], None)
    assert calls["payloads"] == []


def test_rerank_and_rerank_with_scores_share_the_result(calls):
    body = {"data": [{"index": 2, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.2}]}
    calls["responses"].extend([_Resp(body=body), _Resp(body=body)])
    r = APIReranker("rerank-3", "voyage")
    assert r.rerank("q", DOCS, top_k=2) == [2, 1]
    assert r.rerank_with_scores("q", DOCS, top_k=2) == [(2, 0.9), (1, 0.2)]


def test_voyage_instruction_is_prepended_to_the_query(calls):
    calls["responses"].append(_Resp(body={"data": [{"index": 0, "relevance_score": 1.0}]}))
    r = APIReranker("rerank-3", "voyage", instruction="Prefer empirical findings.")
    r.rerank_detailed("parental control", DOCS, top_k=1)
    sent = calls["payloads"][0]
    assert "instruction" not in sent
    assert sent["query"].startswith("Prefer empirical findings.")
    assert sent["query"].endswith("parental control")


def test_contextual_instruction_is_a_field(calls):
    calls["responses"].append(_Resp(body={"results": [{"index": 0, "relevance_score": 1.0}]}))
    r = APIReranker("ctxl-rerank-v2-instruct", "contextual", instruction="Recent first.")
    r.rerank_detailed("q", DOCS, top_k=1)
    sent = calls["payloads"][0]
    assert sent["instruction"] == "Recent first."
    assert sent["query"] == "q"


def test_no_instruction_sends_neither(calls):
    calls["responses"].append(_Resp(body={"data": [{"index": 0, "relevance_score": 1.0}]}))
    APIReranker("rerank-3", "voyage").rerank_detailed("q", DOCS, top_k=1)
    assert calls["payloads"][0]["query"] == "q"
    assert "instruction" not in calls["payloads"][0]


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY")
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        APIReranker("rerank-3", "voyage")


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown rerank provider"):
        APIReranker("m", "nonesuch")


def test_custom_endpoint_needs_url_and_key_env(calls):
    calls["responses"].append(_Resp(body={"results": [{"index": 0, "relevance_score": 0.5}]}))
    r = APIReranker("m", "selfhosted", base_url="https://rr.example/v1/rerank", api_key_env="MY_KEY")
    pairs, error = r.rerank_detailed("q", DOCS, top_k=1)
    assert error is None and pairs == [(0, 0.5)]
    assert calls["urls"][0] == "https://rr.example/v1/rerank"


def test_cache_is_keyed_by_provider(monkeypatch):
    monkeypatch.setattr(semantic_search, "_RERANKER_CACHE", {})
    a = get_cached_reranker("rerank-3", {"provider": "voyage"})
    b = get_cached_reranker("rerank-3", {"provider": "voyage"})
    c = get_cached_reranker("rerank-3", {"provider": "openrouter"})
    assert a is b
    assert a is not c
    assert isinstance(a, APIReranker) and a.provider == "voyage"


def test_unavailable_reranker_degrades_to_no_reranking(monkeypatch):
    monkeypatch.setattr(semantic_search, "_RERANKER_CACHE", {})
    monkeypatch.delenv("VOYAGE_API_KEY")
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    s = semantic_search.ZoteroSemanticSearch(chroma_client=object())
    s._reranker_config = {"enabled": True, "model": "rerank-3", "provider": "voyage"}
    assert s._get_reranker() is None


# --- search() reports what the re-ranker did -------------------------------


class _Chroma:
    embedding_max_tokens = 8000

    def search(self, query_texts=None, n_results=10, where=None, where_document=None):
        return {
            "ids": [["A#0", "B#0", "C#0"]],
            "distances": [[0.1, 0.2, 0.3]],
            "documents": [["a text", "b text", "c text"]],
            "metadatas": [[{"title": "A"}, {"title": "B"}, {"title": "C"}]],
        }


def _search_with(monkeypatch, reranker):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    s = semantic_search.ZoteroSemanticSearch(chroma_client=_Chroma())
    s._chunking_config = {"enabled": True}
    s._reranker_config = {"enabled": True, "candidate_multiplier": 3}
    s._reranker = reranker
    s._attach_zotero_items = lambda enriched: None
    return s


def test_search_reports_applied_rerank_and_scores(monkeypatch, calls):
    calls["responses"].append(_Resp(body={"data": [
        {"index": 2, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.5},
        {"index": 1, "relevance_score": 0.1},
    ]}))
    s = _search_with(monkeypatch, APIReranker("rerank-3", "voyage"))
    out = s.search("q", limit=3)
    assert [r["item_key"] for r in out["results"]] == ["C", "A", "B"]
    assert [r["rerank_score"] for r in out["results"]] == [0.9, 0.5, 0.1]
    assert out["rerank"] == {
        "model": "voyage/rerank-3", "applied": True, "error": None, "candidates": 3,
    }


def test_search_reports_failed_rerank(monkeypatch, calls):
    calls["responses"].append(_Resp(status=401, text="invalid key"))
    s = _search_with(monkeypatch, APIReranker("rerank-3", "voyage"))
    out = s.search("q", limit=3)
    assert [r["item_key"] for r in out["results"]] == ["A", "B", "C"]
    assert all("rerank_score" not in r for r in out["results"])
    assert out["rerank"]["applied"] is False
    assert out["rerank"]["error"].startswith("HTTP 401")


# --- the semantic search tool states how results were ordered ----------------


class _FixedSemanticSearch:
    def __init__(self, rerank):
        self._rerank = rerank

    def search(self, query, limit=10, filters=None, group_id=None):
        hit = {
            "item_key": "ITEM0001",
            "similarity_score": 0.52,
            "matched_passage": "a passage",
            "metadata": {},
            "zotero_item": {"key": "ITEM0001", "data": {"title": "A Paper", "itemType": "journalArticle"}},
        }
        if self._rerank and self._rerank.get("applied"):
            hit["rerank_score"] = 0.91
        return {"results": [hit], "total_found": 1, "rerank": self._rerank}


def _run_tool(monkeypatch, tmp_path, rerank):
    from conftest import DummyContext

    from zotero_mcp import client as _client
    from zotero_mcp.tools import search as search_module

    config_dir = tmp_path / ".config" / "zotero-mcp"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text("{}")
    monkeypatch.setattr(search_module.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(semantic_search, "create_semantic_search", lambda *a, **kw: _FixedSemanticSearch(rerank))
    monkeypatch.setattr(search_module, "_maybe_fire_presearch_sync", lambda _s: None)
    monkeypatch.setattr(_client, "get_active_group_id", lambda: 0)
    return search_module.semantic_search(query="q", ctx=DummyContext())


def test_tool_output_names_the_reranker(monkeypatch, tmp_path):
    out = _run_tool(monkeypatch, tmp_path, {
        "model": "voyage/rerank-3", "applied": True, "error": None, "candidates": 80,
    })
    assert "Ranked by voyage/rerank-3 over 80 candidate passages" in out
    assert "**Relevance:** 0.910" in out
    assert "**Similarity:** 0.520" in out


def test_tool_output_flags_a_failed_rerank(monkeypatch, tmp_path):
    out = _run_tool(monkeypatch, tmp_path, {
        "model": "voyage/rerank-3", "applied": False, "error": "HTTP 401: invalid key", "candidates": 80,
    })
    assert "re-ranking with voyage/rerank-3 failed (HTTP 401: invalid key)" in out
    assert "**Relevance:** 0.520" in out
    assert "Similarity" not in out


def test_tool_output_without_reranker_is_unchanged(monkeypatch, tmp_path):
    out = _run_tool(monkeypatch, tmp_path, None)
    assert "Ranked by" not in out and "re-ranking" not in out
    assert "**Relevance:** 0.520" in out
