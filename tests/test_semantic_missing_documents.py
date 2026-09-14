"""A hit whose document was deleted under a running server must not fail search (#545)."""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip("chromadb is incompatible with Python 3.14+", allow_module_level=True)

from zotero_mcp.semantic_search import _drop_missing_documents


def _results(docs):
    n = len(docs)
    return {
        "ids": [[f"K{i}" for i in range(n)]],
        "distances": [[float(i) for i in range(n)]],
        "documents": [list(docs)],
        "metadatas": [[{"item_key": f"K{i}"} for i in range(n)]],
    }


def test_none_documents_are_dropped_and_columns_stay_aligned():
    results = _results(["alpha", None, "gamma", None])
    assert _drop_missing_documents(results) == 2
    assert results["documents"][0] == ["alpha", "gamma"]
    assert results["ids"][0] == ["K0", "K2"]
    assert results["distances"][0] == [0.0, 2.0]
    assert [m["item_key"] for m in results["metadatas"][0]] == ["K0", "K2"]


def test_clean_results_are_untouched():
    results = _results(["alpha", "beta"])
    assert _drop_missing_documents(results) == 0
    assert results["ids"][0] == ["K0", "K1"]


def test_empty_results_are_fine():
    assert _drop_missing_documents({"documents": [[]]}) == 0
    assert _drop_missing_documents({}) == 0
