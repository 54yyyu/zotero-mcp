"""The Zotero items of a search's hits are fetched 50 to a request.

Each hit's item was fetched with a request of its own: 10 for a default
search, and up to 200 for the keyword search's filtered semantic fallback.
"""

import importlib.util
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

from test_passage_chunking import _chunking_search


class _Zotero:
    def __init__(self, batch=True, missing=()):
        self.batch = batch
        self.missing = set(missing)
        self.calls = []

    def _item(self, key):
        return {"key": key, "data": {"title": f"Title {key}"}}

    def items(self, **kwargs):
        self.calls.append(("items", kwargs.get("itemKey")))
        if not self.batch:
            raise RuntimeError("400 Bad Request")
        return [self._item(k) for k in kwargs["itemKey"].split(",") if k not in self.missing]

    def item(self, key):
        self.calls.append(("item", key))
        if key == "BROKEN":
            raise RuntimeError("500 Internal Server Error")
        return self._item(key)


def _results(keys, **extra):
    return {
        "ids": [list(keys)],
        "distances": [[0.1] * len(keys)],
        "documents": [[f"text of {k}" for k in keys]],
        "metadatas": [[{"item_key": k} for k in keys]],
        **extra,
    }


def test_hits_are_hydrated_fifty_to_a_request(monkeypatch):
    search = _chunking_search(monkeypatch)
    search.zotero_client = zot = _Zotero()
    keys = [f"K{i:03d}" for i in range(120)]

    enriched = search._enrich_search_results(_results(keys), "q", limit=200)

    assert [r["zotero_item"]["data"]["title"] for r in enriched] == [f"Title {k}" for k in keys]
    assert [name for name, _ in zot.calls] == ["items"] * 3


def test_a_hit_the_batch_does_not_return_is_asked_for_on_its_own(monkeypatch):
    search = _chunking_search(monkeypatch)
    search.zotero_client = zot = _Zotero(missing={"B", "BROKEN"})

    enriched = search._enrich_search_results(_results(["A", "B", "BROKEN"]), "q", limit=10)

    by_key = {r["item_key"]: r for r in enriched}
    assert by_key["A"]["zotero_item"]["key"] == "A"
    assert by_key["B"]["zotero_item"]["key"] == "B"
    assert "500" in by_key["BROKEN"]["error"] and "zotero_item" not in by_key["BROKEN"]
    assert ("item", "A") not in zot.calls and ("item", "B") in zot.calls


def test_a_failed_batch_request_hydrates_per_item(monkeypatch):
    search = _chunking_search(monkeypatch)
    search.zotero_client = zot = _Zotero(batch=False)

    enriched = search._enrich_search_results(_results(["A", "B"]), "q", limit=10)

    assert [r["zotero_item"]["key"] for r in enriched] == ["A", "B"]
    assert zot.calls == [("items", "A,B"), ("item", "A"), ("item", "B")]


def test_a_single_hit_is_one_request(monkeypatch):
    search = _chunking_search(monkeypatch)
    search.zotero_client = zot = _Zotero()

    search._enrich_search_results(_results(["A"]), "q", limit=10)

    assert zot.calls == [("item", "A")]
