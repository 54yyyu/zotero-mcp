"""Regression: item-level metadata lookup must find chunked records.

With passage chunking enabled, items are indexed only under their chunk ids
(``<key>#<n>``). ``get_document_metadata`` looked the item up by its bare key,
which never matched, so the local-fulltext skip-check in ``_get_items_from_local_db``
treated every already-indexed item as new and re-extracted plus re-embedded the
entire library on every update.
"""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp.chroma_client import ChromaClient


class FakeCollection:
    """Stand-in for a Chroma collection with exact-id `get` semantics."""

    def __init__(self, store):
        self.store = store

    def get(self, ids, include=None):
        found = [i for i in ids if i in self.store]
        return {"ids": found, "metadatas": [self.store[i] for i in found]}


def _client(store):
    client = object.__new__(ChromaClient)
    client.collection = FakeCollection(store)
    return client


def test_finds_chunked_item_by_bare_key():
    """A chunked item is found by its item key, via chunk 0."""
    client = _client({
        "ABCD1234#0": {"item_key": "ABCD1234", "date_modified": "2025-05-07T09:58:54Z"},
        "ABCD1234#1": {"item_key": "ABCD1234", "date_modified": "2025-05-07T09:58:54Z"},
    })

    metadata = client.get_document_metadata("ABCD1234")

    assert metadata is not None
    assert metadata["date_modified"] == "2025-05-07T09:58:54Z"


def test_finds_unchunked_item_by_bare_key():
    """Item-level (unchunked) records keep working."""
    client = _client({"ABCD1234": {"item_key": "ABCD1234", "date_modified": "2025-05-07T09:58:54Z"}})

    metadata = client.get_document_metadata("ABCD1234")

    assert metadata is not None
    assert metadata["date_modified"] == "2025-05-07T09:58:54Z"


def test_returns_none_for_unindexed_item():
    """An item that is not indexed is still reported as absent."""
    client = _client({"OTHER999#0": {"item_key": "OTHER999"}})

    assert client.get_document_metadata("ABCD1234") is None
