"""A pass over the whole library asks the index about its items in one read.

The local full-text scan asked ChromaDB about each item on its own: 0.9 ms an
item, on every update, changed or not. What each item is decided to need must
not depend on how the answer was fetched.
"""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from test_failed_marker_retry import DATE_MODIFIED, FakeChromaClient, FakeItem, FakeReader

from zotero_mcp import semantic_search


class ManyItemsReader(FakeReader):
    def __init__(self, keys):
        super().__init__()
        self._keys = keys

    def get_all_item_keys(self):
        return set(self._keys)

    def get_key_group_map(self):
        return ({key: 0 for key in self._keys}, set())

    def get_items_with_text(self, limit=None, include_fulltext=False, key_filter=None, collection_keys=None):
        items = []
        for index, key in enumerate(self._keys, 1):
            item = FakeItem()
            item.item_id, item.key, item.title = index, key, f"Paper {key}"
            items.append(item)
        return items


class BatchChroma(FakeChromaClient):
    """Answers for many items at once, and counts how it was asked."""

    def __init__(self, stored_by_key):
        super().__init__(None)
        self._by_key = stored_by_key
        self.single_lookups = 0
        self.batch_lookups = []

    def get_document_metadata(self, key):
        self.single_lookups += 1
        return self._by_key.get(key)

    def get_metadatas(self, ids):
        self.batch_lookups.append(list(ids))
        return {doc_id: self._by_key[doc_id] for doc_id in ids if doc_id in self._by_key}


class SingleChroma(BatchChroma):
    get_metadatas = None  # a client that can only answer per record


def _scan(monkeypatch, chroma, keys):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    reader = ManyItemsReader(keys)
    monkeypatch.setattr(semantic_search, "LocalZoteroReader", lambda *a, **kw: reader)
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma)
    monkeypatch.setattr(
        search, "_get_items_from_api", lambda *a, **kw: pytest.fail("unexpected fallback to API path")
    )
    return search._get_items_from_source(extract_fulltext=True, chroma_client=chroma, force_rebuild=False)


def _stored():
    current = {"group_id": 0, "date_modified": DATE_MODIFIED, "attachment_keys": ""}
    return {
        "CURRENT1": dict(current),
        "CURRENT2": dict(current),
        "UNTAGGED": {"date_modified": DATE_MODIFIED, "attachment_keys": ""},
    }


KEYS = ["CURRENT1", "NEW", "UNTAGGED", "CURRENT2"]


def test_the_local_scan_asks_about_every_item_in_one_read(monkeypatch):
    chroma = BatchChroma(_stored())

    items = _scan(monkeypatch, chroma, KEYS)

    # One read for the slice: each item's record and, where it may have
    # passages, its first passage (the item-keyed scan reads state per slice).
    assert len(chroma.batch_lookups) == 1
    assert set(KEYS) <= set(chroma.batch_lookups[0])
    assert chroma.single_lookups == 0
    assert [item["key"] for item in items] == ["NEW", "UNTAGGED"]


def test_the_scan_decides_the_same_either_way(monkeypatch):
    batched = _scan(monkeypatch, BatchChroma(_stored()), KEYS)
    single_client = SingleChroma(_stored())
    single = _scan(monkeypatch, single_client, KEYS)

    assert single_client.single_lookups >= len(KEYS)
    assert [item["key"] for item in single] == [item["key"] for item in batched]
