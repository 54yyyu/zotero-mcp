import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search


class FakeChromaClient:
    def __init__(self):
        self.upserted_ids = []

    def get_existing_ids(self, ids):
        # Pretend item A already exists and item B is new.
        return {"ITEMA001"} & set(ids)

    def upsert_documents(self, documents, metadatas, ids):
        self.upserted_ids.extend(ids)


def test_process_item_batch_tracks_added_vs_updated(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())

    items = [
        {
            "key": "ITEMA001",
            "data": {
                "title": "Existing Item",
                "itemType": "journalArticle",
                "abstractNote": "A",
                "creators": [],
            },
        },
        {
            "key": "ITEMB002",
            "data": {
                "title": "New Item",
                "itemType": "journalArticle",
                "abstractNote": "B",
                "creators": [],
            },
        },
    ]

    stats = search._process_item_batch(items, force_rebuild=False)

    assert stats["processed"] == 2
    assert stats["updated"] == 1
    assert stats["added"] == 1
