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
        self.embedding_max_tokens = 8000

    def get_existing_ids(self, ids):
        # Pretend item A already exists and item B is new.
        return {"ITEMA001"} & set(ids)

    def upsert_documents(self, documents, metadatas, ids):
        self.upserted_ids.extend(ids)

    def truncate_text(self, text, max_tokens=None):
        return text


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


class FailingChromaClient(FakeChromaClient):
    """Fakes a ChromaDB client whose upsert raises a transient error.

    Exercises the deferred-retry path in _process_item_batch.
    """

    def upsert_documents(self, documents, metadatas, ids):
        raise RuntimeError("simulated transient ChromaDB upsert failure")


def test_process_item_batch_defers_failures_when_failed_docs_provided(monkeypatch):
    """Failed batches should append to _failed_docs and bump errors instead of crashing.

    Before the _failed_docs scope fix, this path raised NameError because
    `_failed_docs` was referenced inside `_process_item_batch` but only
    defined in the caller (`update_database`). Every transient ChromaDB
    error therefore crashed the entire reindex.
    """
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    search = semantic_search.ZoteroSemanticSearch(chroma_client=FailingChromaClient())

    items = [
        {
            "key": f"ITEM{i:03d}",
            "data": {
                "title": f"Item {i}",
                "itemType": "journalArticle",
                "abstractNote": "x",
                "creators": [],
            },
        }
        for i in range(3)
    ]

    failed_docs = []
    stats = search._process_item_batch(items, force_rebuild=False, _failed_docs=failed_docs)

    # All 3 items prepared successfully (text built, truncated, queued)
    assert stats["processed"] == 3
    # All 3 items deferred to the retry list when the upsert raised
    assert len(failed_docs) == 3
    assert {doc_id for _, _, doc_id in failed_docs} == {"ITEM000", "ITEM001", "ITEM002"}
    # Errors bumped by the deferred count so the totals stay accurate
    assert stats["errors"] == 3
    # No items classified as added/updated because the batch never reached
    # the existing-id lookup
    assert stats["added"] == 0
    assert stats["updated"] == 0


def test_process_item_batch_reraises_when_no_retry_list(monkeypatch):
    """Without `_failed_docs`, the batch failure path must re-raise.

    Legacy callers (or test fixtures) that don't opt into deferred retry
    should see the real exception instead of silent swallowing.
    """
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    search = semantic_search.ZoteroSemanticSearch(chroma_client=FailingChromaClient())

    items = [
        {
            "key": "ITEM000",
            "data": {
                "title": "Item",
                "itemType": "journalArticle",
                "abstractNote": "x",
                "creators": [],
            },
        }
    ]

    with pytest.raises(RuntimeError, match="simulated transient"):
        search._process_item_batch(items, force_rebuild=False)


def test_update_database_stats_includes_recovered_items_field(monkeypatch):
    """The stats dict initialized by update_database must include recovered_items.

    Pinning the field name so future refactors don't silently drop it
    from the stats schema. The retry loop relies on this field existing.
    """
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())

    # Force update_database to short-circuit by feeding an empty source.
    monkeypatch.setattr(
        search, "_get_items_from_source", lambda **kwargs: []
    )
    monkeypatch.setattr(search, "_save_update_config", lambda: None)

    stats = search.update_database()

    assert "recovered_items" in stats
    assert stats["recovered_items"] == 0
