"""Tests for --fulltext + ZOTERO_LOCAL interaction."""

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
        self.embedding_max_tokens = 8000

    def get_existing_ids(self, ids):
        return set()

    def upsert_documents(self, documents, metadatas, ids):
        pass

    def reset_collection(self):
        pass


def test_get_items_from_source_aborts_when_fulltext_without_local_mode(monkeypatch):
    """Requesting fulltext extraction without local mode should raise RuntimeError."""
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())

    with pytest.raises(RuntimeError, match="ZOTERO_LOCAL"):
        search._get_items_from_source(extract_fulltext=True)


def test_get_items_from_source_proceeds_when_fulltext_with_local_mode(monkeypatch):
    """Requesting fulltext extraction with local mode should not raise."""
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)

    # Mock _get_items_from_local_db to avoid needing a real DB
    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())
    monkeypatch.setattr(search, "_get_items_from_local_db", lambda *a, **kw: [])

    # Should not raise
    result = search._get_items_from_source(extract_fulltext=True)
    assert result == []


def test_get_items_from_source_no_abort_without_fulltext(monkeypatch):
    """Without --fulltext, should proceed regardless of local mode."""
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)

    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())
    monkeypatch.setattr(search, "_get_items_from_api", lambda *a, **kw: [])

    # Should not raise — fulltext not requested
    result = search._get_items_from_source(extract_fulltext=False)
    assert result == []
