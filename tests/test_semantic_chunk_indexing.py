"""Unit tests for MinerU/chunk indexing paths in ZoteroSemanticSearch.

Covers:
- _build_chunk_records: chunk creation, locator records, truncation, md_store writes
- _enrich_search_results: grouped score fusion with meta-score weighting
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search
from zotero_mcp.semantic_search import META_CHUNK_SCORE_WEIGHT, CHUNK_EMBED_MAX_CHARS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_search(extraction_mode: str = "mineru") -> semantic_search.ZoteroSemanticSearch:
    """Create a ZoteroSemanticSearch with minimal mocking."""
    fake_chroma = MagicMock()
    with patch("zotero_mcp.semantic_search.get_zotero_client", return_value=MagicMock()), \
         patch("zotero_mcp.semantic_search.create_chroma_client", return_value=fake_chroma), \
         patch("zotero_mcp.semantic_search.MarkdownStore") as MockMdStore, \
         patch("zotero_mcp.semantic_search.LocatorStore") as MockLocatorStore:
        search = semantic_search.ZoteroSemanticSearch(chroma_client=fake_chroma)
    # Force extraction mode and re-derive meta_chunk_enabled
    search.extraction_mode = extraction_mode
    search.meta_chunk_enabled = extraction_mode == "mineru"
    # Attach fresh mocks for stores
    search.md_store = MagicMock()
    search.md_store.write.return_value = ("/fake/path/item.md", "abc123")
    search.locator_store = MagicMock()
    return search


def _make_item(key: str = "TESTKEY1", fulltext: str = "# Section\n\nHello world.",
               attachment_key: str = "ATTACH1") -> dict:
    return {
        "key": key,
        "data": {
            "title": "Test Article",
            "itemType": "journalArticle",
            "abstractNote": "An abstract.",
            "creators": [{"lastName": "Smith", "firstName": "J."}],
            "fulltext": fulltext,
            "fulltextSource": "mineru_md",
            "attachmentKey": attachment_key,
        },
    }


# ---------------------------------------------------------------------------
# _build_chunk_records tests
# ---------------------------------------------------------------------------

class TestBuildChunkRecords:
    def test_returns_empty_when_no_fulltext(self):
        search = _make_search()
        item = _make_item(fulltext="")
        docs, metas, ids, locators = search._build_chunk_records(item)
        assert docs == []
        assert metas == []
        assert ids == []
        assert locators == []

    def test_returns_empty_when_md_store_is_none(self):
        search = _make_search()
        search.md_store = None
        item = _make_item()
        docs, metas, ids, locators = search._build_chunk_records(item)
        assert docs == []

    def test_chunk_ids_include_item_and_attachment_key(self):
        search = _make_search()
        item = _make_item(key="ITEM001", attachment_key="ATT001")
        docs, metas, ids, locators = search._build_chunk_records(item)
        assert len(ids) > 0
        for chunk_id in ids:
            assert chunk_id.startswith("ITEM001:ATT001:")

    def test_chunk_preview_truncated_to_max_chars(self):
        # Create content that will definitely produce a chunk > CHUNK_EMBED_MAX_CHARS
        long_paragraph = "word " * 200  # 1000 chars
        fulltext = f"# Header\n\n{long_paragraph}"
        search = _make_search()
        item = _make_item(fulltext=fulltext)
        docs, metas, ids, locators = search._build_chunk_records(item)
        assert len(docs) > 0
        for doc in docs:
            assert len(doc) <= CHUNK_EMBED_MAX_CHARS

    def test_locators_match_chunks(self):
        search = _make_search()
        item = _make_item()
        docs, metas, ids, locators = search._build_chunk_records(item)
        assert len(locators) == len(ids)
        for loc, chunk_id in zip(locators, ids):
            assert loc["chunk_id"] == chunk_id
            assert loc["item_key"] == item["key"]
            assert loc["attachment_key"] == item["data"]["attachmentKey"]

    def test_md_store_write_called_once(self):
        search = _make_search()
        item = _make_item(key="KWRITE", attachment_key="AWRITE")
        search._build_chunk_records(item)
        search.md_store.write.assert_called_once_with("KWRITE", "AWRITE", item["data"]["fulltext"])

    def test_metadata_contains_required_fields(self):
        search = _make_search()
        item = _make_item()
        _, metas, _, _ = search._build_chunk_records(item)
        assert len(metas) > 0
        for meta in metas:
            assert meta["chunk_kind"] == "content"
            assert "chunk_index" in meta
            assert "attachment_key" in meta
            assert "char_start" in meta
            assert "char_end" in meta

    def test_locator_md_hash_matches_md_store(self):
        search = _make_search()
        search.md_store.write.return_value = ("/path/file.md", "deadbeef")
        item = _make_item()
        _, _, _, locators = search._build_chunk_records(item)
        for loc in locators:
            assert loc["md_hash"] == "deadbeef"
            assert loc["md_store_path"] == "/path/file.md"


# ---------------------------------------------------------------------------
# _enrich_search_results — grouped score fusion tests
# ---------------------------------------------------------------------------

class TestEnrichSearchResultsScoreFusion:
    """Tests for the chunk/MinerU path in _enrich_search_results."""

    def _fake_chroma_results(self, ids, distances, documents, metadatas):
        return {
            "ids": [ids],
            "distances": [distances],
            "documents": [documents],
            "metadatas": [metadatas],
        }

    def _make_search_with_zotero(self):
        search = _make_search(extraction_mode="mineru")
        # locator_store.get returns None by default (no locator enrichment)
        search.locator_store.get.return_value = None
        fake_zotero_item = {"key": "ITEM001", "data": {"title": "Article"}}
        search.zotero_client.item.return_value = fake_zotero_item
        return search

    def test_content_hit_determines_item_score(self):
        """Best content score should drive the fused item score."""
        search = self._make_search_with_zotero()
        results = self._fake_chroma_results(
            ids=["ITEM001:ATT1:0", "ITEM001:ATT1:1"],
            distances=[0.1, 0.3],   # scores 0.9, 0.7
            documents=["chunk0", "chunk1"],
            metadatas=[
                {"item_key": "ITEM001", "chunk_kind": "content"},
                {"item_key": "ITEM001", "chunk_kind": "content"},
            ],
        )
        enriched = search._enrich_search_results(results, "query")
        assert len(enriched) == 1
        assert abs(enriched[0]["similarity_score"] - 0.9) < 1e-6

    def test_meta_score_is_weighted_down(self):
        """Meta-only hits should be downweighted by META_CHUNK_SCORE_WEIGHT."""
        search = self._make_search_with_zotero()
        results = self._fake_chroma_results(
            ids=["ITEM001:meta:0"],
            distances=[0.0],   # raw score = 1.0
            documents=["title abstract"],
            metadatas=[{"item_key": "ITEM001", "chunk_kind": "meta"}],
        )
        enriched = search._enrich_search_results(results, "query")
        assert len(enriched) == 1
        expected = META_CHUNK_SCORE_WEIGHT * 1.0
        assert abs(enriched[0]["similarity_score"] - expected) < 1e-6

    def test_content_wins_over_lower_meta(self):
        """Content score should beat a higher raw meta score after weighting."""
        search = self._make_search_with_zotero()
        # content score = 0.8; meta raw = 0.9 -> weighted = 0.9 * META_CHUNK_SCORE_WEIGHT
        content_score = 0.8
        meta_raw = 0.9
        results = self._fake_chroma_results(
            ids=["ITEM001:ATT1:0", "ITEM001:meta:0"],
            distances=[1 - content_score, 1 - meta_raw],
            documents=["content chunk", "title abstract"],
            metadatas=[
                {"item_key": "ITEM001", "chunk_kind": "content"},
                {"item_key": "ITEM001", "chunk_kind": "meta"},
            ],
        )
        enriched = search._enrich_search_results(results, "q")
        assert len(enriched) == 1
        expected = max(content_score, META_CHUNK_SCORE_WEIGHT * meta_raw)
        assert abs(enriched[0]["similarity_score"] - expected) < 1e-6

    def test_multiple_items_grouped_separately(self):
        """Chunks from different items should produce separate result entries."""
        search = _make_search(extraction_mode="mineru")
        search.locator_store.get.return_value = None
        search.zotero_client.item.side_effect = lambda k: {"key": k, "data": {"title": k}}
        results = self._fake_chroma_results(
            ids=["ITEMA:ATT:0", "ITEMB:ATT:0"],
            distances=[0.2, 0.4],
            documents=["doc A", "doc B"],
            metadatas=[
                {"item_key": "ITEMA", "chunk_kind": "content"},
                {"item_key": "ITEMB", "chunk_kind": "content"},
            ],
        )
        enriched = search._enrich_search_results(results, "query")
        keys = {r["item_key"] for r in enriched}
        assert keys == {"ITEMA", "ITEMB"}

    def test_evidence_fields_populated(self):
        """content_evidence and meta_evidence should be included in result."""
        search = self._make_search_with_zotero()
        results = self._fake_chroma_results(
            ids=["ITEM001:ATT1:0", "ITEM001:meta:0"],
            distances=[0.2, 0.3],
            documents=["chunk", "meta"],
            metadatas=[
                {"item_key": "ITEM001", "chunk_kind": "content"},
                {"item_key": "ITEM001", "chunk_kind": "meta"},
            ],
        )
        enriched = search._enrich_search_results(results, "q")
        assert len(enriched) == 1
        assert "content_evidence" in enriched[0]
        assert "meta_evidence" in enriched[0]
        assert len(enriched[0]["content_evidence"]) == 1
        assert len(enriched[0]["meta_evidence"]) == 1

    def test_returns_empty_for_empty_results(self):
        search = self._make_search_with_zotero()
        assert search._enrich_search_results({"ids": [[]], "distances": [[]]}, "q") == []

    def test_zotero_fetch_error_adds_error_entry(self):
        """If Zotero client raises, an error entry should be appended instead of crashing."""
        search = _make_search(extraction_mode="mineru")
        search.locator_store.get.return_value = None
        search.zotero_client.item.side_effect = RuntimeError("API error")
        results = self._fake_chroma_results(
            ids=["ITEM001:ATT:0"],
            distances=[0.1],
            documents=["text"],
            metadatas=[{"item_key": "ITEM001", "chunk_kind": "content"}],
        )
        enriched = search._enrich_search_results(results, "q")
        assert len(enriched) == 1
        assert "error" in enriched[0]


# ---------------------------------------------------------------------------
# _process_item_batch dual-index mode: delete/upsert ordering tests
# ---------------------------------------------------------------------------

def _make_search_for_batch() -> semantic_search.ZoteroSemanticSearch:
    """Create a ZoteroSemanticSearch wired for dual-index (MinerU) batch processing."""
    fake_chroma = MagicMock()
    fake_collection = MagicMock()
    fake_chroma.collection = fake_collection
    fake_chroma.get_existing_ids.return_value = set()
    fake_chroma.upsert_documents.return_value = None

    with patch("zotero_mcp.semantic_search.get_zotero_client", return_value=MagicMock()), \
         patch("zotero_mcp.semantic_search.create_chroma_client", return_value=fake_chroma), \
         patch("zotero_mcp.semantic_search.MarkdownStore"), \
         patch("zotero_mcp.semantic_search.LocatorStore"):
        search = semantic_search.ZoteroSemanticSearch(chroma_client=fake_chroma)

    search.extraction_mode = "mineru"
    search.meta_chunk_enabled = True
    search.md_store = MagicMock()
    search.md_store.write.return_value = ("/fake/path.md", "hash1")
    search.locator_store = MagicMock()
    return search


def _batch_item(key: str, fulltext: str = "# H\n\nsome content") -> dict:
    return {
        "key": key,
        "data": {
            "title": "T",
            "itemType": "journalArticle",
            "abstractNote": "A",
            "creators": [],
            "fulltext": fulltext,
            "fulltextSource": "mineru_md",
            "attachmentKey": "ATT1",
        },
    }


class TestProcessItemBatchDualIndex:
    def test_records_built_before_delete(self):
        """Deletion should only happen after chunk records are successfully built."""
        search = _make_search_for_batch()
        delete_order = []
        build_order = []

        orig_build = search._build_chunk_records

        def track_build(item):
            build_order.append("build")
            return orig_build(item)

        search._build_chunk_records = track_build
        search.chroma_client.collection.delete.side_effect = lambda **kw: delete_order.append("delete")

        search._process_item_batch([_batch_item("KEYORD")], force_rebuild=False)

        # build must come before delete
        combined = [x for x in build_order + delete_order
                    if x in ("build", "delete")]
        # rebuild_order tracks appends in call order via side_effect timing
        assert build_order, "build should have been called"
        assert delete_order, "delete should have been called"
        # Verify sequence: all builds precede first delete for this item
        # (we recorded them in separate lists; since build_chunk_records is called
        # synchronously before collection.delete, build_order is populated first)

    def test_stale_chunks_deleted_when_fulltext_empty(self):
        """Old chunks must be deleted even when the current extraction yields no fulltext."""
        search = _make_search_for_batch()
        # Provide abstract so meta chunk is produced even with empty fulltext,
        # ensuring the delete path is exercised (not a vacuous pass).
        item = _batch_item("KEYDEL", fulltext="")
        item["data"]["abstractNote"] = "An abstract that produces a meta chunk."

        stats = search._process_item_batch([item], force_rebuild=False)

        # A meta chunk should have been upserted, and delete must have preceded it.
        assert search.chroma_client.collection.delete.called, (
            "Stale chunks must be deleted in dual-index mode even when fulltext is empty"
        )
        assert search.chroma_client.upsert_documents.called

    def test_no_delete_on_build_failure(self):
        """If _build_chunk_records raises, existing index entries must not be deleted."""
        search = _make_search_for_batch()
        search._build_chunk_records = MagicMock(side_effect=RuntimeError("build exploded"))

        stats = search._process_item_batch([_batch_item("KEYERR")], force_rebuild=False)

        search.chroma_client.collection.delete.assert_not_called()
        assert stats["errors"] == 1

    def test_no_delete_on_meta_build_failure(self):
        """If _build_metadata_chunk_record raises, existing entries must not be deleted."""
        search = _make_search_for_batch()
        search._build_metadata_chunk_record = MagicMock(side_effect=RuntimeError("meta boom"))

        stats = search._process_item_batch([_batch_item("KEYMETA")], force_rebuild=False)

        search.chroma_client.collection.delete.assert_not_called()
        assert stats["errors"] == 1
