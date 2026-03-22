"""Tests for semantic search quality improvements.

Covers:
- Fix 1: Structured fields combined with fulltext
- Fix 2: Gemini query/document embedding asymmetry
- Fix 3: Model-aware tokenizer truncation
- Fix 5: Cross-encoder re-ranking
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeChromaClient:
    """Minimal ChromaClient stub for unit tests."""

    def __init__(self):
        self.upserted_docs = []
        self.upserted_ids = []
        self.embedding_max_tokens = 8000
        self._last_query_kwargs = None

    def get_existing_ids(self, ids):
        return set()

    def upsert_documents(self, documents, metadatas, ids):
        self.upserted_docs.extend(documents)
        self.upserted_ids.extend(ids)

    def truncate_text(self, text, max_tokens=None):
        # Pass-through for tests (no actual truncation)
        return text

    def search(self, query_texts=None, n_results=10, where=None, where_document=None):
        self._last_query_kwargs = {
            "query_texts": query_texts,
            "n_results": n_results,
            "where": where,
        }
        return {
            "ids": [["ID1", "ID2", "ID3"]],
            "distances": [[0.1, 0.3, 0.2]],
            "documents": [["doc about cats", "doc about dogs", "doc about birds"]],
            "metadatas": [[{"title": "Cats"}, {"title": "Dogs"}, {"title": "Birds"}]],
        }


def _make_item(key, title="Test", abstract="Abstract", fulltext="", creators=None):
    """Create a minimal Zotero item dict."""
    return {
        "key": key,
        "data": {
            "title": title,
            "itemType": "journalArticle",
            "abstractNote": abstract,
            "creators": creators or [],
            "fulltext": fulltext,
        },
    }


# ---------------------------------------------------------------------------
# Fix 1: Combine structured fields + fulltext
# ---------------------------------------------------------------------------

class TestCombineStructuredAndFulltext:
    def _make_search(self):
        with patch.object(semantic_search, "get_zotero_client", return_value=object()):
            return semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())

    def test_fulltext_prepended_with_structured_fields(self):
        search = self._make_search()
        item = _make_item("K1", title="My Title", abstract="My Abstract", fulltext="Full paper text here.")
        stats = search._process_item_batch([item], force_rebuild=True)

        assert stats["processed"] == 1
        doc = search.chroma_client.upserted_docs[0]
        # Structured fields should appear before fulltext
        assert doc.index("My Title") < doc.index("Full paper text here.")
        assert doc.index("My Abstract") < doc.index("Full paper text here.")

    def test_no_fulltext_uses_structured_only(self):
        search = self._make_search()
        item = _make_item("K2", title="Only Structured", abstract="Just abstract")
        stats = search._process_item_batch([item], force_rebuild=True)

        assert stats["processed"] == 1
        doc = search.chroma_client.upserted_docs[0]
        assert "Only Structured" in doc
        assert "Just abstract" in doc

    def test_empty_structured_with_fulltext(self):
        search = self._make_search()
        item = _make_item("K3", title="", abstract="", fulltext="Only fulltext content.")
        stats = search._process_item_batch([item], force_rebuild=True)

        assert stats["processed"] == 1
        doc = search.chroma_client.upserted_docs[0]
        assert "Only fulltext content." in doc


# ---------------------------------------------------------------------------
# Fix 2: Gemini query/document embedding asymmetry
# ---------------------------------------------------------------------------

class TestGeminiQueryEmbedding:
    def test_gemini_embed_query_uses_retrieval_query(self):
        """Verify GeminiEmbeddingFunction.embed_query passes retrieval_query task type."""
        from zotero_mcp.chroma_client import GeminiEmbeddingFunction

        mock_client = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_response

        mock_types = MagicMock()

        ef = GeminiEmbeddingFunction.__new__(GeminiEmbeddingFunction)
        ef.model_name = "gemini-embedding-001"
        ef.client = mock_client
        ef.types = mock_types

        result = ef.embed_query("test query")

        # Verify embed_content was called
        mock_client.models.embed_content.assert_called_once()
        call_kwargs = mock_client.models.embed_content.call_args
        config_arg = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        # Verify the task_type was retrieval_query
        mock_types.EmbedContentConfig.assert_called_once_with(task_type="retrieval_query")
        assert result == [0.1, 0.2, 0.3]

    def test_gemini_call_uses_retrieval_document(self):
        """Verify GeminiEmbeddingFunction.__call__ still uses retrieval_document."""
        from zotero_mcp.chroma_client import GeminiEmbeddingFunction

        mock_client = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.4, 0.5, 0.6]
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_response

        mock_types = MagicMock()

        ef = GeminiEmbeddingFunction.__new__(GeminiEmbeddingFunction)
        ef.model_name = "gemini-embedding-001"
        ef.client = mock_client
        ef.types = mock_types

        ef(["some document"])

        mock_types.EmbedContentConfig.assert_called_with(
            task_type="retrieval_document",
            title="Zotero library document",
        )


class TestSearchUsesEmbedQuery:
    def test_search_uses_query_embeddings_for_custom_ef(self):
        """ChromaClient.search should use query_embeddings for custom embedding functions."""
        from zotero_mcp.chroma_client import ChromaClient, HuggingFaceEmbeddingFunction

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "distances": [[0.1]],
            "documents": [["text"]],
            "metadatas": [[{}]],
        }

        # Must be an instance of one of our custom classes for embed_query path
        mock_ef = MagicMock(spec=HuggingFaceEmbeddingFunction)
        mock_ef.embed_query.return_value = [0.1, 0.2, 0.3]

        client = ChromaClient.__new__(ChromaClient)
        client.collection = mock_collection
        client.embedding_function = mock_ef

        client.search(query_texts=["hello"])

        # Should have called embed_query, not passed query_texts
        mock_ef.embed_query.assert_called_once_with("hello")
        call_kwargs = mock_collection.query.call_args.kwargs
        assert "query_embeddings" in call_kwargs
        assert "query_texts" not in call_kwargs

    def test_search_falls_back_to_query_texts(self):
        """ChromaClient.search should use query_texts when embed_query is absent."""
        from zotero_mcp.chroma_client import ChromaClient

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "distances": [[0.1]],
            "documents": [["text"]],
            "metadatas": [[{}]],
        }

        # Embedding function WITHOUT embed_query
        mock_ef = MagicMock(spec=[])  # empty spec = no attributes

        client = ChromaClient.__new__(ChromaClient)
        client.collection = mock_collection
        client.embedding_function = mock_ef

        client.search(query_texts=["hello"])

        call_kwargs = mock_collection.query.call_args.kwargs
        assert "query_texts" in call_kwargs
        assert call_kwargs["query_texts"] == ["hello"]


class TestDefaultEFUsesQueryTexts:
    """Verify that DefaultEmbeddingFunction (or any non-custom EF) uses query_texts,
    not embed_query.  This confirms the _is_custom_ef guard prevents calling the
    broken embed_query on ChromaDB's built-in DefaultEmbeddingFunction."""

    def test_default_ef_uses_query_texts_not_embed_query(self):
        from zotero_mcp.chroma_client import ChromaClient

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "distances": [[0.1]],
            "documents": [["text"]],
            "metadatas": [[{}]],
        }

        # A plain MagicMock is NOT an instance of any custom embedding class,
        # mimicking DefaultEmbeddingFunction which also has an embed_query attr.
        mock_ef = MagicMock()
        mock_ef.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])

        client = ChromaClient.__new__(ChromaClient)
        client.collection = mock_collection
        client.embedding_function = mock_ef

        client.search(query_texts=["hello"])

        # embed_query must NOT have been called — default EF path uses query_texts
        mock_ef.embed_query.assert_not_called()

        call_kwargs = mock_collection.query.call_args.kwargs
        assert "query_texts" in call_kwargs
        assert call_kwargs["query_texts"] == ["hello"]
        assert "query_embeddings" not in call_kwargs


# ---------------------------------------------------------------------------
# Fix 3: Model-aware tokenizer
# ---------------------------------------------------------------------------

class TestModelAwareTokenizer:
    def test_openai_truncate_uses_tiktoken(self):
        from zotero_mcp.chroma_client import OpenAIEmbeddingFunction

        ef = OpenAIEmbeddingFunction.__new__(OpenAIEmbeddingFunction)
        # Long text that should be truncated to 5 tokens
        text = "This is a longer text that should be truncated by tiktoken"
        result = ef.truncate(text, max_tokens=5)
        # Result should be shorter than original
        assert len(result) < len(text)
        assert len(result) > 0

    def test_gemini_truncate_uses_char_estimation(self):
        from zotero_mcp.chroma_client import GeminiEmbeddingFunction

        ef = GeminiEmbeddingFunction.__new__(GeminiEmbeddingFunction)
        text = "a" * 10000
        result = ef.truncate(text, max_tokens=100)
        # 100 tokens * 4 chars/token = 400 chars
        assert len(result) == 400

    def test_huggingface_truncate_uses_model_tokenizer(self):
        from zotero_mcp.chroma_client import HuggingFaceEmbeddingFunction

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(20))  # 20 tokens
        mock_tokenizer.decode.return_value = "truncated text"

        mock_model = MagicMock()
        mock_model.tokenizer = mock_tokenizer

        ef = HuggingFaceEmbeddingFunction.__new__(HuggingFaceEmbeddingFunction)
        ef.model = mock_model

        result = ef.truncate("some long text", max_tokens=10)

        mock_tokenizer.encode.assert_called_once_with("some long text", add_special_tokens=False)
        mock_tokenizer.decode.assert_called_once_with(list(range(10)))
        assert result == "truncated text"

    def test_chroma_truncate_text_delegates_to_embedding_function(self):
        from zotero_mcp.chroma_client import ChromaClient

        mock_ef = MagicMock()
        mock_ef.truncate.return_value = "truncated"
        mock_ef.max_input_tokens = 500

        client = ChromaClient.__new__(ChromaClient)
        client.embedding_function = mock_ef

        result = client.truncate_text("long text")

        mock_ef.truncate.assert_called_once_with("long text", 500)
        assert result == "truncated"


# ---------------------------------------------------------------------------
# Fix 5: Cross-encoder re-ranking
# ---------------------------------------------------------------------------

class TestReranking:
    def _make_search_with_reranker(self, enabled=True):
        fake_client = FakeChromaClient()
        with patch.object(semantic_search, "get_zotero_client", return_value=MagicMock()):
            s = semantic_search.ZoteroSemanticSearch(chroma_client=fake_client)
        s._reranker_config = {
            "enabled": enabled,
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "candidate_multiplier": 3,
        }
        return s

    def test_reranker_disabled_by_default(self):
        with patch.object(semantic_search, "get_zotero_client", return_value=object()):
            s = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())
        assert s._get_reranker() is None

    def test_reranker_reorders_results(self):
        s = self._make_search_with_reranker(enabled=True)

        # Mock reranker to reverse the order
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [2, 0, 1]  # Birds, Cats, Dogs
        s._reranker = mock_reranker

        # Mock the Zotero client to avoid API calls
        s.zotero_client = MagicMock()
        s.zotero_client.item.return_value = {"data": {"title": "mock"}}

        result = s.search("birds", limit=3)

        # Verify reranker was called
        mock_reranker.rerank.assert_called_once()
        # First result should be "Birds" (index 2 in original)
        assert result["results"][0]["metadata"]["title"] == "Birds"

    def test_search_overfetches_when_reranker_enabled(self):
        s = self._make_search_with_reranker(enabled=True)
        s._reranker = MagicMock()
        s._reranker.rerank.return_value = [0, 1]

        s.zotero_client = MagicMock()
        s.zotero_client.item.return_value = {"data": {"title": "mock"}}

        s.search("test", limit=2)

        # candidate_multiplier=3, so n_results should be 2*3=6
        assert s.chroma_client._last_query_kwargs["n_results"] == 6

    def test_search_without_reranker_uses_original_limit(self):
        s = self._make_search_with_reranker(enabled=False)

        s.zotero_client = MagicMock()
        s.zotero_client.item.return_value = {"data": {"title": "mock"}}

        s.search("test", limit=5)

        assert s.chroma_client._last_query_kwargs["n_results"] == 5
