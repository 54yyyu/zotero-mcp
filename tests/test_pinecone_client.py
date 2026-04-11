"""Tests for the Pinecone client and vector store abstraction layer."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pinecone_match(id: str, score: float, metadata: dict):
    """Create a mock Pinecone match object."""
    match = MagicMock()
    match.id = id
    match.score = score
    match.metadata = metadata
    return match


def _make_query_response(matches):
    resp = MagicMock()
    resp.matches = matches
    return resp


def _make_fetch_response(vectors_dict):
    """Create a mock Pinecone fetch response."""
    resp = MagicMock()
    resp.vectors = vectors_dict
    return resp


def _make_index_stats(namespace_name, vector_count):
    """Create a mock describe_index_stats response."""
    ns = MagicMock()
    ns.vector_count = vector_count
    stats = MagicMock()
    stats.namespaces = {namespace_name: ns}
    return stats


# ---------------------------------------------------------------------------
# vector_store module tests
# ---------------------------------------------------------------------------


class TestVectorStoreFactory:
    """Tests for vector_store.get_vector_backend and create_vector_client."""

    def test_default_backend_is_chroma(self, tmp_path):
        from zotero_mcp.vector_store import get_vector_backend

        # No config, no env → chroma
        assert get_vector_backend(None) == "chroma"
        assert get_vector_backend(str(tmp_path / "nonexistent.json")) == "chroma"

    def test_env_var_overrides_config(self, tmp_path):
        from zotero_mcp.vector_store import get_vector_backend

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"semantic_search": {"vector_backend": "chroma"}}))

        with patch.dict(os.environ, {"ZOTERO_VECTOR_BACKEND": "pinecone"}):
            assert get_vector_backend(str(config_file)) == "pinecone"

    def test_config_file_backend(self, tmp_path):
        from zotero_mcp.vector_store import get_vector_backend

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"semantic_search": {"vector_backend": "pinecone"}}))

        with patch.dict(os.environ, {}, clear=False):
            # Remove env var if set
            os.environ.pop("ZOTERO_VECTOR_BACKEND", None)
            assert get_vector_backend(str(config_file)) == "pinecone"

    def test_unknown_backend_raises(self, tmp_path):
        from zotero_mcp.vector_store import create_vector_client

        with patch.dict(os.environ, {"ZOTERO_VECTOR_BACKEND": "weaviate"}):
            with pytest.raises(ValueError, match="Unknown vector backend"):
                create_vector_client(None)


# ---------------------------------------------------------------------------
# PineconeClient tests (mocked — no real Pinecone calls)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pinecone_env():
    """Set required env vars and patch the Pinecone SDK."""
    with patch.dict(os.environ, {"PINECONE_API_KEY": "test-key-1234"}):
        yield


@pytest.fixture
def mock_pc_and_index(mock_pinecone_env):
    """Patch Pinecone SDK so PineconeClient can be instantiated."""
    mock_index = MagicMock()
    mock_pc_instance = MagicMock()

    # list_indexes returns existing index
    idx_obj = MagicMock()
    idx_obj.name = "zotero-library"
    mock_pc_instance.list_indexes.return_value = [idx_obj]
    mock_pc_instance.Index.return_value = mock_index
    mock_pc_instance.describe_index.return_value = MagicMock(status={"ready": True})

    with (
        patch("pinecone.Pinecone", return_value=mock_pc_instance) as pc_cls,
        patch("sentence_transformers.SentenceTransformer") as mock_st,
    ):
        # Mock SentenceTransformer for _DefaultEmbedding
        mock_model = MagicMock()
        mock_model.max_seq_length = 256
        import numpy as np

        mock_model.encode.return_value = np.array([[0.1] * 384])
        mock_st.return_value = mock_model

        yield {
            "pc_cls": pc_cls,
            "pc_instance": mock_pc_instance,
            "index": mock_index,
            "st_model": mock_model,
        }


def _create_client(mock_pc_and_index):
    """Helper to create a PineconeClient with all deps mocked."""
    from zotero_mcp.pinecone_client import PineconeClient

    client = PineconeClient(embedding_model="default")
    return client


class TestPineconeClientInit:
    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PINECONE_API_KEY", None)
            from zotero_mcp.pinecone_client import PineconeClient

            with pytest.raises(ValueError, match="Pinecone API key is required"):
                PineconeClient()

    def test_creates_client_with_existing_index(self, mock_pc_and_index):
        client = _create_client(mock_pc_and_index)
        assert client.index_name == "zotero-library"
        assert client.namespace == "zotero_library"
        assert client.embedding_model == "default"

    def test_creates_index_if_not_exists(self, mock_pinecone_env):
        mock_index = MagicMock()
        mock_pc_instance = MagicMock()
        mock_pc_instance.list_indexes.return_value = []  # no indexes
        mock_pc_instance.Index.return_value = mock_index
        desc = MagicMock()
        desc.status = {"ready": True}
        mock_pc_instance.describe_index.return_value = desc

        with (
            patch("pinecone.Pinecone", return_value=mock_pc_instance),
            patch("pinecone.ServerlessSpec"),
            patch("sentence_transformers.SentenceTransformer") as mock_st,
        ):
            import numpy as np

            mock_model = MagicMock()
            mock_model.max_seq_length = 256
            mock_model.encode.return_value = np.array([[0.1] * 384])
            mock_st.return_value = mock_model

            from zotero_mcp.pinecone_client import PineconeClient

            PineconeClient(embedding_model="default")

            mock_pc_instance.create_index.assert_called_once()
            call_kwargs = mock_pc_instance.create_index.call_args
            assert call_kwargs.kwargs["dimension"] == 384
            assert call_kwargs.kwargs["metric"] == "cosine"


class TestPineconeClientOperations:
    def test_upsert_documents(self, mock_pc_and_index):
        client = _create_client(mock_pc_and_index)
        idx = mock_pc_and_index["index"]

        client.upsert_documents(
            documents=["test document one"],
            metadatas=[{"title": "Test", "item_type": "journalArticle"}],
            ids=["key1"],
        )

        idx.upsert.assert_called_once()

    def test_search_returns_chroma_compatible_format(self, mock_pc_and_index):
        client = _create_client(mock_pc_and_index)
        idx = mock_pc_and_index["index"]

        idx.query.return_value = _make_query_response(
            [
                _make_pinecone_match("k1", 0.95, {"title": "Paper A", "_document_text": "Full text A"}),
                _make_pinecone_match("k2", 0.80, {"title": "Paper B", "_document_text": "Full text B"}),
            ]
        )

        results = client.search(query_texts=["machine learning"], n_results=2)

        assert "ids" in results
        assert "distances" in results
        assert "documents" in results
        assert "metadatas" in results

        # Single query → single list
        assert len(results["ids"]) == 1
        assert results["ids"][0] == ["k1", "k2"]

        # Distances = 1 - score (for compatibility with semantic_search.py)
        assert abs(results["distances"][0][0] - 0.05) < 0.001
        assert abs(results["distances"][0][1] - 0.20) < 0.001

        # Document text extracted from metadata
        assert results["documents"][0] == ["Full text A", "Full text B"]

        # _document_text should be removed from returned metadata
        assert "_document_text" not in results["metadatas"][0][0]
        assert results["metadatas"][0][0]["title"] == "Paper A"

    def test_search_with_filter(self, mock_pc_and_index):
        client = _create_client(mock_pc_and_index)
        idx = mock_pc_and_index["index"]
        idx.query.return_value = _make_query_response([])

        client.search(
            query_texts=["test"],
            n_results=5,
            where={"item_type": "journalArticle"},
        )

        call_kwargs = idx.query.call_args.kwargs
        assert call_kwargs["filter"] == {"item_type": {"$eq": "journalArticle"}}

    def test_delete_documents(self, mock_pc_and_index):
        client = _create_client(mock_pc_and_index)
        idx = mock_pc_and_index["index"]

        client.delete_documents(["k1", "k2"])
        idx.delete.assert_called_once_with(ids=["k1", "k2"], namespace="zotero_library")

    def test_reset_collection(self, mock_pc_and_index):
        client = _create_client(mock_pc_and_index)
        idx = mock_pc_and_index["index"]

        client.reset_collection()
        idx.delete.assert_called_once_with(delete_all=True, namespace="zotero_library")

    def test_document_exists(self, mock_pc_and_index):
        client = _create_client(mock_pc_and_index)
        idx = mock_pc_and_index["index"]

        vec = MagicMock()
        vec.metadata = {"title": "Test"}
        idx.fetch.return_value = _make_fetch_response({"k1": vec})

        assert client.document_exists("k1") is True
        assert client.document_exists("k_missing") is False

    def test_get_document_metadata(self, mock_pc_and_index):
        client = _create_client(mock_pc_and_index)
        idx = mock_pc_and_index["index"]

        vec = MagicMock()
        vec.metadata = {"title": "Hello", "_document_text": "full text here"}
        idx.fetch.return_value = _make_fetch_response({"k1": vec})

        meta = client.get_document_metadata("k1")
        assert meta is not None
        assert meta["title"] == "Hello"
        assert "_document_text" not in meta

    def test_get_existing_ids(self, mock_pc_and_index):
        client = _create_client(mock_pc_and_index)
        idx = mock_pc_and_index["index"]

        vec1 = MagicMock()
        vec2 = MagicMock()
        idx.fetch.return_value = _make_fetch_response({"k1": vec1, "k3": vec2})

        existing = client.get_existing_ids(["k1", "k2", "k3"])
        assert existing == {"k1", "k3"}

    def test_get_collection_info(self, mock_pc_and_index):
        client = _create_client(mock_pc_and_index)
        idx = mock_pc_and_index["index"]

        idx.describe_index_stats.return_value = _make_index_stats("zotero_library", 42)

        info = client.get_collection_info()
        assert info["count"] == 42
        assert info["backend"] == "pinecone"
        assert info["name"] == "zotero_library"


class TestFilterTranslation:
    def test_simple_equality(self):
        from zotero_mcp.pinecone_client import _translate_filter

        result = _translate_filter({"item_type": "journalArticle"})
        assert result == {"item_type": {"$eq": "journalArticle"}}

    def test_operator_passthrough(self):
        from zotero_mcp.pinecone_client import _translate_filter

        result = _translate_filter({"year": {"$gte": 2020}})
        assert result == {"year": {"$gte": 2020}}

    def test_logical_operators(self):
        from zotero_mcp.pinecone_client import _translate_filter

        result = _translate_filter(
            {
                "$and": [
                    {"item_type": "journalArticle"},
                    {"year": {"$gte": 2020}},
                ]
            }
        )
        assert result == {
            "$and": [
                {"item_type": {"$eq": "journalArticle"}},
                {"year": {"$gte": 2020}},
            ]
        }

    def test_none_returns_none(self):
        from zotero_mcp.pinecone_client import _translate_filter

        assert _translate_filter(None) is None


class TestSanitizeMetadata:
    def test_removes_none_values(self):
        from zotero_mcp.pinecone_client import _sanitize_metadata

        result = _sanitize_metadata({"a": "hello", "b": None, "c": 42})
        assert result == {"a": "hello", "c": 42}

    def test_converts_unsupported_types(self):
        from zotero_mcp.pinecone_client import _sanitize_metadata

        result = _sanitize_metadata({"a": {"nested": "dict"}})
        assert isinstance(result["a"], str)

    def test_keeps_valid_types(self):
        from zotero_mcp.pinecone_client import _sanitize_metadata

        original = {"s": "str", "i": 1, "f": 1.5, "b": True, "l": ["a", "b"]}
        result = _sanitize_metadata(original)
        assert result == original


class TestEmbeddingDimensions:
    def test_known_dimensions(self, mock_pc_and_index):
        client = _create_client(mock_pc_and_index)
        # default model → 384
        assert client._get_embedding_dimension() == 384

    def test_openai_large_variant(self, mock_pc_and_index):
        client = _create_client(mock_pc_and_index)
        client.embedding_model = "openai"
        client.embedding_config = {"model_name": "text-embedding-3-large"}
        assert client._get_embedding_dimension() == 3072


class TestCreatePineconeClient:
    def test_from_config_file(self, tmp_path, mock_pc_and_index):
        config = {
            "semantic_search": {
                "collection_name": "my_collection",
                "embedding_model": "default",
                "vector_backend": "pinecone",
                "pinecone": {
                    "index_name": "my-index",
                    "cloud": "gcp",
                    "region": "us-central1",
                },
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        from zotero_mcp.pinecone_client import create_pinecone_client

        client = create_pinecone_client(str(config_file))

        assert client.collection_name == "my_collection"
        assert client.namespace == "my_collection"
