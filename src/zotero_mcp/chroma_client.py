"""
ChromaDB client for semantic search functionality.

This module provides persistent vector database storage and embedding functions
for semantic search over Zotero libraries.
"""

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings

logger = logging.getLogger(__name__)


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout temporarily."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class OpenAIEmbeddingFunction(EmbeddingFunction):
    """Custom OpenAI embedding function for ChromaDB."""
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        try:
            import openai
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = openai.OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError("openai package is required for OpenAI embeddings")
    
    def name(self) -> str:
        """Return the name of this embedding function."""
        return "openai"
    
    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using OpenAI API."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=input
        )
        return [data.embedding for data in response.data]


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom Gemini embedding function for ChromaDB using google-genai."""
    
    def __init__(self, model_name: str = "models/text-embedding-004", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.base_url = base_url or os.getenv("GEMINI_BASE_URL")
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        try:
            from google import genai
            from google.genai import types
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                http_options = types.HttpOptions(baseUrl=self.base_url)
                client_kwargs["http_options"] = http_options
            self.client = genai.Client(**client_kwargs)
            self.types = types
        except ImportError:
            raise ImportError("google-genai package is required for Gemini embeddings")
    
    def name(self) -> str:
        """Return the name of this embedding function."""
        return "gemini"
    
    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using Gemini API."""
        embeddings = []
        for text in input:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=[text],
                config=self.types.EmbedContentConfig(
                    task_type="retrieval_document",
                    title="Zotero library document"
                )
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings


class HuggingFaceEmbeddingFunction(EmbeddingFunction):
    """Custom HuggingFace embedding function for ChromaDB using sentence-transformers."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.model_name = model_name

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        except ImportError:
            raise ImportError("sentence-transformers package is required for HuggingFace embeddings. Install with: pip install sentence-transformers")

    def name(self) -> str:
        """Return the name of this embedding function."""
        return f"huggingface-{self.model_name}"

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using HuggingFace model."""
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()


class ChromaClient:
    """ChromaDB client for Zotero semantic search."""
    
    def __init__(self, 
                 collection_name: str = "zotero_library",
                 persist_directory: Optional[str] = None,
                 embedding_model: str = "default",
                 embedding_config: Optional[Dict[str, Any]] = None):
        """
        Initialize ChromaDB client.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Model to use for embeddings ('default', 'openai', 'gemini', 'qwen', 'embeddinggemma', or HuggingFace model name)
            embedding_config: Configuration for the embedding model
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_config = embedding_config or {}
        
        # Set up persistent directory
        if persist_directory is None:
            # Use user's config directory by default
            config_dir = Path.home() / ".config" / "zotero-mcp"
            config_dir.mkdir(parents=True, exist_ok=True)
            persist_directory = str(config_dir / "chroma_db")
        
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client with stdout suppression
        with suppress_stdout():
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Set up embedding function
            self.embedding_function = self._create_embedding_function()
            
            # Get or create collection with embedding function handling
            try:
                # Try to get existing collection first
                self.collection = self.client.get_collection(name=self.collection_name)
                
                # Check if embedding functions are compatible
                existing_ef = getattr(self.collection, '_embedding_function', None)
                if existing_ef is not None:
                    existing_name = getattr(existing_ef, 'name', lambda: 'default')()
                    new_name = getattr(self.embedding_function, 'name', lambda: 'default')()
                    
                    if existing_name != new_name:
                        # Log to stderr instead of letting ChromaDB print to stdout
                        sys.stderr.write(f"ChromaDB: Collection exists with different embedding function: {existing_name} vs {new_name}\n")
                        # Use the existing collection's embedding function to avoid conflicts
                        self.embedding_function = existing_ef
                
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
    
    def _create_embedding_function(self) -> EmbeddingFunction:
        """Create the appropriate embedding function based on configuration."""
        if self.embedding_model == "openai":
            model_name = self.embedding_config.get("model_name", "text-embedding-3-small")
            api_key = self.embedding_config.get("api_key")
            base_url = self.embedding_config.get("base_url")
            return OpenAIEmbeddingFunction(model_name=model_name, api_key=api_key, base_url=base_url)
        
        elif self.embedding_model == "gemini":
            model_name = self.embedding_config.get("model_name", "models/text-embedding-004")
            api_key = self.embedding_config.get("api_key")
            base_url = self.embedding_config.get("base_url")
            return GeminiEmbeddingFunction(model_name=model_name, api_key=api_key, base_url=base_url)

        elif self.embedding_model == "qwen":
            model_name = self.embedding_config.get("model_name", "Qwen/Qwen3-Embedding-0.6B")
            return HuggingFaceEmbeddingFunction(model_name=model_name)

        elif self.embedding_model == "embeddinggemma":
            model_name = self.embedding_config.get("model_name", "google/embeddinggemma-300m")
            return HuggingFaceEmbeddingFunction(model_name=model_name)

        elif self.embedding_model not in ["default", "openai", "gemini"]:
            # Treat any other value as a HuggingFace model name
            return HuggingFaceEmbeddingFunction(model_name=self.embedding_model)
        
        else:
            # Use ChromaDB's default embedding function (all-MiniLM-L6-v2)
            return chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
    
    def add_documents(self, 
                     documents: List[str], 
                     metadatas: List[Dict[str, Any]], 
                     ids: List[str]) -> None:
        """
        Add documents to the collection.
        
        Args:
            documents: List of document texts to embed
            metadatas: List of metadata dictionaries for each document
            ids: List of unique IDs for each document
        """
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to ChromaDB collection")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise
    
    def upsert_documents(self,
                        documents: List[str],
                        metadatas: List[Dict[str, Any]],
                        ids: List[str]) -> None:
        """
        Upsert (update or insert) documents to the collection.
        
        Args:
            documents: List of document texts to embed
            metadatas: List of metadata dictionaries for each document
            ids: List of unique IDs for each document
        """
        try:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Upserted {len(documents)} documents to ChromaDB collection")
        except Exception as e:
            logger.error(f"Error upserting documents to ChromaDB: {e}")
            raise
    
    def search(self, 
               query_texts: List[str], 
               n_results: int = 10,
               where: Optional[Dict[str, Any]] = None,
               where_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for similar documents.
        
        Args:
            query_texts: List of query texts
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions
            
        Returns:
            Search results from ChromaDB
        """
        try:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            logger.info(f"Semantic search returned {len(results.get('ids', [[]])[0])} results")
            return results
        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            raise
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the collection.
        
        Args:
            ids: List of document IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from ChromaDB collection")
        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "name": self.collection_name,
                "count": 0,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory,
                "error": str(e)
            }
    
    def reset_collection(self) -> None:
        """Reset (clear) the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Reset ChromaDB collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise
    
    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the collection."""
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except Exception:
            return False
    
    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a document if it exists.
        
        Args:
            doc_id: Document ID to look up
            
        Returns:
            Metadata dictionary if document exists, None otherwise
        """
        try:
            result = self.collection.get(ids=[doc_id], include=["metadatas"])
            if result['ids'] and result['metadatas']:
                return result['metadatas'][0]
            return None
        except Exception:
            return None


def create_chroma_client(config_path: Optional[str] = None) -> ChromaClient:
    """
    Create a ChromaClient instance from configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured ChromaClient instance
    """
    # Default configuration
    config = {
        "collection_name": "zotero_library",
        "embedding_model": "default",
        "embedding_config": {}
    }
    
    # Load configuration from file if it exists
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config.get("semantic_search", {}))
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {e}")
    
    # Load configuration from environment variables
    env_embedding_model = os.getenv("ZOTERO_EMBEDDING_MODEL")
    if env_embedding_model:
        config["embedding_model"] = env_embedding_model
    
    # Set up embedding config from environment
    if config["embedding_model"] == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        if openai_api_key:
            config["embedding_config"] = {
                "api_key": openai_api_key,
                "model_name": openai_model
            }
            if openai_base_url:
                config["embedding_config"]["base_url"] = openai_base_url
    
    elif config["embedding_model"] == "gemini":
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        gemini_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
        gemini_base_url = os.getenv("GEMINI_BASE_URL")
        if gemini_api_key:
            config["embedding_config"] = {
                "api_key": gemini_api_key,
                "model_name": gemini_model
            }
            if gemini_base_url:
                config["embedding_config"]["base_url"] = gemini_base_url
    
    return ChromaClient(
        collection_name=config["collection_name"],
        embedding_model=config["embedding_model"],
        embedding_config=config["embedding_config"]
    )