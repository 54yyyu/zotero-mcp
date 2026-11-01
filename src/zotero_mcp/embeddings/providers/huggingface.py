"""Local HuggingFace embedding function, backed by sentence-transformers."""

import logging
import threading
from typing import Any

from chromadb import Documents, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

from zotero_mcp.embeddings.base import BaseEmbeddingFunction

logger = logging.getLogger(__name__)

# One SentenceTransformer per model name, shared by every
# HuggingFaceEmbeddingFunction in the process. ChromaDB rebuilds the embedding
# function from the persisted collection config on every upsert and query
# (Schema.deserialize_from_json -> build_from_config), so without this cache
# each call loaded the model again and, on CUDA, kept a copy alive until the
# next GC pass, which exhausted GPU memory during update-db (issue #726).
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _load_model(model_name: str) -> Any:
    """Return the cached SentenceTransformer for ``model_name``, loading it once."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers package is required for HuggingFace embeddings. Install with: pip install sentence-transformers")

    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(model_name)
        if model is None:
            logger.info(f"Loading embedding model: {model_name}")
            model = SentenceTransformer(model_name, trust_remote_code=True)
            _MODEL_CACHE[model_name] = model
    return model


@register_embedding_function
class HuggingFaceEmbeddingFunction(BaseEmbeddingFunction):
    """Custom HuggingFace embedding function for ChromaDB using sentence-transformers.

    Registered under the name "huggingface" so ChromaDB rebuilds it (rather than
    its own incompatible built-in of the same name) when reloading a persisted
    collection's config (see OpenAIEmbeddingFunction for details).
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.model_name = model_name
        self.model = _load_model(model_name)

        # Read limit from model metadata; conservative fallback
        self.max_input_tokens = getattr(self.model, "max_seq_length", 500)

    @staticmethod
    def name() -> str:
        return "huggingface"

    def get_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            # ChromaDB's built-in "huggingface" EF requires api_key_env_var in
            # addition to model_name and asserts without it. Persisting the key
            # keeps the config buildable by either class (issue #382); our own
            # build_from_config ignores it (we embed locally, no API key).
            "api_key_env_var": "HUGGINGFACE_API_KEY",
        }

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HuggingFaceEmbeddingFunction":
        return HuggingFaceEmbeddingFunction(
            model_name=config.get("model_name", "Qwen/Qwen3-Embedding-0.6B"),
        )

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using HuggingFace model."""
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate using the model's own tokenizer."""
        tokenizer = getattr(self.model, 'tokenizer', None)
        if tokenizer is not None:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) > max_tokens:
                encoded = encoded[:max_tokens]
                text = tokenizer.decode(encoded)
        else:
            max_chars = max_tokens * 2
            if len(text) > max_chars:
                text = text[:max_chars]
        return text
