"""HuggingFace (local, sentence-transformers) realtime embedding function.

Moved verbatim out of ``chroma_client.py`` (Phase 5 of the embedding-provider
refactor) — no renames, no behavior changes. ``chroma_client.py`` re-exports
``HuggingFaceEmbeddingFunction`` so every existing import path keeps working.
"""

import logging
from typing import Any

from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

logger = logging.getLogger(__name__)


@register_embedding_function
class HuggingFaceEmbeddingFunction(EmbeddingFunction):
    """Custom HuggingFace embedding function for ChromaDB using sentence-transformers.

    Runs HuggingFace-Hub models **locally** via the ``sentence-transformers``
    package — it downloads model weights and computes embeddings on this
    machine's CPU/GPU. It never calls the hosted HF Inference API.

    Registered under the name "huggingface" so ChromaDB rebuilds it (rather
    than its own incompatible built-in of the same name) when reloading a
    persisted collection's config (see ``OpenAIEmbeddingFunction`` for
    details on the general registration mechanism). This shadowing is
    deliberate: ChromaDB's own built-in ``HuggingFaceEmbeddingFunction`` calls
    the hosted HF *Inference API* (it expects an ``api_key``/endpoint config
    for that hosted service), which is incompatible with — and would reject —
    this class's local ``{model_name}`` config. Without the shadow,
    ChromaDB would resolve the name "huggingface" to its own built-in and
    fail to rebuild a collection created with this local backend.

    Both the class name ``HuggingFaceEmbeddingFunction`` and the registered
    name ``"huggingface"`` are frozen: ChromaDB persists the registered name
    in a collection's config and reconstructs the embedding function by that
    name on every reload. Renaming either would orphan every existing
    collection built with this backend — ChromaDB would no longer be able to
    find (or would resolve to the wrong) embedding function for it.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.model_name = model_name

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        except ImportError:
            raise ImportError("sentence-transformers package is required for HuggingFace embeddings. Install with: pip install sentence-transformers")

        # Read limit from model metadata; conservative fallback
        self.max_input_tokens = getattr(self.model, "max_seq_length", 500)

    @staticmethod
    def name() -> str:
        return "huggingface"

    def get_config(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HuggingFaceEmbeddingFunction":
        return HuggingFaceEmbeddingFunction(
            model_name=config.get("model_name", "Qwen/Qwen3-Embedding-0.6B"),
        )

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using HuggingFace model."""
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string. No special handling needed for HuggingFace."""
        return self.__call__([text])[0]

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
