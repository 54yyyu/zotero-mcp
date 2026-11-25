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
# Keyed by (model name, max_seq_length): the cap is set on the loaded model, so
# two configs with different caps need two models.
_MODEL_CACHE: dict[tuple[str, int | None], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _gpu_model_kwargs() -> dict[str, Any]:
    """Load the model in its own dtype (bf16 for Qwen3) on a CUDA GPU.

    sentence-transformers loads float32 by default, which doubles the memory a
    bf16 model needs. transformers 4.56 renamed ``torch_dtype`` to ``dtype`` and
    5.x warns on the old name; older versions only know the old one.
    """
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    try:
        import transformers
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    except (ImportError, ValueError):
        return {"torch_dtype": "auto"}
    return {"dtype": "auto"} if (major, minor) >= (4, 56) else {"torch_dtype": "auto"}


def _load_model(model_name: str, max_seq_length: int | None = None) -> Any:
    """Return the cached SentenceTransformer for ``model_name`` (with its optional
    ``max_seq_length`` cap), loading it once."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers package is required for HuggingFace embeddings. Install with: pip install sentence-transformers")

    key = (model_name, max_seq_length)
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            logger.info(f"Loading embedding model: {model_name}")
            model_kwargs = _gpu_model_kwargs()
            model = SentenceTransformer(model_name, trust_remote_code=True, model_kwargs=model_kwargs)
            # A long-context model (Qwen3: 32k tokens) pads a batch to its longest
            # text; the cap keeps a full-text update-db inside a small GPU.
            if max_seq_length and max_seq_length < (model.max_seq_length or max_seq_length + 1):
                model.max_seq_length = max_seq_length
            _MODEL_CACHE[key] = model
    return model


@register_embedding_function
class HuggingFaceEmbeddingFunction(BaseEmbeddingFunction):
    """Custom HuggingFace embedding function for ChromaDB using sentence-transformers.

    Registered under the name "huggingface" so ChromaDB rebuilds it (rather than
    its own incompatible built-in of the same name) when reloading a persisted
    collection's config (see OpenAIEmbeddingFunction for details).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        query_instruction: str | None = None,
        max_seq_length: int | None = None,
        batch_size: int | None = None,
    ):
        if query_instruction is not None and not isinstance(query_instruction, str):
            raise ValueError("query_instruction must be a string or null")
        self.query_instruction = (query_instruction or "").strip()
        self.model_name = model_name
        # Optional caps for small GPUs: max_seq_length truncates each text (see
        # _load_model), batch_size bounds how many texts are encoded at once.
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.model = _load_model(model_name, max_seq_length)

        # Read limit from model metadata; conservative fallback
        self.max_input_tokens = getattr(self.model, "max_seq_length", 500)

    @staticmethod
    def name() -> str:
        return "huggingface"

    def get_config(self) -> dict[str, Any]:
        config = {
            "model_name": self.model_name,
            "query_instruction": getattr(self, "query_instruction", ""),
            # ChromaDB's built-in "huggingface" EF requires api_key_env_var in
            # addition to model_name and asserts without it. Persisting the key
            # keeps the config buildable by either class (issue #382); our own
            # build_from_config ignores it (we embed locally, no API key).
            "api_key_env_var": "HUGGINGFACE_API_KEY",
        }
        # Written only when set, so existing collection configs stay as they were.
        if getattr(self, "max_seq_length", None):
            config["max_seq_length"] = self.max_seq_length
        if getattr(self, "batch_size", None):
            config["batch_size"] = self.batch_size
        return config

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HuggingFaceEmbeddingFunction":
        return HuggingFaceEmbeddingFunction(
            model_name=config.get("model_name", "Qwen/Qwen3-Embedding-0.6B"),
            query_instruction=config.get("query_instruction"),
            max_seq_length=config.get("max_seq_length"),
            batch_size=config.get("batch_size"),
        )

    def embed_query_text(self, text: str) -> list[float]:
        """Apply an optional instruction to queries without changing documents."""
        instruction = getattr(self, "query_instruction", "")
        query = f"Instruct: {instruction}\nQuery: {text}" if instruction else text
        return self.__call__([query])[0]

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using HuggingFace model."""
        batch_size = getattr(self, "batch_size", None)
        kwargs = {"batch_size": batch_size} if batch_size else {}
        embeddings = self.model.encode(input, convert_to_numpy=True, **kwargs)
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
