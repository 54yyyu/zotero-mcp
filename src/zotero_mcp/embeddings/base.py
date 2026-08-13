"""Provider-agnostic base class for zotero-mcp's embedding functions.

This holds only the two method bodies that were already byte-identical across
the concrete providers when they lived in ``chroma_client.py``:

- ``embed_query`` -> ``self.__call__([text])[0]``. OpenAI, HuggingFace and
  Ollama each carried their own copy of exactly this. Gemini still overrides
  it, because its query path uses a different task type (v1) or prompt prefix
  (v2) than its document path.
- a character-ratio ``truncate``. Gemini and Ollama each carried their own copy
  at 4 chars/token. OpenAI overrides it with tiktoken and HuggingFace with the
  model's own tokenizer.

Deliberately *not* here: request pacing, retries, sub-batching, parallelism.
Every provider keeps its own ``__call__`` exactly as it was, because unifying
those loops is a behaviour change; it belongs with the streaming and adaptive
rate-limiting work, not with this move.

This class is not registered with ChromaDB — it has no ``name()``,
``get_config()`` or ``build_from_config()``, so it can never be resolved by
name when a persisted collection's config is rebuilt. Only the concrete
subclasses are registered.

``chars_per_token`` is read as a class attribute rather than an instance one on
purpose: several tests construct providers via ``Cls.__new__(Cls)``, setting
only the handful of instance attributes the assertion needs, so anything this
class touches has to resolve without ``__init__`` having run.
"""

from zotero_mcp.utils import install_hint

try:
    from chromadb import EmbeddingFunction
except ImportError as e:
    raise ImportError(
        f"chromadb is required for semantic search. {install_hint('semantic')}"
    ) from e


class BaseEmbeddingFunction(EmbeddingFunction):
    """Shared behaviour for the zotero-mcp embedding functions."""

    #: Characters per token, for the estimate-based :meth:`truncate` below.
    chars_per_token = 4

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string via the document path.

        Correct for any provider that does not tune queries and documents
        differently; Gemini overrides this.
        """
        return self.__call__([text])[0]

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate using character-based estimation (``chars_per_token``).

        The fallback for providers with no tokenizer of their own to consult.
        """
        max_chars = max_tokens * self.chars_per_token
        if len(text) > max_chars:
            text = text[:max_chars]
        return text
