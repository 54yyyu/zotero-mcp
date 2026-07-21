"""Shared realtime-embedding machinery: adaptive rate limiting + the common
sub-batching/retry base class used by the remote provider embedding
functions.

This subpackage holds the provider-agnostic pieces (``ratelimit.py``,
``base.py``) plus the provider registry (``registry.py``, reached directly as
``zotero_mcp.embeddings.registry`` rather than via this ``__init__``) and the
concrete embedding function classes themselves, in ``embeddings/providers/``
(``OpenAIEmbeddingFunction``, ``GeminiEmbeddingFunction``,
``HuggingFaceEmbeddingFunction``, ``OllamaEmbeddingFunction``, and the
extensibility-proof ``VoyageEmbeddingFunction``) — each subclassing
``RemoteEmbeddingFunction`` from this package (HuggingFace excepted, which
runs locally and subclasses ChromaDB's plain ``EmbeddingFunction`` directly).
``zotero_mcp.chroma_client`` re-exports the original four for backward
compatibility.

Importing ``.providers`` here registers every provider's
``@register_embedding_function`` side effect just by importing this package,
with no cycle back to ``chroma_client``: provider modules depend only on
``embeddings.base`` (a plain submodule import, independent of whether this
``__init__`` has finished running), never on ``chroma_client`` or
``embeddings.registry``.
"""

# Import for the @register_embedding_function side effects (see module
# docstring); no attribute of this subpackage is re-exported from it.
from zotero_mcp.embeddings import providers  # noqa: F401
from zotero_mcp.embeddings.base import RemoteEmbeddingFunction
from zotero_mcp.embeddings.ratelimit import AdaptiveRateLimiter

__all__ = ["AdaptiveRateLimiter", "RemoteEmbeddingFunction"]
