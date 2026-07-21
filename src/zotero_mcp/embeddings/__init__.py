"""Shared realtime-embedding machinery: adaptive rate limiting + the common
sub-batching/retry base class used by the remote provider embedding
functions defined in ``zotero_mcp.chroma_client``.

Phase 1 of the embedding-provider refactor: this subpackage holds only the
provider-agnostic pieces. The concrete embedding function classes
(``OpenAIEmbeddingFunction``, ``GeminiEmbeddingFunction``,
``OllamaEmbeddingFunction``) stay defined in ``chroma_client.py`` for now —
moving them here is a later phase — but they subclass ``RemoteEmbeddingFunction``
from this package.
"""

from zotero_mcp.embeddings.base import RemoteEmbeddingFunction
from zotero_mcp.embeddings.ratelimit import AdaptiveRateLimiter

__all__ = ["AdaptiveRateLimiter", "RemoteEmbeddingFunction"]
