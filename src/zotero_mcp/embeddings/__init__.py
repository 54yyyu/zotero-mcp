"""Embedding functions and the provider registry for semantic search.

Layout:

- ``base.py`` — :class:`~zotero_mcp.embeddings.base.BaseEmbeddingFunction`, the
  small amount of behaviour shared by every provider.
- ``providers/`` — one module per concrete embedding function, each registering
  itself with ChromaDB on import.
- ``registry.py`` — the provider registry that maps a configured
  ``embedding_model`` string to a constructed embedding function. Reached
  directly as ``zotero_mcp.embeddings.registry``; deliberately not re-exported
  here, so importing this package never has to build the registry.

Importing this package imports ``providers``, which is enough to register every
embedding function with ChromaDB.
"""

# Imported for the @register_embedding_function side effects.
from zotero_mcp.embeddings import providers  # noqa: F401
from zotero_mcp.embeddings.base import BaseEmbeddingFunction

__all__ = ["BaseEmbeddingFunction"]
