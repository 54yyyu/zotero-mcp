"""The concrete embedding function classes, one module per provider.

Each submodule defines exactly one embedding function decorated with
``@register_embedding_function``, so importing a submodule has the side effect
of claiming that provider's name in ChromaDB's own embedding-function registry.
Importing this package imports all four at once.

Note the two distinct registries in play, which are easy to confuse:

- ChromaDB's ``known_embedding_functions`` — a name -> class map used to rebuild
  an embedding function from a *persisted collection's* config. That is what
  the decorator and :func:`ensure_embedding_functions_registered` below feed.
- zotero-mcp's own provider registry in :mod:`zotero_mcp.embeddings.registry` —
  a name -> :class:`~zotero_mcp.embeddings.registry.ProviderSpec` map used to
  turn a *user's configured* ``embedding_model`` string into a constructed
  embedding function.

These modules depend only on :mod:`zotero_mcp.embeddings.base`, never on
``chroma_client`` or on the provider registry, so neither import direction can
cycle.
"""

import logging

from chromadb.utils.embedding_functions import register_embedding_function

from zotero_mcp.embeddings.providers.gemini import GeminiEmbeddingFunction
from zotero_mcp.embeddings.providers.huggingface import HuggingFaceEmbeddingFunction
from zotero_mcp.embeddings.providers.ollama import OllamaEmbeddingFunction
from zotero_mcp.embeddings.providers.openai import OpenAIEmbeddingFunction

logger = logging.getLogger(__name__)

__all__ = [
    "CUSTOM_EMBEDDING_FUNCTIONS",
    "GeminiEmbeddingFunction",
    "HuggingFaceEmbeddingFunction",
    "OllamaEmbeddingFunction",
    "OpenAIEmbeddingFunction",
    "ensure_embedding_functions_registered",
]

#: Our embedding functions, in registration order. Three of the four names
#: ("openai", "huggingface", "ollama") collide with ChromaDB built-ins.
CUSTOM_EMBEDDING_FUNCTIONS = (
    OpenAIEmbeddingFunction,
    GeminiEmbeddingFunction,
    HuggingFaceEmbeddingFunction,
    OllamaEmbeddingFunction,
)


def ensure_embedding_functions_registered() -> None:
    """(Re-)claim our embedding-function names in ChromaDB's registry.

    ``known_embedding_functions`` is a plain last-write-wins dict, so import
    order decides whether a colliding name resolves to our class or to
    ChromaDB's built-in. Re-registering immediately before a collection is
    opened means a built-in that got imported after this module still cannot
    shadow us and mis-handle our persisted config (issue #382).
    """
    for cls in CUSTOM_EMBEDDING_FUNCTIONS:
        try:
            register_embedding_function(cls)
        except Exception as e:  # pragma: no cover - registry API change
            logger.debug(f"Could not re-register {cls.__name__}: {e}")
