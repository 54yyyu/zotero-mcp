"""Concrete realtime embedding function providers.

Each submodule defines one ``RemoteEmbeddingFunction`` (or, for
``huggingface``, a plain ChromaDB ``EmbeddingFunction``) subclass decorated
with ``@register_embedding_function`` — importing a submodule has the side
effect of registering that provider's embedding function with ChromaDB so
persisted collections can be rebuilt by name.

Importing this package (rather than an individual submodule) registers every
provider in one call — ``chroma_client.py`` re-exports the four original
classes (openai/gemini/huggingface/ollama) individually for backward
compatibility, but new code that only needs the registration side effect
(or wants every provider, including new ones like ``voyage``, registered)
can just do ``import zotero_mcp.embeddings.providers``.
"""

from zotero_mcp.embeddings.providers import (  # noqa: F401
    gemini,
    huggingface,
    ollama,
    openai,
    voyage,
)
