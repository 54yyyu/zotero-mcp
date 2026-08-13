"""Ollama embedding function, backed by Ollama's local HTTP API."""

import os
from typing import Any

from chromadb import Documents, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

from zotero_mcp.embeddings.base import BaseEmbeddingFunction


@register_embedding_function
class OllamaEmbeddingFunction(BaseEmbeddingFunction):
    """Custom Ollama embedding function for ChromaDB.

    Uses Ollama's local HTTP API. Registered under the name ``ollama`` so
    ChromaDB can rebuild persisted collections that were created with this
    embedding function.
    """

    # Ollama models vary; use a conservative, char-based fallback budget.
    max_input_tokens = 8000

    # HTTP timeout (seconds) for /api/embed. Persisted in get_config() because
    # ChromaDB's built-in ollama EF requires a ``timeout`` key.
    DEFAULT_TIMEOUT = 120

    # Documents per /api/embed request. The indexer hands us a whole item
    # batch, which with chunking enabled is (items × max_chunks_per_item)
    # documents — up to thousands. Sending that as one request makes a single
    # HTTP call that has to outlast the entire GPU pass, which is what pushed
    # runs past any sane timeout (#423). Chunking the request keeps each call
    # short; Ollama processes sequentially either way, so on a local server
    # the extra round trips cost approximately nothing.
    DEFAULT_REQUEST_BATCH_SIZE = 64

    def __init__(self, model_name: str = "qwen3-embedding", base_url: str | None = None,
                 url: str | None = None, timeout: int | None = None,
                 request_batch_size: int | None = None):
        self.model_name = model_name
        # ``url`` is ChromaDB's built-in spelling of ``base_url``; accept both
        # so a config written by either class rebuilds here (issue #382).
        self.base_url = (
            base_url or url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        ).rstrip("/")
        # Mirror the attribute under the built-in's name as well.
        self.url = self.base_url
        self.timeout = int(timeout) if timeout else self.DEFAULT_TIMEOUT
        self.request_batch_size = (
            int(request_batch_size) if request_batch_size else self.DEFAULT_REQUEST_BATCH_SIZE
        )

    @staticmethod
    def name() -> str:
        return "ollama"

    def get_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            # ChromaDB ships its own OllamaEmbeddingFunction registered under
            # the same name "ollama". Whichever class wins the registry lookup
            # gets this dict when the persisted collection config is rebuilt at
            # query time; the built-in reads url/model_name/timeout and asserts
            # "This code should not be reached" when any is missing (#382).
            # Carrying both spellings makes the config valid for both classes.
            "url": self.base_url,
            "timeout": self.timeout,
            # Extra keys are ignored by the built-in (it reads only
            # url/model_name/timeout via .get()), so carrying ours is safe.
            "request_batch_size": self.request_batch_size,
        }

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "OllamaEmbeddingFunction":
        return OllamaEmbeddingFunction(
            model_name=config.get("model_name", "qwen3-embedding"),
            base_url=config.get("base_url") or config.get("url"),
            timeout=config.get("timeout"),
            request_batch_size=config.get("request_batch_size"),
        )

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using Ollama's /api/embed endpoint.

        Unlike the deprecated /api/embeddings route (single ``prompt`` -> single
        ``embedding``), /api/embed accepts a batch via ``input`` and returns a
        list under ``embeddings``, so several documents go out per request.

        The caller's list is split into ``request_batch_size`` chunks so one
        request never has to cover an unbounded amount of GPU work; see that
        attribute for why. Vectors are concatenated back in input order.
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests package is required for Ollama embeddings")

        texts = list(input)
        if not texts:
            return []

        endpoint = f"{self.base_url}/api/embed"
        embeddings: list = []
        for start in range(0, len(texts), self.request_batch_size):
            window = texts[start : start + self.request_batch_size]
            response = requests.post(
                endpoint,
                json={"model": self.model_name, "input": window},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            vectors = data.get("embeddings")
            if vectors is None:
                raise ValueError(
                    f"Ollama /api/embed returned no 'embeddings' field: {data}"
                )
            if len(vectors) != len(window):
                # A short response would silently misalign every vector after
                # it with the wrong document, poisoning the index in a way that
                # only shows up as bad search results much later.
                raise ValueError(
                    f"Ollama /api/embed returned {len(vectors)} embeddings for "
                    f"{len(window)} inputs"
                )
            embeddings.extend(vectors)
        return embeddings
