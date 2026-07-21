"""Ollama realtime embedding function.

Moved verbatim out of ``chroma_client.py`` (Phase 5 of the embedding-provider
refactor) — no renames, no behavior changes. ``chroma_client.py`` re-exports
``OllamaEmbeddingFunction`` so every existing import path keeps working.
"""

import os
from typing import Any

from chromadb.utils.embedding_functions import register_embedding_function

from zotero_mcp.embeddings.base import RemoteEmbeddingFunction


@register_embedding_function
class OllamaEmbeddingFunction(RemoteEmbeddingFunction):
    """Custom Ollama embedding function for ChromaDB.

    Uses Ollama's local HTTP API. Registered under the name ``ollama`` so
    ChromaDB can rebuild persisted collections that were created with this
    embedding function.
    """

    # Ollama models vary; use a conservative, char-based fallback budget.
    max_input_tokens = 8000

    # None means "send the whole input as a single request" — Ollama's
    # /api/embed already accepts an arbitrarily long batch locally, so there
    # is no request-size cap to sub-batch against by default.
    default_request_batch_size = None

    def __init__(self, model_name: str = "qwen3-embedding", base_url: str | None = None,
                 request_batch_size: int | None = None, rate_limit_rps: float | None = None,
                 max_parallel_requests: int | None = None, max_retries: int | None = None):
        base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self._init_common(
            model_name=model_name,
            base_url=base_url,
            request_batch_size=request_batch_size,
            rate_limit_rps=rate_limit_rps,
            max_parallel_requests=max_parallel_requests,
            max_retries=max_retries,
        )

    @staticmethod
    def name() -> str:
        return "ollama"

    def get_config(self) -> dict[str, Any]:
        cfg = {"model_name": self.model_name, "base_url": self.base_url}
        cfg.update(self._common_config())
        return cfg

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "OllamaEmbeddingFunction":
        return OllamaEmbeddingFunction(
            model_name=config.get("model_name", "qwen3-embedding"),
            base_url=config.get("base_url"),
            request_batch_size=config.get("request_batch_size"),
            rate_limit_rps=config.get("rate_limit_rps"),
            max_parallel_requests=config.get("max_parallel_requests"),
            max_retries=config.get("max_retries"),
        )

    def _embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """Issue one request to Ollama's /api/embed endpoint.

        Unlike the deprecated /api/embeddings route (single ``prompt`` -> single
        ``embedding``), /api/embed accepts a batch via ``input`` and returns a
        list under ``embeddings``, so the whole batch is sent in one request
        instead of one request per document.
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests package is required for Ollama embeddings")

        if not texts:
            return []

        endpoint = f"{self.base_url}/api/embed"
        response = requests.post(
            endpoint,
            json={"model": self.model_name, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise ValueError(
                f"Ollama /api/embed returned no 'embeddings' field: {data}"
            )
        return embeddings

    def _classify_error(self, exc: Exception) -> tuple[bool, float | None]:
        """429/5xx HTTP responses and connection-level failures are retryable —
        Ollama runs locally, so connection errors are usually the server still
        loading the model rather than something permanently wrong.
        """
        try:
            import requests
        except ImportError:
            return False, None
        if isinstance(exc, requests.HTTPError):
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429 or (isinstance(status, int) and status >= 500):
                return True, None
            return False, None
        if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
            return True, None
        return False, None
