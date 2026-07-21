"""Voyage AI realtime embedding function.

Added in Phase 5 of the embedding-provider refactor as the extensibility
proof: adding a brand-new provider is exactly one
``embeddings/providers/<name>.py`` module (an EF subclass implementing the
``RemoteEmbeddingFunction`` hooks) plus one ``register_provider`` call in
``embeddings/registry.py`` — zero edits to ``semantic_search.py`` or
``cli.py``.
"""

import os
from typing import Any

from chromadb.utils.embedding_functions import register_embedding_function

from zotero_mcp.embeddings.base import RemoteEmbeddingFunction

# Public Voyage AI API base. Overridable via config/env VOYAGE_BASE_URL for
# proxy/self-hosted setups, mirroring the OpenAI/Ollama base_url pattern.
# The embeddings endpoint is "<base_url>/embeddings".
DEFAULT_VOYAGE_BASE_URL = "https://api.voyageai.com/v1"


@register_embedding_function
class VoyageEmbeddingFunction(RemoteEmbeddingFunction):
    """Custom Voyage AI embedding function for ChromaDB.

    Registered under the name "voyage" so ChromaDB can rebuild it from a
    persisted collection's config (see OpenAIEmbeddingFunction for details on
    the general registration mechanism).
    """

    # voyage-3.5's context window is 32k tokens; using half that as the
    # truncation budget keeps the coarse char-based truncate() (inherited
    # chars_per_token default) safely under the model's hard cap even when
    # its char/token estimate is off for dense or non-English text.
    max_input_tokens = 16000

    # Voyage's API caps a single request at 1000 texts, but per-request TOKEN
    # limits make that unsafe in practice for long documents; 128 keeps
    # request payloads comfortably under Voyage's per-request token cap.
    default_request_batch_size = 128

    def __init__(self, model_name: str = "voyage-3.5", api_key: str | None = None,
                 base_url: str | None = None, request_batch_size: int | None = None,
                 rate_limit_rps: float | None = None, max_parallel_requests: int | None = None,
                 max_retries: int | None = None):
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        base_url = (base_url or os.getenv("VOYAGE_BASE_URL") or DEFAULT_VOYAGE_BASE_URL).rstrip("/")
        if not self.api_key:
            raise ValueError("Voyage AI API key is required")

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
        return "voyage"

    def get_config(self) -> dict[str, Any]:
        cfg = {"model_name": self.model_name, "base_url": self.base_url}
        cfg.update(self._common_config())
        return cfg

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "VoyageEmbeddingFunction":
        return VoyageEmbeddingFunction(
            model_name=config.get("model_name", "voyage-3.5"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            request_batch_size=config.get("request_batch_size"),
            rate_limit_rps=config.get("rate_limit_rps"),
            max_parallel_requests=config.get("max_parallel_requests"),
            max_retries=config.get("max_retries"),
        )

    def _embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """Issue one request to Voyage's /embeddings endpoint.

        Voyage's docs recommend an asymmetric ``input_type`` ("document" vs
        "query") for retrieval use cases so the model applies its
        retrieval-tuned encoding on each side.
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests package is required for Voyage embeddings")

        endpoint = f"{self.base_url}/embeddings"
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "input": texts,
                "input_type": "query" if is_query else "document",
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def _classify_error(self, exc: Exception) -> tuple[bool, float | None]:
        """429 (honoring Retry-After when present) and 5xx HTTP responses are
        retryable, as are connection-level failures.
        """
        try:
            import requests
        except ImportError:
            return False, None
        if isinstance(exc, requests.HTTPError):
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429:
                return True, self._parse_retry_after(exc)
            if isinstance(status, int) and status >= 500:
                return True, None
            return False, None
        if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
            return True, None
        return False, None

    @staticmethod
    def _parse_retry_after(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) if response is not None else None
        if not headers:
            return None
        value = headers.get("Retry-After") if hasattr(headers, "get") else None
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
