"""Ollama embedding function, backed by Ollama's local HTTP API."""

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

    A local server has no published request or token ceiling to pace
    against and processes sequentially regardless of client-side concurrency,
    so unlike the cloud providers this class does not raise
    ``max_parallel_requests_default`` or set a ``default_tokens_per_minute``
    — both stay at the base class's conservative defaults (1 and unset).
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
    default_request_batch_size = DEFAULT_REQUEST_BATCH_SIZE

    def __init__(self, model_name: str = "qwen3-embedding", base_url: str | None = None,
                 url: str | None = None, timeout: int | None = None,
                 request_batch_size: int | None = None,
                 rate_limit_rps: float | None = None,
                 max_parallel_requests: int | None = None,
                 max_retries: int | None = None,
                 tokens_per_minute: float | None = None):
        # ``url`` is ChromaDB's built-in spelling of ``base_url``; accept both
        # so a config written by either class rebuilds here (issue #382).
        resolved_base_url = (
            base_url or url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        ).rstrip("/")
        self.timeout = int(timeout) if timeout else self.DEFAULT_TIMEOUT

        self._init_common(
            model_name=model_name,
            base_url=resolved_base_url,
            request_batch_size=request_batch_size,
            rate_limit_rps=rate_limit_rps,
            max_parallel_requests=max_parallel_requests,
            max_retries=max_retries,
            tokens_per_minute=tokens_per_minute,
        )
        # Mirror the attribute under the built-in's name as well.
        self.url = self.base_url

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
            **self._common_config(),
        }

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "OllamaEmbeddingFunction":
        return OllamaEmbeddingFunction(
            model_name=config.get("model_name", "qwen3-embedding"),
            base_url=config.get("base_url") or config.get("url"),
            timeout=config.get("timeout"),
            request_batch_size=config.get("request_batch_size"),
            rate_limit_rps=config.get("rate_limit_rps"),
            max_parallel_requests=config.get("max_parallel_requests"),
            max_retries=config.get("max_retries"),
            tokens_per_minute=config.get("tokens_per_minute"),
        )

    def _embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """One /api/embed request for ``texts``.

        Unlike the deprecated /api/embeddings route (single ``prompt`` ->
        single ``embedding``), /api/embed accepts a batch via ``input`` and
        returns a list under ``embeddings``. The base class now owns splitting
        the caller's full input into ``request_batch_size`` windows so no
        single request has to cover an unbounded amount of GPU work (#423);
        this issues exactly one request for whatever window it is given.
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests package is required for Ollama embeddings")

        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model_name, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        vectors = data.get("embeddings")
        if vectors is None:
            raise ValueError(
                f"Ollama /api/embed returned no 'embeddings' field: {data}"
            )
        if len(vectors) != len(texts):
            # A short response would silently misalign every vector after
            # it with the wrong document, poisoning the index in a way that
            # only shows up as bad search results much later.
            raise ValueError(
                f"Ollama /api/embed returned {len(vectors)} embeddings for "
                f"{len(texts)} inputs"
            )
        return vectors

    def _classify_error(self, exc: Exception) -> tuple[bool, float | None]:
        """Retry connection hiccups and throttled/server-side failures.

        A local server may still be loading the model into memory, which
        shows up as a connection error or timeout rather than an HTTP status;
        both are worth a retry. No reliable Retry-After is available either
        way.
        """
        try:
            import requests
        except ImportError:
            return False, None

        if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            return True, None
        if isinstance(exc, requests.exceptions.HTTPError):
            status = getattr(exc.response, "status_code", None)
            if status == 429 or (status is not None and status >= 500):
                return True, None
        return False, None
