"""OpenAI realtime embedding function.

Moved verbatim out of ``chroma_client.py`` (Phase 5 of the embedding-provider
refactor) — no renames, no behavior changes. ``chroma_client.py`` re-exports
``OpenAIEmbeddingFunction`` so every existing import path keeps working.
"""

import os
from typing import Any

from chromadb.utils.embedding_functions import register_embedding_function

from zotero_mcp.embeddings.base import RemoteEmbeddingFunction


@register_embedding_function
class OpenAIEmbeddingFunction(RemoteEmbeddingFunction):
    """Custom OpenAI embedding function for ChromaDB.

    Registered under the name "openai" so ChromaDB rebuilds it (rather than its
    own incompatible built-in of the same name) when reloading a persisted
    collection's config. ChromaDB >=1.x reconstructs the embedding function by
    name from the stored config during upsert; without registration the name
    collides with the built-in, whose build_from_config rejects our
    {model_name, base_url} config.
    """

    max_input_tokens = 8000  # text-embedding-3-* limit is 8191

    # Per-request input-list cap. OpenAI allows up to 2048 items but many
    # OpenAI-compatible providers are stricter (SiliconFlow is 64 for
    # /v1/embeddings, Mistral is 512, etc.). Defaulting to 64 keeps the code
    # portable; real OpenAI users can raise embedding_config.request_batch_size.
    DEFAULT_REQUEST_BATCH_SIZE = 64
    default_request_batch_size = DEFAULT_REQUEST_BATCH_SIZE

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None,
                 base_url: str | None = None, request_batch_size: int | None = None,
                 rate_limit_rps: float | None = None,
                 max_parallel_requests: int | None = None, max_retries: int | None = None):
        import threading
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        # Legacy fixed-interval throttle state. No longer used by __call__
        # (which now goes through AdaptiveRateLimiter via RemoteEmbeddingFunction),
        # but kept because tests/test_openai_embedding_batching.py calls
        # _wait_for_rate_limit() directly and inspects _last_request_ts.
        self._rate_lock = threading.Lock()
        self._last_request_ts: float = 0.0
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        try:
            import openai
            client_kwargs = {"api_key": self.api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = openai.OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError("openai package is required for OpenAI embeddings")

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
        return "openai"

    def get_config(self) -> dict[str, Any]:
        cfg = {"model_name": self.model_name, "base_url": self.base_url}
        cfg.update(self._common_config())
        return cfg

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "OpenAIEmbeddingFunction":
        return OpenAIEmbeddingFunction(
            model_name=config.get("model_name", "text-embedding-3-small"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            request_batch_size=config.get("request_batch_size"),
            rate_limit_rps=config.get("rate_limit_rps"),
            max_parallel_requests=config.get("max_parallel_requests"),
            max_retries=config.get("max_retries"),
        )

    def _wait_for_rate_limit(self) -> None:
        """Pre-AdaptiveRateLimiter fixed-interval throttle. Deprecated and no
        longer called internally (see AdaptiveRateLimiter in
        embeddings/ratelimit.py) — kept only because
        tests/test_openai_embedding_batching.py invokes it directly.
        """
        rps = self.rate_limit_rps
        if not rps or rps <= 0:
            return
        import time
        with self._rate_lock:
            min_interval = 1.0 / rps
            wait = min_interval - (time.monotonic() - self._last_request_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_request_ts = time.monotonic()

    def _embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """Issue one OpenAI embeddings request.

        ``encoding_format="float"`` is set explicitly. The OpenAI SDK otherwise
        negotiates base64 by default, which OpenRouter's Gemini embedding
        providers (e.g. ``google/gemini-embedding-001``) do not return reliably —
        the SDK then raises "No embedding data received" intermittently. Forcing
        float makes every OpenAI-compatible backend, native OpenAI included,
        respond deterministically.

        No ``service_tier`` is ever sent: OpenAI's flex processing does not
        cover the embeddings endpoint (only Responses/Chat Completions), and
        the SDK's ``embeddings.create`` rejects the kwarg with a TypeError.
        The 50%-discount path for embeddings is the Batch API
        (``openai_batch.py``).
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
            encoding_format="float",
        )
        return [data.embedding for data in response.data]

    def _classify_error(self, exc: Exception) -> tuple[bool, float | None]:
        """openai.RateLimitError and 5xx APIStatusError are retryable; honor
        Retry-After from the response headers when the SDK surfaces it.
        """
        try:
            import openai
        except ImportError:
            return False, None
        if isinstance(exc, openai.RateLimitError):
            return True, self._parse_retry_after(exc)
        if isinstance(exc, openai.APIStatusError) and getattr(exc, "status_code", 0) >= 500:
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

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate using tiktoken cl100k_base (correct for OpenAI models)."""
        try:
            import tiktoken
            if not hasattr(self, '_tokenizer'):
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            tokens = self._tokenizer.encode(text, disallowed_special=())
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                text = self._tokenizer.decode(tokens)
        except ImportError:
            max_chars = max_tokens * 3
            if len(text) > max_chars:
                text = text[:max_chars]
        return text
