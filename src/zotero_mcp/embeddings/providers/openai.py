"""OpenAI (and OpenAI-compatible) embedding function."""

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

    # Tokens per minute the limiter paces against when nothing else supplies a
    # ceiling. text-embedding-3-small is 1,000,000 TPM on Tier 1 and higher on
    # later tiers, so this leaves 5% headroom against the *lowest* tier, which
    # is the safe default to start a run at: the provider does report its
    # ceiling in x-ratelimit-limit-tokens, but only on a response, so the first
    # requests have to be paced against something already known. Users above
    # Tier 1 raise it via embedding_config.tokens_per_minute.
    DEFAULT_TOKENS_PER_MINUTE = 950_000.0
    default_tokens_per_minute = DEFAULT_TOKENS_PER_MINUTE

    # Concurrent in-flight requests. TPM, not request count, is what binds for
    # embeddings — at a 64 x ~500-token payload the 1M ceiling caps throughput
    # near 31 req/min against a 3,000 RPM allowance — so this exists to hide
    # per-request latency, not to raise the throughput ceiling.
    DEFAULT_MAX_PARALLEL_REQUESTS = 4
    max_parallel_requests_default = DEFAULT_MAX_PARALLEL_REQUESTS

    # Only used to estimate a request's token cost for pacing; truncation goes
    # through tiktoken below. Deliberately below the ~4 chars/token rule of
    # thumb: PDF-extracted text (reference lists, URLs, number tables)
    # tokenizes far worse than prose, and overestimating cost merely paces a
    # little conservatively, while underestimating it earns 429s.
    chars_per_token = 3

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None,
                 base_url: str | None = None, request_batch_size: int | None = None,
                 rate_limit_rps: float | None = None,
                 max_parallel_requests: int | None = None,
                 max_retries: int | None = None,
                 tokens_per_minute: float | None = None):
        import threading
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._rate_lock = threading.Lock()
        self._last_request_ts: float = 0.0
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self._init_common(
            model_name=model_name,
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            request_batch_size=request_batch_size,
            rate_limit_rps=rate_limit_rps,
            max_parallel_requests=max_parallel_requests,
            max_retries=max_retries,
            tokens_per_minute=tokens_per_minute,
        )

        try:
            import openai
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            # Deliberately no custom http_client. The SDK already pools at
            # 1000 connections / 100 keepalive with a 600s read timeout, which
            # comfortably covers any max_parallel_requests worth setting;
            # supplying an httpx.Client here would *narrow* both, and a
            # shorter read timeout would fail large embedding requests the
            # default would have completed.
            self.client = openai.OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError("openai package is required for OpenAI embeddings")

    @staticmethod
    def name() -> str:
        return "openai"

    def get_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            # ChromaDB's built-in EF of the same registered name rebuilds from
            # {api_key_env_var, model_name, api_base, ...} and asserts ("This
            # code should not be reached") when those are missing. Persisting
            # its spellings too keeps the stored config buildable by whichever
            # class wins the registry lookup (issue #382).
            "api_key_env_var": "OPENAI_API_KEY",
            "api_base": self.base_url,
            **self._common_config(),
        }

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "OpenAIEmbeddingFunction":
        # Accept either key spelling so a config written by ChromaDB's built-in
        # (api_base / api_key_env_var) rebuilds here too. Every key added since
        # is read with .get(), so a collection persisted before they existed
        # still rebuilds.
        api_key = config.get("api_key")
        if not api_key and config.get("api_key_env_var"):
            api_key = os.getenv(config["api_key_env_var"])
        return OpenAIEmbeddingFunction(
            model_name=config.get("model_name", "text-embedding-3-small"),
            api_key=api_key,
            base_url=config.get("base_url") or config.get("api_base"),
            request_batch_size=config.get("request_batch_size"),
            rate_limit_rps=config.get("rate_limit_rps"),
            max_parallel_requests=config.get("max_parallel_requests"),
            max_retries=config.get("max_retries"),
            tokens_per_minute=config.get("tokens_per_minute"),
        )

    def _wait_for_rate_limit(self) -> None:
        """Fixed-interval pacing to keep requests under ``rate_limit_rps``.

        Superseded by the shared :class:`AdaptiveRateLimiter`, which paces on a
        token budget rather than a request interval and is what the embedding
        path now goes through. Kept because ``rate_limit_rps`` remains a
        supported config key and this is still its most direct expression, and
        because a test exercises it on its own.
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

    def _embed_batch(self, texts: list[str], is_query: bool = False) -> Any:
        """One embeddings request, returning ``(vectors, headers)``.

        ``encoding_format="float"`` is set explicitly. The OpenAI SDK otherwise
        negotiates base64 by default, which OpenRouter's Gemini embedding
        providers (e.g. ``google/gemini-embedding-001``) do not return reliably —
        the SDK then raises "No embedding data received" intermittently. Forcing
        float makes every OpenAI-compatible backend, native OpenAI included,
        respond deterministically.

        Headers come back via ``with_raw_response`` where the SDK offers it, so
        the limiter can read whatever rate-limit headroom the provider reports.
        OpenAI-compatible backends and test doubles that do not expose it fall
        back to the plain call and simply report no headers.
        """
        embeddings_api = self.client.embeddings
        raw_api = getattr(embeddings_api, "with_raw_response", None)
        request = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }

        if raw_api is not None:
            raw = raw_api.create(**request)
            response = raw.parse()
            headers = getattr(raw, "headers", None)
        else:
            response = embeddings_api.create(**request)
            headers = None

        return [data.embedding for data in response.data], headers

    def _classify_error(self, exc: Exception) -> tuple[bool, float | None]:
        """Retry rate limits and server-side failures; fail fast on the rest."""
        try:
            import openai
        except ImportError:
            return False, None

        if isinstance(exc, openai.RateLimitError):
            return True, self._parse_retry_after(exc)
        if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
            return True, None
        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
            return True, None
        return False, None

    @staticmethod
    def _parse_retry_after(exc: Exception) -> float | None:
        """Seconds from a response's Retry-After header, if it carried one."""
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None or not hasattr(headers, "get"):
            return None
        value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate using tiktoken cl100k_base (correct for OpenAI models)."""
        # Every BPE token covers at least one byte, so a text whose UTF-8 byte
        # length is already within max_tokens cannot exceed max_tokens tokens
        # and encoding it would be a no-op. Skipping the encode matters because
        # tiktoken's pre-tokenizer regex falls into fancy_regex backtracking on
        # PDF-extracted text (reference lists, URLs, number tables), where it
        # dominated indexing CPU time and capped throughput well below what the
        # embedding API itself allowed. Output-identical either way.
        if len(text.encode("utf-8")) <= max_tokens:
            return text
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
