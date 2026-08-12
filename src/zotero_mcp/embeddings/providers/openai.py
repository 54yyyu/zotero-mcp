"""OpenAI (and OpenAI-compatible) embedding function."""

import os
from typing import Any

from chromadb import Documents, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

from zotero_mcp.embeddings.base import BaseEmbeddingFunction


@register_embedding_function
class OpenAIEmbeddingFunction(BaseEmbeddingFunction):
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

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None,
                 base_url: str | None = None, request_batch_size: int | None = None,
                 rate_limit_rps: float | None = None):
        import threading
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.request_batch_size = int(request_batch_size) if request_batch_size else self.DEFAULT_REQUEST_BATCH_SIZE
        self.rate_limit_rps: float | None = float(rate_limit_rps) if rate_limit_rps else None
        self._rate_lock = threading.Lock()
        self._last_request_ts: float = 0.0
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        try:
            import openai
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
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
            "request_batch_size": self.request_batch_size,
            "rate_limit_rps": self.rate_limit_rps,
            # ChromaDB's built-in EF of the same registered name rebuilds from
            # {api_key_env_var, model_name, api_base, ...} and asserts ("This
            # code should not be reached") when those are missing. Persisting
            # its spellings too keeps the stored config buildable by whichever
            # class wins the registry lookup (issue #382).
            "api_key_env_var": "OPENAI_API_KEY",
            "api_base": self.base_url,
        }

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "OpenAIEmbeddingFunction":
        # Accept either key spelling so a config written by ChromaDB's built-in
        # (api_base / api_key_env_var) rebuilds here too.
        api_key = config.get("api_key")
        if not api_key and config.get("api_key_env_var"):
            api_key = os.getenv(config["api_key_env_var"])
        return OpenAIEmbeddingFunction(
            model_name=config.get("model_name", "text-embedding-3-small"),
            api_key=api_key,
            base_url=config.get("base_url") or config.get("api_base"),
            request_batch_size=config.get("request_batch_size"),
            rate_limit_rps=config.get("rate_limit_rps"),
        )

    def _wait_for_rate_limit(self) -> None:
        """Sleep as needed so successive embedding requests stay under
        ``rate_limit_rps``. Applied per HTTP request (including each sub-batch)
        so rate-limited providers see a steady cadence regardless of how many
        inputs the caller passed. The lock keeps parallel threads honest.
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

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using the OpenAI-compatible API.

        ``encoding_format="float"`` is set explicitly. The OpenAI SDK otherwise
        negotiates base64 by default, which OpenRouter's Gemini embedding
        providers (e.g. ``google/gemini-embedding-001``) do not return reliably —
        the SDK then raises "No embedding data received" intermittently. Forcing
        float makes every OpenAI-compatible backend, native OpenAI included,
        respond deterministically.
        """
        batch_size = self.request_batch_size or self.DEFAULT_REQUEST_BATCH_SIZE
        vecs: Embeddings = []
        for i in range(0, len(input), batch_size):
            sub = input[i:i + batch_size]
            self._wait_for_rate_limit()
            response = self.client.embeddings.create(
                model=self.model_name,
                input=sub,
                encoding_format="float",
            )
            vecs.extend(data.embedding for data in response.data)
        return vecs

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
