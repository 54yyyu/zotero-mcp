"""Gemini embedding function, backed by the google-genai SDK."""

import os
from typing import Any

from chromadb.utils.embedding_functions import register_embedding_function

from zotero_mcp.embeddings.base import RemoteEmbeddingFunction


@register_embedding_function
class GeminiEmbeddingFunction(RemoteEmbeddingFunction):
    """Custom Gemini embedding function for ChromaDB using google-genai.

    Registered under the name "gemini" so ChromaDB can rebuild it from a
    persisted collection's config (see OpenAIEmbeddingFunction for details).
    """

    # gemini-embedding-2-* models ignore the task_type config field (the API
    # silently drops it). Google's recommended alternative is to embed the
    # task instruction in the prompt text itself, which empirically shifts
    # the embedding space (cos ~0.84 vs raw baseline) and preserves asymmetric
    # doc/query tuning (cos ~0.94 between doc-prefix and query-prefix).
    # These are the canonical prefixes; __call__ and embed_query prepend them
    # to every v2 input. They MUST stay in sync with V2_PREFIX_TOKEN_BUDGET
    # below: if you lengthen a prefix, bump the budget so truncation still
    # leaves room for it under the model's hard cap.
    V2_DOC_PREFIX = "Represent this document for retrieval:\n\n"
    V2_QUERY_PREFIX = "Represent this query for retrieval:\n\n"

    # Token reservation for the v2 prefix above. The longest prefix is
    # V2_DOC_PREFIX at 42 chars ~= 11 tokens with typical English tokenization.
    # We reserve 20 tokens (11 actual + 9 slack) so that truncate() leaves
    # room for the prefix without ever producing a post-prefix payload that
    # exceeds the model's 8192 hard cap even on dense text.
    V2_PREFIX_TOKEN_BUDGET = 20

    # Default for gemini-embedding-001 (hard cap 2048 tokens). Per-instance
    # override in __init__ for models with larger context windows. NOTE: for
    # v2 models this value means "effective budget for the TEXT BODY" —
    # prefix tokens are reserved separately (see V2_PREFIX_TOKEN_BUDGET).
    max_input_tokens = 2000

    # Gemini's embed_content API caps at 100 items per batch (verified
    # empirically: batch=100 OK, batch=250 → 400 INVALID_ARGUMENT with
    # "at most 100 requests can be in one batch").
    GEMINI_MAX_BATCH = 100
    default_request_batch_size = GEMINI_MAX_BATCH

    # Tokens per minute the limiter paces against when nothing else supplies a
    # ceiling. No published per-model TPM figure is available at this
    # granularity, so this mirrors the conservative default used for OpenAI,
    # leaving headroom under a low tier rather than assuming a high one. Users
    # on a higher tier raise it via embedding_config.tokens_per_minute.
    DEFAULT_TOKENS_PER_MINUTE = 950_000.0
    default_tokens_per_minute = DEFAULT_TOKENS_PER_MINUTE

    # Concurrent in-flight requests. As with OpenAI, TPM rather than request
    # count is what binds for embeddings, so this exists to hide per-request
    # latency rather than to raise the throughput ceiling.
    DEFAULT_MAX_PARALLEL_REQUESTS = 4
    max_parallel_requests_default = DEFAULT_MAX_PARALLEL_REQUESTS

    # Gemini's query path (embed_query) bypasses the indexing pipeline's own
    # truncation, so the base class must truncate before preparing the text —
    # matching this class's original embed_query, which truncated first and
    # only then prepended the v2 prefix.
    truncate_queries = True

    def __init__(self, model_name: str = "gemini-embedding-001", api_key: str | None = None,
                 base_url: str | None = None, request_batch_size: int | None = None,
                 rate_limit_rps: float | None = None,
                 max_parallel_requests: int | None = None,
                 max_retries: int | None = None,
                 tokens_per_minute: float | None = None):
        # Model-aware token limit. For v2 models, derive from:
        #   hard_cap (8192) - safety_margin (192, for char-based truncation
        #   imprecision) - V2_PREFIX_TOKEN_BUDGET (20, reserved for the
        #   in-prompt task instruction prepended in __call__/embed_query).
        # Net effective budget for text body: 8192 - 192 - 20 = 7980 tokens.
        # This guarantees post-prefix payload <= hard cap even at the
        # truncation limit, formally closing the cap-enforcement gap.
        if "gemini-embedding-2" in model_name:
            self.max_input_tokens = 8000 - self.V2_PREFIX_TOKEN_BUDGET
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required")

        self._init_common(
            model_name=model_name,
            base_url=base_url or os.getenv("GEMINI_BASE_URL"),
            request_batch_size=request_batch_size,
            rate_limit_rps=rate_limit_rps,
            max_parallel_requests=max_parallel_requests,
            max_retries=max_retries,
            tokens_per_minute=tokens_per_minute,
        )
        # Gemini's API hard-fails above GEMINI_MAX_BATCH items per batch (see
        # the comment on that constant), so a user-configured or persisted
        # request_batch_size can never be allowed to exceed it — clamp here
        # rather than trusting the caller, reproducing today's unconditional
        # GEMINI_MAX_BATCH slicing in __call__.
        self.request_batch_size = min(self.request_batch_size, self.GEMINI_MAX_BATCH)

        try:
            from google import genai
            from google.genai import types
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                http_options = types.HttpOptions(baseUrl=self.base_url)
                client_kwargs["http_options"] = http_options
            self.client = genai.Client(**client_kwargs)
            self.types = types
        except ImportError:
            raise ImportError("google-genai package is required for Gemini embeddings")

    @staticmethod
    def name() -> str:
        return "gemini"

    def get_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            **self._common_config(),
        }

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "GeminiEmbeddingFunction":
        return GeminiEmbeddingFunction(
            model_name=config.get("model_name", "gemini-embedding-001"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            request_batch_size=config.get("request_batch_size"),
            rate_limit_rps=config.get("rate_limit_rps"),
            max_parallel_requests=config.get("max_parallel_requests"),
            max_retries=config.get("max_retries"),
            tokens_per_minute=config.get("tokens_per_minute"),
        )

    def _is_v2(self) -> bool:
        # gemini-embedding-2-* does not support the task_type config field
        # (it is silently ignored by the API). Google's guidance is to put
        # the task hint in the prompt text instead.
        return "gemini-embedding-2" in self.model_name

    def _prepare_document(self, text: str) -> str:
        """Prepend the v2 task-instruction prefix; identity for v1 models."""
        if self._is_v2():
            # v2 models: task instruction goes in the prompt, no config.
            # V2_PREFIX_TOKEN_BUDGET is already reserved from max_input_tokens
            # in __init__, so upstream truncation guarantees the combined
            # payload stays under the model's hard cap.
            return f"{self.V2_DOC_PREFIX}{text}"
        return text

    def _prepare_query(self, text: str) -> str:
        """Prepend the v2 query prefix; identity for v1 models.

        Runs after the base class has already truncated (truncate_queries =
        True), reproducing the original embed_query's truncate-then-prefix
        order.
        """
        if self._is_v2():
            return f"{self.V2_QUERY_PREFIX}{text}"
        return text

    def _embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """One embed_content request; the config argument depends on v1/v2 and task."""
        if self._is_v2():
            # v2 models: task instruction already embedded in the prompt text
            # by _prepare_document/_prepare_query above; no config= argument.
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=texts,
            )
        elif is_query:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=self.types.EmbedContentConfig(
                    task_type="retrieval_query",
                ),
            )
        else:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=self.types.EmbedContentConfig(
                    task_type="retrieval_document",
                    title="Zotero library document",
                ),
            )
        return [e.values for e in response.embeddings]

    def _classify_error(self, exc: Exception) -> tuple[bool, float | None]:
        """Retry rate limits (429) and server errors (5xx); no reliable Retry-After."""
        try:
            from google.genai import errors
        except ImportError:
            return False, None

        if not isinstance(exc, errors.APIError):
            return False, None
        # `code` is not always populated (transport-level APIErrors carry
        # None), and comparing None against an int raises — which would
        # replace the real error with a TypeError from inside the handler.
        code = getattr(exc, "code", None)
        if not isinstance(code, int):
            return False, None
        if code == 429 or code >= 500:
            # Gemini does not reliably surface a Retry-After equivalent, so
            # fall back to the limiter's own exponential backoff.
            return True, None
        return False, None
