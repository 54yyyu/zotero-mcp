"""Gemini realtime embedding function.

Moved verbatim out of ``chroma_client.py`` (Phase 5 of the embedding-provider
refactor) — no renames, no behavior changes. ``chroma_client.py`` re-exports
``GeminiEmbeddingFunction`` so every existing import path keeps working.

``V2_DOC_PREFIX``/``V2_QUERY_PREFIX``/``V2_PREFIX_TOKEN_BUDGET`` live here as
class attributes; ``gemini_batch.apply_text_shaping`` imports them from this
module (rather than from ``chroma_client``) to remove the batch->realtime
coupling that existed before this move.
"""

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
    # These are the canonical prefixes; _prepare_document/_prepare_query
    # prepend them to every v2 input. They MUST stay in sync with
    # V2_PREFIX_TOKEN_BUDGET below: if you lengthen a prefix, bump the budget
    # so truncation still leaves room for it under the model's hard cap.
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

    # embed_query truncates before prepending the task prefix (see
    # embed_query's docstring in the base class + _prepare_query below);
    # OpenAI/Ollama don't truncate at query time, only Gemini does.
    truncate_queries = True

    # Gemini's embed_content API caps at 100 items per batch (verified
    # empirically: batch=100 OK, batch=250 → 400 INVALID_ARGUMENT with
    # "at most 100 requests can be in one batch").
    GEMINI_MAX_BATCH = 100
    default_request_batch_size = GEMINI_MAX_BATCH

    def __init__(self, model_name: str = "gemini-embedding-001", api_key: str | None = None,
                 base_url: str | None = None, request_batch_size: int | None = None,
                 rate_limit_rps: float | None = None, max_parallel_requests: int | None = None,
                 max_retries: int | None = None):
        # Model-aware token limit. For v2 models, derive from:
        #   hard_cap (8192) - safety_margin (192, for char-based truncation
        #   imprecision) - V2_PREFIX_TOKEN_BUDGET (20, reserved for the
        #   in-prompt task instruction prepended in _prepare_document/_prepare_query).
        # Net effective budget for text body: 8192 - 192 - 20 = 7980 tokens.
        # This guarantees post-prefix payload <= hard cap even at the
        # truncation limit, formally closing the cap-enforcement gap.
        if "gemini-embedding-2" in model_name:
            self.max_input_tokens = 8000 - self.V2_PREFIX_TOKEN_BUDGET
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        base_url = base_url or os.getenv("GEMINI_BASE_URL")
        if not self.api_key:
            raise ValueError("Gemini API key is required")

        try:
            from google import genai
            from google.genai import types
            client_kwargs = {"api_key": self.api_key}
            if base_url:
                http_options = types.HttpOptions(baseUrl=base_url)
                client_kwargs["http_options"] = http_options
            self.client = genai.Client(**client_kwargs)
            self.types = types
        except ImportError:
            raise ImportError("google-genai package is required for Gemini embeddings")

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
        return "gemini"

    def get_config(self) -> dict[str, Any]:
        cfg = {"model_name": self.model_name, "base_url": self.base_url}
        cfg.update(self._common_config())
        return cfg

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
        )

    def _is_v2(self) -> bool:
        # gemini-embedding-2-* does not support the task_type config field
        # (it is silently ignored by the API). Google's guidance is to put
        # the task hint in the prompt text instead.
        return "gemini-embedding-2" in self.model_name

    def _prepare_document(self, text: str) -> str:
        # v2 models: task instruction goes in the prompt, no config.
        # V2_PREFIX_TOKEN_BUDGET is already reserved from max_input_tokens
        # in __init__, so upstream truncation guarantees the combined
        # payload stays under the model's hard cap.
        return f"{self.V2_DOC_PREFIX}{text}" if self._is_v2() else text

    def _prepare_query(self, text: str) -> str:
        return f"{self.V2_QUERY_PREFIX}{text}" if self._is_v2() else text

    def _embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """Issue one Gemini embed_content request.

        v2 models never take an EmbedContentConfig (the task hint already
        lives in the prompt text via _prepare_document/_prepare_query); v1
        models select retrieval_document vs retrieval_query via ``is_query``
        since v1's request shape (unlike v2's) differs between the two.
        """
        if self._is_v2():
            response = self.client.models.embed_content(model=self.model_name, contents=texts)
        elif is_query:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=self.types.EmbedContentConfig(task_type="retrieval_query"),
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
        """google.genai.errors.APIError with code 429 or >=500 is retryable."""
        try:
            from google.genai import errors as genai_errors
        except ImportError:
            return False, None
        if isinstance(exc, genai_errors.APIError):
            code = getattr(exc, "code", None)
            if code == 429 or (isinstance(code, int) and code >= 500):
                return True, None
        return False, None
