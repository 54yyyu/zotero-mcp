"""Gemini embedding function, backed by the google-genai SDK."""

import os
from typing import Any

from chromadb import Documents, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

from zotero_mcp.embeddings.base import BaseEmbeddingFunction


@register_embedding_function
class GeminiEmbeddingFunction(BaseEmbeddingFunction):
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

    def __init__(self, model_name: str = "gemini-embedding-001", api_key: str | None = None, base_url: str | None = None):
        self.model_name = model_name
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
        self.base_url = base_url or os.getenv("GEMINI_BASE_URL")
        if not self.api_key:
            raise ValueError("Gemini API key is required")

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
        return {"model_name": self.model_name, "base_url": self.base_url}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "GeminiEmbeddingFunction":
        return GeminiEmbeddingFunction(
            model_name=config.get("model_name", "gemini-embedding-001"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
        )

    # Gemini's embed_content API caps at 100 items per batch (verified
    # empirically: batch=100 OK, batch=250 → 400 INVALID_ARGUMENT with
    # "at most 100 requests can be in one batch").
    GEMINI_MAX_BATCH = 100

    def _is_v2(self) -> bool:
        # gemini-embedding-2-* does not support the task_type config field
        # (it is silently ignored by the API). Google's guidance is to put
        # the task hint in the prompt text instead.
        return "gemini-embedding-2" in self.model_name

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using Gemini API, batching up to 100 per call."""
        is_v2 = self._is_v2()
        # Materialize once so we can slice regardless of input iterable type.
        texts = list(input)
        if is_v2:
            # v2 models: task instruction goes in the prompt, no config.
            # V2_PREFIX_TOKEN_BUDGET is already reserved from max_input_tokens
            # in __init__, so upstream truncation guarantees the combined
            # payload stays under the model's hard cap.
            prepared = [f"{self.V2_DOC_PREFIX}{t}" for t in texts]
        else:
            prepared = texts

        embeddings: list = []
        for start in range(0, len(prepared), self.GEMINI_MAX_BATCH):
            batch = prepared[start:start + self.GEMINI_MAX_BATCH]
            if is_v2:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                )
            else:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                    config=self.types.EmbedContentConfig(
                        task_type="retrieval_document",
                        title="Zotero library document",
                    ),
                )
            embeddings.extend(e.values for e in response.embeddings)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string using retrieval_query task type."""
        # Truncate before any prefix prepending. For v2 models max_input_tokens
        # already excludes V2_PREFIX_TOKEN_BUDGET (reserved in __init__), so
        # the post-prefix payload stays under the model's hard cap. For v1
        # models truncation prevents API errors on pathological queries that
        # the upstream pipeline does not pre-truncate (queries bypass the
        # _process_item_batch truncate_text path that documents go through).
        text = self.truncate(text, self.max_input_tokens)
        if self._is_v2():
            prompt_text = f"{self.V2_QUERY_PREFIX}{text}"
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=[prompt_text],
            )
        else:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=[text],
                config=self.types.EmbedContentConfig(
                    task_type="retrieval_query",
                ),
            )
        return response.embeddings[0].values
