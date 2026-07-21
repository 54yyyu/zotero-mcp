"""Shared machinery for remote (HTTP-API-backed) embedding functions.

``RemoteEmbeddingFunction`` owns everything that is identical across the
OpenAI/Gemini/Ollama embedding functions and would otherwise be copy-pasted
per provider: sub-batching large inputs into request-sized chunks, optional
parallelism across sub-batches, adaptive rate limiting, and retrying
retryable (429/5xx-ish) errors. A concrete provider only implements the four
hooks at the bottom of this file.

Not registered with ChromaDB itself (no ``name()``/``get_config()``/
``build_from_config()``) — only concrete subclasses are registered, so this
class is never resolvable by name from a persisted collection's config.

Design note: some existing tests construct provider EFs via
``Cls.__new__(Cls)`` (bypassing ``__init__`` entirely) and set only a subset
of instance attributes before calling ``__call__``/``embed_query`` directly.
Every attribute this class touches is therefore read with ``getattr(...,
default)`` rather than assumed to exist, and the rate limiter is created
lazily on first use instead of requiring ``__init__`` to have run.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

def _safe_int(val: Any) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


try:
    from chromadb import Documents, EmbeddingFunction, Embeddings
except ImportError as e:  # pragma: no cover - chromadb is a hard dependency of this package
    raise ImportError(
        "chromadb is required for semantic search. "
        "Install it with: pip install 'zotero-mcp-server[semantic]'"
    ) from e

from zotero_mcp.embeddings.ratelimit import AdaptiveRateLimiter

# Common config keys every RemoteEmbeddingFunction subclass round-trips
# through get_config()/build_from_config(), read with .get(...) + a default
# so configs persisted before these keys existed still build.
COMMON_CONFIG_KEYS = (
    "request_batch_size", "rate_limit_rps", "max_parallel_requests", "max_retries", "tokens_per_minute",
)


class RemoteEmbeddingFunction(EmbeddingFunction):
    """Base class for embedding functions backed by a remote HTTP API.

    Class-level knobs a subclass may override:
      - ``default_request_batch_size``: sub-batch size when the caller (or
        persisted config) didn't specify one. ``None`` means "send the whole
        input as a single request" (Ollama's default).
      - ``max_parallel_requests_default`` / ``max_retries_default``: defaults
        for the two additive config keys.
      - ``chars_per_token``: used by the default char-ratio ``truncate()``.
      - ``truncate_queries``: whether ``embed_query`` truncates before
        preparing the text. Only Gemini needs this (its embed_query bypasses
        the pipeline's own truncate_text call).
    """

    default_request_batch_size: int | None = None
    default_tokens_per_minute: float | None = None
    max_parallel_requests_default = 1
    max_retries_default = 5
    chars_per_token = 4.0
    truncate_queries = False

    def _init_common(
        self,
        *,
        model_name: str,
        base_url: str | None,
        request_batch_size: int | None,
        rate_limit_rps: float | None,
        max_parallel_requests: int | None,
        max_retries: int | None,
        tokens_per_minute: float | None = None,
    ) -> None:
        """Set the attributes common to every remote provider.

        Called by each subclass's ``__init__`` after resolving its own
        provider-specific args (api key, client construction, ...).
        """
        import os
        self.model_name = model_name
        self.base_url = base_url
        self.request_batch_size = (
            int(request_batch_size) if request_batch_size else self.default_request_batch_size
        )
        self.rate_limit_rps = float(rate_limit_rps) if rate_limit_rps else None
        self.max_parallel_requests = (
            int(max_parallel_requests) if max_parallel_requests else self.max_parallel_requests_default
        )
        self.max_retries = (
            int(max_retries) if max_retries is not None else self.max_retries_default
        )
        if tokens_per_minute is not None:
            tpm_val = float(tokens_per_minute)
        elif os.getenv("ZOTERO_TOKENS_PER_MINUTE"):
            tpm_val = float(os.getenv("ZOTERO_TOKENS_PER_MINUTE", 0)) or None
        else:
            tpm_val = getattr(self, "default_tokens_per_minute", None)

        self.tokens_per_minute = tpm_val

        # A user-configured rate_limit_rps is a hard cap: adaptation may only
        # lower/restore toward it, never exceed it. Burst is at least the
        # parallelism so workers aren't serialized on a single token.
        self.limiter = AdaptiveRateLimiter(
            initial_rps=self.rate_limit_rps,
            max_rps=self.rate_limit_rps,
            burst=max(4, self.max_parallel_requests),
            tpm=self.tokens_per_minute,
            max_tpm=self.tokens_per_minute,
        )

    def _common_config(self) -> dict[str, Any]:
        """Common config keys for a subclass's get_config(). Additive-only:
        every value is read via getattr with a default so instances built by
        tests via __new__ (skipping __init__) never raise AttributeError.
        """
        return {
            "request_batch_size": getattr(self, "request_batch_size", None),
            "rate_limit_rps": getattr(self, "rate_limit_rps", None),
            "max_parallel_requests": getattr(
                self, "max_parallel_requests", self.max_parallel_requests_default
            ),
            "max_retries": getattr(self, "max_retries", self.max_retries_default),
            "tokens_per_minute": getattr(self, "tokens_per_minute", None),
        }

    def _get_limiter(self) -> AdaptiveRateLimiter:
        limiter = self.__dict__.get("limiter")
        if limiter is None:
            rps = getattr(self, "rate_limit_rps", None)
            max_parallel = getattr(self, "max_parallel_requests", 1) or 1
            tpm = getattr(self, "tokens_per_minute", None)
            limiter = AdaptiveRateLimiter(
                initial_rps=rps, max_rps=rps, burst=max(4, max_parallel), tpm=tpm, max_tpm=tpm,
            )
            self.limiter = limiter
        return limiter

    def __call__(self, input: Documents) -> Embeddings:
        """Embed a list of documents: prepare -> sub-batch -> embed (retrying).

        Sub-batches run sequentially when ``max_parallel_requests <= 1`` (the
        default, matching every provider's pre-refactor behavior) or when
        there is only one sub-batch anyway. Otherwise a thread pool runs them
        concurrently with index-addressed result slots, so the returned
        vector order always matches ``input`` order regardless of which
        sub-batch's HTTP request completes first.
        """
        prepared = [self._prepare_document(t) for t in input]
        batch_size = getattr(self, "request_batch_size", None) or self.default_request_batch_size

        if batch_size:
            sub_batches = [prepared[i : i + batch_size] for i in range(0, len(prepared), batch_size)]
        else:
            # Falsy/None request_batch_size means "whole input, one request"
            # (Ollama's default) — but an empty input must still issue zero
            # requests rather than one empty one.
            sub_batches = [prepared] if prepared else []

        if not sub_batches:
            return []

        max_parallel = getattr(self, "max_parallel_requests", 1) or 1
        if max_parallel <= 1 or len(sub_batches) == 1:
            embeddings: list[list[float]] = []
            for sub in sub_batches:
                embeddings.extend(self._embed_with_retry(sub))
            return embeddings

        slots: list[list[list[float]] | None] = [None] * len(sub_batches)
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            future_to_index = {
                executor.submit(self._embed_with_retry, sub): idx for idx, sub in enumerate(sub_batches)
            }
            for future in future_to_index:
                slots[future_to_index[future]] = future.result()

        embeddings = []
        for chunk in slots:
            embeddings.extend(chunk or [])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string, through the same retry/limiter path.

        Never uses the thread pool (there is only one request), but still
        goes through ``_embed_with_retry`` so a query hitting a 429 is
        retried like any document sub-batch.
        """
        if self.truncate_queries:
            text = self.truncate(text, getattr(self, "max_input_tokens", 8000))
        text = self._prepare_query(text)
        return self._embed_with_retry([text], is_query=True)[0]

    def _embed_with_retry(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """acquire() -> _embed_batch() -> on_success()/on_throttle()+retry."""
        limiter = self._get_limiter()
        max_retries = getattr(self, "max_retries", self.max_retries_default) or 0
        # Rough estimate, not a precise token count: good enough to pace the
        # TPM bucket (see ratelimit.py) without tiktoken's per-request CPU
        # cost. Texts reaching here have already been truncated upstream.
        estimated_tokens = int(sum(len(str(t)) for t in texts) / self.chars_per_token)
        attempt = 0
        t0 = time.monotonic()
        while True:
            limiter.acquire(estimated_tokens=estimated_tokens)
            try:
                res = self._embed_batch(texts, is_query=is_query)
                headers = None
                if isinstance(res, tuple) and len(res) == 2:
                    result, headers = res
                else:
                    result = res
            except Exception as exc:
                retryable, retry_after = self._classify_error(exc)
                if retryable and attempt < max_retries:
                    delay = limiter.on_throttle(retry_after)
                    limiter.wait(delay)
                    attempt += 1
                    continue
                raise
            else:
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                limiter.on_success()
                self._log_telemetry(len(texts), estimated_tokens, elapsed_ms, headers)
                return result

    def _log_telemetry(
        self,
        chunk_count: int,
        estimated_tokens: int,
        elapsed_ms: float,
        headers: Any | None,
    ) -> None:
        provider_name = getattr(self, "name", None)
        name_str = provider_name() if callable(provider_name) else self.__class__.__name__

        info_str = f"\r\033[K[{name_str} API] Embedded {chunk_count} chunks (~{estimated_tokens} tokens) in {elapsed_ms:.1f}ms"


        if headers and hasattr(headers, "get"):
            get_h = lambda k: headers.get(k) or headers.get(k.lower()) or headers.get(k.title())

            rem_tok = _safe_int(get_h("x-ratelimit-remaining-tokens"))
            lim_tok = _safe_int(get_h("x-ratelimit-limit-tokens"))
            rem_req = _safe_int(get_h("x-ratelimit-remaining-requests"))
            lim_req = _safe_int(get_h("x-ratelimit-limit-requests"))

            parts = []
            if rem_tok is not None and lim_tok is not None and lim_tok > 0:
                tok_load = (1.0 - (rem_tok / lim_tok)) * 100.0
                parts.append(f"Token Load: {tok_load:.1f}% ({rem_tok:,}/{lim_tok:,} left)")
            if rem_req is not None and lim_req is not None and lim_req > 0:
                req_load = (1.0 - (rem_req / lim_req)) * 100.0
                parts.append(f"Req Load: {req_load:.1f}% ({rem_req:,}/{lim_req:,} left)")

            if parts:
                info_str += " | " + " | ".join(parts)

        logger.info(info_str)


    # -- provider hooks ------------------------------------------------------

    def _embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """Issue exactly one HTTP request for ``texts`` and return their vectors.

        ``is_query`` distinguishes a query-time call from a document-time
        call for providers whose request shape differs by task (Gemini v1's
        task_type). Providers where the shape never differs can ignore it.
        """
        raise NotImplementedError

    def _prepare_document(self, text: str) -> str:
        """Transform a document's text before it is sent (identity by default)."""
        return text

    def _prepare_query(self, text: str) -> str:
        """Transform a query's text before it is sent (identity by default)."""
        return text

    def _classify_error(self, exc: Exception) -> tuple[bool, float | None]:
        """Decide whether ``exc`` is worth retrying.

        Returns ``(retryable, retry_after)``. Conservative default: nothing
        is retryable. Providers override this with their SDK's exception
        types so an unrecognized error still fails fast instead of retrying
        something that will never succeed.
        """
        return False, None

    def truncate(self, text: str, max_tokens: int) -> str:
        """Default char-ratio truncation using the class's ``chars_per_token``."""
        max_chars = int(max_tokens * self.chars_per_token)
        if len(text) > max_chars:
            text = text[:max_chars]
        return text
