"""Provider-agnostic base classes for zotero-mcp's embedding functions.

Two classes live here. :class:`BaseEmbeddingFunction` is the minimal shared
surface every provider has, local ones included. :class:`RemoteEmbeddingFunction`
sits on top of it and adds what only an HTTP-backed provider needs: sub-batching,
optional parallelism across sub-batches, adaptive rate limiting and retries.
HuggingFace stays on the plain base — it runs a local model, so there is no
request to pace.

``BaseEmbeddingFunction`` holds the two method bodies that were already
byte-identical across the concrete providers when they lived in
``chroma_client.py``, plus ChromaDB's batch query entry point:

- ``embed_query_text`` -> ``self.__call__([text])[0]``, one query string to one
  vector. OpenAI, HuggingFace and Ollama each carried their own copy of exactly
  this. ``RemoteEmbeddingFunction`` overrides it to route the request through
  the limiter, and Gemini's query path differs from its document path — a
  different task type (v1), a different prompt prefix (v2) — which it expresses
  through ``_prepare_query``.
- ``embed_query``, which is ChromaDB's ``EmbeddingFunction`` protocol method:
  a sequence in, a list of vectors out, mapped onto ``embed_query_text``.
- a character-ratio ``truncate``. Gemini and Ollama each carried their own copy
  at 4 chars/token. OpenAI overrides it with tiktoken and HuggingFace with the
  model's own tokenizer.

Neither class is registered with ChromaDB — neither has ``name()``,
``get_config()`` or ``build_from_config()``, so neither can be resolved by
name when a persisted collection's config is rebuilt. Only the concrete
subclasses are registered.

``chars_per_token`` is read as a class attribute rather than an instance one on
purpose: several tests construct providers via ``Cls.__new__(Cls)``, setting
only the handful of instance attributes the assertion needs, so anything these
classes touch has to resolve without ``__init__`` having run. Every attribute
``RemoteEmbeddingFunction`` reads therefore goes through ``getattr(...,
default)``, and its rate limiter is built lazily on first use.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlsplit

from zotero_mcp.embeddings.ratelimit import AdaptiveRateLimiter
from zotero_mcp.utils import install_hint

try:
    from chromadb import Documents, EmbeddingFunction, Embeddings
except ImportError as e:
    raise ImportError(
        f"chromadb is required for semantic search. {install_hint('semantic')}"
    ) from e

logger = logging.getLogger(__name__)

#: Config keys every :class:`RemoteEmbeddingFunction` subclass round-trips
#: through ``get_config()``/``build_from_config()``. All additive and read via
#: ``.get(...)``, so collections persisted before these keys existed still
#: rebuild.
COMMON_CONFIG_KEYS = (
    "request_batch_size",
    "rate_limit_rps",
    "max_parallel_requests",
    "max_retries",
    "tokens_per_minute",
)


class EmbeddingAPIUnreachable(RuntimeError):
    """The embedding API could not be reached, or refused the request.

    Raised for connection errors, timeouts, 401 and 403 once retries are
    spent. Carries the host only, never the URL or the key, so the message
    is safe to show.
    """

    def __init__(self, provider: str, host: str, reason: str):
        self.provider = provider
        self.host = host
        self.reason = reason
        super().__init__(str(self))

    def __str__(self) -> str:
        # Rebuilt rather than taken from args: ChromaDB appends " in upsert."
        # and the like to a passing exception's args.
        if self.host in ("localhost", "127.0.0.1", "::1"):
            hint = "Check that the local server (for example Ollama) is running."
        elif self.reason == "HTTP 401":
            hint = "Check the API key."
        else:
            hint = (
                "Check your network connection; if this API only accepts "
                "requests from a particular network, connect to it or its VPN."
            )
        return f"Could not reach the embedding API at {self.host} ({self.reason}). {hint}"

    def __reduce__(self):
        # The ChromaDB worker process (idle_unload_minutes) sends exceptions
        # back pickled; the default reduction would call __init__ with the
        # message alone.
        return (type(self), (self.provider, self.host, self.reason))


def _safe_int(value: Any) -> int | None:
    """int(value), or None if it is missing or unparseable."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    """float(value), or None if it is missing, unparseable or zero."""
    if value is None:
        return None
    try:
        return float(value) or None
    except (TypeError, ValueError):
        return None


class BaseEmbeddingFunction(EmbeddingFunction):
    """Shared behaviour for the zotero-mcp embedding functions."""

    #: Characters per token, for the estimate-based :meth:`truncate` below.
    chars_per_token = 4

    def embed_query(self, input: Documents) -> Embeddings:
        """Embed query texts — ChromaDB's ``EmbeddingFunction`` contract.

        The parameter name ``input`` is load-bearing: ChromaDB calls this by
        keyword, including on the embedding function it rebuilds from a
        persisted collection's config. Maps :meth:`embed_query_text` over the
        input, one request per query string, as ``ChromaClient.search``
        already did.

        Do NOT delegate to ``super().embed_query``: it does not exist on
        chromadb < 1.1, which is inside our supported range.
        """
        if isinstance(input, str):
            # A string is iterable, so without this the comprehension below
            # would embed it one character at a time and return nonsense.
            raise TypeError(
                "embed_query() takes a sequence of query texts, not a single "
                "string; use embed_query_text() to embed one string."
            )
        return [self.embed_query_text(text) for text in input]

    def embed_query_text(self, text: str) -> list[float]:
        """Embed one query string via the document path.

        Correct for any provider that does not tune queries and documents
        differently; :class:`RemoteEmbeddingFunction` overrides it.
        """
        return self.__call__([text])[0]

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate using character-based estimation (``chars_per_token``).

        The fallback for providers with no tokenizer of their own to consult.
        """
        max_chars = max_tokens * self.chars_per_token
        if len(text) > max_chars:
            text = text[:max_chars]
        return text


class RemoteEmbeddingFunction(BaseEmbeddingFunction):
    """Base class for embedding functions backed by a remote HTTP API.

    Owns everything identical across the OpenAI/Gemini/Ollama providers that
    would otherwise be copy-pasted three times: splitting a large input into
    request-sized sub-batches, optionally running those sub-batches in
    parallel, pacing them through a shared :class:`AdaptiveRateLimiter`, and
    retrying the ones that come back throttled. A concrete provider implements
    only the four hooks at the bottom of this class.

    Class-level knobs a subclass may override:

    - ``default_request_batch_size`` — sub-batch size when neither the caller
      nor a persisted config specified one. ``None`` means "send the whole
      input as a single request".
    - ``default_tokens_per_minute`` — the provider's published TPM ceiling,
      used to arm the limiter's token bucket when nothing else supplies one.
    - ``max_parallel_requests_default`` / ``max_retries_default``.
    - ``truncate_queries`` — whether :meth:`embed_query_text` truncates before
      preparing the text. Only Gemini needs this, because its query path
      bypasses the indexing pipeline's own truncation.
    - ``default_api_host`` — the host named in an "unreachable" message when
      no ``base_url`` is configured.
    """

    default_request_batch_size: int | None = None
    default_api_host: str | None = None
    #: Retries for :meth:`check_reachable` and for a query that cannot reach
    #: the API, so either is reported in seconds, not after the full backoff.
    probe_max_retries = 1
    #: Estimated tokens one request may carry, or None if only the document
    #: count limits it. See :meth:`_split_into_requests`.
    max_request_tokens: int | None = None
    default_tokens_per_minute: float | None = None
    max_parallel_requests_default = 1
    max_retries_default = 5
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

        Called by each subclass's ``__init__`` once it has resolved its own
        provider-specific arguments (API key, client construction, ...).
        """
        self.model_name = model_name
        self.base_url = base_url
        self.request_batch_size = (
            int(request_batch_size)
            if request_batch_size
            else self.default_request_batch_size
        )
        self.rate_limit_rps = float(rate_limit_rps) if rate_limit_rps else None
        self.max_parallel_requests = (
            int(max_parallel_requests)
            if max_parallel_requests
            else self.max_parallel_requests_default
        )
        self.max_retries = (
            int(max_retries) if max_retries is not None else self.max_retries_default
        )

        # Explicit argument wins over the environment, which wins over the
        # provider's published ceiling. The TPM ceiling is documented per model
        # and tier, so it is configured rather than discovered — see
        # ratelimit.py on why a 429 is not used to infer one.
        if tokens_per_minute is not None:
            resolved_tpm: float | None = float(tokens_per_minute)
        else:
            resolved_tpm = _safe_float(os.getenv("ZOTERO_TOKENS_PER_MINUTE"))
            if resolved_tpm is None:
                resolved_tpm = self.default_tokens_per_minute
        self.tokens_per_minute = resolved_tpm

        self.limiter = self._build_limiter(
            self.rate_limit_rps, self.max_parallel_requests, self.tokens_per_minute
        )

    @staticmethod
    def _build_limiter(
        rate_limit_rps: float | None,
        max_parallel_requests: int,
        tokens_per_minute: float | None,
    ) -> AdaptiveRateLimiter:
        """Build the limiter, pinning both ceilings to the configured values.

        ``max_rps``/``max_tpm`` equal the configured rate, so AIMD may back off
        below what the operator asked for and climb back to it, but never past
        it. ``burst`` is at least the parallelism, or workers would serialize
        behind a bucket that only ever holds one token.
        """
        return AdaptiveRateLimiter(
            tpm=tokens_per_minute,
            max_tpm=tokens_per_minute,
            initial_rps=rate_limit_rps,
            max_rps=rate_limit_rps,
            burst=max(4, max_parallel_requests),
        )

    def _get_limiter(self) -> AdaptiveRateLimiter:
        """The shared limiter, built on demand if ``__init__`` never ran."""
        limiter = self.__dict__.get("limiter")
        if limiter is None:
            limiter = self._build_limiter(
                getattr(self, "rate_limit_rps", None),
                getattr(self, "max_parallel_requests", 1) or 1,
                getattr(self, "tokens_per_minute", None),
            )
            self.limiter = limiter
        return limiter

    def _common_config(self) -> dict[str, Any]:
        """The additive config keys for a subclass's ``get_config()``.

        Every value is read via ``getattr`` with a default so instances built
        by tests through ``__new__`` (skipping ``__init__``) never raise.
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

    def __call__(self, input: Documents) -> Embeddings:
        """Embed documents: prepare, split into sub-batches, embed with retry.

        Sub-batches run sequentially when ``max_parallel_requests <= 1`` (the
        default, matching every provider's pre-refactor behaviour) or when
        there is only one of them. Otherwise a thread pool runs them
        concurrently and writes into index-addressed slots, so the returned
        vector order always matches ``input`` order no matter which request
        finishes first.
        """
        prepared = [self._prepare_document(text) for text in input]
        batch_size = (
            getattr(self, "request_batch_size", None) or self.default_request_batch_size
        )

        sub_batches = self._split_into_requests(prepared, batch_size)

        if not sub_batches:
            return []

        max_parallel = getattr(self, "max_parallel_requests", 1) or 1
        if max_parallel <= 1 or len(sub_batches) == 1:
            embeddings: Embeddings = []
            for sub_batch in sub_batches:
                embeddings.extend(self._embed_with_retry(sub_batch))
            return embeddings

        slots: list[list[list[float]] | None] = [None] * len(sub_batches)
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {
                executor.submit(self._embed_with_retry, sub_batch): index
                for index, sub_batch in enumerate(sub_batches)
            }
            for future, index in futures.items():
                slots[index] = future.result()

        embeddings = []
        for chunk in slots:
            embeddings.extend(chunk or [])
        return embeddings

    def _split_into_requests(self, texts: list, batch_size: int | None) -> list[list]:
        """Split ``texts`` into requests, limited by count and by estimated tokens.

        A falsy ``batch_size`` means no count limit. An empty input gives no
        requests rather than one empty one. A document over the token limit
        goes out alone, where the provider can reject it without taking others
        down with it.
        """
        token_limit = self.max_request_tokens
        requests: list[list] = []
        current: list = []
        current_tokens = 0
        for text in texts:
            tokens = self._estimate_request_tokens(text) if token_limit else 0
            over_count = bool(batch_size) and len(current) >= batch_size
            over_tokens = bool(token_limit) and current_tokens + tokens > token_limit
            if current and (over_count or over_tokens):
                requests.append(current)
                current, current_tokens = [], 0
            current.append(text)
            current_tokens += tokens
        if current:
            requests.append(current)
        return requests

    def _estimate_request_tokens(self, text: Any) -> int:
        """A document's share of a request's token limit, without tokenizing it.

        Tokenizing every document here would be expensive on PDF text (see
        ``OpenAIEmbeddingFunction.truncate``), and neither of OpenAI's two
        checks needs it. This is the larger of two bounds, so a request within
        the limit passes both:

        - It counts UTF-8 bytes / 4, however few real tokens they are.
        - It counts real tokens. A token is at least one byte, and documents
          reaching this point are truncated to the input limit, so the smaller
          of those two can't be below the real count.
        """
        data = len(str(text).encode("utf-8"))
        return max(min(data, getattr(self, "max_input_tokens", data)), data // 4 + 1)

    def embed_query_text(self, text: str) -> list[float]:
        """Embed one query string through the same limiter and retry path.

        Never uses the thread pool — there is only one request — but a query
        that hits a 429 is still retried like any document sub-batch.
        ``embed_query`` on the base class maps this over a batch.
        """
        if self.truncate_queries:
            text = self.truncate(text, getattr(self, "max_input_tokens", 8000))
        return self._embed_with_retry([self._prepare_query(text)], is_query=True)[0]

    def api_host(self) -> str:
        """The API's host name, without scheme, port, path or credentials."""
        base_url = getattr(self, "base_url", None)
        if base_url:
            text = str(base_url)
            hostname = urlsplit(text if "//" in text else f"//{text}").hostname
            if hostname:
                return hostname
        return self.default_api_host or "the configured embedding API"

    def check_reachable(self) -> None:
        """Embed one short string to find out whether the API answers at all.

        Raises :class:`EmbeddingAPIUnreachable` when it cannot be reached or
        refuses the request. Any other error is only logged and left to the
        run itself, which handles it as before.
        """
        try:
            self._embed_with_retry(
                [self._prepare_document("zotero-mcp connectivity check")], probe=True
            )
        except EmbeddingAPIUnreachable:
            raise
        except Exception as exc:
            logger.warning(f"Embedding API check failed, continuing: {exc}")

    def _embed_with_retry(
        self, texts: list[str], is_query: bool = False, probe: bool = False
    ) -> list[list[float]]:
        """acquire -> _embed_batch -> on_success, or on_throttle and retry.

        For a query or the ``probe`` request, an error
        :meth:`_unreachable_reason` recognises is retried at most
        ``probe_max_retries`` times and then raised as
        :class:`EmbeddingAPIUnreachable`. Document requests during a run keep
        their errors and the full retry budget.
        """
        limiter = self._get_limiter()
        max_retries = getattr(self, "max_retries", self.max_retries_default) or 0
        if probe:
            max_retries = min(max_retries, self.probe_max_retries)
        # A rough estimate, not a token count: enough to pace the TPM bucket
        # without paying tiktoken's per-request CPU cost. Text reaching here
        # has already been truncated upstream.
        estimated_tokens = int(
            sum(len(str(text)) for text in texts) / self.chars_per_token
        )
        attempt = 0
        limiter_wait_ms = 0.0

        while True:
            limiter_wait_ms += limiter.acquire(estimated_tokens=estimated_tokens) * 1000.0

            started = time.monotonic()
            try:
                response = self._embed_batch(texts, is_query=is_query)
            except Exception as exc:
                retryable, retry_after = self._classify_error(exc)
                reason = self._unreachable_reason(exc) if (is_query or probe) else None
                budget = min(max_retries, self.probe_max_retries) if reason else max_retries
                if retryable and attempt < budget:
                    limiter.wait(limiter.on_throttle(retry_after))
                    attempt += 1
                    continue
                if reason:
                    name = getattr(self, "name", None)
                    provider = name() if callable(name) else type(self).__name__
                    raise EmbeddingAPIUnreachable(provider, self.api_host(), reason) from exc
                raise

            http_ms = (time.monotonic() - started) * 1000.0
            # A provider whose SDK exposes response headers returns them
            # alongside the vectors so the limiter can read its headroom.
            if isinstance(response, tuple) and len(response) == 2:
                vectors, headers = response
            else:
                vectors, headers = response, None

            limiter.on_success(headers)
            self._log_telemetry(
                len(texts), estimated_tokens, http_ms, limiter_wait_ms, headers
            )
            return vectors

    def _log_telemetry(
        self,
        chunk_count: int,
        estimated_tokens: int,
        http_ms: float,
        limiter_wait_ms: float,
        headers: Any | None,
    ) -> None:
        """One INFO line per completed sub-batch, for ``update-db -v``.

        Names the worker thread so parallelism is visible, and separates time
        spent waiting on the limiter from time spent on the wire — the two
        things worth telling apart when a run is slower than expected.
        """
        if not logger.isEnabledFor(logging.INFO):
            return

        name_attr = getattr(self, "name", None)
        provider = name_attr() if callable(name_attr) else self.__class__.__name__
        message = (
            f"[{provider} API] [{threading.current_thread().name}] "
            f"Embedded {chunk_count} chunks (~{estimated_tokens} tokens) "
            f"in {http_ms:.1f}ms"
        )

        server_ms = None
        remaining_tokens = limit_tokens = None
        if headers and hasattr(headers, "get"):

            def get_header(key: str) -> Any:
                # First non-None rather than an `or` chain, so a genuine 0
                # (no headroom left) is reported instead of being dropped.
                for spelling in (key, key.lower(), key.title()):
                    value = headers.get(spelling)
                    if value is not None:
                        return value
                return None

            server_ms = _safe_int(get_header("openai-processing-ms"))
            remaining_tokens = _safe_int(get_header("x-ratelimit-remaining-tokens"))
            limit_tokens = _safe_int(get_header("x-ratelimit-limit-tokens"))

        if server_ms is not None:
            message += f" (server: {server_ms}ms, wait: {limiter_wait_ms:.1f}ms)"
        else:
            message += f" (wait: {limiter_wait_ms:.1f}ms)"

        if remaining_tokens is not None and limit_tokens is not None and limit_tokens > 0:
            load = (1.0 - remaining_tokens / limit_tokens) * 100.0
            message += (
                f" | Token load: {load:.1f}% "
                f"({remaining_tokens:,}/{limit_tokens:,} left)"
            )

        logger.info(message)

    # -- provider hooks ----------------------------------------------------

    def _embed_batch(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]] | tuple[list[list[float]], Any]:
        """Issue exactly one request for ``texts`` and return their vectors.

        May instead return ``(vectors, headers)`` when the provider's SDK
        exposes response headers, so the limiter can read its rate-limit
        headroom from them.

        ``is_query`` distinguishes a query-time call from a document-time one
        for providers whose request shape differs by task (Gemini v1's
        ``task_type``). Providers where it never differs can ignore it.
        """
        raise NotImplementedError

    def _prepare_document(self, text: str) -> str:
        """Transform a document before sending it (identity by default)."""
        return text

    def _prepare_query(self, text: str) -> str:
        """Transform a query before sending it (identity by default)."""
        return text

    def _classify_error(self, exc: Exception) -> tuple[bool, float | None]:
        """Decide whether ``exc`` is worth retrying.

        Returns ``(retryable, retry_after)``. The default is deliberately
        conservative — nothing is retryable — so a provider that has not
        opted in fails fast rather than retrying something that will never
        succeed. Providers override this with their SDK's exception types.
        """
        return False, None

    def _unreachable_reason(self, exc: Exception) -> str | None:
        """A short reason if ``exc`` means the API cannot be reached, else None.

        One of "connection failed", "timed out", "HTTP 401", "HTTP 403", never
        the exception's own text, which may carry a URL. The default
        recognises nothing; providers override.
        """
        return None
