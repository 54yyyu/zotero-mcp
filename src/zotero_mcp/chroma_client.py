"""
ChromaDB client for semantic search functionality.

This module provides persistent vector database storage and embedding functions
for semantic search over Zotero libraries.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

try:
    import chromadb
    from chromadb import Documents, EmbeddingFunction, Embeddings
    from chromadb.config import Settings
    from chromadb.utils.embedding_functions import register_embedding_function
except ImportError as e:
    raise ImportError(
        "chromadb is required for semantic search. "
        "Install it with: pip install 'zotero-mcp-server[semantic]'"
    ) from e

from zotero_mcp.embeddings import RemoteEmbeddingFunction
from zotero_mcp.utils import suppress_stdout

logger = logging.getLogger(__name__)


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
                 rate_limit_rps: float | None = None, service_tier: str | None = None,
                 max_parallel_requests: int | None = None, max_retries: int | None = None):
        import threading
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.service_tier: str | None = service_tier or os.getenv("OPENAI_SERVICE_TIER")
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
        if getattr(self, "service_tier", None):
            cfg["service_tier"] = self.service_tier
        return cfg

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "OpenAIEmbeddingFunction":
        return OpenAIEmbeddingFunction(
            model_name=config.get("model_name", "text-embedding-3-small"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            request_batch_size=config.get("request_batch_size"),
            rate_limit_rps=config.get("rate_limit_rps"),
            service_tier=config.get("service_tier"),
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
        """
        req_kwargs = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }
        if getattr(self, "service_tier", None):
            req_kwargs["service_tier"] = self.service_tier
        response = self.client.embeddings.create(**req_kwargs)
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


@register_embedding_function
class HuggingFaceEmbeddingFunction(EmbeddingFunction):
    """Custom HuggingFace embedding function for ChromaDB using sentence-transformers.

    Registered under the name "huggingface" so ChromaDB rebuilds it (rather than
    its own incompatible built-in of the same name) when reloading a persisted
    collection's config (see OpenAIEmbeddingFunction for details).
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.model_name = model_name

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        except ImportError:
            raise ImportError("sentence-transformers package is required for HuggingFace embeddings. Install with: pip install sentence-transformers")

        # Read limit from model metadata; conservative fallback
        self.max_input_tokens = getattr(self.model, "max_seq_length", 500)

    @staticmethod
    def name() -> str:
        return "huggingface"

    def get_config(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HuggingFaceEmbeddingFunction":
        return HuggingFaceEmbeddingFunction(
            model_name=config.get("model_name", "Qwen/Qwen3-Embedding-0.6B"),
        )

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using HuggingFace model."""
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string. No special handling needed for HuggingFace."""
        return self.__call__([text])[0]

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate using the model's own tokenizer."""
        tokenizer = getattr(self.model, 'tokenizer', None)
        if tokenizer is not None:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) > max_tokens:
                encoded = encoded[:max_tokens]
                text = tokenizer.decode(encoded)
        else:
            max_chars = max_tokens * 2
            if len(text) > max_chars:
                text = text[:max_chars]
        return text


@register_embedding_function
class OllamaEmbeddingFunction(RemoteEmbeddingFunction):
    """Custom Ollama embedding function for ChromaDB.

    Uses Ollama's local HTTP API. Registered under the name ``ollama`` so
    ChromaDB can rebuild persisted collections that were created with this
    embedding function.
    """

    # Ollama models vary; use a conservative, char-based fallback budget.
    max_input_tokens = 8000

    # None means "send the whole input as a single request" — Ollama's
    # /api/embed already accepts an arbitrarily long batch locally, so there
    # is no request-size cap to sub-batch against by default.
    default_request_batch_size = None

    def __init__(self, model_name: str = "qwen3-embedding", base_url: str | None = None,
                 request_batch_size: int | None = None, rate_limit_rps: float | None = None,
                 max_parallel_requests: int | None = None, max_retries: int | None = None):
        base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
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
        return "ollama"

    def get_config(self) -> dict[str, Any]:
        cfg = {"model_name": self.model_name, "base_url": self.base_url}
        cfg.update(self._common_config())
        return cfg

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "OllamaEmbeddingFunction":
        return OllamaEmbeddingFunction(
            model_name=config.get("model_name", "qwen3-embedding"),
            base_url=config.get("base_url"),
            request_batch_size=config.get("request_batch_size"),
            rate_limit_rps=config.get("rate_limit_rps"),
            max_parallel_requests=config.get("max_parallel_requests"),
            max_retries=config.get("max_retries"),
        )

    def _embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """Issue one request to Ollama's /api/embed endpoint.

        Unlike the deprecated /api/embeddings route (single ``prompt`` -> single
        ``embedding``), /api/embed accepts a batch via ``input`` and returns a
        list under ``embeddings``, so the whole batch is sent in one request
        instead of one request per document.
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests package is required for Ollama embeddings")

        if not texts:
            return []

        endpoint = f"{self.base_url}/api/embed"
        response = requests.post(
            endpoint,
            json={"model": self.model_name, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise ValueError(
                f"Ollama /api/embed returned no 'embeddings' field: {data}"
            )
        return embeddings

    def _classify_error(self, exc: Exception) -> tuple[bool, float | None]:
        """429/5xx HTTP responses and connection-level failures are retryable —
        Ollama runs locally, so connection errors are usually the server still
        loading the model rather than something permanently wrong.
        """
        try:
            import requests
        except ImportError:
            return False, None
        if isinstance(exc, requests.HTTPError):
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429 or (isinstance(status, int) and status >= 500):
                return True, None
            return False, None
        if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
            return True, None
        return False, None


class ChromaClient:
    """ChromaDB client for Zotero semantic search."""

    def __init__(self,
                 collection_name: str = "zotero_library",
                 persist_directory: str | None = None,
                 embedding_model: str = "default",
                 embedding_config: dict[str, Any] | None = None):
        """
        Initialize ChromaDB client.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Model to use for embeddings ('default', 'openai', 'gemini', 'ollama', 'qwen', 'embeddinggemma', or HuggingFace model name)
            embedding_config: Configuration for the embedding model
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_config = embedding_config or {}

        # Set up persistent directory
        if persist_directory is None:
            # Use user's config directory by default
            config_dir = Path.home() / ".config" / "zotero-mcp"
            config_dir.mkdir(parents=True, exist_ok=True)
            persist_directory = str(config_dir / "chroma_db")

        self.persist_directory = persist_directory

        # Initialize ChromaDB client with stdout suppression and corrupted-DB auto-recovery
        with suppress_stdout():
            try:
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            except Exception as e:
                if "already exists" in str(e).lower() or "operationalerror" in str(e).lower():
                    logger.warning(f"ChromaDB database at {self.persist_directory} is corrupted ({e}). Wiping directory for clean rebuild...")
                    shutil.rmtree(self.persist_directory, ignore_errors=True)
                    self.client = chromadb.PersistentClient(
                        path=self.persist_directory,
                        settings=Settings(
                            anonymized_telemetry=False,
                            allow_reset=True
                        )
                    )
                else:
                    raise

            # Set up embedding function
            self.embedding_function = self._create_embedding_function()

            # Get or create collection with the configured embedding function.
            # If the user switched embedding models, the persisted collection
            # will have stale config.  Detect the mismatch and drop/recreate.
            try:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )

                # ChromaDB may silently persist the old embedding function config.
                # Check if the stored config matches what we want; if not, recreate.
                stored_config = getattr(self.collection, 'metadata', {}) or {}
                if not stored_config:
                    # Try reading config from the collection's config_json_str
                    try:
                        import json as _json
                        rows = self.client._sysdb.get_collections(name=self.collection_name)
                        if rows:
                            raw = getattr(rows[0], 'config_json_str', None) or '{}'
                            cfg = _json.loads(raw)
                            ef_cfg = cfg.get('embedding_function', {}).get('config', {})
                            stored_model = ef_cfg.get('model_name', '')
                            # Compare stored model with configured model
                            configured_model = getattr(self.embedding_function, 'model_name', None)
                            if stored_model and configured_model and stored_model != configured_model:
                                msg = (
                                    f"Stored embedding model '{stored_model}' differs from configured '{configured_model}'. "
                                    "Resetting ChromaDB vector collection for rebuild..."
                                )
                                logger.warning(msg)
                                print(msg, flush=True)
                                try:
                                    self.client.reset()
                                except Exception:
                                    self.client.delete_collection(name=self.collection_name)
                                self.collection = self.client.create_collection(
                                    name=self.collection_name,
                                    embedding_function=self.embedding_function
                                )
                    except Exception:
                        pass  # Best-effort check; proceed with existing collection

            except Exception as e:
                if "embedding function conflict" in str(e).lower():
                    msg = (
                        f"Embedding model changed to '{self.embedding_model}'. "
                        "Resetting ChromaDB vector collection for rebuild (your Zotero library and folders remain untouched)..."
                    )
                    logger.warning(msg)
                    print(msg, flush=True)
                    try:
                        self.client.reset()
                    except Exception:
                        self.client.delete_collection(name=self.collection_name)
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        embedding_function=self.embedding_function
                    )
                else:
                    raise

    def _create_embedding_function(self) -> EmbeddingFunction:
        """Create the appropriate embedding function based on configuration."""
        if self.embedding_model == "openai":
            model_name = self.embedding_config.get("model_name", "text-embedding-3-small")
            api_key = self.embedding_config.get("api_key")
            base_url = self.embedding_config.get("base_url")
            return OpenAIEmbeddingFunction(
                model_name=model_name, api_key=api_key, base_url=base_url,
                request_batch_size=self.embedding_config.get("request_batch_size"),
                rate_limit_rps=self.embedding_config.get("rate_limit_rps"),
                service_tier=self.embedding_config.get("service_tier"),
                max_parallel_requests=self.embedding_config.get("max_parallel_requests"),
                max_retries=self.embedding_config.get("max_retries"),
            )

        elif self.embedding_model == "gemini":
            model_name = self.embedding_config.get("model_name", "gemini-embedding-001")
            api_key = self.embedding_config.get("api_key")
            base_url = self.embedding_config.get("base_url")
            return GeminiEmbeddingFunction(
                model_name=model_name, api_key=api_key, base_url=base_url,
                request_batch_size=self.embedding_config.get("request_batch_size"),
                rate_limit_rps=self.embedding_config.get("rate_limit_rps"),
                max_parallel_requests=self.embedding_config.get("max_parallel_requests"),
                max_retries=self.embedding_config.get("max_retries"),
            )

        elif self.embedding_model == "ollama":
            model_name = self.embedding_config.get("model_name", "qwen3-embedding")
            base_url = self.embedding_config.get("base_url")
            return OllamaEmbeddingFunction(
                model_name=model_name, base_url=base_url,
                request_batch_size=self.embedding_config.get("request_batch_size"),
                rate_limit_rps=self.embedding_config.get("rate_limit_rps"),
                max_parallel_requests=self.embedding_config.get("max_parallel_requests"),
                max_retries=self.embedding_config.get("max_retries"),
            )

        elif self.embedding_model == "qwen":
            model_name = self.embedding_config.get("model_name", "Qwen/Qwen3-Embedding-0.6B")
            return HuggingFaceEmbeddingFunction(model_name=model_name)

        elif self.embedding_model == "embeddinggemma":
            model_name = self.embedding_config.get("model_name", "google/embeddinggemma-300m")
            return HuggingFaceEmbeddingFunction(model_name=model_name)

        elif self.embedding_model not in ["default", "openai", "gemini", "ollama"]:
            # Treat any other value as a HuggingFace model name
            return HuggingFaceEmbeddingFunction(model_name=self.embedding_model)

        else:
            # Use ChromaDB's default embedding function (all-MiniLM-L6-v2)
            ef = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
            ef.max_input_tokens = 256  # all-MiniLM-L6-v2 max_seq_length
            return ef

    @property
    def embedding_max_tokens(self) -> int:
        """Maximum input tokens supported by the configured embedding model."""
        return getattr(self.embedding_function, "max_input_tokens", 8000)

    def truncate_text(self, text: str, max_tokens: int | None = None) -> str:
        """Truncate text using the embedding function's model-aware tokenizer.

        Falls back to tiktoken cl100k_base or character estimation if the
        embedding function does not provide a truncate method.
        """
        if max_tokens is None:
            max_tokens = self.embedding_max_tokens
        if hasattr(self.embedding_function, 'truncate'):
            return self.embedding_function.truncate(text, max_tokens)
        # Fallback for default ChromaDB embedding function
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text, disallowed_special=())
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                text = enc.decode(tokens)
        except Exception:
            max_chars = max_tokens * 2
            if len(text) > max_chars:
                text = text[:max_chars]
        return text

    def upsert_documents(self,
                        documents: list[str],
                        metadatas: list[dict[str, Any]],
                        ids: list[str]) -> None:
        """
        Upsert (update or insert) documents to the collection.

        Args:
            documents: List of document texts to embed
            metadatas: List of metadata dictionaries for each document
            ids: List of unique IDs for each document
        """
        try:
            # ChromaDB rejects batches larger than its max_batch_size
            # (~5461). With passage-chunking enabled a batch of 25 books
            # easily exceeds that, so split instead of failing.
            try:
                max_batch = int(self.client.get_max_batch_size())
            except Exception:
                max_batch = 5000
            for i in range(0, len(ids), max_batch):
                self.collection.upsert(
                    documents=documents[i:i + max_batch],
                    metadatas=metadatas[i:i + max_batch],
                    ids=ids[i:i + max_batch]
                )
            logger.info(f"Upserted {len(documents)} documents to ChromaDB collection")
        except Exception as e:
            logger.error(f"Error upserting documents to ChromaDB: {e}")
            raise

    def upsert_embeddings(self,
                         documents: list[str],
                         metadatas: list[dict[str, Any]],
                         ids: list[str],
                         embeddings: list[list[float]]) -> None:
        """
        Upsert documents with precomputed embeddings.

        Used by OpenAI Batch API imports so ChromaDB stores the vectors
        returned asynchronously without calling the realtime embeddings API.
        """
        try:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings,
            )
            logger.info(f"Upserted {len(documents)} precomputed embeddings to ChromaDB collection")
        except Exception as e:
            logger.error(f"Error upserting precomputed embeddings to ChromaDB: {e}")
            raise

    def search(self,
               query_texts: list[str],
               n_results: int = 10,
               where: dict[str, Any] | None = None,
               where_document: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Search for similar documents.

        Args:
            query_texts: List of query texts
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions

        Returns:
            Search results from ChromaDB
        """
        try:
            query_kwargs = {
                "n_results": n_results,
                "where": where,
                "where_document": where_document,
            }

            # Use embed_query for our custom embedding functions that implement
            # correct query-time task types (e.g. Gemini retrieval_query).
            # Do NOT use embed_query on ChromaDB's DefaultEmbeddingFunction —
            # its embed_query returns chunked results, not a single vector.
            _is_custom_ef = isinstance(
                self.embedding_function,
                (OpenAIEmbeddingFunction, GeminiEmbeddingFunction, HuggingFaceEmbeddingFunction, OllamaEmbeddingFunction),
            )
            if _is_custom_ef and hasattr(self.embedding_function, 'embed_query') and query_texts:
                query_embeddings = []
                for qt in query_texts:
                    emb = self.embedding_function.embed_query(qt)
                    # Ensure plain Python floats (some providers return numpy)
                    if hasattr(emb, 'tolist'):
                        emb = emb.tolist()
                    query_embeddings.append(emb)
                query_kwargs["query_embeddings"] = query_embeddings
            else:
                query_kwargs["query_texts"] = query_texts

            results = self.collection.query(**query_kwargs)
            logger.info(f"Semantic search returned {len(results.get('ids', [[]])[0])} results")
            return results
        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            raise

    def delete_documents(self, ids: list[str]) -> None:
        """
        Delete documents from the collection.

        Args:
            ids: List of document IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from ChromaDB collection")
        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {e}")
            raise

    def delete_item_chunks(self, item_key: str) -> None:
        """Delete all passage chunks belonging to one item (chunked collections).

        Passage chunks carry ``parent_item_key`` in their metadata; deleting by
        that key clears every ``<item_key>#<n>`` entry for the item before its
        chunks are re-upserted, so a document that shrank to fewer passages
        never leaves orphaned chunks behind. No-op-safe on item-level
        collections (nothing matches the filter).
        """
        try:
            self.collection.delete(where={"parent_item_key": item_key})
        except Exception as e:
            logger.debug(f"delete_item_chunks({item_key}) failed: {e}")

    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "name": self.collection_name,
                "count": 0,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory,
                "error": str(e)
            }

    def reset_collection(self) -> None:
        """Reset (clear) the collection."""
        try:
            print("Clearing ChromaDB vector collection for rebuild...", flush=True)
            try:
                self.client.reset()
            except Exception:
                self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Reset ChromaDB collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the collection."""
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except Exception:
            return False

    def get_document_metadata(self, doc_id: str) -> dict[str, Any] | None:
        """
        Get metadata for an item if it is indexed.

        With passage chunking enabled, an item is stored only under its chunk
        ids (``<key>#<n>``) and never under the bare item key, so an exact-id
        lookup on the key alone misses every chunked item. Chunk 0 carries the
        same item-level metadata (``date_modified``, ``has_fulltext``) that
        callers need, so fall back to it.

        Args:
            doc_id: Item key (or full document id) to look up

        Returns:
            Metadata dictionary if the item is indexed, None otherwise
        """
        try:
            result = self.collection.get(ids=[doc_id, f"{doc_id}#0"], include=["metadatas"])
            if result['ids'] and result['metadatas']:
                return result['metadatas'][0]
            return None
        except Exception:
            return None

    def get_existing_ids(self, ids: list[str]) -> set[str]:
        """Return the subset of ids that already exist in the collection."""
        if not ids:
            return set()
        try:
            result = self.collection.get(ids=ids, include=[])
            return set(result.get("ids", []))
        except Exception:
            return set()

    def get_all_ids(self) -> set[str]:
        """Return every id currently stored in the collection.

        Used by incremental sync to compute deletions: items in the local
        collection but no longer present in the Zotero library.
        """
        try:
            result = self.collection.get(include=[])
            return set(result.get("ids", []))
        except Exception as e:
            logger.error(f"Error listing collection ids: {e}")
            return set()


def create_chroma_client(config_path: str | None = None) -> ChromaClient:
    """
    Create a ChromaClient instance from configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured ChromaClient instance
    """
    # Default configuration
    config = {
        "collection_name": "zotero_library",
        "embedding_model": "default",
        "embedding_config": {}
    }

    # Load configuration from file if it exists
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                file_config = json.load(f)
                config.update(file_config.get("semantic_search", {}))
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {e}")

    # Pick the embedding model. config.json is the richer, authoritative source
    # (it also carries the matching api_key / base_url / model_name), so it wins
    # over the ZOTERO_EMBEDDING_MODEL env var whenever it names a concrete model.
    # The env var only fills the gap when config.json is absent or left at the
    # "default" placeholder — which is the normal Claude Desktop case.
    #
    # This deliberately guards against a stale env value silently downgrading an
    # explicitly configured provider: Claude Desktop passes its server `env`
    # block on every launch and can rewrite that file on its own, so a leftover
    # ZOTERO_EMBEDDING_MODEL=default there would otherwise override a Gemini/
    # OpenAI config.json and break search with an opaque embedding-dimension
    # mismatch against the persisted collection.
    env_embedding_model = os.getenv("ZOTERO_EMBEDDING_MODEL")
    if env_embedding_model and config.get("embedding_model", "default") in (None, "default"):
        config["embedding_model"] = env_embedding_model

    # Merge embedding config from environment (config.json wins, env fills gaps).
    # Precedence: explicit config.json value > env var > hardcoded default.
    # Previous code unconditionally REPLACED config["embedding_config"] with env
    # values, silently dropping model_name from config.json whenever any
    # provider env var (e.g. GOOGLE_API_KEY leaked from another tool) was set.
    if config["embedding_model"] == "openai":
        ec = dict(config.get("embedding_config") or {})
        if not ec.get("api_key"):
            env_key = os.getenv("OPENAI_API_KEY")
            if env_key:
                ec["api_key"] = env_key
        if not ec.get("model_name"):
            ec["model_name"] = os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            )
        if not ec.get("base_url"):
            env_base = os.getenv("OPENAI_BASE_URL")
            if env_base:
                ec["base_url"] = env_base
        if ec.get("api_key"):
            config["embedding_config"] = ec

    elif config["embedding_model"] == "gemini":
        ec = dict(config.get("embedding_config") or {})
        if not ec.get("api_key"):
            env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if env_key:
                ec["api_key"] = env_key
        if not ec.get("model_name"):
            ec["model_name"] = os.getenv(
                "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
            )
        if not ec.get("base_url"):
            env_base = os.getenv("GEMINI_BASE_URL")
            if env_base:
                ec["base_url"] = env_base
        if ec.get("api_key"):
            config["embedding_config"] = ec

    elif config["embedding_model"] == "ollama":
        ec = dict(config.get("embedding_config") or {})
        if not ec.get("model_name"):
            ec["model_name"] = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding")
        if not ec.get("base_url"):
            env_base = os.getenv("OLLAMA_BASE_URL")
            if env_base:
                ec["base_url"] = env_base
        config["embedding_config"] = ec

    return ChromaClient(
        collection_name=config["collection_name"],
        embedding_model=config["embedding_model"],
        embedding_config=config["embedding_config"]
    )


class _NoEmbeddingFunction(EmbeddingFunction):
    """Placeholder embedding function used for read-only status reads.

    Passing an explicit embedding function to ``get_collection`` stops ChromaDB
    from reconstructing the collection's persisted embedding function — which,
    for the default backend, eagerly downloads the ~80MB ONNX MiniLM model.
    Counting rows never embeds anything, so this is never actually called; it
    raises if it ever is, to make misuse loud rather than silently wrong.

    ``name()`` MUST return ``"default"``. ChromaDB >=1.x validates the supplied
    embedding function against the collection's persisted config in
    ``validate_embedding_function_conflict_on_get`` and raises a ``ValueError``
    whenever the supplied ``name()`` differs from the persisted one — *unless*
    the supplied name is ``"default"``, which short-circuits the check. Without
    this, opening a collection that was built with any real backend (default,
    openai, gemini, ...) raises a conflict; ``read_collection_status`` then
    swallowed that error and reported "0 documents / not initialized" against a
    fully populated database (issue #362).
    """

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:  # pragma: no cover - never invoked
        raise RuntimeError("embedding is unavailable in status-only mode")

    @staticmethod
    def name() -> str:
        return "default"


def read_collection_status(
    config_path: str | None = None,
    *,
    persist_directory: str | None = None,
) -> dict[str, Any]:
    """Read ChromaDB collection stats WITHOUT loading an embedding model.

    The full :class:`ChromaClient` constructor builds the embedding function,
    which for the default backend downloads the ONNX MiniLM model on first use —
    turning a read-only status check into a multi-minute (or network-blocked,
    indefinite) hang. Reporting readiness only needs the row count and the
    configured model name, neither of which requires the model itself. This
    opens the persisted database directly and reads the count, mirroring the
    shape returned by :meth:`ChromaClient.get_collection_info`.

    ``persist_directory`` defaults to ``ChromaClient``'s location
    (``~/.config/zotero-mcp/chroma_db``); it is parameterised for testing.
    """
    collection_name = "zotero_library"
    embedding_model = "default"

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                semantic_cfg = json.load(f).get("semantic_search", {})
            collection_name = semantic_cfg.get("collection_name", collection_name)
            embedding_model = semantic_cfg.get("embedding_model", embedding_model)
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {e}")

    # Mirror create_chroma_client's precedence: config.json wins; the env var
    # only fills in when the file left the model at the "default" placeholder.
    env_model = os.getenv("ZOTERO_EMBEDDING_MODEL")
    if env_model and embedding_model in (None, "default"):
        embedding_model = env_model

    if persist_directory is None:
        persist_directory = str(Path.home() / ".config" / "zotero-mcp" / "chroma_db")
    base = {
        "name": collection_name,
        "embedding_model": embedding_model,
        "persist_directory": persist_directory,
    }

    try:
        with suppress_stdout():
            client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            try:
                collection = client.get_collection(
                    name=collection_name,
                    embedding_function=_NoEmbeddingFunction(),
                )
            except Exception:
                # Collection does not exist yet — database not initialized.
                return {**base, "count": 0, "initialized": False}
            count = collection.count()
        return {**base, "count": count, "initialized": True}
    except Exception as e:
        logger.error(f"Error reading collection status: {e}")
        return {**base, "count": 0, "error": str(e)}
