"""Provider registry: a single source of truth for "what embedding models
exist and how do we build/configure them", replacing the
``_create_embedding_function`` if/elif chain and the three near-identical
env-merge blocks in ``chroma_client.create_chroma_client``.

Phase 2 of the embedding-provider refactor. This module is deliberately NOT
imported from ``zotero_mcp.embeddings.__init__`` (which is imported by
``chroma_client`` at module scope): ``ef_factory`` callables need the EF
classes that are still defined in ``chroma_client.py`` (they move to
``embeddings/providers/`` in a later phase), so importing them eagerly here
would create ``chroma_client -> embeddings -> registry -> chroma_client``.
Each factory below does a *late* (call-time) ``import zotero_mcp.chroma_client``
instead, and callers reach this module directly as
``zotero_mcp.embeddings.registry`` rather than via the package ``__init__``.

Registered names (``"openai"``, ``"gemini"``, ``"huggingface"``, ``"ollama"``)
are frozen — ChromaDB persists the name in a collection's config and rebuilds
the embedding function by name on reload. This module must reproduce
``resolve_provider``'s selection logic byte-for-byte against the old chain in
``ChromaClient._create_embedding_function`` (see docstring on
``resolve_provider`` below for the exact rules mirrored).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EnvSpec:
    """Environment-variable names a provider's realtime EF is configured from.

    Mirrors the per-provider blocks in ``create_chroma_client`` today:
    ``api_key_vars`` are tried in order (first non-empty wins — this is why
    Gemini needs two: ``GEMINI_API_KEY`` then the more commonly-set
    ``GOOGLE_API_KEY``); ``model_var``/``base_url_var`` are single env var
    names, or ``None`` when the provider doesn't read one (huggingface/default
    have no env wiring at all). ``requires_api_key`` gates whether the merged
    embedding_config is only assigned back when an api_key was actually
    resolved (openai/gemini) versus unconditionally (ollama).
    """

    api_key_vars: tuple[str, ...] = ()
    model_var: str | None = None
    base_url_var: str | None = None
    requires_api_key: bool = False


@dataclass(frozen=True)
class ProviderSpec:
    """Everything needed to resolve, configure, and construct one provider's
    realtime embedding function.

    ``batch`` is a placeholder for the ``BatchAdapter`` that Phase 3 attaches
    to batch-capable providers (openai/gemini); it stays ``None`` here so
    ``batch_capable_providers()`` returns an empty list until then.
    ``model_aliases`` maps a short alias (``"qwen"``, ``"embeddinggemma"``) to
    the concrete HuggingFace model name it expands to when the user hasn't
    given an explicit ``model_name`` in embedding_config.
    """

    name: str
    label: str
    default_model: str | None
    ef_factory: Callable[[dict[str, Any]], Any]
    env: EnvSpec
    chars_per_token: float
    batch: Any | None = None
    model_aliases: Mapping[str, str] = field(default_factory=dict)


PROVIDERS: dict[str, ProviderSpec] = {}


def register_provider(spec: ProviderSpec) -> ProviderSpec:
    """Register (or replace) a provider spec by name. Returns the spec so it
    can be used inline at the registration call site if desired."""
    PROVIDERS[spec.name] = spec
    return spec


def batch_capable_providers() -> list[str]:
    """Names of providers with a batch adapter attached.

    Empty today (batch adapters arrive in Phase 3) — drives the CLI's
    ``--provider`` choices once they exist.
    """
    return [name for name, spec in PROVIDERS.items() if spec.batch is not None]


def _openai_ef_factory(config: dict[str, Any]) -> Any:
    from zotero_mcp.chroma_client import OpenAIEmbeddingFunction

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


def _gemini_ef_factory(config: dict[str, Any]) -> Any:
    from zotero_mcp.chroma_client import GeminiEmbeddingFunction

    return GeminiEmbeddingFunction(
        model_name=config.get("model_name", "gemini-embedding-001"),
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        request_batch_size=config.get("request_batch_size"),
        rate_limit_rps=config.get("rate_limit_rps"),
        max_parallel_requests=config.get("max_parallel_requests"),
        max_retries=config.get("max_retries"),
    )


def _ollama_ef_factory(config: dict[str, Any]) -> Any:
    from zotero_mcp.chroma_client import OllamaEmbeddingFunction

    return OllamaEmbeddingFunction(
        model_name=config.get("model_name", "qwen3-embedding"),
        base_url=config.get("base_url"),
        request_batch_size=config.get("request_batch_size"),
        rate_limit_rps=config.get("rate_limit_rps"),
        max_parallel_requests=config.get("max_parallel_requests"),
        max_retries=config.get("max_retries"),
    )


def _huggingface_ef_factory(config: dict[str, Any]) -> Any:
    from zotero_mcp.chroma_client import HuggingFaceEmbeddingFunction

    return HuggingFaceEmbeddingFunction(model_name=config.get("model_name"))


def _default_ef_factory(config: dict[str, Any]) -> Any:
    import chromadb

    ef = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
    ef.max_input_tokens = 256  # all-MiniLM-L6-v2 max_seq_length (today's fallback branch)
    return ef


# chars_per_token values mirror OPENAI_CHARS_PER_TOKEN (openai_batch.py) and
# GEMINI_CHARS_PER_TOKEN (gemini_batch.py) — kept here as plain floats (no
# import of the batch modules) so this module has no dependency on them.
register_provider(
    ProviderSpec(
        name="openai",
        label="OpenAI",
        default_model="text-embedding-3-small",
        ef_factory=_openai_ef_factory,
        env=EnvSpec(
            api_key_vars=("OPENAI_API_KEY",),
            model_var="OPENAI_EMBEDDING_MODEL",
            base_url_var="OPENAI_BASE_URL",
            requires_api_key=True,
        ),
        chars_per_token=3.0,
    )
)

register_provider(
    ProviderSpec(
        name="gemini",
        label="Gemini",
        default_model="gemini-embedding-001",
        ef_factory=_gemini_ef_factory,
        env=EnvSpec(
            api_key_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            model_var="GEMINI_EMBEDDING_MODEL",
            base_url_var="GEMINI_BASE_URL",
            requires_api_key=True,
        ),
        chars_per_token=3.5,
    )
)

register_provider(
    ProviderSpec(
        name="ollama",
        label="Ollama",
        default_model="qwen3-embedding",
        ef_factory=_ollama_ef_factory,
        env=EnvSpec(
            api_key_vars=(),
            model_var="OLLAMA_EMBEDDING_MODEL",
            base_url_var="OLLAMA_BASE_URL",
            requires_api_key=False,
        ),
        chars_per_token=4.0,
    )
)

register_provider(
    ProviderSpec(
        name="huggingface",
        label="HuggingFace (local, sentence-transformers)",
        default_model=None,
        ef_factory=_huggingface_ef_factory,
        env=EnvSpec(),
        chars_per_token=4.0,
        model_aliases={
            "qwen": "Qwen/Qwen3-Embedding-0.6B",
            "embeddinggemma": "google/embeddinggemma-300m",
        },
    )
)

register_provider(
    ProviderSpec(
        name="default",
        label="ChromaDB default (all-MiniLM-L6-v2)",
        default_model=None,
        ef_factory=_default_ef_factory,
        env=EnvSpec(),
        chars_per_token=4.0,
    )
)


def resolve_provider(embedding_model: str) -> tuple[ProviderSpec, dict[str, Any]]:
    """Resolve an ``embedding_model`` string to its ``ProviderSpec`` plus any
    extra config the model string itself implies (currently only the
    HuggingFace alias -> concrete model_name mapping).

    Reproduces ``ChromaClient._create_embedding_function``'s old if/elif
    chain exactly:
      - ``"openai"`` / ``"gemini"`` / ``"ollama"`` -> that provider, no extras.
      - ``"qwen"`` -> huggingface, extra ``{"model_name": "Qwen/Qwen3-Embedding-0.6B"}``.
      - ``"embeddinggemma"`` -> huggingface, extra ``{"model_name": "google/embeddinggemma-300m"}``.
        Both aliases: an explicit ``model_name`` in embedding_config still wins
        because the caller merges as ``{**extra, **embedding_config}``.
      - any other string that isn't ``"default"`` -> huggingface, extra
        ``{"model_name": embedding_model}`` (the string itself).
      - ``"default"`` -> the "default" spec (ChromaDB's built-in EF).
    """
    huggingface_spec = PROVIDERS["huggingface"]

    if embedding_model in ("openai", "gemini", "ollama"):
        return PROVIDERS[embedding_model], {}

    if embedding_model in huggingface_spec.model_aliases:
        return huggingface_spec, {"model_name": huggingface_spec.model_aliases[embedding_model]}

    if embedding_model != "default":
        return huggingface_spec, {"model_name": embedding_model}

    return PROVIDERS["default"], {}
