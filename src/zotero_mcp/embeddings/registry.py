"""Provider registry: a single source of truth for "what embedding models
exist and how do we build/configure them", replacing the
``_create_embedding_function`` if/elif chain and the three near-identical
env-merge blocks in ``chroma_client.create_chroma_client``.

Phase 2 of the embedding-provider refactor; Phase 5 moved the concrete EF
classes out of ``chroma_client.py`` into ``embeddings/providers/``, so this
module now imports its ``ef_factory`` callables' classes directly from those
provider modules at top level (no more late/call-time imports): the provider
modules depend only on ``embeddings/base.py``, never on ``chroma_client`` or
this module, so there is no import cycle. This module is still deliberately
NOT imported from ``zotero_mcp.embeddings.__init__`` — callers reach it
directly as ``zotero_mcp.embeddings.registry``.

Registered names (``"openai"``, ``"gemini"``, ``"huggingface"``, ``"ollama"``)
are frozen — ChromaDB persists the name in a collection's config and rebuilds
the embedding function by name on reload. This module must reproduce
``resolve_provider``'s selection logic byte-for-byte against the old chain in
``ChromaClient._create_embedding_function`` (see docstring on
``resolve_provider`` below for the exact rules mirrored).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from zotero_mcp.embeddings.providers.gemini import GeminiEmbeddingFunction
from zotero_mcp.embeddings.providers.huggingface import HuggingFaceEmbeddingFunction
from zotero_mcp.embeddings.providers.ollama import OllamaEmbeddingFunction
from zotero_mcp.embeddings.providers.openai import OpenAIEmbeddingFunction
from zotero_mcp.embeddings.providers.voyage import VoyageEmbeddingFunction


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

    Empty until something calls :func:`attach_batch_adapter` — drives the
    CLI's ``--provider`` choices once Phase 4 wires that call in.
    """
    return [name for name, spec in PROVIDERS.items() if spec.batch is not None]


def attach_batch_adapter(name: str, adapter: Any) -> ProviderSpec:
    """Attach a ``BatchAdapter`` to an already-registered provider spec.

    ``ProviderSpec`` is frozen, so this replaces the stored spec via
    ``dataclasses.replace``. Available as of Phase 3 but deliberately not
    called anywhere yet (not from ``openai_batch``/``gemini_batch`` module
    scope, not from ``semantic_search``): ``PROVIDERS`` is a process-wide
    singleton, and both of those modules are imported — for their unrelated
    module-level wrapper functions — by test files that the full suite
    collects before ``test_provider_registry.py`` runs. Calling this at
    import time would mutate ``PROVIDERS`` for the rest of the pytest
    session and break
    ``test_provider_registry.py::test_batch_capable_providers_empty_until_phase_3``,
    which pins the Phase-2 invariant that ``batch`` stays ``None`` until
    something explicit wires it up. Phase 4 (cli.py, which is the first
    consumer of ``batch_capable_providers()``) is the right place to call
    this and to retire that now-superseded test.
    """
    updated = dataclasses.replace(PROVIDERS[name], batch=adapter)
    PROVIDERS[name] = updated
    return updated


def _openai_ef_factory(config: dict[str, Any]) -> Any:
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
    return OllamaEmbeddingFunction(
        model_name=config.get("model_name", "qwen3-embedding"),
        base_url=config.get("base_url"),
        request_batch_size=config.get("request_batch_size"),
        rate_limit_rps=config.get("rate_limit_rps"),
        max_parallel_requests=config.get("max_parallel_requests"),
        max_retries=config.get("max_retries"),
    )


def _huggingface_ef_factory(config: dict[str, Any]) -> Any:
    return HuggingFaceEmbeddingFunction(model_name=config.get("model_name"))


def _default_ef_factory(config: dict[str, Any]) -> Any:
    import chromadb

    ef = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
    ef.max_input_tokens = 256  # all-MiniLM-L6-v2 max_seq_length (today's fallback branch)
    return ef


def _voyage_ef_factory(config: dict[str, Any]) -> Any:
    return VoyageEmbeddingFunction(
        model_name=config.get("model_name", "voyage-3.5"),
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        request_batch_size=config.get("request_batch_size"),
        rate_limit_rps=config.get("rate_limit_rps"),
        max_parallel_requests=config.get("max_parallel_requests"),
        max_retries=config.get("max_retries"),
    )


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

register_provider(
    ProviderSpec(
        name="voyage",
        label="Voyage AI",
        default_model="voyage-3.5",
        ef_factory=_voyage_ef_factory,
        env=EnvSpec(
            api_key_vars=("VOYAGE_API_KEY",),
            model_var="VOYAGE_EMBEDDING_MODEL",
            base_url_var="VOYAGE_BASE_URL",
            requires_api_key=True,
        ),
        chars_per_token=4.0,
    )
)


def resolve_provider(embedding_model: str) -> tuple[ProviderSpec, dict[str, Any]]:
    """Resolve an ``embedding_model`` string to its ``ProviderSpec`` plus any
    extra config the model string itself implies (currently only the
    HuggingFace alias -> concrete model_name mapping).

    Reproduces ``ChromaClient._create_embedding_function``'s old if/elif
    chain exactly, generalized so any newly registered provider is reachable
    by its bare name with zero changes to this function:
      - any ``embedding_model`` that is itself a key in ``PROVIDERS`` other
        than ``"huggingface"``/``"default"`` -> that provider, no extras. This
        is what makes adding a new provider (e.g. ``"voyage"``) resolvable
        just by registering its spec — the literal strings ``"openai"``,
        ``"gemini"``, ``"ollama"`` used to be hardcoded here; "voyage" reaches
        this same branch with no further change to this function.
      - ``"qwen"`` -> huggingface, extra ``{"model_name": "Qwen/Qwen3-Embedding-0.6B"}``.
      - ``"embeddinggemma"`` -> huggingface, extra ``{"model_name": "google/embeddinggemma-300m"}``.
        Both aliases: an explicit ``model_name`` in embedding_config still wins
        because the caller merges as ``{**extra, **embedding_config}``.
      - any other string that isn't ``"default"`` -> huggingface, extra
        ``{"model_name": embedding_model}`` (the string itself). This
        deliberately keeps the legacy quirk that the literal string
        ``"huggingface"`` is treated as an HF model name (looked up via this
        branch, not the direct-name branch above, since "huggingface" is
        explicitly excluded from it) and never resolves to the huggingface
        *provider* spec directly.
      - ``"default"`` -> the "default" spec (ChromaDB's built-in EF).
    """
    huggingface_spec = PROVIDERS["huggingface"]

    if embedding_model in PROVIDERS and embedding_model not in ("huggingface", "default"):
        return PROVIDERS[embedding_model], {}

    if embedding_model in huggingface_spec.model_aliases:
        return huggingface_spec, {"model_name": huggingface_spec.model_aliases[embedding_model]}

    if embedding_model != "default":
        return huggingface_spec, {"model_name": embedding_model}

    return PROVIDERS["default"], {}
