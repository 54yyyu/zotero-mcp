"""One source of truth for "which embedding providers exist, and how are they
configured and constructed".

This replaces two parallel if/elif chains that had to be kept in agreement by
hand, both keyed on the same ``embedding_model`` string:

- ``ChromaClient._create_embedding_function`` — string -> constructed embedding
  function, reproduced here by :func:`resolve_provider` plus each spec's
  ``ef_factory``.
- the three near-identical per-provider blocks in
  ``chroma_client.create_chroma_client`` that merge environment variables into
  ``embedding_config``, reproduced here by :func:`merge_env_config` driven by
  each spec's :class:`EnvSpec`.

Provider names are frozen: ChromaDB persists the embedding function's name in a
collection's config and rebuilds it by name on reload, so renaming one orphans
every index built with it.

This module imports the provider classes at module scope. That is safe in one
direction only: provider modules depend on ``embeddings.base`` alone, never on
this module or on ``chroma_client``.
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from zotero_mcp.embeddings.providers.gemini import GeminiEmbeddingFunction
from zotero_mcp.embeddings.providers.huggingface import HuggingFaceEmbeddingFunction
from zotero_mcp.embeddings.providers.ollama import OllamaEmbeddingFunction
from zotero_mcp.embeddings.providers.openai import OpenAIEmbeddingFunction


@dataclass(frozen=True)
class EnvSpec:
    """Which environment variables configure a provider, and how.

    Mirrors one of the per-provider blocks in ``create_chroma_client``:

    - ``api_key_vars`` are tried in order, first non-empty wins. Gemini needs
      two because ``GOOGLE_API_KEY`` is the more commonly set of the pair.
    - ``model_var`` / ``base_url_var`` are single variable names, or ``None``
      when the provider reads none (huggingface and default have no env wiring
      at all, matching the old chain's lack of a branch for them).
    - ``requires_api_key`` reproduces the difference between the branches: for
      openai/gemini the merged config is assigned back only when an api key was
      actually resolved, for ollama it is assigned back unconditionally.
    """

    api_key_vars: tuple[str, ...] = ()
    model_var: str | None = None
    base_url_var: str | None = None
    requires_api_key: bool = False

    def reads_environment(self) -> bool:
        """Whether this spec touches the environment at all.

        False for the providers the old if/elif chain had no branch for, whose
        ``embedding_config`` must therefore be left exactly as the caller
        supplied it.
        """
        return bool(self.api_key_vars or self.model_var or self.base_url_var)


@dataclass(frozen=True)
class ProviderSpec:
    """Everything needed to resolve, configure and construct one provider.

    ``model_aliases`` maps a short alias the user may put in
    ``embedding_model`` (``"qwen"``, ``"embeddinggemma"``) to the concrete
    model name it stands for.
    """

    name: str
    default_model: str | None
    ef_factory: Callable[[dict[str, Any]], Any]
    env: EnvSpec = field(default_factory=EnvSpec)
    model_aliases: Mapping[str, str] = field(default_factory=dict)


PROVIDERS: dict[str, ProviderSpec] = {}


def register_provider(spec: ProviderSpec) -> ProviderSpec:
    """Register (or replace) a provider spec by name, returning the spec."""
    PROVIDERS[spec.name] = spec
    return spec


def _openai_ef_factory(config: dict[str, Any]) -> Any:
    return OpenAIEmbeddingFunction(
        model_name=config.get("model_name", "text-embedding-3-small"),
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        request_batch_size=config.get("request_batch_size"),
        rate_limit_rps=config.get("rate_limit_rps"),
    )


def _gemini_ef_factory(config: dict[str, Any]) -> Any:
    return GeminiEmbeddingFunction(
        model_name=config.get("model_name", "gemini-embedding-001"),
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
    )


def _ollama_ef_factory(config: dict[str, Any]) -> Any:
    return OllamaEmbeddingFunction(
        model_name=config.get("model_name", "qwen3-embedding"),
        base_url=config.get("base_url"),
        timeout=config.get("timeout"),
        request_batch_size=config.get("request_batch_size"),
    )


def _huggingface_ef_factory(config: dict[str, Any]) -> Any:
    return HuggingFaceEmbeddingFunction(
        model_name=config.get("model_name", "Qwen/Qwen3-Embedding-0.6B"),
    )


def _default_ef_factory(config: dict[str, Any]) -> Any:
    from chromadb.utils import embedding_functions

    ef = embedding_functions.DefaultEmbeddingFunction()
    ef.max_input_tokens = 256  # all-MiniLM-L6-v2 max_seq_length
    return ef


register_provider(
    ProviderSpec(
        name="openai",
        default_model="text-embedding-3-small",
        ef_factory=_openai_ef_factory,
        env=EnvSpec(
            api_key_vars=("OPENAI_API_KEY",),
            model_var="OPENAI_EMBEDDING_MODEL",
            base_url_var="OPENAI_BASE_URL",
            requires_api_key=True,
        ),
    )
)

register_provider(
    ProviderSpec(
        name="gemini",
        default_model="gemini-embedding-001",
        ef_factory=_gemini_ef_factory,
        env=EnvSpec(
            api_key_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            model_var="GEMINI_EMBEDDING_MODEL",
            base_url_var="GEMINI_BASE_URL",
            requires_api_key=True,
        ),
    )
)

register_provider(
    ProviderSpec(
        name="ollama",
        default_model="qwen3-embedding",
        ef_factory=_ollama_ef_factory,
        env=EnvSpec(
            model_var="OLLAMA_EMBEDDING_MODEL",
            base_url_var="OLLAMA_BASE_URL",
            requires_api_key=False,
        ),
    )
)

register_provider(
    ProviderSpec(
        name="huggingface",
        default_model="Qwen/Qwen3-Embedding-0.6B",
        ef_factory=_huggingface_ef_factory,
        model_aliases={
            "qwen": "Qwen/Qwen3-Embedding-0.6B",
            "embeddinggemma": "google/embeddinggemma-300m",
        },
    )
)

register_provider(
    ProviderSpec(
        name="default",
        default_model=None,
        ef_factory=_default_ef_factory,
    )
)


def resolve_provider(
    embedding_model: str,
) -> tuple[ProviderSpec, dict[str, Any], dict[str, Any]]:
    """Map an ``embedding_model`` string to its provider and any config the
    string itself implies.

    Returns ``(spec, defaults, overrides)``. The caller builds the effective
    config as ``{**defaults, **embedding_config, **overrides}`` — i.e.
    ``defaults`` lose to an explicit ``embedding_config`` entry and
    ``overrides`` beat one. Two dicts rather than one because the old chain
    resolved the two alias cases in opposite directions, and this is a pure
    refactor:

    - ``"qwen"`` did ``embedding_config.get("model_name", "Qwen/...")``, so an
      explicit ``model_name`` won -> a *default*.
    - a bare HuggingFace model name passed ``self.embedding_model`` straight to
      the constructor, ignoring ``embedding_config["model_name"]`` entirely ->
      an *override*.

    The rules, mirroring ``_create_embedding_function``'s chain in order:

    - any registered provider name other than ``"huggingface"``/``"default"``
      (``"openai"``, ``"gemini"``, ``"ollama"``) -> that provider, no extras.
      Generalized from the old hardcoded literals, so a provider added later is
      reachable by its bare name with no change to this function.
    - ``"qwen"`` / ``"embeddinggemma"`` -> huggingface, with the aliased model
      name as a default.
    - anything else that is not ``"default"`` -> huggingface, with the string
      itself as an override. This preserves the legacy quirk that the literal
      string ``"huggingface"`` is treated as a *model* name (there is no
      sentence-transformers model called that, so it fails at load time) rather
      than selecting the huggingface provider — which is why ``"huggingface"``
      is excluded from the first rule.
    - ``"default"`` -> ChromaDB's built-in embedding function.
    """
    huggingface_spec = PROVIDERS["huggingface"]

    if embedding_model in PROVIDERS and embedding_model not in ("huggingface", "default"):
        return PROVIDERS[embedding_model], {}, {}

    if embedding_model in huggingface_spec.model_aliases:
        return huggingface_spec, {"model_name": huggingface_spec.model_aliases[embedding_model]}, {}

    if embedding_model != "default":
        return huggingface_spec, {}, {"model_name": embedding_model}

    return PROVIDERS["default"], {}, {}


def create_embedding_function(
    embedding_model: str, embedding_config: dict[str, Any] | None
) -> Any:
    """Construct the embedding function for a configured ``embedding_model``."""
    spec, defaults, overrides = resolve_provider(embedding_model)
    config = {**defaults, **(embedding_config or {}), **overrides}
    return spec.ef_factory(config)


def merge_env_config(
    embedding_model: str, embedding_config: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Fill gaps in ``embedding_config`` from the environment.

    Precedence is unchanged from the blocks this replaces: an explicit
    ``embedding_config`` value wins over an env var, which wins over the
    provider's hardcoded default. Only *absent or falsy* keys are filled, so a
    stray provider env var (a ``GOOGLE_API_KEY`` leaked in from another tool,
    say) can never displace a value the user configured.

    Returns the config to use. Providers with no env wiring, and — for
    api-key-requiring providers — the case where no key could be resolved from
    either source, return ``embedding_config`` untouched, exactly as the old
    chain left it.
    """
    spec = resolve_provider(embedding_model)[0]
    env = spec.env
    if not env.reads_environment():
        return embedding_config

    ec = dict(embedding_config or {})

    if env.api_key_vars and not ec.get("api_key"):
        for var in env.api_key_vars:
            env_key = os.getenv(var)
            if env_key:
                ec["api_key"] = env_key
                break

    if env.model_var and not ec.get("model_name"):
        ec["model_name"] = os.getenv(env.model_var, spec.default_model)

    if env.base_url_var and not ec.get("base_url"):
        env_base = os.getenv(env.base_url_var)
        if env_base:
            ec["base_url"] = env_base

    if env.requires_api_key and not ec.get("api_key"):
        return embedding_config
    return ec
