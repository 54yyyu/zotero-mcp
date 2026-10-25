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

import dataclasses
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

    ``batch`` holds this provider's ``batch_common.BatchAdapter`` when it has
    one, and ``None`` when it does not. Typed ``Any`` rather than imported:
    ``batch_common`` is a leaf module and importing it here would make the
    registry depend on it just to spell a type, for no gain.
    """

    name: str
    default_model: str | None
    ef_factory: Callable[[dict[str, Any]], Any]
    env: EnvSpec = field(default_factory=EnvSpec)
    model_aliases: Mapping[str, str] = field(default_factory=dict)
    batch: Any | None = None


PROVIDERS: dict[str, ProviderSpec] = {}


def register_provider(spec: ProviderSpec) -> ProviderSpec:
    """Register (or replace) a provider spec by name, returning the spec."""
    PROVIDERS[spec.name] = spec
    return spec


def batch_capable_providers() -> list[str]:
    """Names of providers that currently have a batch adapter attached.

    Drives the CLI's ``--batch-provider`` choices, so it reflects what is
    actually wired up rather than a hand-maintained list that can drift.
    """
    return [name for name, spec in PROVIDERS.items() if spec.batch is not None]


def attach_batch_adapter(name: str, adapter: Any) -> ProviderSpec:
    """Attach a ``BatchAdapter`` to an already-registered provider spec.

    ``ProviderSpec`` is frozen, so this replaces the stored spec rather than
    mutating it. ``openai_batch`` and ``gemini_batch`` each call this at module
    scope, which is why importing either module is what makes its provider
    batch-capable — see the note on ``_load_batch_providers`` in ``cli.py``.
    """
    updated = dataclasses.replace(PROVIDERS[name], batch=adapter)
    PROVIDERS[name] = updated
    return updated


# The three remote providers share the pacing/concurrency keys owned by
# RemoteEmbeddingFunction. All are read with .get(), so a config written
# before they existed still resolves — they simply fall back to each
# provider's own class-level defaults.
def _remote_pacing_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "rate_limit_rps": config.get("rate_limit_rps"),
        "max_parallel_requests": config.get("max_parallel_requests"),
        "max_retries": config.get("max_retries"),
        "tokens_per_minute": config.get("tokens_per_minute"),
    }


def _openai_ef_factory(config: dict[str, Any]) -> Any:
    return OpenAIEmbeddingFunction(
        model_name=config.get("model_name", "text-embedding-3-small"),
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        request_batch_size=config.get("request_batch_size"),
        dimensions=config.get("dimensions"),
        **_remote_pacing_kwargs(config),
    )


def _gemini_ef_factory(config: dict[str, Any]) -> Any:
    return GeminiEmbeddingFunction(
        model_name=config.get("model_name", "gemini-embedding-001"),
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        request_batch_size=config.get("request_batch_size"),
        **_remote_pacing_kwargs(config),
    )


def _ollama_ef_factory(config: dict[str, Any]) -> Any:
    return OllamaEmbeddingFunction(
        model_name=config.get("model_name", "qwen3-embedding"),
        base_url=config.get("base_url"),
        timeout=config.get("timeout"),
        request_batch_size=config.get("request_batch_size"),
        **_remote_pacing_kwargs(config),
    )


def _huggingface_ef_factory(config: dict[str, Any]) -> Any:
    return HuggingFaceEmbeddingFunction(
        model_name=config.get("model_name", "Qwen/Qwen3-Embedding-0.6B"),
    )


def _resolve_intra_op_threads() -> int:
    """How many ONNX Runtime intra-op threads the default EF may use.

    ONNX Runtime sizes its thread pool from the *total* logical core count and
    then binds each thread to a specific core index. Under a restricted CPU
    affinity mask (containers, Slurm jobs, ``taskset``) that bind fails with
    ``pthread_setaffinity_np ... Invalid argument`` and floods stderr once per
    thread. An explicit thread count makes ORT skip affinity pinning entirely.

    ``ONNXRUNTIME_INTRA_OP_NUM_THREADS`` wins when set to a positive integer,
    matching what the user in the issue expected the library to honour. It is
    otherwise capped at the number of CPUs this process is actually allowed to
    run on, so a 192-core host with an 8-core mask no longer asks for 192
    threads. A value of ``0`` is ORT's "let it decide" sentinel and is never
    returned.
    """
    env_value = os.environ.get("ONNXRUNTIME_INTRA_OP_NUM_THREADS")
    if env_value is not None:
        try:
            configured = int(env_value)
        except ValueError:
            configured = 0
        if configured > 0:
            return configured

    try:
        allowed = len(os.sched_getaffinity(0))  # Linux-only
    except (AttributeError, OSError):
        allowed = os.cpu_count() or 1
    return max(1, allowed)


def _cap_onnx_intra_op_threads() -> None:
    """Make every ``ort.InferenceSession`` respect :func:`_resolve_intra_op_threads`.

    ChromaDB's ``ONNXMiniLM_L6_V2`` builds its own ``SessionOptions`` and never
    exposes them, so the only interception point is ``InferenceSession`` itself.
    Sessions that already chose a thread count (``!= 0``) are left untouched, so
    an explicit caller configuration always wins. The patch is installed once;
    repeated calls are no-ops.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return

    original_init = ort.InferenceSession.__init__
    if getattr(original_init, "_zotero_mcp_intra_op_capped", False):
        return

    def _patched_init(self, path_or_bytes, sess_options=None, providers=None, **kwargs):
        if sess_options is None:
            sess_options = ort.SessionOptions()
        if getattr(sess_options, "intra_op_num_threads", 0) == 0:
            sess_options.intra_op_num_threads = _resolve_intra_op_threads()
        return original_init(self, path_or_bytes, sess_options, providers, **kwargs)

    _patched_init._zotero_mcp_intra_op_capped = True
    ort.InferenceSession.__init__ = _patched_init


def _default_ef_factory(config: dict[str, Any]) -> Any:
    from chromadb.utils import embedding_functions

    _cap_onnx_intra_op_threads()
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
