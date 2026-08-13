"""Live tests for the OpenAI and Gemini providers, each behind the
config-match gate in tests/live/conftest.py (``configured_provider``).

A test class only actually runs live API calls when this machine's
``~/.config/zotero-mcp/config.json`` has
``semantic_search.embedding_model`` equal to that provider's name — otherwise
it skips with a message naming the actual configured provider. Both code
paths (OpenAI running live, Gemini skipping) must be correct regardless of
which provider happens to be configured on the machine running the suite.

Every EF is built via ``create_embedding_function(provider, config)`` — the
exact path production code (``ChromaClient._create_embedding_function``)
uses — seeded with the machine's real production ``embedding_config`` (model
name, api_key, etc.), so these tests exercise the actual configuration in
use. Inputs are kept to a handful of short strings; total live-API cost is a
small fraction of a cent.
"""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")

from zotero_mcp.embeddings.registry import create_embedding_function  # noqa: E402

DOCS = ["alpha short text", "beta short text", "gamma short text", "delta short text", "epsilon short text"]


class _ProviderLiveTests:
    """Shared test bodies for a provider, parametrized by class attribute
    ``PROVIDER``. Subclasses set ``PROVIDER`` and (optionally) a fixed
    ``EXPECTED_DIM``; when ``EXPECTED_DIM`` is None the dim is derived from
    the first live call instead of hardcoded.
    """

    PROVIDER: str
    EXPECTED_DIM: int | None = None

    @pytest.fixture
    def production_config(self, configured_provider):
        return configured_provider(self.PROVIDER)

    def _build_ef(self, production_config, **overrides):
        config = {**production_config, **overrides}
        return create_embedding_function(self.PROVIDER, config)

    @pytest.mark.timeout(60)
    def test_embed_query_returns_expected_dim(self, production_config):
        ef = self._build_ef(production_config)
        vec = ef.embed_query("live provider smoke test query")
        assert len(vec) > 0
        if self.EXPECTED_DIM is not None:
            assert len(vec) == self.EXPECTED_DIM
        # A second call must be consistent in dimensionality.
        vec2 = ef.embed_query("a different short probe")
        assert len(vec2) == len(vec)

    @pytest.mark.timeout(60)
    def test_document_and_query_embeddings_same_dim(self, production_config):
        ef = self._build_ef(production_config)
        doc_vectors = ef(DOCS[:2])
        query_vector = ef.embed_query(DOCS[0])
        assert len(doc_vectors) == 2
        dims = {len(v) for v in doc_vectors} | {len(query_vector)}
        assert len(dims) == 1

    @pytest.mark.timeout(60)
    def test_sub_batching_request_count(self, production_config, wrap_embed_batch):
        ef = self._build_ef(production_config, request_batch_size=2)
        calls = wrap_embed_batch(ef)

        vectors = ef(DOCS)

        assert len(calls) == 3, "5 texts / request_batch_size=2 must issue exactly 3 _embed_batch calls"
        assert len(vectors) == 5
        dims = {len(v) for v in vectors}
        assert len(dims) == 1

    @pytest.mark.timeout(60)
    def test_config_round_trip(self, production_config, cosine_similarity):
        ef = self._build_ef(production_config)
        cfg = ef.get_config()
        # get_config() does not include api_key (OpenAI/Gemini both omit it
        # deliberately); build_from_config falls back to the provider's env
        # var, which may be unset on this machine. Inject the production
        # api_key explicitly so the round-trip doesn't depend on env state.
        cfg["api_key"] = production_config.get("api_key")
        rebuilt = type(ef).build_from_config(cfg)

        text = "config round trip probe text"
        v1 = ef.embed_query(text)
        v2 = rebuilt.embed_query(text)
        assert cosine_similarity(v1, v2) >= 0.999


class TestOpenAILive(_ProviderLiveTests):
    PROVIDER = "openai"
    EXPECTED_DIM = 1536  # text-embedding-3-small


class TestGeminiLive(_ProviderLiveTests):
    PROVIDER = "gemini"
    EXPECTED_DIM = None  # derive from the live call; model/dim isn't pinned here
