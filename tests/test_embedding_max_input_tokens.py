"""embedding_config.max_input_tokens overrides the per-document token cap."""

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("openai")

from zotero_mcp.embeddings.registry import create_embedding_function  # noqa: E402


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_config_caps_input_tokens(provider):
    ef = create_embedding_function(provider, {"api_key": "x", "max_input_tokens": 3000})
    assert ef.max_input_tokens == 3000
    assert len(ef.truncate("word " * 10_000, ef.max_input_tokens).split()) <= 3000


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_provider_default_without_it(provider):
    assert create_embedding_function(provider, {"api_key": "x"}).max_input_tokens == 8000
