"""EmbeddingGemma task prefixes on the Ollama provider."""

from unittest.mock import patch

from zotero_mcp.embeddings.providers.ollama import OllamaEmbeddingFunction


def _capture_inputs(ef):
    sent = []

    def fake_embed(texts, is_query=False):
        sent.append((list(texts), is_query))
        return [[0.0] for _ in texts]

    return sent, patch.object(ef, "_embed_batch", side_effect=fake_embed)


def test_embeddinggemma_documents_and_queries_get_prefixes():
    ef = OllamaEmbeddingFunction(model_name="embeddinggemma-2:270m-bf16-text")
    sent, patcher = _capture_inputs(ef)
    with patcher:
        ef(["Moral crumple zones"])
        ef.embed_query_text("who is responsible")
    assert sent[0][0] == ["title: none | text: Moral crumple zones"]
    assert sent[1][0] == ["task: search result | query: who is responsible"]
    assert sent[1][1] is True


def test_embeddinggemma_reserves_prefix_budget():
    ef = OllamaEmbeddingFunction(model_name="embeddinggemma:300m")
    assert ef.max_input_tokens == OllamaEmbeddingFunction.max_input_tokens - ef.GEMMA_PREFIX_TOKEN_BUDGET


def test_other_ollama_models_are_unchanged():
    ef = OllamaEmbeddingFunction(model_name="bge-m3:latest")
    sent, patcher = _capture_inputs(ef)
    with patcher:
        ef(["plain doc"])
        ef.embed_query_text("plain query")
    assert sent[0][0] == ["plain doc"]
    assert sent[1][0] == ["plain query"]
    assert ef.max_input_tokens == OllamaEmbeddingFunction.max_input_tokens


def test_config_roundtrip_keeps_prefix_behaviour():
    ef = OllamaEmbeddingFunction(model_name="embeddinggemma-2:270m-bf16-text")
    rebuilt = OllamaEmbeddingFunction.build_from_config(ef.get_config())
    assert rebuilt._prepare_query("q") == "task: search result | query: q"
