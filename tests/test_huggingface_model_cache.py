"""Regression test for issue #726: the HuggingFace provider must load a model once per process.

ChromaDB rebuilds a collection's embedding function from the persisted config on
every upsert and query (``Schema.deserialize_from_json`` -> ``build_from_config``),
so ``HuggingFaceEmbeddingFunction`` is constructed again and again. Each
construction used to create a new ``SentenceTransformer``, which reloaded the
weights every call and, on CUDA, exhausted GPU memory during ``update-db``.
"""

import sys
import types

import numpy as np
import pytest

# chromadb is an optional extra (``[semantic]``); skip where it isn't installed.
pytest.importorskip("chromadb")

from zotero_mcp.embeddings.providers import huggingface  # noqa: E402


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """Stand in for sentence-transformers and record every model construction."""
    constructed = []

    class FakeSentenceTransformer:
        max_seq_length = 512

        def __init__(self, model_name, **kwargs):
            constructed.append(model_name)

        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 4))

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(huggingface, "_MODEL_CACHE", {})
    return constructed


def test_same_model_name_is_loaded_once(fake_sentence_transformers):
    first = huggingface.HuggingFaceEmbeddingFunction("fake/model")
    # What ChromaDB does on every upsert/query.
    rebuilt = huggingface.HuggingFaceEmbeddingFunction.build_from_config(
        {"model_name": "fake/model", "api_key_env_var": "HUGGINGFACE_API_KEY"}
    )

    assert fake_sentence_transformers == ["fake/model"]
    assert rebuilt.model is first.model
    assert rebuilt.max_input_tokens == 512
    assert np.asarray(rebuilt(["a", "b"])).shape == (2, 4)


def test_different_model_names_are_loaded_separately(fake_sentence_transformers):
    huggingface.HuggingFaceEmbeddingFunction("fake/model-a")
    huggingface.HuggingFaceEmbeddingFunction("fake/model-b")
    huggingface.HuggingFaceEmbeddingFunction("fake/model-a")

    assert fake_sentence_transformers == ["fake/model-a", "fake/model-b"]
