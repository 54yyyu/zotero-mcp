"""Optional caps of the HuggingFace provider for small GPUs: max_seq_length, batch_size, and the model's own dtype.

A long-context model (Qwen3-Embedding: 32k tokens) pads each batch to its longest
text, so a full-text update-db can run out of GPU memory; ``max_seq_length`` and
``batch_size`` in ``semantic_search.embedding_config`` cap it.
"""

import sys
import types

import numpy as np
import pytest

# chromadb is an optional extra (``[semantic]``); skip where it isn't installed.
pytest.importorskip("chromadb")

from zotero_mcp.embeddings import registry  # noqa: E402
from zotero_mcp.embeddings.providers import huggingface  # noqa: E402


@pytest.fixture
def fake_st(monkeypatch):
    """Stand in for sentence-transformers; record constructions and encode calls."""
    log = {"built": [], "encode": []}

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            self.max_seq_length = 32768
            log["built"].append((model_name, kwargs.get("model_kwargs")))

        def encode(self, texts, **kwargs):
            log["encode"].append(kwargs)
            return np.zeros((len(texts), 4))

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(huggingface, "_MODEL_CACHE", {})
    monkeypatch.setattr(huggingface, "_gpu_model_kwargs", lambda: {})
    return log


def test_max_seq_length_caps_the_model(fake_st):
    ef = huggingface.HuggingFaceEmbeddingFunction("fake/m", max_seq_length=2048)
    assert ef.model.max_seq_length == 2048
    assert ef.max_input_tokens == 2048


def test_a_cap_above_the_model_limit_is_ignored(fake_st):
    ef = huggingface.HuggingFaceEmbeddingFunction("fake/m", max_seq_length=100_000)
    assert ef.model.max_seq_length == 32768


def test_batch_size_reaches_encode(fake_st):
    ef = huggingface.HuggingFaceEmbeddingFunction("fake/m", batch_size=8)
    ef(["a", "b"])
    assert fake_st["encode"][-1]["batch_size"] == 8


def test_without_caps_nothing_changes(fake_st):
    ef = huggingface.HuggingFaceEmbeddingFunction("fake/m")
    ef(["a"])
    assert ef.model.max_seq_length == 32768
    assert "batch_size" not in fake_st["encode"][-1]
    config = ef.get_config()
    assert "max_seq_length" not in config and "batch_size" not in config


def test_caps_survive_the_config_round_trip(fake_st):
    ef = huggingface.HuggingFaceEmbeddingFunction("fake/m", max_seq_length=2048, batch_size=8)
    rebuilt = huggingface.HuggingFaceEmbeddingFunction.build_from_config(ef.get_config())
    assert (rebuilt.max_seq_length, rebuilt.batch_size) == (2048, 8)
    assert rebuilt.model is ef.model  # same name and cap: one model in the process


def test_different_caps_load_separate_models(fake_st):
    a = huggingface.HuggingFaceEmbeddingFunction("fake/m", max_seq_length=512)
    b = huggingface.HuggingFaceEmbeddingFunction("fake/m", max_seq_length=2048)
    assert a.model is not b.model
    assert (a.model.max_seq_length, b.model.max_seq_length) == (512, 2048)


def test_registry_factory_passes_the_caps(fake_st):
    ef = registry._huggingface_ef_factory(
        {"model_name": "fake/m", "max_seq_length": 1024, "batch_size": 4})
    assert (ef.max_seq_length, ef.batch_size, ef.model.max_seq_length) == (1024, 4, 1024)


@pytest.mark.parametrize("cuda,version,expected", [
    (False, "5.2.0", {}),
    (True, "4.55.4", {"torch_dtype": "auto"}),
    (True, "4.56.0", {"dtype": "auto"}),
    (True, "5.2.0", {"dtype": "auto"}),
])
def test_gpu_dtype_keyword_follows_transformers(monkeypatch, cuda, version, expected):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    transformers = types.ModuleType("transformers")
    transformers.__version__ = version
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    assert huggingface._gpu_model_kwargs() == expected
