"""Query-only instructions for OpenAI-compatible embeddings.

Instruction-tuned models such as Qwen3-Embedding or Harrier are often served
through an OpenAI-compatible API (LM Studio, vLLM, llama-server). #763 added
``embedding_config.query_instruction`` for Ollama and HuggingFace; these tests
pin the same contract for the OpenAI provider.

Request-shape tests build the instance via ``__new__`` with a fake client, so
they run without the optional ``openai`` package. Construction, factory and
persistence tests need it and are skipped without it.
"""

import tempfile
import threading

import pytest

pytest.importorskip("chromadb")

from zotero_mcp.embeddings.providers.openai import OpenAIEmbeddingFunction  # noqa: E402


class _Embeddings:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(list(kwargs["input"]))
        data = [type("D", (), {"embedding": [1.0, 0.0]}) for _ in kwargs["input"]]
        return type("R", (), {"data": data})


class _FakeOpenAI:
    """Stand-in for ``openai.OpenAI``: records every input list sent."""

    instances = []

    def __init__(self, **kwargs):
        self.embeddings = _Embeddings()
        _FakeOpenAI.instances.append(self)


def _sent():
    return [call for inst in _FakeOpenAI.instances for call in inst.embeddings.calls]


def _make(query_instruction=""):
    """An OpenAIEmbeddingFunction with a fake client, bypassing __init__."""
    ef = OpenAIEmbeddingFunction.__new__(OpenAIEmbeddingFunction)
    ef.model_name = "harrier-oss-v1-0.6b"
    ef.base_url = "http://127.0.0.1:1234/v1"
    ef.request_batch_size = 64
    ef.rate_limit_rps = None
    ef.dimensions = None
    ef.max_parallel_requests = 1
    ef.max_retries = 0
    ef._rate_lock = threading.Lock()
    ef._last_request_ts = 0.0
    if query_instruction is not None:
        ef.query_instruction = query_instruction
    ef.client = _FakeOpenAI()
    return ef, ef.client.embeddings.calls


@pytest.fixture
def fake_openai(monkeypatch):
    openai = pytest.importorskip("openai")
    _FakeOpenAI.instances = []
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    return _FakeOpenAI


def test_instruction_applies_to_queries_only():
    ef, calls = _make("Find papers")
    ef(["Document text"])
    assert calls[-1] == ["Document text"]
    assert list(ef.embed_query_text("query")) == [1.0, 0.0]
    assert calls[-1] == ["Instruct: Find papers\nQuery: query"]


def test_batch_queries_each_get_one_instruction():
    ef, calls = _make("Find papers")
    assert [list(v) for v in ef.embed_query(["one", "two"])] == [[1.0, 0.0], [1.0, 0.0]]
    assert calls[-2:] == [["Instruct: Find papers\nQuery: one"], ["Instruct: Find papers\nQuery: two"]]


def test_instance_without_attribute_keeps_plain_queries():
    """An instance that predates the attribute (or bypasses __init__) is unchanged."""
    ef, calls = _make(query_instruction=None)
    ef.embed_query_text("unchanged")
    assert calls[-1] == ["unchanged"]


def test_multiline_unicode_query_is_preserved():
    ef, calls = _make("Find papers")
    ef.embed_query_text("空间转录组\ncell mapping")
    assert calls[-1] == ["Instruct: Find papers\nQuery: 空间转录组\ncell mapping"]


def test_factory_forwards_instruction(fake_openai):
    from zotero_mcp.embeddings.registry import create_embedding_function

    ef = create_embedding_function("openai", {
        "model_name": "harrier", "api_key": "lm-studio",
        "base_url": "http://127.0.0.1:1234/v1", "query_instruction": " Find papers ",
    })
    assert ef.query_instruction == "Find papers"
    ef.embed_query_text("query")
    assert _sent()[-1] == ["Instruct: Find papers\nQuery: query"]


@pytest.mark.parametrize("instruction", [None, "", "  \n "])
def test_empty_instruction_keeps_plain_queries(fake_openai, instruction):
    ef = OpenAIEmbeddingFunction(api_key="x", query_instruction=instruction)
    ef.embed_query_text("unchanged")
    assert _sent()[-1] == ["unchanged"]


@pytest.mark.parametrize("value", [True, 123, ["instruction"]])
def test_invalid_instruction_fails_clearly(fake_openai, value):
    with pytest.raises(ValueError, match="query_instruction"):
        OpenAIEmbeddingFunction(api_key="x", query_instruction=value)


def test_old_config_keeps_plain_queries(fake_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    ef = OpenAIEmbeddingFunction.build_from_config({"model_name": "m", "api_key_env_var": "OPENAI_API_KEY"})
    ef.embed_query_text("unchanged")
    assert _sent()[-1] == ["unchanged"]


def test_instruction_survives_config_round_trip(fake_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    ef = OpenAIEmbeddingFunction(api_key="x", query_instruction=" Find papers ")
    config = ef.get_config()
    assert config["query_instruction"] == "Find papers"
    rebuilt = OpenAIEmbeddingFunction.build_from_config(config)
    rebuilt.embed_query_text("query")
    assert _sent()[-1] == ["Instruct: Find papers\nQuery: query"]


def test_existing_collection_keeps_vectors_when_instruction_is_added(fake_openai):
    """Adding the setting to an indexed library must not reset or re-embed it."""
    from zotero_mcp.chroma_client import ChromaClient

    config = {"model_name": "harrier", "api_key": "lm-studio", "base_url": "http://127.0.0.1:1234/v1"}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        first = ChromaClient(persist_directory=directory, embedding_model="openai", embedding_config=config)
        first.upsert_documents(["Document"], [{"title": "Paper"}], ["item1"])
        collection_id = first.collection.id
        sent_before = len(_sent())

        second = ChromaClient(
            persist_directory=directory, embedding_model="openai",
            embedding_config={**config, "query_instruction": "Find papers"},
        )
        assert second.collection.id == collection_id
        assert second.collection.count() == 1
        assert len(_sent()) == sent_before  # opening did not re-embed
        assert second.search(["query"], n_results=1)["ids"] == [["item1"]]
        assert _sent()[sent_before:] == [["Instruct: Find papers\nQuery: query"]]
