"""Query-only instructions for Ollama and HuggingFace embeddings (#614)."""

import tempfile
import unittest
from unittest.mock import Mock, patch

import chromadb

from zotero_mcp.embeddings.providers import huggingface, ollama
from zotero_mcp.embeddings.registry import create_embedding_function


class FakeModel:
    max_seq_length = 512

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return Mock(tolist=lambda: [[1.0, 0.0] for _ in texts])


class TestQueryInstruction(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        loader = patch.object(huggingface, "_load_model", return_value=self.model)
        loader.start()
        self.addCleanup(loader.stop)
        self.post = Mock(side_effect=self.response)
        requests = patch("requests.post", self.post)
        requests.start()
        self.addCleanup(requests.stop)

    @staticmethod
    def response(url, *, json, timeout):
        return Mock(json=lambda: {"embeddings": [[1.0, 0.0] for _ in json["input"]]})

    def received(self, provider):
        if provider == "ollama":
            return self.post.call_args.kwargs["json"]["input"]
        return self.model.calls[-1]

    def test_factory_applies_instruction_only_to_queries(self):
        for provider in ("ollama", "huggingface"):
            with self.subTest(provider=provider):
                ef = create_embedding_function(provider, {"query_instruction": "Find papers"})
                ef(["Document text"])
                self.assertEqual(self.received(provider), ["Document text"])
                self.assertEqual(ef.embed_query_text("query"), [1.0, 0.0])
                self.assertEqual(self.received(provider), ["Instruct: Find papers\nQuery: query"])

    def test_batch_queries_each_get_one_instruction(self):
        for provider in ("ollama", "huggingface"):
            with self.subTest(provider=provider):
                ef = create_embedding_function(provider, {"query_instruction": "Find papers"})
                self.assertEqual(ef.embed_query(["one", "two"]), [[1.0, 0.0], [1.0, 0.0]])
                self.assertEqual(self.received(provider), ["Instruct: Find papers\nQuery: two"])
                self.assertEqual(ef.embed_query([]), [])

    def test_old_config_keeps_plain_queries(self):
        for cls in (ollama.OllamaEmbeddingFunction, huggingface.HuggingFaceEmbeddingFunction):
            with self.subTest(provider=cls.name()):
                ef = cls.build_from_config({"model_name": "fake/model"})
                ef.embed_query_text("unchanged")
                self.assertEqual(self.received(cls.name()), ["unchanged"])

    def test_empty_instruction_keeps_plain_queries(self):
        for provider in ("ollama", "huggingface"):
            for instruction in (None, "", "  \n "):
                with self.subTest(provider=provider, instruction=instruction):
                    ef = create_embedding_function(provider, {"query_instruction": instruction})
                    ef.embed_query_text("unchanged")
                    self.assertEqual(self.received(provider), ["unchanged"])

    def test_instruction_survives_config_round_trip(self):
        for provider in ("ollama", "huggingface"):
            with self.subTest(provider=provider):
                ef = create_embedding_function(provider, {"query_instruction": " Find papers "})
                rebuilt = type(ef).build_from_config(ef.get_config())
                self.assertEqual(rebuilt.get_config()["query_instruction"], "Find papers")
                rebuilt.embed_query_text("query")
                self.assertEqual(self.received(provider), ["Instruct: Find papers\nQuery: query"])

    def test_invalid_instruction_fails_clearly(self):
        for provider in ("ollama", "huggingface"):
            for value in (True, 123, ["instruction"]):
                with self.subTest(provider=provider, value=value):
                    with self.assertRaisesRegex(ValueError, "query_instruction"):
                        create_embedding_function(provider, {"query_instruction": value})

    def test_multiline_unicode_query_is_preserved(self):
        for provider in ("ollama", "huggingface"):
            with self.subTest(provider=provider):
                ef = create_embedding_function(provider, {"query_instruction": "Find papers"})
                ef.embed_query_text("空间转录组\ncell mapping")
                self.assertEqual(self.received(provider), ["Instruct: Find papers\nQuery: 空间转录组\ncell mapping"])

    def test_shared_huggingface_model_does_not_share_instructions(self):
        first = create_embedding_function("qwen", {"query_instruction": "First task"})
        second = create_embedding_function("qwen", {"query_instruction": "Second task"})
        self.assertIs(first.model, second.model)
        first.embed_query_text("query")
        self.assertEqual(self.model.calls[-1], ["Instruct: First task\nQuery: query"])
        second.embed_query_text("query")
        self.assertEqual(self.model.calls[-1], ["Instruct: Second task\nQuery: query"])
        first(["Document"])
        self.assertEqual(self.model.calls[-1], ["Document"])

    @unittest.skipUnless(hasattr(chromadb, "PersistentClient"), "requires real ChromaDB")
    def test_existing_collection_keeps_vectors_when_instruction_changes(self):
        from zotero_mcp.chroma_client import ChromaClient

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            first = ChromaClient(persist_directory=directory, embedding_model="qwen")
            first.add_documents(["Document"], [{"title": "Paper"}], ["item1"])
            collection_id = first.collection.id
            self.model.calls.clear()

            second = ChromaClient(
                persist_directory=directory, embedding_model="qwen",
                embedding_config={"query_instruction": "Find papers"},
            )
            self.assertEqual(second.collection.id, collection_id)
            self.assertEqual(second.collection.count(), 1)
            self.assertEqual(self.model.calls, [])  # opening did not re-embed
            self.assertEqual(second.search(["query"], n_results=1)["ids"], [["item1"]])
            self.assertEqual(self.model.calls, [["Instruct: Find papers\nQuery: query"]])
