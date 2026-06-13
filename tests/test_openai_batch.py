import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from zotero_mcp import openai_batch, setup_helper

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")

from zotero_mcp import semantic_search  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient  # noqa: E402


def test_build_embedding_request_uses_batch_embeddings_shape():
    record = {"id": "ABC123", "document": "paper text", "metadata": {"title": "A"}}

    request = openai_batch.build_embedding_request(record, "text-embedding-3-small")

    assert request == {
        "custom_id": "ABC123",
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {
            "model": "text-embedding-3-small",
            "input": "paper text",
            "encoding_format": "float",
        },
    }


def test_split_embedding_records_respects_request_limit():
    records = [
        {"id": f"ID{i}", "document": f"text {i}", "metadata": {}}
        for i in range(3)
    ]

    chunks = openai_batch.split_embedding_records(
        records,
        "text-embedding-3-small",
        max_requests=2,
        max_file_bytes=10_000,
    )

    assert [len(chunk_records) for chunk_records, _ in chunks] == [2, 1]
    assert chunks[0][1][0]["custom_id"] == "ID0"
    assert chunks[1][1][0]["custom_id"] == "ID2"


def test_parse_embedding_output_uses_custom_ids_and_keeps_failures():
    output = "\n".join(
        [
            json.dumps({
                "custom_id": "B",
                "response": {"status_code": 200, "body": {"data": [{"embedding": [0.2, 0.3]}]}},
            }),
            json.dumps({
                "custom_id": "A",
                "response": {"status_code": 200, "body": {"data": [{"embedding": [0.1, 0.2]}]}},
            }),
            json.dumps({
                "custom_id": "C",
                "response": {"status_code": 429, "body": {}},
                "error": {"message": "rate limited"},
            }),
        ]
    )

    embeddings, failures = openai_batch.parse_embedding_output(output)

    assert embeddings == {"B": [0.2, 0.3], "A": [0.1, 0.2]}
    assert failures == [{"custom_id": "C", "error": {"message": "rate limited"}, "status_code": 429}]


def test_submit_embedding_batches_writes_manifest_and_jsonl(tmp_path):
    class FakeFiles:
        def create(self, file, purpose):
            assert purpose == "batch"
            assert Path(file.name).read_text(encoding="utf-8")
            return SimpleNamespace(id="file-1")

    class FakeBatches:
        def create(self, **kwargs):
            assert kwargs["endpoint"] == "/v1/embeddings"
            assert kwargs["completion_window"] == "24h"
            return SimpleNamespace(
                id="batch-1",
                status="validating",
                output_file_id=None,
                error_file_id=None,
                request_counts={"total": 1, "completed": 0, "failed": 0},
            )

    records = [{"id": "ABC123", "document": "paper text", "metadata": {"title": "A"}}]

    manifest = openai_batch.submit_embedding_batches(
        records=records,
        model_name="text-embedding-3-small",
        embedding_config={"api_key": "test"},
        config_path=str(tmp_path / "config.json"),
        client=SimpleNamespace(files=FakeFiles(), batches=FakeBatches()),
    )

    assert manifest["batches"][0]["batch_id"] == "batch-1"
    assert Path(manifest["manifest_path"]).exists()
    input_rows = openai_batch.read_jsonl(Path(manifest["batches"][0]["input_path"]))
    assert input_rows[0]["url"] == "/v1/embeddings"


def test_setup_openai_new_config_defaults_to_batch(monkeypatch):
    answers = iter([
        "2",  # OpenAI
        "1",  # text-embedding-3-small
        "",  # default base URL
        "",  # default batch choice: yes for new configs
        "1",  # manual updates
        "",  # default PDF max pages
        "",  # auto-detect DB path
    ])
    monkeypatch.setattr(builtins, "input", lambda *args: next(answers))
    monkeypatch.setattr(setup_helper.getpass, "getpass", lambda *args: "sk-test")

    config = setup_helper.setup_semantic_search()

    assert config["embedding_model"] == "openai"
    assert config["openai_batch"] == {"enabled": True}


class FakeChromaClient:
    def __init__(self, embedding_model="openai"):
        self.embedding_model = embedding_model
        self.embedding_config = {"model_name": "text-embedding-3-small", "api_key": "test"}


def test_update_db_batch_flag_resolution_reads_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"semantic_search": {"openai_batch": {"enabled": True}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    search = semantic_search.ZoteroSemanticSearch(
        chroma_client=FakeChromaClient(),
        config_path=str(config_path),
    )

    assert search._resolve_openai_batch_enabled(None) is True
    assert search._resolve_openai_batch_enabled(False) is False
    assert search._resolve_openai_batch_enabled(True) is True

    non_openai = semantic_search.ZoteroSemanticSearch(
        chroma_client=FakeChromaClient(embedding_model="gemini"),
        config_path=str(config_path),
    )
    assert non_openai._resolve_openai_batch_enabled(True) is False


def test_chroma_client_upsert_embeddings_passes_precomputed_vectors():
    class FakeCollection:
        def __init__(self):
            self.kwargs = None

        def upsert(self, **kwargs):
            self.kwargs = kwargs

    client = ChromaClient.__new__(ChromaClient)
    client.collection = FakeCollection()

    client.upsert_embeddings(
        documents=["doc"],
        metadatas=[{"title": "Title"}],
        ids=["ID1"],
        embeddings=[[0.1, 0.2]],
    )

    assert client.collection.kwargs == {
        "documents": ["doc"],
        "metadatas": [{"title": "Title"}],
        "ids": ["ID1"],
        "embeddings": [[0.1, 0.2]],
    }
