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
        self.embedding_max_tokens = 8000

    def truncate_text(self, text, max_tokens=None):
        return text

    def get_existing_ids(self, ids):
        return {"EXISTING"} & set(ids)


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


def test_failed_batch_submit_does_not_report_added_or_updated(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())

    def fail_submit(**kwargs):
        raise RuntimeError("missing files.write")

    monkeypatch.setattr(semantic_search.openai_batch, "submit_embedding_batches", fail_submit)
    stats = {
        "processed_items": 0,
        "added_items": 0,
        "updated_items": 0,
        "skipped_items": 0,
        "errors": 0,
    }

    with pytest.raises(RuntimeError, match="missing files.write"):
        search._submit_openai_batch_index(
            [
                {
                    "key": "EXISTING",
                    "data": {
                        "title": "Existing item",
                        "itemType": "journalArticle",
                        "abstractNote": "A",
                        "creators": [],
                    },
                }
            ],
            force_full_rebuild=False,
            target_sync_version=1,
            stats=stats,
        )

    assert stats["processed_items"] == 1
    assert stats["added_items"] == 0
    assert stats["updated_items"] == 0
    assert "estimated_added_items" not in stats
    assert "estimated_updated_items" not in stats


def test_batch_and_realtime_indexing_share_prepared_payload(monkeypatch):
    class RecordingChromaClient(FakeChromaClient):
        def __init__(self):
            super().__init__()
            self.upserts = []

        def upsert_documents(self, documents, metadatas, ids):
            self.upserts.append({
                "documents": list(documents),
                "metadatas": list(metadatas),
                "ids": list(ids),
            })

    item = {
        "key": "ITEM1",
        "data": {
            "itemType": "journalArticle",
            "title": "Semantic Batch Paper",
            "abstractNote": "An abstract that should be embedded.",
            "publicationTitle": "Journal of Tests",
            "creators": [{"firstName": "Ada", "lastName": "Lovelace", "creatorType": "author"}],
            "tags": [{"tag": "embeddings"}, {"tag": "batch"}],
            "fulltext": "Full text must be included in both indexing paths.",
            "fulltextSource": "zotero_web_api",
            "date": "2026",
            "DOI": "10.1234/example",
        },
    }

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())

    realtime_client = RecordingChromaClient()
    realtime_search = semantic_search.ZoteroSemanticSearch(chroma_client=realtime_client)
    realtime_search._process_item_batch([item])
    assert realtime_client.upserts

    captured = {}

    def capture_batch_submit(**kwargs):
        captured.update(kwargs)
        return {
            "run_id": "run-1",
            "manifest_path": "/tmp/manifest.json",
            "batches": [{"batch_id": "batch-1"}],
        }

    monkeypatch.setattr(semantic_search.openai_batch, "submit_embedding_batches", capture_batch_submit)
    batch_client = RecordingChromaClient()
    batch_search = semantic_search.ZoteroSemanticSearch(chroma_client=batch_client)
    batch_search._submit_openai_batch_index(
        [item],
        force_full_rebuild=False,
        target_sync_version=123,
        stats={
            "processed_items": 0,
            "skipped_items": 0,
            "errors": 0,
        },
    )

    realtime_payload = realtime_client.upserts[0]
    assert captured["records"] == [{
        "id": realtime_payload["ids"][0],
        "document": realtime_payload["documents"][0],
        "metadata": realtime_payload["metadatas"][0],
    }]
    assert "Full text must be included" in captured["records"][0]["document"]
    assert captured["records"][0]["metadata"]["has_fulltext"] is True


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


def test_import_openai_batch_reports_records_missing_from_output(tmp_path, monkeypatch):
    class ImportChromaClient(FakeChromaClient):
        def __init__(self):
            super().__init__()
            self.upserted = None

        def upsert_embeddings(self, documents, metadatas, ids, embeddings):
            self.upserted = {
                "documents": list(documents),
                "metadatas": list(metadatas),
                "ids": list(ids),
                "embeddings": list(embeddings),
            }

    records_path = tmp_path / "batch-001-records.jsonl"
    openai_batch.write_jsonl(
        records_path,
        [
            {"id": "A", "document": "doc A", "metadata": {"title": "A"}},
            {"id": "B", "document": "doc B", "metadata": {"title": "B"}},
        ],
    )
    output_path = tmp_path / "batch-001-records-output.jsonl"
    output_path.write_text(
        json.dumps({
            "custom_id": "A",
            "response": {"status_code": 200, "body": {"data": [{"embedding": [0.1, 0.2]}]}},
        }) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": "run-1",
        "manifest_path": str(tmp_path / "manifest.json"),
        "force_full_rebuild": False,
        "batches": [
            {
                "batch_id": "batch-1",
                "status": "completed",
                "output_file_id": "file-1",
                "records_path": str(records_path),
                "imported_at": None,
            }
        ],
    }

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search.openai_batch, "find_manifest", lambda **kwargs: manifest)
    monkeypatch.setattr(
        semantic_search.openai_batch,
        "refresh_manifest_status",
        lambda manifest, **kwargs: manifest,
    )
    monkeypatch.setattr(semantic_search.openai_batch, "create_openai_client", lambda config: object())

    chroma_client = ImportChromaClient()
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma_client)

    stats = search.import_openai_batch()

    assert chroma_client.upserted == {
        "documents": ["doc A"],
        "metadatas": [{"title": "A"}],
        "ids": ["A"],
        "embeddings": [[0.1, 0.2]],
    }
    assert stats["imported_items"] == 1
    assert stats["missing_items"] == 1
    assert {
        "custom_id": "B",
        "error": "No embedding or error row returned for batch record",
    } in stats["errors"]
