"""Tests for the Gemini Batch API helpers and orchestration (mirrors test_openai_batch.py)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")

from zotero_mcp import gemini_batch, openai_batch, semantic_search  # noqa: E402


def test_build_embedding_request_v1_uses_task_type():
    record = {"id": "ABC123", "document": "paper text", "metadata": {}}
    request = gemini_batch.build_embedding_request(record, "gemini-embedding-001")
    assert request["key"] == "ABC123"
    assert request["request"]["task_type"] == "RETRIEVAL_DOCUMENT"
    assert request["request"]["content"]["parts"][0]["text"] == "paper text"


def test_build_embedding_request_v2_uses_doc_prefix_not_task_type():
    record = {"id": "ABC123", "document": "paper text", "metadata": {}}
    request = gemini_batch.build_embedding_request(record, "gemini-embedding-2-flash")
    assert "task_type" not in request["request"]
    text = request["request"]["content"]["parts"][0]["text"]
    assert text.startswith("Represent this document for retrieval:")
    assert text.endswith("paper text")


def test_split_embedding_records_respects_request_limit():
    records = [{"id": f"ID{i}", "document": f"text {i}", "metadata": {}} for i in range(3)]
    chunks = gemini_batch.split_embedding_records(
        records, "gemini-embedding-001", max_requests=2, max_file_bytes=100_000
    )
    assert [len(chunk_records) for chunk_records, _ in chunks] == [2, 1]
    assert chunks[0][1][0]["key"] == "ID0"
    assert chunks[1][1][0]["key"] == "ID2"


def test_parse_embedding_output_matches_by_key():
    output = "\n".join([
        json.dumps({"key": "B", "response": {"embedding": {"values": [0.2, 0.3]}}}),
        json.dumps({"key": "A", "response": {"embedding": {"values": [0.1, 0.2]}}}),
        json.dumps({"key": "C", "error": {"message": "rate limited"}}),
    ])
    embeddings, failures = gemini_batch.parse_embedding_output(output, id_order=["A", "B", "C"])
    assert embeddings == {"B": [0.2, 0.3], "A": [0.1, 0.2]}
    assert failures == [{"custom_id": "C", "error": {"message": "rate limited"}}]


def test_parse_embedding_output_falls_back_to_positional_order():
    """Rows without a `key` field are matched by submitted order (SDK-guaranteed)."""
    output = "\n".join([
        json.dumps({"response": {"embedding": {"values": [0.1]}}}),
        json.dumps({"response": {"embedding": {"values": [0.2]}}}),
    ])
    embeddings, failures = gemini_batch.parse_embedding_output(output, id_order=["FIRST", "SECOND"])
    assert embeddings == {"FIRST": [0.1], "SECOND": [0.2]}
    assert failures == []


def test_submit_embedding_batches_writes_manifest_and_jsonl(tmp_path):
    class FakeFiles:
        def upload(self, file, config):
            assert Path(file).exists()
            return SimpleNamespace(name="files/abc123")

        def download(self, file):
            raise AssertionError("download should not be called during submit")

    class FakeBatches:
        def create_embeddings(self, **kwargs):
            assert kwargs["model"] == "gemini-embedding-001"
            assert kwargs["src"] == {"file_name": "files/abc123"}
            return SimpleNamespace(
                name="batches/job-1",
                state=SimpleNamespace(name="JOB_STATE_PENDING"),
                dest=None,
            )

    records = [{"id": "ABC123", "document": "paper text", "metadata": {"title": "A"}}]

    manifest = gemini_batch.submit_embedding_batches(
        records=records,
        model_name="gemini-embedding-001",
        embedding_config={"api_key": "test"},
        config_path=str(tmp_path / "config.json"),
        client=SimpleNamespace(files=FakeFiles(), batches=FakeBatches()),
    )

    assert manifest["provider"] == "gemini"
    assert manifest["batches"][0]["batch_id"] == "batches/job-1"
    assert manifest["batches"][0]["status"] == "JOB_STATE_PENDING"
    assert Path(manifest["manifest_path"]).exists()
    input_rows = openai_batch.read_jsonl(Path(manifest["batches"][0]["input_path"]))
    assert input_rows[0]["key"] == "ABC123"


def test_refresh_manifest_status_maps_job_state_enum(tmp_path):
    manifest = {
        "run_id": "run-1",
        "manifest_path": str(tmp_path / "manifest.json"),
        "batches": [{"batch_id": "batches/job-1", "status": "JOB_STATE_PENDING"}],
    }

    class FakeBatches:
        def get(self, name):
            assert name == "batches/job-1"
            return SimpleNamespace(state=SimpleNamespace(name="JOB_STATE_SUCCEEDED"), dest=None)

    refreshed = gemini_batch.refresh_manifest_status(
        manifest, embedding_config=None, client=SimpleNamespace(batches=FakeBatches())
    )
    assert refreshed["batches"][0]["status"] == "JOB_STATE_SUCCEEDED"
    assert "JOB_STATE_SUCCEEDED" in gemini_batch.GEMINI_IMPORTABLE_STATES


class FakeChromaClient:
    def __init__(self, embedding_model="gemini"):
        self.embedding_model = embedding_model
        self.embedding_config = {"model_name": "gemini-embedding-001", "api_key": "test"}
        self.embedding_max_tokens = 8000

    def truncate_text(self, text, max_tokens=None):
        return text

    def get_existing_ids(self, ids):
        return set()


def test_resolve_gemini_batch_enabled_reads_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"semantic_search": {"gemini_batch": {"enabled": True}}}), encoding="utf-8"
    )
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    search = semantic_search.ZoteroSemanticSearch(
        chroma_client=FakeChromaClient(), config_path=str(config_path)
    )
    assert search._resolve_gemini_batch_enabled(None) is True
    assert search._resolve_gemini_batch_enabled(False) is False

    non_gemini = semantic_search.ZoteroSemanticSearch(
        chroma_client=FakeChromaClient(embedding_model="openai"), config_path=str(config_path)
    )
    assert non_gemini._resolve_gemini_batch_enabled(True) is False


def test_import_gemini_batch_evicts_fulltext_cache_only_for_succeeded_ids(tmp_path, monkeypatch):
    class ImportChromaClient(FakeChromaClient):
        def __init__(self):
            super().__init__()
            self.upserted = None

        def upsert_embeddings(self, documents, metadatas, ids, embeddings):
            self.upserted = {"ids": list(ids)}

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
        json.dumps({"key": "A", "response": {"embedding": {"values": [0.1, 0.2]}}}) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": "run-1",
        "manifest_path": str(tmp_path / "manifest.json"),
        "force_full_rebuild": False,
        "batches": [
            {
                "batch_id": "batches/job-1",
                "status": "JOB_STATE_SUCCEEDED",
                "records_path": str(records_path),
                "imported_at": None,
            }
        ],
    }

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search.gemini_batch, "find_manifest", lambda **kwargs: manifest)
    monkeypatch.setattr(
        semantic_search.gemini_batch, "refresh_manifest_status", lambda manifest, **kwargs: manifest
    )
    monkeypatch.setattr(semantic_search.gemini_batch, "create_gemini_client", lambda config: object())

    evicted = {}

    def _fake_evict_many(keys, config_path=None):
        evicted["keys"] = set(keys)
        return len(evicted["keys"])

    monkeypatch.setattr(semantic_search.fulltext_cache, "evict_many", _fake_evict_many)

    chroma_client = ImportChromaClient()
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma_client)

    stats = search.import_gemini_batch()

    assert chroma_client.upserted == {"ids": ["A"]}
    assert stats["imported_items"] == 1
    assert stats["missing_items"] == 1  # "B" got no output row
    # Only the succeeded id's cache entry is evicted — "B" must survive so a
    # rerun can find its extracted text without redoing extraction.
    assert evicted["keys"] == {"A"}
