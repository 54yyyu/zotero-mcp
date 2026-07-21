"""Tests for throttled Batch API submissions and the --auto-loop pipeline.

Covers token estimation, token-aware record slicing, throttled submission
(pending manifest entries), pending-chunk promotion, throttle config
resolution, and auto-loop termination with a scripted fake provider.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")

from zotero_mcp import batch_common, gemini_batch, openai_batch, semantic_search  # noqa: E402


# ---------------------------------------------------------------------------
# Token estimation + token-aware slicing
# ---------------------------------------------------------------------------


def test_estimate_tokens_is_deterministic_and_conservative():
    assert batch_common.estimate_tokens("") == 1
    assert batch_common.estimate_tokens("abc", 3.0) == 1
    assert batch_common.estimate_tokens("abcdefg", 3.0) == 3  # ceil(7/3)
    assert batch_common.estimate_tokens("abcdefg", 3.0) >= len("abcdefg") / 3


def test_split_respects_max_tokens():
    records = [
        {"id": f"ID{i}", "document": "x" * 150, "metadata": {}}  # 50 tokens each
        for i in range(5)
    ]
    chunks = openai_batch.split_embedding_records(
        records, "text-embedding-3-small", max_file_bytes=10_000_000, max_tokens=100
    )
    sizes = [len(chunk_records) for chunk_records, _ in chunks]
    assert sizes == [2, 2, 1]  # 100-token budget admits two 50-token records


def test_split_without_max_tokens_ignores_token_cap():
    records = [{"id": f"ID{i}", "document": "x" * 150, "metadata": {}} for i in range(5)]
    chunks = openai_batch.split_embedding_records(
        records, "text-embedding-3-small", max_file_bytes=10_000_000
    )
    assert [len(chunk_records) for chunk_records, _ in chunks] == [5]


# ---------------------------------------------------------------------------
# Throttled submission
# ---------------------------------------------------------------------------


class _FakeOpenAIClient:
    """Fake OpenAI client; each created batch gets a sequential id."""

    def __init__(self):
        self.created_batches = []

        client = self

        class FakeFiles:
            def create(self, file, purpose):
                assert purpose == "batch"
                return SimpleNamespace(id=f"file-{len(client.created_batches) + 1}")

        class FakeBatches:
            def create(self, **kwargs):
                batch_id = f"batch-{len(client.created_batches) + 1}"
                client.created_batches.append(batch_id)
                return SimpleNamespace(
                    id=batch_id,
                    status="validating",
                    output_file_id=None,
                    error_file_id=None,
                    request_counts={"total": 1, "completed": 0, "failed": 0},
                )

        self.files = FakeFiles()
        self.batches = FakeBatches()


def _records(n, doc_chars=150):
    return [{"id": f"ID{i}", "document": "x" * doc_chars, "metadata": {}} for i in range(n)]


def test_throttled_submit_parks_overflow_as_pending(tmp_path):
    client = _FakeOpenAIClient()
    manifest = openai_batch.submit_embedding_batches(
        records=_records(5),
        model_name="text-embedding-3-small",
        embedding_config={"api_key": "test"},
        config_path=str(tmp_path / "config.json"),
        client=client,
        max_enqueued_tokens=100,
    )

    assert client.created_batches == ["batch-1"]  # only the first chunk submitted
    statuses = [b["status"] for b in manifest["batches"]]
    assert statuses == ["validating", "pending", "pending"]
    assert manifest["batches"][0]["batch_id"] == "batch-1"
    assert manifest["batches"][1]["batch_id"] is None
    # Token accounting recorded for headroom math.
    assert manifest["batches"][0]["request_tokens"] == 100
    assert manifest["batches"][1]["request_tokens"] == 100
    assert manifest["batches"][2]["request_tokens"] == 50
    # Pending chunks are fully written to disk for later submission.
    for batch in manifest["batches"]:
        assert Path(batch["input_path"]).exists()
        assert Path(batch["records_path"]).exists()


def test_unthrottled_submit_keeps_legacy_behavior(tmp_path):
    client = _FakeOpenAIClient()
    manifest = openai_batch.submit_embedding_batches(
        records=_records(5),
        model_name="text-embedding-3-small",
        embedding_config={"api_key": "test"},
        config_path=str(tmp_path / "config.json"),
        client=client,
    )
    assert len(client.created_batches) == 1  # one chunk, submitted immediately
    assert all(b.get("batch_id") for b in manifest["batches"])
    assert not any(b["status"] == "pending" for b in manifest["batches"])


def test_oversized_single_chunk_is_still_submitted(tmp_path):
    """A chunk bigger than the whole budget must not deadlock the run."""
    client = _FakeOpenAIClient()
    manifest = openai_batch.submit_embedding_batches(
        records=_records(1, doc_chars=900),  # 300 tokens > 100-token budget
        model_name="text-embedding-3-small",
        embedding_config={"api_key": "test"},
        config_path=str(tmp_path / "config.json"),
        client=client,
        max_enqueued_tokens=100,
    )
    assert client.created_batches == ["batch-1"]
    assert manifest["batches"][0]["batch_id"] == "batch-1"


def test_submit_pending_batches_uses_freed_headroom(tmp_path):
    client = _FakeOpenAIClient()
    manifest = openai_batch.submit_embedding_batches(
        records=_records(5),
        model_name="text-embedding-3-small",
        embedding_config={"api_key": "test"},
        config_path=str(tmp_path / "config.json"),
        client=client,
        max_enqueued_tokens=100,
    )

    # First chunk completes -> headroom frees exactly one more chunk.
    manifest["batches"][0]["status"] = "completed"
    submitted = openai_batch.submit_pending_batches(manifest, {"api_key": "test"}, client=client)
    assert submitted == 1
    assert manifest["batches"][1]["batch_id"] == "batch-2"
    assert manifest["batches"][2]["status"] == "pending"  # 100 + 50 > 100

    # Second chunk completes -> the small tail chunk now fits.
    manifest["batches"][1]["status"] = "completed"
    submitted = openai_batch.submit_pending_batches(manifest, {"api_key": "test"}, client=client)
    assert submitted == 1
    assert manifest["batches"][2]["batch_id"] == "batch-3"
    assert client.created_batches == ["batch-1", "batch-2", "batch-3"]


def test_refresh_skips_pending_entries(tmp_path):
    class FakeBatches:
        def retrieve(self, batch_id):
            assert batch_id == "batch-1"
            return SimpleNamespace(
                status="completed",
                output_file_id="out-1",
                error_file_id=None,
                request_counts={"total": 1},
            )

    client = SimpleNamespace(batches=FakeBatches())
    manifest = openai_batch.submit_embedding_batches(
        records=_records(5),
        model_name="text-embedding-3-small",
        embedding_config={"api_key": "test"},
        config_path=str(tmp_path / "config.json"),
        client=_FakeOpenAIClient(),
        max_enqueued_tokens=100,
    )
    refreshed = openai_batch.refresh_manifest_status(manifest, {"api_key": "test"}, client=client)
    assert refreshed["batches"][0]["status"] == "completed"
    assert refreshed["batches"][1]["status"] == "pending"  # untouched, no API call


def test_gemini_throttled_submit_parks_pending(tmp_path):
    class FakeFiles:
        def upload(self, file, config):
            return SimpleNamespace(name="files/abc123")

    class FakeBatches:
        def __init__(self):
            self.created = []

        def create_embeddings(self, **kwargs):
            self.created.append(kwargs["model"])
            return SimpleNamespace(
                name=f"batches/job-{len(self.created)}",
                state=SimpleNamespace(name="JOB_STATE_PENDING"),
                dest=None,
            )

    fake_batches = FakeBatches()
    client = SimpleNamespace(files=FakeFiles(), batches=fake_batches)
    manifest = gemini_batch.submit_embedding_batches(
        records=_records(4, doc_chars=175),  # 50 tokens each at 3.5 chars/token
        model_name="gemini-embedding-001",
        embedding_config={"api_key": "test"},
        config_path=str(tmp_path / "config.json"),
        client=client,
        max_enqueued_tokens=100,
    )
    assert len(fake_batches.created) == 1
    assert [b["status"] for b in manifest["batches"]] == [
        "JOB_STATE_PENDING",
        "pending",
    ]


# ---------------------------------------------------------------------------
# Throttle config resolution
# ---------------------------------------------------------------------------


class _FakeChroma:
    def __init__(self):
        self.embedding_model = "openai"
        self.embedding_config = {"model_name": "text-embedding-3-small", "api_key": "test"}
        self.embedding_max_tokens = 8000

    def truncate_text(self, text, max_tokens=None):
        return text

    def get_existing_ids(self, ids):
        return set()


def test_load_batch_throttle_config_defaults_and_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"semantic_search": {}}), encoding="utf-8")
    search = semantic_search.ZoteroSemanticSearch(chroma_client=_FakeChroma(), config_path=str(cfg))
    throttle = search._load_batch_throttle_config("openai")
    assert throttle["batch_max_enqueued_tokens"] == openai_batch.OPENAI_BATCH_MAX_ENQUEUED_TOKENS
    assert throttle["batch_max_requests"] == 50_000

    cfg.write_text(
        json.dumps({
            "semantic_search": {
                "openai_batch": {"batch_max_enqueued_tokens": 18_000_000, "batch_max_requests": 10_000}
            }
        }),
        encoding="utf-8",
    )
    throttle = search._load_batch_throttle_config("openai")
    assert throttle["batch_max_enqueued_tokens"] == 18_000_000
    assert throttle["batch_max_requests"] == 10_000


def test_submit_batch_index_forwards_throttle_kwargs(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    captured = {}

    def capture_submit(**kwargs):
        captured.update(kwargs)
        return {"run_id": "r", "manifest_path": "/tmp/m.json", "batches": []}

    monkeypatch.setattr(semantic_search.openai_batch, "submit_embedding_batches", capture_submit)
    search = semantic_search.ZoteroSemanticSearch(chroma_client=_FakeChroma())
    item = {"key": "K1", "data": {"title": "T", "itemType": "journalArticle", "creators": []}}
    stats = {"processed_items": 0, "skipped_items": 0, "errors": 0}
    search._submit_batch_index(
        "openai", [item], False, None, stats,
        max_enqueued_tokens=123, max_requests=456,
    )
    assert captured["max_enqueued_tokens"] == 123
    assert captured["max_requests"] == 456


# ---------------------------------------------------------------------------
# Auto-loop pipeline
# ---------------------------------------------------------------------------


def test_auto_loop_drives_run_to_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    provider_client = _FakeOpenAIClient()
    monkeypatch.setattr(
        semantic_search.openai_batch, "create_openai_client", lambda cfg: provider_client
    )

    search = semantic_search.ZoteroSemanticSearch(
        chroma_client=_FakeChroma(), config_path=str(tmp_path / "config.json")
    )

    manifest = openai_batch.submit_embedding_batches(
        records=_records(5),
        model_name="text-embedding-3-small",
        embedding_config={"api_key": "test"},
        config_path=str(tmp_path / "config.json"),
        client=provider_client,
        max_enqueued_tokens=100,
    )

    def fake_import(self, provider, batch_ids=None, _skip_lock=False):
        """Each poll: complete+import the oldest submitted-but-not-imported batch."""
        assert _skip_lock is True  # auto-loop must not re-acquire the update lock
        m = openai_batch.find_manifest(config_path=str(tmp_path / "config.json"))
        imported = 0
        for batch in m["batches"]:
            if batch.get("batch_id") and not batch.get("imported_at"):
                batch["status"] = "completed"
                batch["imported_at"] = datetime.now().isoformat()
                imported = batch["request_count"]
                break
        openai_batch.save_manifest(m)
        return {"imported_items": imported}

    monkeypatch.setattr(semantic_search.ZoteroSemanticSearch, "_import_batch", fake_import)

    result = search.auto_loop_batch_pipeline(
        "openai", poll_interval=0, max_enqueued_tokens=100
    )

    assert "stalled" not in result
    assert result["submitted_chunks"] == 2  # the two pending chunks were promoted
    final = openai_batch.find_manifest(config_path=str(tmp_path / "config.json"))
    assert all(b.get("imported_at") for b in final["batches"])
    assert provider_client.created_batches == ["batch-1", "batch-2", "batch-3"]


def test_auto_loop_reports_stall_when_nothing_can_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(
        semantic_search.openai_batch, "create_openai_client", lambda cfg: _FakeOpenAIClient()
    )
    search = semantic_search.ZoteroSemanticSearch(
        chroma_client=_FakeChroma(), config_path=str(tmp_path / "config.json")
    )

    manifest = openai_batch.submit_embedding_batches(
        records=_records(2),
        model_name="text-embedding-3-small",
        embedding_config={"api_key": "test"},
        config_path=str(tmp_path / "config.json"),
        client=_FakeOpenAIClient(),
        max_enqueued_tokens=100,
    )
    # The only submitted batch fails terminally and nothing imports it.
    manifest["batches"][0]["status"] = "failed"
    openai_batch.save_manifest(manifest)

    monkeypatch.setattr(
        semantic_search.ZoteroSemanticSearch,
        "_import_batch",
        lambda self, provider, batch_ids=None, _skip_lock=False: {"imported_items": 0},
    )

    result = search.auto_loop_batch_pipeline("openai", poll_interval=0, max_enqueued_tokens=100)
    assert "stalled" in result
    assert result["polls"] == 1  # no infinite loop
