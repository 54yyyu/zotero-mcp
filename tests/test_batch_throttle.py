"""Tests for throttled Batch API submissions and the --auto-loop pipeline.

Covers token estimation, token-aware record slicing, throttled submission
(pending manifest entries), pending-chunk promotion, throttle config
resolution, and auto-loop termination with a scripted fake provider.
"""

import contextlib
import json
import re
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

    openai_batch.submit_embedding_batches(
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


def test_auto_loop_counts_chunks_the_import_submitted(tmp_path, monkeypatch):
    """The import submits parked chunks itself now, so the loop must count those too.

    Otherwise the run summary ("N pending chunks submitted") reports 0 for a
    run whose chunks the import submitted.
    """
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    provider_client = _FakeOpenAIClient()
    monkeypatch.setattr(
        semantic_search.openai_batch, "create_openai_client", lambda cfg: provider_client
    )
    search = semantic_search.ZoteroSemanticSearch(
        chroma_client=_FakeChroma(), config_path=str(tmp_path / "config.json")
    )
    openai_batch.submit_embedding_batches(
        records=_records(5),
        model_name="text-embedding-3-small",
        embedding_config={"api_key": "test"},
        config_path=str(tmp_path / "config.json"),
        client=provider_client,
        max_enqueued_tokens=100,
    )

    def fake_import(self, provider, batch_ids=None, _skip_lock=False):
        """Stand in for the real import: submit the parked chunks, import everything."""
        m = openai_batch.find_manifest(config_path=str(tmp_path / "config.json"))
        submitted = 0
        for batch in m["batches"]:
            if not batch.get("batch_id"):
                batch_id = f"batch-{len(provider_client.created_batches) + 1}"
                provider_client.created_batches.append(batch_id)
                batch.update({"batch_id": batch_id, "status": "completed"})
                submitted += 1
            batch["imported_at"] = datetime.now().isoformat()
        openai_batch.save_manifest(m)
        return {"imported_items": 5, "batches_submitted": submitted}

    monkeypatch.setattr(semantic_search.ZoteroSemanticSearch, "_import_batch", fake_import)

    result = search.auto_loop_batch_pipeline("openai", poll_interval=0, max_enqueued_tokens=100)

    assert result["polls"] == 1
    assert result["submitted_chunks"] == 2  # both parked chunks, counted exactly once


# ---------------------------------------------------------------------------
# batch-import submits pending chunks (previously only --auto-loop did)
# ---------------------------------------------------------------------------


class _FakeOpenAIClientCompleting:
    """Provider that accepts any submission and reports every batch completed."""

    def __init__(self):
        self.created_batches = []
        client = self

        class FakeFiles:
            def create(self, file, purpose):
                return SimpleNamespace(id=f"file-{len(client.created_batches) + 1}")

        class FakeBatches:
            def create(self, **kwargs):
                batch_id = f"batch-{len(client.created_batches) + 1}"
                client.created_batches.append(batch_id)
                return SimpleNamespace(
                    id=batch_id, status="validating", output_file_id=None, error_file_id=None,
                    request_counts={"total": 1, "completed": 0, "failed": 0},
                )

            def retrieve(self, batch_id):
                return SimpleNamespace(
                    id=batch_id, status="completed", output_file_id=f"out-{batch_id}", error_file_id=None,
                    request_counts={"total": 2, "completed": 2, "failed": 0},
                )

        self.files = FakeFiles()
        self.batches = FakeBatches()


class _FakeChromaForImport:
    def __init__(self):
        self.embedding_model = "openai"
        self.embedding_config = {"model_name": "text-embedding-3-small", "api_key": "test"}
        self.embedding_max_tokens = 8000
        self.upserted = []
        self.reset_calls = 0

    def get_existing_ids(self, ids):
        return set()

    def upsert_embeddings(self, documents, metadatas, ids, embeddings):
        self.upserted.extend(ids)

    def reset_collection(self):
        self.reset_calls += 1


def _write_fake_outputs(manifest):
    """Pretend every submitted batch's output file was downloaded already."""
    for b in manifest["batches"]:
        if not b.get("batch_id"):
            continue
        rp = Path(b["records_path"])
        rows = [
            {"custom_id": r["id"], "response": {"status_code": 200, "body": {"data": [{"embedding": [0.1, 0.2]}]}}}
            for r in batch_common.read_jsonl(rp)
        ]
        rp.with_name(rp.stem + "-output.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


@pytest.fixture
def import_env(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    client = _FakeOpenAIClientCompleting()
    monkeypatch.setattr(openai_batch, "create_openai_client", lambda cfg: client)
    monkeypatch.setattr(semantic_search, "_acquire_update_lock", lambda p: contextlib.nullcontext(True))
    chroma = _FakeChromaForImport()
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma, config_path=str(cfg))
    promoted = []
    monkeypatch.setattr(search, "_save_update_config", lambda **kw: promoted.append(kw))
    return SimpleNamespace(search=search, client=client, chroma=chroma, cfg=str(cfg), promoted=promoted)


def _throttled_run(env, force):
    # 5 records, budget 100 tokens -> 3 chunks, only the first submitted.
    manifest = openai_batch.submit_embedding_batches(
        records=_records(5), model_name="text-embedding-3-small", embedding_config={"api_key": "t"},
        config_path=env.cfg, client=env.client, max_enqueued_tokens=100, force_full_rebuild=force,
    )
    assert [b["status"] for b in manifest["batches"]] == ["validating", "pending", "pending"]
    return manifest


def test_batch_import_submits_pending_chunks(import_env):
    # Chunks are 100, 100 and 50 tokens against a 100-token budget (verified
    # 2026-09-10), so each import round frees room for exactly one more chunk.
    env = import_env
    manifest = _throttled_run(env, force=False)
    _write_fake_outputs(manifest)

    stats = env.search._import_batch("openai")

    assert stats["batches_imported"] == 1
    assert stats["batches_submitted"] == 1
    assert env.client.created_batches == ["batch-1", "batch-2"]
    assert env.promoted == [], "watermark must not be promoted while chunks are outstanding"

    _write_fake_outputs(openai_batch.find_manifest(config_path=env.cfg))
    stats = env.search._import_batch("openai")

    assert stats["batches_imported"] == 1
    assert stats["batches_submitted"] == 1
    assert env.client.created_batches == ["batch-1", "batch-2", "batch-3"]
    assert env.promoted == []

    _write_fake_outputs(openai_batch.find_manifest(config_path=env.cfg))
    stats = env.search._import_batch("openai")

    assert stats["batches_imported"] == 1
    assert stats["batches_submitted"] == 0
    assert len(env.promoted) == 1, "all chunks imported -> watermark promoted once"


def test_force_rebuild_batch_import_submits_pending_instead_of_refusing(import_env):
    env = import_env
    manifest = _throttled_run(env, force=True)
    _write_fake_outputs(manifest)

    stats = env.search._import_batch("openai")  # must not raise

    assert stats["batches_imported"] == 0
    assert stats["batches_submitted"] == 1
    assert stats.get("deferred"), "the run is still incomplete; the stats must say so"
    assert env.chroma.reset_calls == 0, "the collection is reset only when the whole run imports"
    assert env.client.created_batches == ["batch-1", "batch-2"]

    _write_fake_outputs(openai_batch.find_manifest(config_path=env.cfg))
    stats = env.search._import_batch("openai")

    assert stats["batches_imported"] == 0
    assert stats["batches_submitted"] == 1
    assert stats.get("deferred")
    assert env.client.created_batches == ["batch-1", "batch-2", "batch-3"]

    _write_fake_outputs(openai_batch.find_manifest(config_path=env.cfg))
    stats = env.search._import_batch("openai")

    assert stats["batches_imported"] == 3
    assert stats["batches_submitted"] == 0
    assert env.chroma.reset_calls == 1
    assert len(env.promoted) == 1


def test_force_rebuild_batch_import_still_refuses_when_nothing_can_be_submitted(import_env):
    env = import_env
    _throttled_run(env, force=True)
    # Make the in-flight batch non-terminal so the budget stays consumed and
    # the pending chunks cannot be submitted.
    env.client.batches.retrieve = lambda batch_id: SimpleNamespace(
        id=batch_id, status="in_progress", output_file_id=None, error_file_id=None,
        request_counts={"total": 2, "completed": 0, "failed": 0},
    )

    with pytest.raises(RuntimeError, match="can only be imported after all batches complete"):
        env.search._import_batch("openai")
    assert env.client.created_batches == ["batch-1"]


def test_batch_import_refuses_to_submit_into_a_superseded_run(import_env):
    env = import_env
    old = _throttled_run(env, force=False)
    _write_fake_outputs(old)
    # A later update-db --batch mints a newer run; the old run's pending
    # chunks must not be submitted (the new run re-submitted everything).
    import time
    time.sleep(0.02)
    newer = openai_batch.submit_embedding_batches(
        records=_records(5), model_name="text-embedding-3-small", embedding_config={"api_key": "t"},
        config_path=env.cfg, client=env.client, max_enqueued_tokens=None, force_full_rebuild=False,
    )
    assert newer["manifest_path"] != old["manifest_path"]
    created_before = list(env.client.created_batches)

    stats = env.search._import_batch("openai", batch_ids=["batch-1"])

    assert stats["batches_submitted"] == 0
    assert env.client.created_batches == created_before
    assert any("superseded" in str(e) for e in stats["errors"])


def _set_created_at(manifest_path, value):
    """Runs are minutes apart in reality; make the ordering explicit in tests."""
    manifest = openai_batch.load_manifest(Path(manifest_path))
    manifest["created_at"] = value
    openai_batch.save_manifest(manifest)


def _two_runs(env):
    """An old throttled run plus a newer unthrottled one that supersedes it."""
    old = _throttled_run(env, force=False)
    _write_fake_outputs(old)
    newer = openai_batch.submit_embedding_batches(
        records=_records(5), model_name="text-embedding-3-small", embedding_config={"api_key": "t"},
        config_path=env.cfg, client=env.client, max_enqueued_tokens=None, force_full_rebuild=False,
    )
    _set_created_at(old["manifest_path"], "2026-09-10T10:00:00.000000Z")
    _set_created_at(newer["manifest_path"], "2026-09-10T10:05:00.000000Z")
    return old, newer


def test_run_order_is_by_created_at_then_run_id_and_survives_a_resave(tmp_path):
    """Ranking must use fields written once at submission, not file mtime.

    Every status refresh and every import re-saves a manifest, so an mtime
    order would make "newest run" mean "run looked at most recently".
    """
    cfg = str(tmp_path / "config.json")
    assert openai_batch.newest_run_path(config_path=cfg) is None  # no runs yet

    client = _FakeOpenAIClientCompleting()
    kwargs = {
        "records": _records(2), "model_name": "text-embedding-3-small",
        "embedding_config": {"api_key": "t"}, "config_path": cfg, "client": client,
    }
    first = openai_batch.submit_embedding_batches(**kwargs)
    second = openai_batch.submit_embedding_batches(**kwargs)
    # Identical created_at: the run id, not the clock, has to settle the order.
    _set_created_at(first["manifest_path"], "2026-09-10T10:00:00.000000Z")
    _set_created_at(second["manifest_path"], "2026-09-10T10:00:00.000000Z")
    newest_run_id = max(first["run_id"], second["run_id"])
    loser = first if first["run_id"] != newest_run_id else second

    def newest():
        return openai_batch.load_manifest(Path(openai_batch.newest_run_path(config_path=cfg)))["run_id"]

    assert newest() == newest_run_id
    assert openai_batch.find_manifest(config_path=cfg)["run_id"] == newest_run_id
    assert [openai_batch.load_manifest(p)["run_id"] for p in openai_batch.iter_manifests(cfg)] == sorted(
        [first["run_id"], second["run_id"]], reverse=True
    )

    # A status check on the run that lost re-saves its manifest...
    openai_batch.refresh_manifest_status(
        openai_batch.load_manifest(Path(loser["manifest_path"])), {"api_key": "t"}, client=client
    )
    # ...which must not promote it.
    assert newest() == newest_run_id
    assert openai_batch.find_manifest(config_path=cfg)["run_id"] == newest_run_id


def test_submission_timestamps_separate_runs_minted_in_the_same_second():
    import time

    # Two back-to-back calls land in the same microsecond about a third of the
    # time, so wait for the clock to move rather than assume it has - real runs
    # are never that close. What must hold is that the wait is far shorter than
    # the second the two stamps share.
    started = time.monotonic()
    stamps = {batch_common._utc_now()}
    while len(stamps) < 2:
        stamps.add(batch_common._utc_now())
        assert time.monotonic() - started < 0.5, "stamps within one second must already differ"

    assert len(stamps) == 2
    # Fixed width and microsecond resolution: the order is lexicographic.
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z", stamp) for stamp in stamps)
    earlier, later = sorted(stamps)
    assert later > earlier


def test_superseded_run_stays_refused_after_a_refusal_bumps_its_mtime(import_env):
    """Refusing an import re-saves the manifest; that must not promote the run.

    Runs are ordered by ``created_at`` then ``run_id``, both stamped once at
    submission, precisely so this cannot happen: under the file-mtime order
    this used to have, every import re-saved the manifest it refreshed, so the
    run just refused became the "newest" one and the identical next command
    submitted its chunks after all.
    """
    env = import_env
    old, _ = _two_runs(env)
    created_before = list(env.client.created_batches)

    for attempt in (1, 2):
        stats = env.search._import_batch("openai", batch_ids=["batch-1"])
        assert stats["batches_submitted"] == 0, f"attempt {attempt} submitted into a superseded run"
        assert any("superseded" in str(e) for e in stats["errors"])
    assert env.client.created_batches == created_before
    assert [b["status"] for b in openai_batch.load_manifest(Path(old["manifest_path"]))["batches"][1:]] == [
        "pending", "pending",
    ]


def test_a_status_check_on_an_old_run_does_not_redirect_the_plain_import(import_env):
    """'batch-status' on an old run must not hand the plain import that run.

    A status refresh re-saves the manifest it read, which under an mtime order
    made the old run the newest one - so the next plain ``batch-import`` landed
    on it and submitted its parked chunks.
    """
    env = import_env
    old, newer = _two_runs(env)
    _write_fake_outputs(newer)
    created_before = list(env.client.created_batches)

    openai_batch.refresh_manifest_status(
        openai_batch.load_manifest(Path(old["manifest_path"])), {"api_key": "t"}, client=env.client
    )
    assert openai_batch.find_manifest(config_path=env.cfg)["run_id"] == newer["run_id"]

    stats = env.search._import_batch("openai")  # no batch ids: the plain path

    assert stats["run_id"] == newer["run_id"], "the plain import must work on the current run"
    assert stats["batches_submitted"] == 0
    assert env.client.created_batches == created_before
    assert [b["status"] for b in openai_batch.load_manifest(Path(old["manifest_path"]))["batches"][1:]] == [
        "pending", "pending",
    ]


def test_a_newer_run_for_another_library_does_not_supersede(import_env):
    """Supersession is per library: another library's run covers none of these items."""
    env = import_env
    old = _throttled_run(env, force=False)  # group_id None: the personal library
    openai_batch.submit_embedding_batches(
        records=_records(5), model_name="text-embedding-3-small", embedding_config={"api_key": "t"},
        config_path=env.cfg, client=env.client, max_enqueued_tokens=None, group_id=7,
    )
    created_before = list(env.client.created_batches)
    old["batches"][0]["status"] = "completed"  # frees the enqueued-token headroom

    superseded_by = env.search._superseded_by("openai", old)

    assert superseded_by is None
    assert env.search._submit_pending_chunks("openai", old, env.client, superseded_by) == 1
    assert env.client.created_batches == created_before + ["batch-3"]


def test_a_newer_incremental_run_does_not_supersede_a_force_rebuild_run(import_env):
    """An incremental run re-embeds nothing a force-rebuild run was rebuilding."""
    env = import_env
    force_run = _throttled_run(env, force=True)
    common = {
        "records": _records(5), "model_name": "text-embedding-3-small",
        "embedding_config": {"api_key": "t"}, "config_path": env.cfg,
        "client": env.client, "max_enqueued_tokens": None,
    }
    openai_batch.submit_embedding_batches(**common, force_full_rebuild=False)
    assert env.search._superseded_by("openai", force_run) is None

    newer_force = openai_batch.submit_embedding_batches(**common, force_full_rebuild=True)
    assert env.search._superseded_by("openai", force_run) == newer_force["run_id"]


def test_a_superseded_completed_run_imports_without_promoting_the_watermark(import_env):
    """The newer run was cut from the same watermark and owns advancing it."""
    env = import_env
    old = openai_batch.submit_embedding_batches(
        records=_records(2), model_name="text-embedding-3-small", embedding_config={"api_key": "t"},
        config_path=env.cfg, client=env.client, max_enqueued_tokens=None,
    )
    _write_fake_outputs(old)
    openai_batch.submit_embedding_batches(
        records=_records(5), model_name="text-embedding-3-small", embedding_config={"api_key": "t"},
        config_path=env.cfg, client=env.client, max_enqueued_tokens=None,
    )

    stats = env.search._import_batch("openai", batch_ids=["batch-1"])

    assert stats["batches_imported"] == 1  # the embeddings are still worth having
    assert env.promoted == [], "a superseded run must not promote the sync watermark"
    assert any("watermark" in str(e) for e in stats["errors"])


def test_import_by_batch_id_never_submits_pending_chunks(import_env):
    """Naming batch ids asks for those batches, not for the run to move on."""
    env = import_env
    manifest = _throttled_run(env, force=False)
    _write_fake_outputs(manifest)
    assert env.search._superseded_by("openai", manifest) is None  # it is the current run

    stats = env.search._import_batch("openai", batch_ids=["batch-1"])

    assert stats["batches_imported"] == 1
    assert stats["batches_submitted"] == 0
    assert env.client.created_batches == ["batch-1"]
    assert [b["status"] for b in openai_batch.find_manifest(config_path=env.cfg)["batches"][1:]] == [
        "pending", "pending",
    ]


def test_auto_loop_does_not_submit_into_a_superseded_run(import_env, monkeypatch):
    """The loop submits through the same guard as every other path."""
    env = import_env
    old = _throttled_run(env, force=False)
    # The one submitted chunk fails terminally, so its tokens stop counting and
    # an unguarded loop would have room to submit the parked chunks.
    old["batches"][0]["status"] = "failed"
    openai_batch.save_manifest(old)
    openai_batch.submit_embedding_batches(
        records=_records(5), model_name="text-embedding-3-small", embedding_config={"api_key": "t"},
        config_path=env.cfg, client=env.client, max_enqueued_tokens=None,
    )
    created_before = list(env.client.created_batches)

    # The loop drives whatever find_manifest hands it; pin that to the
    # superseded run so this exercises the guard, not the run ordering.
    monkeypatch.setattr(
        openai_batch, "find_manifest",
        lambda config_path=None, batch_id=None: openai_batch.load_manifest(Path(old["manifest_path"])),
    )
    monkeypatch.setattr(
        semantic_search.ZoteroSemanticSearch, "_import_batch",
        lambda self, provider, batch_ids=None, _skip_lock=False: {"imported_items": 0},
    )

    result = env.search.auto_loop_batch_pipeline("openai", poll_interval=0, max_enqueued_tokens=100)

    assert result["submitted_chunks"] == 0
    assert result["polls"] == 1  # nothing can progress: it stops rather than spinning
    assert "stalled" in result
    assert env.client.created_batches == created_before


def test_print_batch_import_reports_submitted_and_deferred(capsys):
    from zotero_mcp.cli import _print_batch_import

    _print_batch_import({"run_id": "r", "manifest_path": "m", "batches_seen": 3, "batches_imported": 0,
                         "batches_skipped": 3, "batches_submitted": 2,
                         "deferred": "2 pending chunk(s) submitted; the force-rebuild run imports once all batches complete"},
                        "openai")
    out = capsys.readouterr().out
    assert "Pending chunks submitted: 2" in out
    assert "imports once all batches complete" in out
    assert "batch-import" in out
