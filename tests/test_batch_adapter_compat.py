"""Compatibility tests for the ``BatchAdapter`` + normalized-state refactor.

Pins three things that must survive the move of the near-duplicated
submit/refresh/pending-promotion bodies out of ``openai_batch``/``gemini_batch``
and into ``batch_common``, parameterized by a per-provider adapter:

1. Pre-refactor (state-less) v2 manifests on disk still work: ``state`` is
   backfilled from the raw ``status`` without disturbing anything else, and
   import-eligibility/terminal-ness decisions computed from it agree with the
   old per-provider status sets.
2. Every raw status string the old ``GEMINI_TERMINAL_STATES``/
   ``GEMINI_IMPORTABLE_STATES`` and OpenAI's hardcoded
   ``{"completed", "failed", "expired", "cancelled"}`` covered, plus in-flight
   statuses, normalize to a state whose TERMINAL/IMPORTABLE membership agrees
   with the old sets.
3. Freshly-written manifests carry both the raw ``status`` and the new
   ``state`` on every entry, and still declare ``"version": 2``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from zotero_mcp import batch_common, gemini_batch, openai_batch

# ---------------------------------------------------------------------------
# Fake SDK clients (mirrors tests/test_batch_throttle.py's fakes)
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


class _FakeGeminiClient:
    """Fake google-genai client; each created batch job gets a sequential name."""

    def __init__(self):
        self.created = []

        class FakeFiles:
            def upload(self, file, config):
                return SimpleNamespace(name="files/abc123")

        class FakeBatches:
            def create_embeddings(self, **kwargs):
                self.created = getattr(self, "created", [])
                self.created.append(kwargs["model"])
                return SimpleNamespace(
                    name=f"batches/job-{len(self.created)}",
                    state=SimpleNamespace(name="JOB_STATE_PENDING"),
                    dest=None,
                )

        self.files = FakeFiles()
        self.batches = FakeBatches()


def _records(n, doc_chars=150):
    return [{"id": f"ID{i}", "document": "x" * doc_chars, "metadata": {}} for i in range(n)]


# ---------------------------------------------------------------------------
# Pre-refactor v2 manifest fixture (no ``state`` fields at all)
# ---------------------------------------------------------------------------


def _legacy_openai_manifest(tmp_path: Path) -> dict:
    return {
        "version": 2,
        "provider": "openai",
        "run_id": "legacy-run",
        "manifest_path": str(tmp_path / "manifest.json"),
        "max_enqueued_tokens": 1000,
        "batches": [
            {"batch_id": "batch-1", "status": "completed", "request_tokens": 100, "imported_at": None},
            {"batch_id": "batch-2", "status": "in_progress", "request_tokens": 100, "imported_at": None},
            {"batch_id": None, "status": "pending", "request_tokens": 50, "imported_at": None,
             "input_path": str(tmp_path / "batch-003-input.jsonl")},
        ],
    }


def _legacy_gemini_manifest(tmp_path: Path) -> dict:
    return {
        "version": 2,
        "provider": "gemini",
        "run_id": "legacy-run",
        "manifest_path": str(tmp_path / "manifest.json"),
        "max_enqueued_tokens": 1000,
        "batches": [
            {"batch_id": "batches/job-1", "status": "JOB_STATE_SUCCEEDED", "request_tokens": 100, "imported_at": None},
            {"batch_id": "batches/job-2", "status": "JOB_STATE_RUNNING", "request_tokens": 100, "imported_at": None},
            {"batch_id": None, "status": "pending", "request_tokens": 50, "imported_at": None,
             "input_path": str(tmp_path / "batch-003-input.jsonl")},
        ],
    }


def test_no_state_fields_in_legacy_fixture(tmp_path):
    """Sanity check on the fixture itself: pins the pre-refactor shape."""
    for manifest in (_legacy_openai_manifest(tmp_path), _legacy_gemini_manifest(tmp_path)):
        assert all("state" not in batch for batch in manifest["batches"])


# ---------------------------------------------------------------------------
# State backfill
# ---------------------------------------------------------------------------


def test_ensure_states_backfills_openai_manifest(tmp_path):
    manifest = _legacy_openai_manifest(tmp_path)
    backfilled = batch_common.ensure_states(manifest, openai_batch.ADAPTER)
    states = [b["state"] for b in backfilled["batches"]]
    assert states == [
        batch_common.STATE_SUCCEEDED,
        batch_common.STATE_IN_PROGRESS,
        batch_common.STATE_PENDING,
    ]
    # Raw status is untouched.
    assert [b["status"] for b in backfilled["batches"]] == ["completed", "in_progress", "pending"]


def test_ensure_states_backfills_gemini_manifest(tmp_path):
    manifest = _legacy_gemini_manifest(tmp_path)
    backfilled = batch_common.ensure_states(manifest, gemini_batch.ADAPTER)
    states = [b["state"] for b in backfilled["batches"]]
    assert states == [
        batch_common.STATE_SUCCEEDED,
        batch_common.STATE_IN_PROGRESS,
        batch_common.STATE_PENDING,
    ]
    assert [b["status"] for b in backfilled["batches"]] == [
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_RUNNING",
        "pending",
    ]


def test_ensure_states_does_not_override_an_existing_state(tmp_path):
    """Backfill is additive only: a manifest that already has ``state`` (even
    a nonsense value) is left alone — it's only for state-less legacy entries."""
    manifest = {
        "manifest_path": str(tmp_path / "manifest.json"),
        "batches": [{"status": "completed", "state": "custom-value"}],
    }
    batch_common.ensure_states(manifest, openai_batch.ADAPTER)
    assert manifest["batches"][0]["state"] == "custom-value"


def test_openai_find_manifest_backfills_state_from_disk(tmp_path):
    """The module-level ``find_manifest``/``load_manifest`` wrappers backfill
    ``state`` transparently for manifests already on disk, without changing
    their ``(config_path)``/``(path)`` signatures."""
    config_path = tmp_path / "config.json"
    root = openai_batch.get_openai_batch_root(str(config_path))
    run_dir = root / "legacy-run"
    run_dir.mkdir(parents=True)
    manifest = _legacy_openai_manifest(tmp_path)
    manifest["manifest_path"] = str(run_dir / "manifest.json")
    with open(manifest["manifest_path"], "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    loaded = openai_batch.find_manifest(config_path=str(config_path))
    assert [b["state"] for b in loaded["batches"]] == [
        batch_common.STATE_SUCCEEDED,
        batch_common.STATE_IN_PROGRESS,
        batch_common.STATE_PENDING,
    ]

    loaded_by_path = openai_batch.load_manifest(Path(manifest["manifest_path"]))
    assert loaded_by_path["batches"][0]["state"] == batch_common.STATE_SUCCEEDED


def test_gemini_find_manifest_backfills_state_from_disk(tmp_path):
    config_path = tmp_path / "config.json"
    root = gemini_batch.get_gemini_batch_root(str(config_path))
    run_dir = root / "legacy-run"
    run_dir.mkdir(parents=True)
    manifest = _legacy_gemini_manifest(tmp_path)
    manifest["manifest_path"] = str(run_dir / "manifest.json")
    with open(manifest["manifest_path"], "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    loaded = gemini_batch.find_manifest(config_path=str(config_path))
    assert [b["state"] for b in loaded["batches"]] == [
        batch_common.STATE_SUCCEEDED,
        batch_common.STATE_IN_PROGRESS,
        batch_common.STATE_PENDING,
    ]


# ---------------------------------------------------------------------------
# Import-eligibility / terminal-ness parity with the old per-provider sets
# ---------------------------------------------------------------------------


def test_openai_normalize_status_matches_old_terminal_and_importable_sets():
    old_terminal = {"completed", "failed", "expired", "cancelled"}
    old_importable = {"completed"}
    # "cancelling" is deliberately excluded here: the plan calls for mapping
    # it to STATE_CANCELLED (a batch mid-cancellation will never complete),
    # a one-time intentional change from the old code's implicit
    # not-in-terminal-set handling — see the dedicated test below.
    in_flight = {"validating", "in_progress", "finalizing", "pending"}

    for status in old_terminal | in_flight:
        normalized = openai_batch.ADAPTER.normalize_status(status)
        assert (normalized in batch_common.TERMINAL_STATES) == (status in old_terminal), status
        assert (normalized in batch_common.IMPORTABLE_STATES) == (status in old_importable), status


def test_openai_cancelling_status_normalizes_to_cancelled():
    """Deliberate mapping (per the refactor plan): a batch mid-cancellation
    is treated as terminal-cancelled rather than left "active" forever."""
    assert openai_batch.ADAPTER.normalize_status("cancelling") == batch_common.STATE_CANCELLED


def test_gemini_normalize_status_matches_old_terminal_and_importable_sets():
    old_terminal = gemini_batch.GEMINI_TERMINAL_STATES
    old_importable = gemini_batch.GEMINI_IMPORTABLE_STATES
    in_flight = {"JOB_STATE_PENDING", "JOB_STATE_QUEUED", "JOB_STATE_RUNNING", "JOB_STATE_UNSPECIFIED", "pending"}

    for status in old_terminal | in_flight:
        normalized = gemini_batch.ADAPTER.normalize_status(status)
        assert (normalized in batch_common.TERMINAL_STATES) == (status in old_terminal), status
        assert (normalized in batch_common.IMPORTABLE_STATES) == (status in old_importable), status


def test_partially_succeeded_is_importable_but_only_gemini_has_it():
    # Gemini has a "partial success" state OpenAI's Batch API has no analog for.
    assert gemini_batch.ADAPTER.normalize_status("JOB_STATE_PARTIALLY_SUCCEEDED") == batch_common.STATE_PARTIAL
    assert batch_common.STATE_PARTIAL in batch_common.IMPORTABLE_STATES
    assert batch_common.STATE_PARTIAL in batch_common.TERMINAL_STATES


def test_unknown_status_normalizes_to_non_terminal_in_progress():
    """An unrecognized future SDK status must not be mistaken for terminal —
    an auto-loop should keep polling rather than stall or misfire an import."""
    assert openai_batch.ADAPTER.normalize_status("some_new_status") == batch_common.STATE_IN_PROGRESS
    assert gemini_batch.ADAPTER.normalize_status("JOB_STATE_SOMETHING_NEW") == batch_common.STATE_IN_PROGRESS
    assert batch_common.STATE_IN_PROGRESS not in batch_common.TERMINAL_STATES


# ---------------------------------------------------------------------------
# submit_pending_batches: enqueued-token accounting unchanged
# ---------------------------------------------------------------------------


def test_submit_pending_batches_terminal_state_frees_budget_without_cached_state(tmp_path):
    """A manifest entry marked terminal only via raw ``status`` (no ``state``
    key at all — the legacy/on-the-fly-mutated shape) must free its tokens
    from the enqueued-budget count exactly as the old status-only check did."""
    input_path = tmp_path / "batch-003-input.jsonl"
    openai_batch.write_jsonl(input_path, [{"custom_id": "X"}])
    manifest = {
        "version": 2,
        "provider": "openai",
        "run_id": "run-1",
        "manifest_path": str(tmp_path / "manifest.json"),
        "max_enqueued_tokens": 100,
        "batches": [
            {"batch_id": "batch-1", "status": "completed", "request_tokens": 80,
             "input_path": str(tmp_path / "b1-input.jsonl")},
            {"batch_id": None, "status": "pending", "request_tokens": 50, "input_path": str(input_path)},
        ],
    }
    client = _FakeOpenAIClient()
    submitted = openai_batch.submit_pending_batches(manifest, {"api_key": "test"}, client=client)
    assert submitted == 1  # batch-1 is terminal -> full 100-token budget is free for the 50-token pending chunk
    assert manifest["batches"][1]["batch_id"] == "batch-1"  # first id the fake client hands out


def test_submit_pending_batches_non_terminal_state_still_blocks_budget(tmp_path):
    """A submitted-but-not-terminal entry keeps counting against the budget,
    exactly like the old ``status not in terminal`` check did."""
    input_path = tmp_path / "batch-003-input.jsonl"
    openai_batch.write_jsonl(input_path, [{"custom_id": "X"}])
    manifest = {
        "version": 2,
        "provider": "openai",
        "run_id": "run-1",
        "manifest_path": str(tmp_path / "manifest.json"),
        "max_enqueued_tokens": 100,
        "batches": [
            {"batch_id": "batch-1", "status": "in_progress", "request_tokens": 80,
             "input_path": str(tmp_path / "b1-input.jsonl")},
            {"batch_id": None, "status": "pending", "request_tokens": 50, "input_path": str(input_path)},
        ],
    }
    client = _FakeOpenAIClient()
    submitted = openai_batch.submit_pending_batches(manifest, {"api_key": "test"}, client=client)
    assert submitted == 0  # 80 + 50 > 100: still blocked, batch-1 isn't terminal


def test_submit_pending_batches_pending_entries_never_count_toward_enqueued(tmp_path):
    """Parked/pending chunks (``batch_id`` is ``None``) must not themselves be
    counted in the enqueued-token sum, matching the pre-refactor behavior."""
    first_pending = tmp_path / "batch-002-input.jsonl"
    second_pending = tmp_path / "batch-003-input.jsonl"
    openai_batch.write_jsonl(first_pending, [{"custom_id": "X"}])
    openai_batch.write_jsonl(second_pending, [{"custom_id": "Y"}])
    manifest = {
        "version": 2,
        "provider": "openai",
        "run_id": "run-1",
        "manifest_path": str(tmp_path / "manifest.json"),
        "max_enqueued_tokens": 60,
        "batches": [
            {"batch_id": None, "status": "pending", "request_tokens": 200, "input_path": str(first_pending)},
            {"batch_id": None, "status": "pending", "request_tokens": 40, "input_path": str(second_pending)},
        ],
    }
    client = _FakeOpenAIClient()
    submitted = openai_batch.submit_pending_batches(manifest, {"api_key": "test"}, client=client)
    # Nothing is currently enqueued (both entries are pending, not submitted,
    # so neither counts toward the sum) -> "always submit at least one" lets
    # the first one out even though 200 > 60; the second then overflows
    # (200 + 40 > 60) and stays parked.
    assert submitted == 1
    assert manifest["batches"][0]["batch_id"] == "batch-1"
    assert manifest["batches"][0]["state"] != batch_common.STATE_PENDING
    assert manifest["batches"][1]["batch_id"] is None
    assert manifest["batches"][1]["status"] == "pending"


# ---------------------------------------------------------------------------
# Freshly-written manifests: additive ``state`` next to raw ``status``, v2 kept
# ---------------------------------------------------------------------------


def test_new_openai_manifest_has_status_and_state_on_every_entry(tmp_path):
    client = _FakeOpenAIClient()
    manifest = openai_batch.submit_embedding_batches(
        records=_records(1),
        model_name="text-embedding-3-small",
        embedding_config={"api_key": "test"},
        config_path=str(tmp_path / "config.json"),
        client=client,
    )
    assert manifest["version"] == 2
    for batch in manifest["batches"]:
        assert "status" in batch and batch["status"]
        assert "state" in batch and batch["state"]
    assert manifest["batches"][0]["state"] == batch_common.STATE_SUBMITTED  # raw "validating"


def test_new_gemini_manifest_has_status_and_state_on_every_entry(tmp_path):
    client = _FakeGeminiClient()
    manifest = gemini_batch.submit_embedding_batches(
        records=_records(1),
        model_name="gemini-embedding-001",
        embedding_config={"api_key": "test"},
        config_path=str(tmp_path / "config.json"),
        client=client,
    )
    assert manifest["version"] == 2
    for batch in manifest["batches"]:
        assert "status" in batch and batch["status"]
        assert "state" in batch and batch["state"]
    assert manifest["batches"][0]["status"] == "JOB_STATE_PENDING"
    assert manifest["batches"][0]["state"] == batch_common.STATE_SUBMITTED


def test_new_openai_manifest_parked_pending_entry_has_pending_state(tmp_path):
    client = _FakeOpenAIClient()
    manifest = openai_batch.submit_embedding_batches(
        records=_records(3),
        model_name="text-embedding-3-small",
        embedding_config={"api_key": "test"},
        config_path=str(tmp_path / "config.json"),
        client=client,
        max_enqueued_tokens=50,
    )
    pending = [b for b in manifest["batches"] if b["status"] == "pending"]
    assert pending
    assert all(b["state"] == batch_common.STATE_PENDING for b in pending)
