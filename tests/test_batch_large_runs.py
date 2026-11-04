"""What a Batch API run needs once it spans many files and days.

Runs the real manifest, submit and import code against a fake provider, whose
batches finish however a test scripts them.
"""

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

from zotero_mcp import batch_common, openai_batch, semantic_search  # noqa: E402


class FakeOpenAI:
    """Batches stay in progress until a test finishes them."""

    def __init__(self):
        self.uploads: dict[str, str] = {}
        self.outputs: dict[str, str] = {}
        self.errors: dict[str, str] = {}
        self.jobs: dict[str, SimpleNamespace] = {}
        self.downloads: list[str] = []
        self.files = SimpleNamespace(create=self._upload, content=self._content)
        self.batches = SimpleNamespace(create=self._create, retrieve=lambda job_id: self.jobs[job_id])

    def _upload(self, file, purpose):
        file_id = f"file-{len(self.uploads)}"
        self.uploads[file_id] = Path(file.name).read_text(encoding="utf-8")
        return SimpleNamespace(id=file_id)

    def _content(self, file_id):
        self.downloads.append(file_id)
        return SimpleNamespace(text=self.outputs[file_id] if file_id in self.outputs else self.errors[file_id])

    def _create(self, input_file_id, **kwargs):
        job_id = f"job-{len(self.jobs)}"
        self.jobs[job_id] = SimpleNamespace(
            id=job_id,
            status="in_progress",
            output_file_id=None,
            error_file_id=None,
            request_counts=None,
            input_file_id=input_file_id,
        )
        return self.jobs[job_id]

    def ids_in(self, job_id):
        rows = self.uploads[self.jobs[job_id].input_file_id].splitlines()
        return [json.loads(row)["custom_id"] for row in rows]

    def finish(self, job_id, *, fail=(), status="completed"):
        """Complete a job, leaving out the ids in ``fail`` with an error row each."""
        job = self.jobs[job_id]
        rows = [
            {"custom_id": doc_id, "response": {"status_code": 200, "body": {"data": [{"embedding": [1.0, 0.0]}]}}}
            for doc_id in self.ids_in(job_id)
            if doc_id not in fail
        ]
        job.status = status
        job.output_file_id = f"out-{job_id}"
        self.outputs[job.output_file_id] = "\n".join(json.dumps(row) for row in rows)
        if fail:
            job.error_file_id = f"err-{job_id}"
            self.errors[job.error_file_id] = "\n".join(
                json.dumps({"custom_id": doc_id, "error": {"code": "batch_expired"}}) for doc_id in fail
            )

    def fail(self, job_id):
        self.jobs[job_id].status = "failed"

    def unfinished(self):
        return [job_id for job_id, job in self.jobs.items() if job.status == "in_progress"]


class FakeChroma:
    embedding_model = "openai"
    embedding_config = {"model_name": "text-embedding-3-small", "api_key": "test"}
    embedding_max_tokens = 8000

    def __init__(self):
        self.stored: dict[str, dict] = {"EXISTING": {}}
        self.resets = 0

    def get_existing_ids(self, ids):
        return {i for i in ids if i in self.stored}

    def upsert_embeddings(self, documents, metadatas, ids, embeddings):
        self.stored.update(zip(ids, metadatas))

    def reset_collection(self):
        self.resets += 1
        self.stored = {}


def _records(n):
    return [{"id": f"K{i:03d}", "document": "x" * 30, "metadata": {"item_key": f"K{i:03d}"}} for i in range(n)]


@pytest.fixture()
def run(monkeypatch, tmp_path):
    """Submit a run of two-record files, one file allowed in the queue at a time."""
    fake = FakeOpenAI()
    monkeypatch.setattr(openai_batch, "create_openai_client", lambda *_a, **_k: fake)
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"semantic_search": {"update_config": {}}}))
    chroma = FakeChroma()
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma, config_path=str(config))

    def _submit(n=4, *, force=False, budget=25):
        manifest = openai_batch.submit_embedding_batches(
            records=_records(n),
            model_name="text-embedding-3-small",
            embedding_config=chroma.embedding_config,
            config_path=str(config),
            force_full_rebuild=force,
            target_sync_version=7,
            group_id=0,
            client=fake,
            max_enqueued_tokens=budget,
            max_requests=2,
        )
        return manifest

    return SimpleNamespace(
        fake=fake,
        chroma=chroma,
        search=search,
        submit=_submit,
        config=config,
        manifest=lambda: openai_batch.find_manifest(str(config)),
        watermark=lambda: search._load_last_sync_version(),
    )


def _finish_everything(run):
    """Import and finish until no job is left."""
    for _ in range(20):
        run.search._import_batch("openai")
        waiting = run.fake.unfinished()
        if not waiting:
            break
        for job_id in waiting:
            run.fake.finish(job_id)
    run.search._import_batch("openai")


def test_a_failed_file_is_submitted_again(run):
    run.submit()
    run.fake.fail("job-0")

    stats = run.search._import_batch("openai")

    assert stats["batches_submitted"] == 1
    assert run.fake.ids_in("job-1") == ["K000", "K001"]


def test_a_failed_file_stops_after_its_retries_and_the_rest_still_lands(run):
    run.submit()
    # The first file fails each time it is submitted, whatever else is running.
    for _ in range(10):
        first = [j for j in run.fake.unfinished() if "K000" in run.fake.ids_in(j)]
        others = [j for j in run.fake.unfinished() if j not in first]
        for job_id in first:
            run.fake.fail(job_id)
        for job_id in others:
            run.fake.finish(job_id)
        run.search._import_batch("openai")

    attempts = [j for j in run.fake.jobs if "K000" in run.fake.ids_in(j)]
    assert len(attempts) == batch_common.MAX_BATCH_ATTEMPTS
    assert set(run.chroma.stored) == {"EXISTING", "K002", "K003"}
    manifest = run.manifest()
    assert sorted(manifest["failed_ids"]) == ["K000", "K001"]


def test_an_abandoned_file_holds_the_sync_version_back(run):
    run.submit(n=2)
    run.fake.fail("job-0")
    run.search._import_batch("openai")
    run.fake.fail("job-1")
    run.search._import_batch("openai")
    run.fake.fail("job-2")
    stats = run.search._import_batch("openai")

    assert run.watermark() == 0
    assert any("after retries" in str(e.get("error")) for e in stats["errors"])


def test_failed_rows_are_sent_again_on_their_own(run):
    run.submit(n=2)
    run.fake.finish("job-0", fail=["K001"])

    stats = run.search._import_batch("openai")

    assert set(run.chroma.stored) == {"EXISTING", "K000"}
    assert stats["batches_submitted"] == 1
    assert run.fake.ids_in("job-1") == ["K001"]
    assert run.watermark() == 0, "not until the retry has landed"

    run.fake.finish("job-1")
    run.search._import_batch("openai")

    assert set(run.chroma.stored) == {"EXISTING", "K000", "K001"}
    assert run.watermark() == 7


def test_rows_that_keep_failing_hold_the_sync_version_back(run):
    """Realtime holds it back too, or an incremental run never sees the item again (#610)."""
    run.submit(n=2)
    run.fake.finish("job-0", fail=["K001"])
    run.search._import_batch("openai")
    run.fake.finish("job-1", fail=["K001"])
    run.search._import_batch("openai")
    run.fake.finish("job-2", fail=["K001"])
    run.search._import_batch("openai")

    assert len(run.fake.jobs) == batch_common.MAX_BATCH_ATTEMPTS
    assert run.manifest()["failed_ids"] == ["K001"]
    assert run.watermark() == 0


def test_an_expired_batch_keeps_its_finished_rows_and_retries_the_rest(run):
    """OpenAI charges for rows that finished before the window closed."""
    run.submit(n=2)
    run.fake.finish("job-0", fail=["K001"], status="expired")

    run.search._import_batch("openai")

    assert set(run.chroma.stored) == {"EXISTING", "K000"}
    assert run.fake.ids_in("job-1") == ["K001"]


def test_finished_files_are_removed_after_import(run):
    """At a million passages these outputs are tens of gigabytes."""
    run.submit(n=2)
    entry = run.manifest()["batches"][0]
    records_path = Path(entry["records_path"])
    run.fake.finish("job-0")

    run.search._import_batch("openai")

    leftovers = [p.name for p in records_path.parent.iterdir() if p.suffix == ".jsonl"]
    assert leftovers == []
    assert run.manifest()["batches"][0]["imported_at"]


def test_an_interrupted_import_does_not_import_a_finished_file_again(run, monkeypatch):
    run.submit(budget=100)
    for job_id in list(run.fake.jobs):
        run.fake.finish(job_id)
    real = run.chroma.upsert_embeddings
    calls = {"n": 0}

    def fail_second(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("interrupted")
        real(*args, **kwargs)

    run.chroma.upsert_embeddings = fail_second
    with pytest.raises(RuntimeError, match="interrupted"):
        run.search._import_batch("openai")

    run.fake.downloads.clear()
    run.chroma.upsert_embeddings = real
    stats = run.search._import_batch("openai")

    assert stats["batches_skipped"] == 1
    assert stats["batches_imported"] == 1
    assert set(run.chroma.stored) == {"EXISTING", "K000", "K001", "K002", "K003"}


def test_force_rebuild_downloads_results_before_the_last_file_finishes(run):
    """Provider retention can be shorter than a throttled run."""
    run.submit(force=True, budget=100)
    run.fake.finish("job-0")

    with pytest.raises(RuntimeError, match="after all batches complete"):
        run.search._import_batch("openai")

    entry = run.manifest()["batches"][0]
    records_path = Path(entry["records_path"])
    assert records_path.with_name(records_path.stem + "-output.jsonl").exists()
    assert run.chroma.resets == 0, "the live index stays untouched until the whole run is in"


def test_force_rebuild_is_not_blocked_forever_by_a_file_that_keeps_failing(run):
    run.submit(force=True, budget=100)
    for _ in range(batch_common.MAX_BATCH_ATTEMPTS):
        failing = [j for j in run.fake.unfinished() if "K000" in run.fake.ids_in(j)]
        others = [j for j in run.fake.unfinished() if j not in failing]
        for job_id in failing:
            run.fake.fail(job_id)
        for job_id in others:
            run.fake.finish(job_id)
        try:
            run.search._import_batch("openai")
        except RuntimeError:
            pass

    assert set(run.chroma.stored) == {"EXISTING", "K002", "K003"}
    assert run.chroma.resets == 0, "an incomplete rebuild must not replace the index"
    assert run.manifest()["failed_ids"] == ["K000", "K001"]
    assert run.watermark() == 0


def test_force_rebuild_where_every_file_fails_leaves_the_index_alone(run):
    run.submit(n=2, force=True, budget=100)
    for _ in range(batch_common.MAX_BATCH_ATTEMPTS + 2):
        for job_id in run.fake.unfinished():
            run.fake.fail(job_id)
        try:
            run.search._import_batch("openai")
        except RuntimeError:
            pass

    assert run.chroma.resets == 0
    assert set(run.chroma.stored) == {"EXISTING"}


def test_force_rebuild_row_that_keeps_failing_is_retried_without_a_full_scan(run):
    """Going back to a full scan would embed the whole library again for one record."""
    run.submit(n=2, force=True, budget=100)
    run.fake.finish("job-0", fail=["K001"])
    run.search._import_batch("openai")
    run.fake.finish("job-1", fail=["K001"])
    run.search._import_batch("openai")
    run.fake.finish("job-2", fail=["K001"])
    run.search._import_batch("openai")

    assert run.chroma.resets == 1
    assert set(run.chroma.stored) == {"K000"}
    assert run.watermark() == 7
    assert run.search.update_config["retry_item_keys"] == ["K001"]


def test_a_record_that_keeps_failing_does_not_hold_back_a_split_index(run):
    """Holding the sync version would pay for the run's other items again, every run."""
    run.chroma.separate_passages = lambda: True
    run.submit(n=2)
    for job_id in ("job-0", "job-1", "job-2"):
        run.fake.finish(job_id, fail=["K001"])
        run.search._import_batch("openai")

    assert run.watermark() == 7
    assert run.search.update_config["retry_item_keys"] == ["K001"]


def test_a_superseded_force_rebuild_does_not_reset_the_index(run):
    older = run.submit(n=2, force=True, budget=100)
    run.submit(n=2, force=True, budget=100)
    older_id = older["batches"][0]["batch_id"]
    run.fake.finish(older_id)

    run.search._import_batch("openai", [older_id])

    assert run.chroma.resets == 0


def test_repeating_an_import_does_not_reset_the_index_again(run):
    run.submit(n=2, force=True, budget=100)
    run.fake.finish("job-0")
    run.search._import_batch("openai")
    assert run.chroma.resets == 1

    run.search._import_batch("openai")

    assert run.chroma.resets == 1


def test_a_superseded_run_retries_nothing(run):
    run.submit(n=2)
    newer_manifest = run.submit(n=2)
    older = [p for p in openai_batch.iter_manifests(str(run.config))][-1]
    older_id = json.loads(Path(older).read_text())["batches"][0]["batch_id"]
    assert newer_manifest["run_id"]
    run.fake.finish(older_id, fail=["K001"])
    before = len(run.fake.jobs)

    run.search._import_batch("openai", [older_id])

    assert len(run.fake.jobs) == before


def test_a_clean_run_still_promotes_the_sync_version(run):
    run.submit()

    _finish_everything(run)

    assert set(run.chroma.stored) == {"EXISTING", "K000", "K001", "K002", "K003"}
    assert run.watermark() == 7


def test_auto_loop_stops_when_only_abandoned_files_are_left(run, monkeypatch):
    run.submit(n=2)
    monkeypatch.setattr("time.sleep", lambda _s: run.fake.unfinished() and run.fake.fail(run.fake.unfinished()[0]))
    run.fake.fail("job-0")

    aggregate = run.search.auto_loop_batch_pipeline("openai", poll_interval=0)

    assert aggregate["abandoned"] == 1
    assert run.manifest()["failed_ids"] == ["K000", "K001"]
