"""Passage chunking on the Batch API path, OpenAI and Gemini (#416).

Runs update_database and the real manifest, split and import code against a fake
provider and a real ChromaDB index, so a batch run has to end in the same index
state as a realtime one.
"""

import importlib.util
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

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

from chromadb import EmbeddingFunction  # noqa: E402

from zotero_mcp import batch_common, gemini_batch, openai_batch, semantic_search  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient, read_index_layout  # noqa: E402

NAME = "batch_levels"
LONG = " ".join(f"Sentence number {i} of the abstract." for i in range(100))


class NoEmbedding(EmbeddingFunction):
    """A batch run must never embed through the realtime path."""

    def __call__(self, input):
        raise AssertionError(f"realtime embedding of {len(input)} documents")

    def truncate(self, text, max_tokens):
        return text


class Client(ChromaClient):
    def _create_embedding_function(self):
        return NoEmbedding()


class FakeZotero:
    def __init__(self, abstracts):
        self.abstracts = dict(abstracts)
        self.version = 5
        self.changed: set[str] = set()

    def item(self, key):
        return {
            "key": key,
            "version": self.version,
            "data": {
                "key": key,
                "itemType": "journalArticle",
                "title": f"Title {key}",
                "abstractNote": self.abstracts[key],
                "creators": [],
                "dateAdded": "2024-01-01T00:00:00Z",
                "dateModified": "2024-01-01T00:00:00Z",
            },
        }

    def items(self, start=0, limit=100, **kwargs):
        return [] if start else [self.item(k) for k in self.abstracts]

    def item_versions(self, since=None, **kwargs):
        keys = self.changed if since is not None else self.abstracts
        return {k: self.version for k in keys}

    def children(self, *a, **kw):
        return []

    def last_modified_version(self, **kwargs):
        return self.version


class FakeOpenAI:
    def __init__(self):
        self.uploads = {}
        self.outputs = {}
        self.jobs = {}
        self.files = SimpleNamespace(create=self._upload, content=self._content)
        self.batches = SimpleNamespace(create=self._create, retrieve=lambda job_id: self.jobs[job_id])

    def _upload(self, file, purpose):
        file_id = f"file-{len(self.uploads)}"
        self.uploads[file_id] = Path(file.name).read_text(encoding="utf-8")
        return SimpleNamespace(id=file_id)

    def _content(self, file_id):
        return SimpleNamespace(text=self.outputs[file_id])

    def _create(self, input_file_id, **kwargs):
        job_id = f"job-{len(self.jobs)}"
        self.jobs[job_id] = SimpleNamespace(
            id=job_id, status="in_progress", output_file_id=None, error_file_id=None,
            request_counts=None, input_file_id=input_file_id,
        )
        return self.jobs[job_id]

    def ids_in(self, job_id):
        return [json.loads(r)["custom_id"] for r in self.uploads[self.jobs[job_id].input_file_id].splitlines()]

    def finish_all(self, skip=()):
        for job_id, job in self.jobs.items():
            if job.status != "in_progress":
                continue
            rows = [
                {"custom_id": i, "response": {"status_code": 200, "body": {"data": [{"embedding": [1.0, 0.0]}]}}}
                if i not in skip
                else {"custom_id": i, "response": {"status_code": 500, "body": {}}, "error": {"code": "server_error"}}
                for i in self.ids_in(job_id)
            ]
            job.status = "completed"
            job.output_file_id = f"out-{job_id}"
            self.outputs[job.output_file_id] = "\n".join(json.dumps(r) for r in rows)


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    fake = FakeOpenAI()
    monkeypatch.setattr(openai_batch, "create_openai_client", lambda *_a, **_k: fake)
    config_path = tmp_path / "config.json"

    def build(zotero, *, chunking, route=True, batch_extra=None):
        monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: zotero)
        existing = json.loads(config_path.read_text()) if config_path.exists() else {"semantic_search": {}}
        existing["semantic_search"].update(
            include_fulltext=False,
            index_schema_version=semantic_search._INDEX_SCHEMA_VERSION,
            chunking={"enabled": chunking, "chunk_size": 400, "overlap": 50, "max_chunks_per_item": 5},
            openai_batch={"enabled": True, **(batch_extra or {})},
        )
        config_path.write_text(json.dumps(existing))
        client = Client(collection_name=NAME, persist_directory=str(tmp_path / "chroma"))
        client.route_passages = route
        client.embedding_model = "openai"
        client.embedding_config = {"model_name": "text-embedding-3-large", "api_key": "test"}
        return semantic_search.ZoteroSemanticSearch(chroma_client=client, config_path=str(config_path))

    return SimpleNamespace(fake=fake, build=build, config=str(config_path))


def _land(env, search, **finish):
    """Finish every job, and import, until the run has nothing left."""
    stats = None
    for _ in range(10):
        env.fake.finish_all(**finish)
        stats = search._import_batch("openai")
        if not [j for j in env.fake.jobs.values() if j.status == "in_progress"]:
            break
    return stats


def _submitted(env):
    return [i for job_id in env.fake.jobs for i in env.fake.ids_in(job_id)]


def _ids(search, name=NAME):
    return set(search.chroma_client.client.get_collection(name).get(include=[])["ids"])


def _passages(search):
    return _ids(search, f"{NAME}_passages")


def test_chunking_on_submits_an_article_and_its_passages(env):
    search = env.build(FakeZotero({"AAA": LONG}), chunking=True)

    stats = search.update_database()

    assert stats["batch_submitted"] is True
    assert "chunking_ignored" not in stats
    assert _submitted(env) == ["AAA", "AAA#0", "AAA#1", "AAA#2", "AAA#3", "AAA#4"]


def test_import_ends_in_the_same_index_as_a_realtime_run(env):
    search = env.build(FakeZotero({"AAA": LONG}), chunking=True)
    search.update_database()

    _land(env, search)

    assert _ids(search) == {"AAA"}
    assert len(_passages(search)) == 5
    stored = search.chroma_client.client.get_collection(f"{NAME}_passages").get(ids=["AAA#2"])
    assert stored["metadatas"][0]["parent_item_key"] == "AAA"
    assert stored["metadatas"][0]["chunk_index"] == 2
    assert stored["metadatas"][0]["n_chunks"] == 5


def test_chunking_off_submits_only_the_article(env):
    search = env.build(FakeZotero({"AAA": LONG}), chunking=False)

    search.update_database()

    assert _submitted(env) == ["AAA"]


def test_enabling_chunking_submits_only_the_missing_passages(env):
    zotero = FakeZotero({"AAA": LONG})
    search = env.build(zotero, chunking=False)
    search.update_database()
    _land(env, search)
    env.fake.jobs.clear()
    env.fake.uploads.clear()

    search = env.build(zotero, chunking=True)
    search.update_database()

    assert _submitted(env) == [f"AAA#{i}" for i in range(5)]


def test_a_document_that_shrank_loses_its_extra_passages_on_import(env):
    zotero = FakeZotero({"AAA": LONG})
    search = env.build(zotero, chunking=True)
    search.update_database()
    _land(env, search)

    zotero.version = 6
    zotero.changed = {"AAA"}
    zotero.abstracts["AAA"] = "A much shorter abstract now."
    search = env.build(zotero, chunking=True)
    search.update_database()
    _land(env, search)

    assert _ids(search) == {"AAA"}
    assert _passages(search) == {"AAA#0"}


def test_a_passage_split_across_files_is_not_pruned_away(env):
    """An item's passages can land in different files, in any order."""
    search = env.build(FakeZotero({"AAA": LONG}), chunking=True, batch_extra={"batch_max_requests": 2})
    search.update_database()

    _land(env, search)

    assert len(env.fake.jobs) >= 3
    assert len(_passages(search)) == 5


def test_stats_count_items_not_passages(env):
    search = env.build(FakeZotero({"AAA": LONG, "BBB": LONG}), chunking=True)

    stats = search.update_database()

    assert stats["submitted_items"] == 2
    assert stats["submitted_records"] == 12
    assert stats["estimated_added_items"] == 2

    env.fake.finish_all()
    imported = search._import_batch("openai")
    assert imported["added_items"] == 2
    assert imported["imported_items"] == 12


def test_a_passage_that_keeps_failing_sends_its_item_back_to_the_gap_fill(env):
    """Only that item goes back, and importing the run again changes nothing."""
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    search = env.build(zotero, chunking=True)
    search.update_database()

    _land(env, search, skip={"AAA#3"})

    assert len([p for p in _passages(search) if p.startswith("AAA")]) == 4
    assert len([p for p in _passages(search) if p.startswith("BBB")]) == 5
    assert search._find_index_gaps(limit=None, force_rebuild=False) == {"AAA": {"passages"}}

    search._import_batch("openai")

    assert len([p for p in _passages(search) if p.startswith("AAA")]) == 4


def test_filling_articles_with_chunking_off_keeps_the_passages(env):
    """The batch path of the same promise the realtime path makes."""
    zotero = FakeZotero({"AAA": LONG})
    search = env.build(zotero, chunking=True)
    search.update_database()
    _land(env, search)
    search.chroma_client.delete_documents(["AAA"])
    env.fake.jobs.clear()
    env.fake.uploads.clear()

    search = env.build(zotero, chunking=False)
    search.update_database()
    assert _submitted(env) == ["AAA"]
    _land(env, search)

    assert _ids(search) == {"AAA"}
    assert len(_passages(search)) == 5


def test_a_library_change_does_not_strand_an_unfinished_fill(env):
    """A small new run would supersede the fill, whose other files then never go out."""
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    search = env.build(zotero, chunking=False)
    search.update_database()
    _land(env, search)

    search = env.build(zotero, chunking=True)
    search.update_database()
    submitted = len(env.fake.jobs)

    zotero.version = 6
    zotero.changed = {"AAA"}
    stats = search.update_database()

    assert stats["batch_deferred"]
    assert len(env.fake.jobs) == submitted

    _land(env, search)
    assert len(_passages(search)) == 10

    search.update_database()
    assert len(env.fake.jobs) > submitted, "the change goes out once the fill has landed"


def test_no_second_submission_while_a_fill_is_unfinished(env):
    zotero = FakeZotero({"AAA": LONG})
    search = env.build(zotero, chunking=False)
    search.update_database()
    _land(env, search)

    search = env.build(zotero, chunking=True)
    search.update_database()
    submitted = len(env.fake.jobs)
    assert env.fake.ids_in(f"job-{submitted - 1}") == [f"AAA#{i}" for i in range(5)]

    search = env.build(zotero, chunking=True)
    search.update_database()

    assert len(env.fake.jobs) == submitted


def test_a_superseded_run_does_not_write_its_stale_passages(env):
    zotero = FakeZotero({"AAA": LONG})
    search = env.build(zotero, chunking=True)
    search.update_database()
    older_id = "job-0"
    # Past the provider's window, so it no longer holds the next run back.
    older = openai_batch.find_manifest(env.config)
    older["created_at"] = "2020-01-01T00:00:00+00:00"
    for entry in older["batches"]:
        entry["submitted_at"] = "2020-01-01T00:00:00+00:00"
    openai_batch.save_manifest(older)

    zotero.abstracts["AAA"] = "A much shorter abstract now."
    search = env.build(zotero, chunking=True)
    search.update_database()
    env.fake.finish_all()

    newer_id = "job-1"
    search._import_batch("openai", [newer_id])
    search._import_batch("openai", [older_id])

    assert _passages(search) == {"AAA#0"}


def test_unmigrated_index_gets_its_batch_passages_where_the_others_are(env):
    zotero = FakeZotero({"AAA": LONG})
    search = env.build(zotero, chunking=True, route=False)
    search.update_database()
    _land(env, search)

    assert read_index_layout(search.chroma_client.persist_directory, NAME)["unmigrated_passages"] == 5
    assert _ids(search) == {f"AAA#{i}" for i in range(5)}


def test_db_status_no_longer_distinguishes_requested_from_effective(env):
    status = env.build(FakeZotero({"AAA": LONG}), chunking=True).get_database_status()

    assert status["chunking"]["enabled"] is True
    assert "effective" not in status["chunking"]


def test_passage_ids_round_trip_through_the_gemini_format():
    records = [
        {"id": "AAA#0", "document": "first passage", "metadata": {"parent_item_key": "AAA"}},
        {"id": "AAA#1", "document": "second passage", "metadata": {"parent_item_key": "AAA"}},
    ]
    requests = [gemini_batch.build_embedding_request(r, "gemini-embedding-001") for r in records]
    assert [r["key"] for r in requests] == ["AAA#0", "AAA#1"]

    output = "\n".join(
        json.dumps({"key": r["key"], "response": {"embedding": {"values": [0.5, 0.5]}}}) for r in requests
    )
    embeddings, failures = gemini_batch.parse_embedding_output(output, [r["id"] for r in records])

    assert failures == []
    assert set(embeddings) == {"AAA#0", "AAA#1"}


@pytest.mark.parametrize(
    ("module", "per_file"),
    [(openai_batch, 5000), (gemini_batch, 1050)],
)
def test_default_budgets_split_passages_into_files_that_fit(module, per_file):
    records = [{"id": f"K#{i}", "document": "x" * 1500, "metadata": {}} for i in range(per_file * 2 + 10)]

    chunks = module.split_embedding_records(
        records, "any-model", max_tokens=module.ADAPTER.default_max_enqueued_tokens
    )

    assert len(chunks) == 3
    sizes = [len(c[0]) for c in chunks]
    assert max(sizes) <= per_file
    budget = module.ADAPTER.default_max_enqueued_tokens
    for chunk_records, _ in chunks:
        estimated = sum(batch_common.estimate_tokens(r["document"], module.ADAPTER.chars_per_token) for r in chunk_records)
        assert estimated <= budget
