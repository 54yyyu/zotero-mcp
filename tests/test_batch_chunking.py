"""Passage chunking on the Batch API path, OpenAI and Gemini (#416).

Runs update_database and the real manifest, split and import code against a fake
provider and a real ChromaDB index, so a batch run has to end in the same index
state as a realtime one.
"""

import hashlib
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
    """Each item ``KEY`` has one PDF ``PKEY``; ``abstracts`` holds the PDFs' text.

    Passages come from files only (#826). A key in ``changed`` was edited:
    its dateModified moves on.
    """

    def __init__(self, abstracts):
        self.abstracts = dict(abstracts)
        self.version = 5
        self.changed: set[str] = set()

    def item(self, key):
        if key.startswith("P") and key[1:] in self.abstracts:
            return self._attachment(key[1:])
        return {
            "key": key,
            "version": self.version,
            "data": {
                "key": key,
                "itemType": "journalArticle",
                "title": f"Title {key}",
                "abstractNote": "An abstract.",
                "creators": [],
                "dateAdded": "2024-01-01T00:00:00Z",
                "dateModified": (
                    f"2024-02-{self.version:02d}T00:00:00Z" if key in self.changed else "2024-01-01T00:00:00Z"
                ),
            },
        }

    def _attachment(self, parent):
        return {
            "key": f"P{parent}",
            "version": self.version,
            "data": {
                "key": f"P{parent}",
                "itemType": "attachment",
                "parentItem": parent,
                "contentType": "application/pdf",
                "md5": hashlib.md5(self.abstracts[parent].encode()).hexdigest(),
                "mtime": 1,
                "dateModified": "2024-01-01T00:00:00Z",
            },
        }

    def items(self, start=0, limit=100, **kwargs):
        return [] if start else [self.item(k) for k in self.abstracts]

    def item_versions(self, since=None, **kwargs):
        if since is not None:
            return {k: self.version for k in self.changed}
        return {k: self.version for key in self.abstracts for k in (key, f"P{key}")}

    def children(self, key, *a, **kw):
        return [self._attachment(key)] if key in self.abstracts else []

    def fulltext_item(self, key):
        return {"content": self.abstracts[key[1:]]}

    def last_modified_version(self, **kwargs):
        return self.version


def _records(*keys):
    """Each item's metadata record, its PDF's record, then the PDF's five passages."""
    return [i for key in keys for i in (key, f"P{key}", *(f"P{key}#{n}" for n in range(5)))]


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
            include_fulltext=True,
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
    assert _submitted(env) == _records("AAA")


def test_import_ends_in_the_same_index_as_a_realtime_run(env):
    search = env.build(FakeZotero({"AAA": LONG}), chunking=True)
    search.update_database()

    _land(env, search)

    assert _ids(search) == {"AAA", "PAAA"}
    assert len(_passages(search)) == 5
    stored = search.chroma_client.client.get_collection(f"{NAME}_passages").get(ids=["PAAA#2"])
    # Keyed by the PDF itself, with nothing of its parent (#826).
    assert stored["metadatas"][0]["item_key"] == "PAAA"
    assert "parent_item_key" not in stored["metadatas"][0]
    assert stored["metadatas"][0]["chunk_index"] == 2
    assert stored["metadatas"][0]["n_chunks"] == 5
    article = search.chroma_client.collection.get(ids=["PAAA"], include=["metadatas"])["metadatas"][0]
    assert article["file_stamp"].startswith("api:")
    assert "title" not in article


def test_chunking_off_submits_only_the_article(env):
    search = env.build(FakeZotero({"AAA": LONG}), chunking=False)

    search.update_database()

    assert _submitted(env) == ["AAA", "PAAA"]


def test_enabling_chunking_submits_only_the_missing_passages(env):
    zotero = FakeZotero({"AAA": LONG})
    search = env.build(zotero, chunking=False)
    search.update_database()
    _land(env, search)
    env.fake.jobs.clear()
    env.fake.uploads.clear()

    search = env.build(zotero, chunking=True)
    search.update_database()

    assert _submitted(env) == [f"PAAA#{i}" for i in range(5)]


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

    assert _ids(search) == {"AAA", "PAAA"}
    assert _passages(search) == {"PAAA#0"}


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

    # Counted by record key: each item's metadata record and its PDF's.
    assert stats["submitted_items"] == 4
    assert stats["submitted_records"] == 14
    assert stats["estimated_added_items"] == 4

    env.fake.finish_all()
    imported = search._import_batch("openai")
    assert imported["added_items"] == 4
    assert imported["imported_items"] == 14


def test_a_passage_that_keeps_failing_sends_its_item_back_to_the_gap_fill(env):
    """Only that item goes back, and importing the run again changes nothing."""
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    search = env.build(zotero, chunking=True)
    search.update_database()

    _land(env, search, skip={"PAAA#3"})

    assert len([p for p in _passages(search) if p.startswith("PAAA")]) == 4
    assert len([p for p in _passages(search) if p.startswith("PBBB")]) == 5
    assert search._find_index_gaps(limit=None, force_rebuild=False) == {"PAAA": {"passages"}}

    search._import_batch("openai")

    assert len([p for p in _passages(search) if p.startswith("PAAA")]) == 4


def test_filling_articles_with_chunking_off_keeps_the_passages(env):
    """The batch path of the same promise the realtime path makes."""
    zotero = FakeZotero({"AAA": LONG})
    search = env.build(zotero, chunking=True)
    search.update_database()
    _land(env, search)
    search.chroma_client.delete_documents(["PAAA"])
    env.fake.jobs.clear()
    env.fake.uploads.clear()

    search = env.build(zotero, chunking=False)
    search.update_database()
    assert _submitted(env) == ["PAAA"]
    _land(env, search)

    assert _ids(search) == {"AAA", "PAAA"}
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

    assert stats["skipped_reason"] == "batch_run_unfinished"
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
    assert env.fake.ids_in(f"job-{submitted - 1}") == [f"PAAA#{i}" for i in range(5)]

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

    assert _passages(search) == {"PAAA#0"}


def test_unmigrated_index_gets_its_batch_passages_where_the_others_are(env):
    zotero = FakeZotero({"AAA": LONG})
    search = env.build(zotero, chunking=True, route=False)
    search.update_database()
    _land(env, search)

    assert read_index_layout(search.chroma_client.persist_directory, NAME)["unmigrated_passages"] == 5
    # Every record in the one collection, article-level records too (#826).
    assert _ids(search) == set(_records("AAA"))


def test_a_changed_legacy_item_lands_in_the_item_keyed_shape(env, monkeypatch, tmp_path):
    """Its metadata record replaces the legacy one, and its legacy passages go (#826)."""
    _isolate_home(monkeypatch, tmp_path)
    zotero = FakeZotero({"AAA": LONG})
    search = env.build(zotero, chunking=True)
    legacy = {
        "item_key": "AAA", "title": "Title AAA", "group_id": search._pinned_group_id(),
        "date_modified": "2023-01-01T00:00:00Z", "has_fulltext": True,
    }
    ids = ["AAA", *(f"AAA#{n}" for n in range(3))]
    metas = [legacy, *({**legacy, "parent_item_key": "AAA", "chunk_index": n, "n_chunks": 3} for n in range(3))]
    search.chroma_client.upsert_embeddings(
        documents=["old"] * 4, metadatas=metas, ids=ids, embeddings=[[0.0, 1.0]] * 4
    )

    stats = search.update_database()
    # The legacy passages wait for the files' records; the import settles them.
    manifest = json.loads(Path(stats["batch_manifest"]).read_text())
    assert list(manifest["legacy_settle"]) == ["AAA"]
    _land(env, search)

    assert _ids(search) == {"AAA", "PAAA"}
    assert _passages(search) == {f"PAAA#{n}" for n in range(5)}
    stored = search.chroma_client.collection.get(ids=["AAA"], include=["metadatas"])["metadatas"][0]
    assert set(stored) == {"item_key", "group_id", "date_modified"}


def _legacy_item(search):
    legacy = {
        "item_key": "AAA", "title": "Title AAA", "group_id": search._pinned_group_id(),
        "date_modified": "2023-01-01T00:00:00Z", "has_fulltext": True,
    }
    ids = ["AAA", *(f"AAA#{n}" for n in range(3))]
    metas = [legacy, *({**legacy, "parent_item_key": "AAA", "chunk_index": n, "n_chunks": 3} for n in range(3))]
    search.chroma_client.upsert_embeddings(
        documents=["old"] * 4, metadatas=metas, ids=ids, embeddings=[[0.0, 1.0]] * 4
    )


def test_a_failed_attachment_record_keeps_a_legacy_items_passages_through_the_import(env, monkeypatch, tmp_path):
    """The passages are dropped only once every file of the item has landed (fix round 2)."""
    _isolate_home(monkeypatch, tmp_path)
    search = env.build(FakeZotero({"AAA": LONG}), chunking=True)
    _legacy_item(search)

    search.update_database()
    _land(env, search, skip={"PAAA"})

    stored = _ids(search) | _passages(search)
    assert "PAAA" not in stored
    assert {f"AAA#{n}" for n in range(3)} <= stored, "the only copy of the text stays"


def _import_only(env, search, finish_jobs, **finish):
    """Finish and import only the first ``finish_jobs`` jobs; the rest stay in progress."""
    jobs = list(env.fake.jobs.values())
    later = jobs[finish_jobs:]
    for job in later:
        job.status = "validating"  # still running: finish_all leaves it
    env.fake.finish_all(**finish)
    for job in later:
        job.status = "in_progress"
    search._import_batch("openai")


def _all_stored(search):
    return _ids(search) | _passages(search)


def test_legacy_passages_stay_until_every_passage_of_the_file_has_landed(env, monkeypatch, tmp_path):
    """The attachment's passages span several files; two of five landed is not replaced (fix round 2b)."""
    _isolate_home(monkeypatch, tmp_path)
    search = env.build(FakeZotero({"AAA": LONG}), chunking=True, batch_extra={"batch_max_requests": 2})
    _legacy_item(search)
    search.update_database()
    assert len(env.fake.jobs) >= 3

    # Jobs hold [AAA, PAAA], [PAAA#0, PAAA#1], then the rest of the passages.
    _import_only(env, search, 2)

    assert {"PAAA", "PAAA#0", "PAAA#1"} <= _all_stored(search)
    assert "PAAA#2" not in _all_stored(search)
    assert {f"AAA#{n}" for n in range(3)} <= _all_stored(search), "the new set is incomplete"

    _land(env, search)

    assert _ids(search) == {"AAA", "PAAA"}
    assert _passages(search) == {f"PAAA#{n}" for n in range(5)}


def test_a_failed_passage_row_keeps_the_legacy_passages(env, monkeypatch, tmp_path):
    _isolate_home(monkeypatch, tmp_path)
    search = env.build(FakeZotero({"AAA": LONG}), chunking=True)
    _legacy_item(search)
    search.update_database()

    env.fake.finish_all(skip={"PAAA#3"})
    search._import_batch("openai")

    assert "PAAA#3" not in _all_stored(search)
    assert {f"AAA#{n}" for n in range(3)} <= _all_stored(search), "one new passage is missing"


def test_a_passage_set_missing_one_is_refilled_by_the_next_run(env):
    """Only that file's passages are submitted again (fix round 2b)."""
    zotero = FakeZotero({"AAA": LONG})
    search = env.build(zotero, chunking=True)
    search.update_database()
    env.fake.finish_all(skip={"PAAA#3"})
    search._import_batch("openai")
    assert "PAAA#3" not in _passages(search)
    env.fake.jobs.clear()
    env.fake.uploads.clear()

    search.update_database()

    assert _submitted(env) and all(i.startswith("PAAA#") for i in _submitted(env)), _submitted(env)
    _land(env, search)
    assert _passages(search) == {f"PAAA#{n}" for n in range(5)}


def test_api_planning_judges_an_item_by_its_bare_record_whatever_the_storage_order(env, monkeypatch, tmp_path):
    """Legacy passages stored before the item's new bare record must not make it a legacy item (fix round 2b)."""
    _isolate_home(monkeypatch, tmp_path)
    zotero = FakeZotero({"AAA": LONG})
    search = env.build(zotero, chunking=True, route=False)
    legacy = {
        "item_key": "AAA", "title": "Title AAA", "group_id": search._pinned_group_id(),
        "date_modified": "2023-01-01T00:00:00Z", "has_fulltext": True,
    }
    passages = [{**legacy, "parent_item_key": "AAA", "chunk_index": n, "n_chunks": 3} for n in range(3)]
    search.chroma_client.upsert_embeddings(
        documents=["old"] * 3, metadatas=passages, ids=[f"AAA#{n}" for n in range(3)], embeddings=[[0.0, 1.0]] * 3
    )
    # Inserted last: the item's own record, in the new shape and current.
    search.chroma_client.upsert_embeddings(
        documents=["Title AAA"],
        metadatas=[{
            "item_key": "AAA", "group_id": search._pinned_group_id(), "date_modified": "2024-01-01T00:00:00Z",
        }],
        ids=["AAA"], embeddings=[[1.0, 0.0]],
    )

    units = search._plan_units([zotero.item("AAA")], force_rebuild=False)

    assert not [u for u in units if u.get("legacy_rewrite")], "the bare record is new-shape and current"
    assert [u["key"] for u in units if u["key"] == "AAA"] == []


def test_an_explicit_batch_run_counts_as_covering_standalone_files(env):
    """Else automatic runs would hold standalone files for good and repeat the hint (fix round 2)."""
    search = env.build(FakeZotero({"AAA": LONG}), chunking=True)

    search.update_database()

    saved = json.loads(Path(env.config).read_text())["semantic_search"]["update_config"]
    assert saved["standalone_files_indexed"] is True


def test_an_automatic_batch_run_does_not(env):
    search = env.build(FakeZotero({"AAA": LONG}), chunking=True)

    search.update_database(automatic=True)

    saved = json.loads(Path(env.config).read_text())["semantic_search"].get("update_config", {})
    assert not saved.get("standalone_files_indexed")


def test_db_status_no_longer_distinguishes_requested_from_effective(env):
    status = env.build(FakeZotero({"AAA": LONG}), chunking=True).get_database_status()

    assert status["chunking"]["enabled"] is True
    assert "effective" not in status["chunking"]


def test_passage_ids_round_trip_through_the_gemini_format():
    records = [
        {"id": "AAA#0", "document": "first passage", "metadata": {"item_key": "AAA"}},
        {"id": "AAA#1", "document": "second passage", "metadata": {"item_key": "AAA"}},
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


def _isolate_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)


def test_article_records_are_submitted_before_any_passage(env, monkeypatch, tmp_path):
    """Article-level search is complete once the first files land."""
    _isolate_home(monkeypatch, tmp_path)
    keys = ["AAA", "BBB", "CCC"]
    search = env.build(FakeZotero({k: LONG for k in keys}), chunking=True, batch_extra={"batch_max_requests": 2})

    search.update_database()

    submitted = _submitted(env)
    assert submitted[:6] == ["AAA", "PAAA", "BBB", "PBBB", "CCC", "PCCC"]
    assert all("#" in i for i in submitted[6:])
    assert len(submitted) == 21


def test_items_are_counted_once_when_articles_and_passages_land_apart(env, monkeypatch, tmp_path):
    _isolate_home(monkeypatch, tmp_path)
    keys = ["AAA", "BBB", "CCC"]
    search = env.build(FakeZotero({k: LONG for k in keys}), chunking=True, batch_extra={"batch_max_requests": 2})
    search.update_database()

    added = updated = 0
    for _ in range(10):
        env.fake.finish_all()
        stats = search._import_batch("openai")
        added += stats["added_items"]
        updated += stats["updated_items"]
        if not [j for j in env.fake.jobs.values() if j.status == "in_progress"]:
            break

    assert (added, updated) == (6, 0)
    assert _ids(search) == {k for key in keys for k in (key, f"P{key}")}
    assert len(_passages(search)) == 15


CHANGED = " ".join(f"Rewritten sentence {i} of the new abstract." for i in range(100))


def _finish_where(env, wanted):
    """Finish only the jobs whose ids all satisfy ``wanted``."""
    for job_id, job in env.fake.jobs.items():
        ids = env.fake.ids_in(job_id)
        if job.status != "in_progress" or not all(wanted(i) for i in ids):
            continue
        rows = [
            {"custom_id": i, "response": {"status_code": 200, "body": {"data": [{"embedding": [1.0, 0.0]}]}}}
            for i in ids
        ]
        job.status = "completed"
        job.output_file_id = f"out-{job_id}"
        env.fake.outputs[job.output_file_id] = "\n".join(json.dumps(r) for r in rows)


def _passage_texts(search, key):
    return search.chroma_client.client.get_collection(f"{NAME}_passages").get(
        where={"item_key": key}, include=["documents"]
    )["documents"]


def _changed_run(env, monkeypatch, tmp_path):
    _isolate_home(monkeypatch, tmp_path)
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG, "CCC": LONG})
    search = env.build(zotero, chunking=True)
    search.update_database()
    _land(env, search)
    zotero.version = 6
    zotero.changed = {"AAA", "BBB"}
    zotero.abstracts["AAA"] = zotero.abstracts["BBB"] = CHANGED
    search = env.build(zotero, chunking=True, batch_extra={"batch_max_requests": 2})
    search.update_database()
    return search


def test_an_article_landing_drops_its_items_stale_passages(env, monkeypatch, tmp_path):
    """Should the passage files never land, the gap fill finds the item."""
    search = _changed_run(env, monkeypatch, tmp_path)

    _finish_where(env, lambda i: "#" not in i)
    search._import_batch("openai")

    assert _passage_texts(search, "PAAA") == []
    assert len(_passage_texts(search, "PCCC")) == 5, "an unchanged file keeps its passages"
    assert "PAAA" in search._find_index_gaps(limit=None, force_rebuild=False)


def test_an_article_landing_after_its_passages_keeps_them(env, monkeypatch, tmp_path):
    search = _changed_run(env, monkeypatch, tmp_path)

    _finish_where(env, lambda i: "#" in i)
    search._import_batch("openai")
    _finish_where(env, lambda i: True)
    search._import_batch("openai")

    texts = _passage_texts(search, "PAAA")
    assert len(texts) == 5
    assert all("Sentence number" not in t for t in texts)


def test_an_article_landing_keeps_the_cached_text_for_its_passages(env, monkeypatch, tmp_path):
    search = _changed_run(env, monkeypatch, tmp_path)
    evicted: list[str] = []
    monkeypatch.setattr(
        semantic_search.fulltext_cache, "evict_many",
        lambda keys, config_path=None: evicted.extend(keys) or 0,
    )

    _finish_where(env, lambda i: "#" not in i)
    search._import_batch("openai")
    assert "PAAA" not in evicted

    _land(env, search)
    assert "PAAA" in evicted


def test_a_superseded_runs_article_landing_keeps_the_newer_passages(env, monkeypatch, tmp_path):
    """A newer run wrote these passages; the older run's own are ignored."""
    search = _changed_run(env, monkeypatch, tmp_path)
    article_jobs = [
        job_id for job_id in env.fake.jobs
        if all("#" not in i for i in env.fake.ids_in(job_id))
    ]
    older = openai_batch.find_manifest(env.config)
    older["created_at"] = "2020-01-01T00:00:00+00:00"
    for entry in older["batches"]:
        entry["submitted_at"] = "2020-01-01T00:00:00+00:00"
    openai_batch.save_manifest(older)

    zotero = FakeZotero({"AAA": "Newer text for AAA."})
    zotero.version = 7
    zotero.changed = {"AAA"}
    search = env.build(zotero, chunking=True)
    search.update_database()
    newer = [f"Newer passage {n} of AAA." for n in range(5)]
    search.chroma_client.upsert_embeddings(
        documents=newer,
        metadatas=[{"item_key": "PAAA", "chunk_index": n, "n_chunks": 5} for n in range(5)],
        ids=[f"PAAA#{n}" for n in range(5)],
        embeddings=[[1.0, 0.0]] * 5,
    )
    _finish_where(env, lambda i: "#" not in i)

    stats = search._import_batch("openai", batch_ids=article_jobs[:1])

    assert stats["imported_items"] >= 1
    assert sorted(_passage_texts(search, "PAAA")) == sorted(newer)


@pytest.mark.parametrize("failure", ["read", "delete"])
def test_stale_passages_an_article_landing_could_not_drop_are_noted_and_warned(
    env, monkeypatch, tmp_path, caplog, failure
):
    import logging

    search = _changed_run(env, monkeypatch, tmp_path)
    if failure == "read":
        def broken(key):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(search.chroma_client.__class__, "passage_documents", lambda self, key: broken(key))
    else:
        monkeypatch.setattr(search.chroma_client.__class__, "delete_documents", lambda self, ids: None)
    _finish_where(env, lambda i: "#" not in i)

    noted: list[str] = []
    original_note = search._note_undropped_passages

    def note(item_key):
        original_note(item_key)
        noted.append(item_key)

    monkeypatch.setattr(search, "_note_undropped_passages", note)

    with caplog.at_level(logging.WARNING, logger="zotero_mcp.semantic_search"):
        search._import_batch("openai")

    assert "PAAA" in noted
    assert any("PAAA" in r.getMessage() for r in caplog.records if r.name == "zotero_mcp.semantic_search")
