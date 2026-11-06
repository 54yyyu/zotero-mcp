"""Records that failed to embed are written again, without stalling updates (#769).

On a split index a record that keeps failing must not hold the sync version
back (every later run would pay for the same changes again), must not be lost
(an unchanged item never shows up as a library change), and must not be paid
for forever: its item is kept as a retry key, rewritten whatever levels it has,
and given up on after a few runs, with the user told.

Real ChromaDB in a temp directory, a fake Zotero, an embedding function that
records its inputs and fails on a marker. HOME and Path.home point at tmp_path.
"""

import importlib.util
import json
import sys
import time
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

from zotero_mcp import batch_common, openai_batch, semantic_search  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient  # noqa: E402
from zotero_mcp.embeddings.base import EmbeddingAPIUnreachable  # noqa: E402

NAME = "failed_records"
LONG = " ".join(f"Sentence number {i} of the abstract." for i in range(100))
POISON = "POISON"


class FailingEF(EmbeddingFunction):
    """Embeds anything but text holding POISON; can also act unreachable."""

    def __init__(self):
        self.embedded: list[str] = []
        self.unreachable = False

    def __call__(self, input):
        if self.unreachable:
            raise EmbeddingAPIUnreachable("openai", "api.example.invalid", "connection refused")
        if any(POISON in text for text in input):
            raise ValueError("the provider rejected this text")
        self.embedded.extend(input)
        return [[1.0, 0.0] for _ in input]

    def truncate(self, text, max_tokens):
        return text


class FailingClient(ChromaClient):
    def _create_embedding_function(self):
        return FailingEF()

    @property
    def embedded(self):
        return self.embedding_function.embedded


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
        if kwargs.get("itemType") == "attachment":
            return []
        return [] if start else [self.item(k) for k in self.abstracts]

    def item_versions(self, since=None, **kwargs):
        keys = self.changed if since is not None else self.abstracts
        return {k: self.version for k in keys}

    def children(self, *a, **kw):
        return []

    def last_modified_version(self, **kwargs):
        return self.version


@pytest.fixture()
def build(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    config_path = tmp_path / "config.json"

    def _build(zotero, *, chunking=True, split=True):
        monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: zotero)
        existing = json.loads(config_path.read_text()) if config_path.exists() else {"semantic_search": {}}
        existing["semantic_search"].update(
            include_fulltext=False,
            index_schema_version=semantic_search._INDEX_SCHEMA_VERSION,
            chunking={"enabled": chunking, "chunk_size": 400, "overlap": 50, "max_chunks_per_item": 5},
            openai_batch={"enabled": False},
        )
        config_path.write_text(json.dumps(existing))
        client = FailingClient(collection_name=NAME, persist_directory=str(tmp_path / "chroma"))
        client.route_passages = split
        return semantic_search.ZoteroSemanticSearch(chroma_client=client, config_path=str(config_path))

    return _build


def _saved(tmp_path) -> dict:
    return json.loads((tmp_path / "config.json").read_text())["semantic_search"]["update_config"]


def _set_update_config(tmp_path, **values):
    config_path = tmp_path / "config.json"
    data = json.loads(config_path.read_text())
    data["semantic_search"]["update_config"].update(values)
    config_path.write_text(json.dumps(data))


def _poison(zotero, key):
    """Change ``key`` in Zotero so that its new text can never be embedded."""
    zotero.version += 1
    zotero.changed = {key}
    zotero.abstracts[key] = f"{POISON} {LONG}"


# --- the realtime path on a split index -------------------------------------------


def test_a_record_that_fails_moves_the_sync_version_and_keeps_its_item(build, tmp_path):
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero).update_database()
    _poison(zotero, "BBB")

    search = build(zotero)
    stats = search.update_database()

    assert stats.get("error") is None
    assert search._load_last_sync_version() == 6, "holding it back would re-buy every change"
    assert _saved(tmp_path)["retry_item_keys"] == ["BBB"]
    assert _saved(tmp_path)["retry_item_attempts"] == {"BBB": 1}


def test_a_kept_item_is_rewritten_though_it_has_both_levels(build, tmp_path):
    """A changed item whose write failed still has its old records."""
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero).update_database()
    zotero.abstracts["BBB"] = "A new abstract for BBB."
    _set_update_config(tmp_path, retry_item_keys=["BBB"], retry_item_attempts={"BBB": 1})

    search = build(zotero)
    search.update_database()

    embedded = search.chroma_client.embedded
    assert any("A new abstract for BBB." in text for text in embedded)
    assert not any("Title AAA" in text for text in embedded)
    assert search.chroma_client.get_document_metadata("BBB") is not None
    assert _saved(tmp_path).get("retry_item_keys") == []
    assert _saved(tmp_path).get("retry_item_attempts") == {}


def test_a_kept_item_stays_until_its_write_succeeds_then_is_given_up(build, tmp_path):
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero).update_database()
    _poison(zotero, "BBB")
    build(zotero).update_database()

    zotero.changed = set()
    second = build(zotero).update_database()
    assert _saved(tmp_path)["retry_item_keys"] == ["BBB"]
    assert _saved(tmp_path)["retry_item_attempts"] == {"BBB": 2}
    assert not second.get("retry_given_up")

    third = build(zotero).update_database()
    assert _saved(tmp_path)["retry_item_keys"] == []
    assert _saved(tmp_path)["retry_item_attempts"] == {}
    assert _saved(tmp_path)["retry_item_keys_given_up"] == ["BBB"]
    assert "BBB" in third["retry_given_up"]
    assert "3" in third["retry_given_up"]

    search = build(zotero)
    fourth = search.update_database()
    assert search.chroma_client.embedded == [], "given up: no more paid attempts"
    assert not fourth.get("retry_given_up")


def _passage_ids(search):
    return set(search.chroma_client.client.get_collection(f"{NAME}_passages").get(include=[])["ids"])


def test_a_kept_item_rewritten_with_chunking_off_loses_its_stale_passages(build, tmp_path):
    """A rewrite is no level fill: the passages left from before the change must go."""
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero).update_database()
    zotero.abstracts["BBB"] = "A new abstract for BBB."
    _set_update_config(tmp_path, retry_item_keys=["BBB"], retry_item_attempts={"BBB": 1})

    search = build(zotero, chunking=False)
    search.update_database()

    assert not [p for p in _passage_ids(search) if p.startswith("BBB")]
    assert [p for p in _passage_ids(search) if p.startswith("AAA")]


def test_a_given_up_item_missing_a_level_is_not_filled_on_every_run(build, tmp_path):
    """Its first passages got written, its article never did: a gap the cap must cover."""
    zotero = FakeZotero({"AAA": LONG, "CCC": f"{LONG} {POISON}"})
    build(zotero).update_database()
    build(zotero).update_database()
    third = build(zotero).update_database()
    assert _saved(tmp_path)["retry_item_keys_given_up"] == ["CCC"]
    assert third.get("retry_given_up")

    search = build(zotero)
    search.update_database()

    assert search.chroma_client.embedded == []
    assert _saved(tmp_path)["retry_item_keys_given_up"] == ["CCC"]


def test_a_given_up_item_is_written_again_when_it_changes(build, tmp_path):
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero).update_database()
    _set_update_config(tmp_path, retry_item_keys_given_up=["BBB"])
    zotero.version = 6
    zotero.changed = {"BBB"}
    zotero.abstracts["BBB"] = "Fixed in Zotero."

    search = build(zotero)
    search.update_database()

    assert any("Fixed in Zotero." in text for text in search.chroma_client.embedded)
    assert _saved(tmp_path)["retry_item_keys_given_up"] == []


def test_an_unreachable_api_does_not_use_up_a_retry(build, tmp_path):
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero).update_database()
    _set_update_config(tmp_path, retry_item_keys=["BBB"], retry_item_attempts={"BBB": 2})

    search = build(zotero)
    search.chroma_client.embedding_function.unreachable = True
    stats = search.update_database()

    assert stats.get("embedding_api_unreachable")
    assert _saved(tmp_path)["retry_item_keys"] == ["BBB"]
    assert _saved(tmp_path)["retry_item_attempts"] == {"BBB": 2}


def test_an_unmigrated_index_still_holds_the_sync_version(build, tmp_path):
    """No gap fill there before a rebuild; the watermark is how it retries (#610)."""
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero, split=False).update_database()
    _poison(zotero, "BBB")

    search = build(zotero, split=False)
    search.update_database()

    assert search._load_last_sync_version() == 5
    assert not _saved(tmp_path).get("retry_item_keys")


def test_an_unmigrated_index_moves_on_when_only_a_kept_item_fails(build, tmp_path):
    """The retry key already tracks it; holding the watermark would re-buy every change."""
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG, "CCC": LONG})
    build(zotero, split=False).update_database()
    zotero.abstracts["BBB"] = f"{POISON} {LONG}"
    _set_update_config(tmp_path, retry_item_keys=["BBB"], retry_item_attempts={"BBB": 1})
    zotero.version = 6
    zotero.changed = {"CCC"}
    zotero.abstracts["CCC"] = "A new abstract for CCC."

    search = build(zotero, split=False)
    search.update_database()

    assert search._load_last_sync_version() == 6
    assert _saved(tmp_path)["retry_item_keys"] == ["BBB"]
    zotero.changed = set()
    search = build(zotero, split=False)
    search.update_database()
    assert not any("A new abstract for CCC." in text for text in search.chroma_client.embedded)


# --- add_failed_ids ---------------------------------------------------------------


def test_add_failed_ids_keeps_order_and_drops_repeats():
    manifest = {"failed_ids": ["B", "A"]}

    batch_common.add_failed_ids(manifest, ["A", "C", "C", "D"])

    assert manifest["failed_ids"] == ["B", "A", "C", "D"]


def test_add_failed_ids_is_linear():
    """A large run gives up whole files of ids; a quadratic check takes minutes."""
    # Quadratic, this takes minutes and hits the test timeout; linear, milliseconds.
    # The bound leaves a slow runner plenty of room.
    manifest = {"failed_ids": [f"old{i}" for i in range(100000)]}
    started = time.monotonic()

    batch_common.add_failed_ids(manifest, [f"new{i}" for i in range(100000)])

    assert time.monotonic() - started < 10
    assert len(manifest["failed_ids"]) == 200000


# --- the Batch API path ------------------------------------------------------------


class FakeOpenAI:
    """Batches stay in progress until a test finishes them."""

    def __init__(self):
        self.uploads: dict[str, str] = {}
        self.outputs: dict[str, str] = {}
        self.errors: dict[str, str] = {}
        self.jobs: dict[str, SimpleNamespace] = {}
        self.files = SimpleNamespace(create=self._upload, content=self._content)
        self.batches = SimpleNamespace(create=self._create, retrieve=lambda job_id: self.jobs[job_id])

    def _upload(self, file, purpose):
        file_id = f"file-{len(self.uploads)}"
        self.uploads[file_id] = Path(file.name).read_text(encoding="utf-8")
        return SimpleNamespace(id=file_id)

    def _content(self, file_id):
        return SimpleNamespace(text=self.outputs[file_id] if file_id in self.outputs else self.errors[file_id])

    def _create(self, input_file_id, **kwargs):
        job_id = f"job-{len(self.jobs)}"
        self.jobs[job_id] = SimpleNamespace(
            id=job_id, status="in_progress", output_file_id=None, error_file_id=None,
            request_counts=None, input_file_id=input_file_id,
        )
        return self.jobs[job_id]

    def ids_in(self, job_id):
        rows = self.uploads[self.jobs[job_id].input_file_id].splitlines()
        return [json.loads(row)["custom_id"] for row in rows]

    def finish_all(self, *, fail=()):
        """Complete every job in progress, with an error row for each id in ``fail``."""
        for job_id, job in self.jobs.items():
            if job.status != "in_progress":
                continue
            ids = self.ids_in(job_id)
            rows = [
                {"custom_id": i, "response": {"status_code": 200, "body": {"data": [{"embedding": [1.0, 0.0]}]}}}
                for i in ids
                if i not in fail
            ]
            job.status = "completed"
            job.output_file_id = f"out-{job_id}"
            self.outputs[job.output_file_id] = "\n".join(json.dumps(row) for row in rows)
            failed = [i for i in ids if i in fail]
            if failed:
                job.error_file_id = f"err-{job_id}"
                self.errors[job.error_file_id] = "\n".join(
                    json.dumps({"custom_id": i, "error": {"code": "invalid_input"}}) for i in failed
                )


@pytest.fixture()
def batch(build, monkeypatch):
    fake = FakeOpenAI()
    monkeypatch.setattr(openai_batch, "create_openai_client", lambda *_a, **_k: fake)

    def _batch(zotero, **kwargs):
        search = build(zotero, **kwargs)
        search.chroma_client.embedding_model = "openai"
        search.chroma_client.embedding_config = {
            "model_name": "text-embedding-3-small",
            "api_key": "test",
            "base_url": "http://127.0.0.1:9",
        }
        return search

    return SimpleNamespace(fake=fake, search=_batch)


def _land(batch, zotero, *, fail=(), **kwargs):
    """Finish every job, retried files included, and import what came back."""
    stats = {}
    for _ in range(6):
        batch.fake.finish_all(fail=fail)
        stats = batch.search(zotero, **kwargs)._import_batch("openai")
        if not any(job.status == "in_progress" for job in batch.fake.jobs.values()):
            break
    return stats


def test_a_batch_run_rewrites_a_kept_item_and_drops_it_once_landed(build, batch, tmp_path):
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero, chunking=False).update_database()
    _set_update_config(tmp_path, retry_item_keys=["BBB"], retry_item_attempts={"BBB": 1})

    stats = batch.search(zotero, chunking=False).update_database(use_batch=True)

    assert stats.get("batch_submitted")
    assert batch.fake.ids_in("job-0") == ["BBB"]
    assert _saved(tmp_path)["retry_item_keys"] == ["BBB"], "not dropped before the write lands"

    _land(batch, zotero, chunking=False)

    assert _saved(tmp_path)["retry_item_keys"] == []
    assert _saved(tmp_path)["retry_item_attempts"] == {}


def test_a_batch_record_that_keeps_failing_is_counted_across_runs(build, batch, tmp_path):
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero, chunking=False).update_database()
    _set_update_config(tmp_path, retry_item_keys=["BBB"], retry_item_attempts={"BBB": 2})

    batch.search(zotero, chunking=False).update_database(use_batch=True)
    stats = _land(batch, zotero, fail={"BBB"}, chunking=False)

    assert _saved(tmp_path)["retry_item_keys"] == []
    assert _saved(tmp_path)["retry_item_keys_given_up"] == ["BBB"]
    errors = " ".join(str(error) for error in stats["errors"])
    assert "BBB" in errors
    assert "embeds them again" not in errors


def test_an_update_that_collects_a_run_says_what_it_gave_up(build, batch, tmp_path):
    """A background collect lands most runs, so its summary must carry the news."""
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero, chunking=False).update_database()
    _set_update_config(tmp_path, retry_item_keys=["BBB"], retry_item_attempts={"BBB": 2})
    batch.search(zotero, chunking=False).update_database(use_batch=True)

    stats = {}
    for _ in range(6):
        batch.fake.finish_all(fail={"BBB"})
        stats = batch.search(zotero, chunking=False).update_database()
        if not any(job.status == "in_progress" for job in batch.fake.jobs.values()):
            break

    assert _saved(tmp_path)["retry_item_keys_given_up"] == ["BBB"]
    errors = " ".join(stats["batch_collect"]["errors"])
    assert "Gave up" in errors and "BBB" in errors


def test_the_import_says_an_unmigrated_index_keeps_its_sync_version(build, batch, tmp_path):
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero, chunking=False, split=False).update_database()
    zotero.version = 6
    zotero.changed = {"BBB"}

    batch.search(zotero, chunking=False, split=False).update_database(use_batch=True)
    stats = _land(batch, zotero, fail={"BBB"}, chunking=False, split=False)

    errors = " ".join(str(error) for error in stats["errors"])
    assert "sync version stays put" in errors
    assert "embeds them again" not in errors
    assert build(zotero, chunking=False, split=False)._load_last_sync_version() == 5


def test_an_unmigrated_batch_run_moves_on_when_only_a_kept_item_fails(build, batch, tmp_path):
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG, "CCC": LONG})
    build(zotero, chunking=False, split=False).update_database()
    _set_update_config(tmp_path, retry_item_keys=["BBB"], retry_item_attempts={"BBB": 1})
    zotero.version = 6
    zotero.changed = {"CCC"}

    batch.search(zotero, chunking=False, split=False).update_database(use_batch=True)
    stats = _land(batch, zotero, fail={"BBB"}, chunking=False, split=False)

    assert build(zotero, chunking=False, split=False)._load_last_sync_version() == 6
    assert _saved(tmp_path)["retry_item_keys"] == ["BBB"]
    errors = " ".join(str(error) for error in stats["errors"])
    assert "sync version stays put" not in errors


def test_a_change_run_does_not_cover_a_run_rewriting_kept_items():
    """The newer run doesn't carry the rewrites, so superseding would pay for them twice."""
    older = {"run_id": "a", "group_id": 0, "since_version": 5, "retry_keys": ["BBB"]}
    newer = {"run_id": "b", "group_id": 0, "since_version": 5}

    assert not batch_common.run_covers(newer, older)
    assert batch_common.run_covers({**newer, "retry_keys": ["BBB", "CCC"]}, older)
    assert batch_common.run_covers({**newer, "force_full_rebuild": True}, older)


def test_a_superseded_run_settles_the_rewrites_it_landed(build, batch, tmp_path):
    """Its records are in the index even though it doesn't move the sync version."""
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG, "CCC": LONG})
    build(zotero, chunking=False).update_database()
    zotero.abstracts["CCC"] = f"{POISON} {LONG}"
    _set_update_config(
        tmp_path, retry_item_keys=["BBB", "CCC"], retry_item_attempts={"BBB": 1, "CCC": 1}
    )
    batch.search(zotero, chunking=False).update_database(use_batch=True)

    batch.fake.finish_all(fail={"CCC"})
    search = batch.search(zotero, chunking=False)
    search._superseded_by = lambda provider, manifest: "a-newer-run"
    stats = search._import_batch("openai")

    assert stats["imported_items"] == 1
    assert _saved(tmp_path)["retry_item_keys"] == ["CCC"], "not written, so still kept"
    errors = " ".join(str(error) for error in stats["errors"])
    assert "rewrites the items kept for a retry" in errors


def test_a_kept_item_without_text_is_dropped_by_a_batch_run(build, batch, tmp_path):
    """Nothing to embed is not a failure, and keeping it would fetch it on every run."""
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG, "EMPTY": LONG})
    build(zotero, chunking=False).update_database()
    zotero.abstracts["BBB"] = "A new abstract for BBB."
    _set_update_config(
        tmp_path, retry_item_keys=["BBB", "EMPTY"], retry_item_attempts={"BBB": 1, "EMPTY": 1}
    )
    search = batch.search(zotero, chunking=False)
    original = search._build_item_records

    def build_records(item, **kwargs):
        return [] if item.get("key") == "EMPTY" else original(item, **kwargs)

    search._build_item_records = build_records
    search.update_database(use_batch=True)
    _land(batch, zotero, chunking=False)

    assert _saved(tmp_path)["retry_item_keys"] == []
