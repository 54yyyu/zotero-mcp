"""Batch runs are collected before anything else, and never paid for twice.

These run ``update_database`` end to end against a real temporary ChromaDB
index; only the provider SDK client and the Zotero client are fakes. No
embedding function runs: vectors arrive precomputed from the fake provider.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")

from zotero_mcp import batch_common, gemini_batch, openai_batch, semantic_search  # noqa: E402
from zotero_mcp import client as zclient  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient  # noqa: E402
from zotero_mcp.cli import main as cli_main  # noqa: E402
from zotero_mcp.tools import search as search_tools  # noqa: E402

GROUP_ID = 6015547


class FakeProvider:
    """OpenAI SDK double. Batches stay in progress until a test finishes them."""

    def __init__(self):
        self.created = []  # batch ids, in submission order
        self.network_calls = 0
        self.statuses = {}
        self.fail_status_with = None
        self.fail_download_with = None
        self.expired_with_output = False
        self._file_ids = {}  # uploaded file id -> custom ids
        self._batch_ids = {}  # batch id -> custom ids
        provider = self

        class Files:
            def create(self, file, purpose):
                provider.network_calls += 1
                file_id = f"file-{len(provider._file_ids) + 1}"
                provider._file_ids[file_id] = [
                    json.loads(line)["custom_id"] for line in file.read().splitlines()
                ]
                return SimpleNamespace(id=file_id)

            def content(self, file_id):
                provider.network_calls += 1
                if provider.fail_download_with:
                    raise provider.fail_download_with
                batch_id = file_id.removeprefix("out-")
                return SimpleNamespace(text="\n".join(
                    json.dumps({
                        "custom_id": custom_id,
                        "response": {"status_code": 200, "body": {"data": [{"embedding": [0.1, 0.2, 0.3]}]}},
                    })
                    for custom_id in provider._batch_ids[batch_id]
                ))

        class Batches:
            def create(self, input_file_id, **kwargs):
                provider.network_calls += 1
                batch_id = f"batch-{len(provider.created) + 1}"
                provider.created.append(batch_id)
                provider._batch_ids[batch_id] = provider._file_ids[input_file_id]
                provider.statuses[batch_id] = "in_progress"
                return _batch_object(batch_id, "in_progress")

            def retrieve(self, batch_id):
                provider.network_calls += 1
                if provider.fail_status_with:
                    raise provider.fail_status_with
                return _batch_object(batch_id, provider.statuses[batch_id], provider.expired_with_output)

        self.files = Files()
        self.batches = Batches()

    def finish_all(self, status="completed"):
        for batch_id in self.statuses:
            self.statuses[batch_id] = status


def _batch_object(batch_id, status, has_output=False):
    return SimpleNamespace(
        id=batch_id,
        status=status,
        output_file_id=f"out-{batch_id}" if status == "completed" or has_output else None,
        error_file_id=None,
        request_counts={"total": 1, "completed": 0, "failed": 0},
    )


class FakeZotero:
    library_type = "user"
    library_id = "0"

    def __init__(self, keys, version):
        self.library_version = version
        self.versions = {key: version for key in keys}

    def add(self, key, version):
        self.versions[key] = version
        self.library_version = version

    def items(self, start=0, limit=100, **kwargs):
        return [_paper(key) for key in list(self.versions)[start:start + limit]]

    def item(self, key):
        return _paper(key)

    def children(self, key, **kwargs):
        return []

    def item_versions(self, since=None, **kwargs):
        if since is None:
            return dict(self.versions)
        return {key: version for key, version in self.versions.items() if version > since}

    def last_modified_version(self, **kwargs):
        return self.library_version


class GroupZotero(FakeZotero):
    library_type = "group"
    library_id = str(GROUP_ID)


def _paper(key):
    return {
        "key": key,
        "version": 1,
        "data": {
            "key": key,
            "itemType": "journalArticle",
            "title": f"Paper {key}",
            "abstractNote": f"abstract of {key}",
            "creators": [{"creatorType": "author", "firstName": "A", "lastName": "Author"}],
            "dateAdded": "2024-01-01T00:00:00Z",
            "dateModified": "2024-01-01T00:00:00Z",
        },
    }


class Index(ChromaClient):
    """The real client; only the tokenizer download is cut out."""

    def truncate_text(self, text, max_tokens=None):
        return text


@pytest.fixture
def env(tmp_path, monkeypatch):
    for name in ("ZOTERO_LOCAL", "ZOTERO_LIBRARY_ID", "ZOTERO_LIBRARY_TYPE", "ZOTERO_MCP_FORCE_UPDATE"):
        monkeypatch.delenv(name, raising=False)
    zclient.clear_active_library()
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    monkeypatch.setattr(search_tools._utils, "is_local_mode", lambda: False)

    provider = FakeProvider()
    monkeypatch.setattr(openai_batch, "create_openai_client", lambda cfg=None: provider)

    index = Index(
        collection_name="batch_collect_test",
        persist_directory=str(tmp_path / "chroma"),
        embedding_model="default",
    )
    index.embedding_model = "openai"
    index.embedding_config = {"model_name": "text-embedding-3-small", "api_key": "test"}

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "semantic_search": {
            "embedding_model": "openai",
            "include_fulltext": False,
            "openai_batch": {"enabled": True},
            "update_config": {"auto_update": True, "update_frequency": "daily"},
        }
    }))
    zotero = FakeZotero(["AAA", "BBB"], version=10)

    def new_search(client=None):
        monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: client or env_ns.zotero)
        return semantic_search.ZoteroSemanticSearch(chroma_client=index, config_path=str(config_path))

    env_ns = SimpleNamespace(
        provider=provider, index=index, zotero=zotero, config_path=config_path, new_search=new_search,
    )
    yield env_ns
    zclient.clear_active_library()


def _section(env):
    return json.loads(env.config_path.read_text())["semantic_search"]


@pytest.fixture
def presearch(env, monkeypatch):
    """Fire the pre-search sync synchronously, on a clock the test controls."""

    class InlineThread:
        def __init__(self, target, **kwargs):
            self._target = target

        def start(self):
            self._target()

    clock = SimpleNamespace(now=100_000.0)
    monkeypatch.setattr(search_tools, "_threading", SimpleNamespace(Thread=InlineThread))
    monkeypatch.setattr(search_tools, "_time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(search_tools, "_last_presearch_sync_ts", 0.0)
    monkeypatch.setattr(search_tools, "_last_presearch_collect_ts", float("-inf"))

    def fire(after_seconds=0.0):
        clock.now += after_seconds
        # The search tool builds a new instance for every call.
        search_tools._maybe_fire_presearch_sync(env.new_search())

    return fire


# ---------------------------------------------------------------------------
# Repeat submission
# ---------------------------------------------------------------------------


def test_two_background_syncs_in_a_row_submit_once(env, presearch):
    presearch()
    assert env.provider.created == ["batch-1"]

    presearch(after_seconds=61)

    assert env.provider.created == ["batch-1"], "the second background sync paid for the same changes again"


def test_submitting_a_batch_saves_last_update_but_not_the_sync_version(env):
    before = datetime.now()

    stats = env.new_search().update_database()

    assert stats["batch_submitted"] is True
    section = _section(env)
    assert datetime.fromisoformat(section["update_config"]["last_update"]) >= before
    assert not section.get("last_sync_versions", {}).get("0"), "nothing is indexed yet"
    assert env.new_search().should_update_database() is False


# ---------------------------------------------------------------------------
# Which run is unfinished, which run covers which, and which one a lookup finds
# ---------------------------------------------------------------------------


def _submit_run(env, *, group_id=0, force=False, created_at=None, scope=None, records=2, budget=None, target=20):
    manifest = openai_batch.submit_embedding_batches(
        records=[
            {"id": f"R{i}", "document": "x" * 150, "metadata": {"item_key": f"R{i}", "group_id": group_id or 0}}
            for i in range(records)
        ],
        model_name="text-embedding-3-small",
        embedding_config={"api_key": "test"},
        config_path=str(env.config_path),
        client=env.provider,
        force_full_rebuild=force,
        target_sync_version=target,
        group_id=group_id,
        max_enqueued_tokens=budget,
        scope={"limited": False, "since_version": 10} if scope is None else scope,
    )
    if created_at:
        # Back-dating a run back-dates what it has sent to the provider, too.
        manifest["created_at"] = created_at
        for entry in manifest["batches"]:
            entry.pop("submitted_at", None)
        openai_batch.save_manifest(manifest)
    return manifest


def _entry(status="in_progress", imported=False, batch_id="b1", output_file_id="out"):
    return {
        "batch_id": batch_id,
        "status": status,
        "output_file_id": output_file_id,
        "imported_at": "2026-01-01T00:00:00" if imported else None,
    }


@pytest.mark.parametrize(
    "force, entries, unfinished",
    [
        (False, [_entry()], True),
        (False, [_entry("pending", batch_id=None)], True),
        (False, [_entry("completed")], True),
        (False, [_entry("completed", imported=True)], False),
        (False, [_entry("completed", imported=True), _entry("in_progress")], True),
        # Nothing more can happen; the next update re-covers these items.
        (False, [_entry("failed", output_file_id=None)], False),
        (False, [_entry("expired", output_file_id=None), _entry("completed", imported=True)], False),
        # An expired chunk's file still holds paid rows to import.
        (False, [_entry("expired")], True),
        ({"lost": True}, [], False),
        # Completed with every row failed: there is no output file to import.
        (False, [_entry("completed", output_file_id=None)], False),
        # A rebuild is all-or-nothing, so one dead chunk kills the whole run.
        (True, [_entry("failed"), _entry("completed")], False),
        (True, [_entry("in_progress"), _entry("completed")], True),
        (False, [], False),
    ],
)
def test_run_is_unfinished(force, entries, unfinished):
    if isinstance(force, dict):  # a batch the provider no longer has
        force, entries = False, [{**_entry(), **force}]
    manifest = {"force_full_rebuild": force, "batches": entries}
    assert batch_common.run_is_unfinished(manifest, openai_batch.ADAPTER) is unfinished


def _run(run_id="new", group_id=0, force=False, limited=False, since=10):
    return {
        "run_id": run_id,
        "group_id": group_id,
        "force_full_rebuild": force,
        "limited": limited,
        "since_version": since,
    }


@pytest.mark.parametrize(
    "newer, older, covers",
    [
        # Same changes.
        (_run(), _run("old"), True),
        # A rebuild covers anything, but not when it stops after --limit items.
        (_run(force=True), _run("old"), True),
        (_run(force=True), _run("old", force=True), True),
        (_run(force=True, limited=True), _run("old"), False),
        # --limit covers only its own items, and is covered only by a rebuild.
        (_run(limited=True), _run("old"), False),
        (_run(), _run("old", limited=True), False),
        (_run(force=True), _run("old", limited=True), True),
        # Small incremental runs do not cover a rebuild or a first build.
        (_run(), _run("old", force=True), False),
        (_run(since=10), _run("old", since=0), False),
        # A first build does cover an incremental run; an older cut covers a newer one.
        (_run(since=0), _run("old", since=10), True),
        (_run(since=10), _run("old", since=20), True),
        (_run(since=20), _run("old", since=10), False),
        # A scan that skips indexed items (--fulltext, a collection filter) proves nothing.
        (_run(since=None), _run("old"), False),
        (_run(), _run("old", since=None), False),
        # Other libraries and the run itself.
        (_run(group_id=7), _run("old"), False),
        (_run("old"), _run("old"), False),
        # Manifests from before these fields keep the rule they had.
        ({"run_id": "new", "group_id": 0}, {"run_id": "old", "group_id": 0}, True),
        ({"run_id": "new", "group_id": 0}, {"run_id": "old", "group_id": 0, "force_full_rebuild": True}, False),
        # Against one that has them nothing can be proved.
        (_run(), {"run_id": "old", "group_id": 0}, False),
        ({"run_id": "new", "group_id": 0}, _run("old"), False),
    ],
)
def test_run_covers(newer, older, covers):
    assert batch_common.run_covers(newer, older) is covers


def test_only_a_full_rebuild_may_supersede_an_unfinished_run():
    assert batch_common.may_supersede(_run(force=True), _run("old")) is True
    assert batch_common.may_supersede(_run(force=True, limited=True), _run("old")) is False
    # Covers it, but paying for it again gains nothing the wait does not.
    assert batch_common.may_supersede(_run(), _run("old")) is False


def test_lookup_prefers_this_librarys_unfinished_run_over_another_librarys_newer_one(env):
    ours = _submit_run(env, group_id=0, created_at="2026-01-01T10:00:00.000000Z")
    _submit_run(env, group_id=GROUP_ID, created_at="2026-01-01T11:00:00.000000Z")

    found = openai_batch.find_manifest(config_path=str(env.config_path), group_id=0)

    assert found["run_id"] == ours["run_id"]


def test_lookup_finds_an_older_unfinished_rebuild_behind_a_finished_incremental_run(env):
    rebuild = _submit_run(env, force=True, created_at="2026-01-01T10:00:00.000000Z")
    newer = _submit_run(env, created_at="2026-01-01T11:00:00.000000Z")
    for entry in newer["batches"]:
        entry.update(status="completed", imported_at="2026-01-01T12:00:00")
    openai_batch.save_manifest(newer)

    found = openai_batch.find_manifest(config_path=str(env.config_path), group_id=0)

    assert found["run_id"] == rebuild["run_id"]


def test_lookup_skips_an_unfinished_run_a_newer_run_covers(env):
    _submit_run(env, created_at="2026-01-01T10:00:00.000000Z")
    newer = _submit_run(env, force=True, created_at="2026-01-01T11:00:00.000000Z")

    found = openai_batch.find_manifest(config_path=str(env.config_path), group_id=0)

    assert found["run_id"] == newer["run_id"]


def test_lookup_falls_back_to_this_librarys_newest_run_then_the_newest_overall(env):
    ours = _submit_run(env, group_id=0, created_at="2026-01-01T10:00:00.000000Z")
    for entry in ours["batches"]:
        entry.update(status="completed", imported_at="2026-01-01T12:00:00")
    openai_batch.save_manifest(ours)
    other = _submit_run(env, group_id=GROUP_ID, created_at="2026-01-01T11:00:00.000000Z")
    cfg = str(env.config_path)

    assert openai_batch.find_manifest(config_path=cfg, group_id=0)["run_id"] == ours["run_id"]
    assert openai_batch.find_manifest(config_path=cfg, group_id=999)["run_id"] == other["run_id"]
    assert openai_batch.find_manifest(config_path=cfg)["run_id"] == other["run_id"]


def test_gemini_lookup_prefers_the_active_librarys_unfinished_run(tmp_path):
    cfg = tmp_path / "config.json"
    root = gemini_batch.get_gemini_batch_root(str(cfg))
    for name, group_id, created_at in (("ours", 0, "10"), ("theirs", GROUP_ID, "11")):
        (root / name).mkdir()
        (root / name / "manifest.json").write_text(json.dumps({
            "run_id": name,
            "created_at": f"2026-01-01T{created_at}:00:00.000000Z",
            "group_id": group_id,
            "batches": [{"batch_id": "b", "status": "JOB_STATE_RUNNING", "imported_at": None}],
        }))

    assert gemini_batch.find_manifest(config_path=str(cfg), group_id=0)["run_id"] == "ours"


def test_a_submission_cut_off_part_way_records_the_chunks_it_never_sent(env):
    """The run must not look complete once the chunks it did send finish."""

    real_create = env.provider.batches.create
    calls = []

    def create_then_fail(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise ConnectionError("provider went away")
        return real_create(**kwargs)

    env.provider.batches.create = create_then_fail

    with pytest.raises(ConnectionError):
        openai_batch.submit_embedding_batches(
            records=[{"id": f"R{i}", "document": "x" * 150, "metadata": {}} for i in range(3)],
            model_name="text-embedding-3-small",
            embedding_config={"api_key": "test"},
            config_path=str(env.config_path),
            client=env.provider,
            max_requests=1,
            max_enqueued_tokens=1_000_000,
        )

    manifest = openai_batch.find_manifest(config_path=str(env.config_path))
    assert [b["status"] for b in manifest["batches"]] == ["in_progress", "pending", "pending"]
    assert [bool(b.get("batch_id")) for b in manifest["batches"]] == [True, False, False]
    assert batch_common.run_is_unfinished(manifest, openai_batch.ADAPTER) is True


def test_a_submission_that_fails_on_its_first_chunk_is_still_on_record(env):
    def fail(**kwargs):
        raise ConnectionError("provider went away")

    env.provider.batches.create = fail

    with pytest.raises(ConnectionError):
        _submit_run(env, records=1)

    manifest = openai_batch.find_manifest(config_path=str(env.config_path))
    assert [b["status"] for b in manifest["batches"]] == ["pending"]


@pytest.mark.parametrize(
    "status, unfinished",
    [
        ("JOB_STATE_RUNNING", True),
        ("JOB_STATE_SUCCEEDED", True),
        ("pending", True),
        ("JOB_STATE_FAILED", False),
        ("JOB_STATE_EXPIRED", False),
    ],
)
def test_run_is_unfinished_reads_gemini_states(status, unfinished):
    # Gemini has no separate output file id to go missing.
    manifest = {"batches": [{"batch_id": "b", "status": status, "imported_at": None}]}
    assert batch_common.run_is_unfinished(manifest, gemini_batch.ADAPTER) is unfinished


# ---------------------------------------------------------------------------
# Every update collects first
# ---------------------------------------------------------------------------


def _indexed(env):
    return set(env.index.get_all_ids())


def _versions(env):
    return _section(env).get("last_sync_versions", {})


def _set_version(env, version, library="0"):
    config = json.loads(env.config_path.read_text())
    config["semantic_search"]["last_sync_versions"] = {library: version}
    env.config_path.write_text(json.dumps(config))


def _batch_off(env):
    return env.new_search().update_database(use_batch=False)


def test_a_run_with_nothing_unfinished_makes_no_provider_call(env, monkeypatch):
    _set_version(env, 10)  # library unchanged: the update stops at the version check

    def no_client(cfg=None):
        raise AssertionError("a provider client was built with nothing to collect")

    monkeypatch.setattr(openai_batch, "create_openai_client", no_client)

    stats = env.new_search().update_database()

    assert env.provider.network_calls == 0
    assert "batch_collect" not in stats


def test_settled_runs_are_not_looked_at_again(env):
    env.new_search().update_database()
    env.provider.finish_all()
    env.new_search().update_database()
    calls_after_import = env.provider.network_calls

    env.new_search().update_database()

    assert env.provider.network_calls == calls_after_import


def test_a_batch_update_imports_a_finished_run_and_moves_the_sync_version(env):
    env.new_search().update_database()
    env.provider.finish_all()

    stats = env.new_search().update_database()

    assert stats["batch_collect"]["imported_items"] == 2
    assert _indexed(env) == {"AAA", "BBB"}
    assert _versions(env)["0"] == 10
    assert env.provider.created == ["batch-1"], "the library is unchanged since the import"


def test_a_realtime_update_still_collects_results_already_paid_for(env, monkeypatch):
    env.new_search().update_database()
    env.provider.finish_all()
    # A real update would embed here, and this index has no embedding function
    # to do that with; nothing has changed, so nothing should reach it.
    monkeypatch.setattr(Index, "upsert_documents", lambda *a, **k: pytest.fail("embedded in realtime"))

    stats = _batch_off(env)

    assert stats["batch_collect"]["imported_items"] == 2
    assert _indexed(env) == {"AAA", "BBB"}


def test_collecting_submits_waiting_chunks(env):
    run = _submit_run(env, budget=100, records=5)  # 3 chunks; only the first goes out
    assert env.provider.created == ["batch-1"]
    env.provider.finish_all()
    _set_version(env, 5)
    env.zotero = FakeZotero(["AAA"], version=5)

    stats = env.new_search().update_database()

    assert stats["batch_collect"]["submitted_chunks"] == 1
    assert env.provider.created == ["batch-1", "batch-2"]
    assert stats["batch_deferred"], "the run is still unfinished"
    manifest = openai_batch.load_manifest(Path(run["manifest_path"]))
    assert [b["status"] for b in manifest["batches"]] == ["completed", "in_progress", "pending"]


def test_an_update_cut_off_while_submitting_is_finished_by_the_next_one(env):
    real_create = env.provider.batches.create
    calls = []

    def create_then_fail(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise ConnectionError("provider went away")
        return real_create(**kwargs)

    env.provider.batches.create = create_then_fail

    stats = env.new_search().update_database(batch_max_requests=1)

    assert "provider went away" in stats["error"]
    assert not _versions(env).get("0")
    assert env.provider.created == ["batch-1"]

    env.provider.batches.create = real_create
    stats = env.new_search().update_database(batch_max_requests=1)

    assert stats["batch_collect"]["submitted_chunks"] == 1
    assert env.provider.created == ["batch-1", "batch-2"], "the rest of the same run, not a new one"
    assert stats["batch_deferred"]

    env.provider.finish_all()
    env.new_search().update_database()
    assert _indexed(env) == {"AAA", "BBB"}
    assert _versions(env)["0"] == 10


def test_auto_loop_drives_a_new_run_to_the_end(env):
    real_create = env.provider.batches.create

    def create_finished(**kwargs):
        created = real_create(**kwargs)
        env.provider.statuses[created.id] = "completed"
        return created

    env.provider.batches.create = create_finished

    stats = env.new_search().update_database(auto_loop=True, batch_poll_interval=0)

    assert stats["auto_loop"]["imported_items"] == 2
    assert _indexed(env) == {"AAA", "BBB"}
    assert _versions(env)["0"] == 10


def test_auto_loop_is_deferred_too_rather_than_waiting_on_a_run_it_did_not_start(env):
    env.new_search().update_database()

    stats = env.new_search().update_database(auto_loop=True, batch_poll_interval=0)

    assert stats["batch_deferred"]
    assert "auto_loop" not in stats


def test_the_update_tool_says_what_was_collected_and_deferred(env, monkeypatch):
    env.new_search().update_database()
    monkeypatch.setattr(semantic_search, "create_semantic_search", lambda path: env.new_search())
    ctx = SimpleNamespace(info=lambda *a: None, error=lambda *a: None)

    output = search_tools.update_search_database(ctx=ctx)

    assert "**Deferred:** Not submitting a new batch" in output
    assert "**Unfinished batch runs checked:** 1" in output


def test_the_update_tool_reports_a_submission(env, monkeypatch):
    monkeypatch.setattr(semantic_search, "create_semantic_search", lambda path: env.new_search())
    ctx = SimpleNamespace(info=lambda *a: None, error=lambda *a: None)

    output = search_tools.update_search_database(ctx=ctx)

    assert "**Batch submitted:** run" in output


def test_a_collect_failure_is_reported_and_moves_nothing(env):
    env.new_search().update_database()
    env.provider.fail_status_with = ConnectionError("provider unreachable")

    stats = env.new_search().update_database()

    assert "ConnectionError" in str(stats["batch_collect"]["errors"])
    assert not stats.get("error")
    assert stats["batch_deferred"], "the update still says what became of it"
    assert not _versions(env).get("0")
    assert env.provider.created == ["batch-1"]

    # And it is only a delay.
    env.provider.fail_status_with = None
    env.provider.finish_all()
    env.new_search().update_database()
    assert _indexed(env) == {"AAA", "BBB"}
    assert _versions(env)["0"] == 10


def test_nothing_is_collected_into_an_index_chromadb_cannot_read(env, monkeypatch):
    env.new_search().update_database()
    env.provider.finish_all()
    calls = env.provider.network_calls
    monkeypatch.setattr(Index, "read_error", lambda self: "Failed to apply logs")

    stats = env.new_search().update_database()
    collected = env.new_search().collect_batch_results()

    assert "Failed to apply logs" in stats["error"]
    assert "Failed to apply logs" in collected["error"]
    assert env.provider.network_calls == calls
    assert not _versions(env).get("0")


def test_a_failed_download_before_a_rebuild_import_leaves_the_index_alone(env):
    env.index.upsert_embeddings(
        documents=["old"], metadatas=[{"item_key": "OLD", "group_id": 0}], ids=["OLD"], embeddings=[[0.1, 0.2, 0.3]]
    )
    env.new_search().update_database(force_full_rebuild=True)
    env.provider.finish_all()
    env.provider.fail_download_with = ConnectionError("download failed")

    stats = env.new_search().update_database()

    assert "ConnectionError" in str(stats["batch_collect"]["errors"])
    assert _indexed(env) == {"OLD"}, "the reset has to wait until every file is in hand"


def test_an_unattended_rebuild_import_will_not_drop_another_librarys_documents(env):
    env.new_search().update_database(force_full_rebuild=True)
    env.provider.finish_all()
    # Another library gets indexed while the rebuild runs.
    env.index.upsert_embeddings(
        documents=["theirs"],
        metadatas=[{"item_key": "THEIRS", "group_id": GROUP_ID}],
        ids=["THEIRS"],
        embeddings=[[0.1, 0.2, 0.3]],
    )

    stats = env.new_search().update_database()

    assert "batch-import" in str(stats["batch_collect"]["errors"])
    assert _indexed(env) == {"THEIRS"}

    # Importing it by hand is the user's decision to make.
    env.new_search()._import_batch("openai")
    assert _indexed(env) == {"AAA", "BBB"}


def test_collecting_leaves_another_librarys_run_alone(env):
    _submit_run(env, group_id=GROUP_ID)
    env.provider.finish_all()
    calls = env.provider.network_calls

    env.new_search().update_database()

    assert env.provider.created[-1] == "batch-2", "this library's own run goes out"
    assert env.provider.network_calls - calls == 2, "upload and create, no status call for the other run"
    assert not _indexed(env)


def test_a_run_from_another_embedding_model_is_neither_collected_nor_blocking(env):
    run = _submit_run(env)
    manifest = openai_batch.load_manifest(Path(run["manifest_path"]))
    manifest["model"] = "text-embedding-3-large"
    openai_batch.save_manifest(manifest)
    env.provider.finish_all()

    stats = env.new_search().update_database()

    assert "batch_collect" not in stats
    assert not _indexed(env), "those vectors don't belong in this ChromaDB collection"
    assert stats["batch_submitted"] is True


def test_a_rebuild_that_can_no_longer_finish_is_reported(env):
    env.new_search().update_database(force_full_rebuild=True)
    env.provider.finish_all("expired")

    stats = env.new_search().update_database()

    assert "--force-rebuild" in str(stats["batch_collect"]["errors"])
    assert stats["batch_submitted"] is True, "a run that can't finish holds nothing up"


def test_failed_chunks_are_reported_and_covered_by_the_next_run(env):
    env.new_search().update_database()
    env.provider.finish_all("failed")

    stats = env.new_search().update_database()

    assert "failed" in str(stats["batch_collect"]["errors"])
    assert not _versions(env).get("0")
    assert env.provider.created == ["batch-1", "batch-2"]


def test_a_second_process_holding_the_lock_skips_collecting(env):
    env.new_search().update_database()
    env.provider.finish_all()
    calls = env.provider.network_calls

    with semantic_search._acquire_update_lock(semantic_search._update_lock_path()) as acquired:
        assert acquired
        # flock is per open file, so a second open in this process stands in
        # for the other process.
        stats = env.new_search().update_database()
        collected = env.new_search().collect_batch_results()

    assert stats["skipped_reason"] == "another_update_in_progress"
    assert collected["skipped_reason"] == "another_update_in_progress"
    assert env.provider.network_calls == calls


def test_batch_status_does_not_save_without_the_lock(env):
    env.new_search().update_database()
    run = openai_batch.find_manifest(config_path=str(env.config_path))
    env.provider.finish_all()
    search = env.new_search()

    with semantic_search._acquire_update_lock(semantic_search._update_lock_path()) as acquired:
        assert acquired
        status = search._get_batch_status("openai")

    assert status["batches"][0]["status"] == "completed", "it still reports the live status"
    saved = openai_batch.load_manifest(Path(run["manifest_path"]))
    assert saved["batches"][0]["status"] == "in_progress"


def test_batch_import_refreshes_only_with_the_lock_held(env):
    env.new_search().update_database()
    env.provider.finish_all()
    calls = env.provider.network_calls

    with semantic_search._acquire_update_lock(semantic_search._update_lock_path()) as acquired:
        assert acquired
        with pytest.raises(RuntimeError, match="already running"):
            env.new_search()._import_batch("openai")

    assert env.provider.network_calls == calls


# ---------------------------------------------------------------------------
# A new run does not supersede what it doesn't cover
# ---------------------------------------------------------------------------


def test_a_limited_run_is_deferred_while_a_run_is_unfinished(env):
    env.new_search().update_database()

    stats = env.new_search().update_database(limit=1)

    assert env.provider.created == ["batch-1"]
    assert not stats.get("batch_submitted")
    run_id = openai_batch.find_manifest(config_path=str(env.config_path))["run_id"]
    assert run_id in stats["batch_deferred"]
    assert "batch-status" in stats["batch_deferred"]


def test_a_limited_run_never_supersedes_when_one_does_get_written(env):
    older = _submit_run(env, created_at="2026-01-01T10:00:00.000000Z")
    _submit_run(env, scope={"limited": True, "since_version": None}, created_at="2026-01-01T11:00:00.000000Z")

    assert env.new_search()._superseded_by("openai", older) is None


def test_an_older_run_stays_superseded_behind_a_newer_limited_run(env):
    older = _submit_run(env, created_at="2026-01-01T10:00:00.000000Z")
    rebuild = _submit_run(env, force=True, created_at="2026-01-01T11:00:00.000000Z")
    _submit_run(env, scope={"limited": True, "since_version": None}, created_at="2026-01-01T12:00:00.000000Z")

    assert env.new_search()._superseded_by("openai", older) == rebuild["run_id"]


def test_an_incremental_run_is_deferred_behind_an_unfinished_first_build(env):
    env.new_search().update_database()
    env.zotero.add("CCC", 11)

    stats = env.new_search().update_database()

    assert env.provider.created == ["batch-1"]
    assert stats["batch_deferred"]
    assert not _versions(env).get("0")


def test_an_incremental_update_it_does_cover_is_deferred_too_and_nothing_is_lost(env):
    _set_version(env, 10)
    env.zotero.add("CCC", 11)
    env.new_search().update_database()
    assert env.provider.created == ["batch-1"]
    env.zotero.add("DDD", 12)

    stats = env.new_search().update_database()

    assert stats["batch_deferred"], "it would pay for CCC again"
    assert env.provider.created == ["batch-1"]
    assert _versions(env)["0"] == 10, "DDD stays behind the sync version"

    env.provider.finish_all()
    stats = env.new_search().update_database()  # imports CCC, then submits DDD

    assert stats["batch_collect"]["imported_items"] == 1
    assert _versions(env)["0"] == 11
    assert env.provider.created == ["batch-1", "batch-2"]
    assert stats["submitted_items"] == 1


def test_a_rebuild_may_supersede_and_does_not_pay_for_waiting_chunks(env):
    older = _submit_run(env, budget=100, records=5)
    env.provider.finish_all()

    stats = env.new_search().update_database(force_full_rebuild=True)

    assert stats["batch_submitted"] is True
    assert env.provider.created == ["batch-1", "batch-2"], "batch-2 is the rebuild; no parked chunk went out"
    assert env.new_search()._superseded_by("openai", older) == stats["batch_run_id"]


def test_a_superseded_run_is_not_collected(env):
    _submit_run(env, created_at="2026-01-01T10:00:00.000000Z")
    _submit_run(env, force=True, created_at="2026-01-01T11:00:00.000000Z")
    env.provider.statuses["batch-1"] = "completed"
    search = env.new_search()

    assert [run["force_full_rebuild"] for run in search._unfinished_batch_runs("openai", 0)] == [True]


def test_a_limited_rebuild_does_not_supersede(env):
    env.new_search().update_database()

    stats = env.new_search().update_database(force_full_rebuild=True, limit=1, allow_mass_deletion=True)

    assert stats["batch_deferred"]
    assert env.provider.created == ["batch-1"]


def test_a_realtime_sync_overtakes_an_unfinished_run(env):
    """Its items are in, and newer; its results would write older text over them."""
    run = _submit_run(env, budget=100, records=5, target=20)
    _set_version(env, 25)
    env.provider.finish_all()
    calls = env.provider.network_calls
    search = env.new_search()

    assert search._unfinished_batch_runs("openai", 0) == []
    assert search._overtaken_by_sync(run) == 25
    stats = search.collect_batch_results()

    assert env.provider.network_calls == calls
    assert not stats.get("batch_collect")
    assert not _indexed(env)


# ---------------------------------------------------------------------------
# The sync version
# ---------------------------------------------------------------------------


def test_importing_a_settled_run_again_does_not_lower_the_sync_version(env):
    env.new_search().update_database()
    env.provider.finish_all()
    env.new_search()._import_batch("openai")
    assert _versions(env)["0"] == 10
    _set_version(env, 50)

    env.new_search()._import_batch("openai")

    assert _versions(env)["0"] == 50


def test_a_run_that_lands_late_does_not_lower_the_sync_version(env):
    _submit_run(env, target=20, scope={"limited": False, "since_version": None})
    _set_version(env, 50)
    env.provider.finish_all()

    env.new_search()._import_batch("openai")

    assert _indexed(env) == {"R0", "R1"}
    assert _versions(env)["0"] == 50


def test_an_import_still_raises_the_sync_version(env):
    _submit_run(env, target=20)
    _set_version(env, 10)
    env.provider.finish_all()

    env.new_search()._import_batch("openai")

    assert _versions(env)["0"] == 20


def test_a_rebuild_landing_sets_the_sync_version_to_what_the_collection_now_holds(env):
    """It replaced the collection, so a higher number would hide the gap for good."""
    _submit_run(env, force=True, target=20, scope={"limited": False, "since_version": 0, "allow_mass_deletion": True})
    _set_version(env, 50)
    env.provider.finish_all()

    env.new_search()._import_batch("openai")
    assert _versions(env)["0"] == 20

    _set_version(env, 50)
    env.new_search()._import_batch("openai")  # settled now: the plain rule again
    assert _versions(env)["0"] == 50


# ---------------------------------------------------------------------------
# The pre-search sync
# ---------------------------------------------------------------------------


def test_presearch_only_collects_while_a_run_is_unfinished_at_most_hourly(env, presearch):
    presearch()
    assert env.provider.created == ["batch-1"]
    statuses_seen = env.provider.network_calls

    presearch(after_seconds=61)
    first_collect = env.provider.network_calls
    assert first_collect == statuses_seen + 1, "one status call: it checks well before the daily schedule"

    presearch(after_seconds=61)
    presearch(after_seconds=1800)
    assert env.provider.network_calls == first_collect, "not again within the hour"

    presearch(after_seconds=3600)
    assert env.provider.network_calls == first_collect + 1
    assert env.provider.created == ["batch-1"]

    env.provider.finish_all()
    presearch(after_seconds=3700)
    assert _indexed(env) == {"AAA", "BBB"}
    assert _versions(env)["0"] == 10
    assert env.provider.created == ["batch-1"]


def test_presearch_collecting_never_starts_a_new_run_even_when_an_update_is_due(env, presearch):
    presearch()
    config = json.loads(env.config_path.read_text())
    config["semantic_search"]["update_config"]["last_update"] = "2020-01-01T00:00:00"
    env.config_path.write_text(json.dumps(config))
    env.zotero.add("CCC", 11)

    presearch(after_seconds=3700)

    assert env.provider.created == ["batch-1"]


def test_presearch_does_not_touch_the_provider_with_auto_update_off(env, presearch):
    env.new_search().update_database()
    calls = env.provider.network_calls
    config = json.loads(env.config_path.read_text())
    config["semantic_search"]["update_config"].update(auto_update=False, update_frequency="manual")
    env.config_path.write_text(json.dumps(config))

    presearch(after_seconds=3700)

    assert env.provider.network_calls == calls


def test_presearch_with_no_run_and_no_update_due_does_nothing(env, presearch):
    config = json.loads(env.config_path.read_text())
    config["semantic_search"]["update_config"]["last_update"] = datetime.now().isoformat()
    env.config_path.write_text(json.dumps(config))

    presearch()

    assert env.provider.network_calls == 0


# ---------------------------------------------------------------------------
# What the user is told
# ---------------------------------------------------------------------------


def test_cli_output_for_a_collect_and_a_deferred_submission(capsys):
    from zotero_mcp.cli import _print_update_stats

    _print_update_stats({
        "batch_mode": True,
        "batch_provider": "openai",
        "batch_deferred": "Not submitting a new batch: run R is still unfinished.",
        "batch_collect": {
            "runs": ["R"],
            "imported_items": 3,
            "submitted_chunks": 1,
            "waiting": ["the force-rebuild run imports once all batches complete"],
            "errors": ["run R: 2 item(s) came back without an embedding"],
        },
    })

    out = capsys.readouterr().out
    assert "Embeddings imported: 3" in out
    assert "Waiting chunks submitted: 1" in out
    assert "Still waiting: the force-rebuild run" in out
    assert "WARNING: run R: 2 item(s)" in out
    assert "deferred" in out
    assert "run R is still unfinished" in out


# ---------------------------------------------------------------------------
# Review round 1
# ---------------------------------------------------------------------------

LONG_AGO = "2026-01-01T00:00:00+00:00"


class NotFound(Exception):
    """What the OpenAI SDK raises for a batch or file it no longer has."""

    status_code = 404


def _seed_fresh_index(env, n=5, **extra):
    env.index.collection.upsert(
        ids=[f"FRESH{i}#0" for i in range(n)],
        documents=[f"passage {i}" for i in range(n)],
        metadatas=[{"item_key": f"FRESH{i}", "group_id": 0, **extra} for i in range(n)],
        embeddings=[[0.1, 0.2, 0.3] for _ in range(n)],
    )


def _batch_config(env, enabled, last_update=None):
    cfg = json.loads(env.config_path.read_text())
    cfg["semantic_search"]["openai_batch"]["enabled"] = enabled
    if last_update:
        cfg["semantic_search"]["update_config"]["last_update"] = last_update
    env.config_path.write_text(json.dumps(cfg))


def _notice(env):
    return env.new_search().batch_notice()


REBUILD_SCOPE = {"limited": False, "since_version": 0}


def test_c1_a_stale_rebuild_does_not_replace_a_newer_index_on_a_plain_update(env):
    run = _submit_run(env, force=True, target=20, created_at=LONG_AGO, scope=REBUILD_SCOPE)
    env.provider.finish_all()
    _seed_fresh_index(env)
    _set_version(env, 50)
    before = _indexed(env)
    env.zotero.add("NEW", 60)

    stats = env.new_search().update_database()

    assert _indexed(env) == before
    assert _versions(env)["0"] == 50
    assert "batch-import" in str(stats["batch_collect"]["errors"])
    assert run["run_id"] in str(_notice(env))
    assert stats["batch_submitted"] is True, "a stale rebuild isn't blocking"


def test_c1_a_legacy_rebuild_stranded_behind_an_imported_run_stays_stranded(env):
    _submit_run(env, force=True, target=20, created_at=LONG_AGO, scope={})
    newer = _submit_run(env, target=30, created_at="2026-02-01T00:00:00+00:00", scope={})
    for batch in newer["batches"]:
        batch.update(status="completed", imported_at="2026-02-02T00:00:00")
    openai_batch.save_manifest(newer)
    env.provider.finish_all()
    _seed_fresh_index(env)
    _set_version(env, 50)
    before = _indexed(env)

    env.new_search().collect_batch_results()

    assert _indexed(env) == before


def test_c1_a_realtime_rebuild_retires_an_in_flight_batch_rebuild(env, monkeypatch):
    _submit_run(env, force=True, target=20, scope=REBUILD_SCOPE)
    monkeypatch.setattr(Index, "upsert_documents", lambda *a, **k: None)

    stats = env.new_search().update_database(use_batch=False, force_full_rebuild=True, allow_mass_deletion=True)

    assert not stats.get("error")
    assert _section(env)["last_rebuild_at"]["0"]
    assert not env.new_search().has_unfinished_batch_run()
    _seed_fresh_index(env)
    before = _indexed(env)
    env.provider.finish_all()

    env.new_search().collect_batch_results()

    assert _indexed(env) == before


def test_c1_a_force_rebuilds_own_collect_skips_an_older_rebuild(env):
    older = _submit_run(env, force=True, target=20, scope=REBUILD_SCOPE)
    env.provider.finish_all()
    _seed_fresh_index(env)
    before = _indexed(env)
    calls = env.provider.network_calls

    stats = env.new_search().update_database(force_full_rebuild=True, allow_mass_deletion=True)

    assert _indexed(env) == before, "the older rebuild must not wipe the index first"
    assert older["run_id"] not in str(stats.get("batch_collect", {}).get("runs"))
    assert env.provider.network_calls == calls + 2, "upload and create for the new run, no status call"
    assert stats["batch_submitted"] is True


def test_c1_a_rebuild_with_different_chunking_is_refused_unattended(env):
    run = _submit_run(env, force=True, target=20, scope=REBUILD_SCOPE)
    env.provider.finish_all()
    _seed_fresh_index(env, chunk_index=0)
    before = _indexed(env)

    stats = env.new_search().update_database()

    assert _indexed(env) == before
    assert "batch-import" in str(stats["batch_collect"]["errors"])
    assert run["run_id"] in str(_notice(env))


def test_c1_a_current_rebuild_still_lands_on_a_plain_update(env):
    _submit_run(env, force=True, target=20, scope=REBUILD_SCOPE)
    env.provider.finish_all()
    _seed_fresh_index(env)

    env.new_search().collect_batch_results()

    assert _indexed(env) == {"R0", "R1"}
    assert _versions(env)["0"] == 20
    assert _section(env)["last_rebuild_at"]["0"]


def test_c1_batch_import_can_still_replace_the_index_with_a_stale_rebuild(env):
    _submit_run(env, force=True, target=20, created_at=LONG_AGO, scope=REBUILD_SCOPE)
    env.provider.finish_all()
    _seed_fresh_index(env)
    _set_version(env, 50)

    env.new_search()._import_batch("openai")

    assert _indexed(env) == {"R0", "R1"}


def test_c2_a_stuck_run_does_not_stop_realtime_auto_updates(env, presearch, monkeypatch):
    _batch_config(env, False, last_update="2026-01-01T00:00:00")
    _submit_run(env, target=20, created_at=LONG_AGO)
    env.provider.fail_status_with = RuntimeError("no answer")
    updates = []
    monkeypatch.setattr(
        semantic_search.ZoteroSemanticSearch, "update_database", lambda self, **k: updates.append(k) or {}
    )

    presearch()

    assert len(updates) == 1


def test_c2_a_realtime_update_past_the_window_says_so(env):
    _batch_config(env, False)
    _set_version(env, 10)
    _submit_run(env, target=20, created_at=LONG_AGO)
    env.provider.fail_status_with = RuntimeError("no answer")

    stats = env.new_search().update_database()

    assert not stats.get("batch_deferred")
    assert "window" in str(stats["batch_collect"]["errors"])


def test_c2_a_stuck_batch_mode_user_stops_being_deferred(env):
    run = _submit_run(env, target=20, created_at=LONG_AGO)
    env.provider.fail_download_with = RuntimeError("file past retention")
    env.provider.finish_all()

    stats = env.new_search().update_database()

    assert not stats.get("batch_deferred")
    assert stats["batch_submitted"] is True
    assert run["run_id"] in str(_notice(env))


def test_c2_a_run_inside_its_window_still_defers_when_the_status_call_fails(env):
    env.new_search().update_database()
    env.provider.fail_status_with = ConnectionError("temporary")

    stats = env.new_search().update_database()

    assert stats["batch_deferred"]


def test_c2_a_batch_the_provider_no_longer_has_is_dead(env):
    env.new_search().update_database()
    env.provider.fail_status_with = NotFound("no such batch")

    stats = env.new_search().update_database()

    assert not stats.get("batch_deferred")
    assert stats["batch_submitted"] is True
    assert "no longer has" in str(stats["batch_collect"]["errors"])
    assert "every change since" in str(stats["batch_collect"]["errors"])


def test_c2_an_output_file_the_provider_no_longer_has_is_dead(env):
    env.new_search().update_database()
    env.provider.finish_all()
    env.provider.fail_download_with = NotFound("no such file")

    stats = env.new_search().update_database()

    assert not stats.get("batch_deferred")
    assert stats["batch_submitted"] is True
    assert "no longer has" in str(stats["batch_collect"]["errors"])
    assert not _versions(env).get("0")


def test_c2_the_last_error_and_deferral_are_kept_and_cleared(env):
    env.new_search().update_database()
    env.provider.fail_status_with = ConnectionError("temporary")

    env.new_search().update_database()

    notice = _notice(env)
    assert any("run" in note for note in notice["notes"])
    assert "Not submitting" in notice["deferred"]

    env.provider.fail_status_with = None
    env.provider.finish_all()
    env.new_search().update_database()

    assert _notice(env) == {}


def test_c2_the_database_status_shows_the_notice(env):
    env.new_search().update_database()
    env.provider.fail_status_with = ConnectionError("temporary")
    env.new_search().update_database()

    status = env.new_search().get_database_status()

    assert "Not submitting" in status["batch_notice"]["deferred"]


def test_c2_db_status_prints_the_notice(capsys):
    from zotero_mcp.cli import _print_batch_notice

    _print_batch_notice({"batch_notice": {"notes": ["run R: stuck"], "deferred": "Not submitting: R"}})

    out = capsys.readouterr().out
    assert "run R: stuck" in out
    assert "Not submitting: R" in out


def test_c2_the_status_tool_shows_the_notice(env, monkeypatch):
    env.new_search().update_database()
    env.provider.fail_status_with = ConnectionError("temporary")
    env.new_search().update_database()
    config_dir = env.config_path.parent / ".config" / "zotero-mcp"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(env.config_path.read_text())
    (config_dir / "batch_notices.json").write_text((env.config_path.parent / "batch_notices.json").read_text())
    monkeypatch.setattr(search_tools.Path, "home", lambda: env.config_path.parent)
    ctx = SimpleNamespace(info=lambda *a: None, error=lambda *a: None)

    output = search_tools.get_search_database_status(ctx=ctx)

    assert "Not submitting" in output


@pytest.mark.parametrize("results", [[], [{"item_key": "AAA", "similarity_score": 0.5, "matched_text": "x"}]])
def test_c2_semantic_search_output_carries_a_one_line_note(env, monkeypatch, results):
    env.new_search().update_database()
    env.provider.fail_status_with = ConnectionError("temporary")
    env.new_search().update_database()
    search = env.new_search()
    monkeypatch.setattr(search, "search", lambda **k: {"results": results})
    monkeypatch.setattr(semantic_search, "create_semantic_search", lambda *a, **k: search)
    monkeypatch.setattr(search_tools, "_maybe_fire_presearch_sync", lambda _s: None)
    monkeypatch.setattr(search_tools.Path, "home", lambda: env.config_path.parent)
    ctx = SimpleNamespace(info=lambda *a: None, error=lambda *a: None)

    output = search_tools.semantic_search(query="q", ctx=ctx)

    notes = [line for line in output.splitlines() if line.startswith("**Index note:**")]
    assert len(notes) == 1
    assert "zotero_get_search_database_status" in notes[0]


def test_c2_giving_up_is_a_command(env):
    env.new_search().update_database()
    run = openai_batch.find_manifest(config_path=str(env.config_path))
    env.zotero.add("CCC", 11)

    result = env.new_search().abandon_batch_run("openai")

    assert result["abandoned"] == run["run_id"]
    assert not env.new_search().has_unfinished_batch_run()
    stats = env.new_search().update_database()
    assert stats["batch_submitted"] is True
    assert not _versions(env).get("0"), "giving up doesn't claim anything was indexed"
    assert openai_batch.load_manifest(Path(run["manifest_path"]))["abandoned_at"]


def test_c2_a_run_given_up_on_is_finished_and_not_found_again(env):
    run = _submit_run(env, created_at="2026-11-01T00:00:00+00:00")
    finished = _submit_run(env, created_at="2026-10-01T00:00:00+00:00")
    for batch in finished["batches"]:
        batch.update(status="completed", imported_at="2026-10-01T12:00:00")
    openai_batch.save_manifest(finished)
    env.new_search().abandon_batch_run("openai", run_id=run["run_id"])

    manifest = openai_batch.load_manifest(Path(run["manifest_path"]))

    assert batch_common.run_is_unfinished(manifest, openai_batch.ADAPTER) is False
    assert env.new_search().import_openai_batch()["run_id"] == finished["run_id"]


def test_c2_giving_up_with_several_candidates_asks_which(env):
    first = _submit_run(env, force=True, created_at="2026-11-01T00:00:00+00:00", scope=REBUILD_SCOPE)
    second = _submit_run(env, group_id=0, created_at="2026-11-02T00:00:00+00:00")

    result = env.new_search().abandon_batch_run("openai")

    assert "abandoned" not in result
    assert first["run_id"] in result["candidates"]
    assert second["run_id"] in result["candidates"]
    assert env.new_search().abandon_batch_run("openai", run_id=first["run_id"])["abandoned"] == first["run_id"]


def test_c2_giving_up_is_on_the_command_line(capsys):
    with patch("sys.argv", ["zotero-mcp", "help", "batch-abandon"]):
        with pytest.raises(SystemExit) as exc:
            cli_main()
    assert exc.value.code == 0
    assert "--run-id" in capsys.readouterr().out


def test_c2_the_deferred_message_is_true_for_a_run_that_cannot_finish(env):
    env.new_search().update_database()

    stats = env.new_search().update_database()

    assert "batch-abandon" in stats["batch_deferred"]
    assert "window" in stats["batch_deferred"]
    assert "delete" not in stats["batch_deferred"]


def test_i1_an_implicit_realtime_update_defers_behind_an_in_flight_batch(env, monkeypatch):
    _set_version(env, 10)
    env.zotero.add("NEW1", 20)
    env.new_search().update_database()
    _batch_config(env, False)
    embedded = []
    monkeypatch.setattr(Index, "upsert_documents", lambda self, documents, metadatas, ids: embedded.extend(ids))

    stats = env.new_search().update_database()

    assert embedded == []
    assert stats["batch_deferred"]
    assert _versions(env)["0"] == 10
    assert env.new_search().has_unfinished_batch_run()


def test_i1_an_explicit_no_batch_proceeds_and_warns(env, monkeypatch):
    _set_version(env, 10)
    env.zotero.add("NEW1", 20)
    env.new_search().update_database()
    embedded = []
    monkeypatch.setattr(Index, "upsert_documents", lambda self, documents, metadatas, ids: embedded.extend(ids))

    stats = _batch_off(env)

    assert embedded == ["NEW1"]
    assert "discarded" in stats["batch_discard_warning"]


def test_i1_a_run_past_its_window_defers_no_realtime_update(env, monkeypatch):
    _batch_config(env, False)
    _set_version(env, 10)
    env.zotero.add("NEW1", 20)
    _submit_run(env, target=15, created_at=LONG_AGO)
    env.provider.fail_status_with = ConnectionError("temporary")
    embedded = []
    monkeypatch.setattr(Index, "upsert_documents", lambda self, documents, metadatas, ids: embedded.extend(ids))

    stats = env.new_search().update_database()

    assert not stats.get("batch_deferred")
    assert embedded == ["NEW1"]


def test_i1_a_limited_implicit_realtime_rebuild_defers_before_the_reset(env):
    _batch_config(env, False)
    _submit_run(env)
    _seed_fresh_index(env)
    before = _indexed(env)

    stats = env.new_search().update_database(force_full_rebuild=True, limit=1, allow_mass_deletion=True)

    assert stats["batch_deferred"]
    assert _indexed(env) == before


def test_i2_plain_batch_import_passes_over_an_overtaken_run(env):
    finished = _submit_run(env, target=25, created_at="2026-10-01T00:00:00+00:00")
    for batch in finished["batches"]:
        batch.update(status="completed", imported_at="2026-10-01T12:00:00")
    openai_batch.save_manifest(finished)
    overtaken = _submit_run(env, target=20, created_at="2026-11-01T00:00:00+00:00")
    _set_version(env, 25)
    env.provider.finish_all()
    assert not env.new_search()._unfinished_batch_runs("openai", 0)

    stats = env.new_search().import_openai_batch()
    assert stats["run_id"] != overtaken["run_id"]
    assert stats["imported_items"] == 0
    assert env.new_search()._get_batch_status("openai")["run_id"] != overtaken["run_id"]


def test_i2_plain_batch_import_refuses_when_only_an_overtaken_run_is_left(env):
    run = _submit_run(env, target=20)
    _set_version(env, 25)
    env.provider.finish_all()

    stats = env.new_search().import_openai_batch()

    assert stats["imported_items"] == 0
    assert not _indexed(env)
    assert "--batch-id" in str(stats["errors"])

    # By explicit id it is the user's call.
    batch_id = run["batches"][0]["batch_id"]
    stats = env.new_search().import_openai_batch(batch_ids=[batch_id])
    assert stats["imported_items"] == 2


def test_i2_plain_batch_import_refuses_another_models_run(env):
    run = _submit_run(env)
    manifest = openai_batch.load_manifest(Path(run["manifest_path"]))
    manifest["model"] = "text-embedding-3-large"
    openai_batch.save_manifest(manifest)
    env.provider.finish_all()

    stats = env.new_search().import_openai_batch()

    assert not _indexed(env)
    assert "--batch-id" in str(stats["errors"])


def test_i3_a_stale_legacy_run_buys_and_imports_nothing(env):
    _submit_run(env, budget=100, records=5, target=20, created_at=LONG_AGO, scope={})
    _set_version(env, 500)
    env.provider.finish_all()
    created_before = list(env.provider.created)

    env.new_search().collect_batch_results()

    assert env.provider.created == created_before
    assert not _indexed(env)


def test_i3_a_run_past_the_age_limit_does_not_submit_waiting_chunks_unasked(env):
    _submit_run(env, budget=100, records=5, target=20, created_at=LONG_AGO)
    _set_version(env, 10)
    env.provider.finish_all()
    created_before = list(env.provider.created)

    stats = env.new_search().collect_batch_results()

    assert env.provider.created == created_before
    assert "batch-import" in str(stats["batch_collect"]["errors"])

    # Asking is what batch-import is for.
    assert env.new_search()._import_batch("openai")["batches_submitted"] == 1


def test_i3_a_throttled_run_keeps_going_when_its_last_chunk_went_out_recently(env):
    run = _submit_run(env, budget=100, records=5, target=20)
    manifest = openai_batch.load_manifest(Path(run["manifest_path"]))
    manifest["created_at"] = LONG_AGO  # started long ago; the chunk went out just now
    openai_batch.save_manifest(manifest)
    env.provider.finish_all()
    _set_version(env, 10)

    stats = env.new_search().collect_batch_results()

    assert stats["batch_collect"]["submitted_chunks"] == 1


def test_i4_an_expired_chunk_with_a_file_is_imported_but_does_not_promote(env):
    run = _submit_run(env, target=20)
    env.provider.statuses["batch-1"] = "expired"
    env.provider.expired_with_output = True

    stats = env.new_search().collect_batch_results()

    assert _indexed(env) == {"R0", "R1"}, "these rows are paid for"
    assert not _versions(env).get("0"), "other rows of that chunk never ran"
    saved = openai_batch.load_manifest(Path(run["manifest_path"]))
    assert not batch_common.run_is_unfinished(saved, openai_batch.ADAPTER)
    assert "every change since" in str(stats["batch_collect"]["errors"])


def test_i4_a_rebuild_with_an_expired_chunk_stays_all_or_nothing(env):
    manifest = {
        "force_full_rebuild": True,
        "batches": [_entry("expired"), _entry("completed")],
    }
    assert batch_common.run_is_unfinished(manifest, openai_batch.ADAPTER) is False


def test_i4_the_message_says_what_is_actually_bought_again(env):
    _set_version(env, 10)
    env.zotero.add("N1", 20)
    env.zotero.add("N2", 20)
    env.zotero.versions.pop("AAA")
    env.zotero.versions.pop("BBB")
    env.new_search().update_database(batch_max_requests=1)
    assert len(env.provider.created) == 2
    env.provider.statuses["batch-1"] = "completed"
    env.provider.statuses["batch-2"] = "failed"

    stats = env.new_search().collect_batch_results()

    assert "every change since" in str(stats["batch_collect"]["errors"])
    assert "those items" not in str(stats["batch_collect"]["errors"])


def test_i5_the_other_libraries_guard_defers_when_it_cannot_read(env, monkeypatch):
    env.index.collection.upsert(
        ids=["OTHER#0"], documents=["o"], metadatas=[{"group_id": 999}], embeddings=[[0.1, 0.2, 0.3]],
    )
    _submit_run(env, force=True, target=20, scope=REBUILD_SCOPE)
    env.provider.finish_all()
    real = env.index.collection.get

    def boom(*a, **k):
        if "group_id" in (k.get("where") or {}):
            raise RuntimeError("transient read failure")
        return real(*a, **k)

    monkeypatch.setattr(env.index.collection, "get", boom, raising=False)

    stats = env.new_search().collect_batch_results()

    assert "could not be checked" in str(stats["batch_collect"]["errors"])
    assert "OTHER#0" in set(real(include=[])["ids"])


def test_i5_unattributed_documents_count_for_the_other_libraries_guard(env):
    env.index.collection.upsert(
        ids=["UNTAGGED"], documents=["u"], metadatas=[{"title": "no library"}], embeddings=[[0.1, 0.2, 0.3]],
    )
    _submit_run(env, force=True, target=20, scope=REBUILD_SCOPE)
    env.provider.finish_all()

    stats = env.new_search().collect_batch_results()

    assert "UNTAGGED" in _indexed(env)
    assert "batch-import" in str(stats["batch_collect"]["errors"])


def test_i6_presearch_never_imports_a_rebuild_and_says_what_to_run(env, presearch):
    run = _submit_run(env, force=True, target=20, scope=REBUILD_SCOPE)
    env.provider.finish_all()
    _seed_fresh_index(env)
    before = _indexed(env)

    presearch()

    assert _indexed(env) == before
    notes = _notice(env)["notes"]
    assert run["run_id"] in str(notes)
    assert "batch-import" in str(notes)
    assert env.new_search().has_unfinished_batch_run()


def test_i6_an_update_fired_by_presearch_never_imports_a_rebuild_either(env, presearch):
    _submit_run(env, force=True, target=20, created_at=LONG_AGO, scope=REBUILD_SCOPE)
    env.provider.finish_all()
    _seed_fresh_index(env)
    before = _indexed(env)

    presearch()  # past its window, so not blocking: this is a due update

    assert _indexed(env) == before


def test_i6_presearch_still_moves_a_rebuild_along(env, presearch):
    _submit_run(env, force=True, target=20, budget=100, records=5, scope=REBUILD_SCOPE)
    env.provider.finish_all()

    presearch()

    assert len(env.provider.created) == 2


def test_m1_a_download_cut_off_leaves_no_output_file(tmp_path, monkeypatch):
    output = tmp_path / "batch-001-records-output.jsonl"

    def cut_off(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(batch_common.os, "replace", cut_off)
    with pytest.raises(OSError):
        batch_common.write_text_atomic(output, "partial")

    assert list(tmp_path.iterdir()) == []


def test_m1_downloads_go_through_the_atomic_write(tmp_path, monkeypatch):
    written = []
    monkeypatch.setattr(batch_common, "write_text_atomic", lambda path, text: written.append(path))
    client = SimpleNamespace(files=SimpleNamespace(content=lambda file_id: SimpleNamespace(text="x")))
    openai_batch.download_file_text(client, "f", tmp_path / "a")

    gemini_client = SimpleNamespace(
        batches=SimpleNamespace(get=lambda name: SimpleNamespace(dest=SimpleNamespace(file_name="f"))),
        files=SimpleNamespace(download=lambda file: b"x"),
    )
    gemini_batch.download_results(gemini_client, {"batch_id": "b"}, tmp_path / "b")

    assert written == [tmp_path / "a", tmp_path / "b"]


def test_m2_a_crash_before_the_sync_version_save_is_finished_by_the_next_collect(env, monkeypatch):
    env.new_search().update_database()
    env.provider.finish_all()
    search = env.new_search()

    def crash(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(search, "_save_update_config", crash)
    with pytest.raises(KeyboardInterrupt):
        search.collect_batch_results()
    assert not _versions(env).get("0")
    calls = env.provider.network_calls

    env.new_search().collect_batch_results()

    assert _versions(env)["0"] == 10
    assert env.provider.network_calls == calls
    manifest = openai_batch.find_manifest(config_path=str(env.config_path))
    assert manifest["promoted_at"]


def test_m3_auto_loop_looks_at_this_librarys_run(env, monkeypatch):
    seen = []
    real = openai_batch.find_manifest

    def spy(**kwargs):
        seen.append(kwargs.get("group_id"))
        return real(**kwargs)

    monkeypatch.setattr(openai_batch, "find_manifest", spy)
    real_create = env.provider.batches.create

    def create_finished(**kwargs):
        created = real_create(**kwargs)
        env.provider.statuses[created.id] = "completed"
        return created

    env.provider.batches.create = create_finished
    env.new_search().update_database(auto_loop=True, batch_poll_interval=0)

    assert seen and all(group_id == 0 for group_id in seen)


def test_m4_a_landing_rebuild_with_no_target_resets_the_sync_version(env):
    _submit_run(env, force=True, target=None, scope=REBUILD_SCOPE)
    _set_version(env, 50)
    env.provider.finish_all()

    env.new_search()._import_batch("openai")

    assert _versions(env)["0"] == 0


def test_m5_a_check_with_nothing_ready_does_not_read_as_success(env, capsys):
    env.new_search().update_database()
    capsys.readouterr()

    env.new_search().collect_batch_results()

    err = capsys.readouterr().err
    assert "Checked 1 unfinished batch run" in err
    assert "Collected" not in err


def test_m5_only_errors_is_not_printed_as_collected(capsys):
    from zotero_mcp.cli import _print_batch_collect

    _print_batch_collect({
        "batch_collect": {"runs": [], "imported_items": 0, "submitted_chunks": 0, "errors": ["run R: stuck"]}
    })

    out = capsys.readouterr().out
    assert "Collected" not in out
    assert "run R: stuck" in out


def test_m5_raw_sdk_text_is_not_shown(env):
    env.new_search().update_database()
    env.provider.fail_status_with = ConnectionError("Bearer sk-secret at https://internal")

    stats = env.new_search().update_database()

    assert "sk-secret" not in str(stats["batch_collect"]["errors"])
    assert "sk-secret" not in str(_notice(env))
    assert "ConnectionError" in str(stats["batch_collect"]["errors"])


def test_m6_a_dead_rebuild_note_offers_giving_up_and_clears_with_it(env):
    run = _submit_run(env, force=True, scope=REBUILD_SCOPE)
    env.provider.finish_all("failed")
    stats = env.new_search().collect_batch_results()
    assert "batch-abandon" in str(stats["batch_collect"]["errors"])

    env.new_search().abandon_batch_run("openai", run_id=run["run_id"])

    assert "batch_collect" not in env.new_search().collect_batch_results()
    assert _notice(env) == {}


def test_m6_a_realtime_rebuild_clears_a_dead_rebuild_note(env, monkeypatch):
    _submit_run(env, force=True, scope=REBUILD_SCOPE)
    env.provider.finish_all("failed")
    env.new_search().collect_batch_results()
    monkeypatch.setattr(Index, "upsert_documents", lambda *a, **k: None)
    env.new_search().update_database(use_batch=False, force_full_rebuild=True, allow_mass_deletion=True)

    assert "batch_collect" not in env.new_search().collect_batch_results()


def test_m6_a_dead_rebuild_from_another_model_is_not_mentioned(env):
    run = _submit_run(env, force=True, scope=REBUILD_SCOPE)
    manifest = openai_batch.load_manifest(Path(run["manifest_path"]))
    manifest["model"] = "text-embedding-3-large"
    openai_batch.save_manifest(manifest)
    env.provider.finish_all("failed")

    assert "batch_collect" not in env.new_search().collect_batch_results()
