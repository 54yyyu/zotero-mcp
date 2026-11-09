"""Batch runs are collected before anything else, and never paid for twice.

These run ``update_database`` end to end against a real temporary ChromaDB
index; only the provider SDK client and the Zotero client are fakes. No
embedding function runs: vectors arrive precomputed from the fake provider.
"""

import json
import sys
from datetime import datetime
from types import SimpleNamespace

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")

from zotero_mcp import client as zclient  # noqa: E402
from zotero_mcp import openai_batch, semantic_search  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient  # noqa: E402
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
                return _batch_object(batch_id, provider.statuses[batch_id])

        self.files = Files()
        self.batches = Batches()

    def finish_all(self, status="completed"):
        for batch_id in self.statuses:
            self.statuses[batch_id] = status


def _batch_object(batch_id, status):
    return SimpleNamespace(
        id=batch_id,
        status=status,
        output_file_id=f"out-{batch_id}" if status == "completed" else None,
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
        def __init__(self, target, args=(), **kwargs):
            self._target, self._args = target, args

        def start(self):
            self._target(*self._args)

    clock = SimpleNamespace(now=100_000.0)
    monkeypatch.setattr(search_tools, "_threading", SimpleNamespace(Thread=InlineThread))
    monkeypatch.setattr(search_tools, "_time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(search_tools, "_last_presearch_sync_ts", 0.0)

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
# An unfinished run holds new submissions for its library
# ---------------------------------------------------------------------------


def _set_frequency(env, frequency):
    config = json.loads(env.config_path.read_text())
    config["semantic_search"]["update_config"]["update_frequency"] = frequency
    env.config_path.write_text(json.dumps(config))


def test_startup_syncs_do_not_resubmit_while_a_run_is_unfinished(env, presearch):
    # "startup" is due on every server start whatever last_update says.
    _set_frequency(env, "startup")
    presearch()
    env.zotero.add("CCC", version=11)

    presearch(after_seconds=61)
    stats = env.new_search().update_database()

    assert env.provider.created == ["batch-1"]
    assert stats["skipped_reason"] == "batch_run_unfinished"
    assert "batch-import" in stats["skipped_message"]


def test_after_the_import_the_next_update_submits_the_new_changes(env):
    env.new_search().update_database()
    env.zotero.add("CCC", version=11)
    env.provider.finish_all()

    env.new_search()._import_batch("openai")
    stats = env.new_search().update_database()

    assert env.provider.created == ["batch-1", "batch-2"]
    assert stats["submitted_items"] == 1


def test_a_force_rebuild_still_goes_ahead(env):
    env.new_search().update_database()

    env.new_search().update_database(force_full_rebuild=True)

    assert env.provider.created == ["batch-1", "batch-2"]


def test_a_run_older_than_the_hold_no_longer_holds(env):
    stats = env.new_search().update_database()
    manifest_path = stats["batch_manifest"]
    manifest = json.loads(open(manifest_path).read())
    manifest["created_at"] = "2020-01-01T00:00:00.000000Z"
    open(manifest_path, "w").write(json.dumps(manifest))

    env.new_search().update_database()

    assert env.provider.created == ["batch-1", "batch-2"]
