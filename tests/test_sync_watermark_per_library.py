"""Tests for issue #393: the incremental-sync watermark must be tracked per
library.

Every Zotero library (personal + each group) has its own independent,
monotonically increasing version counter, so a single shared
``semantic_search.last_sync_version`` scalar corrupted sync state for both
libraries as soon as ``zotero_switch_library`` was used: the new library's
counter was compared against a stale watermark from a different library, and
then overwrote it.

Watermarks now live in ``semantic_search.last_sync_versions``, keyed by the
same group_id identity used to tag documents ("0" = personal library).
"""

import json
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import client as zclient
from zotero_mcp import semantic_search

GROUP_ID = 6015547


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    """Keep these tests independent of the host shell's Zotero env vars and
    never leak the process-wide active-library override."""
    monkeypatch.delenv("ZOTERO_LOCAL", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_TYPE", raising=False)
    zclient.clear_active_library()
    yield
    zclient.clear_active_library()


class FakeChromaClient:
    """Minimal ChromaClient stand-in recording upserts and deletions."""

    def __init__(self, preloaded_ids=None):
        self.embedding_max_tokens = 8000
        # Metadata store so get_all_ids(where=...) honors the real class's
        # DB-side group_id filtering; preloaded docs model previously-indexed
        # personal-library items (group_id 0, the post-migration steady state).
        self._metas = {i: {"item_key": i, "group_id": 0} for i in (preloaded_ids or [])}
        self.added = []
        self.deleted = []
        self.reset_calls = 0

    def truncate_text(self, text, max_tokens=None):
        return text

    def get_existing_ids(self, ids):
        return {i for i in ids if i in self._metas}

    def get_all_ids(self, where=None):
        if where and "group_id" in where:
            return {
                i for i, m in self._metas.items()
                if m.get("group_id") == where["group_id"]
            }
        return set(self._metas)

    def get_document_metadata(self, doc_id):
        return None

    def iter_metadatas(self, batch_size=500):
        return iter(())

    def update_metadatas(self, ids, metadatas):
        for i, m in zip(ids, metadatas):
            self._metas.setdefault(i, {}).update(m)

    def upsert_documents(self, documents, metadatas, ids):
        self.added.append((list(documents), list(metadatas), list(ids)))
        for i, m in zip(ids, metadatas):
            self._metas[i] = dict(m)

    def add_documents(self, documents, metadatas, ids):
        self.upsert_documents(documents, metadatas, ids)

    def delete_documents(self, ids):
        self.deleted.extend(list(ids))
        for i in ids:
            self._metas.pop(i, None)

    def reset_collection(self):
        self.reset_calls += 1
        self._metas = {}


class FakeZoteroClient:
    """pyzotero double scoped to a single library with its own counter."""

    def __init__(self, keys, library_version):
        self._items = {k: _paper(k) for k in keys}
        self.versions_state = {k: library_version for k in keys}
        self.library_version = library_version
        self.calls = []

    def items(self, start=0, limit=100, **kwargs):
        self.calls.append(("items", start, limit))
        keys = list(self._items)[start:start + limit]
        return [self._items[k] for k in keys]

    def item(self, key):
        self.calls.append(("item", key))
        if key not in self._items:
            raise LookupError(key)
        return self._items[key]

    def children(self, key, start=0, limit=25, **kwargs):
        return []

    def fulltext_item(self, key):
        raise RuntimeError("404 no fulltext")

    def item_versions(self, since=None, **kwargs):
        self.calls.append(("item_versions", since))
        if since is None:
            return dict(self.versions_state)
        return {k: v for k, v in self.versions_state.items() if v > since}

    def last_modified_version(self, **kwargs):
        self.calls.append(("last_modified_version",))
        return self.library_version


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


def _write_config(tmp_path, semantic_extra=None):
    cfg = {
        "semantic_search": {
            "embedding_model": "default",
            "update_config": {"auto_update": False, "update_frequency": "manual"},
            "include_fulltext": False,
        }
    }
    if semantic_extra:
        cfg["semantic_search"].update(semantic_extra)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    return str(config_path)


def _build_search(monkeypatch, zot, chroma, config_path=None):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: zot)
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    return semantic_search.ZoteroSemanticSearch(
        chroma_client=chroma, config_path=config_path
    )


def _saved(config_path):
    return json.loads(open(config_path).read())["semantic_search"]


# ---------------------------------------------------------------------------
# _load_last_sync_version / _save_update_config
# ---------------------------------------------------------------------------

def test_load_reads_the_active_librarys_watermark(monkeypatch, tmp_path):
    config_path = _write_config(
        tmp_path, {"last_sync_versions": {"0": 50000, str(GROUP_ID): 1200}}
    )
    search = _build_search(monkeypatch, FakeZoteroClient([], 0), FakeChromaClient(),
                           config_path=config_path)

    assert search._load_last_sync_version() == 50000
    zclient.set_active_library(str(GROUP_ID), "group")
    assert search._load_last_sync_version() == 1200
    zclient.clear_active_library()
    assert search._load_last_sync_version() == 50000


def test_load_bootstraps_library_missing_from_the_map(monkeypatch, tmp_path):
    """A library absent from the map has never been synced; it must not
    inherit another library's counter (nor a leftover legacy scalar)."""
    config_path = _write_config(
        tmp_path, {"last_sync_versions": {"0": 50000}, "last_sync_version": 50000}
    )
    search = _build_search(monkeypatch, FakeZoteroClient([], 0), FakeChromaClient(),
                           config_path=config_path)

    zclient.set_active_library(str(GROUP_ID), "group")
    assert search._load_last_sync_version() == 0


def test_save_touches_only_the_active_librarys_entry(monkeypatch, tmp_path):
    config_path = _write_config(
        tmp_path, {"last_sync_versions": {"0": 50000, str(GROUP_ID): 1200}}
    )
    search = _build_search(monkeypatch, FakeZoteroClient([], 0), FakeChromaClient(),
                           config_path=config_path)

    zclient.set_active_library(str(GROUP_ID), "group")
    search._save_update_config(last_sync_version=1300)

    versions = _saved(config_path)["last_sync_versions"]
    assert versions == {"0": 50000, str(GROUP_ID): 1300}
    # A group's counter must never leak into the legacy scalar, which older
    # versions apply to whatever library they are pointed at.
    assert "last_sync_version" not in _saved(config_path)


def test_save_mirrors_legacy_scalar_for_personal_library(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, {"last_sync_version": 10})
    search = _build_search(monkeypatch, FakeZoteroClient([], 0), FakeChromaClient(),
                           config_path=config_path)

    search._save_update_config(last_sync_version=42)

    saved = _saved(config_path)
    assert saved["last_sync_versions"] == {"0": 42}
    assert saved["last_sync_version"] == 42


def test_save_honors_explicit_library_key(monkeypatch, tmp_path):
    """The OpenAI-batch import promotes the watermark of the library the batch
    was submitted against, not whichever library is active at import time."""
    config_path = _write_config(tmp_path)
    search = _build_search(monkeypatch, FakeZoteroClient([], 0), FakeChromaClient(),
                           config_path=config_path)

    search._save_update_config(last_sync_version=1200, library_key=str(GROUP_ID))

    saved = _saved(config_path)
    assert saved["last_sync_versions"] == {str(GROUP_ID): 1200}
    assert "last_sync_version" not in saved


# ---------------------------------------------------------------------------
# Migration from the pre-#393 scalar
# ---------------------------------------------------------------------------

def test_legacy_scalar_migrates_to_the_default_library(monkeypatch, tmp_path):
    """Without a runtime switch the client is scoped to the env-configured
    default library, the only library an old config could have tracked across
    restarts — so the scalar is reused and no full re-scan is forced."""
    config_path = _write_config(tmp_path, {"last_sync_version": 50000})
    search = _build_search(monkeypatch, FakeZoteroClient([], 0), FakeChromaClient(),
                           config_path=config_path)

    assert search._load_last_sync_version() == 50000


def test_legacy_scalar_migrates_to_a_group_default_library(monkeypatch, tmp_path):
    """Same for someone whose env default is a group library."""
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", str(GROUP_ID))
    monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "group")
    config_path = _write_config(tmp_path, {"last_sync_version": 1200})
    search = _build_search(monkeypatch, FakeZoteroClient([], 0), FakeChromaClient(),
                           config_path=config_path)

    assert search._load_last_sync_version() == 1200
    search._save_update_config(last_sync_version=1250)
    assert _saved(config_path)["last_sync_versions"] == {str(GROUP_ID): 1250}


def test_legacy_scalar_discarded_while_a_switch_is_active(monkeypatch, tmp_path):
    """The scalar's provenance is unknowable once a library override is in
    play; a redundant full scan beats silently skipping the whole library."""
    config_path = _write_config(tmp_path, {"last_sync_version": 50000})
    search = _build_search(monkeypatch, FakeZoteroClient([], 0), FakeChromaClient(),
                           config_path=config_path)

    zclient.set_active_library(str(GROUP_ID), "group")
    assert search._load_last_sync_version() == 0


def test_migrated_config_keeps_other_libraries_intact(monkeypatch, tmp_path):
    """Upgrading writes the map without dropping the migrated scalar's value
    for the library it belonged to."""
    config_path = _write_config(tmp_path, {"last_sync_version": 50000})
    search = _build_search(monkeypatch, FakeZoteroClient([], 0), FakeChromaClient(),
                           config_path=config_path)

    search._save_update_config(last_sync_version=search._load_last_sync_version())
    zclient.set_active_library(str(GROUP_ID), "group")
    search._save_update_config(last_sync_version=1200)

    assert _saved(config_path)["last_sync_versions"] == {"0": 50000, str(GROUP_ID): 1200}


# ---------------------------------------------------------------------------
# End-to-end through update_database
# ---------------------------------------------------------------------------

def test_switching_libraries_keeps_watermarks_independent(monkeypatch, tmp_path):
    """personal -> group -> personal: neither library's watermark is clobbered,
    and the group is bootstrapped instead of being compared against the
    personal library's much higher counter (which would return no changed
    items at all)."""
    config_path = _write_config(tmp_path)

    # 1. Personal library bootstraps to version 50000.
    personal = FakeZoteroClient(["PERS1"], library_version=50000)
    chroma = FakeChromaClient()
    search = _build_search(monkeypatch, personal, chroma, config_path=config_path)
    search.update_database()
    assert _saved(config_path)["last_sync_versions"] == {"0": 50000}

    # 2. Switch to a group whose independent counter is far lower.
    zclient.set_active_library(str(GROUP_ID), "group")
    group = FakeZoteroClient(["GRP1"], library_version=1200)
    group_search = _build_search(monkeypatch, group, chroma, config_path=config_path)
    stats = group_search.update_database()

    # The group must actually be indexed, not skipped as "unchanged"
    assert stats["processed_items"] == 1
    assert "GRP1" in chroma.get_all_ids()
    # No since-based fetch against the personal library's watermark
    assert not any(c[0] == "item_versions" and c[1] for c in group.calls)
    versions = _saved(config_path)["last_sync_versions"]
    assert versions == {"0": 50000, str(GROUP_ID): 1200}

    # 3. Back to personal: its watermark survived and still drives incremental.
    zclient.clear_active_library()
    personal_again = FakeZoteroClient(["PERS1", "PERS2"], library_version=50010)
    personal_again.versions_state = {"PERS1": 40000, "PERS2": 50010}
    back = _build_search(monkeypatch, personal_again, chroma, config_path=config_path)
    stats = back.update_database()

    assert ("item_versions", 50000) in personal_again.calls
    assert stats["processed_items"] == 1  # only PERS2 changed since 50000
    versions = _saved(config_path)["last_sync_versions"]
    assert versions == {"0": 50010, str(GROUP_ID): 1200}
    # The personal sync's deletion pass is scoped to group_id 0, so the
    # group's document — indexed in step 2 and untouched since — survives
    # the round-trip (#404: it used to be deleted here).
    assert "GRP1" in chroma.get_all_ids()
    assert chroma.deleted == []


def test_legacy_scalar_is_not_adopted_by_a_client_scoped_elsewhere(monkeypatch, tmp_path):
    """The pre-#393 scalar's provenance is the env-configured default
    library. Whether to trust it must be judged against the RUN's pinned
    library — not the live override, which can be cleared/changed mid-run:
    a group-scoped run would otherwise inherit the personal library's
    counter and silently skip every group change below it."""
    config_path = _write_config(tmp_path, {"last_sync_version": 500})
    group = FakeZoteroClient(["GRP1"], library_version=1200)
    group.library_id = str(GROUP_ID)
    group.library_type = "groups"
    chroma = FakeChromaClient()
    # No override active (module state says "personal"); the client — and
    # so the run — is scoped to the group.
    search = _build_search(monkeypatch, group, chroma, config_path=config_path)

    stats = search.update_database()

    assert stats["processed_items"] == 1
    # Bootstrap full scan, not incremental against the foreign scalar:
    assert not any(c[0] == "item_versions" and c[1] for c in group.calls)
    assert _saved(config_path)["last_sync_versions"][str(GROUP_ID)] == 1200


def test_update_run_scopes_to_the_clients_library_not_the_module_override(monkeypatch, tmp_path):
    """The updater must take its library identity from the Zotero client it
    reads keys and versions from. The module-level active-library override
    is mutable shared state: a zotero_switch_library tool call can land
    while the server's background update is mid-run, and any identity read
    at call time would attach the wrong library to this run's tagging,
    watermark — and, once deletion is group_id-scoped, deletion authority."""
    config_path = _write_config(tmp_path)
    group = FakeZoteroClient(["GRP1"], library_version=1200)
    # pyzotero clients expose their scope (library_type in its plural URL
    # form); the fake mirrors that.
    group.library_id = str(GROUP_ID)
    group.library_type = "groups"
    chroma = FakeChromaClient()
    search = _build_search(monkeypatch, group, chroma, config_path=config_path)
    # The override — module state — says "personal" for the whole run.

    search.update_database()

    metas = [m for batch in chroma.added for m in batch[1]]
    assert metas, "expected the group item to be indexed"
    assert all(m["group_id"] == GROUP_ID for m in metas)
    assert _saved(config_path)["last_sync_versions"] == {str(GROUP_ID): 1200}


def test_watermark_ahead_of_library_version_forces_full_scan(monkeypatch, tmp_path):
    """Defence in depth for configs that already carry a foreign watermark: a
    library's counter never decreases, so a watermark ahead of it cannot be
    ours and must not drive item_versions(since=...)."""
    config_path = _write_config(
        tmp_path, {"last_sync_versions": {str(GROUP_ID): 50000}}
    )
    zclient.set_active_library(str(GROUP_ID), "group")
    group = FakeZoteroClient(["GRP1"], library_version=1200)
    chroma = FakeChromaClient()
    search = _build_search(monkeypatch, group, chroma, config_path=config_path)

    stats = search.update_database()

    assert stats["processed_items"] == 1
    assert not any(c[0] == "item_versions" and c[1] for c in group.calls)
    assert _saved(config_path)["last_sync_versions"] == {str(GROUP_ID): 1200}
