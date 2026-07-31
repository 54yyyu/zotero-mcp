"""The incremental deletion pass must be library-scoped (issue #404).

The pass used to diff EVERY stored id against ``item_versions()`` of the
ACTIVE library only, so syncing one library deleted every other library's
documents from the index (a real-world run wiped 738 live group-library
docs). Deletion authority now requires a doc to be positively attributed
(``group_id``) to the syncing library; unattributed docs are structurally
undeletable, a failed or empty ``item_versions()`` skips deletion instead
of wiping, and a mass deletion needs an explicit opt-in.
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
    monkeypatch.delenv("ZOTERO_LOCAL", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_TYPE", raising=False)
    zclient.clear_active_library()
    yield
    zclient.clear_active_library()


class FakeZoteroClient:
    """pyzotero double scoped to one library."""

    def __init__(self, versions_state=None, library_version=9,
                 bare_versions_error=None):
        self.versions_state = dict(versions_state or {})
        self.library_version = library_version
        self.bare_versions_error = bare_versions_error
        self.calls = []

    def item_versions(self, since=None, **kwargs):
        self.calls.append(("item_versions", since))
        if since is None:
            if self.bare_versions_error is not None:
                raise self.bare_versions_error
            return dict(self.versions_state)
        return {k: v for k, v in self.versions_state.items() if v > since}

    def item(self, key):
        return {"key": key, "version": self.versions_state.get(key, 1), "data": {
            "key": key, "itemType": "journalArticle", "title": f"Paper {key}",
            "dateAdded": "2024-01-01T00:00:00Z", "dateModified": "2024-01-01T00:00:00Z",
        }}

    def children(self, *a, **kw):
        return []

    def last_modified_version(self, **kwargs):
        return self.library_version


class RecordingChroma:
    """Metadata-carrying ChromaClient double that honors get_all_ids(where=)."""

    def __init__(self, docs=None):
        self.embedding_max_tokens = 8000
        self._docs = {k: dict(v) for k, v in (docs or {}).items()}
        self.deleted = []
        self.chunk_deletes = []
        self.reset_calls = 0

    def truncate_text(self, text, max_tokens=None):
        return text

    def get_existing_ids(self, ids):
        return {i for i in ids if i in self._docs}

    def get_all_ids(self, where=None):
        if where and "group_id" in where:
            return {
                i for i, m in self._docs.items()
                if m.get("group_id") == where["group_id"]
            }
        return set(self._docs)

    def get_document_metadata(self, doc_id):
        return self._docs.get(doc_id)

    def iter_metadatas(self, batch_size=500):
        ids = list(self._docs)
        if ids:
            yield ids, [self._docs[i] for i in ids]

    def update_metadatas(self, ids, metadatas):
        for i, m in zip(ids, metadatas):
            self._docs.setdefault(i, {}).update(m)

    def upsert_documents(self, documents, metadatas, ids):
        for i, m in zip(ids, metadatas):
            self._docs[i] = dict(m)

    def add_documents(self, documents, metadatas, ids):
        self.upsert_documents(documents, metadatas, ids)

    def delete_documents(self, ids):
        self.deleted.extend(ids)
        for i in ids:
            self._docs.pop(i, None)

    def delete_item_chunks(self, item_key):
        self.chunk_deletes.append(item_key)
        for i in [d for d, m in self._docs.items() if m.get("parent_item_key") == item_key]:
            self._docs.pop(i, None)

    def reset_collection(self):
        self.reset_calls += 1
        self._docs = {}


def _write_config(tmp_path, last_sync_versions):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "semantic_search": {
            "embedding_model": "default",
            "update_config": {"auto_update": False, "update_frequency": "manual"},
            "include_fulltext": False,
            # Backfill gate closed: these tests isolate deletion behavior on
            # an already-attributed collection.
            "index_schema_version": semantic_search._INDEX_SCHEMA_VERSION,
            "last_sync_versions": last_sync_versions,
        }
    }))
    return str(config_path)


def _build_search(monkeypatch, zot, chroma, config_path):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: zot)
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    return semantic_search.ZoteroSemanticSearch(
        chroma_client=chroma, config_path=config_path
    )


def _personal_doc(key):
    return {"item_key": key, "title": key, "group_id": 0}


def _group_doc(key):
    return {"item_key": key, "title": key, "group_id": GROUP_ID}


# ---------------------------------------------------------------------------
# Scoping
# ---------------------------------------------------------------------------

def test_personal_sync_deletes_only_personal_docs(monkeypatch, tmp_path):
    """The scenario that wiped 738 live group docs: an incremental personal
    sync must clean the personal library's dead docs and NOTHING else."""
    chroma = RecordingChroma({
        "PERS_LIVE": _personal_doc("PERS_LIVE"),
        "PERS_DEAD": _personal_doc("PERS_DEAD"),
        "GRP_LIVE": _group_doc("GRP_LIVE"),
        "UNTAGGED": {"item_key": "UNTAGGED", "title": "no attribution"},
    })
    zot = FakeZoteroClient(versions_state={"PERS_LIVE": 9})
    config_path = _write_config(tmp_path, {"0": 5})
    search = _build_search(monkeypatch, zot, chroma, config_path)

    stats = search.update_database()

    assert stats["deleted_items"] == 1
    assert chroma.deleted == ["PERS_DEAD"]
    assert "GRP_LIVE" in chroma._docs, "another library's doc was deleted"
    assert "UNTAGGED" in chroma._docs, "an unattributed doc was deleted"


def test_group_sync_deletes_only_that_groups_docs(monkeypatch, tmp_path):
    chroma = RecordingChroma({
        "PERS_LIVE": _personal_doc("PERS_LIVE"),
        "GRP_LIVE": _group_doc("GRP_LIVE"),
        "GRP_DEAD": _group_doc("GRP_DEAD"),
    })
    zot = FakeZoteroClient(versions_state={"GRP_LIVE": 9})
    zot.library_id = str(GROUP_ID)
    zot.library_type = "groups"
    config_path = _write_config(tmp_path, {str(GROUP_ID): 5})
    zclient.set_active_library(str(GROUP_ID), "group")
    search = _build_search(monkeypatch, zot, chroma, config_path)

    stats = search.update_database()

    assert stats["deleted_items"] == 1
    assert chroma.deleted == ["GRP_DEAD"]
    assert "PERS_LIVE" in chroma._docs


def test_deletion_only_sync_still_cleans_up(monkeypatch, tmp_path):
    """A bumped library version with zero changed items (something was
    deleted) must still run the scoped deletion pass."""
    chroma = RecordingChroma({
        "PERS_LIVE": _personal_doc("PERS_LIVE"),
        "PERS_DEAD": _personal_doc("PERS_DEAD"),
    })
    zot = FakeZoteroClient(versions_state={"PERS_LIVE": 3})  # nothing changed since 5
    config_path = _write_config(tmp_path, {"0": 5})
    search = _build_search(monkeypatch, zot, chroma, config_path)

    stats = search.update_database()

    assert stats["deleted_items"] == 1
    assert chroma.deleted == ["PERS_DEAD"]


def test_chunked_collections_scope_deletion_the_same_way(monkeypatch, tmp_path):
    chroma = RecordingChroma({
        "PERS_LIVE#0": dict(_personal_doc("PERS_LIVE"), parent_item_key="PERS_LIVE"),
        "PERS_DEAD#0": dict(_personal_doc("PERS_DEAD"), parent_item_key="PERS_DEAD"),
        "PERS_DEAD#1": dict(_personal_doc("PERS_DEAD"), parent_item_key="PERS_DEAD"),
        "GRP_LIVE#0": dict(_group_doc("GRP_LIVE"), parent_item_key="GRP_LIVE"),
    })
    zot = FakeZoteroClient(versions_state={"PERS_LIVE": 9})
    config_path = _write_config(tmp_path, {"0": 5})
    search = _build_search(monkeypatch, zot, chroma, config_path)
    monkeypatch.setattr(
        type(search), "_chunking_enabled", property(lambda self: True)
    )

    search.update_database()

    # PERS_LIVE may also appear in chunk_deletes: chunked ingest clears an
    # item's old chunks before re-upserting it. The deletion-pass claims are
    # that the dead personal item is cleaned and the group's chunks are not.
    assert "PERS_DEAD" in chroma.chunk_deletes
    assert "GRP_LIVE" not in chroma.chunk_deletes
    assert "PERS_DEAD#0" not in chroma._docs
    assert "GRP_LIVE#0" in chroma._docs


# ---------------------------------------------------------------------------
# Failure modes must skip deletion, never wipe
# ---------------------------------------------------------------------------

def test_item_versions_failure_skips_deletion_instead_of_wiping(monkeypatch, tmp_path):
    """A transient API failure used to degrade to 'the library is empty',
    turning the deletion pass into a full wipe of everything in scope."""
    chroma = RecordingChroma({
        "PERS_LIVE": _personal_doc("PERS_LIVE"),
        "PERS_ALSO": _personal_doc("PERS_ALSO"),
    })
    zot = FakeZoteroClient(
        versions_state={"PERS_LIVE": 9},
        bare_versions_error=RuntimeError("503 from Zotero"),
    )
    config_path = _write_config(tmp_path, {"0": 5})
    search = _build_search(monkeypatch, zot, chroma, config_path)

    stats = search.update_database()

    assert chroma.deleted == []
    assert stats["deleted_items"] == 0
    assert stats["deletion_skipped_reason"] == "item_versions_unavailable"


def test_empty_item_versions_against_nonempty_store_skips_deletion(monkeypatch, tmp_path):
    """An HTTP-200-but-empty (or truncated-to-empty) item_versions() body is
    indistinguishable from an API fault. Wiping a small library this way
    slips under any count-based guard, so emptiness itself must not be
    treated as evidence of deletion."""
    chroma = RecordingChroma({
        "PERS_LIVE": _personal_doc("PERS_LIVE"),
        "PERS_ALSO": _personal_doc("PERS_ALSO"),
    })
    zot = FakeZoteroClient(versions_state={}, library_version=9)
    config_path = _write_config(tmp_path, {"0": 5})
    search = _build_search(monkeypatch, zot, chroma, config_path)

    stats = search.update_database()

    assert chroma.deleted == []
    assert stats["deleted_items"] == 0
    assert stats["deletion_skipped_reason"] == "empty_item_versions"


# ---------------------------------------------------------------------------
# Mass-deletion guard
# ---------------------------------------------------------------------------

def _many_personal_docs(n_live, n_dead):
    docs = {}
    live = [f"LIVE{i:03d}" for i in range(n_live)]
    dead = [f"DEAD{i:03d}" for i in range(n_dead)]
    for k in live + dead:
        docs[k] = _personal_doc(k)
    return docs, live, dead


def test_mass_deletion_is_guarded(monkeypatch, tmp_path):
    """Deleting most of a library in one sync is far more likely to be a
    truncated item_versions() response or a scoping regression than a real
    purge; require an explicit opt-in."""
    docs, live, dead = _many_personal_docs(n_live=5, n_dead=35)
    chroma = RecordingChroma(docs)
    zot = FakeZoteroClient(versions_state={k: 3 for k in live})
    config_path = _write_config(tmp_path, {"0": 5})
    search = _build_search(monkeypatch, zot, chroma, config_path)

    stats = search.update_database()

    assert chroma.deleted == []
    assert stats["deleted_items"] == 0
    assert stats["deletion_skipped_reason"] == "mass_deletion_guard"


def test_allow_mass_deletion_overrides_the_guard(monkeypatch, tmp_path):
    docs, live, dead = _many_personal_docs(n_live=5, n_dead=35)
    chroma = RecordingChroma(docs)
    zot = FakeZoteroClient(versions_state={k: 3 for k in live})
    config_path = _write_config(tmp_path, {"0": 5})
    search = _build_search(monkeypatch, zot, chroma, config_path)

    stats = search.update_database(allow_mass_deletion=True)

    assert sorted(chroma.deleted) == dead
    assert stats["deleted_items"] == 35
    assert "deletion_skipped_reason" not in stats


def test_allow_mass_deletion_also_accepts_an_emptied_library(monkeypatch, tmp_path):
    """The empty-item_versions skip protects against API faults; a user who
    really emptied their library clears it with the same explicit opt-in."""
    chroma = RecordingChroma({
        "PERS_A": _personal_doc("PERS_A"),
        "PERS_B": _personal_doc("PERS_B"),
    })
    zot = FakeZoteroClient(versions_state={}, library_version=9)
    config_path = _write_config(tmp_path, {"0": 5})
    search = _build_search(monkeypatch, zot, chroma, config_path)

    stats = search.update_database(allow_mass_deletion=True)

    assert sorted(chroma.deleted) == ["PERS_A", "PERS_B"]
    assert stats["deleted_items"] == 2


def test_small_scale_deletions_pass_the_guard(monkeypatch, tmp_path):
    """Ordinary cleanup — a few items below both thresholds — needs no flag."""
    docs, live, dead = _many_personal_docs(n_live=37, n_dead=3)
    chroma = RecordingChroma(docs)
    zot = FakeZoteroClient(versions_state={k: 3 for k in live})
    config_path = _write_config(tmp_path, {"0": 5})
    search = _build_search(monkeypatch, zot, chroma, config_path)

    stats = search.update_database()

    assert sorted(chroma.deleted) == dead
    assert stats["deleted_items"] == 3


def test_get_all_ids_returning_nothing_deletes_nothing(monkeypatch, tmp_path):
    """get_all_ids swallows backend errors into an empty set; the deletion
    pass must treat that as 'nothing eligible', not crash or over-delete."""
    chroma = RecordingChroma({"PERS_LIVE": _personal_doc("PERS_LIVE")})
    monkeypatch.setattr(chroma, "get_all_ids", lambda where=None: set())
    zot = FakeZoteroClient(versions_state={"PERS_LIVE": 9})
    config_path = _write_config(tmp_path, {"0": 5})
    search = _build_search(monkeypatch, zot, chroma, config_path)

    stats = search.update_database()

    assert chroma.deleted == []
    assert stats["deleted_items"] == 0
