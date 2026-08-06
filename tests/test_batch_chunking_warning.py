"""`semantic_search.chunking` is ignored on the Batch API path (issue #416).

`_prepare_index_records`, the record builder behind
`_submit_openai_batch_index`, never reads the chunking config and truncates
each document at the embedding limit. `_process_item_batch` on the realtime
path does the opposite. So with `openai_batch.enabled: true`, turning
`chunking.enabled` on changed nothing and said nothing: the reporter found it
only by diffing the generated `batch-001-input.jsonl` by hand and measuring 30
of 58 documents truncated.

Making the batch path chunk-aware is tracked separately. These tests cover the
signal: an `update-db` run that takes the batch path with chunking requested
must say so once, and `db-status` must report the effective behaviour rather
than the requested setting.
"""

import json
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")

from zotero_mcp import semantic_search  # noqa: E402
from zotero_mcp.cli import _format_chunking_status  # noqa: E402


class FakeZoteroClient:
    """Minimal pyzotero double: one item, one library version."""

    def __init__(self):
        self.versions_state = {"ITEM0001": 9}

    def items(self, start=0, limit=100, **kwargs):
        if start:
            return []
        return [self.item(k) for k in self.versions_state]

    def item_versions(self, since=None, **kwargs):
        if since is None:
            return dict(self.versions_state)
        return {k: v for k, v in self.versions_state.items() if v > since}

    def item(self, key):
        return {
            "key": key,
            "version": self.versions_state.get(key, 1),
            "data": {
                "key": key,
                "itemType": "journalArticle",
                "title": f"Paper {key}",
                "abstractNote": "An abstract long enough to embed.",
                "creators": [],
                "dateAdded": "2024-01-01T00:00:00Z",
                "dateModified": "2024-01-01T00:00:00Z",
            },
        }

    def children(self, *a, **kw):
        return []

    def last_modified_version(self, **kwargs):
        return 9


class FakeChroma:
    def __init__(self, embedding_model="openai"):
        self.embedding_model = embedding_model
        self.embedding_config = {"model_name": "text-embedding-3-small", "api_key": "test"}
        self.embedding_max_tokens = 8000
        self._docs = {}

    def truncate_text(self, text, max_tokens=None):
        return text

    def get_existing_ids(self, ids):
        return {i for i in ids if i in self._docs}

    def get_all_ids(self, where=None):
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
        for i in ids:
            self._docs.pop(i, None)

    def delete_item_chunks(self, item_key, group_id=None):
        pass

    def reset_collection(self):
        self._docs = {}

    def get_collection_info(self):
        return {"name": "zotero", "count": len(self._docs), "embedding_model": self.embedding_model}


def _write_config(tmp_path, *, batch, chunking):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "semantic_search": {
            "embedding_model": "openai",
            "update_config": {"auto_update": False, "update_frequency": "manual"},
            "include_fulltext": False,
            "index_schema_version": semantic_search._INDEX_SCHEMA_VERSION,
            "openai_batch": {"enabled": batch},
            "chunking": {
                "enabled": chunking,
                "chunk_size": 1500,
                "overlap": 200,
                "max_chunks_per_item": 20,
            },
        }
    }))
    return str(config_path)


def _build_search(monkeypatch, tmp_path, *, batch, chunking, embedding_model="openai"):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: FakeZoteroClient())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    monkeypatch.setattr(
        semantic_search.openai_batch,
        "submit_embedding_batches",
        lambda **kwargs: {
            "run_id": "run-1",
            "manifest_path": "/tmp/manifest.json",
            "batches": [{"batch_id": "batch-1"}],
        },
    )
    return semantic_search.ZoteroSemanticSearch(
        chroma_client=FakeChroma(embedding_model),
        config_path=_write_config(tmp_path, batch=batch, chunking=chunking),
    )


# ---------------------------------------------------------------------------
# update-db warning
# ---------------------------------------------------------------------------

def test_batch_run_with_chunking_requested_warns_on_stderr(monkeypatch, tmp_path, capsys):
    search = _build_search(monkeypatch, tmp_path, batch=True, chunking=True)

    stats = search.update_database()

    assert stats["batch_submitted"] is True
    assert stats["chunking_ignored"] is True
    err = capsys.readouterr().err
    assert "Passage chunking is NOT applied on the OpenAI Batch API path" in err
    # The user has to be told what to do about it, not only that it happened.
    assert "--no-openai-batch" in err
    # Warned exactly once per run.
    assert err.count("Passage chunking is NOT applied") == 1


def test_batch_run_without_chunking_stays_silent(monkeypatch, tmp_path, capsys):
    search = _build_search(monkeypatch, tmp_path, batch=True, chunking=False)

    stats = search.update_database()

    assert stats["batch_submitted"] is True
    assert "chunking_ignored" not in stats
    assert "Passage chunking" not in capsys.readouterr().err


def test_realtime_run_with_chunking_stays_silent(monkeypatch, tmp_path, capsys):
    """Chunking works on the realtime path, so there is nothing to warn about."""
    search = _build_search(monkeypatch, tmp_path, batch=False, chunking=True)

    stats = search.update_database()

    assert not stats.get("batch_submitted")
    assert "chunking_ignored" not in stats
    assert "Passage chunking" not in capsys.readouterr().err


def test_no_warning_when_batch_is_configured_but_inapplicable(monkeypatch, tmp_path, capsys):
    """`openai_batch.enabled` with a non-OpenAI model never takes the batch
    path, so chunking still applies and the warning would be wrong."""
    search = _build_search(
        monkeypatch, tmp_path, batch=True, chunking=True, embedding_model="gemini"
    )

    stats = search.update_database()

    assert not stats.get("batch_submitted")
    assert "chunking_ignored" not in stats
    assert "Passage chunking" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# db-status reports effective behaviour
# ---------------------------------------------------------------------------

def test_db_status_reports_chunking_as_not_applied_on_batch_path(monkeypatch, tmp_path):
    search = _build_search(monkeypatch, tmp_path, batch=True, chunking=True)

    status = search.get_database_status()

    assert status["chunking"] == {"enabled": True, "effective": False}
    assert "NOT applied" in _format_chunking_status(status)


def test_db_status_reports_chunking_as_effective_off_the_batch_path(monkeypatch, tmp_path):
    search = _build_search(monkeypatch, tmp_path, batch=False, chunking=True)

    status = search.get_database_status()

    assert status["chunking"] == {"enabled": True, "effective": True}
    assert _format_chunking_status(status) == "enabled"


def test_db_status_reports_chunking_disabled(monkeypatch, tmp_path):
    search = _build_search(monkeypatch, tmp_path, batch=True, chunking=False)

    status = search.get_database_status()

    assert status["chunking"] == {"enabled": False, "effective": False}
    assert _format_chunking_status(status) == "disabled (item-level indexing)"
