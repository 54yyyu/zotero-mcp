"""`db-status` wording for chunking, and the warning for full text on a small model."""

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

    def read_error(self):

        return None


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
# db-status
# ---------------------------------------------------------------------------

def test_db_status_reports_chunking_enabled_on_either_path(monkeypatch, tmp_path):
    for batch in (True, False):
        search = _build_search(monkeypatch, tmp_path, batch=batch, chunking=True)

        status = search.get_database_status()

        assert status["chunking"] == {"enabled": True, "max_tokens": 8000}
        assert _format_chunking_status(status) == "enabled"


def test_db_status_reports_chunking_disabled(monkeypatch, tmp_path):
    search = _build_search(monkeypatch, tmp_path, batch=True, chunking=False)

    status = search.get_database_status()

    assert status["chunking"] == {"enabled": False, "max_tokens": 8000}
    assert _format_chunking_status(status) == "disabled (item-level indexing)"


# ---------------------------------------------------------------------------
# Fulltext without chunking on a small-context model
# ---------------------------------------------------------------------------

def _fit_search(monkeypatch, tmp_path, *, batch=False, chunking=False, max_tokens=256):
    search = _build_search(monkeypatch, tmp_path, batch=batch, chunking=chunking)
    search.chroma_client.embedding_max_tokens = max_tokens
    # Fulltext extraction needs the local sqlite reader; the warning comes first.
    monkeypatch.setattr(search, "_get_items_from_source", lambda *a, **k: [])
    return search


def test_fulltext_without_chunking_warns_when_model_limit_is_small(monkeypatch, tmp_path, capsys):
    search = _fit_search(monkeypatch, tmp_path)

    stats = search.update_database(extract_fulltext=True)

    assert stats["fulltext_truncated"] is True
    err = capsys.readouterr().err
    assert "only the first ~256 tokens of each item" in err
    assert "run update-db again" in err
    assert err.count("Full text is being indexed as one vector per item") == 1


def test_fulltext_warning_not_shown_without_fulltext_chunking_or_for_large_models(
    monkeypatch, tmp_path, capsys
):
    for kwargs, extract in [
        ({}, False),  # no --fulltext
        ({"chunking": True}, True),  # chunking on
        ({"max_tokens": 8000}, True),  # model limit is large
    ]:
        search = _fit_search(monkeypatch, tmp_path, **kwargs)
        stats = search.update_database(extract_fulltext=extract)
        assert "fulltext_truncated" not in stats
        assert "Full text is being indexed" not in capsys.readouterr().err


def test_db_status_notes_small_model_limit_when_chunking_is_off(monkeypatch, tmp_path):
    search = _fit_search(monkeypatch, tmp_path)

    status = search.get_database_status()

    assert "first ~256 tokens" in _format_chunking_status(status)
