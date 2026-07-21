"""Tests for optional multicore PDF extraction (extraction_workers > 1).

Verifies: workers=1 keeps identical (sequential-equivalent) results, workers>1
processes every item without duplication/loss, the circuit breaker still
trips under concurrency, and LocalZoteroReader's sqlite connection tolerates
being used from multiple threads.
"""

import sys
import threading

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")

from zotero_mcp import semantic_search
from zotero_mcp.local_db import _EXTRACTION_TIMEOUT


class FakeItem:
    def __init__(self, n):
        self.item_id = n
        self.key = f"ITEM{n:04d}"
        self.item_type = "journalArticle"
        self.title = f"Paper {n}"
        self.abstract = ""
        self.extra = ""
        self.doi = None
        self.notes = None
        self.creators = None
        self.date_added = "2026-01-01 00:00:00"
        self.date_modified = "2026-01-01 00:00:00"
        self.fulltext = None
        self.fulltext_source = None


class FakeReader:
    """Minimal LocalZoteroReader stand-in with a fake per-item extraction result."""

    def __init__(self, n_items, results, extract_lock=None):
        self._items = [FakeItem(i) for i in range(n_items)]
        self._results = results  # dict item_id -> extract_fulltext_for_item() return
        self._extract_lock = extract_lock
        self.calls = []
        self._calls_lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_all_item_keys(self):
        return {it.key for it in self._items}

    def get_items_with_text(self, limit=None, include_fulltext=False, key_filter=None, collection_keys=None):
        return list(self._items)

    def get_fulltext_meta_for_item(self, item_id):
        return []

    def extract_fulltext_for_item(self, item_id, item_key=None):
        with self._calls_lock:
            self.calls.append(item_id)
        if self._extract_lock:
            with self._extract_lock:
                pass
        return self._results[item_id]


class FakeChromaClient:
    def __init__(self):
        self.embedding_max_tokens = 8000

    def get_document_metadata(self, key):
        return None

    def get_existing_ids(self, ids):
        return set()

    def upsert_documents(self, documents, metadatas, ids):
        pass

    def reset_collection(self):
        pass


def _run_scan(monkeypatch, reader, config_path=None):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    monkeypatch.setattr(semantic_search, "LocalZoteroReader", lambda *a, **kw: reader)

    chroma = FakeChromaClient()
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma, config_path=config_path)
    monkeypatch.setattr(
        search, "_get_items_from_api", lambda *a, **kw: pytest.fail("unexpected fallback to API path")
    )
    return search._get_items_from_source(
        extract_fulltext=True, chroma_client=chroma, force_rebuild=False, extraction_workers=None
    )


def test_workers_one_matches_sequential_baseline(monkeypatch):
    n = 10
    results = {i: (f"text {i}", "pdf", False) for i in range(n)}
    reader = FakeReader(n, results)
    items = _run_scan(monkeypatch, reader)

    assert len(items) == n
    assert sorted(reader.calls) == list(range(n))
    got = {item["key"]: item["data"]["fulltext"] for item in items}
    assert got == {f"ITEM{i:04d}": f"text {i}" for i in range(n)}


def test_workers_four_processes_all_items_without_duplication(monkeypatch, tmp_path):
    n = 20
    results = {i: (f"text {i}", "pdf", False) for i in range(n)}
    reader = FakeReader(n, results)
    config_path = tmp_path / "config.json"
    import json
    config_path.write_text(json.dumps({"semantic_search": {"extraction": {"workers": 4}}}))

    items = _run_scan(monkeypatch, reader, config_path=str(config_path))

    assert len(items) == n
    assert sorted(reader.calls) == list(range(n))  # every item extracted exactly once
    got = {item["key"] for item in items}
    assert got == {f"ITEM{i:04d}" for i in range(n)}


def test_circuit_breaker_trips_under_concurrency(monkeypatch, tmp_path):
    n = 15
    # All extractions time out.
    results = {i: (_EXTRACTION_TIMEOUT, "timeout") for i in range(n)}
    reader = FakeReader(n, results)
    config_path = tmp_path / "config.json"
    import json
    config_path.write_text(json.dumps({"semantic_search": {"extraction": {"workers": 3}}}))

    items = _run_scan(monkeypatch, reader, config_path=str(config_path))

    # Every item is still returned (indexed metadata-only), but none has fulltext.
    assert len(items) == n
    assert all(not item["data"]["fulltext"] for item in items)
    assert all(item["data"]["fulltext_attempted"] for item in items)


def test_threaded_sqlite_reads_do_not_raise(tmp_path):
    """LocalZoteroReader's connection must tolerate concurrent threads."""
    from concurrent.futures import ThreadPoolExecutor

    from zotero_mcp.local_db import LocalZoteroReader

    db_path = tmp_path / "zotero.sqlite"
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT)")
    for i in range(30):
        conn.execute("INSERT INTO items (key) VALUES (?)", (f"K{i}",))
    conn.commit()
    conn.close()

    reader = LocalZoteroReader(db_path=str(db_path), extraction_workers=4)

    def read_keys(_):
        return reader.get_all_item_keys()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(read_keys, range(8)))

    for keys in results:
        assert len(keys) == 30
    reader.close()
