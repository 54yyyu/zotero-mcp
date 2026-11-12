"""The local ``--fulltext`` scan writes the item-keyed shape (#826).

A regular item is one bibliographic record; every file the attachment
priority selects (and every standalone file) is an attachment record with
passages. Change detection is per record: ``date_modified`` for the item,
``file_stamp`` for a file. Legacy records (they carry ``title``) are left
alone until their item changes.

A real ChromaDB index in a temp directory and an embedding function that
records what it embeds; the Zotero side is a fake reader. HOME, Path.home
and the config path all point into tmp_path.
"""

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

from _scan_fakes import ScanReader, target  # noqa: E402
from chromadb import EmbeddingFunction  # noqa: E402

from zotero_mcp import semantic_search  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient  # noqa: E402
from zotero_mcp.local_db import LocalZoteroReader  # noqa: E402

NAME = "item_keyed_scan"
DATE = "2024-01-01 00:00:00"
TEXT = " ".join(f"Body sentence {i} of the paper." for i in range(100))


class RecordingEF(EmbeddingFunction):
    def __init__(self):
        self.embedded: list[str] = []

    def __call__(self, input):
        self.embedded.extend(input)
        return [[1.0, 0.0] for _ in input]

    def truncate(self, text, max_tokens):
        return text


class RecordingClient(ChromaClient):
    def _create_embedding_function(self):
        return RecordingEF()

    @property
    def embedded(self):
        return self.embedding_function.embedded


class Item:
    """A regular item as LocalZoteroReader.get_items_with_text returns it."""

    def __init__(self, key, item_id, title="A paper", date_modified=DATE):
        self.key = key
        self.item_id = item_id
        self.item_type = "journalArticle"
        self.title = title
        self.abstract = "An abstract."
        self.extra = ""
        self.doi = None
        self.notes = None
        self.creators = "Doe, Jane"
        self.date_added = DATE
        self.date_modified = date_modified


class LibReader(ScanReader):
    """A library: regular items, their selected files, standalone files."""

    def __init__(self, items, **kwargs):
        super().__init__(**kwargs)
        self.items = list(items)

    def all_keys(self):
        keys = {it.key for it in self.items}
        keys |= {t.key for targets in self.targets.values() for t in targets}
        keys |= {row[0] for row in self.standalone}
        return keys

    def get_all_item_keys(self):
        return self.all_keys()

    def get_key_group_map(self):
        return ({key: 0 for key in self.all_keys()}, set())

    def get_items_with_text(self, limit=None, include_fulltext=False, key_filter=None, collection_keys=None):
        return list(self.items)

    def extract_targets(self, targets):
        for t in targets:
            self.extracted.append(t.key)
            yield t.key, self.texts.get(t.key, (TEXT, "pdf"))


class FakeZotero:
    def __init__(self, reader):
        self.reader = reader

    def item_versions(self, since=None, **kwargs):
        # Zotero still has files the priority no longer selects.
        return {key: 5 for key in self.reader.all_keys() | getattr(self.reader, "unselected", set())}

    def last_modified_version(self, **kwargs):
        return 5


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "semantic_search": {
                    "index_schema_version": semantic_search._INDEX_SCHEMA_VERSION,
                    "chunking": {"enabled": True, "chunk_size": 400, "overlap": 50, "max_chunks_per_item": 5},
                }
            }
        )
    )

    def run(reader, update=True, **kwargs):
        monkeypatch.setattr(semantic_search, "LocalZoteroReader", lambda *a, **kw: reader)
        monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: FakeZotero(reader))
        client = RecordingClient(collection_name=NAME, persist_directory=str(tmp_path / "chroma"))
        search = semantic_search.ZoteroSemanticSearch(chroma_client=client, config_path=str(config_path))
        if not update:
            return search, None
        reader.extracted.clear()
        stats = search.update_database(extract_fulltext=True, **kwargs)
        assert stats.get("error") is None, stats
        return search, stats

    return run


def _ids(search, name=NAME):
    return set(search.chroma_client.client.get_collection(name).get(include=[])["ids"])


def _all_ids(search):
    return _ids(search) | _ids(search, f"{NAME}_passages")


def _meta(search, doc_id):
    return search.chroma_client.get_document_metadata(doc_id)


# --- selection ------------------------------------------------------------------


def _make_db(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT, libraryID INT, dateModified TEXT);
        CREATE TABLE itemAttachments (itemID INTEGER PRIMARY KEY, parentItemID INT, path TEXT, contentType TEXT);
        CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY);
        """
    )
    conn.execute("INSERT INTO items VALUES (1, 'ITEM0001', 1, ?)", (DATE,))
    rows = [
        (10, "PDFAAAAA", "storage:a.pdf", "application/pdf"),
        (11, "PDFBBBBB", "storage:b.pdf", "application/pdf"),
        (12, "SNAPSHOT", "storage:s.html", "text/html"),
    ]
    for item_id, key, path, ctype in rows:
        conn.execute("INSERT INTO items VALUES (?, ?, 1, ?)", (item_id, key, DATE))
        conn.execute("INSERT INTO itemAttachments VALUES (?, 1, ?, ?)", (item_id, path, ctype))
    conn.commit()
    conn.close()


def test_every_pdf_is_indexed_under_its_own_key_and_the_snapshot_is_skipped(env, tmp_path):
    db = tmp_path / "zotero.sqlite"
    _make_db(db)
    storage = tmp_path / "storage"
    for key, name in (("PDFAAAAA", "a.pdf"), ("PDFBBBBB", "b.pdf"), ("SNAPSHOT", "s.html")):
        (storage / key).mkdir(parents=True)
        (storage / key / name).write_bytes(b"file bytes")
    real = LocalZoteroReader(db_path=str(db))
    real._get_storage_dir = lambda: storage
    reader = LibReader([Item("ITEM0001", 1)], texts={"PDFAAAAA": (TEXT, "pdf"), "PDFBBBBB": (TEXT, "pdf")})
    reader._resolve_extraction_targets = real._resolve_extraction_targets
    try:
        search, _ = env(reader)
    finally:
        real.close()

    assert _ids(search) == {"ITEM0001", "PDFAAAAA", "PDFBBBBB"}
    passages = _ids(search, f"{NAME}_passages")
    assert {p.split("#")[0] for p in passages} == {"PDFAAAAA", "PDFBBBBB"}
    assert len(passages) == 10
    assert sorted(reader.extracted) == ["PDFAAAAA", "PDFBBBBB"]
    meta = _meta(search, "PDFAAAAA")
    assert set(meta) == {"item_key", "group_id", "date_modified", "file_stamp", "file_hash"}
    assert meta["file_hash"]  # the real file, so a real hash
    assert set(_meta(search, "ITEM0001")) == {"item_key", "group_id", "date_modified"}


def test_a_standalone_pdf_is_indexed_and_records_nothing_of_a_parent(env):
    reader = LibReader(
        [],
        standalone=[("LONEPDF1", "storage:x.pdf", "application/pdf", DATE, 1)],
        standalone_targets={"LONEPDF1": target("LONEPDF1")},
    )

    search, _ = env(reader)

    assert _ids(search) == {"LONEPDF1"}
    assert len(_ids(search, f"{NAME}_passages")) == 5
    passage = search.chroma_client.client.get_collection(f"{NAME}_passages").get(
        ids=["LONEPDF1#0"], include=["metadatas"]
    )["metadatas"][0]
    assert passage["item_key"] == "LONEPDF1"
    assert "parent_item_key" not in passage


# --- change detection -------------------------------------------------------------


def _library(**kwargs):
    return LibReader([Item("ITEM0001", 1)], targets={1: [target("PDFAAAAA")]}, **kwargs)


def test_a_title_edit_re_embeds_only_the_bibliographic_record(env):
    reader = _library()
    env(reader)

    reader.items = [Item("ITEM0001", 1, title="A better title", date_modified="2024-03-01 00:00:00")]
    search, _ = env(reader)

    assert len(search.chroma_client.embedded) == 1
    assert "A better title" in search.chroma_client.embedded[0]
    assert reader.extracted == [], "the file is not parsed again"


def test_a_second_run_writes_nothing(env):
    reader = _library()
    env(reader)

    search, stats = env(reader)

    assert search.chroma_client.embedded == []
    assert reader.extracted == []
    assert stats["total_items"] == 0


def test_a_new_pdf_on_an_unchanged_item_embeds_only_that_file(env):
    reader = _library()
    env(reader)

    reader.targets[1] = [target("PDFAAAAA"), target("PDFBBBBB")]
    search, _ = env(reader)

    assert reader.extracted == ["PDFBBBBB"]
    assert len(search.chroma_client.embedded) == 6, "its record and five passages"
    assert "PDFBBBBB" in _ids(search)


def test_a_replaced_file_is_re_embedded_by_its_stamp(env):
    reader = _library()
    env(reader)

    reader.targets[1] = [target("PDFAAAAA", stamp="2:200")]
    search, _ = env(reader)

    assert reader.extracted == ["PDFAAAAA"]
    assert _meta(search, "PDFAAAAA")["file_stamp"] == "2:200"
    assert len(search.chroma_client.embedded) == 6


def test_a_failed_extraction_waits_for_the_file_to_change(env):
    reader = _library(texts={"PDFAAAAA": None})
    search, _ = env(reader)
    assert reader.extracted == ["PDFAAAAA"]
    assert "PDFAAAAA" not in _ids(search)

    env(reader)
    assert reader.extracted == [], "the same file is not parsed again"

    reader.targets[1] = [target("PDFAAAAA", stamp="2:200")]
    reader.texts = {}
    search, _ = env(reader)
    assert reader.extracted == ["PDFAAAAA"]
    assert "PDFAAAAA" in _ids(search)


# --- legacy records -------------------------------------------------------------


def _seed_legacy(search, date_modified=DATE):
    """Records as 0.17 wrote them: the item's text under its own key, with its metadata."""
    meta = {
        "item_key": "ITEM0001",
        "title": "A paper",
        "item_type": "journalArticle",
        "group_id": 0,
        "date_modified": date_modified,
        "has_fulltext": True,
        "attachment_keys": "PDFAAAAA",
        "attachment_priority": "pdf,html,other",
    }
    ids = ["ITEM0001", *(f"ITEM0001#{n}" for n in range(3))]
    metas = [meta, *({**meta, "parent_item_key": "ITEM0001", "chunk_index": n, "n_chunks": 3} for n in range(3))]
    search.chroma_client.upsert_embeddings(
        documents=[f"legacy {i}" for i in ids], metadatas=metas, ids=ids, embeddings=[[0.0, 1.0]] * len(ids)
    )


def test_an_unchanged_legacy_item_writes_nothing(env):
    reader = _library()
    search, _ = env(reader, update=False)
    _seed_legacy(search)

    search, _ = env(reader)

    assert search.chroma_client.embedded == []
    assert reader.extracted == []
    assert _all_ids(search) == {"ITEM0001", "ITEM0001#0", "ITEM0001#1", "ITEM0001#2"}
    assert "title" in _meta(search, "ITEM0001")


def test_a_changed_legacy_item_is_rewritten_and_its_old_passages_dropped_after(env, monkeypatch):
    reader = _library()
    search, _ = env(reader, update=False)
    _seed_legacy(search, date_modified="2023-01-01 00:00:00")

    events: list[tuple[str, str]] = []
    real_upsert = ChromaClient.upsert_documents
    real_delete = ChromaClient.delete_item_chunks

    def upsert(self, documents, metadatas, ids):
        events.extend(("upsert", doc_id) for doc_id in ids)
        return real_upsert(self, documents, metadatas, ids)

    def delete(self, item_key, group_id=None):
        events.append(("delete", item_key))
        return real_delete(self, item_key, group_id=group_id)

    monkeypatch.setattr(ChromaClient, "upsert_documents", upsert)
    monkeypatch.setattr(ChromaClient, "delete_item_chunks", delete)
    search, _ = env(reader)

    assert _ids(search) == {"ITEM0001", "PDFAAAAA"}
    assert {p.split("#")[0] for p in _ids(search, f"{NAME}_passages")} == {"PDFAAAAA"}
    # Written over the legacy record, and none of its fields left behind.
    assert set(_meta(search, "ITEM0001")) == {"item_key", "group_id", "date_modified"}
    assert events.index(("delete", "ITEM0001")) > events.index(("upsert", "ITEM0001"))

    search, _ = env(reader)
    assert search.chroma_client.embedded == [], "the rewritten item is current from now on"


# --- the deletion pass ------------------------------------------------------------


def test_an_outranked_file_loses_its_records(env):
    """A snapshot indexed while it was the best file goes once a PDF outranks it."""
    reader = LibReader([Item("ITEM0001", 1)], targets={1: [target("SNAPSHOT", category="html")]})
    env(reader)

    reader.targets[1] = [target("PDFAAAAA")]
    reader.unselected = {"SNAPSHOT"}
    search, stats = env(reader)

    assert _ids(search) == {"ITEM0001", "PDFAAAAA"}
    assert {p.split("#")[0] for p in _ids(search, f"{NAME}_passages")} == {"PDFAAAAA"}
    assert stats["deleted_items"] == 1


def test_a_trashed_file_loses_its_records_and_its_item_keeps_its_own(env):
    reader = _library()
    env(reader)

    reader.targets[1] = []
    reader.unselected = {"PDFAAAAA"}
    search, _ = env(reader)

    assert _all_ids(search) == {"ITEM0001"}


def test_a_file_that_now_fails_loses_the_records_of_its_old_version(env):
    reader = _library()
    env(reader)

    reader.targets[1] = [target("PDFAAAAA", stamp="2:200")]
    reader.texts = {"PDFAAAAA": None}
    search, _ = env(reader)

    assert _all_ids(search) == {"ITEM0001"}


def test_legacy_passages_stay_while_their_item_exists(env):
    reader = _library()
    search, _ = env(reader, update=False)
    _seed_legacy(search)

    search, stats = env(reader)

    assert stats.get("deleted_items", 0) == 0
    assert _all_ids(search) == {"ITEM0001", "ITEM0001#0", "ITEM0001#1", "ITEM0001#2"}


def test_the_mass_deletion_guard_still_trips(env):
    items = [Item(f"ITEM{n:04d}", n) for n in range(30)]
    reader = LibReader(items, targets={n: [target(f"PDF{n:05d}")] for n in range(30)})
    env(reader)

    reader.unselected = {f"PDF{n:05d}" for n in range(30)}
    reader.targets = {}
    search, stats = env(reader)

    assert stats["deletion_skipped_reason"] == "mass_deletion_guard"
    assert {f"PDF{n:05d}" for n in range(30)} <= _ids(search)
