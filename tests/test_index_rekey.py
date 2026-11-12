"""``migrate-index`` re-keys legacy passages to their attachment (#826).

Real persisted indexes in temp directories with given vectors; any request for
a new embedding fails the test. HOME, Path.home and the config path point into
tmp_path, and the Zotero side is a fake reader.
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

import numpy as np  # noqa: E402
from _scan_fakes import ScanReader, target  # noqa: E402
from chromadb.api.models.CollectionCommon import CollectionCommon  # noqa: E402

from zotero_mcp import file_hash, index_migration  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient, read_index_layout  # noqa: E402
from zotero_mcp.index_migration import LocalSources, file_text_offset, migrate_index  # noqa: E402
from zotero_mcp.local_db import ExtractionTarget  # noqa: E402
from zotero_mcp.utils import format_creators  # noqa: E402

NAME = "zotero_library"
PASSAGES = "zotero_library_passages"
QUIET = {"progress": lambda _m: None}

ITEM = "ITEM0001"
ATT = "ATTACH01"
TITLE = "A paper"
ABSTRACT = "An abstract."
PREFIX = f"{TITLE} {format_creators([{'creatorType': 'author', 'firstName': 'Jane', 'lastName': 'Doe'}])} {ABSTRACT}"
BODY = " ".join(f"Sentence {i} of the file." for i in range(12))
DOC = PREFIX + "\n\n" + BODY
SPANS = [(0, 120), (100, 220), (200, len(DOC))]


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Nothing reaches the real home, and migrating moves stored vectors: embedding fails the test."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

    def refuse(self, *args, **kwargs):
        raise AssertionError("the migration asked for an embedding")

    monkeypatch.setattr(CollectionCommon, "_embed", refuse)


@pytest.fixture()
def index(tmp_path):
    directory = tmp_path / "chroma"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "semantic_search": {
                    "persist_directory": str(directory),
                    "embedding_model": "default",
                }
            }
        )
    )
    return str(directory), str(config)


class MigReader(ScanReader):
    """The Zotero side of the migration: regular items, and the files each one has."""

    def __init__(self, items=(), **kwargs):
        super().__init__(**kwargs)
        self.items = {it.key: it for it in items}

    def get_item_by_key(self, key):
        return self.items.get(key)


def _item(key=ITEM, item_id=1):
    return SimpleNamespace(
        key=key, item_id=item_id, title=TITLE, abstract=ABSTRACT, creators="Doe, Jane", item_type="journalArticle"
    )


def _legacy_meta(key=ITEM, source="pdf", att_keys=ATT, priority="pdf,html,other"):
    meta = {
        "item_key": key,
        "item_type": "journalArticle",
        "title": TITLE,
        "date": "",
        "date_added": "2024-01-01",
        "date_modified": "2024-01-01",
        "creators": "Doe, Jane",
        "publication": "",
        "url": "",
        "doi": "",
        "group_id": 0,
        "has_fulltext": True,
        "fulltext_source": source,
        "attachment_priority": priority,
        "tags": "",
        "citation_key": "",
    }
    if att_keys is not None:
        meta["attachment_keys"] = att_keys
    return meta


def _legacy_records(key=ITEM, bare=True, **meta_args):
    """Records as 0.17 wrote them: the item's text under its own key, then its passages."""
    meta = _legacy_meta(key, **meta_args)
    ids, metas, docs = [], [], []
    if bare:
        ids.append(key)
        metas.append(meta)
        docs.append(DOC[:200])
    for n, (start, end) in enumerate(SPANS):
        ids.append(f"{key}#{n}")
        extra = {"page": 2} if n == 1 else {"section": "Intro"} if n == 2 else {}
        metas.append(
            {
                **meta,
                "parent_item_key": key,
                "chunk_index": n,
                "n_chunks": len(SPANS),
                "char_start": start,
                "char_end": end,
                **extra,
            }
        )
        docs.append(DOC[start:end])
    return ids, metas, docs


def _vectors(ids):
    rng = np.random.default_rng(3)
    return {doc_id: rng.normal(size=4).astype(float).tolist() for doc_id in ids}


def _seed_split(directory, client=None, **kwargs):
    ids, metas, docs = _legacy_records(**kwargs)
    vectors = _vectors(ids)
    client = client or ChromaClient(collection_name=NAME, persist_directory=directory)
    client.upsert_embeddings(documents=docs, metadatas=metas, ids=ids, embeddings=[vectors[i] for i in ids])
    return vectors


def _read(directory, name):
    client = ChromaClient(collection_name=NAME, persist_directory=directory)
    got = client.client.get_collection(name).get(include=["embeddings", "documents", "metadatas"])
    return {
        doc_id: (doc, meta, [float(x) for x in emb])
        for doc_id, doc, meta, emb in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"])
    }


def _sources(config, reader=None, targets=None):
    reader = reader if reader is not None else MigReader([_item()], targets={1: targets or [target(ATT)]})
    return LocalSources(reader, config)


def _all_files(directory):
    return {str(p): p.stat().st_mtime_ns for p in Path(directory).rglob("*") if p.is_file()}


def test_a_split_index_is_rekeyed_to_the_attachment_without_touching_a_vector(index):
    directory, config = index
    vectors = _seed_split(directory)

    report = migrate_index(config, sources=_sources(config), **QUIET)

    stored = _read(directory, PASSAGES)
    assert set(stored) == {f"{ATT}#{n}" for n in range(3)}
    offset = len(PREFIX) + 2
    for n, (start, end) in enumerate(SPANS):
        doc, meta, vector = stored[f"{ATT}#{n}"]
        assert doc == DOC[start:end]
        assert vector == pytest.approx(vectors[f"{ITEM}#{n}"], abs=1e-5)
        expected = {
            "item_key": ATT,
            "group_id": 0,
            "chunk_index": n,
            "n_chunks": 3,
            "char_start": max(0, start - offset),
            "char_end": end - offset,
        }
        if n == 1:
            expected["page"] = 2
        if n == 2:
            expected["section"] = "Intro"
        assert meta == expected
    assert (report.legacy_items, report.legacy_passages) == (1, 3)
    assert (report.rekeyed_items, report.rekeyed_passages, report.dropped_items) == (1, 3, 0)
    # The item's own article record stays, with a note that its passages moved.
    bare = _read(directory, NAME)[ITEM][1]
    assert bare["title"] == TITLE
    assert bare["rekeyed_to"] == ATT
    assert read_index_layout(directory, NAME)["legacy_passages"] == 0


def test_rekeying_keeps_the_files_hash_in_the_hash_cache(index, tmp_path):
    directory, config = index
    _seed_split(directory)
    path = tmp_path / "file.pdf"
    path.write_bytes(b"the file")
    live = ExtractionTarget(
        key=ATT, path=path, content_type="application/pdf", category="pdf", file_stamp="1:8", size=8
    )

    migrate_index(config, sources=_sources(config, targets=[live]), **QUIET)

    path.unlink()
    assert file_hash.get_file_hash(ATT, path, "1:8", config_path=config) == hashlib.sha256(b"the file").hexdigest()


def test_pre_0_17_chunked_index_is_split_and_rekeyed_in_one_run(index):
    directory, config = index
    legacy = ChromaClient(collection_name=NAME, persist_directory=directory)
    legacy.route_passages = False
    _seed_split(directory, client=legacy, bare=False)

    report = migrate_index(config, sources=_sources(config), **QUIET)

    assert report.renamed is True
    assert set(_read(directory, PASSAGES)) == {f"{ATT}#{n}" for n in range(3)}
    assert _read(directory, NAME) == {}
    layout = read_index_layout(directory, NAME)
    assert layout["unmigrated_passages"] == 0
    assert layout["legacy_passages"] == 0


def test_pre_0_17_unchunked_index_has_nothing_to_rekey(index):
    directory, config = index
    ids, metas, docs = _legacy_records()
    ChromaClient(collection_name=NAME, persist_directory=directory).upsert_embeddings(
        documents=docs[:1], metadatas=metas[:1], ids=ids[:1], embeddings=[_vectors(ids)[ITEM]]
    )

    report = migrate_index(config, **QUIET)

    assert report.needed is False
    assert report.legacy_passages == 0
    assert set(_read(directory, NAME)) == {ITEM}
    assert "rekeyed_to" not in _read(directory, NAME)[ITEM][1]


@pytest.mark.parametrize("why", ["two files", "fingerprint changed", "priority changed", "item gone"])
def test_an_item_whose_attachment_is_uncertain_loses_its_passages(index, why):
    directory, config = index
    _seed_split(directory)
    reader = MigReader([_item()], targets={1: [target(ATT), target("ATTACH02")]})
    if why == "fingerprint changed":
        reader = MigReader([_item()], targets={1: [target("ATTACH02")]})
    elif why == "priority changed":
        reader = MigReader([_item()], targets={1: [target(ATT)]})
        reader.attachment_priority = ("html", "pdf", "other")
    elif why == "item gone":
        reader = MigReader([], targets={})

    report = migrate_index(config, sources=LocalSources(reader, config), **QUIET)

    assert (report.rekeyed_items, report.dropped_items, report.dropped_passages) == (0, 1, 3)
    assert _read(directory, PASSAGES) == {}
    # The article vector, paid for, stays; with its passages gone the next update embeds them again.
    assert set(_read(directory, NAME)) == {ITEM}
    assert "rekeyed_to" not in _read(directory, NAME)[ITEM][1]


def test_the_web_api_source_names_the_attachment(index):
    directory, config = index
    _seed_split(directory, source=f"web-api:attachment:{ATT}", att_keys=None)

    # No Zotero database at all: only the stored source can say.
    report = migrate_index(config, sources=LocalSources(None, config), **QUIET)

    assert report.rekeyed_items == 1
    assert set(_read(directory, PASSAGES)) == {f"{ATT}#{n}" for n in range(3)}


def test_a_run_that_stopped_after_writing_the_new_ids_finishes_on_the_next(index, monkeypatch):
    directory, config = index
    vectors = _seed_split(directory)
    real_hook = index_migration._after_rekey_upsert

    def stop():
        raise RuntimeError("interrupted")

    monkeypatch.setattr(index_migration, "_after_rekey_upsert", stop)
    with pytest.raises(RuntimeError, match="interrupted"):
        migrate_index(config, sources=_sources(config), **QUIET)
    both = {f"{ITEM}#{n}" for n in range(3)} | {f"{ATT}#{n}" for n in range(3)}
    assert set(_read(directory, PASSAGES)) == both
    monkeypatch.setattr(index_migration, "_after_rekey_upsert", real_hook)

    report = migrate_index(config, sources=_sources(config), **QUIET)

    stored = _read(directory, PASSAGES)
    assert set(stored) == {f"{ATT}#{n}" for n in range(3)}
    assert stored[f"{ATT}#0"][2] == pytest.approx(vectors[f"{ITEM}#0"], abs=1e-5)
    assert report.rekeyed_items == 1
    assert migrate_index(config, **QUIET).needed is False


def test_dry_run_reports_both_steps_and_changes_nothing(index):
    directory, config = index
    _seed_split(directory)
    _seed_split(directory, key="ITEM0002", att_keys="ATTACH09")
    before = _all_files(directory)

    report = migrate_index(config, dry_run=True, sources=_sources(config), **QUIET)

    assert (report.legacy_items, report.legacy_passages) == (2, 6)
    assert (report.rekeyed_items, report.rekeyed_passages) == (1, 3)
    assert (report.dropped_items, report.dropped_passages) == (1, 3)
    assert _all_files(directory) == before
    assert len(_read(directory, PASSAGES)) == 6


def test_dry_run_of_an_unmigrated_index_changes_nothing(index):
    directory, config = index
    legacy = ChromaClient(collection_name=NAME, persist_directory=directory)
    legacy.route_passages = False
    _seed_split(directory, client=legacy)
    before = _all_files(directory)

    report = migrate_index(config, dry_run=True, sources=_sources(config), **QUIET)

    assert report.rekeyed_items == 1
    assert _all_files(directory) == before


def test_the_layout_counts_legacy_passages(index):
    directory, _ = index
    _seed_split(directory)
    layout = read_index_layout(directory, NAME)
    assert (layout["legacy_passages"], layout["legacy_passage_items"]) == (3, 1)


def test_offsets_come_from_the_live_metadata_text_or_from_what_chunk_0_stored():
    first = _legacy_meta()
    chunk_0 = DOC[:120]
    assert file_text_offset(chunk_0, first, PREFIX) == len(PREFIX) + 2
    # Zotero's record changed since: the stored title and creators still find where the text starts.
    assert file_text_offset(chunk_0, first, "Another title") == len(PREFIX) + 2
    assert file_text_offset(chunk_0, first, None) == len(PREFIX) + 2
    assert file_text_offset("no metadata prefix here", {}, None) == 0
    assert file_text_offset(BODY, first, "") == 0


def test_the_note_names_the_command():
    lines = index_migration.rekey_note(2, 40)
    assert "40" in lines[0] and "2" in lines[0]
    assert any("migrate-index" in line for line in lines)
    assert index_migration.rekey_note(0, 0) == []
