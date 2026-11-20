"""``migrate-index`` re-keys legacy passages to their attachment (#826).

Real persisted indexes in temp directories with given vectors; any request for
a new embedding fails the test. HOME, Path.home and the config path point into
tmp_path, and the Zotero side is a fake reader.
"""

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
MINIMAL = {"item_key", "group_id", "chunk_index", "n_chunks", "char_start", "char_end", "parent_hint"}


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

    def __init__(self, items=(), ft_caches=None, **kwargs):
        super().__init__(**kwargs)
        self.items = {it.key: it for it in items}
        self.ft_caches = dict(ft_caches or {})  # attachment key -> Zotero's .zotero-ft-cache text

    def _read_zotero_ft_cache(self, attachment_key):
        return self.ft_caches.get(attachment_key)

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
            # The legacy key was the parent's: it becomes the passage's hint (#826).
            "parent_hint": ITEM,
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


def test_rekeying_does_not_hash_the_file(index, tmp_path):
    """Hashing waits for the first update that needs it; migrate-index reads no file bytes."""
    directory, config = index
    _seed_split(directory)
    path = tmp_path / "file.pdf"
    path.write_bytes(b"the file")
    live = ExtractionTarget(
        key=ATT, path=path, content_type="application/pdf", category="pdf", file_stamp="1:8", size=8
    )

    migrate_index(config, sources=_sources(config, targets=[live]), **QUIET)

    path.unlink()
    assert file_hash.get_file_hash(ATT, path, "1:8", config_path=config) is None, "nothing was cached, and the file is gone"


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
        raise KeyboardInterrupt("interrupted")  # a kill is no Exception: a failing item does not hide it

    monkeypatch.setattr(index_migration, "_after_rekey_upsert", stop)
    with pytest.raises(KeyboardInterrupt, match="interrupted"):
        migrate_index(config, sources=_sources(config), **QUIET)
    both = {f"{ITEM}#{n}" for n in range(3)} | {f"{ATT}#{n}" for n in range(3)}
    assert set(_read(directory, PASSAGES)) == both
    monkeypatch.setattr(index_migration, "_after_rekey_upsert", real_hook)

    report = migrate_index(config, sources=_sources(config), **QUIET)

    stored = _read(directory, PASSAGES)
    assert set(stored) == {f"{ATT}#{n}" for n in range(3)}
    assert stored[f"{ATT}#0"][2] == pytest.approx(vectors[f"{ITEM}#0"], abs=1e-5)
    assert report.rekeyed_items == 1
    # Written over the ids the first run left, and still nothing but the minimal set.
    for n, (_doc, meta, _vector) in enumerate(stored[f"{ATT}#{k}"] for k in range(3)):
        assert set(meta) == MINIMAL | ({"page"} if n == 1 else {"section"} if n == 2 else set())
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


def test_dry_run_creates_no_cache_file(index, tmp_path):
    directory, config = index
    _seed_split(directory)
    path = tmp_path / "file.pdf"
    path.write_bytes(b"the file")
    live = ExtractionTarget(
        key=ATT, path=path, content_type="application/pdf", category="pdf", file_stamp="1:8", size=8
    )
    before = _all_files(tmp_path)

    migrate_index(config, dry_run=True, sources=_sources(config, targets=[live]), **QUIET)

    assert _all_files(tmp_path) == before, "no hash or fulltext cache file, nor anything else, was written"


def test_a_local_record_without_a_stored_fingerprint_is_rekeyed_when_one_file_is_selected(index):
    directory, config = index
    _seed_split(directory, att_keys=None)

    report = migrate_index(config, sources=_sources(config), **QUIET)

    assert (report.rekeyed_items, report.dropped_items) == (1, 0)
    assert set(_read(directory, PASSAGES)) == {f"{ATT}#{n}" for n in range(3)}


def test_a_local_record_without_a_stored_fingerprint_and_two_files_is_left_as_it_is(index):
    """The update path keeps such an item as legacy; dropping it would cost a paid re-embed."""
    directory, config = index
    _seed_split(directory, att_keys=None)
    reader = MigReader([_item()], targets={1: [target(ATT), target("ATTACH02")]})
    sources = LocalSources(reader, config)

    dry = migrate_index(config, dry_run=True, sources=sources, **QUIET)
    assert (dry.skipped_items, dry.skipped_passages, dry.dropped_items, dry.rekeyed_items) == (1, 3, 0, 0)

    report = migrate_index(config, sources=sources, **QUIET)

    assert (report.rekeyed_items, report.dropped_items, report.skipped_items) == (0, 0, 1)
    assert set(_read(directory, PASSAGES)) == {f"{ITEM}#{n}" for n in range(3)}


class BrokenReader(MigReader):
    def get_item_by_key(self, key):
        raise RuntimeError("database is locked")


@pytest.mark.parametrize("why", ["no reader", "reader raises"])
def test_nothing_is_dropped_while_zotero_cannot_be_asked(index, why):
    directory, config = index
    vectors = _seed_split(directory)
    sources = LocalSources(None if why == "no reader" else BrokenReader([_item()]), config)
    before = _all_files(directory)

    dry = migrate_index(config, dry_run=True, sources=sources, **QUIET)
    assert (dry.skipped_items, dry.skipped_passages, dry.dropped_items, dry.rekeyed_items) == (1, 3, 0, 0)
    assert _all_files(directory) == before

    report = migrate_index(config, sources=sources, **QUIET)

    assert (report.skipped_items, report.skipped_passages, report.dropped_items) == (1, 3, 0)
    stored = _read(directory, PASSAGES)
    assert set(stored) == {f"{ITEM}#{n}" for n in range(3)}
    assert stored[f"{ITEM}#0"][2] == pytest.approx(vectors[f"{ITEM}#0"], abs=1e-5)
    assert "title" in stored[f"{ITEM}#0"][1], "left exactly as it was"
    assert read_index_layout(directory, NAME)["legacy_passages"] == 3


def test_the_source_is_read_from_a_surviving_passage_when_the_first_is_gone(index):
    directory, config = index
    _seed_split(directory, source=f"web-api:attachment:{ATT}", att_keys=None)
    ChromaClient(collection_name=NAME, persist_directory=directory).delete_records([f"{ITEM}#0"])

    report = migrate_index(config, sources=LocalSources(None, config), **QUIET)

    assert (report.rekeyed_items, report.dropped_items, report.skipped_items) == (1, 0, 0)
    assert set(_read(directory, PASSAGES)) == {f"{ATT}#1", f"{ATT}#2"}


def test_dry_run_of_an_unmigrated_index_changes_nothing(index):
    directory, config = index
    legacy = ChromaClient(collection_name=NAME, persist_directory=directory)
    legacy.route_passages = False
    _seed_split(directory, client=legacy)
    before = _all_files(directory)

    report = migrate_index(config, dry_run=True, sources=_sources(config), **QUIET)

    assert report.rekeyed_items == 1
    assert _all_files(directory) == before


def _searcher(config, directory, monkeypatch):
    from zotero_mcp import semantic_search

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: None)
    cfg = json.loads(Path(config).read_text())
    cfg["semantic_search"]["chunking"] = {"enabled": True}
    Path(config).write_text(json.dumps(cfg))
    client = ChromaClient(collection_name=NAME, persist_directory=directory)
    return semantic_search.ZoteroSemanticSearch(chroma_client=client, config_path=config)


def test_article_records_the_migration_leaves_missing_wait_for_an_explicit_update(index, monkeypatch):
    directory, config = index
    legacy = ChromaClient(collection_name=NAME, persist_directory=directory)
    legacy.route_passages = False
    _seed_split(directory, client=legacy, bare=False)
    migrate_index(config, sources=_sources(config), **QUIET)
    search = _searcher(config, directory, monkeypatch)

    automatic = search._find_index_gaps(limit=None, force_rebuild=False, fill_passages=False)
    explicit = search._find_index_gaps(limit=None, force_rebuild=False, fill_passages=True)

    assert automatic == {}
    assert explicit == {ATT: {"articles"}}


def test_an_item_the_0_17_split_left_without_an_article_is_still_filled_automatically(index, monkeypatch):
    directory, config = index
    _seed_split(directory, bare=False)
    search = _searcher(config, directory, monkeypatch)

    gaps = search._find_index_gaps(limit=None, force_rebuild=False, fill_passages=False)

    assert gaps == {ITEM: {"articles"}}


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


# --- a file out of reach is no evidence about the item (final review C1) ------------

PDF_ROW = [ATT, "storage:a.pdf", "application/pdf"]
OTHER_ROW = ["ATTACH02", "storage:b.pdf", "application/pdf"]


def _unreachable(index, rows, targets, **seed):
    directory, config = index
    _seed_split(directory, **seed)
    reader = MigReader([_item()], targets={1: targets}, rows={1: rows})
    return directory, config, LocalSources(reader, config)


def test_a_file_missing_on_disk_leaves_the_passages_where_they_are(index):
    """An unmounted drive, a file kept on demand, an EPUB: the database still lists the attachment."""
    directory, config, sources = _unreachable(index, [PDF_ROW], [])

    report = migrate_index(config, sources=sources, **QUIET)

    assert (report.skipped_items, report.skipped_passages, report.dropped_items) == (1, 3, 0)
    assert set(_read(directory, PASSAGES)) == {f"{ITEM}#{n}" for n in range(3)}


def test_one_of_two_selected_files_out_of_reach_leaves_the_passages_where_they_are(index):
    directory, config, sources = _unreachable(
        index, [PDF_ROW, OTHER_ROW], [target(ATT)], att_keys="ATTACH01,ATTACH02"
    )

    report = migrate_index(config, sources=sources, **QUIET)

    assert (report.skipped_items, report.dropped_items, report.rekeyed_items) == (1, 0, 0)
    assert len(_read(directory, PASSAGES)) == 3


def test_an_item_without_any_attachment_still_loses_its_passages(index):
    directory, config, sources = _unreachable(index, [], [], att_keys="")

    report = migrate_index(config, sources=sources, **QUIET)

    assert (report.dropped_items, report.skipped_items) == (1, 0)


def _sized(key, size):
    t = target(key)
    return ExtractionTarget(
        key=key, path=t.path, content_type=t.content_type, category=t.category, file_stamp=t.file_stamp, size=size
    )


def test_a_legacy_item_with_two_files_is_rekeyed_to_the_larger_when_nothing_changed(index):
    """Before #826 the source was deterministic: the first category's largest file."""
    directory, config = index
    _seed_split(directory, att_keys="ATTACH01,ATTACH02")
    reader = MigReader([_item()], targets={1: [_sized(ATT, 100), _sized("ATTACH02", 900)]})

    report = migrate_index(config, sources=LocalSources(reader, config), **QUIET)

    assert (report.rekeyed_items, report.dropped_items) == (1, 0)
    assert set(_read(directory, PASSAGES)) == {f"ATTACH02#{n}" for n in range(3)}


def test_a_legacy_item_with_a_file_known_only_by_its_cache_is_not_rekeyed_by_size(index):
    """The cache's size says nothing about the file's, so the largest cannot be told (fix round 2)."""
    directory, config = index
    _seed_split(directory, att_keys="ATTACH01,ATTACH02")
    on_disk = _sized(ATT, 100)
    cached = ExtractionTarget(
        key="ATTACH02", path=on_disk.path, content_type=on_disk.content_type, category=on_disk.category,
        file_stamp="ftcache:1:900", size=900, ft_cache=True,
    )
    reader = MigReader([_item()], targets={1: [on_disk, cached]})

    report = migrate_index(config, sources=LocalSources(reader, config), **QUIET)

    assert (report.rekeyed_items, report.dropped_items, report.skipped_items) == (0, 0, 1)
    assert len(_read(directory, PASSAGES)) == 3


def test_a_legacy_item_with_two_files_and_a_changed_fingerprint_is_dropped(index):
    directory, config = index
    _seed_split(directory, att_keys="ATTACH01")
    reader = MigReader([_item()], targets={1: [_sized(ATT, 100), _sized("ATTACH02", 900)]})

    report = migrate_index(config, sources=LocalSources(reader, config), **QUIET)

    assert (report.rekeyed_items, report.dropped_items) == (0, 1)


# --- the text 0.17 read from Zotero's cache names its own attachment (review of #840, item 5b) ---

def test_a_zotero_cache_item_is_filed_under_the_attachment_whose_cache_was_read(index):
    """0.17 read the first listed attachment with a cache, which can be a sibling of the largest file."""
    directory, config = index
    _seed_split(directory, source="zotero-cache", att_keys="ATTACH01,ATTACH02")
    reader = MigReader(
        [_item()],
        targets={1: [_sized("ATTACH02", 100), _sized(ATT, 900)]},  # listed order: the small sibling first
        ft_caches={"ATTACH02": "cached text", ATT: "the large file's own cache"},
    )

    report = migrate_index(config, sources=LocalSources(reader, config), **QUIET)

    assert (report.rekeyed_items, report.dropped_items, report.skipped_items) == (1, 0, 0)
    assert set(_read(directory, PASSAGES)) == {f"ATTACH02#{n}" for n in range(3)}


def test_a_zotero_cache_item_whose_cache_belongs_to_a_file_the_scan_does_not_index_is_left_alone(index):
    directory, config = index
    _seed_split(directory, source="zotero-cache", att_keys="ATTACH01,ATTACH02")
    snapshot = ["ATTACH02", "storage:s.html", "text/html"]
    reader = MigReader(
        [_item()],
        targets={1: [_sized(ATT, 900)]},
        rows={1: [snapshot, PDF_ROW]},
        ft_caches={"ATTACH02": "the snapshot's cache"},
    )

    report = migrate_index(config, sources=LocalSources(reader, config), **QUIET)

    assert (report.rekeyed_items, report.dropped_items, report.skipped_items) == (0, 0, 1)
    assert set(_read(directory, PASSAGES)) == {f"{ITEM}#{n}" for n in range(3)}


def test_a_zotero_cache_item_with_one_file_is_filed_under_it(index):
    directory, config = index
    _seed_split(directory, source="zotero-cache")
    reader = MigReader([_item()], targets={1: [target(ATT)]}, ft_caches={ATT: "cached text"})

    report = migrate_index(config, sources=LocalSources(reader, config), **QUIET)

    assert report.rekeyed_items == 1
    assert set(_read(directory, PASSAGES)) == {f"{ATT}#{n}" for n in range(3)}


# --- a mark that did not land must not cost the old passages (review of #840, item 5c) ---

def _two_items(directory):
    """Two legacy items, each with its own attachment (ITEM0001 -> ATTACH01, ITEM0002 -> ATTACH02)."""
    _seed_split(directory)
    _seed_split(directory, key="ITEM0002", att_keys="ATTACH02")
    second = SimpleNamespace(**{**vars(_item("ITEM0002", 2))})
    return MigReader([_item(), second], targets={1: [target(ATT)], 2: [target("ATTACH02")]})


@pytest.mark.parametrize("how", ["silent", "raises"])
def test_old_passages_stay_when_the_rekeyed_to_mark_does_not_land(index, monkeypatch, how):
    directory, config = index
    reader = _two_items(directory)
    real = ChromaClient.update_metadatas

    def update(self, ids, metadatas):
        if ids == [ITEM]:
            if how == "raises":
                raise RuntimeError("disk full")
            return  # what a swallowed error looks like to the caller
        return real(self, ids, metadatas)

    monkeypatch.setattr(ChromaClient, "update_metadatas", update)

    report = migrate_index(config, sources=LocalSources(reader, config), **QUIET)

    stored = _read(directory, PASSAGES)
    # The failed item keeps its old passages, so an update does not embed it again ...
    assert {f"{ITEM}#{n}" for n in range(3)} <= set(stored)
    assert "rekeyed_to" not in _read(directory, NAME)[ITEM][1]
    # ... and the migration goes on with the next item, reporting the failure.
    assert {f"ITEM0002#{n}" for n in range(3)}.isdisjoint(stored)
    assert {f"ATTACH02#{n}" for n in range(3)} <= set(stored)
    assert (report.failed_items, report.rekeyed_items, report.dropped_items) == (1, 1, 0)

    monkeypatch.setattr(ChromaClient, "update_metadatas", real)
    again = migrate_index(config, sources=LocalSources(reader, config), **QUIET)
    assert (again.rekeyed_items, again.failed_items) == (1, 0)
    assert set(_read(directory, PASSAGES)) == {f"{att}#{n}" for att in (ATT, "ATTACH02") for n in range(3)}
    assert _read(directory, NAME)[ITEM][1]["rekeyed_to"] == ATT


def test_an_item_without_an_article_record_is_rekeyed_and_its_old_passages_go(index):
    """A passages-only 0.16 item has no record to mark; nothing waits on a mark."""
    directory, config = index
    _seed_split(directory, bare=False)

    report = migrate_index(config, sources=_sources(config), **QUIET)

    assert (report.rekeyed_items, report.failed_items) == (1, 0)
    assert set(_read(directory, PASSAGES)) == {f"{ATT}#{n}" for n in range(3)}


# --- an update never drops; only migrate-index does (review of #840, item 6) ---

def _rekey(directory, config, sources, **kwargs):
    client = ChromaClient(collection_name=NAME, persist_directory=directory)
    report = index_migration.MigrationReport(directory, dry_run=False)
    index_migration.rekey_legacy_passages(
        directory, NAME, client, sources, report, lambda _m: None, config_path=config, **kwargs
    )
    return report


def test_without_drop_an_item_with_no_file_is_held_and_with_drop_it_is_dropped(index):
    directory, config = index
    _seed_split(directory)
    sources = LocalSources(MigReader([], targets={}), config)  # the item is gone from Zotero

    held = _rekey(directory, config, sources, drop=False)

    assert (held.held_items, held.held_passages, held.dropped_items, held.rekeyed_items) == (1, 3, 0, 0)
    assert set(_read(directory, PASSAGES)) == {f"{ITEM}#{n}" for n in range(3)}

    dropped = _rekey(directory, config, sources)

    assert (dropped.dropped_items, dropped.held_items) == (1, 0)
    assert _read(directory, PASSAGES) == {}


def test_without_drop_what_can_be_placed_is_still_rekeyed(index):
    directory, config = index
    _seed_split(directory)

    report = _rekey(directory, config, _sources(config), drop=False)

    assert (report.rekeyed_items, report.held_items) == (1, 0)
    assert set(_read(directory, PASSAGES)) == {f"{ATT}#{n}" for n in range(3)}


def test_the_notes_say_who_drops():
    assert "migrate-index" in index_migration.held_note(2, 40)
    assert "40" in index_migration.held_note(2, 40)
    assert "only migrate-index drops" in " ".join(index_migration.rekey_note(2, 40))


def test_the_mcp_update_tool_passes_the_note_on(monkeypatch, tmp_path):
    from zotero_mcp import semantic_search
    from zotero_mcp.tools import search as search_tools

    class FakeSearch:
        def update_database(self, **kwargs):
            assert kwargs.get("automatic", False) is False
            return {"processed_items": 0, "passages_rekeyed": {"held_items": 2, "note": index_migration.held_note(2, 40)}}

    monkeypatch.setattr(semantic_search, "create_semantic_search", lambda _config: FakeSearch())
    ctx = SimpleNamespace(info=lambda _m: None, error=lambda _m: None)

    out = search_tools.update_search_database.__wrapped__(ctx=ctx) if hasattr(
        search_tools.update_search_database, "__wrapped__"
    ) else search_tools.update_search_database(ctx=ctx)

    assert "migrate-index" in out
