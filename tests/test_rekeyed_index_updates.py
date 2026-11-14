"""Updates of an index the migration re-keyed never re-embed what was paid for (#826, final review C2/C3/I5).

A pre-0.17 chunked index (passages ``KEY#n`` in the article collection, nothing
else) is split and re-keyed by ``migrate_index``; its attachments then have
passages and no attachment record. An automatic update holds them; an explicit
one writes only the missing attachment record. Real Chroma in a temp dir, a
recording embedding function that counts what is embedded, and a fake Zotero
reader. HOME, Path.home and the config path point into tmp_path.
"""

import contextlib
import importlib.util
import json
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

from _scan_fakes import target  # noqa: E402
from chromadb.api.models.CollectionCommon import CollectionCommon  # noqa: E402
from test_item_keyed_local_scan import DATE, NAME, FakeZotero, Item, LibReader  # noqa: E402

from zotero_mcp import index_migration, semantic_search  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient, read_index_layout  # noqa: E402
from zotero_mcp.index_migration import LocalSources, migrate_index  # noqa: E402
from zotero_mcp.local_db import ExtractionTarget  # noqa: E402

ITEM = "ITEM0001"
ATT = "PDFAAAAA"
PASSAGES = f"{NAME}_passages"
QUIET = {"progress": lambda _m: None}


class Reader(LibReader):
    def get_item_by_key(self, key):
        return next((item for item in self.items if item.key == key), None)


class World:
    def __init__(self, monkeypatch, tmp_path):
        self.monkeypatch = monkeypatch
        # The only embedder: it records what it is asked for, and no provider is reachable.
        self.embedded: list[str] = []

        def embed(collection, input, **kwargs):
            self.embedded.extend(input)
            return [[1.0, 0.0] for _ in input]

        monkeypatch.setattr(CollectionCommon, "_embed", embed)
        monkeypatch.setattr(ChromaClient, "truncate_text", lambda self, text, max_tokens=None: text)
        self.directory = str(tmp_path / "chroma")
        self.config = str(tmp_path / "config.json")
        Path(self.config).write_text(
            json.dumps(
                {
                    "semantic_search": {
                        "persist_directory": self.directory,
                        "collection_name": NAME,
                        "index_schema_version": semantic_search._INDEX_SCHEMA_VERSION,
                        "chunking": {"enabled": True, "chunk_size": 400, "overlap": 50, "max_chunks_per_item": 5},
                    }
                }
            )
        )

    def seed_pre_0_17(self, key=ITEM, att_keys=ATT, date_modified=DATE, bare=False):
        """A chunked index as 0.16 wrote it: passages in the article collection, with their item's metadata."""
        client = ChromaClient(collection_name=NAME, persist_directory=self.directory)
        client.route_passages = False
        meta = {
            "item_key": key,
            "title": "A paper",
            "item_type": "journalArticle",
            "group_id": 0,
            "date_modified": date_modified,
            "has_fulltext": True,
            "fulltext_source": "pdf",
            "attachment_keys": att_keys,
            "attachment_priority": "pdf,html,other",
        }
        ids = ([key] if bare else []) + [f"{key}#{n}" for n in range(3)]
        metas = ([meta] if bare else []) + [
            {
                **meta,
                "parent_item_key": key,
                "chunk_index": n,
                "n_chunks": 3,
                "char_start": n * 50,
                "char_end": n * 50 + 90,
            }
            for n in range(3)
        ]
        client.upsert_embeddings(
            documents=[f"legacy {i}" for i in ids],
            metadatas=metas,
            ids=ids,
            embeddings=[[0.5, float(n)] for n, _ in enumerate(ids)],
        )

    def migrate(self, reader):
        return migrate_index(self.config, sources=LocalSources(reader, self.config), **QUIET)

    def search(self, reader, as_migration_source=True):
        self.monkeypatch.setattr(semantic_search, "LocalZoteroReader", lambda *a, **kw: reader)
        self.monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: FakeZotero(reader))
        if as_migration_source:

            @contextlib.contextmanager
            def sources(config_path, db_path=None):
                yield LocalSources(reader, config_path)

            self.monkeypatch.setattr(index_migration, "open_sources", sources)
        client = ChromaClient(collection_name=NAME, persist_directory=self.directory)
        return semantic_search.ZoteroSemanticSearch(chroma_client=client, config_path=self.config)

    def update(self, reader, **kwargs):
        search = self.search(reader)
        reader.extracted.clear()
        stats = search.update_database(extract_fulltext=True, **kwargs)
        assert stats.get("error") is None, stats
        return search, stats

    def stored(self, name):
        client = ChromaClient(collection_name=NAME, persist_directory=self.directory)
        got = client.client.get_collection(name).get(include=["embeddings", "metadatas"])
        return {
            doc_id: (meta, [float(x) for x in emb])
            for doc_id, meta, emb in zip(got["ids"], got["metadatas"], got["embeddings"])
        }


@pytest.fixture()
def world(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    return World(monkeypatch, tmp_path)


def _reader(**kwargs):
    return Reader([Item(ITEM, 1)], targets={1: [target(ATT)]}, **kwargs)


def _passage_vectors(world):
    return {doc_id: vec for doc_id, (_m, vec) in world.stored(PASSAGES).items()}


def test_automatic_update_of_a_migrated_pre_0_17_index_leaves_every_passage_alone(world):
    reader = _reader()
    world.seed_pre_0_17()
    world.migrate(reader)
    before = _passage_vectors(world)
    assert set(before) == {f"{ATT}#{n}" for n in range(3)}

    _search, _stats = world.update(reader, automatic=True)

    assert reader.extracted == [], "nothing is parsed for a file whose passages are held"
    assert _passage_vectors(world) == before
    assert not any(text.startswith(("Body sentence", "extracted")) for text in world.embedded)
    assert ATT not in world.stored(NAME), "no attachment record is embedded by an automatic run"


def test_explicit_update_of_an_unmigrated_pre_0_17_index_splits_it_and_rekeys_without_embedding_a_passage(world):
    reader = _reader()
    world.seed_pre_0_17()
    old = world.stored(NAME)
    assert set(old) == {f"{ITEM}#{n}" for n in range(3)}, "the passages are in the article collection"

    _search, _stats = world.update(reader)

    layout = read_index_layout(world.directory, NAME)
    assert layout["unmigrated_passages"] == 0, "the index is split"
    assert set(world.stored(PASSAGES)) == {f"{ATT}#{n}" for n in range(3)}
    vectors = {doc_id: vec for doc_id, (_m, vec) in world.stored(PASSAGES).items()}
    assert vectors == {f"{ATT}#{n}": old[f"{ITEM}#{n}"][1] for n in range(3)}, "the paid vectors, re-keyed"
    assert set(world.stored(NAME)) == {ITEM, ATT}, "an article record each for the item and the file"
    assert len(world.embedded) == 2, "article records only"


def test_a_file_that_yields_no_text_for_its_missing_record_keeps_its_passages(world):
    reader = _reader(texts={ATT: None})
    world.seed_pre_0_17()
    world.migrate(reader)
    before = _passage_vectors(world)

    world.update(reader)

    assert _passage_vectors(world) == before, "a transient failure does not cost the paid passages"
    assert ATT not in world.stored(NAME)


def test_a_file_changed_since_the_migration_is_a_real_change(world, tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"the file")
    reader = _reader()
    reader.targets[1] = [_live(path, "1:8")]
    world.seed_pre_0_17()
    world.migrate(reader)  # records the file's stamp "1:8"
    before = _passage_vectors(world)
    path.write_bytes(b"the file, edited")
    reader.targets[1] = [_live(path, "2:16")]

    world.update(reader)

    after = _passage_vectors(world)
    assert set(after) == {f"{ATT}#{n}" for n in range(5)}
    assert all(after[f"{ATT}#{n}"] != before[f"{ATT}#{n}"] for n in range(3)), "its passages are cut again"
    assert len(world.embedded) == 7


def _live(path, stamp):
    return ExtractionTarget(
        key=ATT, path=path, content_type="application/pdf", category="pdf", file_stamp=stamp, size=8
    )


def test_explicit_update_after_the_migration_writes_only_the_attachment_record(world):
    reader = _reader()
    world.seed_pre_0_17()
    world.migrate(reader)
    before = _passage_vectors(world)

    _search, _stats = world.update(reader)

    assert _passage_vectors(world) == before, "passages: same ids, same vectors"
    assert ATT in world.stored(NAME), "the missing attachment record is written"
    assert reader.extracted == [ATT], "its text is read once, for that record"
    assert len(world.embedded) == 2, "the attachment record and the item's bibliographic record"


def _edited_reader():
    """The item was edited in Zotero after it was indexed (a tag, say): its date_modified moved."""
    reader = Reader([Item(ITEM, 1, date_modified="2025-05-05 00:00:00")], targets={1: [target(ATT)]})
    return reader


@pytest.mark.parametrize("automatic", [True, False])
def test_a_metadata_edit_of_a_migrated_item_leaves_its_rekeyed_passages_alone(world, automatic):
    reader = _edited_reader()
    world.seed_pre_0_17(bare=True)
    world.migrate(reader)
    before = _passage_vectors(world)

    world.update(reader, automatic=automatic)

    assert _passage_vectors(world) == before, "no passage is cut or embedded again"
    assert ITEM in world.stored(NAME) and "title" not in world.stored(NAME)[ITEM][0], "the item is rewritten"
    assert (ATT in world.stored(NAME)) is (not automatic), "an explicit run adds the attachment record"
    assert len(world.embedded) == (1 if automatic else 2)
