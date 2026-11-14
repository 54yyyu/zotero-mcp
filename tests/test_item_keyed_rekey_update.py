"""``update-db`` re-keys legacy passages first; automatic updates never do (#826).

The harness is the local-scan tests': a real ChromaDB index in a temp directory,
an embedding function that records what it embeds, and a fake Zotero reader.
"""

import contextlib
import importlib.util
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

from _scan_fakes import target  # noqa: E402
from test_item_keyed_local_scan import (  # noqa: E402
    DATE,
    Item,
    LibReader,
    _all_ids,
    _ids,
    _library,
    _meta,
    env,  # noqa: F401  (the fixture)
)

from zotero_mcp import index_migration  # noqa: E402

PREFIX = "A paper Doe, Jane An abstract."
DOC = PREFIX + "\n\n" + "Body sentence of the paper. " * 20
SPANS = [(0, 150), (130, 280), (260, 400)]


def _seed_legacy(search, att_keys="PDFAAAAA"):
    """Records as 0.17 wrote them for a chunked index: the item, then its passages."""
    meta = {
        "item_key": "ITEM0001",
        "title": "A paper",
        "item_type": "journalArticle",
        "group_id": 0,
        "date_modified": DATE,
        "has_fulltext": True,
        "fulltext_source": "pdf",
        "attachment_keys": att_keys,
        "attachment_priority": "pdf,html,other",
        "creators": "Doe, Jane",
        "publication": "",
        "tags": "",
    }
    ids = ["ITEM0001", *(f"ITEM0001#{n}" for n in range(3))]
    metas = [
        meta,
        *(
            {
                **meta,
                "parent_item_key": "ITEM0001",
                "chunk_index": n,
                "n_chunks": 3,
                "char_start": start,
                "char_end": end,
            }
            for n, (start, end) in enumerate(SPANS)
        ),
    ]
    docs = [DOC[:150], *(DOC[start:end] for start, end in SPANS)]
    search.chroma_client.upsert_embeddings(documents=docs, metadatas=metas, ids=ids, embeddings=[[0.0, 1.0]] * len(ids))


@pytest.fixture()
def zotero(monkeypatch, tmp_path):
    """Makes the migration's sources the fake reader (a real one would open the user's Zotero)."""

    def use(reader):
        reader.get_item_by_key = lambda key: next((it for it in reader.items if it.key == key), None)

        @contextlib.contextmanager
        def open_sources(config_path, db_path=None):
            yield index_migration.LocalSources(reader, config_path)

        monkeypatch.setattr(index_migration, "open_sources", open_sources)

    return use


def test_an_explicit_update_rekeys_the_passages_and_embeds_nothing(env, zotero):  # noqa: F811
    reader = _library()
    zotero(reader)
    search, _ = env(reader, update=False)
    _seed_legacy(search)

    search, stats = env(reader)

    assert search.chroma_client.embedded == []
    assert reader.extracted == []
    assert stats["passages_rekeyed"]["items"] == 1
    assert _ids(search) == {"ITEM0001"}
    assert _all_ids(search) == {"ITEM0001", "PDFAAAAA#0", "PDFAAAAA#1", "PDFAAAAA#2"}
    assert "title" in _meta(search, "ITEM0001"), "the unchanged legacy item keeps its record"

    # And it stays that way: the moved passages are not a gap, and not a change.
    search, stats = env(reader)
    assert search.chroma_client.embedded == []
    assert _all_ids(search) == {"ITEM0001", "PDFAAAAA#0", "PDFAAAAA#1", "PDFAAAAA#2"}


def test_an_explicit_update_drops_nothing_while_zotero_cannot_be_asked(env, monkeypatch):  # noqa: F811
    @contextlib.contextmanager
    def no_zotero(config_path, db_path=None):
        yield index_migration.LocalSources(None, config_path)

    monkeypatch.setattr(index_migration, "open_sources", no_zotero)
    reader = _library()
    search, _ = env(reader, update=False)
    _seed_legacy(search)

    search, stats = env(reader)

    assert stats["passages_rekeyed"]["skipped_items"] == 1
    assert stats["passages_rekeyed"]["dropped_items"] == 0
    assert {"ITEM0001#0", "ITEM0001#1", "ITEM0001#2"} <= _all_ids(search)
    assert search.chroma_client.embedded == []


def test_an_automatic_update_leaves_legacy_passages_where_they_are(env, zotero):  # noqa: F811
    reader = _library()
    zotero(reader)
    search, _ = env(reader, update=False)
    _seed_legacy(search)

    search, stats = env(reader, automatic=True)

    assert "passages_rekeyed" not in stats
    assert search.chroma_client.embedded == []
    assert _all_ids(search) == {"ITEM0001", "ITEM0001#0", "ITEM0001#1", "ITEM0001#2"}


def test_an_item_whose_passages_were_dropped_is_written_again_by_the_next_update(env, zotero):  # noqa: F811
    # A second PDF was attached since the passages were written: which one the old text came from cannot be told.
    reader = LibReader([Item("ITEM0001", 1)], targets={1: [target("PDFAAAAA"), target("PDFBBBBB")]})
    zotero(reader)
    search, _ = env(reader, update=False)
    _seed_legacy(search, att_keys="PDFAAAAA")

    search, stats = env(reader)

    assert stats["passages_rekeyed"]["dropped_items"] == 1
    assert not any(i.startswith("ITEM0001#") for i in _all_ids(search))
    assert search.chroma_client.embedded, "the dropped passages are embedded again, not skipped as unchanged"
    # One file is indexed per item: the largest of the first category (the first of equals).
    assert "PDFAAAAA" in _ids(search) and "PDFBBBBB" not in _ids(search)
    assert "title" not in _meta(search, "ITEM0001"), "the item is written in the new shape"
