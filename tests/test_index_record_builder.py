"""The one place an item becomes index records, for the realtime and batch paths."""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")

from zotero_mcp import semantic_search  # noqa: E402

LONG_TEXT = " ".join(f"Sentence number {i} of the full text." for i in range(200))


class _FakeChroma:
    """The parts of ChromaClient that record building touches; nothing is stored."""

    embedding_max_tokens = 8000
    embedding_model = "default"
    embedding_config: dict = {}
    persist_directory = "/nonexistent"

    def __init__(self, fail_truncate_on: str | None = None):
        self.fail_truncate_on = fail_truncate_on

    def truncate_text(self, text, max_tokens=None):
        if self.fail_truncate_on and self.fail_truncate_on in text:
            raise RuntimeError("truncate failed")
        return text[:50_000]

    def get_existing_ids(self, ids):
        return set()


def _item(key="ITEM0001", fulltext=LONG_TEXT, title="A paper"):
    return {
        "key": key,
        "version": 1,
        "data": {
            "key": key,
            "itemType": "journalArticle",
            "title": title,
            "abstractNote": "An abstract.",
            "creators": [],
            "dateAdded": "2024-01-01T00:00:00Z",
            "dateModified": "2024-01-01T00:00:00Z",
            "fulltext": fulltext,
        },
    }


@pytest.fixture
def make_search(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())

    def _make(chunking: bool, chroma=None):
        search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma or _FakeChroma())
        search._chunking_config = {
            "enabled": chunking,
            "chunk_size": 400,
            "overlap": 50,
            "max_chunks_per_item": 5,
        }
        search._run_group_id = 0
        return search

    return _make


def test_item_level_builds_one_record_under_the_item_key(make_search):
    records = make_search(chunking=False)._build_item_records(_item(), chunking=False)

    assert [r["id"] for r in records] == ["ITEM0001"]
    assert "parent_item_key" not in records[0]["metadata"]
    assert LONG_TEXT in records[0]["document"]


def test_chunked_builds_numbered_passages_with_provenance(make_search):
    records = make_search(chunking=True)._build_item_records(_item(), chunking=True)

    assert [r["id"] for r in records] == [f"ITEM0001#{i}" for i in range(5)]
    for index, record in enumerate(records):
        meta = record["metadata"]
        assert meta["parent_item_key"] == "ITEM0001"
        assert meta["chunk_index"] == index
        assert meta["n_chunks"] == 5
        assert meta["char_end"] > meta["char_start"]


def test_item_without_key_or_text_builds_nothing(make_search):
    search = make_search(chunking=True)
    # An annotation with no text; other item types always embed their "No authors listed".
    empty = {"key": "ITEM0002", "data": {"itemType": "annotation"}}

    assert search._build_item_records({"data": {"title": "No key"}}, chunking=True) == []
    assert search._build_item_records(empty, chunking=True) == []
    assert search._build_item_records(empty, chunking=False) == []


def test_realtime_path_writes_what_the_builder_builds(make_search):
    search = make_search(chunking=True)
    expected = search._build_item_records(_item(), chunking=True)

    prepared = search._prepare_and_classify_slice([_item()], force_rebuild=True)

    assert prepared["ids"] == [r["id"] for r in expected]
    assert prepared["documents"] == [r["document"] for r in expected]
    assert prepared["metadatas"] == [r["metadata"] for r in expected]
    assert prepared["item_keys_order"] == ["ITEM0001"]
    assert prepared["item_doc_counts"] == [5]


def test_batch_path_submits_what_the_builder_builds(make_search):
    search = make_search(chunking=False)
    expected = search._build_item_records(_item(), chunking=False)

    records, stats = search._prepare_index_records([_item()])

    assert records == expected
    assert stats == {"processed": 1, "skipped": 0, "errors": 0}


def test_batch_path_stays_item_level_with_chunking_on(make_search):
    """Unchanged by the refactor; chunking on this path is a later change (#416)."""
    records, _ = make_search(chunking=True)._prepare_index_records([_item()])

    assert [r["id"] for r in records] == ["ITEM0001"]


def test_failure_part_way_through_an_item_keeps_later_items_aligned(make_search):
    """A failed item contributes no documents, so counts still slice the flat lists."""
    bad = _item(key="BADITEM1", fulltext=LONG_TEXT + " POISON " + LONG_TEXT)
    chroma = _FakeChroma(fail_truncate_on="POISON")
    search = make_search(chunking=True, chroma=chroma)
    search._chunking_config["max_chunks_per_item"] = 50

    prepared = search._prepare_and_classify_slice([bad, _item()], force_rebuild=True)

    assert prepared["prep_stats"]["errors"] == 1
    assert prepared["item_keys_order"] == ["ITEM0001"]
    assert all(doc_id.startswith("ITEM0001#") for doc_id in prepared["ids"])
    assert len(prepared["ids"]) == sum(prepared["item_doc_counts"])
