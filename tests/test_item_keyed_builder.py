"""Records in the item-keyed shape (#826): what each work unit becomes.

A regular item becomes one bibliographic record with no file text; an
attachment becomes one attachment record plus, with chunking, its passages,
all keyed by the attachment's own key. Records carry only their own state.
"""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")

from zotero_mcp import semantic_search  # noqa: E402
from zotero_mcp.semantic_search import _attachment_unit, _item_unit  # noqa: E402

FILE_TEXT = "\n\n".join(f"Paragraph {i}. " + " ".join(f"word{i}_{j}" for j in range(40)) for i in range(30))


class _FakeChroma:
    embedding_max_tokens = 8000
    embedding_model = "default"
    embedding_config: dict = {}
    persist_directory = "/nonexistent"

    def truncate_text(self, text, max_tokens=None):
        return text[:50_000]

    def get_existing_ids(self, ids):
        return set()


@pytest.fixture
def make_search(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())

    def _make(chunking: bool, max_chunks: int = 5):
        search = semantic_search.ZoteroSemanticSearch(
            chroma_client=_FakeChroma(), config_path=str(tmp_path / "config.json")
        )
        search._chunking_config = {
            "enabled": chunking,
            "chunk_size": 400,
            "overlap": 50,
            "max_chunks_per_item": max_chunks,
        }
        return search

    return _make


def _regular(key="ITEM0001", group_id=0, fulltext=""):
    data = {
        "key": key,
        "itemType": "journalArticle",
        "title": "A paper",
        "abstractNote": "An abstract.",
        "creators": [{"creatorType": "author", "lastName": "Doe", "firstName": "Jane"}],
        "DOI": "10.1/x",
        "dateModified": "2024-01-02T00:00:00Z",
        "fulltext": fulltext,
    }
    if group_id is not None:
        data["group_id"] = group_id
    return {"key": key, "version": 3, "data": data}


def _att(key="ATTKEY01", text=FILE_TEXT, file_hash="ab" * 32, group_id=0):
    return _attachment_unit(
        key,
        text,
        file_stamp="123:456",
        file_hash=file_hash,
        date_modified="2024-01-03 00:00:00",
        group_id=group_id,
        owner_key="ITEM0001",
    )


def test_bibliographic_record_has_no_file_text_and_only_its_own_fields(make_search):
    search = make_search(chunking=True)
    item = _item_unit(_regular(fulltext="FILE BODY TEXT"))

    records = search._build_item_records(item, chunking=True)

    assert [r["id"] for r in records] == ["ITEM0001"]
    assert records[0]["document"] == search._create_document_text(item)
    assert "FILE BODY TEXT" not in records[0]["document"]
    assert records[0]["metadata"] == {
        "item_key": "ITEM0001",
        "group_id": 0,
        "date_modified": "2024-01-02T00:00:00Z",
    }


def test_bibliographic_record_omits_an_unknown_library(make_search):
    records = make_search(chunking=False)._build_item_records(_item_unit(_regular(group_id=None)), chunking=False)

    assert "group_id" not in records[0]["metadata"]


def test_attachment_record_is_file_text_under_its_own_key(make_search):
    records = make_search(chunking=False)._build_item_records(_att(), chunking=False)

    assert [r["id"] for r in records] == ["ATTKEY01"]
    assert records[0]["document"] == FILE_TEXT
    assert records[0]["metadata"] == {
        "item_key": "ATTKEY01",
        "group_id": 0,
        "date_modified": "2024-01-03 00:00:00",
        "file_stamp": "123:456",
        "file_hash": "ab" * 32,
    }


def test_attachment_without_a_hash_omits_it(make_search):
    records = make_search(chunking=False)._build_item_records(_att(file_hash=None), chunking=False)

    assert "file_hash" not in records[0]["metadata"]
    assert records[0]["metadata"]["file_stamp"] == "123:456"


def test_passages_are_keyed_by_the_attachment_with_offsets_into_the_file(make_search):
    records = make_search(chunking=True, max_chunks=50)._build_item_records(_att(), chunking=True)

    article, passages = records[0], records[1:]
    assert article["id"] == "ATTKEY01"
    assert passages and [r["id"] for r in passages] == [f"ATTKEY01#{i}" for i in range(len(passages))]
    allowed = {"item_key", "group_id", "chunk_index", "n_chunks", "char_start", "char_end", "page", "section"}
    for index, record in enumerate(passages):
        meta = record["metadata"]
        assert set(meta) <= allowed
        assert meta["item_key"] == "ATTKEY01"
        assert meta["group_id"] == 0
        assert meta["chunk_index"] == index
        assert meta["n_chunks"] == len(passages)
        assert FILE_TEXT[meta["char_start"] : meta["char_end"]].strip() == record["document"]


def test_max_chunks_applies_per_attachment(make_search):
    search = make_search(chunking=True, max_chunks=3)
    units = [_item_unit(_regular()), _att("ATTKEY01"), _att("ATTKEY02")]

    records, stats = search._prepare_index_records(units)

    ids = [r["id"] for r in records]
    assert ids.count("ITEM0001") == 1
    assert [i for i in ids if i.startswith("ATTKEY01")] == ["ATTKEY01", "ATTKEY01#0", "ATTKEY01#1", "ATTKEY01#2"]
    assert [i for i in ids if i.startswith("ATTKEY02")] == ["ATTKEY02", "ATTKEY02#0", "ATTKEY02#1", "ATTKEY02#2"]
    assert stats == {"processed": 3, "skipped": 0, "errors": 0}


def test_attachment_without_text_builds_nothing(make_search):
    assert make_search(chunking=True)._build_item_records(_att(text="  "), chunking=True) == []


def test_web_api_text_becomes_a_unit_of_its_attachment(make_search):
    search = make_search(chunking=False)
    item = _regular(fulltext="Body of the PDF.")
    item["data"]["fulltextSource"] = "web-api:attachment:PDFKEY01"
    item["data"]["fulltextAttachment"] = {"file_stamp": "api:md5x:99", "date_modified": "2024-02-01T00:00:00Z"}

    units = search._as_units([item])

    assert [(u["key"], u["unit"]) for u in units] == [("ITEM0001", "item"), ("PDFKEY01", "attachment")]
    records, _ = search._prepare_index_records(units)
    assert [r["id"] for r in records] == ["ITEM0001", "PDFKEY01"]
    assert records[1]["document"] == "Body of the PDF."
    assert records[1]["metadata"] == {
        "item_key": "PDFKEY01",
        "group_id": 0,
        "date_modified": "2024-02-01T00:00:00Z",
        "file_stamp": "api:md5x:99",
    }
    assert "Body of the PDF." not in records[0]["document"]


def test_realtime_and_batch_paths_build_the_same_records(make_search):
    search = make_search(chunking=True)
    units = [_item_unit(_regular()), _att()]

    records, _ = search._prepare_index_records(units)
    prepared = search._prepare_and_classify_slice(units, force_rebuild=True)

    assert [r["id"] for r in records] == prepared["ids"]
    assert [r["metadata"] for r in records] == prepared["metadatas"]
    assert prepared["item_keys_order"] == ["ITEM0001", "ATTKEY01"]
