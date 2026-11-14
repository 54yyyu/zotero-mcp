"""Filters keep 0.17's contract (#826): the live evaluator agrees with ChromaDB.

0.17 stored an item's metadata in the index and passed ``filters`` to a ChromaDB
``where``. Now the same filter is checked by ``check_where`` and evaluated by
``matches`` over live Zotero data. These tests build a real in-memory
ChromaDB collection with 0.17-shaped metadata and require the same matches and
the same error text for a set of filters. No files are touched.
"""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

chromadb = pytest.importorskip("chromadb")

from zotero_mcp.semantic_filters import check_where, live_fields, matches  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")


def _item(key, **data):
    base = {"key": key, "itemType": "journalArticle", "title": "", "tags": [], "creators": []}
    return {"key": key, "data": {**base, **data}}


ITEMS = [
    # item, stored extras (library, has_fulltext as 0.17 wrote it)
    (
        _item(
            "AAAAAAAA", title="Mindfulness", date="2020-05-01", DOI="10.1/a", publicationTitle="J Psych",
            url="https://a.test", dateAdded="2021-01-01T00:00:00Z", dateModified="2022-02-02T00:00:00Z",
            tags=[{"tag": "x"}], creators=[{"lastName": "Smith", "firstName": "Ann"}],
            extra="Citation Key: smith2020",
        ),
        {"group_id": 0, "has_fulltext": True},
    ),
    (
        _item("BBBBBBBB", itemType="book", title="Books", tags=[{"tag": "x"}, {"tag": "y"}]),
        {"group_id": 0},
    ),
    (
        _item("CCCCCCCC", title="Zed", DOI="10.1/c", tags=[{"tag": "to read"}], extra="citationkey: zed1999"),
        {"group_id": 777, "has_fulltext": "failed"},
    ),
]

FILTERS = [
    {"item_type": "book"},
    {"item_type": {"$ne": "book"}},
    {"item_type": {"$in": ["book", "thesis"]}},
    {"item_type": {"$nin": ["book"]}},
    {"tags": "x"},
    {"tags": "x y"},
    {"tags": {"$ne": "x"}},
    {"tags": {"$in": ["x", "to read"]}},
    {"tags": ""},
    {"tags": {"$contains": "x"}},
    {"tags": {"$not_contains": "x"}},
    {"title": {"$gt": 1}},
    {"title": {"$in": ["Zed", "Books"]}},
    {"doi": {"$ne": "10.1/a"}},
    {"doi": ""},
    {"citation_key": "smith2020"},
    {"citation_key": {"$ne": ""}},
    {"creators": "Smith, Ann"},
    {"creators": "No authors listed"},
    {"date": "2020-05-01"},
    {"url": {"$ne": "https://a.test"}},
    {"date_added": {"$in": ["2021-01-01T00:00:00Z"]}},
    {"date_modified": "2022-02-02T00:00:00Z"},
    {"publication": "J Psych"},
    {"item_key": {"$in": ["AAAAAAAA", "BBBBBBBB"]}},
    {"group_id": 777},
    {"group_id": {"$ne": 0}},
    {"group_id": {"$gt": 0}},
    {"group_id": {"$in": [0.0]}},
    {"group_id": "0"},
    {"has_fulltext": True},
    {"has_fulltext": 1},
    {"has_fulltext": {"$ne": True}},
    {"has_fulltext": {"$nin": [True]}},
    {"fulltext_source": {"$ne": "x"}},
    {"fulltext_source": "x"},
    {"$and": [{"item_type": "journalArticle"}, {"tags": {"$ne": "x"}}]},
    {"$or": [{"item_type": "book"}, {"doi": "10.1/c"}]},
    {"$and": [{"$or": [{"item_type": "book"}, {"group_id": 777}]}, {"title": {"$ne": "Zed"}}]},
    {"unknown_field": {"$ne": 1}},
    {"unknown_field": 1},
]

ERRORS = [
    {"item_type": "book", "tags": "x"},
    {"$and": [{"item_type": "book"}]},
    {"$or": [{"item_type": "book"}]},
    {"$and": []},
    {},
    {"title": {"$gt": "a"}},
    {"group_id": {"$gt": True}},
    {"title": {"$bogus": 1}},
    {"title": {"$in": []}},
    {"title": {"$in": ["a", 1]}},
    {"title": {"$in": "a"}},
    {"title": None},
    {"title": {"$eq": None}},
    {"title": {"$gt": 1, "$lt": 2}},
    {"title": ["a"]},
    {"$bogus": 1},
    {"$in": ["a"]},
    {"$and": [{"item_type": "book", "tags": "x"}, {"title": "a"}]},
    {"$contains": "a"},
    ["item_type", "book"],
    "item_type",
]


@pytest.fixture
def collection():
    client = chromadb.EphemeralClient()
    name = "parity" + str(id(object()))
    col = client.create_collection(name, embedding_function=None)
    metas = []
    for item, extra in ITEMS:
        data = item["data"]
        meta = {
            "item_key": item["key"],
            "item_type": data["itemType"],
            "title": data.get("title", ""),
            "date": data.get("date", ""),
            "date_added": data.get("dateAdded", ""),
            "date_modified": data.get("dateModified", ""),
            "creators": live_fields(item)["creators"],
            "publication": data.get("publicationTitle", ""),
            "url": data.get("url", ""),
            "doi": data.get("DOI", ""),
            "tags": " ".join(t["tag"] for t in data["tags"]),
            "citation_key": live_fields(item)["citation_key"],
            **extra,
        }
        metas.append(meta)
    col.add(
        ids=[i["key"] for i, _ in ITEMS],
        embeddings=[[1.0, float(n)] for n in range(len(ITEMS))],
        metadatas=metas,
    )
    yield col
    client.delete_collection(name)


def _chroma(collection, where):
    try:
        got = collection.query(query_embeddings=[[1.0, 0.0]], n_results=len(ITEMS), where=where)
    except Exception as e:  # noqa: BLE001 - the text is the point
        return ("error", str(e))
    return ("ids", sorted(got["ids"][0]))


def _ours(where):
    try:
        check_where(where)
    except ValueError as e:
        return ("error", str(e))
    out = []
    for item, extra in ITEMS:
        fields = {**live_fields(item), **extra}
        if matches(where, fields):
            out.append(item["key"])
    return ("ids", sorted(out))


@pytest.mark.parametrize("where", FILTERS, ids=[str(f) for f in FILTERS])
def test_a_filter_matches_the_same_items_as_chroma_does(collection, where):
    assert _ours(where) == _chroma(collection, where)


@pytest.mark.parametrize("where", ERRORS, ids=[str(f) for f in ERRORS])
def test_a_filter_chroma_refuses_fails_with_its_text(collection, where):
    expected = _chroma(collection, where)

    assert expected[0] == "error"
    assert _ours(where) == expected


def test_a_library_clause_is_validated_as_part_of_the_where(collection):
    where = {"$and": [{"item_type": "book", "tags": "x"}, {"group_id": 0}]}
    expected = _chroma(collection, where)

    with pytest.raises(ValueError) as raised:
        check_where({"item_type": "book", "tags": "x"}, group_id=0)

    assert ("error", str(raised.value)) == expected
    check_where({"item_type": "book"}, group_id=0)
    check_where({}, group_id=0)  # no filter beyond the library: 0.17 sent just the library clause
