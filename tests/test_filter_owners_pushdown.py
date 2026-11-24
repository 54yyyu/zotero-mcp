"""``get_filter_owners`` narrows its candidates in SQLite (#826).

``$eq`` / ``$in`` (and the plain-value form) on ``item_type``, ``tags`` and
``item_key`` become SQL conditions; the Python check (``matches``) stays the
authority, so a pushed-down filter returns exactly what an unnarrowed one
does once checked.

Real sqlite files in tmp_path; HOME and Path.home point there too.
"""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip("chromadb relies on pydantic v1 paths incompatible with Python 3.14+", allow_module_level=True)

from test_semantic_owner_resolution import _isolated, _owner_db  # noqa: E402, F401

from zotero_mcp.local_db import LocalZoteroReader  # noqa: E402
from zotero_mcp.semantic_filters import equality_clauses, live_fields, matches  # noqa: E402


@pytest.fixture
def reader(tmp_path):
    db = tmp_path / "zotero.sqlite"
    _owner_db(db)
    import sqlite3

    conn = sqlite3.connect(db)
    # A second article with two tags, one with a space, and a book with none.
    conn.execute("INSERT INTO itemTypes VALUES (4, 'book')")
    conn.execute("INSERT INTO items VALUES (8, 'PAPER002', 1, 1, '2024-01-01', '2024-01-01')")
    conn.execute("INSERT INTO items VALUES (9, 'BOOK0001', 4, 1, '2024-01-01', '2024-01-01')")
    conn.execute("INSERT INTO tags VALUES (2, 'two words')")
    conn.execute("INSERT INTO tags VALUES (3, 'other')")
    conn.execute("INSERT INTO itemTags VALUES (8, 2, 0)")
    conn.execute("INSERT INTO itemTags VALUES (8, 1, 0)")
    conn.execute("INSERT INTO itemTags VALUES (9, 3, 0)")
    conn.commit()
    conn.close()
    r = LocalZoteroReader(db_path=str(db))
    yield r
    r.close()


FILTERS = [
    {"item_type": "journalArticle"},
    {"item_type": {"$eq": "book"}},
    {"item_type": {"$in": ["book", "attachment"]}},
    {"item_type": {"$in": []}},
    {"item_type": {"$ne": "book"}},
    {"item_type": 5},
    {"tags": "keep"},
    {"tags": {"$eq": "other"}},
    {"tags": {"$in": ["keep", "other"]}},
    {"tags": "two words"},
    {"tags": "keep two words"},
    {"tags": "two words keep"},
    {"tags": ""},
    {"tags": {"$nin": ["keep"]}},
    {"item_key": "PAPER001"},
    {"item_key": {"$in": ["PAPER001", "BOOK0001", "NOSUCH"]}},
    {"$and": [{"item_type": "journalArticle"}, {"tags": "keep"}]},
    {"$and": [{"item_type": "journalArticle"}, {"item_key": "BOOK0001"}]},
    {"$and": [{"tags": "keep"}, {"tags": "other"}]},
    {"$or": [{"item_type": "book"}, {"tags": "keep"}]},
    {"$and": [{"item_type": "book"}, {"title": "x"}]},
    {"title": "Paper One"},
]


@pytest.mark.parametrize("where", FILTERS, ids=[str(f) for f in FILTERS])
def test_the_pushed_down_candidates_match_the_same_owners_as_all_of_them(reader, where):
    def matching(owners):
        return {o["key"] for o in owners if matches(where, live_fields(o))}

    everything = reader.get_filter_owners(group_id=None)
    narrowed = reader.get_filter_owners(group_id=None, where=where)

    assert matching(narrowed) == matching(everything)
    assert {o["key"] for o in narrowed} <= {o["key"] for o in everything}
    by_key = {o["key"]: o for o in everything}
    assert all(o == by_key[o["key"]] for o in narrowed)  # a candidate is complete: tags, attachments


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        ({"item_type": "book"}, {"BOOK0001"}),
        ({"item_type": {"$in": ["book", "attachment"]}}, {"BOOK0001", "STANDATT"}),
        ({"tags": "other"}, {"BOOK0001"}),
        ({"tags": {"$in": ["keep", "other"]}}, {"PAPER001", "PAPER002", "BOOK0001"}),
        ({"item_key": {"$in": ["PAPER001", "BOOK0001"]}}, {"PAPER001", "BOOK0001"}),
        ({"$and": [{"item_type": "journalArticle"}, {"tags": "keep"}]}, {"PAPER001", "PAPER002"}),
        ({"$and": [{"item_type": "book"}, {"item_key": "PAPER001"}]}, set()),
    ],
)
def test_sqlite_returns_only_the_narrowed_candidates(reader, where, expected):
    assert {o["key"] for o in reader.get_filter_owners(group_id=None, where=where)} == expected


def test_a_filter_on_a_field_not_pushed_down_returns_every_owner(reader):
    everything = {o["key"] for o in reader.get_filter_owners(group_id=None)}
    assert {o["key"] for o in reader.get_filter_owners(group_id=None, where={"title": "x"})} == everything
    assert {o["key"] for o in reader.get_filter_owners(group_id=None, where={"$or": [{"item_type": "book"}]})} == everything


def test_the_detail_queries_are_restricted_to_the_candidates(reader):
    statements: list[str] = []
    reader._get_connection().set_trace_callback(statements.append)

    reader.get_filter_owners(group_id=None, where={"item_type": "book"})

    detail = [s for s in statements if any(h in s for h in ("SELECT id.itemID", "SELECT itg.itemID", "SELECT ia.parentItemID"))]
    assert len(detail) == 3
    assert all("typeName IN" in s for s in detail)  # each reads the narrowed owners, not the library


def test_equality_clauses_keeps_what_a_conjunction_can_push_down():
    where = {"$and": [{"item_type": {"$in": ["a", "b"]}}, {"tags": "t"}, {"title": "x"}, {"tags": {"$ne": "u"}},
                      {"$or": [{"tags": "v"}]}, {"item_key": {"$eq": "K"}}, {"item_type": 3}]}
    assert equality_clauses(where, ("item_type", "tags", "item_key")) == {
        "item_type": [["a", "b"]],
        "tags": [["t"]],
        "item_key": [["K"]],
    }
