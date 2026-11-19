"""SqliteBackend.collection_page: one page of a collection's entries, their
children and the total, without hydrating the rest of the collection.

zotero_get_collection_items used to hydrate every item in the collection and
every child of every item to print one page of 50.
"""

import pytest

from _full_db import FullDb
from zotero_mcp.library import ApiBackend, SqliteBackend
from zotero_mcp.local_db import LocalZoteroReader


@pytest.fixture
def reader(tmp_path):
    db = FullDb(tmp_path / "zotero.sqlite")
    db.collection("COLL0001")
    db.collection("SUBCOLL1", parent="COLL0001")
    for n in range(6):
        paper = db.item(f"PAPER{n:03d}", fields={"title": f"Paper {n}"}, collections=("COLL0001",))
        pdf = db.item(f"PDF00{n:03d}", "attachment", parent=paper)
        db.item(f"ANNOT{n:03d}", "annotation", parent=pdf)
        if n % 2:
            db.item(f"NOTE0{n:03d}", "note", parent=paper, note="child note")
    db.item("LONEPDF1", "attachment", collections=("COLL0001",))
    db.item("LONENOTE", "note", note="standalone", collections=("COLL0001",))
    db.item("TRASHED1", fields={"title": "Trashed"}, collections=("COLL0001",), deleted=True)
    db.item("SUBPAPER", fields={"title": "In the subcollection"}, collections=("SUBCOLL1",))
    db.item("ELSEWHER", fields={"title": "Not filed"})
    with LocalZoteroReader(db_path=db.done()) as r:
        yield r


def _keys(items):
    return [item["key"] for item in items]


def test_page_children_and_total(reader):
    entries, children, total = SqliteBackend(reader, 0).collection_page(
        "COLL0001", limit=2, offset=1
    )
    assert _keys(entries) == ["PAPER001", "PAPER002"]
    assert sorted(_keys(children)) == ["NOTE0001", "PDF00001", "PDF00002"]
    # Six papers and the standalone PDF; the note, the trashed item and the
    # subcollection's paper are not entries.
    assert total == 7


def test_last_page_and_past_the_end(reader):
    backend = SqliteBackend(reader, 0)
    entries, _children, total = backend.collection_page("COLL0001", limit=5, offset=5)
    assert _keys(entries) == ["PAPER005", "LONEPDF1"] and total == 7
    assert backend.collection_page("COLL0001", limit=5, offset=50) == ([], [], 7)


def test_subcollections_and_unknown_key(reader):
    backend = SqliteBackend(reader, 0)
    entries, _children, total = backend.collection_page(
        "COLL0001", limit=50, include_subcollections=True
    )
    assert total == 8 and "SUBPAPER" in _keys(entries)
    assert backend.collection_page("MISSING0", limit=50) is None


def test_only_the_page_is_hydrated(reader):
    hydrated = []
    original = reader._build_full_items
    reader._build_full_items = lambda conn, rows: hydrated.extend(rows) or original(conn, rows)
    SqliteBackend(reader, 0).collection_page("COLL0001", limit=2)
    # Two entries, then their children (annotations are fetched with them and
    # dropped); never the other four papers.
    assert {row["key"] for row in hydrated} == {"PAPER000", "PAPER001", "PDF00000", "PDF00001", "NOTE0001"}


def test_api_backend_answers_the_same_shape(reader):
    """Both backends cut the same window out of the same listing."""
    everything = SqliteBackend(reader, 0).collection_items("COLL0001")

    class _Zot:
        def collection(self, key):
            return {"key": key}

        def collection_items(self, key, start=0, limit=100, **params):
            return everything[start:start + limit]

    api = ApiBackend(_Zot()).collection_page("COLL0001", limit=2, offset=1)
    sqlite = SqliteBackend(reader, 0).collection_page("COLL0001", limit=2, offset=1)
    assert _keys(api[0]) == _keys(sqlite[0])
    assert sorted(_keys(api[1])) == sorted(_keys(sqlite[1]))
    assert api[2] == sqlite[2]
