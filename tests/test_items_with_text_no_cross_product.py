"""get_items_with_text lists each creator and each note once, in order.

It joined itemNotes and itemCreators side by side and concatenated both, so
an item with N notes and M creators got every creator N times and every note
M times, in whatever order the join produced. The semantic index embeds and
stores that text.
"""

import sqlite3

from _search_corpus import Item, _ITEM_TYPE_IDS, USER_LIBRARY_ID, build_sqlite
from zotero_mcp.local_db import LocalZoteroReader


def _db(tmp_path):
    db = tmp_path / "zotero.sqlite"
    build_sqlite(db, [
        Item("PAPER001", title="Paper with notes",
             creators=[("Zed", "Young"), ("Amy", "Adams"), (None, "Consortium")]),
        Item("PAPER002", title="Bare paper"),
    ])
    conn = sqlite3.connect(db)
    for n, text in enumerate(["first note", "second note"]):
        conn.execute(
            "INSERT INTO items (itemID, key, itemTypeID, libraryID, dateAdded, dateModified) "
            "VALUES (?, ?, ?, ?, '2024-01-01 00:00:00', '2024-01-01 00:00:00')",
            (50 + n, f"NOTE000{n}", _ITEM_TYPE_IDS["note"], USER_LIBRARY_ID),
        )
        conn.execute("INSERT INTO itemNotes VALUES (?, 1, ?)", (50 + n, text))
    conn.commit()
    conn.close()
    return db


def test_creators_and_notes_are_listed_once_in_order(tmp_path):
    with LocalZoteroReader(db_path=str(_db(tmp_path))) as reader:
        item = reader.get_item_by_key("PAPER001")
        bare = reader.get_item_by_key("PAPER002")

    assert item.creators == "Young, Zed; Adams, Amy; Consortium"
    assert item.notes == "first note second note"
    assert bare.creators is None and bare.notes is None


def test_every_regular_item_is_returned_once(tmp_path):
    with LocalZoteroReader(db_path=str(_db(tmp_path))) as reader:
        keys = [item.key for item in reader.get_items_with_text()]
    assert sorted(keys) == ["PAPER001", "PAPER002"]


def test_text_items_by_keys_is_one_query_that_uses_the_library_key_index(tmp_path):
    """The re-key looks a batch of items up in one query per library, through Zotero's (libraryID, key) index."""
    db = _db(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute("CREATE UNIQUE INDEX itemsUnique ON items (libraryID, key)")  # as Zotero's schema has it
    conn.commit()
    conn.close()
    with LocalZoteroReader(db_path=str(db)) as reader:
        statements: list[str] = []
        reader._get_connection().set_trace_callback(statements.append)
        found = reader.get_text_items_by_keys(["PAPER001", "NOSUCH01", "PAPER002"], 0)
        reader._get_connection().set_trace_callback(None)
        assert set(found) == {"PAPER001", "PAPER002"}
        assert reader.get_text_items_by_keys(["PAPER001"], 999) == {}, "a library that is not here has no items"

        queries = [s for s in statements if "i.key IN" in s]
        assert len(queries) == 1 and "i.libraryID IN" in queries[0]
        plan = reader._get_connection().execute("EXPLAIN QUERY PLAN " + queries[0]).fetchall()
        items_steps = [row[3] for row in plan if " items" in f" {row[3]}" and "itemData" not in row[3]]
        assert any("SEARCH" in step and "libraryID" in step for step in items_steps), items_steps
        assert not any(step.startswith("SCAN") and "i" == step.split()[-1] for step in items_steps), items_steps
