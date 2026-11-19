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
