"""Collection scoping in `get_items_with_text`.

`semantic_search.collection_keys` restricts indexing to the configured
collections plus everything nested beneath them, resolved by walking
`collections.parentCollectionID` in `local_db.py`. That walk had no test
coverage at all, so most of what follows pins existing behaviour rather than
new behaviour: nesting is followed to any depth, siblings are excluded,
unknown keys are ignored.

The one behaviour change is that the walk now remembers which collections it
has already visited. A `parentCollection` cycle previously made it loop
forever, and configuring both a collection and one of its ancestors bound the
same collectionID as several SQL parameters.
"""

import sqlite3

from zotero_mcp.local_db import LocalZoteroReader

BASE_SCHEMA = [
    "CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT)",
    """CREATE TABLE items (
        itemID INTEGER PRIMARY KEY, itemTypeID INT, dateAdded TEXT,
        dateModified TEXT, clientDateModified TEXT, libraryID INT,
        key TEXT UNIQUE, version INT, synced INT
    )""",
    "CREATE TABLE libraries (libraryID INTEGER PRIMARY KEY, type TEXT, "
    "editable INT, filesEditable INT)",
    "CREATE TABLE groups (groupID INTEGER PRIMARY KEY, libraryID INT UNIQUE, "
    "name TEXT, description TEXT, version INT)",
    "CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY)",
    "CREATE TABLE itemData (itemID INT, fieldID INT, valueID INT)",
    "CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT)",
    "CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT)",
    "CREATE TABLE itemNotes (itemID INT, parentItemID INT, note TEXT)",
    "CREATE TABLE itemCreators (itemID INT, creatorID INT)",
    "CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, "
    "lastName TEXT)",
    # The two tables the collection walk reads. No other test in the suite
    # creates them, which is why this file exists.
    "CREATE TABLE collections (collectionID INTEGER PRIMARY KEY, "
    "collectionName TEXT, parentCollectionID INT, libraryID INT, key TEXT)",
    "CREATE TABLE collectionItems (collectionID INT, itemID INT)",
]


def make_db(path, collections, memberships):
    """Build a minimal zotero.sqlite.

    ``collections`` maps a collection key to its parent's key (None for a
    top-level collection). ``memberships`` maps a collection key to the item
    keys filed directly in it. Every referenced item is created.
    """
    conn = sqlite3.connect(path)
    for statement in BASE_SCHEMA:
        conn.execute(statement)
    conn.execute("INSERT INTO itemTypes VALUES (1, 'journalArticle')")
    conn.execute("INSERT INTO libraries VALUES (1, 'user', 1, 1)")

    ids = {key: index for index, key in enumerate(collections, start=1)}
    for key, parent in collections.items():
        conn.execute(
            "INSERT INTO collections VALUES (?, ?, ?, 1, ?)",
            (ids[key], f"Collection {key}", ids.get(parent), key),
        )

    item_ids = {}
    for collection_key, item_keys in memberships.items():
        for item_key in item_keys:
            if item_key not in item_ids:
                item_ids[item_key] = len(item_ids) + 1
                conn.execute(
                    "INSERT INTO items VALUES (?, 1, '2026-01-01 00:00:00', "
                    "'2026-01-01 00:00:00', '2026-01-01 00:00:00', 1, ?, 1, 0)",
                    (item_ids[item_key], item_key),
                )
            conn.execute(
                "INSERT INTO collectionItems VALUES (?, ?)",
                (ids[collection_key], item_ids[item_key]),
            )
    conn.commit()
    conn.close()


def scoped_item_keys(db_path, collection_keys):
    with LocalZoteroReader(db_path=str(db_path)) as reader:
        items = reader.get_items_with_text(collection_keys=collection_keys)
    return sorted(item.key for item in items)


def test_expands_nested_subcollections(tmp_path):
    """A configured collection pulls in its descendants at any depth."""
    db = tmp_path / "zotero.sqlite"
    make_db(
        db,
        collections={"ROOT": None, "CHILD": "ROOT", "GRANDCHILD": "CHILD"},
        memberships={"ROOT": ["ITEMROOT"], "CHILD": ["ITEMCHILD"],
                     "GRANDCHILD": ["ITEMGRAND"]},
    )
    assert scoped_item_keys(db, ["ROOT"]) == ["ITEMCHILD", "ITEMGRAND", "ITEMROOT"]


def test_sibling_collections_are_excluded(tmp_path):
    """Scoping is real: a sibling branch is not swept in."""
    db = tmp_path / "zotero.sqlite"
    make_db(
        db,
        collections={"ROOT": None, "WANTED": "ROOT", "UNWANTED": "ROOT"},
        memberships={"WANTED": ["ITEMKEEP"], "UNWANTED": ["ITEMDROP"]},
    )
    assert scoped_item_keys(db, ["WANTED"]) == ["ITEMKEEP"]


def test_unknown_collection_key_is_ignored(tmp_path):
    """An unknown key contributes nothing rather than raising."""
    db = tmp_path / "zotero.sqlite"
    make_db(
        db,
        collections={"ROOT": None},
        memberships={"ROOT": ["ITEMROOT"]},
    )
    assert scoped_item_keys(db, ["ROOT", "NOSUCHKEY"]) == ["ITEMROOT"]


def test_parent_cycle_terminates(tmp_path):
    """A parentCollection cycle must not loop forever.

    Zotero's own client cannot produce one, but a corrupted, partially synced
    or hand-edited database can. Without a visited set the walk pops A, pushes
    B, pops B, pushes A, appending to the collection-id list on every pass —
    an unbounded loop rather than a wrong answer, which is why this test would
    hang rather than fail before the fix.
    """
    db = tmp_path / "zotero.sqlite"
    make_db(
        db,
        collections={"AAA": "BBB", "BBB": "AAA"},
        memberships={"AAA": ["ITEMA"], "BBB": ["ITEMB"]},
    )
    # Both are reachable from either entry point once the cycle is followed.
    assert scoped_item_keys(db, ["AAA"]) == ["ITEMA", "ITEMB"]


def test_self_parenting_collection_terminates(tmp_path):
    """The degenerate one-node cycle terminates too."""
    db = tmp_path / "zotero.sqlite"
    make_db(
        db,
        collections={"SELF": "SELF"},
        memberships={"SELF": ["ITEMSELF"]},
    )
    assert scoped_item_keys(db, ["SELF"]) == ["ITEMSELF"]


def test_overlapping_configured_keys_do_not_duplicate(tmp_path):
    """Configuring a collection and its ancestor yields each item once.

    Passes before and after the fix, deliberately: it is the no-regression
    half of the dedupe. The generated SQL already selects DISTINCT itemID, so
    the repeats never changed the result — what they changed is that the same
    collectionID was bound as another parameter each time, against a cap
    SQLite places on how many one statement may carry.
    """
    db = tmp_path / "zotero.sqlite"
    make_db(
        db,
        collections={"ROOT": None, "CHILD": "ROOT"},
        memberships={"ROOT": ["ITEMROOT"], "CHILD": ["ITEMCHILD"]},
    )
    assert scoped_item_keys(db, ["ROOT", "CHILD"]) == ["ITEMCHILD", "ITEMROOT"]
