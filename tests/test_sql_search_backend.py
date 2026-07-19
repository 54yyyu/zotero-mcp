"""Tests for the #167 SQLite metadata search backend (Phase B).

Builds a realistic (subset) zotero.sqlite fixture with items across a
personal and a group library, then exercises LocalZoteroReader.search_items_sql
/ advanced_search_sql directly, plus the tools/search.py branch that picks
between the SQL backend and the existing pyzotero-based path.
"""

import sqlite3
from pathlib import Path

from zotero_mcp.local_db import LocalZoteroReader

GROUP_ID = 6015547


def _build_db(db_path: Path) -> dict[str, int]:
    """Build a fixture DB; returns a name -> itemID map for convenience."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE libraries (
            libraryID INTEGER PRIMARY KEY, type TEXT NOT NULL,
            editable INT NOT NULL, filesEditable INT NOT NULL
        );
        CREATE TABLE groups (
            groupID INTEGER PRIMARY KEY, libraryID INT NOT NULL UNIQUE,
            name TEXT NOT NULL, description TEXT NOT NULL, version INT NOT NULL
        );
        CREATE TABLE items (
            itemID INTEGER PRIMARY KEY, key TEXT, itemTypeID INTEGER,
            libraryID INT, dateAdded TEXT, dateModified TEXT
        );
        CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
        CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
        CREATE TABLE itemData (itemID INTEGER, fieldID INTEGER, valueID INTEGER);
        CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE itemCreators (
            itemID INTEGER, creatorID INTEGER, creatorTypeID INTEGER, orderIndex INTEGER
        );
        CREATE TABLE creators (
            creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT
        );
        CREATE TABLE creatorTypes (creatorTypeID INTEGER PRIMARY KEY, creatorType TEXT);
        CREATE TABLE itemTags (itemID INTEGER, tagID INTEGER, type INTEGER);
        CREATE TABLE tags (tagID INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE itemNotes (itemID INTEGER, parentItemID INTEGER, note TEXT);
        CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY);
        CREATE TABLE collections (
            collectionID INTEGER PRIMARY KEY, collectionName TEXT,
            parentCollectionID INTEGER, libraryID INTEGER, key TEXT
        );
        CREATE TABLE collectionItems (collectionID INTEGER, itemID INTEGER);
        """
    )

    conn.execute("INSERT INTO libraries VALUES (1, 'user', 1, 1)")
    conn.execute("INSERT INTO libraries VALUES (5, 'group', 1, 1)")
    conn.execute(
        f"INSERT INTO groups VALUES ({GROUP_ID}, 5, 'Test Group', '', 1)"
    )

    conn.executemany(
        "INSERT INTO itemTypes (itemTypeID, typeName) VALUES (?, ?)",
        [(1, "journalArticle"), (2, "attachment"), (3, "note")],
    )
    conn.executemany(
        "INSERT INTO fields (fieldID, fieldName) VALUES (?, ?)",
        [(1, "title"), (2, "abstractNote"), (13, "date"), (26, "DOI"), (27, "publicationTitle")],
    )
    conn.execute("INSERT INTO creatorTypes VALUES (1, 'author')")

    def add_item(item_id, key, item_type_id, library_id, title=None, date=None,
                 abstract=None, doi=None, pub_title=None,
                 date_added="2024-01-01 00:00:00", date_modified="2024-01-01 00:00:00"):
        conn.execute(
            "INSERT INTO items (itemID, key, itemTypeID, libraryID, dateAdded, dateModified) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (item_id, key, item_type_id, library_id, date_added, date_modified),
        )
        field_values = [(1, title), (2, abstract), (13, date), (26, doi), (27, pub_title)]
        value_id = item_id * 100
        for field_id, value in field_values:
            if value is None:
                continue
            value_id += 1
            conn.execute("INSERT INTO itemDataValues (valueID, value) VALUES (?, ?)", (value_id, value))
            conn.execute("INSERT INTO itemData (itemID, fieldID, valueID) VALUES (?, ?, ?)",
                         (item_id, field_id, value_id))

    add_item(1, "PERS0001", 1, 1, title="Quantum Networks and Learning",
             date="2024-01-15", abstract="A paper about quantum stuff",
             doi="10.1/quantum", pub_title="Journal of Quantum",
             date_modified="2024-06-01 00:00:00")
    add_item(2, "PERS0002", 1, 1, title="Classical Literature Review", date="2018-05-01")
    add_item(3, "PERS0003", 2, 1, title="Ignored Attachment")  # itemType=attachment
    add_item(4, "PERS0004", 3, 1)  # standalone note, no title
    add_item(5, "PERS0005", 1, 1, title="Org Author Paper")
    add_item(6, "DELETEDKEY", 1, 1, title="Should Never Appear")
    add_item(7, "GRP00001", 1, 5, title="Group Library Paper about quantum")

    conn.execute("INSERT INTO deletedItems (itemID) VALUES (6)")

    conn.execute("INSERT INTO itemNotes (itemID, parentItemID, note) VALUES (4, NULL, ?)",
                 ("Some note text mentioning mindfulness practices",))

    # Creators: item 1 -> Jane Doe; item 2 -> Alex Smith; item 5 -> org (lastName only)
    conn.execute("INSERT INTO creators (creatorID, firstName, lastName) VALUES (1, 'Jane', 'Doe')")
    conn.execute("INSERT INTO creators (creatorID, firstName, lastName) VALUES (2, 'Alex', 'Smith')")
    conn.execute("INSERT INTO creators (creatorID, firstName, lastName) VALUES (3, NULL, 'Big Organization')")
    conn.execute("INSERT INTO itemCreators VALUES (1, 1, 1, 0)")
    conn.execute("INSERT INTO itemCreators VALUES (2, 2, 1, 0)")
    conn.execute("INSERT INTO itemCreators VALUES (5, 3, 1, 0)")

    # Tags: item 1 -> physics; item 2 -> history; item 7 -> physics
    conn.execute("INSERT INTO tags (tagID, name) VALUES (1, 'physics')")
    conn.execute("INSERT INTO tags (tagID, name) VALUES (2, 'history')")
    conn.execute("INSERT INTO itemTags VALUES (1, 1, 0)")
    conn.execute("INSERT INTO itemTags VALUES (2, 2, 0)")
    conn.execute("INSERT INTO itemTags VALUES (7, 1, 0)")

    # Collections: Root (COLLA001) > Child (COLLB001); item 1 filed in Child.
    conn.execute("INSERT INTO collections VALUES (100, 'Root', NULL, 1, 'COLLA001')")
    conn.execute("INSERT INTO collections VALUES (101, 'Child', 100, 1, 'COLLB001')")
    conn.execute("INSERT INTO collectionItems VALUES (101, 1)")

    conn.commit()
    conn.close()
    return {}


def _reader(tmp_path) -> LocalZoteroReader:
    db_path = tmp_path / "zotero.sqlite"
    _build_db(db_path)
    return LocalZoteroReader(db_path=str(db_path))


# ---------------------------------------------------------------------------
# search_items_sql
# ---------------------------------------------------------------------------

def test_search_items_sql_matches_title(tmp_path):
    reader = _reader(tmp_path)
    try:
        results = reader.search_items_sql("Quantum", group_id=0)
    finally:
        reader.close()
    assert results is not None
    keys = {r["key"] for r in results}
    assert "PERS0001" in keys
    assert "PERS0002" not in keys


def test_search_items_sql_matches_creator(tmp_path):
    reader = _reader(tmp_path)
    try:
        results = reader.search_items_sql("Smith", group_id=0)
    finally:
        reader.close()
    keys = {r["key"] for r in results}
    assert keys == {"PERS0002"}
    creators = results[0]["data"]["creators"]
    assert creators == [{"creatorType": "author", "firstName": "Alex", "lastName": "Smith"}]


def test_search_items_sql_org_creator_uses_name_key(tmp_path):
    reader = _reader(tmp_path)
    try:
        results = reader.search_items_sql("Organization", group_id=0)
    finally:
        reader.close()
    assert results
    creators = results[0]["data"]["creators"]
    assert creators == [{"creatorType": "author", "name": "Big Organization"}]


def test_search_items_sql_excludes_deleted_items(tmp_path):
    reader = _reader(tmp_path)
    try:
        results = reader.search_items_sql("Never Appear", group_id=0)
    finally:
        reader.close()
    assert results == []


def test_search_items_sql_default_item_type_excludes_attachments(tmp_path):
    reader = _reader(tmp_path)
    try:
        results = reader.search_items_sql("Ignored Attachment", group_id=0)
    finally:
        reader.close()
    assert results == []


def test_search_items_sql_bare_item_type_includes_only_that_type(tmp_path):
    reader = _reader(tmp_path)
    try:
        results = reader.search_items_sql("Ignored Attachment", item_type="attachment", group_id=0)
    finally:
        reader.close()
    assert {r["key"] for r in results} == {"PERS0003"}


def test_search_items_sql_scopes_to_active_library(tmp_path):
    reader = _reader(tmp_path)
    try:
        personal = reader.search_items_sql("quantum", group_id=0)
        group = reader.search_items_sql("quantum", group_id=GROUP_ID)
    finally:
        reader.close()
    assert {r["key"] for r in personal} == {"PERS0001"}
    assert {r["key"] for r in group} == {"GRP00001"}


def test_search_items_sql_unknown_group_returns_none(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.search_items_sql("quantum", group_id=999999)
    finally:
        reader.close()
    assert result is None


def test_search_items_sql_tag_filter_unsupported_falls_back(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.search_items_sql("quantum", tag=["physics"], group_id=0)
    finally:
        reader.close()
    assert result is None


def test_search_items_sql_boolean_item_type_unsupported_falls_back(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.search_items_sql("quantum", item_type="book || journalArticle", group_id=0)
    finally:
        reader.close()
    assert result is None


def test_search_items_sql_everything_mode_matches_abstract(tmp_path):
    reader = _reader(tmp_path)
    try:
        title_mode = reader.search_items_sql("quantum stuff", qmode="titleCreatorYear", group_id=0)
        everything_mode = reader.search_items_sql("quantum stuff", qmode="everything", group_id=0)
    finally:
        reader.close()
    assert title_mode == []
    assert {r["key"] for r in everything_mode} == {"PERS0001"}


def test_search_items_sql_everything_mode_matches_tag(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.search_items_sql("physics", qmode="everything", group_id=0)
    finally:
        reader.close()
    assert {r["key"] for r in result} == {"PERS0001"}


def test_search_items_sql_everything_mode_matches_note_content(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.search_items_sql("mindfulness", qmode="everything", item_type="note", group_id=0)
    finally:
        reader.close()
    assert {r["key"] for r in result} == {"PERS0004"}


def test_search_items_sql_respects_limit(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.search_items_sql("a", qmode="everything", item_type="journalArticle", limit=1, group_id=0)
    finally:
        reader.close()
    assert len(result) == 1


# ---------------------------------------------------------------------------
# advanced_search_sql
# ---------------------------------------------------------------------------

def test_advanced_search_sql_title_and_year(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.advanced_search_sql(
            conditions=[
                {"field": "title", "operation": "contains", "value": "Quantum"},
                {"field": "year", "operation": "isGreaterThan", "value": "2020"},
            ],
            join_mode="all",
            group_id=0,
        )
    finally:
        reader.close()
    assert {r["key"] for r in result} == {"PERS0001"}


def test_advanced_search_sql_year_excludes_older_item(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.advanced_search_sql(
            conditions=[{"field": "year", "operation": "isGreaterThan", "value": "2020"}],
            group_id=0,
        )
    finally:
        reader.close()
    keys = {r["key"] for r in result}
    assert "PERS0001" in keys
    assert "PERS0002" not in keys  # 2018, excluded


def test_advanced_search_sql_always_excludes_attachments_notes(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.advanced_search_sql(
            conditions=[{"field": "dateAdded", "operation": "contains", "value": "2024"}],
            group_id=0,
        )
    finally:
        reader.close()
    keys = {r["key"] for r in result}
    assert "PERS0003" not in keys  # attachment
    assert "PERS0004" not in keys  # note


def test_advanced_search_sql_unsupported_operation_returns_none(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.advanced_search_sql(
            conditions=[{"field": "title", "operation": "regex", "value": ".*"}],
            group_id=0,
        )
    finally:
        reader.close()
    assert result is None


def test_advanced_search_sql_unsupported_field_returns_none(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.advanced_search_sql(
            conditions=[{"field": "volume", "operation": "is", "value": "3"}],
            group_id=0,
        )
    finally:
        reader.close()
    assert result is None


def test_advanced_search_sql_creator_doesnotcontain_excludes_matching_creator(tmp_path):
    """doesNotContain on creator must not match items that simply have no
    creators at all (mirrors tools/search.py's `_matches_condition`: an
    absent value never satisfies any operator, negated or not)."""
    reader = _reader(tmp_path)
    try:
        result = reader.advanced_search_sql(
            conditions=[{"field": "creator", "operation": "doesNotContain", "value": "Smith"}],
            group_id=0,
        )
    finally:
        reader.close()
    keys = {r["key"] for r in result}
    # PERS0001 (Jane Doe) and PERS0005 (Big Organization) don't contain
    # "Smith" -> doesNotContain matches; PERS0002 (Alex Smith) must be excluded.
    assert "PERS0002" not in keys
    assert "PERS0001" in keys
    assert "PERS0005" in keys
    # Item with NO creators at all (e.g. PERS0003/PERS0004, also excluded by
    # itemType) must not spuriously satisfy the negated operator either.
    assert "PERS0003" not in keys
    assert "PERS0004" not in keys


def test_advanced_search_sql_creator_isnot_is_exact_match_not_substring(tmp_path):
    """isNot compares the WHOLE extracted creator value ("First Last") for
    exact equality — matching tools/search.py's `_compare`'s `left != right` —
    so "isNot 'Smith'" does NOT exclude "Alex Smith" (only an exact full-name
    match would); only "isNot 'Alex Smith'" does."""
    reader = _reader(tmp_path)
    try:
        not_smith = reader.advanced_search_sql(
            conditions=[{"field": "creator", "operation": "isNot", "value": "Smith"}],
            group_id=0,
        )
        not_alex_smith = reader.advanced_search_sql(
            conditions=[{"field": "creator", "operation": "isNot", "value": "Alex Smith"}],
            group_id=0,
        )
    finally:
        reader.close()
    assert "PERS0002" in {r["key"] for r in not_smith}
    assert "PERS0002" not in {r["key"] for r in not_alex_smith}


def test_advanced_search_sql_tag_condition(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.advanced_search_sql(
            conditions=[{"field": "tag", "operation": "is", "value": "physics"}],
            group_id=0,
        )
    finally:
        reader.close()
    assert {r["key"] for r in result} == {"PERS0001"}


def test_advanced_search_sql_collection_condition_includes_subcollection(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.advanced_search_sql(
            conditions=[{"field": "collection", "operation": "is", "value": "COLLA001"}],
            group_id=0,
        )
    finally:
        reader.close()
    # Item 1 is filed in COLLB001, a child of COLLA001 — recursive resolution
    # must include it.
    assert {r["key"] for r in result} == {"PERS0001"}


def test_advanced_search_sql_collection_condition_isnot(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.advanced_search_sql(
            conditions=[
                {"field": "collection", "operation": "isNot", "value": "COLLA001"},
                {"field": "itemType", "operation": "is", "value": "journalArticle"},
            ],
            group_id=0,
        )
    finally:
        reader.close()
    keys = {r["key"] for r in result}
    assert "PERS0001" not in keys
    assert "PERS0002" in keys


def test_advanced_search_sql_collection_unsupported_operation_returns_none(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.advanced_search_sql(
            conditions=[{"field": "collection", "operation": "contains", "value": "COLLA001"}],
            group_id=0,
        )
    finally:
        reader.close()
    assert result is None


def test_advanced_search_sql_unknown_collection_key_returns_none(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.advanced_search_sql(
            conditions=[{"field": "collection", "operation": "is", "value": "NOPE0000"}],
            group_id=0,
        )
    finally:
        reader.close()
    assert result is None


def test_advanced_search_sql_join_mode_any(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.advanced_search_sql(
            conditions=[
                {"field": "title", "operation": "contains", "value": "Quantum"},
                {"field": "title", "operation": "contains", "value": "Classical"},
            ],
            join_mode="any",
            group_id=0,
        )
    finally:
        reader.close()
    assert {r["key"] for r in result} == {"PERS0001", "PERS0002"}


def test_advanced_search_sql_scopes_to_group_library(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.advanced_search_sql(
            conditions=[{"field": "title", "operation": "contains", "value": "quantum"}],
            group_id=GROUP_ID,
        )
    finally:
        reader.close()
    assert {r["key"] for r in result} == {"GRP00001"}


def test_advanced_search_sql_unknown_group_returns_none(tmp_path):
    reader = _reader(tmp_path)
    try:
        result = reader.advanced_search_sql(
            conditions=[{"field": "title", "operation": "contains", "value": "quantum"}],
            group_id=999999,
        )
    finally:
        reader.close()
    assert result is None
