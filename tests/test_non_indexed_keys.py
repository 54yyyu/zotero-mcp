"""The local reader names the notes and annotations of the library (#604, fix round 2)."""

import sqlite3

from zotero_mcp.local_db import LocalZoteroReader


def test_notes_and_annotations_are_named_by_one_query(tmp_path):
    db = tmp_path / "zotero.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT, itemTypeID INT, libraryID INT);
        CREATE TABLE libraries (libraryID INTEGER PRIMARY KEY, type TEXT);
        CREATE TABLE groups (groupID INTEGER PRIMARY KEY, libraryID INT);
        INSERT INTO libraries VALUES (1, 'user');
        INSERT INTO itemTypes VALUES (1, 'journalArticle'), (2, 'attachment'), (3, 'note'), (4, 'annotation');
        INSERT INTO items VALUES (1, 'ARTICLE1', 1, 1), (2, 'PDFFILE1', 2, 1), (3, 'NOTE0001', 3, 1), (4, 'ANNOT001', 4, 1);
        """
    )
    conn.commit()
    conn.close()

    with LocalZoteroReader(db_path=str(db)) as reader:
        assert reader.get_non_indexed_item_keys() == {"NOTE0001", "ANNOT001"}


def test_keys_are_those_of_the_asked_library_only(tmp_path):
    db = tmp_path / "zotero.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT, itemTypeID INT, libraryID INT);
        CREATE TABLE libraries (libraryID INTEGER PRIMARY KEY, type TEXT);
        CREATE TABLE groups (groupID INTEGER PRIMARY KEY, libraryID INT);
        INSERT INTO libraries VALUES (1, 'user'), (2, 'group');
        INSERT INTO groups VALUES (77, 2);
        INSERT INTO itemTypes VALUES (3, 'note'), (4, 'annotation');
        INSERT INTO items VALUES (1, 'PERSNOTE', 3, 1), (2, 'GROUPANN', 4, 2);
        """
    )
    conn.commit()
    conn.close()

    with LocalZoteroReader(db_path=str(db)) as reader:
        assert reader.get_non_indexed_item_keys(group_id=0) == {"PERSNOTE"}
        assert reader.get_non_indexed_item_keys(group_id=77) == {"GROUPANN"}
        assert reader.get_non_indexed_item_keys(group_id=99) == set()
        assert reader.get_non_indexed_item_keys() == {"PERSNOTE", "GROUPANN"}
