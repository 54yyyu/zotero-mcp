"""The local reader names the notes and annotations of the library (#604, fix round 2)."""

import sqlite3

from zotero_mcp.local_db import LocalZoteroReader


def test_notes_and_annotations_are_named_by_one_query(tmp_path):
    db = tmp_path / "zotero.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT, itemTypeID INT);
        INSERT INTO itemTypes VALUES (1, 'journalArticle'), (2, 'attachment'), (3, 'note'), (4, 'annotation');
        INSERT INTO items VALUES (1, 'ARTICLE1', 1), (2, 'PDFFILE1', 2), (3, 'NOTE0001', 3), (4, 'ANNOT001', 4);
        """
    )
    conn.commit()
    conn.close()

    with LocalZoteroReader(db_path=str(db)) as reader:
        assert reader.get_non_indexed_item_keys() == {"NOTE0001", "ANNOT001"}
