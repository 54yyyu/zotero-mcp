"""`search_notes_local` must apply its limit after dropping markup-only hits.

Zotero 7 notes with inserted citations carry ``data-citation-items`` /
``http://zotero.org/users/...`` URIs in their HTML, so ``LIKE '%zotero%'``
matches every such note. Those rows are discarded by the clean-text check;
if SQL ``LIMIT`` ran first, they consumed the whole budget and real matches
further down were never seen.
"""

import sqlite3

from zotero_mcp.local_db import LocalZoteroReader

CITED = (
    '<div data-schema-version="9"><p>Some thoughts '
    '<span class="citation" data-citation="%7B%22uris%22%3A%5B%22'
    'http%3A%2F%2Fzotero.org%2Fusers%2F1%2Fitems%2FABCD1234%22%5D%7D">'
    "(Smith, 2020)</span></p></div>"
)


def _build(db_path, n_markup_only):
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, itemTypeID INT,
                            key TEXT, libraryID INT);
        CREATE TABLE itemNotes (itemID INTEGER, parentItemID INTEGER,
                                note TEXT, title TEXT);
        CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY);
        CREATE TABLE libraries (libraryID INTEGER PRIMARY KEY, type TEXT,
                                editable INT, filesEditable INT);
        INSERT INTO libraries VALUES (1, 'user', 1, 1);
        CREATE TABLE groups (groupID INTEGER PRIMARY KEY, libraryID INT,
                             name TEXT, description TEXT, version INT);
        CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
        INSERT INTO fields VALUES (1, 'title');
        CREATE TABLE itemData (itemID INT, fieldID INT, valueID INT);
        CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value);
        CREATE TABLE baseFieldMappingsCombined (itemTypeID INT,
                                                baseFieldID INT, fieldID INT);
        """
    )
    for i in range(1, n_markup_only + 1):
        conn.execute("INSERT INTO items VALUES (?, 1, ?, 1)", (i, f"CITE{i:04d}"))
        conn.execute("INSERT INTO itemNotes VALUES (?, NULL, ?, '')", (i, CITED))
    real = n_markup_only + 1
    conn.execute("INSERT INTO items VALUES (?, 1, 'REALHIT1', 1)", (real,))
    conn.execute(
        "INSERT INTO itemNotes VALUES (?, NULL, ?, '')",
        (real, "<p>How to export from Zotero to Word</p>"),
    )
    conn.commit()
    conn.close()


def test_real_match_not_crowded_out_by_markup_only_hits(tmp_path):
    db_path = tmp_path / "zotero.sqlite"
    _build(db_path, n_markup_only=25)
    reader = LocalZoteroReader(db_path=str(db_path))
    try:
        results = reader.search_notes_local("zotero", limit=20)
    finally:
        reader.close()
    assert [r["key"] for r in results] == ["REALHIT1"]


def test_limit_still_caps_results(tmp_path):
    db_path = tmp_path / "zotero.sqlite"
    _build(db_path, n_markup_only=0)
    reader = LocalZoteroReader(db_path=str(db_path))
    try:
        assert len(reader.search_notes_local("zotero", limit=0)) == 0
        assert len(reader.search_notes_local("zotero", limit=5)) == 1
    finally:
        reader.close()
