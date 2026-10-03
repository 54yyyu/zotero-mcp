"""Local note/annotation search must stay inside the active library.

``zotero_get_notes(query=...)`` promises "Scope: active library only", and
its API path honours that because pyzotero is scoped to one library. The
local-mode SQLite path queried ``itemNotes``/``itemAnnotations`` with no
``libraryID`` filter, so a personal-library search returned notes and
annotations from every group library synced to the machine (and vice versa
after ``zotero_switch_library``) — keys the follow-up tools then fail to
resolve in the active library.
"""

from __future__ import annotations

import sqlite3

import pytest

from tests._search_corpus import SCHEMA
from zotero_mcp import client as _client
from zotero_mcp import server
from zotero_mcp.local_db import LocalZoteroReader

GROUP_ID = 4242
USER_LIB = 1
GROUP_LIB = 2

EXTRA_SCHEMA = """
-- Zotero's real itemNotes also has a title column; the shared corpus omits it.
ALTER TABLE itemNotes ADD COLUMN title TEXT;
CREATE TABLE itemAttachments (itemID INTEGER PRIMARY KEY, parentItemID INT,
    linkMode INT, contentType TEXT, path TEXT);
CREATE TABLE itemAnnotations (itemID INTEGER PRIMARY KEY, parentItemID INT,
    type INT, text TEXT, comment TEXT, color TEXT, pageLabel TEXT);
"""


class DummyContext:
    def info(self, *_a, **_k):
        return None

    warning = error = info


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "zotero.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA + EXTRA_SCHEMA)
    conn.executemany(
        "INSERT INTO libraries VALUES (?, ?, 1, 1)",
        [(USER_LIB, "user"), (GROUP_LIB, "group")],
    )
    conn.execute(
        "INSERT INTO groups VALUES (?, ?, 'Lab group', '', 1)", (GROUP_ID, GROUP_LIB)
    )
    conn.executemany(
        "INSERT INTO itemTypes VALUES (?, ?)",
        [(1, "journalArticle"), (2, "attachment"), (3, "note"), (4, "annotation")],
    )
    # itemID, key, type, library
    rows = [
        (1, "PERSPAP1", 1, USER_LIB), (2, "PERSNOTE", 3, USER_LIB),
        (3, "PERSATT1", 2, USER_LIB), (4, "PERSANNO", 4, USER_LIB),
        (11, "GRPPAP01", 1, GROUP_LIB), (12, "GRPNOTE1", 3, GROUP_LIB),
        (13, "GRPATT01", 2, GROUP_LIB), (14, "GRPANNO1", 4, GROUP_LIB),
    ]
    conn.executemany(
        "INSERT INTO items VALUES (?, ?, ?, ?, '2024-01-01', '2024-01-01')", rows
    )
    conn.executemany(
        "INSERT INTO itemNotes (itemID, parentItemID, note) VALUES (?, ?, ?)",
        [
            (2, 1, "<p>Personal thoughts on attention mechanisms</p>"),
            (12, 11, "<p>Group discussion of attention mechanisms</p>"),
        ],
    )
    conn.executemany(
        "INSERT INTO itemAttachments VALUES (?, ?, 2, 'application/pdf', NULL)",
        [(3, 1), (13, 11)],
    )
    conn.executemany(
        "INSERT INTO itemAnnotations VALUES (?, ?, 1, ?, '', '#ffd400', '3')",
        [(4, 3, "attention is all you need"), (14, 13, "attention heads")],
    )
    conn.commit()
    conn.close()
    return path


def _keys(results):
    return {r["key"] for r in results}


def test_reader_note_search_is_scoped_to_the_requested_library(db_path):
    with LocalZoteroReader(db_path=str(db_path)) as reader:
        personal = reader.search_notes_local("attention", 20, group_id=0)
        group = reader.search_notes_local("attention", 20, group_id=GROUP_ID)
    assert _keys(personal) == {"PERSNOTE"}
    assert _keys(group) == {"GRPNOTE1"}


def test_reader_annotation_search_is_scoped_to_the_requested_library(db_path):
    with LocalZoteroReader(db_path=str(db_path)) as reader:
        personal = reader.search_annotations_local("attention", 20, group_id=0)
        group = reader.search_annotations_local("attention", 20, group_id=GROUP_ID)
    assert _keys(personal) == {"PERSANNO"}
    assert _keys(group) == {"GRPANNO1"}


@pytest.mark.parametrize(
    "active, expected, foreign",
    [
        ({}, ("PERSNOTE", "PERSANNO"), ("GRPNOTE1", "GRPANNO1")),
        (
            {"library_id": str(GROUP_ID), "library_type": "group"},
            ("GRPNOTE1", "GRPANNO1"),
            ("PERSNOTE", "PERSANNO"),
        ),
    ],
)
def test_search_notes_tool_local_mode_returns_only_active_library(
    monkeypatch, db_path, active, expected, foreign
):
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: True)
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_TYPE", raising=False)
    monkeypatch.setattr(
        "zotero_mcp.local_db.get_local_zotero_reader",
        lambda: LocalZoteroReader(db_path=str(db_path)),
    )
    monkeypatch.setattr(_client, "_active_library_override", dict(active))

    result = server.search_notes(query="attention", limit=20, ctx=DummyContext())

    for key in expected:
        assert key in result
    for key in foreign:
        assert key not in result
