"""A small builder for a synthetic ``zotero.sqlite`` with every table the
full-record readers touch (attachments, notes, annotations, collections,
relations), for tests that need parents and children.

The schema is the one ``test_strict_sqlite_read_tools`` declares, so the two
cannot drift.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from test_strict_sqlite_read_tools import SCHEMA

STAMP = "2024-05-01 10:00:00"
_CHILD_TABLES = {"attachment", "note", "annotation"}


class FullDb:
    def __init__(self, path: Path):
        self.path = str(path)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.conn.execute("INSERT INTO libraries VALUES (1, 'user', 1, 1)")
        self.conn.execute("INSERT INTO creatorTypes VALUES (1, 'author')")
        self._ids: dict[tuple[str, str], int] = {}
        self._next_item = 1

    def _id(self, table: str, id_col: str, name_col: str, name: str) -> int:
        """The row id for `name` in a lookup table, inserted on first use."""
        if (table, name) not in self._ids:
            new_id = len([k for k in self._ids if k[0] == table]) + 1
            self.conn.execute(
                f"INSERT INTO {table} ({id_col}, {name_col}) VALUES (?, ?)", (new_id, name)
            )
            self._ids[(table, name)] = new_id
        return self._ids[(table, name)]

    def base_field(self, item_type: str, base: str, actual: str) -> None:
        """Record that `item_type` keeps the base field `base` in `actual`."""
        self.conn.execute(
            "INSERT INTO baseFieldMappingsCombined VALUES (?, ?, ?)",
            (self._id("itemTypes", "itemTypeID", "typeName", item_type),
             self._id("fields", "fieldID", "fieldName", base),
             self._id("fields", "fieldID", "fieldName", actual)),
        )

    def collection(self, key: str, name: str | None = None, parent: str | None = None) -> int:
        coll_id = len([k for k in self._ids if k[0] == "collections"]) + 1
        self._ids[("collections", key)] = coll_id
        self.conn.execute(
            "INSERT INTO collections (collectionID, collectionName, parentCollectionID,"
            " libraryID, key, clientDateModified) VALUES (?, ?, ?, 1, ?, ?)",
            (coll_id, name or key, self._ids.get(("collections", parent)), key, STAMP),
        )
        return coll_id

    def item(
        self,
        key: str,
        item_type: str = "journalArticle",
        *,
        parent: int | None = None,
        fields: dict[str, str] | None = None,
        added: str = STAMP,
        modified: str = STAMP,
        note: str = "",
        content_type: str = "application/pdf",
        collections: tuple[str, ...] = (),
        deleted: bool = False,
    ) -> int:
        item_id = self._next_item
        self._next_item += 1
        self.conn.execute(
            "INSERT INTO items (itemID, itemTypeID, dateAdded, dateModified, clientDateModified,"
            " libraryID, key, version) VALUES (?, ?, ?, ?, ?, 1, ?, 1)",
            (item_id, self._id("itemTypes", "itemTypeID", "typeName", item_type),
             added, modified, modified, key),
        )
        for name, value in (fields or {}).items():
            self.conn.execute(
                "INSERT INTO itemData VALUES (?, ?, ?)",
                (item_id, self._id("fields", "fieldID", "fieldName", name),
                 self._id("itemDataValues", "valueID", "value", value)),
            )
        if item_type == "attachment":
            self.conn.execute(
                "INSERT INTO itemAttachments (itemID, parentItemID, linkMode, contentType, path)"
                " VALUES (?, ?, 0, ?, 'storage:file.pdf')", (item_id, parent, content_type),
            )
        elif item_type == "note":
            self.conn.execute(
                "INSERT INTO itemNotes VALUES (?, ?, ?, '')", (item_id, parent, note),
            )
        elif item_type == "annotation":
            self.conn.execute(
                "INSERT INTO itemAnnotations (itemID, parentItemID, type, text, sortIndex,"
                " position, isExternal) VALUES (?, ?, 1, 'quoted', '00000|000000|00000', '{}', 0)",
                (item_id, parent),
            )
        for coll_key in collections:
            self.conn.execute(
                "INSERT INTO collectionItems (collectionID, itemID) VALUES (?, ?)",
                (self._ids[("collections", coll_key)], item_id),
            )
        if deleted:
            self.conn.execute("INSERT INTO deletedItems (itemID) VALUES (?)", (item_id,))
        return item_id

    def done(self) -> str:
        # The lookup tables the readers resolve by name, present even when
        # no item of that kind was added.
        for name in ("attachment", "note", "annotation"):
            self._id("itemTypes", "itemTypeID", "typeName", name)
        for name in ("title", "date", "extra", "citationKey", "abstractNote", "DOI",
                     "publicationTitle"):
            self._id("fields", "fieldID", "fieldName", name)
        self.conn.commit()
        self.conn.close()
        return self.path
