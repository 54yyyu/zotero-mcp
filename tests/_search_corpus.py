"""One declaration of a test corpus, materialised for both search backends.

`zotero_advanced_search` has two implementations — direct SQL against
`zotero.sqlite`, and pyzotero-plus-Python-filtering — and they must answer
every condition identically. Testing that needs the *same* items in both
shapes, so this module declares the corpus once and builds it two ways:

* :func:`build_sqlite` writes a synthetic `zotero.sqlite` for the SQL backend.
* :func:`build_api_items` emits the pyzotero item dicts the API path filters.

Keeping one declaration is the point. If the two corpora could drift, a parity
test could pass while comparing different libraries.

The items are chosen to provoke the divergences that actually occurred while
#417 was in review: accented names that ASCII `LIKE` folding misses, values
containing `LIKE` metacharacters, and items with no creators or tags at all,
which is where the negated operators stop agreeing if the "empty satisfies
nothing" rule is implemented only once.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

GROUP_ID = 6015547
USER_LIBRARY_ID = 1
GROUP_LIBRARY_ID = 5

_ITEM_TYPE_IDS = {"journalArticle": 1, "attachment": 2, "note": 3, "book": 4}
_FIELD_IDS = {"title": 1, "abstractNote": 2, "date": 13, "DOI": 26, "publicationTitle": 27}

#: Schema shared with ``test_sql_search_backend._build_db`` so the two fixtures
#: cannot drift apart. A subset of Zotero's real schema — only what the search
#: backend reads.
SCHEMA = """
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


@dataclass
class Item:
    """One corpus item, in backend-neutral form."""

    key: str
    item_type: str = "journalArticle"
    title: str | None = None
    date: str | None = None
    """Raw Zotero storage form. Multipart dates are ``"<ISO> <display text>"``."""
    abstract: str | None = None
    doi: str | None = None
    pub_title: str | None = None
    creators: list[tuple[str | None, str]] = field(default_factory=list)
    """``(firstName, lastName)``; ``firstName=None`` means a single-field name."""
    tags: list[str] = field(default_factory=list)
    collections: list[str] = field(default_factory=list)
    library_id: int = USER_LIBRARY_ID
    date_added: str = "2024-01-01 00:00:00"
    date_modified: str = "2024-01-01 00:00:00"

    @property
    def display_date(self) -> str:
        """What the Zotero API exposes: the display half of a multipart date.

        Zotero stores ``"2016-10-01 October 1, 2016"`` but the API returns only
        ``"October 1, 2016"``. The SQL backend can see the ISO prefix and the
        API path cannot — the one deliberate divergence between them.
        """
        if not self.date:
            return ""
        parts = self.date.split(" ", 1)
        if len(parts) == 2 and len(parts[0]) == 10 and parts[0][4] == "-":
            return parts[1]
        return self.date


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------

CORPUS: list[Item] = [
    # --- accented vs unaccented creators: the case that returned 4 hits via
    # SQL and 15 via the API on a real library before normalization ---
    Item("ACCENT01", title="Perception and Attention",
         creators=[("Hans", "Müller")], tags=["méthode"], date="2016-10-01 October 1, 2016"),
    Item("ACCENT02", title="A Replication Study",
         creators=[("Hans", "Muller")], tags=["method"], date="2018-03-02 March 2, 2018"),
    Item("ACCENT03", title="Uppercase Umlaut Author",
         creators=[("Ingrid", "MÜLLER")], tags=["Méthode"], date="2019"),
    Item("ACCENT04", title="Nordic Vowel", creators=[("Åsa", "Ångström")], date="2020"),
    Item("ACCENT05", title="Transliterated Name", creators=[("Wei", "王")], date="2021"),

    # --- LIKE metacharacters: '%' and '_' must be literals, not wildcards ---
    Item("PCTSIGN1", title="Growth of 50% in Adoption", date="2022"),
    Item("PCTSIGN2", title="Underscore_Separated_Title", date="2022"),
    Item("PCTSIGN3", title="A Plain Title With Neither", date="2022"),

    # --- dashes and spacing, which _normalize_for_search folds ---
    Item("DASHED01", title="Cladder-Micus Effect", creators=[("Mira", "Cladder-Micus")]),
    Item("DASHED02", title="En Dash – Study", creators=[("Leo", "Van Der Berg")]),

    # --- empty fields: negated operators must not sweep these in ---
    Item("NOCREAT1", title="Anonymous Report", tags=["policy"]),
    Item("NOTAGS01", title="Untagged Paper", creators=[("Ada", "Lovelace")]),
    Item("BARE0001", title="Nothing But A Title"),

    # --- ordinary items, for range operators and general matching ---
    Item("PLAIN001", title="Quantum Networks and Learning", date="2024-01-15 January 15, 2024",
         abstract="A paper about quantum stuff", doi="10.1/quantum",
         pub_title="Journal of Quantum", creators=[("Jane", "Doe")], tags=["physics"],
         collections=["COLLB001"], date_modified="2024-06-01 00:00:00"),
    Item("PLAIN002", title="Classical Literature Review", date="2018-05-01 May 1, 2018",
         creators=[("Alex", "Smith")], tags=["history"], pub_title="Review of Letters"),
    Item("ORGAUTH1", title="Org Author Paper", creators=[(None, "Big Organization")],
         date="2015"),
    Item("BOOKITM1", item_type="book", title="A Monograph", date="2012",
         creators=[("Ida", "Nordberg")]),

    # --- excluded from results by both backends ---
    Item("ATTACH01", item_type="attachment", title="Ignored Attachment"),
    Item("NOTEITM1", item_type="note"),
    Item("DELETED1", title="Should Never Appear"),

    # --- group library, to keep library scoping honest ---
    Item("GRPITEM1", title="Group Library Paper about quantum",
         library_id=GROUP_LIBRARY_ID, tags=["physics"]),
]

DELETED_KEYS = {"DELETED1"}

#: ``key -> (collectionKey, parentCollectionKey|None)``
COLLECTIONS = {"COLLA001": None, "COLLB001": "COLLA001"}

#: Item types the search tools never return.
EXCLUDED_TYPES = {"attachment", "note", "annotation"}


def visible_keys(library_id: int = USER_LIBRARY_ID) -> set[str]:
    """Keys both backends should be able to return for *library_id*."""
    return {
        it.key
        for it in CORPUS
        if it.library_id == library_id
        and it.key not in DELETED_KEYS
        and it.item_type not in EXCLUDED_TYPES
    }


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_sqlite(db_path: Path, items: list[Item] | None = None) -> None:
    """Materialise the corpus as a synthetic ``zotero.sqlite``."""
    items = CORPUS if items is None else items
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    conn.execute(f"INSERT INTO libraries VALUES ({USER_LIBRARY_ID}, 'user', 1, 1)")
    conn.execute(f"INSERT INTO libraries VALUES ({GROUP_LIBRARY_ID}, 'group', 1, 1)")
    conn.execute(
        f"INSERT INTO groups VALUES ({GROUP_ID}, {GROUP_LIBRARY_ID}, 'Test Group', '', 1)"
    )
    conn.executemany(
        "INSERT INTO itemTypes (itemTypeID, typeName) VALUES (?, ?)",
        [(v, k) for k, v in _ITEM_TYPE_IDS.items()],
    )
    conn.executemany(
        "INSERT INTO fields (fieldID, fieldName) VALUES (?, ?)",
        [(v, k) for k, v in _FIELD_IDS.items()],
    )
    conn.execute("INSERT INTO creatorTypes VALUES (1, 'author')")

    for coll_key, parent_key in COLLECTIONS.items():
        coll_id = 100 + list(COLLECTIONS).index(coll_key)
        parent_id = (
            None if parent_key is None else 100 + list(COLLECTIONS).index(parent_key)
        )
        conn.execute(
            "INSERT INTO collections VALUES (?, ?, ?, ?, ?)",
            (coll_id, coll_key, parent_id, USER_LIBRARY_ID, coll_key),
        )

    creator_ids: dict[tuple[str | None, str], int] = {}
    tag_ids: dict[str, int] = {}

    for index, item in enumerate(items, start=1):
        conn.execute(
            "INSERT INTO items (itemID, key, itemTypeID, libraryID, dateAdded, dateModified) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (index, item.key, _ITEM_TYPE_IDS[item.item_type], item.library_id,
             item.date_added, item.date_modified),
        )
        value_id = index * 100
        for field_name, value in (
            ("title", item.title), ("abstractNote", item.abstract),
            ("date", item.date), ("DOI", item.doi),
            ("publicationTitle", item.pub_title),
        ):
            if value is None:
                continue
            value_id += 1
            conn.execute("INSERT INTO itemDataValues (valueID, value) VALUES (?, ?)",
                         (value_id, value))
            conn.execute("INSERT INTO itemData (itemID, fieldID, valueID) VALUES (?, ?, ?)",
                         (index, _FIELD_IDS[field_name], value_id))

        for order, creator in enumerate(item.creators):
            if creator not in creator_ids:
                creator_ids[creator] = len(creator_ids) + 1
                conn.execute(
                    "INSERT INTO creators (creatorID, firstName, lastName) VALUES (?, ?, ?)",
                    (creator_ids[creator], creator[0], creator[1]),
                )
            conn.execute("INSERT INTO itemCreators VALUES (?, ?, 1, ?)",
                         (index, creator_ids[creator], order))

        for tag in item.tags:
            if tag not in tag_ids:
                tag_ids[tag] = len(tag_ids) + 1
                conn.execute("INSERT INTO tags (tagID, name) VALUES (?, ?)",
                             (tag_ids[tag], tag))
            conn.execute("INSERT INTO itemTags VALUES (?, ?, 0)", (index, tag_ids[tag]))

        for coll_key in item.collections:
            conn.execute("INSERT INTO collectionItems VALUES (?, ?)",
                         (100 + list(COLLECTIONS).index(coll_key), index))

        if item.item_type == "note":
            conn.execute(
                "INSERT INTO itemNotes (itemID, parentItemID, note) VALUES (?, NULL, ?)",
                (index, "Some note text mentioning mindfulness practices"),
            )
        if item.key in DELETED_KEYS:
            conn.execute("INSERT INTO deletedItems (itemID) VALUES (?)", (index,))

    conn.commit()
    conn.close()


def build_api_items(
    items: list[Item] | None = None, library_id: int = USER_LIBRARY_ID
) -> list[dict]:
    """Materialise the corpus as the pyzotero item dicts the API path filters.

    Trashed items are omitted rather than flagged: the web API does not return
    them from ``items()`` either, which is why the API path has no
    deleted-item check of its own.
    """
    items = CORPUS if items is None else items
    out: list[dict] = []
    for item in items:
        if item.library_id != library_id or item.key in DELETED_KEYS:
            continue
        creators = []
        for first, last in item.creators:
            if first is None:
                creators.append({"creatorType": "author", "name": last})
            else:
                creators.append(
                    {"creatorType": "author", "firstName": first, "lastName": last}
                )
        out.append(
            {
                "key": item.key,
                "data": {
                    "key": item.key,
                    "itemType": item.item_type,
                    "title": item.title or "",
                    # The API exposes only the display half of a multipart date.
                    "date": item.display_date,
                    "dateAdded": item.date_added,
                    "dateModified": item.date_modified,
                    "abstractNote": item.abstract or "",
                    "DOI": item.doi or "",
                    "publicationTitle": item.pub_title or "",
                    "creators": creators,
                    "tags": [{"tag": t} for t in item.tags],
                    "collections": list(item.collections),
                },
            }
        )
    return out
