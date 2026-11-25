"""A tiny fake Zotero and an offline embedding, shared by the generator and the tests.

Standard library only: ``make_legacy_indexes.py`` imports it inside the throwaway
environments of old zotero-mcp releases, and the tests import it to rebuild the
very same ``zotero.sqlite`` and ``storage/`` the old releases indexed. Everything
here is deterministic (no clock, no randomness), so rebuilding is byte-for-byte
the same library.

The library (user library 1, nothing else):

``ITEMONE1``  one PDF  (``ATTONEA1``)
``ITEMTWO2``  two PDFs of different sizes, same category (``ATTTWOA2`` small,
              ``ATTTWOB2`` large; 0.17 indexes the large one)
``ITEMTHR3``  no file at all (metadata and an abstract only)
``ITEMFOU4``  one PDF with a long text (``ATTFOUR4``), several passages
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from pathlib import Path

DATE_MODIFIED = "2024-03-01 12:00:00"
DIM = 384
#: Chunking as configured for the old releases, small so every file gives passages.
CHUNKING = {"enabled": True, "chunk_size": 400, "overlap": 50, "max_chunks_per_item": 20}

SCHEMA = """
CREATE TABLE libraries (libraryID INTEGER PRIMARY KEY, type TEXT NOT NULL,
    editable INT NOT NULL, filesEditable INT NOT NULL);
CREATE TABLE groups (groupID INTEGER PRIMARY KEY, libraryID INT NOT NULL UNIQUE,
    name TEXT NOT NULL, description TEXT NOT NULL, version INT NOT NULL);
CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT, itemTypeID INTEGER,
    libraryID INT, dateAdded TEXT, dateModified TEXT, version INT NOT NULL DEFAULT 1);
CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
CREATE TABLE itemData (itemID INTEGER, fieldID INTEGER, valueID INTEGER);
CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
CREATE TABLE itemCreators (itemID INTEGER, creatorID INTEGER, creatorTypeID INTEGER, orderIndex INTEGER);
CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT);
CREATE TABLE creatorTypes (creatorTypeID INTEGER PRIMARY KEY, creatorType TEXT);
CREATE TABLE itemTags (itemID INTEGER, tagID INTEGER, type INTEGER);
CREATE TABLE tags (tagID INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE itemNotes (itemID INTEGER, parentItemID INTEGER, note TEXT, title TEXT);
CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY);
CREATE TABLE collections (collectionID INTEGER PRIMARY KEY, collectionName TEXT,
    parentCollectionID INTEGER, libraryID INTEGER, key TEXT);
CREATE TABLE collectionItems (collectionID INTEGER, itemID INTEGER);
CREATE TABLE itemRelations (itemID INTEGER, predicateID INTEGER, object TEXT);
CREATE TABLE relationPredicates (predicateID INTEGER PRIMARY KEY, predicate TEXT);
CREATE TABLE baseFieldMappingsCombined (itemTypeID INT, baseFieldID INT, fieldID INT,
    PRIMARY KEY (itemTypeID, baseFieldID, fieldID));
CREATE TABLE itemAttachments (itemID INTEGER PRIMARY KEY, parentItemID INT, linkMode INT,
    contentType TEXT, path TEXT);
CREATE TABLE itemAnnotations (itemID INTEGER PRIMARY KEY, parentItemID INT, type INT, text TEXT,
    comment TEXT, color TEXT, pageLabel TEXT, sortIndex TEXT, position TEXT);
"""

_FIELDS = {"title": 1, "abstractNote": 2, "date": 13, "DOI": 26, "publicationTitle": 27, "extra": 28}

# key, item id, title, abstract, creators (first, last), tags, DOI, journal, collection
ITEMS = [
    ("ITEMONE1", 1, "Neural networks for soil moisture retrieval",
     "We train neural networks to retrieve soil moisture from radar backscatter.",
     [("Aino", "Virtanen")], ["remote sensing", "soil"], "10.1000/soil.1", "Journal of Hydrology"),
    ("ITEMTWO2", 2, "Bayesian methods in community ecology",
     "A hierarchical Bayesian model of species abundance across forest plots.",
     [("Jonas", "Berg"), ("Mei", "Tanaka")], ["ecology", "bayesian"], "10.1000/eco.2", "Ecology Letters"),
    ("ITEMTHR3", 3, "A note on lichen taxonomy",
     "A short taxonomic note on crustose lichens without any attached file.",
     [("Sara", "Lindqvist")], ["taxonomy"], "10.1000/lichen.3", "The Bryologist"),
    ("ITEMFOU4", 4, "Glacier retreat in the Alps since 1850",
     "Length records of alpine glaciers show a sustained retreat since the end of the little ice age.",
     [("Luca", "Moretti")], ["glaciology", "climate"], "10.1000/ice.4", "The Cryosphere"),
]

# attachment key, item id, parent item key, file name, text
_SENT = {
    "soil": [
        "Radar backscatter responds to the dielectric constant of wet soil.",
        "The network takes incidence angle, polarisation and vegetation index as inputs.",
        "Validation against in situ probes gives a root mean square error of four percent.",
        "Frozen ground and dense canopy remain the main sources of retrieval error.",
    ],
    "eco": [
        "Species abundance varies strongly between forest plots and across seasons.",
        "A hierarchical model shares information among rare species and plots.",
        "Posterior predictive checks show the model reproduces the observed occupancy.",
        "Priors on the detection probability were chosen from earlier field surveys.",
    ],
    "ice": [
        "Length records from alpine glaciers extend back to the middle of the nineteenth century.",
        "Retreat accelerated after the nineteen eighties as summer temperatures rose.",
        "Debris cover slows the melt of tongues but not of the thinning upper basins.",
        "Photographs and moraine mapping extend the record before systematic measurements.",
        "Mass balance stakes on the Hintereis glacier show a negative balance in almost every year.",
    ],
}


def _text(topic: str, paragraphs: int, tag: str) -> str:
    sentences = _SENT[topic]
    paras = []
    for p in range(paragraphs):
        picked = [sentences[(p + j) % len(sentences)] for j in range(len(sentences))]
        paras.append(f"{tag} section {p + 1}. " + " ".join(picked))
    return "\n".join(paras)


ATTACHMENTS = [
    ("ATTONEA1", 11, "ITEMONE1", "soil.pdf", _text("soil", 3, "Soil moisture")),
    ("ATTTWOA2", 12, "ITEMTWO2", "appendix.pdf", _text("eco", 2, "Ecology appendix")),
    ("ATTTWOB2", 13, "ITEMTWO2", "paper.pdf", _text("eco", 5, "Ecology paper")),
    ("ATTFOUR4", 14, "ITEMFOU4", "glaciers.pdf", _text("ice", 12, "Glacier retreat")),
]

#: item key -> attachment key 0.17 indexes for it (the largest file of the category)
LARGEST = {"ITEMONE1": "ATTONEA1", "ITEMTWO2": "ATTTWOB2", "ITEMFOU4": "ATTFOUR4"}
ITEM_KEYS = [i[0] for i in ITEMS]
ATTACHMENT_KEYS = [a[0] for a in ATTACHMENTS]


def _pdf(text: str) -> bytes:
    """A minimal one-font PDF, a page per twenty-five wrapped lines."""
    lines: list[str] = []
    for para in text.split("\n"):
        words, line = para.split(), ""
        for w in words:
            if len(line) + len(w) + 1 > 85:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        lines.append(line)
    pages = [lines[i : i + 25] for i in range(0, len(lines), 25)]
    objs: list[bytes] = []
    n_pages = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n_pages))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, page in enumerate(pages):
        esc = [ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for ln in page]
        stream = "BT /F1 10 Tf 12 TL 40 760 Td " + " T* ".join(f"({ln}) Tj" for ln in esc) + " ET"
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {5 + 2 * i} 0 R >>".encode()
        )
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream".encode())
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def build(root: Path) -> Path:
    """Write ``root/zotero.sqlite`` and ``root/storage/<KEY>/<file>``; return the sqlite path."""
    root.mkdir(parents=True, exist_ok=True)
    db = root / "zotero.sqlite"
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO libraries VALUES (1, 'user', 1, 1)")
    for type_id, name in [(1, "journalArticle"), (2, "attachment"), (3, "note")]:
        conn.execute("INSERT INTO itemTypes VALUES (?, ?)", (type_id, name))
    for name, field_id in _FIELDS.items():
        conn.execute("INSERT INTO fields VALUES (?, ?)", (field_id, name))
    conn.execute("INSERT INTO creatorTypes VALUES (1, 'author')")
    value_id = creator_id = 0
    tag_ids: dict[str, int] = {}
    for key, item_id, title, abstract, creators, tags, doi, journal in ITEMS:
        conn.execute("INSERT INTO items VALUES (?, ?, 1, 1, '2024-01-01 00:00:00', ?, 5)", (item_id, key, DATE_MODIFIED))
        for field, value in (("title", title), ("abstractNote", abstract), ("DOI", doi),
                             ("publicationTitle", journal), ("date", "2020-05-01 2020")):
            value_id += 1
            conn.execute("INSERT INTO itemDataValues VALUES (?, ?)", (value_id, value))
            conn.execute("INSERT INTO itemData VALUES (?, ?, ?)", (item_id, _FIELDS[field], value_id))
        for order, (first, last) in enumerate(creators):
            creator_id += 1
            conn.execute("INSERT INTO creators VALUES (?, ?, ?)", (creator_id, first, last))
            conn.execute("INSERT INTO itemCreators VALUES (?, ?, 1, ?)", (item_id, creator_id, order))
        for tag in tags:
            if tag not in tag_ids:
                tag_ids[tag] = len(tag_ids) + 1
                conn.execute("INSERT INTO tags VALUES (?, ?)", (tag_ids[tag], tag))
            tag_id = tag_ids[tag]
            conn.execute("INSERT INTO itemTags VALUES (?, ?, 0)", (item_id, tag_id))
    for att_key, att_id, parent_key, name, text in ATTACHMENTS:
        parent_id = next(i[1] for i in ITEMS if i[0] == parent_key)
        conn.execute("INSERT INTO items VALUES (?, ?, 2, 1, '2024-01-01 00:00:00', ?, 5)", (att_id, att_key, DATE_MODIFIED))
        conn.execute(
            "INSERT INTO itemAttachments VALUES (?, ?, 1, 'application/pdf', ?)", (att_id, parent_id, f"storage:{name}")
        )
        folder = root / "storage" / att_key
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(_pdf(text))
    conn.commit()
    conn.close()
    return db


# --- the offline embedding ----------------------------------------------------------


def embed(text: str) -> list[float]:
    """A deterministic 384-dimensional bag-of-words vector (hashed words, unit length).

    Texts that share words are close, so a query like "glacier retreat" finds the
    glacier item, with no model and no network.
    """
    vec = [0.0] * DIM
    for word in re.findall(r"[a-z]+", text.lower()):
        h = int.from_bytes(hashlib.sha1(word.encode()).digest()[:4], "big")
        vec[h % DIM] += 1.0 if (h >> 16) & 1 else -1.0
        vec[(h >> 8) % DIM] += 0.5
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]
