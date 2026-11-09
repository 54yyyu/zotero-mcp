"""The read tools answer from zotero.sqlite alone in local mode.

Runs each read tool under ZOTERO_STRICT_SQLITE against a small synthetic
database, with every HTTP request answered below the guard. A tool that still
reads through the Zotero API fails here with the call site that made the read,
so a new API read path is caught in CI rather than by a user with Zotero closed.
"""

import sqlite3
from unittest.mock import Mock

import pytest

from zotero_mcp import client as _client
from zotero_mcp import library, strict_sqlite

# Column names follow Zotero's userdata.sql; only what local_db reads.
SCHEMA = """
CREATE TABLE libraries (libraryID INTEGER PRIMARY KEY, type TEXT, editable INT, filesEditable INT);
CREATE TABLE groups (groupID INTEGER PRIMARY KEY, libraryID INT, name TEXT, description TEXT, version INT);
CREATE TABLE feeds (libraryID INTEGER PRIMARY KEY, name TEXT, url TEXT, lastUpdate TIMESTAMP,
    lastCheck TIMESTAMP, lastCheckError TEXT, cleanupReadAfter INT, cleanupUnreadAfter INT,
    refreshInterval INT);
CREATE TABLE feedItems (itemID INTEGER PRIMARY KEY, guid TEXT, readTime TIMESTAMP, translatedTime TIMESTAMP);
CREATE TABLE items (itemID INTEGER PRIMARY KEY, itemTypeID INT, dateAdded TIMESTAMP,
    dateModified TIMESTAMP, clientDateModified TIMESTAMP, libraryID INT, key TEXT,
    version INT DEFAULT 0, synced INT DEFAULT 0);
CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
CREATE TABLE itemData (itemID INT, fieldID INT, valueID INT);
CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
CREATE TABLE baseFieldMappingsCombined (itemTypeID INT, baseFieldID INT, fieldID INT);
CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT, fieldMode INT);
CREATE TABLE creatorTypes (creatorTypeID INTEGER PRIMARY KEY, creatorType TEXT);
CREATE TABLE itemCreators (itemID INT, creatorID INT, creatorTypeID INT, orderIndex INT);
CREATE TABLE tags (tagID INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE itemTags (itemID INT, tagID INT, type INT);
CREATE TABLE itemNotes (itemID INTEGER PRIMARY KEY, parentItemID INT, note TEXT, title TEXT);
CREATE TABLE itemAttachments (itemID INTEGER PRIMARY KEY, parentItemID INT, linkMode INT,
    contentType TEXT, charsetID INT, path TEXT, syncState INT, storageModTime INT,
    storageHash TEXT, lastProcessedModificationTime INT, lastRead INT);
CREATE TABLE itemAnnotations (itemID INTEGER PRIMARY KEY, parentItemID INT, type INT,
    authorName TEXT, text TEXT, comment TEXT, color TEXT, pageLabel TEXT, sortIndex TEXT,
    position TEXT, isExternal INT);
CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY, dateDeleted DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE collections (collectionID INTEGER PRIMARY KEY, collectionName TEXT,
    parentCollectionID INT, clientDateModified TIMESTAMP, libraryID INT, key TEXT,
    version INT DEFAULT 0, synced INT DEFAULT 0);
CREATE TABLE collectionItems (collectionID INT, itemID INT, orderIndex INT DEFAULT 0);
CREATE TABLE deletedCollections (collectionID INTEGER PRIMARY KEY, dateDeleted DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE relationPredicates (predicateID INTEGER PRIMARY KEY, predicate TEXT);
CREATE TABLE itemRelations (itemID INT, predicateID INT, object TEXT);
"""

TYPES = {"journalArticle": 1, "book": 2, "attachment": 3, "note": 4, "annotation": 5}
FIELDS = {"title": 1, "date": 2, "DOI": 3, "extra": 4, "abstractNote": 5, "publicationTitle": 6, "url": 7}
STAMP = "2024-05-01 10:00:00"

# (itemID, key, type, parentID or None, fields)
ITEMS = [
    (1, "PAPER001", "journalArticle", None, {
        "title": "Attention Is All You Need", "date": "2017-06-12 2017-06-12",
        "DOI": "10.1000/attention", "publicationTitle": "NeurIPS",
        "extra": "Citation Key: vaswani2017",
    }),
    (2, "PAPER002", "book", None, {"title": "Deep Learning", "extra": "DOI: 10.1000/deepbook"}),
    (3, "PDFATT01", "attachment", 1, {"title": "Full Text PDF"}),
    (4, "NOTE0001", "note", 1, {}),
    (5, "NOTE0002", "note", None, {}),
    (6, "ANNO0001", "annotation", 3, {}),
]


def _pdf_bytes() -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Attention is all you need. The Transformer.", fontsize=12)
    page.draw_rect(fitz.Rect(72, 200, 400, 400), fill=(0.6, 0.6, 0.6))
    page.insert_text((72, 420), "Figure 1: The Transformer architecture.", fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def build_db(data_dir) -> str:
    db = data_dir / "zotero.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO libraries VALUES (1, 'user', 1, 1)")
    conn.executemany("INSERT INTO itemTypes VALUES (?, ?)", [(v, k) for k, v in TYPES.items()])
    conn.executemany("INSERT INTO fields VALUES (?, ?)", [(v, k) for k, v in FIELDS.items()])
    conn.execute("INSERT INTO creatorTypes VALUES (1, 'author')")
    conn.execute("INSERT INTO creators VALUES (1, 'Ashish', 'Vaswani', 0)")
    conn.execute("INSERT INTO itemCreators VALUES (1, 1, 1, 0)")
    conn.execute("INSERT INTO tags VALUES (1, 'transformers')")
    conn.executemany("INSERT INTO itemTags VALUES (?, 1, 0)", [(1,), (2,)])
    conn.execute(
        "INSERT INTO collections (collectionID, collectionName, libraryID, key, clientDateModified)"
        " VALUES (1, 'Reading', 1, 'COLLA001', ?)", (STAMP,),
    )
    conn.executemany("INSERT INTO collectionItems (collectionID, itemID) VALUES (1, ?)", [(1,), (2,)])
    value_id = 0
    for item_id, key, item_type, _parent, fields in ITEMS:
        conn.execute(
            "INSERT INTO items (itemID, itemTypeID, dateAdded, dateModified, clientDateModified,"
            " libraryID, key, version) VALUES (?, ?, ?, ?, ?, 1, ?, 1)",
            (item_id, TYPES[item_type], STAMP, STAMP, STAMP, key),
        )
        for name, value in fields.items():
            value_id += 1
            conn.execute("INSERT INTO itemDataValues VALUES (?, ?)", (value_id, value))
            conn.execute("INSERT INTO itemData VALUES (?, ?, ?)", (item_id, FIELDS[name], value_id))
    conn.execute(
        "INSERT INTO itemAttachments (itemID, parentItemID, linkMode, contentType, path)"
        " VALUES (3, 1, 0, 'application/pdf', 'storage:paper.pdf')"
    )
    conn.executemany(
        "INSERT INTO itemNotes VALUES (?, ?, ?, ?)",
        [(4, 1, "<p>Child note on self-attention</p>", "Child note on self-attention"),
         (5, None, "<p>Standalone reading list</p>", "Standalone reading list")],
    )
    conn.execute(
        "INSERT INTO itemAnnotations VALUES (6, 3, 1, '', 'scaled dot-product attention',"
        " 'key idea', '#ffd400', '1', '00000|000100|00100',"
        " '{\"pageIndex\":0,\"rects\":[[72,100,300,112]]}', 0)"
    )
    conn.commit()
    conn.close()
    storage = data_dir / "storage" / "PDFATT01"
    storage.mkdir(parents=True)
    (storage / "paper.pdf").write_bytes(_pdf_bytes())
    return str(db)


@pytest.fixture
def strict_library(monkeypatch, tmp_path):
    """Local mode over the synthetic database, strict mode on, no network."""
    home = tmp_path / "home"
    (home / ".config" / "zotero-mcp").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    from zotero_mcp import config as _config

    monkeypatch.setattr(_config, "ZOTERO_MCP_CONFIG_PATH", home / ".config" / "zotero-mcp" / "config.json")
    data_dir = tmp_path / "Zotero"
    data_dir.mkdir()
    monkeypatch.setenv("ZOTERO_DB_PATH", build_db(data_dir))
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    for var in ("ZOTERO_BACKEND", "ZOTERO_SEARCH_BACKEND", "ZOTERO_API_KEY", "ZOTERO_LIBRARY_ID",
                "ZOTERO_LIBRARY_TYPE", "ZOTERO_WEBDAV_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_client, "_active_library", {}, raising=False)

    # Anything that gets past the guard is recorded and answered empty, so a
    # third-party request cannot reach the network either.
    http = _client._pyzotero_httpx()
    sent = []

    def handle(self, request):
        sent.append(f"{request.method} {request.url}")
        return http.Response(200, json=[], request=request)

    monkeypatch.setattr(http.HTTPTransport, "handle_request", handle)
    import requests

    def refuse(self, request, *args, **kwargs):
        sent.append(f"{request.method} {request.url}")
        raise requests.ConnectionError("network is off in this test")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", refuse)
    # Better BibTeX is a separate plugin with its own endpoint; most users
    # have none, and item-scoped get_annotations asks it first when it runs.
    from zotero_mcp.better_bibtex_client import ZoteroBetterBibTexAPI

    monkeypatch.setattr(ZoteroBetterBibTexAPI, "is_zotero_running", lambda self: False)
    library.reset_sqlite_reader()
    strict_sqlite.activate()
    yield sent
    strict_sqlite.deactivate()
    library.reset_sqlite_reader()


def _scite_stub(monkeypatch):
    from zotero_mcp import scite_client

    paper = {"tallies": {"supporting": 1, "contrasting": 0, "mentioning": 2}, "editorialNotices": [
        {"status": "retracted", "date": "2020-01-01"}]}
    monkeypatch.setattr(scite_client, "get_papers_batch", lambda dois, **_: {d: paper for d in dois})
    monkeypatch.setattr(scite_client, "get_tallies_batch", lambda dois: {d: paper["tallies"] for d in dois})
    monkeypatch.setattr(scite_client, "get_tally", lambda doi: paper["tallies"])
    monkeypatch.setattr(scite_client, "get_paper", lambda doi: paper)


#: tool -> (call, text the answer must contain). Every module-level tool
#: function is called the way the CLI calls it.
TOOLS = {
    "get_item_metadata": (lambda t: t.retrieval.get_item_metadata("PAPER001", ctx=Mock()), "Attention"),
    "get_item_fulltext": (lambda t: t.retrieval.get_item_fulltext("PAPER001", ctx=Mock()), "Transformer"),
    "get_attachment_path": (lambda t: t.retrieval.get_attachment_path("PAPER001", ctx=Mock()), "paper.pdf"),
    "get_collections": (lambda t: t.retrieval.get_collections(ctx=Mock()), "Reading"),
    "get_collection_items": (lambda t: t.retrieval.get_collection_items("COLLA001", ctx=Mock()), "Deep Learning"),
    "get_item_children": (lambda t: t.retrieval.get_item_children("PAPER001", ctx=Mock()), "PDFATT01"),
    "get_tags": (lambda t: t.retrieval.get_tags(ctx=Mock()), "transformers"),
    "list_libraries": (lambda t: t.retrieval.list_libraries(ctx=Mock()), "My Library"),
    "list_feeds": (lambda t: t.retrieval.list_feeds(ctx=Mock()), "No RSS feeds"),
    "get_recent": (lambda t: t.retrieval.get_recent(ctx=Mock()), "Attention"),
    "get_item_related": (lambda t: t.retrieval.get_item_related("PAPER001", ctx=Mock()), ""),
    "search_items": (lambda t: t.search.search_items("Attention", ctx=Mock()), "PAPER001"),
    "search_items by tag": (lambda t: t.search.search_items(tag="transformers", ctx=Mock()), "Deep Learning"),
    "search_by_citation_key": (lambda t: t.search.search_by_citation_key("vaswani2017", ctx=Mock()), "Attention"),
    "advanced_search": (lambda t: t.search.advanced_search(
        [{"field": "title", "operation": "contains", "value": "Attention"}], ctx=Mock()), "Attention"),
    "get_annotations item": (lambda t: t.annotations.get_annotations(item_key="PAPER001", ctx=Mock()),
                             "scaled dot-product"),
    "get_annotations library": (lambda t: t.annotations.get_annotations(ctx=Mock()), "scaled dot-product"),
    "get_notes item": (lambda t: t.annotations.get_notes(item_key="PAPER001", ctx=Mock()), "self-attention"),
    "get_notes library": (lambda t: t.annotations.get_notes(ctx=Mock()), "Standalone reading list"),
    "search_notes": (lambda t: t.annotations.search_notes(query="attention", ctx=Mock()), "NOTE0001"),
    "get_page_layout": (lambda t: t.annotations.get_page_layout("PDFATT01", 1, ctx=Mock()), "PDFATT01"),
    "read_pdf_pages": (lambda t: t.read_pdf.read_pdf_pages("PAPER001", 1, ctx=Mock()), "Transformer"),
    "get_pdf_outline": (lambda t: t.write.get_pdf_outline("PAPER001", ctx=Mock()), "outline"),
    "find_duplicates": (lambda t: t.write.find_duplicates(ctx=Mock()), "No duplicates"),
    "synthesize_annotations": (lambda t: t.synthesis.synthesize_annotations(ctx=Mock()), "1 highlights"),
    "library_coverage": (lambda t: t.discovery.library_coverage(ctx=Mock()), "With PDF: 1"),
    "scite_enrich_item": (lambda t: t.scite.enrich_item(item_key="PAPER002", ctx=Mock()), "10.1000/deepbook"),
    "scite_enrich_search": (lambda t: t.scite.enrich_search("Attention", ctx=Mock()), "1/1 items"),
    # PAPER002 keeps its DOI in Extra, which SQLite's search rows leave out.
    "scite_enrich_search extra doi": (lambda t: t.scite.enrich_search("Deep", ctx=Mock()), "1/1 items"),
    "scite_check_retractions recent": (lambda t: t.scite.check_retractions(ctx=Mock()), "Deep Learning"),
    "scite_check_retractions tag": (lambda t: t.scite.check_retractions(tag="transformers", ctx=Mock()),
                                    "Attention"),
    "scite_check_retractions collection": (
        lambda t: t.scite.check_retractions(collection="Reading", ctx=Mock()), "Deep Learning"),
}


@pytest.mark.parametrize("name", list(TOOLS))
def test_read_tool_needs_no_api_in_local_mode(strict_library, monkeypatch, name):
    import types

    from zotero_mcp.tools import annotations, discovery, read_pdf, retrieval, scite, search, synthesis, write

    _scite_stub(monkeypatch)
    modules = types.SimpleNamespace(
        annotations=annotations, discovery=discovery, read_pdf=read_pdf, retrieval=retrieval,
        scite=scite, search=search, synthesis=synthesis, write=write,
    )
    call, expected = TOOLS[name]
    text = str(call(modules))
    assert "Error" not in text.split("\n", 1)[0], text
    assert expected in text, text
    assert strict_library == []
