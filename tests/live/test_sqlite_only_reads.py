"""Live: every read operation must still work when neither Zotero API is up.

The scenario this file pins down is the ordinary one for a local-only user:
Zotero desktop is closed (so the local API on port 23119 refuses
connections) and no web credentials are configured (so api.zotero.org is
not an option either) — but ``zotero.sqlite`` is sitting right there on
disk, fully readable. Reading a library in that state needs no server at
all, so every read tool is expected to answer from SQLite rather than
report a connection error.

Both Zotero APIs are made unreachable *deliberately* here rather than
assumed down, so the suite asserts the same thing on a machine where
Zotero happens to be running. The block is scoped to the two Zotero
hosts (127.0.0.1:23119 and api.zotero.org); unrelated HTTP is left alone
so a tool that legitimately calls some other service is not failed for
the wrong reason.

Gated by ZOTERO_MCP_LIVE_TESTS=1 (see conftest.py). Assertions use item /
collection / tag values discovered from the tester's own database at
runtime via plain SQL — never via the code under test, which would make a
broken reader agree with itself.
"""

import os
import sqlite3

import httpx
import pytest
import requests

# Hosts that are "the Zotero API" for the purposes of this test: the
# desktop app's local server (which also serves Better BibTeX's JSON-RPC)
# and the web API.
_LOCAL_API_PORTS = {23119}
_WEB_API_HOSTS = {"api.zotero.org"}


class ZoteroApiDown(Exception):
    """Raised in place of any HTTP call to a Zotero API endpoint."""


def _is_zotero_api_url(url: str) -> bool:
    parsed = httpx.URL(str(url))
    host = parsed.host
    if host in _WEB_API_HOSTS:
        return True
    return host in ("localhost", "127.0.0.1", "::1") and parsed.port in _LOCAL_API_PORTS


# --------------------------------------------------------------------------
# discovery: real values from the tester's own zotero.sqlite, via plain SQL
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sqlite_db_path() -> str:
    """Path to this machine's zotero.sqlite, or skip if there isn't one.

    Deliberately independent of the ``sql_reader`` fixture, which is gated
    on the local Zotero *API* being reachable — the state this suite exists
    to run without.
    """
    from zotero_mcp.config import load_config
    from zotero_mcp.local_db import LocalZoteroReader

    try:
        # Configured path wins; None means "auto-detect", which is what
        # LocalZoteroReader itself does at construction time.
        path = load_config().resolve_zotero_db_path() or LocalZoteroReader()._find_zotero_db()
    except Exception as exc:
        pytest.skip(f"no readable zotero.sqlite on this machine: {exc}")
    if not path or not os.path.exists(path):
        pytest.skip("no readable zotero.sqlite on this machine")
    return str(path)


@pytest.fixture(scope="session")
def facts(sqlite_db_path) -> dict:
    """Real keys/names pulled straight out of the database with SQL.

    Everything the assertions below compare against comes from here, so a
    regression in the reader cannot mask itself by supplying both the
    answer and the expectation.
    """
    conn = sqlite3.connect(f"file:{sqlite_db_path}?immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT libraryID FROM libraries WHERE type='user'").fetchone()
        if row is None:
            pytest.skip("database has no personal library")
        library_id = row[0]

        def one(sql, params=()):
            r = conn.execute(sql, params).fetchone()
            return dict(r) if r is not None else None

        item = one(
            """
            SELECT i.key AS key, idv.value AS title
            FROM items i
            JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
            JOIN itemData id ON id.itemID = i.itemID
            JOIN itemDataValues idv ON idv.valueID = id.valueID
            JOIN fields f ON f.fieldID = id.fieldID AND f.fieldName = 'title'
            WHERE i.libraryID = ?
              AND it.typeName NOT IN ('attachment', 'note', 'annotation')
              AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
            ORDER BY i.itemID DESC LIMIT 1
            """,
            (library_id,),
        )
        pdf_item = one(
            """
            SELECT i.key AS key, idv.value AS title
            FROM items i
            JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
            JOIN itemData id ON id.itemID = i.itemID
            JOIN itemDataValues idv ON idv.valueID = id.valueID
            JOIN fields f ON f.fieldID = id.fieldID AND f.fieldName = 'title'
            WHERE i.libraryID = ?
              AND it.typeName NOT IN ('attachment', 'note', 'annotation')
              AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
              AND i.itemID IN (
                    SELECT parentItemID FROM itemAttachments
                    WHERE parentItemID IS NOT NULL
                      AND contentType = 'application/pdf'
              )
            ORDER BY i.itemID DESC LIMIT 1
            """,
            (library_id,),
        )
        collection = one(
            """
            SELECT c.key AS key, c.collectionName AS name, COUNT(ci.itemID) AS n
            FROM collections c
            JOIN collectionItems ci ON ci.collectionID = c.collectionID
            WHERE c.libraryID = ?
            GROUP BY c.collectionID
            HAVING n > 0
            ORDER BY n ASC LIMIT 1
            """,
            (library_id,),
        )
        tag = one(
            """
            SELECT t.name AS name, COUNT(*) AS n
            FROM tags t
            JOIN itemTags itg ON itg.tagID = t.tagID
            JOIN items i ON i.itemID = itg.itemID
            WHERE i.libraryID = ?
            GROUP BY t.tagID
            ORDER BY n DESC LIMIT 1
            """,
            (library_id,),
        )
        first_tag = one(
            """
            SELECT t.name AS name
            FROM tags t
            JOIN itemTags itg ON itg.tagID = t.tagID
            JOIN items i ON i.itemID = itg.itemID
            WHERE i.libraryID = ?
            ORDER BY t.name ASC LIMIT 1
            """,
            (library_id,),
        )
        note_parent = one(
            """
            SELECT i.key AS key
            FROM itemNotes n
            JOIN items i ON i.itemID = n.parentItemID
            WHERE i.libraryID = ?
              AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
            LIMIT 1
            """,
            (library_id,),
        )
        annotated = one(
            """
            SELECT parent.key AS key
            FROM itemAnnotations a
            JOIN items att ON att.itemID = a.parentItemID
            JOIN itemAttachments ia ON ia.itemID = att.itemID
            JOIN items parent ON parent.itemID = ia.parentItemID
            WHERE parent.libraryID = ?
              AND parent.itemID NOT IN (SELECT itemID FROM deletedItems)
            LIMIT 1
            """,
            (library_id,),
        )
        feed = one("SELECT libraryID FROM libraries WHERE type = 'feed' LIMIT 1")

        if item is None:
            pytest.skip("database has no titled items in the personal library")

        return {
            "library_id": library_id,
            "item": item,
            "pdf_item": pdf_item,
            "collection": collection,
            "tag": tag,
            "first_tag": first_tag,
            "note_parent": note_parent,
            "annotated": annotated,
            "feed_library_id": feed["libraryID"] if feed else None,
        }
    finally:
        conn.close()


# --------------------------------------------------------------------------
# the scenario: both Zotero APIs unreachable, zotero.sqlite readable
# --------------------------------------------------------------------------


@pytest.fixture
def sqlite_only(monkeypatch, sqlite_db_path):
    """Local API refuses connections, web API has no credentials.

    Leaves the SQLite database completely untouched — that is the point:
    the data is there, only the servers are gone.
    """
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_SEARCH_BACKEND", "sqlite")
    monkeypatch.setenv("ZOTERO_DB_PATH", sqlite_db_path)
    for var in ("ZOTERO_LIBRARY_ID", "ZOTERO_API_KEY", "ZOTERO_LIBRARY_TYPE"):
        monkeypatch.delenv(var, raising=False)

    # Any runtime library override left behind by another test would point the
    # tools at a library this scenario says nothing about.
    import zotero_mcp.client as _client

    monkeypatch.setattr(_client, "_active_library_override", {}, raising=False)

    real_httpx_send = httpx.Client.send

    def guarded_httpx_send(self, request, *args, **kwargs):
        if _is_zotero_api_url(request.url):
            raise httpx.ConnectError(f"[test] Zotero API is down: {request.url}")
        return real_httpx_send(self, request, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", guarded_httpx_send)

    real_requests_send = requests.adapters.HTTPAdapter.send

    def guarded_requests_send(self, request, *args, **kwargs):
        if _is_zotero_api_url(request.url):
            raise requests.exceptions.ConnectionError(
                f"[test] Zotero API is down: {request.url}"
            )
        return real_requests_send(self, request, *args, **kwargs)

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", guarded_requests_send)


def assert_served(text: str, label: str) -> str:
    """Fail with the tool's own message when it reported an error."""
    assert isinstance(text, str) and text.strip(), f"{label}: returned no output"
    first = text.strip().splitlines()[0]
    assert not first.lower().startswith("error"), f"{label}: {text[:600]}"
    lowered = text.lower()
    for phrase in ("connection", "connecterror", "failed to establish", "refused"):
        assert phrase not in lowered, f"{label}: leaked a connection failure: {text[:600]}"
    return text


# --------------------------------------------------------------------------
# item reads
# --------------------------------------------------------------------------


def test_get_item_metadata_markdown(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.retrieval import get_item_metadata

    key = facts["item"]["key"]
    out = assert_served(
        get_item_metadata(item_key=key, ctx=dummy_ctx), "get_item_metadata"
    )
    assert facts["item"]["title"][:40] in out


def test_get_item_metadata_json(sqlite_only, facts, dummy_ctx):
    import json

    from zotero_mcp.tools.retrieval import get_item_metadata

    key = facts["item"]["key"]
    out = assert_served(
        get_item_metadata(item_key=key, format="json", ctx=dummy_ctx),
        "get_item_metadata(json)",
    )
    parsed = json.loads(out)
    assert parsed["data"]["key"] == key


def test_get_item_metadata_bibtex(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.retrieval import get_item_metadata

    out = assert_served(
        get_item_metadata(item_key=facts["item"]["key"], format="bibtex", ctx=dummy_ctx),
        "get_item_metadata(bibtex)",
    )
    assert out.lstrip().startswith("@")


def test_get_item_children(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.retrieval import get_item_children

    if facts["pdf_item"] is None:
        pytest.skip("library has no item with a PDF attachment")
    out = assert_served(
        get_item_children(item_key=facts["pdf_item"]["key"], ctx=dummy_ctx),
        "get_item_children",
    )
    assert "attachment" in out.lower()


def test_get_item_related(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.retrieval import get_item_related

    assert_served(
        get_item_related(item_key=facts["item"]["key"], ctx=dummy_ctx),
        "get_item_related",
    )


def test_get_attachment_path(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.retrieval import get_attachment_path

    if facts["pdf_item"] is None:
        pytest.skip("library has no item with a PDF attachment")
    out = assert_served(
        get_attachment_path(item_key=facts["pdf_item"]["key"], ctx=dummy_ctx),
        "get_attachment_path",
    )
    assert "Local path" in out


@pytest.mark.timeout(120)
def test_get_item_fulltext(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.retrieval import get_item_fulltext

    if facts["pdf_item"] is None:
        pytest.skip("library has no item with a PDF attachment")
    assert_served(
        get_item_fulltext(item_key=facts["pdf_item"]["key"], ctx=dummy_ctx),
        "get_item_fulltext",
    )


# --------------------------------------------------------------------------
# library structure
# --------------------------------------------------------------------------


def test_get_collections(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.retrieval import get_collections

    if facts["collection"] is None:
        pytest.skip("library has no non-empty collection")
    out = assert_served(get_collections(limit=5000, ctx=dummy_ctx), "get_collections")
    assert facts["collection"]["key"] in out


def test_get_collection_items(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.retrieval import get_collection_items

    if facts["collection"] is None:
        pytest.skip("library has no non-empty collection")
    out = assert_served(
        get_collection_items(collection_key=facts["collection"]["key"], ctx=dummy_ctx),
        "get_collection_items",
    )
    assert facts["collection"]["name"][:30] in out


def test_search_collections(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.write import search_collections

    if facts["collection"] is None:
        pytest.skip("library has no non-empty collection")
    name = facts["collection"]["name"]
    out = assert_served(
        search_collections(query=name, ctx=dummy_ctx), "search_collections"
    )
    assert facts["collection"]["key"] in out


def test_get_tags(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.retrieval import get_tags

    if facts["first_tag"] is None:
        pytest.skip("library has no tags")
    out = assert_served(get_tags(limit=5000, ctx=dummy_ctx), "get_tags")
    # Alphabetically first, so it is inside the display cap whatever the
    # library's tag count.
    assert facts["first_tag"]["name"] in out


def test_list_libraries(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.retrieval import list_libraries

    assert_served(list_libraries(ctx=dummy_ctx), "list_libraries")


def test_list_feeds(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.retrieval import list_feeds

    if facts["feed_library_id"] is None:
        pytest.skip("library has no RSS feeds")
    assert_served(list_feeds(ctx=dummy_ctx), "list_feeds")


def test_get_feed_items(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.retrieval import get_feed_items

    if facts["feed_library_id"] is None:
        pytest.skip("library has no RSS feeds")
    assert_served(
        get_feed_items(library_id=facts["feed_library_id"], ctx=dummy_ctx),
        "get_feed_items",
    )


def test_get_recent(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.retrieval import get_recent

    assert_served(get_recent(limit=5, ctx=dummy_ctx), "get_recent")


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------


@pytest.mark.timeout(120)
def test_search_items(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.search import search_items

    query = facts["item"]["title"].split()[0]
    assert_served(
        search_items(query=query, limit=5, ctx=dummy_ctx), "search_items"
    )


@pytest.mark.timeout(120)
def test_search_by_tag(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.search import search_by_tag

    if facts["tag"] is None:
        pytest.skip("library has no tags")
    assert_served(
        search_by_tag(tag=[facts["tag"]["name"]], limit=5, ctx=dummy_ctx),
        "search_by_tag",
    )


@pytest.mark.timeout(120)
def test_advanced_search(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.search import advanced_search

    out = assert_served(
        advanced_search(
            conditions=[
                {"field": "title", "operation": "contains",
                 "value": facts["item"]["title"].split()[0]}
            ],
            limit=10,
            ctx=dummy_ctx,
        ),
        "advanced_search",
    )
    assert out.strip()


# --------------------------------------------------------------------------
# notes and annotations
# --------------------------------------------------------------------------


def test_get_notes_for_item(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.annotations import get_notes_tool

    if facts["note_parent"] is None:
        pytest.skip("library has no child notes")
    assert_served(
        get_notes_tool(item_key=facts["note_parent"]["key"], ctx=dummy_ctx),
        "get_notes(item_key)",
    )


def test_search_notes(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.annotations import get_notes_tool

    if facts["note_parent"] is None:
        pytest.skip("library has no child notes")
    assert_served(
        get_notes_tool(query="the", limit=3, ctx=dummy_ctx), "get_notes(query)"
    )


@pytest.mark.timeout(120)
def test_get_annotations(sqlite_only, facts, dummy_ctx):
    from zotero_mcp.tools.annotations import get_annotations

    if facts["annotated"] is None:
        pytest.skip("library has no annotations")
    assert_served(
        get_annotations(item_key=facts["annotated"]["key"], ctx=dummy_ctx),
        "get_annotations",
    )


@pytest.mark.timeout(120)
def test_get_annotations_json(sqlite_only, facts, dummy_ctx):
    """JSON output resolves annotation -> attachment -> paper context.

    A separate test because that two-hop resolution runs only for
    ``format="json"`` and for library-wide listings — the markdown
    item-scoped path above skips it entirely, so it would otherwise go
    unexercised.
    """
    import json

    from zotero_mcp.tools.annotations import get_annotations

    if facts["annotated"] is None:
        pytest.skip("library has no annotations")
    out = assert_served(
        get_annotations(item_key=facts["annotated"]["key"], format="json", ctx=dummy_ctx),
        "get_annotations(json)",
    )
    records = json.loads(out)
    assert isinstance(records, list) and records, "expected annotation records"
