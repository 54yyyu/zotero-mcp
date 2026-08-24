"""Tests for global search across all libraries (#163, phase 2).

`group_id=None` is the "every accessible library" sentinel on the SQLite
metadata backend, matching the convention `_parse_library_id_param` already
uses for `zotero_semantic_search`. Feeds and "My Publications" are excluded,
mirroring `get_key_group_map`'s rule for the semantic index.

Builds on test_sql_search_backend's fixture, which already carries a personal
library, a group library (GROUP_ID) and items in both.
"""

import sqlite3

from conftest import DummyContext
from test_sql_search_backend import GROUP_ID, _build_db

from zotero_mcp.local_db import LocalZoteroReader
from zotero_mcp.tools import search as search_module


def _reader(tmp_path) -> LocalZoteroReader:
    db_path = tmp_path / "zotero.sqlite"
    _build_db(db_path)
    return LocalZoteroReader(db_path=str(db_path))


# ---------------------------------------------------------------------------
# search_items_sql — global scope
# ---------------------------------------------------------------------------

def test_search_items_sql_global_spans_personal_and_group(tmp_path):
    reader = _reader(tmp_path)
    try:
        results = reader.search_items_sql("quantum", group_id=None)
    finally:
        reader.close()
    assert results is not None
    keys = {r["key"] for r in results}
    assert "PERS0001" in keys, "personal-library hit missing from a global search"
    assert "GRP00001" in keys, "group-library hit missing from a global search"


def test_global_results_name_the_library_they_came_from(tmp_path):
    reader = _reader(tmp_path)
    try:
        results = reader.search_items_sql("quantum", group_id=None)
    finally:
        reader.close()
    by_key = {r["key"]: r for r in results}
    assert by_key["PERS0001"]["library"] == {
        "id": 0, "type": "user", "name": "My Library",
    }
    assert by_key["GRP00001"]["library"] == {
        "id": GROUP_ID, "type": "group", "name": "Test Group",
    }


def test_feed_libraries_are_outside_global_scope(tmp_path):
    # Feeds have no group_id equivalent, so a global hit from one could not
    # name its library. get_key_group_map already excludes them from the
    # semantic index; global search follows the same rule.
    db_path = tmp_path / "zotero.sqlite"
    _build_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO libraries VALUES (9, 'feed', 1, 1)")
    conn.execute(
        "INSERT INTO items (itemID, key, itemTypeID, libraryID, dateAdded, dateModified) "
        "VALUES (900, 'FEED0001', 1, 9, '2024-01-01 00:00:00', '2024-01-01 00:00:00')"
    )
    conn.execute("INSERT INTO itemDataValues (valueID, value) VALUES (99001, 'Quantum Feed Item')")
    conn.execute("INSERT INTO itemData (itemID, fieldID, valueID) VALUES (900, 1, 99001)")
    conn.commit()
    conn.close()

    reader = LocalZoteroReader(db_path=str(db_path))
    try:
        results = reader.search_items_sql("quantum", group_id=None)
    finally:
        reader.close()
    assert "FEED0001" not in {r["key"] for r in results}


def test_the_same_paper_in_two_libraries_is_returned_twice(tmp_path):
    # Cross-library duplicates are expected and kept: an item key is unique
    # per library, so two copies are two distinct items, not one to collapse.
    db_path = tmp_path / "zotero.sqlite"
    _build_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO items (itemID, key, itemTypeID, libraryID, dateAdded, dateModified) "
        "VALUES (901, 'GRP00002', 1, 5, '2024-01-01 00:00:00', '2024-01-01 00:00:00')"
    )
    conn.execute(
        "INSERT INTO itemDataValues (valueID, value) VALUES (99002, 'Quantum Networks and Learning')"
    )
    conn.execute("INSERT INTO itemData (itemID, fieldID, valueID) VALUES (901, 1, 99002)")
    conn.commit()
    conn.close()

    reader = LocalZoteroReader(db_path=str(db_path))
    try:
        results = reader.search_items_sql("Quantum Networks and Learning", group_id=None)
    finally:
        reader.close()
    titles = [r["data"]["title"] for r in results]
    assert titles.count("Quantum Networks and Learning") == 2


# ---------------------------------------------------------------------------
# advanced_search_sql — global scope
# ---------------------------------------------------------------------------

def test_advanced_search_sql_global_spans_libraries(tmp_path):
    reader = _reader(tmp_path)
    try:
        results = reader.advanced_search_sql(
            [{"field": "title", "operation": "contains", "value": "quantum"}],
            group_id=None,
        )
    finally:
        reader.close()
    assert results is not None
    assert {r["key"] for r in results} == {"PERS0001", "GRP00001"}


def test_advanced_search_sql_tag_condition_spans_libraries(tmp_path):
    # physics is on a personal item and a group item under one tagID.
    reader = _reader(tmp_path)
    try:
        results = reader.advanced_search_sql(
            [{"field": "tag", "operation": "is", "value": "physics"}], group_id=None
        )
    finally:
        reader.close()
    assert {r["key"] for r in results} == {"PERS0001", "GRP00001"}


def test_advanced_search_sql_refuses_a_collection_condition_when_global(tmp_path):
    # A collection key belongs to exactly one library, so honouring it would
    # answer a one-library question while claiming to have searched them all.
    reader = _reader(tmp_path)
    try:
        scoped = reader.advanced_search_sql(
            [{"field": "collection", "operation": "is", "value": "COLLB001"}], group_id=0
        )
        globally = reader.advanced_search_sql(
            [{"field": "collection", "operation": "is", "value": "COLLB001"}], group_id=None
        )
    finally:
        reader.close()
    assert scoped is not None, "the single-library case must keep working"
    assert globally is None


# ---------------------------------------------------------------------------
# Boolean tag DSL (`tag=["a OR b", "-draft"]`) in SQL
# ---------------------------------------------------------------------------
#
# Every item is titled "... Paper ..." so one query matches them all and each
# test isolates the tag clause. Tags live in a database-wide table with no
# libraryID, so the same tag name is shared by items in both libraries.

_TAG_ITEMS = [
    # (itemID, key, libraryID, title, [tags])
    (1, "TAGPERSA", 1, "Alpha Paper", ["methods"]),
    (2, "TAGPERSB", 1, "Beta Paper", ["methods", "draft"]),
    (3, "TAGPERSC", 1, "Gamma Paper", ["methodology"]),
    (4, "TAGPERSD", 1, "Delta Paper", []),
    (5, "TAGGRPA", 5, "Epsilon Paper", ["methods"]),
]


def _build_tag_db(db_path):
    import _search_corpus

    conn = sqlite3.connect(db_path)
    conn.executescript(_search_corpus.SCHEMA)
    conn.execute("INSERT INTO libraries VALUES (1, 'user', 1, 1)")
    conn.execute("INSERT INTO libraries VALUES (5, 'group', 1, 1)")
    conn.execute(f"INSERT INTO groups VALUES ({GROUP_ID}, 5, 'Test Group', '', 1)")
    conn.execute("INSERT INTO itemTypes (itemTypeID, typeName) VALUES (1, 'journalArticle')")
    conn.execute("INSERT INTO fields (fieldID, fieldName) VALUES (1, 'title')")
    conn.execute("INSERT INTO fields (fieldID, fieldName) VALUES (13, 'date')")

    tag_ids: dict[str, int] = {}
    for item_id, key, library_id, title, tags in _TAG_ITEMS:
        conn.execute(
            "INSERT INTO items (itemID, key, itemTypeID, libraryID, dateAdded, dateModified) "
            "VALUES (?, ?, 1, ?, '2024-01-01 00:00:00', '2024-01-01 00:00:00')",
            (item_id, key, library_id),
        )
        conn.execute("INSERT INTO itemDataValues (valueID, value) VALUES (?, ?)", (item_id, title))
        conn.execute("INSERT INTO itemData (itemID, fieldID, valueID) VALUES (?, 1, ?)",
                     (item_id, item_id))
        for name in tags:
            if name not in tag_ids:
                tag_ids[name] = len(tag_ids) + 1
                conn.execute("INSERT INTO tags (tagID, name) VALUES (?, ?)",
                             (tag_ids[name], name))
            conn.execute("INSERT INTO itemTags VALUES (?, ?, 0)", (item_id, tag_ids[name]))

    conn.commit()
    conn.close()


def _tag_reader(tmp_path) -> LocalZoteroReader:
    db_path = tmp_path / "zotero.sqlite"
    _build_tag_db(db_path)
    return LocalZoteroReader(db_path=str(db_path))


def _tag_search(tmp_path, tag, group_id=0):
    reader = _tag_reader(tmp_path)
    try:
        results = reader.search_items_sql("Paper", tag=tag, limit=50, group_id=group_id)
    finally:
        reader.close()
    return results if results is None else {r["key"] for r in results}


def test_tag_filter_is_served_in_sql_not_punted_to_pyzotero(tmp_path):
    assert _tag_search(tmp_path, ["methods"]) == {"TAGPERSA", "TAGPERSB"}


def test_tag_filter_matches_the_whole_name_not_a_substring(tmp_path):
    assert _tag_search(tmp_path, ["method"]) == set()


def test_tag_entries_are_anded_together(tmp_path):
    assert _tag_search(tmp_path, ["methods", "draft"]) == {"TAGPERSB"}


def test_or_within_one_entry_is_a_disjunction(tmp_path):
    assert _tag_search(tmp_path, ["methods OR methodology"]) == {
        "TAGPERSA", "TAGPERSB", "TAGPERSC"
    }


def test_zotero_pipe_spelling_of_or_agrees_with_the_word_form(tmp_path):
    reader = _tag_reader(tmp_path)
    try:
        pipes = reader.search_items_sql("Paper", tag=["methods || methodology"], limit=50)
        words = reader.search_items_sql("Paper", tag=["methods OR methodology"], limit=50)
    finally:
        reader.close()
    assert {r["key"] for r in pipes} == {r["key"] for r in words}
    assert pipes, "the shared corpus should match something, or this proves nothing"


def test_leading_dash_excludes_and_an_untagged_item_still_matches(tmp_path):
    # Zotero's own `tag=-draft` returns items that lack the tag, which
    # includes items carrying no tags at all. This is NOT the advanced-search
    # `isNot` rule, where a missing value satisfies nothing.
    assert _tag_search(tmp_path, ["-draft"]) == {"TAGPERSA", "TAGPERSC", "TAGPERSD"}


def test_disjunction_and_exclusion_combine(tmp_path):
    assert _tag_search(tmp_path, ["methods OR methodology", "-draft"]) == {
        "TAGPERSA", "TAGPERSC"
    }


def test_wildcard_tag_is_unsupported_and_falls_back(tmp_path):
    assert _tag_search(tmp_path, ["meth%"]) is None


def test_tag_search_spans_libraries_when_global(tmp_path):
    assert _tag_search(tmp_path, ["methods"], group_id=None) == {
        "TAGPERSA", "TAGPERSB", "TAGGRPA"
    }


# ---------------------------------------------------------------------------
# zotero_semantic_search — library scope (#163)
# ---------------------------------------------------------------------------
#
# Before #163 phase 2 this tool searched every indexed library implicitly.
# It now defaults to the active library like the other two search tools, and
# global is opt-in behind the same flag and the same SQLite gate.

class _RecordingSemanticSearch:
    """create_semantic_search() stand-in that records the scope it was asked for."""

    def __init__(self, results=None):
        self.calls: list[dict] = []
        self._results = results or []

    def search(self, query, limit=10, filters=None, group_id=None):
        self.calls.append({"query": query, "group_id": group_id, "filters": filters})
        return {"results": self._results, "total_found": len(self._results)}


def _semantic_env(monkeypatch, tmp_path, sem, *, backend="sqlite", active_group=0):
    """Point the tool at `sem`, with a config file present and the SQLite gate
    satisfied (or not, per `backend`)."""
    from zotero_mcp import client as _client
    from zotero_mcp import semantic_search as semantic_module
    from zotero_mcp import utils as _utils
    from zotero_mcp.tools import _helpers

    config_dir = tmp_path / ".config" / "zotero-mcp"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text("{}")
    monkeypatch.setattr(search_module.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(semantic_module, "create_semantic_search", lambda *a, **kw: sem)
    monkeypatch.setattr(search_module, "_maybe_fire_presearch_sync", lambda _s: None)
    monkeypatch.setattr(_client, "get_active_group_id", lambda: active_group)
    monkeypatch.setattr(_utils, "get_search_backend", lambda: backend)

    db_path = tmp_path / "zotero.sqlite"
    if not db_path.exists():
        _build_db(db_path)
    monkeypatch.setattr(
        _helpers, "get_local_zotero_reader", lambda: LocalZoteroReader(db_path=str(db_path))
    )


def test_semantic_search_defaults_to_the_active_library(monkeypatch, tmp_path):
    sem = _RecordingSemanticSearch()
    _semantic_env(monkeypatch, tmp_path, sem, active_group=GROUP_ID)

    search_module.semantic_search(query="quantum", ctx=DummyContext())

    assert sem.calls[0]["group_id"] == GROUP_ID


def test_semantic_search_all_libraries_drops_the_filter(monkeypatch, tmp_path):
    sem = _RecordingSemanticSearch()
    _semantic_env(monkeypatch, tmp_path, sem, active_group=GROUP_ID)

    search_module.semantic_search(
        query="quantum", search_all_libraries=True, ctx=DummyContext()
    )

    assert sem.calls[0]["group_id"] is None


def test_semantic_global_search_needs_the_sqlite_backend(monkeypatch, tmp_path):
    sem = _RecordingSemanticSearch()
    _semantic_env(monkeypatch, tmp_path, sem, backend="api")

    result = search_module.semantic_search(
        query="quantum", search_all_libraries=True, ctx=DummyContext()
    )

    assert "ZOTERO_SEARCH_BACKEND=sqlite" in result
    assert sem.calls == []


def test_semantic_library_id_and_all_libraries_conflict(monkeypatch, tmp_path):
    sem = _RecordingSemanticSearch()
    _semantic_env(monkeypatch, tmp_path, sem)

    result = search_module.semantic_search(
        query="quantum", library_id=GROUP_ID, search_all_libraries=True, ctx=DummyContext()
    )

    assert "Error" in result
    assert sem.calls == []


def test_semantic_global_results_name_their_library(monkeypatch, tmp_path):
    hit = {
        "item_key": "GRP00001",
        "similarity_score": 0.9,
        "matched_passage": "quantum",
        "metadata": {"group_id": GROUP_ID},
        "zotero_item": {
            "key": "GRP00001",
            "library": {"id": GROUP_ID, "type": "group", "name": "Test Group"},
            "data": {"itemType": "journalArticle", "title": "Group Library Paper",
                     "creators": [], "tags": []},
        },
    }
    sem = _RecordingSemanticSearch(results=[hit])
    _semantic_env(monkeypatch, tmp_path, sem)

    result = search_module.semantic_search(
        query="quantum", search_all_libraries=True, ctx=DummyContext()
    )

    assert f"**Library:** Test Group (groupID={GROUP_ID})" in result


# ---------------------------------------------------------------------------
# get_items_by_keys — hydration for keys in any library
# ---------------------------------------------------------------------------

def test_get_items_by_keys_hydrates_across_libraries(tmp_path):
    reader = _reader(tmp_path)
    try:
        items = reader.get_items_by_keys(["PERS0001", "GRP00001", "NOSUCHKY"])
    finally:
        reader.close()
    assert set(items) == {"PERS0001", "GRP00001"}
    assert items["GRP00001"]["data"]["title"] == "Group Library Paper about quantum"
    assert items["GRP00001"]["library"]["id"] == GROUP_ID
    assert items["PERS0001"]["library"]["type"] == "user"


def test_get_items_by_keys_excludes_trashed_items(tmp_path):
    reader = _reader(tmp_path)
    try:
        items = reader.get_items_by_keys(["DELETEDKEY"])
    finally:
        reader.close()
    assert items == {}


def test_get_items_by_keys_with_no_keys_does_not_query(tmp_path):
    reader = _reader(tmp_path)
    try:
        assert reader.get_items_by_keys([]) == {}
    finally:
        reader.close()
