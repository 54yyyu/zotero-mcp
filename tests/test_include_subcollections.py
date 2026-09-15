"""`include_subcollections` across the four collection-scoped tools.

Zotero's own advanced-search window has a "Search subcollections" checkbox,
unchecked by default, and these tools had no equivalent — a collection scope
always meant direct membership. That was inconsistent in one direction already:
`LocalZoteroReader.get_items_with_text(collection_keys=...)`, which scopes the
semantic index, has always expanded subcollections.

The tree used throughout:

    Root (COLLROOT)
    ├── Child A (COLLCHDA)
    │   └── Grandchild (COLLGCHD)
    └── Child B (COLLCHDB)
    Unrelated (COLLUNRL)
"""

from __future__ import annotations

import pytest
from conftest import DummyContext

from zotero_mcp import client as _client
from zotero_mcp import server
from zotero_mcp.tools import _helpers

ROOT, CHILD_A, GRANDCHILD, CHILD_B, UNRELATED = (
    "COLLROOT", "COLLCHDA", "COLLGCHD", "COLLCHDB", "COLLUNRL",
)

COLLECTIONS = [
    {"key": ROOT, "data": {"name": "Root", "parentCollection": False}},
    {"key": CHILD_A, "data": {"name": "Child A", "parentCollection": ROOT}},
    {"key": GRANDCHILD, "data": {"name": "Grandchild", "parentCollection": CHILD_A}},
    {"key": CHILD_B, "data": {"name": "Child B", "parentCollection": ROOT}},
    {"key": UNRELATED, "data": {"name": "Unrelated", "parentCollection": False}},
]

# One item per collection, plus one filed in two of them.
ITEMS_BY_COLLECTION = {
    ROOT: ["ITEMROOT"],
    CHILD_A: ["ITEMCHDA", "ITEMBOTH"],
    GRANDCHILD: ["ITEMGCHD"],
    CHILD_B: ["ITEMCHDB", "ITEMBOTH"],
    UNRELATED: ["ITEMUNRL"],
}


def _item(key: str, collections: list[str]) -> dict:
    return {
        "key": key,
        "data": {
            "key": key,
            "itemType": "journalArticle",
            "title": f"Title {key}",
            "date": "2024",
            "creators": [],
            "tags": [{"tag": "shared"}],
            "collections": collections,
            "abstractNote": "",
        },
    }


def _collections_for(item_key: str) -> list[str]:
    return [c for c, keys in ITEMS_BY_COLLECTION.items() if item_key in keys]


ALL_ITEMS = [
    _item(k, _collections_for(k))
    for k in sorted({key for keys in ITEMS_BY_COLLECTION.values() for key in keys})
]


class _FakeZotero:
    """Serves the tree above; records which collections were queried."""

    def __init__(self):
        self.collection_items_calls: list[str] = []
        self.collections_calls = 0

    def collections(self, *args, **kwargs):
        self.collections_calls += 1
        return list(COLLECTIONS)

    def collection(self, key):
        for coll in COLLECTIONS:
            if coll["key"] == key:
                return coll
        raise Exception(f"Collection not found: {key}")

    def collection_items(self, key, *args, **kwargs):
        self.collection_items_calls.append(key)
        return [_item(k, _collections_for(k)) for k in ITEMS_BY_COLLECTION.get(key, [])]

    def add_parameters(self, **kwargs):
        pass

    def items(self, start: int = 0, limit: int = 100, **kwargs):
        return ALL_ITEMS[start : start + limit]


@pytest.fixture
def zot(monkeypatch):
    fake = _FakeZotero()
    monkeypatch.setattr(_client, "get_zotero_client", lambda: fake)
    monkeypatch.setattr(_client, "get_active_group_id", lambda: 0)
    return fake


def _keys_in(text: str) -> set[str]:
    return {k for k in
            {key for keys in ITEMS_BY_COLLECTION.values() for key in keys}
            if k in text}


# ---------------------------------------------------------------------------
# The pure tree walk
# ---------------------------------------------------------------------------

def test_descendants_are_breadth_first_and_include_the_root():
    assert _helpers.collection_descendants(COLLECTIONS, ROOT) == [
        ROOT, CHILD_A, CHILD_B, GRANDCHILD,
    ]


def test_descendants_of_a_leaf_is_just_itself():
    assert _helpers.collection_descendants(COLLECTIONS, GRANDCHILD) == [GRANDCHILD]


def test_descendants_reaches_arbitrary_depth():
    deep = [{"key": "C0", "data": {"parentCollection": False}}] + [
        {"key": f"C{i}", "data": {"parentCollection": f"C{i - 1}"}} for i in range(1, 12)
    ]
    assert _helpers.collection_descendants(deep, "C0") == [f"C{i}" for i in range(12)]


def test_descendants_of_unknown_key_returns_that_key_alone():
    """The caller's own 'collection not found' handling should decide, not this."""
    assert _helpers.collection_descendants(COLLECTIONS, "NOSUCHKY") == ["NOSUCHKY"]


def test_descendants_terminates_on_a_parent_cycle():
    """A corrupt or half-synced database can contain one; don't hang on it."""
    cyclic = [
        {"key": "CYCA", "data": {"parentCollection": "CYCB"}},
        {"key": "CYCB", "data": {"parentCollection": "CYCA"}},
    ]
    assert sorted(_helpers.collection_descendants(cyclic, "CYCA")) == ["CYCA", "CYCB"]


def test_descendants_tolerates_missing_keys_and_data():
    messy = COLLECTIONS + [{"data": {"parentCollection": ROOT}}, {"key": "NODATA00"}]
    assert _helpers.collection_descendants(messy, ROOT) == [ROOT, CHILD_A, CHILD_B, GRANDCHILD]


def test_expand_scope_costs_no_api_call_when_not_requested(zot):
    assert _helpers.expand_collection_scope(zot, ROOT, False) == [ROOT]
    assert zot.collections_calls == 0


def test_expand_scope_fetches_once_when_requested(zot):
    assert set(_helpers.expand_collection_scope(zot, ROOT, True)) == {
        ROOT, CHILD_A, CHILD_B, GRANDCHILD,
    }
    assert zot.collections_calls == 1


# ---------------------------------------------------------------------------
# get_collection_items
# ---------------------------------------------------------------------------

def test_get_collection_items_defaults_to_direct_membership(zot):
    out = server.get_collection_items(collection_key=ROOT, ctx=DummyContext())
    assert _keys_in(out) == {"ITEMROOT"}
    assert zot.collection_items_calls == [ROOT]


def test_get_collection_items_includes_the_whole_subtree(zot):
    out = server.get_collection_items(
        collection_key=ROOT, include_subcollections=True, ctx=DummyContext()
    )
    assert _keys_in(out) == {"ITEMROOT", "ITEMCHDA", "ITEMCHDB", "ITEMGCHD", "ITEMBOTH"}
    assert "ITEMUNRL" not in out


def test_get_collection_items_deduplicates_an_item_in_two_subcollections(zot):
    """ITEMBOTH is filed in both Child A and Child B; it must appear once."""
    out = server.get_collection_items(
        collection_key=ROOT, include_subcollections=True, ctx=DummyContext()
    )
    assert out.count("**Item Key:** ITEMBOTH") == 1
    assert "(5 items)" in out


def test_get_collection_items_reports_an_empty_subtree_as_such(zot):
    out = server.get_collection_items(
        collection_key=UNRELATED, include_subcollections=True, ctx=DummyContext()
    )
    assert "ITEMUNRL" in out  # sanity: this one is not empty


# ---------------------------------------------------------------------------
# search_items / search_by_tag
# ---------------------------------------------------------------------------

def test_search_items_scope_defaults_to_direct(zot):
    out = server.search_items(
        query="Title", collection_key=ROOT, limit=50, ctx=DummyContext()
    )
    assert zot.collection_items_calls == [ROOT]
    assert "ITEMGCHD" not in out


def test_search_items_scope_can_include_subcollections(zot):
    out = server.search_items(
        query="Title", collection_key=ROOT, include_subcollections=True,
        limit=50, ctx=DummyContext(),
    )
    assert set(zot.collection_items_calls) == {ROOT, CHILD_A, CHILD_B, GRANDCHILD}
    assert "ITEMGCHD" in out
    assert "ITEMUNRL" not in out


def test_search_by_tag_scope_defaults_to_direct(zot):
    out = server.search_by_tag(
        tag=["shared"], collection_key=ROOT, limit=50, ctx=DummyContext()
    )
    assert zot.collection_items_calls == [ROOT]
    assert "ITEMGCHD" not in out


def test_search_by_tag_scope_can_include_subcollections(zot):
    out = server.search_by_tag(
        tag=["shared"], collection_key=ROOT, include_subcollections=True,
        limit=50, ctx=DummyContext(),
    )
    assert "ITEMGCHD" in out
    assert "ITEMUNRL" not in out


def test_subcollection_scope_is_ignored_without_a_collection_key(zot):
    """The flag is about scoping a collection, so alone it must change nothing."""
    plain = server.search_items(query="Title", limit=50, ctx=DummyContext())
    flagged = server.search_items(
        query="Title", include_subcollections=True, limit=50, ctx=DummyContext()
    )
    assert _keys_in(plain) == _keys_in(flagged)
    assert zot.collections_calls == 0


# ---------------------------------------------------------------------------
# advanced_search — membership rather than string comparison
# ---------------------------------------------------------------------------

def _adv(collection_key, operation="is", **kwargs):
    return server.advanced_search(
        conditions=[{"field": "collection", "operation": operation, "value": collection_key}],
        limit=100, ctx=DummyContext(), **kwargs,
    )


def test_advanced_search_collection_defaults_to_direct_membership(zot):
    assert _keys_in(_adv(ROOT)) == {"ITEMROOT"}


def test_advanced_search_collection_can_include_subcollections(zot):
    out = _adv(ROOT, include_subcollections=True)
    assert _keys_in(out) == {"ITEMROOT", "ITEMCHDA", "ITEMCHDB", "ITEMGCHD", "ITEMBOTH"}


def test_advanced_search_isnot_excludes_the_whole_subtree(zot):
    """isNot must negate subtree membership, not per-collection membership.

    ITEMCHDA is not filed in ROOT directly, so a per-collection negation would
    wrongly return it when the subtree is what was asked about.
    """
    out = _adv(ROOT, operation="isNot", include_subcollections=True)
    assert _keys_in(out) == {"ITEMUNRL"}


def test_advanced_search_isnot_without_the_flag_is_unchanged(zot):
    out = _adv(ROOT, operation="isNot")
    assert "ITEMGCHD" in out  # not in ROOT directly, so it satisfies isNot


def test_advanced_search_scopes_only_the_collection_field(zot):
    """A non-collection condition whose value happens to be a collection key
    must not pick up subtree treatment."""
    out = server.advanced_search(
        conditions=[{"field": "title", "operation": "is", "value": ROOT}],
        include_subcollections=True, limit=100, ctx=DummyContext(),
    )
    assert _keys_in(out) == set()


def test_advanced_search_expands_each_distinct_value_once(zot):
    server.advanced_search(
        conditions=[
            {"field": "collection", "operation": "is", "value": ROOT},
            {"field": "collection", "operation": "is", "value": ROOT},
        ],
        join_mode="any", include_subcollections=True, limit=100, ctx=DummyContext(),
    )
    assert zot.collections_calls == 1
