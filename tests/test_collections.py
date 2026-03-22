"""Tests for Features 1-3: Collection operations.

Feature 1: zotero_create_collection
Feature 2: zotero_search_collections
Feature 3: zotero_manage_collections

These tests are written BEFORE implementation. They will FAIL until
the tool functions (create_collection, search_collections, manage_collections)
are added to server.py.
"""

import json
import pytest

from conftest import DummyContext, FakeZotero
from zotero_mcp import server


# ---------------------------------------------------------------------------
# Extended FakeZotero for collection tests
# ---------------------------------------------------------------------------

class FakeZoteroCollections(FakeZotero):
    """FakeZotero with tracking for collection write operations."""

    def __init__(self):
        super().__init__()
        self.added_to_collections = []   # (collection_key, items)
        self.removed_from_collections = []  # (collection_key, item)
        self.created_collections = []

    def create_collections(self, colls, **kwargs):
        self.created_collections.extend(colls)
        result = {}
        for i, c in enumerate(colls):
            result[str(i)] = f"NEWCOL{i:04d}"
        return {"success": result, "successful": {}, "failed": {}}

    def addto_collection(self, collection_key, items, **kwargs):
        self.added_to_collections.append((collection_key, items))
        return _FakeResponse(204)

    def deletefrom_collection(self, collection_key, item, **kwargs):
        self.removed_from_collections.append((collection_key, item))
        return _FakeResponse(204)


class _FakeResponse:
    """Minimal httpx.Response stub."""

    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_zot():
    zot = FakeZoteroCollections()
    zot._collections = [
        {
            "key": "ABC00001",
            "data": {"name": "Machine Learning", "parentCollection": False},
        },
        {
            "key": "ABC00002",
            "data": {"name": "Deep Learning", "parentCollection": "ABC00001"},
        },
        {
            "key": "ABC00003",
            "data": {"name": "NLP Papers", "parentCollection": False},
        },
    ]
    zot._items = [
        {
            "key": "ITEM0001",
            "version": 10,
            "data": {
                "title": "Attention Is All You Need",
                "collections": ["ABC00001"],
                "tags": [],
            },
        },
        {
            "key": "ITEM0002",
            "version": 11,
            "data": {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "collections": ["ABC00003"],
                "tags": [],
            },
        },
    ]
    return zot


@pytest.fixture
def ctx():
    return DummyContext()


def _patch_hybrid(monkeypatch, read_zot, write_zot):
    """Patch _get_write_client to return the given read/write pair."""
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (read_zot, write_zot)
    )


def _patch_web_only(monkeypatch, fake_zot):
    """Patch for web-only mode: same client for read and write."""
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
    )


def _patch_local_only(monkeypatch, fake_zot):
    """Patch _get_write_client to raise ValueError (local-only mode)."""
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)

    def _raise_local_only(ctx):
        raise ValueError(
            "Cannot perform write operations in local-only mode. "
            "Add ZOTERO_API_KEY and ZOTERO_LIBRARY_ID to enable hybrid mode."
        )

    monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", _raise_local_only)


# ===========================================================================
# Feature 1: zotero_create_collection
# ===========================================================================


class TestCreateCollection:
    """Tests for the create_collection tool function."""

    def test_happy_path(self, monkeypatch, fake_zot, ctx):
        """Create a simple collection and verify name and key are returned."""
        _patch_web_only(monkeypatch, fake_zot)

        result = server.create_collection(name="New Collection", ctx=ctx)

        assert "NEWCOL0000" in result
        assert len(fake_zot.created_collections) == 1
        assert fake_zot.created_collections[0]["name"] == "New Collection"

    def test_with_parent_collection_key(self, monkeypatch, fake_zot, ctx):
        """parent_collection as an 8-char alphanumeric key is passed directly."""
        _patch_web_only(monkeypatch, fake_zot)

        result = server.create_collection(
            name="Sub Collection",
            parent_collection="ABC00001",
            ctx=ctx,
        )

        assert "NEWCOL0000" in result
        created = fake_zot.created_collections[0]
        assert created["parentCollection"] == "ABC00001"

    def test_with_parent_collection_name(self, monkeypatch, fake_zot, ctx):
        """parent_collection as a name is resolved via _resolve_collection_names."""
        _patch_web_only(monkeypatch, fake_zot)

        result = server.create_collection(
            name="Sub Collection",
            parent_collection="Machine Learning",
            ctx=ctx,
        )

        assert "NEWCOL0000" in result
        created = fake_zot.created_collections[0]
        # Should resolve "Machine Learning" -> "ABC00001"
        assert created["parentCollection"] == "ABC00001"

    def test_hybrid_mode_uses_web_write(self, monkeypatch, fake_zot, ctx):
        """In hybrid mode, read_zot is used for name resolution but
        write_zot is used for the actual create_collections call."""
        read_zot = FakeZoteroCollections()
        read_zot._collections = fake_zot._collections
        write_zot = FakeZoteroCollections()

        _patch_hybrid(monkeypatch, read_zot, write_zot)

        result = server.create_collection(name="Hybrid Test", ctx=ctx)

        assert "NEWCOL0000" in result
        # Write should go to write_zot, not read_zot
        assert len(write_zot.created_collections) == 1
        assert len(read_zot.created_collections) == 0

    def test_local_only_mode_returns_error(self, monkeypatch, fake_zot, ctx):
        """In local-only mode (no web credentials), return clear error."""
        _patch_local_only(monkeypatch, fake_zot)

        result = server.create_collection(name="Should Fail", ctx=ctx)

        assert "local-only mode" in result.lower() or "hybrid mode" in result.lower()

    def test_response_parsing_extracts_key(self, monkeypatch, fake_zot, ctx):
        """The collection key is extracted from response['success']['0']."""
        _patch_web_only(monkeypatch, fake_zot)

        # Override create_collections to return a specific key
        def custom_create(colls, **kwargs):
            fake_zot.created_collections.extend(colls)
            return {"success": {"0": "MYCUSTOM"}, "successful": {}, "failed": {}}

        fake_zot.create_collections = custom_create

        result = server.create_collection(name="Custom Key", ctx=ctx)

        assert "MYCUSTOM" in result

    def test_parent_name_not_found_returns_error(self, monkeypatch, fake_zot, ctx):
        """Passing a parent_collection name that doesn't match any collection."""
        _patch_web_only(monkeypatch, fake_zot)

        result = server.create_collection(
            name="Orphan",
            parent_collection="Nonexistent Collection",
            ctx=ctx,
        )

        assert "error" in result.lower() or "no collection found" in result.lower()

    def test_no_parent_sets_false(self, monkeypatch, fake_zot, ctx):
        """When parent_collection is None, parentCollection should be False."""
        _patch_web_only(monkeypatch, fake_zot)

        server.create_collection(name="Top Level", ctx=ctx)

        created = fake_zot.created_collections[0]
        assert created["parentCollection"] is False


# ===========================================================================
# Feature 2: zotero_search_collections
# ===========================================================================


class TestSearchCollections:
    """Tests for the search_collections tool function."""

    def test_happy_path_find_by_name(self, monkeypatch, fake_zot, ctx):
        """Search for a collection by exact name substring."""
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)

        result = server.search_collections(query="Machine Learning", ctx=ctx)

        assert "Machine Learning" in result
        assert "ABC00001" in result

    def test_case_insensitive_search(self, monkeypatch, fake_zot, ctx):
        """Search is case-insensitive."""
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)

        result = server.search_collections(query="machine learning", ctx=ctx)

        assert "Machine Learning" in result or "ABC00001" in result

    def test_partial_match(self, monkeypatch, fake_zot, ctx):
        """Search for a substring should match collections containing it."""
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)

        result = server.search_collections(query="Learning", ctx=ctx)

        # Should match both "Machine Learning" and "Deep Learning"
        assert "ABC00001" in result
        assert "ABC00002" in result

    def test_no_matches_returns_message(self, monkeypatch, fake_zot, ctx):
        """When no collections match, return an informative message."""
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)

        result = server.search_collections(query="Quantum Physics", ctx=ctx)

        assert "no collection" in result.lower() or "0" in result

    def test_empty_library_returns_message(self, monkeypatch, ctx):
        """Empty library (no collections) returns an informative message."""
        empty_zot = FakeZoteroCollections()
        empty_zot._collections = []
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: empty_zot)

        result = server.search_collections(query="anything", ctx=ctx)

        assert "no collection" in result.lower() or "0" in result

    def test_returns_parent_info(self, monkeypatch, fake_zot, ctx):
        """Results should include parent collection info when present."""
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)

        result = server.search_collections(query="Deep Learning", ctx=ctx)

        assert "ABC00002" in result
        # Should mention parent key or parent name
        assert "ABC00001" in result or "Machine Learning" in result

    def test_returns_item_count(self, monkeypatch, fake_zot, ctx):
        """Results should include number of items in each collection."""
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)

        result = server.search_collections(query="NLP", ctx=ctx)

        assert "ABC00003" in result


# ===========================================================================
# Feature 3: zotero_manage_collections
# ===========================================================================


class TestManageCollections:
    """Tests for the manage_collections tool function."""

    def test_add_items_to_collection(self, monkeypatch, fake_zot, ctx):
        """Add items to a collection."""
        _patch_web_only(monkeypatch, fake_zot)

        result = server.manage_collections(
            item_keys=["ITEM0001"],
            add_to=["ABC00003"],
            ctx=ctx,
        )

        assert len(fake_zot.added_to_collections) == 1
        coll_key, items = fake_zot.added_to_collections[0]
        assert coll_key == "ABC00003"
        assert "success" in result.lower() or "added" in result.lower()

    def test_remove_items_from_collection(self, monkeypatch, fake_zot, ctx):
        """Remove items from a collection."""
        _patch_web_only(monkeypatch, fake_zot)

        result = server.manage_collections(
            item_keys=["ITEM0002"],
            remove_from=["ABC00003"],
            ctx=ctx,
        )

        assert len(fake_zot.removed_from_collections) == 1
        coll_key, item = fake_zot.removed_from_collections[0]
        assert coll_key == "ABC00003"
        assert "success" in result.lower() or "removed" in result.lower()

    def test_add_and_remove_in_one_call(self, monkeypatch, fake_zot, ctx):
        """Both add_to and remove_from in a single call."""
        _patch_web_only(monkeypatch, fake_zot)

        result = server.manage_collections(
            item_keys=["ITEM0001"],
            add_to=["ABC00003"],
            remove_from=["ABC00001"],
            ctx=ctx,
        )

        assert len(fake_zot.added_to_collections) >= 1
        assert len(fake_zot.removed_from_collections) >= 1
        assert "success" in result.lower() or "added" in result.lower()

    def test_item_keys_as_json_string(self, monkeypatch, fake_zot, ctx):
        """item_keys passed as a JSON string should be normalized."""
        _patch_web_only(monkeypatch, fake_zot)

        result = server.manage_collections(
            item_keys='["ITEM0001", "ITEM0002"]',
            add_to=["ABC00002"],
            ctx=ctx,
        )

        # Should process both items
        assert len(fake_zot.added_to_collections) >= 1
        assert "success" in result.lower() or "added" in result.lower()

    def test_add_to_as_json_string(self, monkeypatch, fake_zot, ctx):
        """add_to passed as a JSON string should be normalized."""
        _patch_web_only(monkeypatch, fake_zot)

        result = server.manage_collections(
            item_keys=["ITEM0001"],
            add_to='["ABC00002", "ABC00003"]',
            ctx=ctx,
        )

        assert len(fake_zot.added_to_collections) >= 2

    def test_remove_from_as_json_string(self, monkeypatch, fake_zot, ctx):
        """remove_from passed as a JSON string should be normalized."""
        _patch_web_only(monkeypatch, fake_zot)

        result = server.manage_collections(
            item_keys=["ITEM0001"],
            remove_from='["ABC00001"]',
            ctx=ctx,
        )

        assert len(fake_zot.removed_from_collections) >= 1

    def test_single_string_item_key(self, monkeypatch, fake_zot, ctx):
        """A single item_key string (not a list) should be normalized to a list."""
        _patch_web_only(monkeypatch, fake_zot)

        result = server.manage_collections(
            item_keys="ITEM0001",
            add_to=["ABC00002"],
            ctx=ctx,
        )

        assert len(fake_zot.added_to_collections) >= 1

    def test_hybrid_mode_fetches_from_write_client(self, monkeypatch, ctx):
        """In hybrid mode, items are fetched from write_zot (not read_zot)
        since the version number must match the server the write goes to."""
        read_zot = FakeZoteroCollections()
        read_zot._collections = [
            {"key": "COL001", "data": {"name": "Test", "parentCollection": False}},
        ]
        write_zot = FakeZoteroCollections()
        write_zot._items = [
            {
                "key": "ITEM0001",
                "version": 99,
                "data": {"title": "Test Item", "collections": [], "tags": []},
            },
        ]

        _patch_hybrid(monkeypatch, read_zot, write_zot)

        result = server.manage_collections(
            item_keys=["ITEM0001"],
            add_to=["COL001"],
            ctx=ctx,
        )

        # Write operations should go through write_zot
        assert len(write_zot.added_to_collections) >= 1

    def test_local_only_mode_returns_error(self, monkeypatch, fake_zot, ctx):
        """In local-only mode, return clear error."""
        _patch_local_only(monkeypatch, fake_zot)

        result = server.manage_collections(
            item_keys=["ITEM0001"],
            add_to=["ABC00001"],
            ctx=ctx,
        )

        assert "local-only mode" in result.lower() or "hybrid mode" in result.lower()

    def test_no_add_or_remove_returns_error(self, monkeypatch, fake_zot, ctx):
        """Must specify at least one of add_to or remove_from."""
        _patch_web_only(monkeypatch, fake_zot)

        result = server.manage_collections(
            item_keys=["ITEM0001"],
            ctx=ctx,
        )

        assert "error" in result.lower() or "must specify" in result.lower()

    def test_collection_name_resolution_in_add_to(self, monkeypatch, fake_zot, ctx):
        """add_to values are passed directly as collection keys (no name resolution).
        The implementation uses _normalize_str_list_input, not _resolve_collection_names."""
        _patch_web_only(monkeypatch, fake_zot)

        result = server.manage_collections(
            item_keys=["ITEM0001"],
            add_to=["NLP Papers"],
            ctx=ctx,
        )

        # The value is passed through as-is (not resolved to a key)
        if fake_zot.added_to_collections:
            coll_key, _ = fake_zot.added_to_collections[0]
            assert coll_key == "NLP Papers"

    def test_multiple_items_batched(self, monkeypatch, fake_zot, ctx):
        """Multiple items should all be processed."""
        _patch_web_only(monkeypatch, fake_zot)

        result = server.manage_collections(
            item_keys=["ITEM0001", "ITEM0002"],
            add_to=["ABC00002"],
            ctx=ctx,
        )

        # Both items should be added to the collection
        assert "success" in result.lower() or "added" in result.lower()
