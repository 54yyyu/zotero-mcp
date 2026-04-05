"""Tests for item related/relation functionality."""

import pytest

from zotero_mcp import server
from conftest import DummyContext, FakeZotero, _FakeResponse


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_item(key="ABCD1234", version=10, title="Test Title",
               relations=None, **kwargs):
    """Build a Zotero item dict with optional relations."""
    return {
        "key": key,
        "version": version,
        "data": {
            "key": key,
            "version": version,
            "itemType": "journalArticle",
            "title": title,
            "creators": [{"creatorType": "author",
                          "firstName": "Jane", "lastName": "Doe"}],
            "relations": relations or {},
            **kwargs
        },
    }


class FakeZoteroForRelations(FakeZotero):
    """Extends FakeZotero with relation-specific behaviour."""

    def __init__(self, items=None, library_type="user", library_id="12345"):
        super().__init__()
        self._items = {it["key"]: it for it in (items or [])}
        self.update_calls = []
        self.library_type = library_type
        self.library_id = library_id

    def item(self, item_key):
        if item_key in self._items:
            return self._items[item_key]
        raise Exception(f"Item {item_key} not found")

    def update_item(self, item, **kwargs):
        self.update_calls.append(item)
        # Update internal state
        key = item.get("key")
        if key in self._items:
            self._items[key] = item
        return _FakeResponse(204)


# -----------------------------------------------------------------------------
# Get Related Items Tests
# -----------------------------------------------------------------------------

class TestGetItemRelated:

    def test_no_relations(self, monkeypatch):
        """Item with no relations returns appropriate message."""
        item = _make_item(key="ITEM0001", relations={})
        fake = FakeZoteroForRelations(items=[item])

        monkeypatch.setattr("zotero_mcp.tools.retrieval._client.get_zotero_client",
                            lambda: fake)

        result = server.get_item_related("ITEM0001", ctx=DummyContext())

        assert "No related items found" in result
        assert "ITEM0001" in result

    def test_with_relations(self, monkeypatch):
        """Item with relations returns formatted list."""
        item1 = _make_item(key="ITEM0001", title="First Paper")
        item2 = _make_item(key="ITEM0002", title="Second Paper")
        item1["data"]["relations"] = {
            "dc:relation": ["http://zotero.org/users/12345/items/ITEM0002"]
        }

        fake = FakeZoteroForRelations(items=[item1, item2])

        monkeypatch.setattr("zotero_mcp.tools.retrieval._client.get_zotero_client",
                            lambda: fake)

        result = server.get_item_related("ITEM0001", ctx=DummyContext())

        assert "Second Paper" in result
        assert "ITEM0002" in result
        assert "dc:relation" in result

    def test_nonexistent_item(self, monkeypatch):
        """Fetching relations for nonexistent item returns error."""
        fake = FakeZoteroForRelations(items=[])

        monkeypatch.setattr("zotero_mcp.tools.retrieval._client.get_zotero_client",
                            lambda: fake)

        result = server.get_item_related("NOTFOUND", ctx=DummyContext())

        assert "not found" in result.lower()


# -----------------------------------------------------------------------------
# Add Relation Tests
# -----------------------------------------------------------------------------

class TestAddItemRelation:

    def test_add_relation_success(self, monkeypatch):
        """Successfully add a relation between two items."""
        item1 = _make_item(key="ITEM0001", title="Paper One")
        item2 = _make_item(key="ITEM0002", title="Paper Two")
        fake = FakeZoteroForRelations(items=[item1, item2])

        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.add_item_relation(
            item_key="ITEM0001",
            related_item_key="ITEM0002",
            ctx=DummyContext(),
        )

        assert "Successfully added relation" in result
        assert "ITEM0001" in result
        assert "ITEM0002" in result
        assert len(fake.update_calls) >= 1

        # Verify the relation was added
        updated_item = fake.update_calls[0]
        relations = updated_item["data"]["relations"]
        assert "dc:relation" in relations

    def test_cannot_relate_to_self(self, monkeypatch):
        """Cannot add relation from an item to itself."""
        item1 = _make_item(key="ITEM0001")
        fake = FakeZoteroForRelations(items=[item1])

        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.add_item_relation(
            item_key="ITEM0001",
            related_item_key="ITEM0001",
            ctx=DummyContext(),
        )

        assert "Cannot relate an item to itself" in result
        assert len(fake.update_calls) == 0

    def test_nonexistent_item(self, monkeypatch):
        """Adding relation to nonexistent item returns error."""
        item1 = _make_item(key="ITEM0001")
        fake = FakeZoteroForRelations(items=[item1])

        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.add_item_relation(
            item_key="ITEM0001",
            related_item_key="NOTFOUND",
            ctx=DummyContext(),
        )

        assert "not found" in result.lower()

    def test_duplicate_relation(self, monkeypatch):
        """Adding duplicate relation returns informative message."""
        item1 = _make_item(key="ITEM0001", relations={
            "dc:relation": ["http://zotero.org/users/12345/items/ITEM0002"]
        })
        item2 = _make_item(key="ITEM0002")
        fake = FakeZoteroForRelations(items=[item1, item2])

        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.add_item_relation(
            item_key="ITEM0001",
            related_item_key="ITEM0002",
            ctx=DummyContext(),
        )

        assert "already related" in result.lower()


# -----------------------------------------------------------------------------
# Remove Relation Tests
# -----------------------------------------------------------------------------

class TestRemoveItemRelation:

    def test_remove_relation_success(self, monkeypatch):
        """Successfully remove a relation."""
        item1 = _make_item(key="ITEM0001", relations={
            "dc:relation": ["http://zotero.org/users/12345/items/ITEM0002"]
        })
        item2 = _make_item(key="ITEM0002", relations={
            "dc:relation": ["http://zotero.org/users/12345/items/ITEM0001"]
        })
        fake = FakeZoteroForRelations(items=[item1, item2])

        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.remove_item_relation(
            item_key="ITEM0001",
            related_item_key="ITEM0002",
            ctx=DummyContext(),
        )

        assert "Successfully removed relation" in result
        assert "ITEM0001" in result
        assert "ITEM0002" in result

    def test_remove_nonexistent_relation(self, monkeypatch):
        """Removing nonexistent relation returns error."""
        item1 = _make_item(key="ITEM0001", relations={})
        item2 = _make_item(key="ITEM0002")
        fake = FakeZoteroForRelations(items=[item1, item2])

        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.remove_item_relation(
            item_key="ITEM0001",
            related_item_key="ITEM0002",
            ctx=DummyContext(),
        )

        assert "not related" in result.lower() or "no relations" in result.lower()

    def test_no_relations_field(self, monkeypatch):
        """Item with no relations field returns appropriate message."""
        item1 = _make_item(key="ITEM0001")
        item1["data"]["relations"] = {}  # Explicitly empty
        item2 = _make_item(key="ITEM0002")
        fake = FakeZoteroForRelations(items=[item1, item2])

        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.remove_item_relation(
            item_key="ITEM0001",
            related_item_key="ITEM0002",
            ctx=DummyContext(),
        )

        assert "no relations" in result.lower()
