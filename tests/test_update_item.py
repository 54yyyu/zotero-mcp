"""Tests for Feature 6: update_item (zotero_update_item)."""

import pytest

from zotero_mcp import server
from conftest import DummyContext, FakeZotero, _FakeResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(key="ABCD1234", version=10, title="Original Title",
               tags=None, collections=None, extra="", abstract="",
               date="2024-01-01", doi="", url=""):
    """Build a realistic Zotero item dict for stubbing."""
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
            "date": date,
            "abstractNote": abstract,
            "publicationTitle": "Test Journal",
            "tags": [{"tag": t} for t in (tags or [])],
            "collections": list(collections or []),
            "DOI": doi,
            "url": url,
            "extra": extra,
            "relations": {},
        },
    }


class FakeZoteroForUpdate(FakeZotero):
    """Extends FakeZotero with update-specific behaviour."""

    def __init__(self, items=None, collections=None):
        super().__init__()
        self._items = items or []
        self._collections = collections or []
        # Track the exact item dict passed to update_item
        self.update_calls = []

    def item(self, item_key):
        for it in self._items:
            if it.get("key") == item_key:
                return it
        raise Exception(f"Item {item_key} not found")

    def update_item(self, item, **kwargs):
        self.update_calls.append(item)
        return _FakeResponse(204)


# ---------------------------------------------------------------------------
# Happy-path: update title
# ---------------------------------------------------------------------------

class TestUpdateItemHappyPath:

    def test_update_title(self, monkeypatch):
        item = _make_item(title="Old Title")
        fake = FakeZoteroForUpdate(items=[item])

        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            title="New Title",
            ctx=DummyContext(),
        )

        assert len(fake.update_calls) == 1
        updated = fake.update_calls[0]
        assert updated["data"]["title"] == "New Title"
        assert "New Title" in result


# ---------------------------------------------------------------------------
# Multiple fields at once
# ---------------------------------------------------------------------------

class TestUpdateMultipleFields:

    def test_update_title_date_abstract(self, monkeypatch):
        item = _make_item(title="Old", date="2020-01-01", abstract="old abs")
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            title="Brand New Title",
            date="2025-06-15",
            abstract="Updated abstract",
            ctx=DummyContext(),
        )

        updated = fake.update_calls[0]
        assert updated["data"]["title"] == "Brand New Title"
        assert updated["data"]["date"] == "2025-06-15"
        assert updated["data"]["abstractNote"] == "Updated abstract"


# ---------------------------------------------------------------------------
# Tag operations
# ---------------------------------------------------------------------------

class TestUpdateItemTags:

    def test_tags_replace(self, monkeypatch):
        """tags= replaces ALL existing tags."""
        item = _make_item(tags=["old1", "old2"])
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        server.update_item(
            item_key="ABCD1234",
            tags=["new"],
            ctx=DummyContext(),
        )

        updated_tags = [t["tag"] for t in fake.update_calls[0]["data"]["tags"]]
        assert updated_tags == ["new"]

    def test_add_tags_additive(self, monkeypatch):
        """add_tags= adds to existing tags without removing any."""
        item = _make_item(tags=["existing"])
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        server.update_item(
            item_key="ABCD1234",
            add_tags=["extra"],
            ctx=DummyContext(),
        )

        updated_tags = {t["tag"] for t in fake.update_calls[0]["data"]["tags"]}
        assert "existing" in updated_tags
        assert "extra" in updated_tags

    def test_remove_tags(self, monkeypatch):
        """remove_tags= removes specified tags, keeps the rest."""
        item = _make_item(tags=["keep", "old", "also-keep"])
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        server.update_item(
            item_key="ABCD1234",
            remove_tags=["old"],
            ctx=DummyContext(),
        )

        updated_tags = [t["tag"] for t in fake.update_calls[0]["data"]["tags"]]
        assert "old" not in updated_tags
        assert "keep" in updated_tags
        assert "also-keep" in updated_tags

    def test_tags_and_add_tags_mutually_exclusive(self, monkeypatch):
        """Providing both tags= and add_tags= should produce an error."""
        item = _make_item(tags=["x"])
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            tags=["replacement"],
            add_tags=["extra"],
            ctx=DummyContext(),
        )

        # Should return an error message, NOT call update_item
        assert len(fake.update_calls) == 0
        assert "Cannot use" in result or "mutually exclusive" in result.lower() \
            or "tags" in result.lower()

    def test_tags_and_remove_tags_mutually_exclusive(self, monkeypatch):
        """Providing both tags= and remove_tags= should produce an error."""
        item = _make_item(tags=["x"])
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            tags=["replacement"],
            remove_tags=["x"],
            ctx=DummyContext(),
        )

        assert len(fake.update_calls) == 0
        assert "Cannot use" in result or "tags" in result.lower()


# ---------------------------------------------------------------------------
# Collection names resolved and added
# ---------------------------------------------------------------------------

class TestUpdateItemCollections:

    def test_collection_names_resolved(self, monkeypatch):
        """collection_names should resolve names to keys and add them."""
        item = _make_item(collections=["EXISTCOL"])
        fake = FakeZoteroForUpdate(
            items=[item],
            collections=[
                {"key": "COL001", "data": {"name": "My Papers"}},
                {"key": "COL002", "data": {"name": "Reviews"}},
            ],
        )
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        server.update_item(
            item_key="ABCD1234",
            collection_names=["My Papers"],
            ctx=DummyContext(),
        )

        updated_colls = fake.update_calls[0]["data"]["collections"]
        # Should contain BOTH the existing collection and the resolved one
        assert "EXISTCOL" in updated_colls
        assert "COL001" in updated_colls

    def test_collection_names_unknown_raises_error(self, monkeypatch):
        """Unknown collection name should produce an error."""
        item = _make_item()
        fake = FakeZoteroForUpdate(
            items=[item],
            collections=[
                {"key": "COL001", "data": {"name": "My Papers"}},
            ],
        )
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            collection_names=["Nonexistent Collection"],
            ctx=DummyContext(),
        )

        # Should get an error, no update call
        assert len(fake.update_calls) == 0
        assert "No collection found" in result or "not found" in result.lower()


# ---------------------------------------------------------------------------
# Extra field is a string
# ---------------------------------------------------------------------------

class TestUpdateItemExtra:

    def test_extra_field_string(self, monkeypatch):
        """extra param should be stored as-is (string)."""
        item = _make_item(extra="old extra")
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        server.update_item(
            item_key="ABCD1234",
            extra="PMID: 12345\noriginal-date: 2020",
            ctx=DummyContext(),
        )

        assert fake.update_calls[0]["data"]["extra"] == "PMID: 12345\noriginal-date: 2020"


# ---------------------------------------------------------------------------
# Version from write client (not read client)
# ---------------------------------------------------------------------------

class TestUpdateItemVersion:

    def test_version_from_write_client(self, monkeypatch):
        """Item should be fetched from the write client for correct version."""
        read_item = _make_item(version=5, title="Read Version")
        write_item = _make_item(version=42, title="Write Version")

        read_fake = FakeZoteroForUpdate(items=[read_item])
        write_fake = FakeZoteroForUpdate(items=[write_item])

        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (read_fake, write_fake))

        server.update_item(
            item_key="ABCD1234",
            title="Updated",
            ctx=DummyContext(),
        )

        # The update should go through write_fake, not read_fake
        assert len(write_fake.update_calls) == 1
        assert len(read_fake.update_calls) == 0
        # The version in the updated dict should be from the write client
        assert write_fake.update_calls[0]["data"]["version"] == 42


# ---------------------------------------------------------------------------
# Before/after diff returned
# ---------------------------------------------------------------------------

class TestUpdateItemDiff:

    def test_diff_returned(self, monkeypatch):
        """Result should show before/after for changed fields."""
        item = _make_item(title="Old Title", date="2020-01-01")
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            title="New Title",
            ctx=DummyContext(),
        )

        # Result should mention both old and new values
        assert "Old Title" in result
        assert "New Title" in result


# ---------------------------------------------------------------------------
# Hybrid mode / local-only rejection
# ---------------------------------------------------------------------------

class TestUpdateItemHybridMode:

    def test_local_only_rejected(self, monkeypatch):
        """Local-only mode (no web credentials) should return clear error."""
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (_ for _ in ()).throw(
                                ValueError(
                                    "Cannot perform write operations in local-only mode. "
                                    "Add ZOTERO_API_KEY and ZOTERO_LIBRARY_ID to enable hybrid mode."
                                )
                            ))

        result = server.update_item(
            item_key="ABCD1234",
            title="Anything",
            ctx=DummyContext(),
        )

        assert "local-only" in result.lower() or "Cannot perform write" in result

    def test_hybrid_mode_uses_web_for_write(self, monkeypatch):
        """In hybrid mode, update_item should be called on the write client."""
        read_item = _make_item(version=1, title="Local Read")
        write_item = _make_item(version=99, title="Web Write")

        read_zot = FakeZoteroForUpdate(items=[read_item])
        write_zot = FakeZoteroForUpdate(items=[write_item])

        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (read_zot, write_zot))

        server.update_item(
            item_key="ABCD1234",
            title="Changed",
            ctx=DummyContext(),
        )

        # Write should happen on write_zot, not read_zot
        assert len(write_zot.update_calls) == 1
        assert len(read_zot.update_calls) == 0


# ---------------------------------------------------------------------------
# Nonexistent item key -> error
# ---------------------------------------------------------------------------

class TestUpdateItemErrors:

    def test_nonexistent_item_key(self, monkeypatch):
        """An item key that doesn't exist should produce a clear error."""
        fake = FakeZoteroForUpdate(items=[])  # no items at all

        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ZZZZZZZZ",
            title="Anything",
            ctx=DummyContext(),
        )

        assert "not found" in result.lower() or "error" in result.lower()

    def test_no_fields_provided(self, monkeypatch):
        """Calling update_item with no fields to change should give feedback."""
        item = _make_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            ctx=DummyContext(),
        )

        # Should either return a message or succeed with no update
        assert len(fake.update_calls) == 0 or "no changes" in result.lower() \
            or "nothing" in result.lower()

    def test_write_failure_reported(self, monkeypatch):
        """If the API returns a non-success status, report it."""
        item = _make_item()

        class FailingZotero(FakeZoteroForUpdate):
            def update_item(self, item_dict, **kwargs):
                self.update_calls.append(item_dict)
                return _FakeResponse(412, text="Precondition Failed")

        fake = FailingZotero(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            title="Anything",
            ctx=DummyContext(),
        )

        assert "fail" in result.lower() or "error" in result.lower()


# ---------------------------------------------------------------------------
# Additional field updates
# ---------------------------------------------------------------------------

class TestUpdateItemFieldVariants:

    def test_update_doi(self, monkeypatch):
        item = _make_item(doi="10.1234/old")
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        server.update_item(
            item_key="ABCD1234",
            doi="10.5678/new",
            ctx=DummyContext(),
        )

        assert fake.update_calls[0]["data"]["DOI"] == "10.5678/new"

    def test_update_url(self, monkeypatch):
        item = _make_item(url="https://old.example.com")
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        server.update_item(
            item_key="ABCD1234",
            url="https://new.example.com",
            ctx=DummyContext(),
        )

        assert fake.update_calls[0]["data"]["url"] == "https://new.example.com"

    def test_update_publication_title(self, monkeypatch):
        item = _make_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        server.update_item(
            item_key="ABCD1234",
            publication_title="Nature",
            ctx=DummyContext(),
        )

        assert fake.update_calls[0]["data"]["publicationTitle"] == "Nature"

    def test_update_creators(self, monkeypatch):
        item = _make_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        new_creators = [
            {"creatorType": "author", "firstName": "Alice", "lastName": "Smith"},
            {"creatorType": "editor", "firstName": "Bob", "lastName": "Jones"},
        ]

        server.update_item(
            item_key="ABCD1234",
            creators=new_creators,
            ctx=DummyContext(),
        )

        assert fake.update_calls[0]["data"]["creators"] == new_creators

    def test_collections_additive(self, monkeypatch):
        """collections= adds to existing collections (does not replace)."""
        item = _make_item(collections=["OLD_COL"])
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        server.update_item(
            item_key="ABCD1234",
            collections=["NEW_COL1", "NEW_COL2"],
            ctx=DummyContext(),
        )

        updated_colls = fake.update_calls[0]["data"]["collections"]
        assert "OLD_COL" in updated_colls  # existing collection preserved
        assert "NEW_COL1" in updated_colls
        assert "NEW_COL2" in updated_colls
