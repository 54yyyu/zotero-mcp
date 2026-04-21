"""Tests for Feature 6: update_item (zotero_update_item)."""

import pytest

from zotero_mcp import server
from conftest import DummyContext, FakeZotero, _FakeResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(key="ABCD1234", version=10, title="Original Title",
               tags=None, collections=None, extra="", abstract="",
               date="2024-01-01", doi="", url="",
               volume="", issue="", pages="", publisher="",
               issn="", language="", short_title="",
               publication_title="Test Journal"):
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
            "publicationTitle": publication_title,
            "volume": volume,
            "issue": issue,
            "pages": pages,
            "publisher": publisher,
            "ISSN": issn,
            "language": language,
            "shortTitle": short_title,
            "tags": [{"tag": t} for t in (tags or [])],
            "collections": list(collections or []),
            "DOI": doi,
            "url": url,
            "extra": extra,
            "relations": {},
        },
    }


def _make_webpage_item(key="WEBP1234", version=10, title="A Web Page",
                      access_date="", url="https://example.com",
                      tags=None, collections=None, extra=""):
    """Build a realistic Zotero webpage item dict (supports accessDate)."""
    return {
        "key": key,
        "version": version,
        "data": {
            "key": key,
            "version": version,
            "itemType": "webpage",
            "title": title,
            "creators": [],
            "date": "",
            "accessDate": access_date,
            "abstractNote": "",
            "url": url,
            "language": "",
            "shortTitle": "",
            "tags": [{"tag": t} for t in (tags or [])],
            "collections": list(collections or []),
            "extra": extra,
            "relations": {},
        },
    }


def _make_book_item(key="BOOK1234", version=10, title="Original Book",
                    tags=None, collections=None, extra="",
                    publisher="", edition="", isbn="", volume="",
                    issn="", language="", short_title=""):
    """Build a realistic Zotero book item dict for stubbing."""
    return {
        "key": key,
        "version": version,
        "data": {
            "key": key,
            "version": version,
            "itemType": "book",
            "title": title,
            "creators": [{"creatorType": "author",
                          "firstName": "Jane", "lastName": "Doe"}],
            "date": "2024-01-01",
            "abstractNote": "",
            "publisher": publisher,
            "place": "",
            "ISBN": isbn,
            "numPages": "",
            "edition": edition,
            "volume": volume,
            "ISSN": issn,
            "language": language,
            "shortTitle": short_title,
            "tags": [{"tag": t} for t in (tags or [])],
            "collections": list(collections or []),
            "DOI": "",
            "url": "",
            "extra": extra,
            "relations": {},
        },
    }


def _make_book_section_item(key="BSEC1234", version=10,
                            title="Original Chapter",
                            tags=None, collections=None, extra="",
                            book_title="", publisher="", edition="",
                            isbn="", pages="", volume="",
                            issn="", language="", short_title=""):
    """Build a realistic Zotero bookSection item dict for stubbing."""
    return {
        "key": key,
        "version": version,
        "data": {
            "key": key,
            "version": version,
            "itemType": "bookSection",
            "title": title,
            "creators": [{"creatorType": "author",
                          "firstName": "Jane", "lastName": "Doe"}],
            "date": "2024-01-01",
            "abstractNote": "",
            "bookTitle": book_title,
            "publisher": publisher,
            "place": "",
            "ISBN": isbn,
            "pages": pages,
            "edition": edition,
            "volume": volume,
            "ISSN": issn,
            "language": language,
            "shortTitle": short_title,
            "tags": [{"tag": t} for t in (tags or [])],
            "collections": list(collections or []),
            "DOI": "",
            "url": "",
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


# ---------------------------------------------------------------------------
# New field parameters (volume, issue, pages, publisher, issn, language,
# short_title, edition, isbn, book_title)
# ---------------------------------------------------------------------------

class TestUpdateItemNewFields:

    def test_update_volume(self, monkeypatch):
        item = _make_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            volume="42",
            ctx=DummyContext(),
        )

        assert fake.update_calls[0]["data"]["volume"] == "42"
        assert "42" in result

    def test_update_issue(self, monkeypatch):
        item = _make_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            issue="3",
            ctx=DummyContext(),
        )

        assert fake.update_calls[0]["data"]["issue"] == "3"
        assert "3" in result

    def test_update_pages(self, monkeypatch):
        item = _make_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            pages="27-61",
            ctx=DummyContext(),
        )

        assert fake.update_calls[0]["data"]["pages"] == "27-61"
        assert "27-61" in result

    def test_update_publisher(self, monkeypatch):
        item = _make_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            publisher="Oxford University Press",
            ctx=DummyContext(),
        )

        assert fake.update_calls[0]["data"]["publisher"] == "Oxford University Press"
        assert "Oxford University Press" in result

    def test_update_issn(self, monkeypatch):
        item = _make_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            issn="0028-0836",
            ctx=DummyContext(),
        )

        assert fake.update_calls[0]["data"]["ISSN"] == "0028-0836"
        assert "0028-0836" in result

    def test_update_language(self, monkeypatch):
        item = _make_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            language="en",
            ctx=DummyContext(),
        )

        assert fake.update_calls[0]["data"]["language"] == "en"
        assert "en" in result

    def test_update_short_title(self, monkeypatch):
        item = _make_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            short_title="Brief",
            ctx=DummyContext(),
        )

        assert fake.update_calls[0]["data"]["shortTitle"] == "Brief"
        assert "Brief" in result

    def test_update_edition_on_book(self, monkeypatch):
        item = _make_book_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="BOOK1234",
            edition="3rd",
            ctx=DummyContext(),
        )

        assert fake.update_calls[0]["data"]["edition"] == "3rd"
        assert "3rd" in result

    def test_update_isbn_on_book(self, monkeypatch):
        item = _make_book_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="BOOK1234",
            isbn="978-0-123456-78-9",
            ctx=DummyContext(),
        )

        assert fake.update_calls[0]["data"]["ISBN"] == "978-0-123456-78-9"
        assert "978-0-123456-78-9" in result

    def test_update_access_date_on_webpage(self, monkeypatch):
        """accessDate should be writable on webpage items (issue #240)."""
        item = _make_webpage_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="WEBP1234",
            access_date="2026-04-21",
            ctx=DummyContext(),
        )

        assert fake.update_calls[0]["data"]["accessDate"] == "2026-04-21"
        assert "2026-04-21" in result

    def test_update_place_on_book(self, monkeypatch):
        """place should be writable on book items (issue #238 round-trip)."""
        item = _make_book_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="BOOK1234",
            place="Cambridge, MA",
            ctx=DummyContext(),
        )

        assert fake.update_calls[0]["data"]["place"] == "Cambridge, MA"
        assert "Cambridge, MA" in result

    def test_access_date_skipped_on_book(self, monkeypatch):
        """accessDate is not valid for books — should be in skip warning."""
        item = _make_book_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="BOOK1234",
            access_date="2026-04-21",
            ctx=DummyContext(),
        )

        # No update should happen (accessDate not in book schema)
        assert len(fake.update_calls) == 0
        assert "access_date" in result
        assert "book" in result.lower()

    def test_update_book_title_on_book_section(self, monkeypatch):
        item = _make_book_section_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="BSEC1234",
            book_title="The Oxford Handbook of Philosophy",
            ctx=DummyContext(),
        )

        assert fake.update_calls[0]["data"]["bookTitle"] == "The Oxford Handbook of Philosophy"
        assert "Oxford Handbook" in result

    def test_update_multiple_new_fields(self, monkeypatch):
        """Update several new fields simultaneously on a journalArticle."""
        item = _make_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            volume="21",
            issue="4",
            pages="27-61",
            publisher="Springer",
            ctx=DummyContext(),
        )

        d = fake.update_calls[0]["data"]
        assert d["volume"] == "21"
        assert d["issue"] == "4"
        assert d["pages"] == "27-61"
        assert d["publisher"] == "Springer"
        assert "Successfully" in result

    def test_update_book_section_multiple_fields(self, monkeypatch):
        """Update bookSection-specific fields together."""
        item = _make_book_section_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="BSEC1234",
            book_title="Collected Essays",
            edition="2nd",
            pages="100-150",
            isbn="978-0-000000-00-0",
            ctx=DummyContext(),
        )

        d = fake.update_calls[0]["data"]
        assert d["bookTitle"] == "Collected Essays"
        assert d["edition"] == "2nd"
        assert d["pages"] == "100-150"
        assert d["ISBN"] == "978-0-000000-00-0"
        assert "Successfully" in result


# ---------------------------------------------------------------------------
# Silent-skip warning: fields not valid for item type
# ---------------------------------------------------------------------------

class TestUpdateItemSkippedFields:

    def test_skipped_field_warning(self, monkeypatch):
        """Passing issue= on a book item should produce a skip warning."""
        item = _make_book_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="BOOK1234",
            issue="3",
            ctx=DummyContext(),
        )

        # No update should happen (only field was skipped)
        assert len(fake.update_calls) == 0
        # Warning should mention the param name (snake_case) and item type
        assert "issue" in result
        assert "book" in result.lower()
        assert "skip" in result.lower() or "not valid" in result.lower()

    def test_skipped_uses_param_names(self, monkeypatch):
        """Warning should use snake_case param names, not camelCase API names."""
        item = _make_book_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="BOOK1234",
            short_title="Brief",
            issue="3",
            ctx=DummyContext(),
        )

        # short_title should succeed on book, issue should be skipped
        assert len(fake.update_calls) == 1
        # The warning should say "issue" not "issue" (same here), but
        # for shortTitle -> short_title, if it were skipped it would
        # use "short_title" not "shortTitle"
        assert "issue" in result

    def test_mixed_valid_and_skipped(self, monkeypatch):
        """Valid fields should apply; invalid ones should be warned about."""
        item = _make_book_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="BOOK1234",
            edition="2nd",
            issue="3",
            pages="100-200",
            ctx=DummyContext(),
        )

        # edition should be applied (valid for book)
        assert len(fake.update_calls) == 1
        assert fake.update_calls[0]["data"]["edition"] == "2nd"
        assert "Successfully" in result
        # issue and pages should be skipped (not valid for book)
        assert "issue" in result
        assert "pages" in result
        assert "book" in result.lower()

    def test_all_fields_skipped(self, monkeypatch):
        """If all fields are skipped, return no-changes message with warning."""
        item = _make_book_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="BOOK1234",
            issue="3",
            pages="100-200",
            ctx=DummyContext(),
        )

        assert len(fake.update_calls) == 0
        assert "No changes" in result
        assert "issue" in result
        assert "pages" in result

    def test_existing_field_skipped_on_wrong_type(self, monkeypatch):
        """Existing fields (e.g., publication_title) should also warn if not valid for type."""
        item = _make_book_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="BOOK1234",
            publication_title="Some Journal",
            edition="2nd",
            ctx=DummyContext(),
        )

        # edition should apply, publication_title should be skipped
        assert len(fake.update_calls) == 1
        assert "publication_title" in result
        assert "book" in result.lower()

    def test_same_value_valid_plus_invalid(self, monkeypatch):
        """Same-value valid field + invalid field: no changes but skip warning shown."""
        item = _make_book_item(publisher="OUP")
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="BOOK1234",
            publisher="OUP",
            issue="2",
            ctx=DummyContext(),
        )

        # publisher value unchanged -> no changes; issue skipped -> warning
        assert len(fake.update_calls) == 0
        assert "No changes" in result
        assert "issue" in result

    def test_clear_field_with_empty_string(self, monkeypatch):
        """Setting a field to empty string should clear it and show in diff."""
        item = _make_item(abstract="Some abstract text")
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            abstract="",
            ctx=DummyContext(),
        )

        assert fake.update_calls[0]["data"]["abstractNote"] == ""
        assert "Successfully" in result
        assert "Some abstract text" in result

    def test_no_op_same_value(self, monkeypatch):
        """Providing a value identical to existing should return no changes."""
        item = _make_item(title="Same Title")
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            title="Same Title",
            ctx=DummyContext(),
        )

        assert len(fake.update_calls) == 0
        assert "No changes" in result


# ---------------------------------------------------------------------------
# item_type migration (#234)
# ---------------------------------------------------------------------------

class TestUpdateItemType:
    """Covers programmatic migration of records across Zotero item types.

    The typical failure mode: the Zotero Connector or DOI resolution
    misclassifies an item (e.g., a book landing page saved as journalArticle).
    Without MCP-side repair the only recourse is manual desktop editing,
    which defeats automated library management.
    """

    def test_migrate_journal_article_to_book(self, monkeypatch):
        """Tillman case from issue #234: a book miscoded as journalArticle."""
        item = _make_item(
            title="Stripped and Script",
            publication_title="Wrong Journal",  # stale type-specific field
            volume="21",
        )
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            item_type="book",
            ctx=DummyContext(),
        )

        assert len(fake.update_calls) == 1
        d = fake.update_calls[0]["data"]
        # Type flipped
        assert d["itemType"] == "book"
        # Title preserved (overlapping field)
        assert d["title"] == "Stripped and Script"
        # journalArticle-only field dropped
        assert "publicationTitle" not in d
        # Book-specific fields present (from template) — empty strings ok
        assert "publisher" in d
        assert "ISBN" in d
        # Creators, tags, collections preserved (internal bookkeeping)
        assert len(d["creators"]) == 1
        assert d["creators"][0]["lastName"] == "Doe"
        # Diff mentions the type change
        assert "item_type" in result
        assert "book" in result

    def test_migrate_preserves_tags_and_collections(self, monkeypatch):
        item = _make_item(
            tags=["keep-me", "also-me"],
            collections=["COLL0001", "COLL0002"],
        )
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        server.update_item(
            item_key="ABCD1234",
            item_type="book",
            ctx=DummyContext(),
        )

        d = fake.update_calls[0]["data"]
        tag_values = {t["tag"] for t in d["tags"]}
        assert tag_values == {"keep-me", "also-me"}
        assert set(d["collections"]) == {"COLL0001", "COLL0002"}

    def test_migrate_and_update_new_type_fields_together(self, monkeypatch):
        """After migrating, fields specific to the NEW type should be settable."""
        item = _make_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            item_type="book",
            isbn="978-0-8173-2044-5",
            edition="1st",
            ctx=DummyContext(),
        )

        d = fake.update_calls[0]["data"]
        assert d["itemType"] == "book"
        assert d["ISBN"] == "978-0-8173-2044-5"
        assert d["edition"] == "1st"
        assert "Successfully" in result

    def test_migrate_to_same_type_noop(self, monkeypatch):
        """item_type equal to current type should not reshape."""
        item = _make_item()
        fake = FakeZoteroForUpdate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            item_type="journalArticle",
            ctx=DummyContext(),
        )

        # No change recorded for item_type
        assert len(fake.update_calls) == 0
        assert "No changes" in result

    def test_invalid_item_type_rejected(self, monkeypatch):
        """An invalid item_type should produce a clear error, not a silent drop."""
        item = _make_item()

        class ZoteroThatRejectsTemplate(FakeZoteroForUpdate):
            def item_template(self, item_type_name):
                raise Exception(f"unknown type: {item_type_name}")

        fake = ZoteroThatRejectsTemplate(items=[item])
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (fake, fake))

        result = server.update_item(
            item_key="ABCD1234",
            item_type="notATypeAtAll",
            ctx=DummyContext(),
        )

        assert len(fake.update_calls) == 0
        assert "invalid" in result.lower() or "error" in result.lower()
        assert "notATypeAtAll" in result
