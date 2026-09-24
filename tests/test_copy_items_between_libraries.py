"""Tests for zotero_copy_items_between_libraries (Issue #560)."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import DummyContext, FakeZotero

from zotero_mcp import client as _client
from zotero_mcp import server
from zotero_mcp.tools import _helpers
from zotero_mcp.tools.write import _copy_library, copy_items_between_libraries
from zotero_mcp.toolsets import TOOLSETS, validate_toolsets


@pytest.fixture
def dummy_ctx():
    return DummyContext()


# ---------------------------------------------------------------------------
# 1. Library Specification Resolution
# ---------------------------------------------------------------------------


class TestLibraryResolution:
    def test_default_is_active_personal_library_in_web_mode(self, monkeypatch):
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)
        monkeypatch.setenv("ZOTERO_LIBRARY_ID", "12345")
        monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "user")
        _client.clear_active_library()

        assert _copy_library(None, None) == ("12345", "user", 0)

    def test_default_follows_active_group(self):
        _client.set_active_library("6069773", "group")
        try:
            assert _copy_library(None, None) == ("6069773", "group", 6069773)
        finally:
            _client.clear_active_library()

    def test_personal_library_is_0_in_local_mode(self, monkeypatch):
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: True)
        for alias in ("0", "user", "USER", 0):
            assert _copy_library(alias, None) == ("0", "user", 0)

    def test_personal_library_is_the_user_id_in_web_mode(self, monkeypatch):
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)
        monkeypatch.setenv("ZOTERO_LIBRARY_ID", "98765")
        assert _copy_library("0", None) == ("98765", "user", 0)

    def test_group_library_numeric(self):
        assert _copy_library("54321", None) == ("54321", "group", 54321)
        assert _copy_library(54321, "group") == ("54321", "group", 54321)

    def test_feed_library(self):
        assert _copy_library("12", "feed") == ("12", "feed", 12)

    def test_invalid_library_id_raises(self):
        with pytest.raises(ValueError, match="Invalid library_id"):
            _copy_library("not_a_number_or_user", None)

    def test_unknown_group_is_rejected_in_local_mode(self, monkeypatch):
        monkeypatch.setattr(
            "zotero_mcp.tools.retrieval.validate_library_switch",
            lambda library_id, library_type: f"Group '{library_id}' not found.",
        )
        with pytest.raises(ValueError, match="Group '999' not found"):
            _copy_library("999", "group")


# ---------------------------------------------------------------------------
# 2. Item Data Sanitization
# ---------------------------------------------------------------------------


class TestItemSanitization:
    def test_strips_read_only_and_library_fields(self):
        zot = FakeZotero()
        source_data = {
            "key": "OLD12345",
            "version": 42,
            "itemType": "journalArticle",
            "title": "Quantum Supremacy",
            "creators": [{"creatorType": "author", "firstName": "Alice", "lastName": "Smith"}],
            "DOI": "10.1000/182",
            "publicationTitle": "Nature",
            "collections": ["COL1", "COL2"],
            "relations": {"owl:sameAs": ["http://zotero.org/users/0/items/OLD12345"]},
            "dateAdded": "2024-01-01T00:00:00Z",
            "dateModified": "2024-01-02T00:00:00Z",
            "deleted": 1,
            "parentItem": "PARENT1",
            "tags": [{"tag": "physics"}],
            "extra": "Citation Key: smith2024",
        }

        sanitized = _helpers._sanitize_item_for_creation(source_data, zot, copy_tags=True)

        assert "key" not in sanitized
        assert "version" not in sanitized
        assert "dateAdded" not in sanitized
        assert "dateModified" not in sanitized
        assert "collections" not in sanitized
        assert "relations" not in sanitized
        assert "deleted" not in sanitized
        assert "parentItem" not in sanitized

        assert sanitized["itemType"] == "journalArticle"
        assert sanitized["title"] == "Quantum Supremacy"
        assert sanitized["DOI"] == "10.1000/182"
        assert sanitized["publicationTitle"] == "Nature"
        assert sanitized["extra"] == "Citation Key: smith2024"
        assert sanitized["creators"] == source_data["creators"]
        assert sanitized["tags"] == [{"tag": "physics"}]

    def test_tags_excluded_when_copy_tags_false(self):
        zot = FakeZotero()
        source_data = {
            "itemType": "book",
            "title": "Clean Code",
            "tags": [{"tag": "programming"}],
        }

        sanitized = _helpers._sanitize_item_for_creation(source_data, zot, copy_tags=False)
        assert sanitized["tags"] == []

    def test_missing_item_type_raises(self):
        zot = FakeZotero()
        with pytest.raises(ValueError, match="missing 'itemType'"):
            _helpers._sanitize_item_for_creation({"title": "No Type"}, zot)


# ---------------------------------------------------------------------------
# 3. Validation and Error Cases in copy_items_between_libraries
# ---------------------------------------------------------------------------


class TestCopyItemsValidation:
    def test_empty_item_keys(self, dummy_ctx):
        res = copy_items_between_libraries("", target_library_id="6069773", ctx=dummy_ctx)
        assert "No item_keys provided" in res

    def test_invalid_if_exists(self, dummy_ctx):
        res = copy_items_between_libraries(
            "KEY1", target_library_id="6069773", if_exists="invalid", ctx=dummy_ctx
        )
        assert "if_exists must be one of" in res

    def test_same_source_and_target_library(self, dummy_ctx):
        _client.clear_active_library()
        res = copy_items_between_libraries(
            "KEY1",
            source_library_id="6069773",
            source_library_type="group",
            target_library_id="6069773",
            target_library_type="group",
            ctx=dummy_ctx,
        )
        assert "Source and target libraries are the same" in res

    def test_target_library_is_feed(self, dummy_ctx):
        res = copy_items_between_libraries(
            "KEY1",
            source_library_id="0",
            source_library_type="user",
            target_library_id="12",
            target_library_type="feed",
            ctx=dummy_ctx,
        )
        assert "Cannot copy items to an RSS feed library" in res


# ---------------------------------------------------------------------------
# 4. Copy Execution (Single, Batch, Notes, Attachments, Collections, Dedup)
# ---------------------------------------------------------------------------


class TestCopyItemsExecution:
    @pytest.fixture
    def setup_clients(self, monkeypatch):
        source_zot = FakeZotero()
        source_zot.library_id = "0"
        source_zot.library_type = "users"

        target_zot = FakeZotero()
        target_zot.library_id = "6069773"
        target_zot.library_type = "groups"

        def fake_get_client(library_id=None, library_type=None):
            active = _client.get_active_library()
            if active.get("library_id") == "6069773":
                return target_zot
            return source_zot

        monkeypatch.setattr(_client, "get_zotero_client", fake_get_client)
        monkeypatch.setattr(_helpers, "_get_write_client", lambda ctx: (target_zot, target_zot))
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

        return source_zot, target_zot

    def test_copy_single_item(self, setup_clients, dummy_ctx):
        source_zot, target_zot = setup_clients
        source_zot._items = [
            {
                "key": "SRC0001",
                "version": 1,
                "data": {
                    "key": "SRC0001",
                    "version": 1,
                    "itemType": "journalArticle",
                    "title": "Attention Is All You Need",
                    "creators": [{"creatorType": "author", "firstName": "Ashish", "lastName": "Vaswani"}],
                    "DOI": "10.5555/3295222.3295349",
                    "extra": "Citation Key: vaswani2017",
                },
            }
        ]

        res = copy_items_between_libraries(
            "SRC0001",
            source_library_id="0",
            source_library_type="user",
            target_library_id="6069773",
            target_library_type="group",
            ctx=dummy_ctx,
        )

        assert "✅ Attention Is All You Need" in res
        assert "**Source Key:** `SRC0001`" in res
        assert "**Target Key:** `KEY0000`" in res
        assert len(target_zot.created) == 1
        created_data = target_zot.created[0]
        assert created_data["title"] == "Attention Is All You Need"
        assert created_data["DOI"] == "10.5555/3295222.3295349"
        assert created_data["extra"] == "Citation Key: vaswani2017"

    def test_copy_batch_items(self, setup_clients, dummy_ctx):
        source_zot, target_zot = setup_clients
        source_zot._items = [
            {
                "key": "SRC0001",
                "version": 1,
                "data": {"key": "SRC0001", "itemType": "journalArticle", "title": "Paper 1"},
            },
            {
                "key": "SRC0002",
                "version": 1,
                "data": {"key": "SRC0002", "itemType": "book", "title": "Book 2"},
            },
        ]

        res = copy_items_between_libraries(
            ["SRC0001", "SRC0002"],
            source_library_id="0",
            target_library_id="6069773",
            ctx=dummy_ctx,
        )

        assert "Paper 1" in res
        assert "Book 2" in res
        assert len(target_zot.created) == 2

    def test_source_item_not_found(self, setup_clients, dummy_ctx):
        source_zot, target_zot = setup_clients
        source_zot._items = []

        res = copy_items_between_libraries(
            "NONEXISTENT",
            source_library_id="0",
            target_library_id="6069773",
            ctx=dummy_ctx,
        )

        assert "❌ Item `NONEXISTENT`" in res
        assert "not found in source library" in res
        assert len(target_zot.created) == 0

    def test_rejects_top_level_attachment_copy(self, setup_clients, dummy_ctx):
        source_zot, target_zot = setup_clients
        source_zot._items = [
            {
                "key": "ATT0001",
                "version": 1,
                "data": {"key": "ATT0001", "itemType": "attachment", "title": "file.pdf"},
            }
        ]

        res = copy_items_between_libraries(
            "ATT0001",
            source_library_id="0",
            target_library_id="6069773",
            ctx=dummy_ctx,
        )

        assert "❌ Item `ATT0001`" in res
        assert "Top-level copying of attachments, notes, or annotations is not supported" in res

    def test_copies_child_notes(self, setup_clients, dummy_ctx):
        source_zot, target_zot = setup_clients
        source_zot._items = [
            {
                "key": "SRC0001",
                "version": 1,
                "data": {"key": "SRC0001", "itemType": "journalArticle", "title": "Parent Paper"},
            }
        ]
        source_zot._children["SRC0001"] = [
            {
                "key": "NOTE001",
                "data": {
                    "itemType": "note",
                    "note": "<p>Important reading note</p>",
                    "tags": [{"tag": "summary"}],
                },
            }
        ]

        res = copy_items_between_libraries(
            "SRC0001",
            source_library_id="0",
            target_library_id="6069773",
            copy_notes=True,
            ctx=dummy_ctx,
        )

        assert "**Notes copied:** 1" in res
        # target_zot should have 2 creations: parent item and child note
        assert len(target_zot.created) == 2
        note_created = target_zot.created[1]
        assert note_created["itemType"] == "note"
        assert note_created["note"] == "<p>Important reading note</p>"
        assert note_created["parentItem"] == "KEY0000"
        assert note_created["tags"] == [{"tag": "summary"}]

    def test_copies_child_linked_url_attachment(self, setup_clients, dummy_ctx):
        source_zot, target_zot = setup_clients
        source_zot._items = [
            {
                "key": "SRC0001",
                "version": 1,
                "data": {"key": "SRC0001", "itemType": "journalArticle", "title": "Parent Paper"},
            }
        ]
        source_zot._children["SRC0001"] = [
            {
                "key": "ATT_LINK",
                "data": {
                    "itemType": "attachment",
                    "linkMode": "linked_url",
                    "title": "Online Version",
                    "url": "https://example.com/paper.html",
                },
            }
        ]

        res = copy_items_between_libraries(
            "SRC0001",
            source_library_id="0",
            target_library_id="6069773",
            copy_attachments=True,
            ctx=dummy_ctx,
        )

        assert "**Attachments copied:** 1" in res
        assert len(target_zot.created) == 2
        att_created = target_zot.created[1]
        assert att_created["itemType"] == "attachment"
        assert att_created["url"] == "https://example.com/paper.html"
        assert att_created["parentItem"] == "KEY0000"

    @staticmethod
    def _parent_with_file(source_zot, **child_extra):
        source_zot._items = [
            {
                "key": "SRC0001",
                "version": 1,
                "data": {"key": "SRC0001", "itemType": "journalArticle", "title": "Parent Paper"},
            }
        ]
        source_zot._children["SRC0001"] = [
            {
                "key": "ATT_FILE",
                "data": {
                    "itemType": "attachment",
                    "linkMode": "imported_file",
                    "title": "paper.pdf",
                    "filename": "paper.pdf",
                    "contentType": "application/pdf",
                    **child_extra,
                },
            }
        ]

    def test_copies_child_file_attachment_from_the_source_library(
        self, setup_clients, monkeypatch, dummy_ctx
    ):
        """The file is fetched through a client scoped to the source library,
        not the target one that is active while the copy runs."""
        source_zot, target_zot = setup_clients
        self._parent_with_file(source_zot)

        real_download = _client.download_attachment_file
        download_clients = []

        def spy_download(*args, **kwargs):
            download_clients.append(kwargs.get("web_client"))
            return real_download(*args, **kwargs)

        monkeypatch.setattr(_client, "download_attachment_file", spy_download)

        attached_files = []

        def fake_attach_and_verify(write_zot, display_name, path, item_key, ctx, content_type=None):
            attached_files.append((display_name, Path(path).read_bytes(), item_key))
            return True, "", "NEW_ATT_KEY"

        monkeypatch.setattr(_helpers, "_attach_and_verify", fake_attach_and_verify)

        res = copy_items_between_libraries(
            "SRC0001",
            source_library_id="0",
            target_library_id="6069773",
            copy_attachments=True,
            ctx=dummy_ctx,
        )

        assert "**Attachments copied:** 1 (File: paper.pdf)" in res
        assert download_clients == [source_zot]
        # FakeZotero.dump wrote these bytes: the Web API step served the file.
        assert attached_files == [("paper.pdf", b"%PDF-1.4 fake", "KEY0000")]

    def test_failed_file_copy_is_reported(self, setup_clients, monkeypatch, dummy_ctx):
        source_zot, target_zot = setup_clients
        self._parent_with_file(source_zot)
        monkeypatch.setattr(
            _client,
            "download_attachment_file",
            lambda *a, **k: _client.AttachmentDownloadResult(
                path=None, source=None, errors=["Web API: 404 Not Found"]
            ),
        )

        res = copy_items_between_libraries(
            "SRC0001",
            source_library_id="0",
            target_library_id="6069773",
            ctx=dummy_ctx,
        )

        assert "**Attachments copied:** 0" in res
        assert (
            "**Not copied:** File: paper.pdf: could not fetch the file "
            "(Web API: 404 Not Found)" in res
        )
        assert "annotations are not copied" in res
        assert len(target_zot.created) == 1  # the parent only

    def test_trashed_children_are_not_copied(self, setup_clients, dummy_ctx):
        source_zot, target_zot = setup_clients
        self._parent_with_file(source_zot, deleted=1)
        source_zot._children["SRC0001"].append(
            {"key": "NOTE_DEL", "data": {"itemType": "note", "note": "<p>old</p>", "deleted": 1}}
        )

        res = copy_items_between_libraries(
            "SRC0001",
            source_library_id="0",
            target_library_id="6069773",
            ctx=dummy_ctx,
        )

        assert "**Notes copied:** 0" in res
        assert "**Attachments copied:** 0" in res
        assert "Not copied" not in res
        assert len(target_zot.created) == 1

    def test_trashed_item_is_not_copied(self, setup_clients, dummy_ctx):
        source_zot, target_zot = setup_clients
        source_zot._items = [
            {
                "key": "SRC0001",
                "version": 1,
                "data": {"key": "SRC0001", "itemType": "book", "title": "Old", "deleted": 1},
            }
        ]

        res = copy_items_between_libraries(
            "SRC0001",
            source_library_id="0",
            target_library_id="6069773",
            ctx=dummy_ctx,
        )

        assert "is in the trash" in res
        assert target_zot.created == []

    def test_source_collections_not_copied(self, setup_clients, dummy_ctx):
        source_zot, target_zot = setup_clients
        source_zot._items = [
            {
                "key": "SRC0001",
                "version": 1,
                "data": {
                    "key": "SRC0001",
                    "itemType": "journalArticle",
                    "title": "Paper In Collection",
                    "collections": ["COL_SOURCE_1", "COL_SOURCE_2"],
                },
            }
        ]

        res = copy_items_between_libraries(
            "SRC0001",
            source_library_id="0",
            target_library_id="6069773",
            ctx=dummy_ctx,
        )

        assert "✅ Paper In Collection" in res
        assert len(target_zot.created) == 1
        # Collections are library-specific and must not be copied
        assert "collections" not in target_zot.created[0]

    def test_dedup_skip(self, setup_clients, monkeypatch, dummy_ctx):
        source_zot, target_zot = setup_clients
        source_zot._items = [
            {
                "key": "SRC0001",
                "version": 1,
                "data": {
                    "key": "SRC0001",
                    "itemType": "journalArticle",
                    "title": "Existing Paper",
                    "DOI": "10.1000/182",
                },
            }
        ]

        # find_existing_items finds an existing item in target library
        monkeypatch.setattr(
            _helpers,
            "find_existing_items",
            lambda zot, **kwargs: [{"key": "EXISTING_KEY", "data": {"key": "EXISTING_KEY", "collections": ["COL_EX"]}}],
        )

        res = copy_items_between_libraries(
            "SRC0001",
            source_library_id="0",
            target_library_id="6069773",
            if_exists="skip",
            ctx=dummy_ctx,
        )

        assert "**Status:** Skipped" in res
        assert "**Target Key:** `EXISTING_KEY`" in res
        # No new item created in target library
        assert len(target_zot.created) == 0

    def test_if_exists_defaults_to_skip(self, setup_clients, monkeypatch, dummy_ctx):
        """A retried/repeated call must not double items unless 'duplicate' is explicit."""
        source_zot, target_zot = setup_clients
        source_zot._items = [
            {
                "key": "SRC0001",
                "version": 1,
                "data": {
                    "key": "SRC0001",
                    "itemType": "journalArticle",
                    "title": "Existing Paper",
                    "DOI": "10.1000/182",
                },
            }
        ]

        monkeypatch.setattr(
            _helpers,
            "find_existing_items",
            lambda zot, **kwargs: [{"key": "EXISTING_KEY", "data": {"key": "EXISTING_KEY"}}],
        )

        res = copy_items_between_libraries(
            "SRC0001",
            source_library_id="0",
            target_library_id="6069773",
            ctx=dummy_ctx,
        )

        assert "**Status:** Skipped" in res
        assert len(target_zot.created) == 0

    def test_if_exists_duplicate_must_be_explicit(self, setup_clients, monkeypatch, dummy_ctx):
        source_zot, target_zot = setup_clients
        source_zot._items = [
            {
                "key": "SRC0001",
                "version": 1,
                "data": {
                    "key": "SRC0001",
                    "itemType": "journalArticle",
                    "title": "Existing Paper",
                    "DOI": "10.1000/182",
                },
            }
        ]

        monkeypatch.setattr(
            _helpers,
            "find_existing_items",
            lambda zot, **kwargs: [{"key": "EXISTING_KEY", "data": {"key": "EXISTING_KEY"}}],
        )

        res = copy_items_between_libraries(
            "SRC0001",
            source_library_id="0",
            target_library_id="6069773",
            if_exists="duplicate",
            ctx=dummy_ctx,
        )

        assert "**Status:** Copied" in res
        assert len(target_zot.created) == 1


    def test_hyphenated_isbn_is_normalized_before_matching(
        self, setup_clients, monkeypatch, dummy_ctx
    ):
        source_zot, target_zot = setup_clients
        source_zot._items = [
            {
                "key": "SRC0001",
                "version": 1,
                "data": {
                    "key": "SRC0001",
                    "itemType": "book",
                    "title": "Introduction to Algorithms",
                    # Zotero's ISBN field may hold several, hyphenated.
                    "ISBN": "978-0-262-03384-8 0-262-03384-4",
                },
            }
        ]
        lookups = []

        def fake_find(zot, **kwargs):
            lookups.append(kwargs)
            return [{"key": "EXISTING_KEY", "data": {"key": "EXISTING_KEY"}}]

        monkeypatch.setattr(_helpers, "find_existing_items", fake_find)

        res = copy_items_between_libraries(
            "SRC0001",
            source_library_id="0",
            target_library_id="6069773",
            ctx=dummy_ctx,
        )

        assert lookups[0]["isbn"] == "9780262033848"
        assert "**Status:** Skipped" in res
        assert target_zot.created == []


# ---------------------------------------------------------------------------
# 5. Reading the source from zotero.sqlite
# ---------------------------------------------------------------------------


class _FakeReader:
    """Records the scope each read asked for; one group and one feed hold items."""

    def __init__(self):
        self.calls = []
        self.scoped = {
            ("group", 6015547): {"key": "GRP0001", "version": 1,
                                 "data": {"key": "GRP0001", "itemType": "book", "title": "Group Book"}},
            ("library", 12): {"key": "FEED001", "version": 1,
                              "data": {"key": "FEED001", "itemType": "journalArticle",
                                       "title": "Feed Paper"}},
        }

    def _lookup(self, group_id, library_id):
        if library_id is not None:
            return self.scoped.get(("library", library_id))
        return self.scoped.get(("group", group_id))

    def get_full_items(self, keys, *, group_id=0, include_trashed=True, library_id=None):
        self.calls.append(("items", group_id, library_id))
        item = self._lookup(group_id, library_id)
        return {k: item for k in keys if item and item["key"] == k}

    def get_children_of(self, keys, *, item_type=None, group_id=0, library_id=None):
        self.calls.append(("children", group_id, library_id))
        return {k: [] for k in keys}


class TestSqliteSource:
    @pytest.fixture
    def sqlite_source(self, monkeypatch):
        from zotero_mcp import library as _library

        reader = _FakeReader()
        monkeypatch.setattr(_library, "configured_backend", lambda: "sqlite")
        monkeypatch.setattr(_library, "_sqlite_reader", lambda: reader)
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: True)
        target_zot = FakeZotero()
        monkeypatch.setattr(_helpers, "_get_write_client", lambda ctx: (target_zot, target_zot))
        _client.clear_active_library()
        return reader, target_zot

    def test_group_source_is_read_in_its_own_scope(self, sqlite_source, dummy_ctx):
        reader, target_zot = sqlite_source

        res = copy_items_between_libraries(
            "GRP0001",
            source_library_id="6015547",
            target_library_id="0",
            copy_attachments=False,
            ctx=dummy_ctx,
        )

        assert "✅ Group Book" in res
        assert reader.calls == [("items", 6015547, None), ("children", 6015547, None)]
        assert target_zot.created[0]["title"] == "Group Book"
        # The source scope does not outlive the call.
        assert _client.get_active_library() == {}

    def test_feed_source_is_read_by_library_id(self, sqlite_source, dummy_ctx):
        reader, target_zot = sqlite_source

        res = copy_items_between_libraries(
            "FEED001",
            source_library_id="12",
            source_library_type="feed",
            target_library_id="0",
            ctx=dummy_ctx,
        )

        assert "✅ Feed Paper" in res
        assert [call[2] for call in reader.calls] == [12, 12]


# ---------------------------------------------------------------------------
# 6. Toolset Registry & Export Verification
# ---------------------------------------------------------------------------


class TestToolsetRegistration:
    def test_tool_in_libraries_copy_toolset(self):
        assert "zotero_copy_items_between_libraries" in TOOLSETS["libraries-copy"]
        assert "zotero_copy_items_between_libraries" not in TOOLSETS["libraries"]

    def test_libraries_copy_toolset_is_opt_in(self):
        from zotero_mcp.toolsets import DEFAULT_ON

        assert "libraries-copy" not in DEFAULT_ON

    def test_toolsets_registry_validates(self):
        import asyncio

        from zotero_mcp.server import mcp
        from zotero_mcp.toolsets import optional_tool_names

        mcp.enable(names=optional_tool_names())
        registered = {t.name for t in asyncio.run(mcp.list_tools())}
        assert not validate_toolsets(registered)

    def test_server_exports_function(self):
        assert hasattr(server, "copy_items_between_libraries")
        assert callable(server.copy_items_between_libraries)

