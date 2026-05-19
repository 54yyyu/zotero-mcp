"""Tests for status surfacing in format_item_metadata.

Regression coverage for two smells caught live during the MCP-smell-
rubric pass (Hasan et al., arXiv:2602.14878):

1. Trashed items (data.deleted == 1) were returned with full metadata
   and NO status indicator. The web API returns the deleted flag; the
   formatter just wasn't looking at it. An agent would reason about a
   trashed paper as if it were live.

2. The Collections line read "**Collections:** N collections" — a bare
   count with no keys. When a collection had been deleted, the item's
   data.collections array still contained the dangling key (the Zotero
   web API doesn't cascade collection-delete to items), so the count was
   stale. Listing the actual keys lets agents verify against
   zotero_search_collections.
"""

import importlib.util
import pathlib
import sys
from unittest.mock import MagicMock

for _mod_name in (
    "markitdown", "pyzotero", "pyzotero.zotero",
    "dotenv", "fastmcp", "mcp", "mcp.server",
    "zotero_mcp", "zotero_mcp.utils", "zotero_mcp._app",
):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

_client_path = pathlib.Path(__file__).parent.parent / "src" / "zotero_mcp" / "client.py"
_spec = importlib.util.spec_from_file_location("zotero_mcp.client", _client_path)
_client_mod = importlib.util.module_from_spec(_spec)
sys.modules["zotero_mcp.client"] = _client_mod
_spec.loader.exec_module(_client_mod)
format_item_metadata = _client_mod.format_item_metadata


def _make_item(deleted=None, collections=None, extra=None):
    data = {
        "key": "TESTKEY1",
        "itemType": "journalArticle",
        "title": "Test Article",
        "creators": [{"creatorType": "author", "lastName": "Smith", "firstName": "J."}],
        "date": "2024",
        "publicationTitle": "Test Journal",
    }
    if deleted is not None:
        data["deleted"] = deleted
    if collections is not None:
        data["collections"] = collections
    if extra:
        data.update(extra)
    return {"data": data}


# ---------------------------------------------------------------------------
# Trash status
# ---------------------------------------------------------------------------

class TestTrashStatus:
    def test_trashed_item_shows_in_trash(self):
        item = _make_item(deleted=1)
        output = format_item_metadata(item)
        assert "Trash" in output, (
            "format_item_metadata silently returned metadata for a trashed "
            "item without any indicator — agents would use the data as if "
            "the item were live."
        )

    def test_untrashed_item_has_no_trash_marker(self):
        item = _make_item(deleted=0)
        output = format_item_metadata(item)
        assert "Trash" not in output

    def test_missing_deleted_field_has_no_trash_marker(self):
        item = _make_item()  # no 'deleted' field at all
        output = format_item_metadata(item)
        assert "Trash" not in output

    def test_trash_status_independent_of_abstract_flag(self):
        """Trash status must appear whether or not abstracts are included."""
        item = _make_item(deleted=1, extra={"abstractNote": "Some abstract."})
        assert "Trash" in format_item_metadata(item, include_abstract=True)
        assert "Trash" in format_item_metadata(item, include_abstract=False)


# ---------------------------------------------------------------------------
# Collections listing
# ---------------------------------------------------------------------------

class TestCollectionsListing:
    def test_collection_keys_are_listed(self):
        item = _make_item(collections=["ABC12345", "XYZ67890"])
        output = format_item_metadata(item)
        assert "ABC12345" in output
        assert "XYZ67890" in output

    def test_empty_collections_omits_line(self):
        item = _make_item(collections=[])
        output = format_item_metadata(item)
        assert "Collections" not in output

    def test_missing_collections_field_omits_line(self):
        item = _make_item()
        output = format_item_metadata(item)
        assert "Collections" not in output

    def test_single_collection_still_listed(self):
        item = _make_item(collections=["9SU943GB"])
        output = format_item_metadata(item)
        assert "9SU943GB" in output


# ---------------------------------------------------------------------------
# Same coverage for format_item_result (list-mode formatter in utils.py)
# ---------------------------------------------------------------------------

_utils_path = pathlib.Path(__file__).parent.parent / "src" / "zotero_mcp" / "utils.py"
_utils_spec = importlib.util.spec_from_file_location("zotero_mcp.utils", _utils_path)
_utils_mod = importlib.util.module_from_spec(_utils_spec)
sys.modules["zotero_mcp.utils"] = _utils_mod
_utils_spec.loader.exec_module(_utils_mod)
format_item_result = _utils_mod.format_item_result


def _make_list_item(deleted=None, key="LISTKEY1"):
    data = {
        "key": key,
        "itemType": "journalArticle",
        "title": "List Mode Item",
        "date": "2024",
        "creators": [{"creatorType": "author", "lastName": "Doe", "firstName": "J."}],
    }
    if deleted is not None:
        data["deleted"] = deleted
    return {"key": key, "data": data}


class TestFormatItemResultTrashStatus:
    """pyzotero filters trashed items out of most list endpoints, but not
    every call site (e.g. includeTrashed=1). The formatter must never
    silently render a trashed item as live."""

    def test_trashed_list_item_shows_in_trash(self):
        item = _make_list_item(deleted=1)
        output = "\n".join(format_item_result(item))
        assert "Trash" in output

    def test_untrashed_list_item_no_marker(self):
        item = _make_list_item(deleted=0)
        output = "\n".join(format_item_result(item))
        assert "Trash" not in output

    def test_missing_deleted_field_no_marker(self):
        item = _make_list_item()
        output = "\n".join(format_item_result(item))
        assert "Trash" not in output
