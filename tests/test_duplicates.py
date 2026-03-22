"""Tests for Features 7-8: find_duplicates and merge_duplicates."""

import pytest

from conftest import DummyContext, FakeZotero, _FakeResponse
from zotero_mcp import server


# ---------------------------------------------------------------------------
# Helpers: item factory and extended FakeZotero for duplicates
# ---------------------------------------------------------------------------

def _make_item(key, title, doi=None, collections=None, tags=None, version=1):
    """Build a minimal Zotero item dict."""
    data = {
        "key": key,
        "title": title,
        "itemType": "journalArticle",
        "creators": [{"firstName": "A", "lastName": "Author", "creatorType": "author"}],
        "date": "2024",
        "DOI": doi or "",
        "tags": [{"tag": t} for t in (tags or [])],
        "collections": collections or [],
        "deleted": False,
    }
    return {"key": key, "version": version, "data": data}


class _FakeHttpClient:
    """Fake httpx.Client that records PATCH calls for trash operations."""

    def __init__(self):
        self.patch_calls = []

    def patch(self, url="", headers=None, content=""):
        self.patch_calls.append({"url": url, "headers": headers, "content": content})
        return _FakeResponse(204)


class FakeZoteroForDuplicates(FakeZotero):
    """Extended stub that tracks write operations for merge testing."""

    def __init__(self):
        super().__init__()
        self.addto_calls = []        # [(collection_key, items)]
        self.update_calls = []       # [item_dict, ...]
        self._version_counter = 100  # auto-increment on update_item
        # Attributes needed for direct PATCH (trash operation)
        self.client = _FakeHttpClient()
        self.endpoint = "https://api.zotero.org"
        self.library_type = "users"
        self.library_id = "12345"

    def item(self, item_key):
        for it in self._items:
            if it.get("key") == item_key:
                return it
        raise KeyError(f"Item {item_key} not found")

    def update_item(self, item, **kwargs):
        self.update_calls.append(item)
        # Simulate server version bump
        item["version"] = self._version_counter
        self._version_counter += 1
        return _FakeResponse(204)

    def addto_collection(self, collection_key, items, **kwargs):
        self.addto_calls.append((collection_key, items))
        return _FakeResponse(204)

    def everything(self, method, *args, **kwargs):
        """Simulate pyzotero everything(): call the method reference."""
        if callable(method):
            return method(*args, **kwargs)
        return method


# ---------------------------------------------------------------------------
# Feature 7: find_duplicates
# ---------------------------------------------------------------------------

class TestFindDuplicates:
    """Tests for zotero_find_duplicates."""

    def test_happy_path_title_grouping(self, monkeypatch, dummy_ctx):
        """Items with the same normalized title are grouped together."""
        fake = FakeZoteroForDuplicates()
        fake._items = [
            _make_item("A1", "Machine Learning Basics"),
            _make_item("A2", "Machine Learning Basics"),
            _make_item("A3", "Deep Learning Overview"),
        ]
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)

        result = server.find_duplicates(method="title", ctx=dummy_ctx)

        # Should find one duplicate group containing A1 and A2
        assert "Machine Learning Basics" in result
        assert "A1" in result
        assert "A2" in result
        # A3 is unique, should NOT appear as a duplicate group
        assert "Deep Learning Overview" not in result or "duplicate" not in result.lower().split("deep")[0]

    def test_doi_matching(self, monkeypatch, dummy_ctx):
        """Items sharing the same DOI are grouped as duplicates."""
        fake = FakeZoteroForDuplicates()
        fake._items = [
            _make_item("B1", "Title Alpha", doi="10.1234/abc"),
            _make_item("B2", "Title Beta", doi="10.1234/abc"),
            _make_item("B3", "Title Gamma", doi="10.5678/xyz"),
        ]
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)

        result = server.find_duplicates(method="doi", ctx=dummy_ctx)

        assert "B1" in result
        assert "B2" in result
        # B3 has a unique DOI, should not be grouped with B1/B2
        assert "10.1234/abc" in result

    def test_title_normalization(self, monkeypatch, dummy_ctx):
        """Normalization strips articles, punctuation, and case differences."""
        fake = FakeZoteroForDuplicates()
        fake._items = [
            _make_item("C1", "The Quick Brown Fox"),
            _make_item("C2", "quick brown fox"),
            _make_item("C3", "  QUICK  BROWN  FOX!  "),
        ]
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)

        result = server.find_duplicates(method="title", ctx=dummy_ctx)

        # All three should be in the same group
        assert "C1" in result
        assert "C2" in result
        assert "C3" in result

    def test_no_duplicates_found(self, monkeypatch, dummy_ctx):
        """When all items are unique, return a message saying no duplicates."""
        fake = FakeZoteroForDuplicates()
        fake._items = [
            _make_item("D1", "Unique Title One"),
            _make_item("D2", "Unique Title Two"),
            _make_item("D3", "Unique Title Three"),
        ]
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)

        result = server.find_duplicates(method="both", ctx=dummy_ctx)

        assert "no duplicate" in result.lower() or "0" in result

    def test_collection_scoping(self, monkeypatch, dummy_ctx):
        """When collection_key is provided, only items in that collection are checked."""
        fake = FakeZoteroForDuplicates()
        fake._items = [
            _make_item("E1", "Same Title", collections=["COL1"]),
            _make_item("E2", "Same Title", collections=["COL1"]),
            _make_item("E3", "Same Title", collections=["COL2"]),  # different collection
        ]
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)

        result = server.find_duplicates(
            method="title", collection_key="COL1", ctx=dummy_ctx
        )

        # E1 and E2 should be grouped; E3 is in COL2, should not appear
        assert "E1" in result
        assert "E2" in result
        assert "E3" not in result

    def test_large_library_cap(self, monkeypatch, dummy_ctx):
        """Libraries with >5000 items return an error asking user to scope by collection."""
        fake = FakeZoteroForDuplicates()
        # Simulate a large library by returning 5001 items
        fake._items = [_make_item(f"X{i:04d}", f"Item {i}") for i in range(5001)]
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)

        result = server.find_duplicates(method="both", ctx=dummy_ctx)

        assert "5,000" in result or "5000" in result or "5001" in result
        assert "collection" in result.lower()

    def test_both_method_combines_title_and_doi(self, monkeypatch, dummy_ctx):
        """method='both' catches duplicates via title OR DOI."""
        fake = FakeZoteroForDuplicates()
        fake._items = [
            _make_item("F1", "Alpha Paper", doi="10.1000/alpha"),
            _make_item("F2", "Different Title", doi="10.1000/alpha"),  # same DOI
            _make_item("F3", "Beta Paper"),
            _make_item("F4", "Beta Paper"),  # same title
        ]
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)

        result = server.find_duplicates(method="both", ctx=dummy_ctx)

        # DOI group
        assert "F1" in result
        assert "F2" in result
        # Title group
        assert "F3" in result
        assert "F4" in result


# ---------------------------------------------------------------------------
# Feature 8: merge_duplicates
# ---------------------------------------------------------------------------

class TestMergeDuplicatesDryRun:
    """Tests for merge_duplicates with confirm=False (dry-run)."""

    def test_dry_run_returns_preview(self, monkeypatch, dummy_ctx):
        """Dry-run shows a preview of what would happen without writing."""
        fake = FakeZoteroForDuplicates()
        fake._items = [
            _make_item("KEEP", "Keeper Item", tags=["tagA"]),
            _make_item("DUP1", "Duplicate One", tags=["tagB"]),
        ]
        fake._children = {
            "DUP1": [
                {"key": "NOTE1", "version": 1, "data": {
                    "itemType": "note", "parentItem": "DUP1", "note": "A note",
                }},
            ],
        }
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake, fake))

        result = server.merge_duplicates(
            keeper_key="KEEP", duplicate_keys=["DUP1"], confirm=False, ctx=dummy_ctx
        )

        # Should mention it is a preview / dry-run
        assert "confirm" in result.lower() or "preview" in result.lower() or "dry" in result.lower()
        # NO write methods should have been called
        assert len(fake.update_calls) == 0
        assert len(fake.addto_calls) == 0

    def test_dry_run_no_writes(self, monkeypatch, dummy_ctx):
        """Explicitly verify zero update_item and addto_collection calls in dry-run."""
        fake = FakeZoteroForDuplicates()
        fake._items = [
            _make_item("KEEP", "Keeper", tags=["t1"], collections=["C1"]),
            _make_item("DUP1", "Dup 1", tags=["t2"], collections=["C2"]),
            _make_item("DUP2", "Dup 2", tags=["t3"], collections=["C3"]),
        ]
        fake._children = {
            "DUP1": [{"key": "CH1", "version": 1, "data": {"itemType": "note", "parentItem": "DUP1"}}],
            "DUP2": [],
        }
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake, fake))

        server.merge_duplicates(
            keeper_key="KEEP", duplicate_keys=["DUP1", "DUP2"], confirm=False, ctx=dummy_ctx
        )

        assert fake.update_calls == []
        assert fake.addto_calls == []


class TestMergeDuplicatesConfirm:
    """Tests for merge_duplicates with confirm=True."""

    def _setup_merge(self, monkeypatch):
        """Shared setup: keeper + two duplicates with tags, collections, children."""
        fake = FakeZoteroForDuplicates()
        # Child items (must also be in _items so write_zot.item(child_key) works
        # during re-parenting)
        note1 = {"key": "NOTE1", "version": 10, "data": {
            "itemType": "note", "parentItem": "DUP1", "note": "child note",
        }}
        att1 = {"key": "ATT1", "version": 11, "data": {
            "itemType": "attachment", "parentItem": "DUP1",
            "contentType": "application/pdf",
        }}
        annot1 = {"key": "ANNOT1", "version": 12, "data": {
            "itemType": "annotation", "parentItem": "DUP2",
        }}
        fake._items = [
            _make_item("KEEP", "Keeper", tags=["shared", "keeperOnly"], collections=["COL_A"], version=1),
            _make_item("DUP1", "Dup1", tags=["shared", "dup1Only"], collections=["COL_B"], version=2),
            _make_item("DUP2", "Dup2", tags=["dup2Only"], collections=["COL_A", "COL_C"], version=3),
            note1, att1, annot1,
        ]
        fake._children = {
            "DUP1": [note1, att1],
            "DUP2": [annot1],
        }
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake, fake))
        return fake

    def test_tags_merged(self, monkeypatch, dummy_ctx):
        """All unique tags from duplicates are consolidated into keeper."""
        fake = self._setup_merge(monkeypatch)

        result = server.merge_duplicates(
            keeper_key="KEEP", duplicate_keys=["DUP1", "DUP2"], confirm=True, ctx=dummy_ctx
        )

        # Find the keeper update that has tags
        keeper_updates = [u for u in fake.update_calls if u.get("key") == "KEEP"]
        assert len(keeper_updates) >= 1
        merged_tags = {t["tag"] for t in keeper_updates[0]["data"]["tags"]}
        assert "shared" in merged_tags
        assert "keeperOnly" in merged_tags
        assert "dup1Only" in merged_tags
        assert "dup2Only" in merged_tags

    def test_children_reparented(self, monkeypatch, dummy_ctx):
        """Child items (notes, attachments, annotations) get parentItem set to keeper."""
        fake = self._setup_merge(monkeypatch)

        server.merge_duplicates(
            keeper_key="KEEP", duplicate_keys=["DUP1", "DUP2"], confirm=True, ctx=dummy_ctx
        )

        # Collect all child reparenting updates
        child_keys = {"NOTE1", "ATT1", "ANNOT1"}
        reparented = [
            u for u in fake.update_calls
            if u.get("key") in child_keys
        ]
        # All children should be reparented
        assert len(reparented) == 3
        for child_update in reparented:
            assert child_update["data"]["parentItem"] == "KEEP"

    def test_duplicates_trashed_not_deleted(self, monkeypatch, dummy_ctx):
        """Duplicates are trashed via direct PATCH (deleted:1), NOT permanently deleted."""
        fake = self._setup_merge(monkeypatch)
        # Ensure delete_item is NOT called (that would permanently delete)
        delete_calls = []
        fake.delete_item = lambda *a, **kw: delete_calls.append(a)

        server.merge_duplicates(
            keeper_key="KEEP", duplicate_keys=["DUP1", "DUP2"], confirm=True, ctx=dummy_ctx
        )

        # delete_item should never be called
        assert delete_calls == []
        # Direct PATCH calls should have been made for each duplicate
        import json
        patch_calls = fake.client.patch_calls
        trashed_contents = [json.loads(c["content"]) for c in patch_calls]
        assert len(trashed_contents) == 2
        assert all(c.get("deleted") == 1 for c in trashed_contents)

    def test_collections_consolidated(self, monkeypatch, dummy_ctx):
        """Keeper is added to every collection the duplicates belonged to."""
        fake = self._setup_merge(monkeypatch)

        server.merge_duplicates(
            keeper_key="KEEP", duplicate_keys=["DUP1", "DUP2"], confirm=True, ctx=dummy_ctx
        )

        # Keeper was already in COL_A, so addto_collection should be called for COL_B and COL_C
        added_colls = {call[0] for call in fake.addto_calls}
        # COL_B comes from DUP1, COL_C comes from DUP2
        assert "COL_B" in added_colls
        assert "COL_C" in added_colls

    def test_keeper_in_duplicate_keys_removed_with_warning(self, monkeypatch, dummy_ctx):
        """If keeper_key appears in duplicate_keys, it is removed (not trashed)."""
        fake = FakeZoteroForDuplicates()
        fake._items = [
            _make_item("KEEP", "Keeper Item", version=1),
            _make_item("DUP1", "Dup Item", version=2),
        ]
        fake._children = {"KEEP": [], "DUP1": []}
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake, fake))

        # Pass keeper_key inside duplicate_keys too
        result = server.merge_duplicates(
            keeper_key="KEEP", duplicate_keys=["KEEP", "DUP1"], confirm=True, ctx=dummy_ctx
        )

        # Keeper should NOT be trashed — check the direct PATCH calls
        import json
        trashed_urls = [c["url"] for c in fake.client.patch_calls]
        assert not any("KEEP" in url for url in trashed_urls)
        # DUP1 should be trashed
        assert any("DUP1" in url for url in trashed_urls)
        # Merge should complete successfully (keeper removal warning goes to ctx.warn)
        assert "merge" in result.lower() or "trashed" in result.lower() or "complete" in result.lower()

    def test_empty_duplicate_list_error(self, monkeypatch, dummy_ctx):
        """Empty duplicate_keys returns an error, no writes performed."""
        fake = FakeZoteroForDuplicates()
        fake._items = [_make_item("KEEP", "Keeper", version=1)]
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake, fake))

        result = server.merge_duplicates(
            keeper_key="KEEP", duplicate_keys=[], confirm=True, ctx=dummy_ctx
        )

        assert "error" in result.lower() or "no duplicate" in result.lower()
        assert fake.update_calls == []

    def test_empty_after_keeper_removal_error(self, monkeypatch, dummy_ctx):
        """If duplicate_keys only contains the keeper, it empties out -> error."""
        fake = FakeZoteroForDuplicates()
        fake._items = [_make_item("KEEP", "Keeper", version=1)]
        fake._children = {"KEEP": []}
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake, fake))

        result = server.merge_duplicates(
            keeper_key="KEEP", duplicate_keys=["KEEP"], confirm=True, ctx=dummy_ctx
        )

        assert "no duplicate" in result.lower() or "empty" in result.lower() or "error" in result.lower()
        assert fake.update_calls == []

    def test_partial_reparent_failure_aborts(self, monkeypatch, dummy_ctx):
        """If a child re-parent fails, stop immediately and don't trash anything."""
        fake = FakeZoteroForDuplicates()
        child_ok = {"key": "CHILD_OK", "version": 10, "data": {
            "itemType": "note", "parentItem": "DUP1",
        }}
        child_fail = {"key": "CHILD_FAIL", "version": 11, "data": {
            "itemType": "attachment", "parentItem": "DUP1",
        }}
        fake._items = [
            _make_item("KEEP", "Keeper", version=1),
            _make_item("DUP1", "Dup", version=2),
            child_ok, child_fail,
        ]
        fake._children = {
            "DUP1": [child_ok, child_fail],
        }
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake, fake))

        # Make update_item fail for the second child
        original_update = fake.update_item
        call_count = [0]

        def failing_update(item, **kwargs):
            call_count[0] += 1
            # Let tag merge and first child succeed; fail on second child
            if item.get("key") == "CHILD_FAIL":
                return _FakeResponse(412, text="Precondition Failed")
            return original_update(item, **kwargs)

        fake.update_item = failing_update

        result = server.merge_duplicates(
            keeper_key="KEEP", duplicate_keys=["DUP1"], confirm=True, ctx=dummy_ctx
        )

        # Should report the failure
        assert "fail" in result.lower() or "error" in result.lower() or "CHILD_FAIL" in result
        # Duplicates should NOT be trashed because re-parenting failed
        assert len(fake.client.patch_calls) == 0

    def test_version_refetch_after_operations(self, monkeypatch, dummy_ctx):
        """Keeper is re-fetched after tag update and collection adds for fresh version."""
        fake = FakeZoteroForDuplicates()
        fake._items = [
            _make_item("KEEP", "Keeper", tags=["t1"], collections=["C1"], version=1),
            _make_item("DUP1", "Dup", tags=["t2"], collections=["C2"], version=2),
        ]
        fake._children = {"DUP1": []}
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake, fake))

        # Track item() fetches to verify re-fetching
        fetch_log = []
        original_item = fake.item

        def tracking_item(key):
            fetch_log.append(key)
            return original_item(key)

        fake.item = tracking_item

        server.merge_duplicates(
            keeper_key="KEEP", duplicate_keys=["DUP1"], confirm=True, ctx=dummy_ctx
        )

        # Keeper should be fetched multiple times: initial + after tag update + after collection add
        keeper_fetches = [k for k in fetch_log if k == "KEEP"]
        assert len(keeper_fetches) >= 2, (
            f"Expected keeper to be re-fetched after updates, got {len(keeper_fetches)} fetches"
        )

    def test_duplicate_keys_as_string_normalized(self, monkeypatch, dummy_ctx):
        """duplicate_keys can be a single string (normalized via _normalize_str_list_input)."""
        fake = FakeZoteroForDuplicates()
        fake._items = [
            _make_item("KEEP", "Keeper", version=1),
            _make_item("DUP1", "Dup", version=2),
        ]
        fake._children = {"DUP1": []}
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake, fake))

        # Pass a single string instead of a list
        result = server.merge_duplicates(
            keeper_key="KEEP", duplicate_keys="DUP1", confirm=True, ctx=dummy_ctx
        )

        # Should succeed — DUP1 trashed via direct PATCH
        assert any("DUP1" in c["url"] for c in fake.client.patch_calls)
