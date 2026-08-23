"""Tests for Features 7-8: find_duplicates and merge_duplicates."""

import json
import re

import pytest

from conftest import DummyContext, FakeZotero, _FakeResponse
from zotero_mcp import server


# ---------------------------------------------------------------------------
# Helpers: item factory and extended FakeZotero for duplicates
# ---------------------------------------------------------------------------

def _make_item(key, title, doi=None, collections=None, tags=None, version=1,
               item_type="journalArticle", abstract=None, date_added=None):
    """Build a minimal Zotero item dict.

    item_type/abstract/date_added exist for the auto-merge keeper heuristic,
    which ranks on child count, then has-an-abstract, then oldest dateAdded.
    """
    data = {
        "key": key,
        "title": title,
        "itemType": item_type,
        "creators": [{"firstName": "A", "lastName": "Author", "creatorType": "author"}],
        "date": "2024",
        "DOI": doi or "",
        "tags": [{"tag": t} for t in (tags or [])],
        "collections": collections or [],
        "deleted": False,
    }
    if abstract is not None:
        data["abstractNote"] = abstract
    if date_added is not None:
        data["dateAdded"] = date_added
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

    @staticmethod
    def _page(rows, kwargs):
        start = kwargs.get("start") or 0
        limit = kwargs.get("limit")
        rows = rows[start:]
        return rows[:limit] if limit else rows

    def items(self, **kwargs):
        """Honour start/limit, so the scan pages the way the real API does.

        The base fake ignores them and returns everything on every call, which
        makes it impossible to test paging past one page (#394).
        """
        return self._page(self._items, kwargs)

    def collection_items(self, key, **kwargs):
        rows = [it for it in self._items
                if key in it.get("data", {}).get("collections", [])]
        return self._page(rows, kwargs)


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


class PagedChildrenFake(FakeZoteroForDuplicates):
    """children() honors start/limit like the real Zotero API.

    Without an explicit ``limit`` the API returns its default page of 25
    results — exactly the behavior that truncates unpaginated call sites.
    """

    def children(self, item_key, start=0, limit=25, **kwargs):
        kids = self._children.get(item_key, [])
        return kids[int(start):int(start) + int(limit)]


def _make_attachment(key, parent, filename, version=1):
    return {"key": key, "version": version, "data": {
        "key": key,
        "itemType": "attachment",
        "parentItem": parent,
        "contentType": "application/pdf",
        "filename": filename,
        "md5": f"md5-{filename}",
        "url": "",
    }}


class TestMergeDuplicatesPagination:
    """merge_duplicates must see ALL children, not the API's first page of 25."""

    def test_all_duplicate_children_reparented_past_first_api_page(self, monkeypatch, dummy_ctx):
        """A duplicate with >100 children gets every child re-parented (not
        just 25 — the rest would silently go to Trash with the duplicate)."""
        fake = PagedChildrenFake()
        n = 130  # > 100 so the fix's page_size=100 must also paginate
        children = [
            {"key": f"N{i:04d}", "version": 1, "data": {
                "itemType": "note", "parentItem": "DUP1", "note": f"note {i}",
            }}
            for i in range(n)
        ]
        fake._items = [
            _make_item("KEEP", "Keeper"),
            _make_item("DUP1", "Dup"),
            *children,
        ]
        fake._children = {"KEEP": [], "DUP1": children}
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake, fake))

        result = server.merge_duplicates(
            keeper_key="KEEP", duplicate_keys=["DUP1"], confirm=True, ctx=dummy_ctx
        )

        reparented = {
            u["key"] for u in fake.update_calls
            if u.get("key", "").startswith("N") and u["data"].get("parentItem") == "KEEP"
        }
        assert len(reparented) == n
        assert f"Children re-parented: {n}" in result

    def test_keeper_dedupe_signatures_scan_past_first_api_page(self, monkeypatch, dummy_ctx):
        """Attachment dedupe must consider keeper children beyond the first
        API page — otherwise a duplicate of keeper attachment #26+ gets
        copied onto the keeper instead of skipped."""
        fake = PagedChildrenFake()
        keeper_children = [
            _make_attachment(f"KA{i:04d}", "KEEP", f"paper-{i}.pdf") for i in range(130)
        ]
        # Same signature (contentType, filename, md5, url) as the keeper's
        # last attachment — far past the first API page.
        dup_att = _make_attachment("DUPATT01", "DUP1", "paper-129.pdf")
        fake._items = [
            _make_item("KEEP", "Keeper"),
            _make_item("DUP1", "Dup"),
            *keeper_children,
            dup_att,
        ]
        fake._children = {"KEEP": keeper_children, "DUP1": [dup_att]}
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake, fake))

        result = server.merge_duplicates(
            keeper_key="KEEP", duplicate_keys=["DUP1"], confirm=True, ctx=dummy_ctx
        )

        reparented_keys = {u.get("key") for u in fake.update_calls}
        assert "DUPATT01" not in reparented_keys, (
            "duplicate attachment should be skipped, not re-parented onto keeper"
        )
        assert "1 duplicate attachments skipped" in result


# ---------------------------------------------------------------------------
# #394: paging past the 100-group ceiling, and honest counts
# ---------------------------------------------------------------------------

def _doi_groups(n, per_group=2, prefix="G"):
    """n duplicate groups, each `per_group` items sharing one DOI."""
    items = []
    for g in range(n):
        for m in range(per_group):
            items.append(_make_item(
                f"{prefix}{g:03d}{m}", f"Paper {g}", doi=f"10.1000/paper{g:03d}"
            ))
    return items


def _dup_fake(monkeypatch, items):
    fake = FakeZoteroForDuplicates()
    fake._items = items
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    return fake


class TestFindDuplicatesPaging:
    """zotero_find_duplicates must be able to reach every group it counted."""

    def test_limit_above_100_is_not_clamped(self, monkeypatch, dummy_ctx):
        """The #394 bug: limit>100 was clamped, so groups 101+ were unreachable."""
        _dup_fake(monkeypatch, _doi_groups(120))

        result = server.find_duplicates(method="doi", limit=150, ctx=dummy_ctx)

        assert result.count("## Group:") == 120
        assert "Found 120 duplicate groups" in result
        # Nothing was withheld, so nothing should advertise a next page.
        assert "more group(s) not shown" not in result

    def test_offset_pages_through_every_group(self, monkeypatch, dummy_ctx):
        """Walking offset by limit yields each group exactly once."""
        _dup_fake(monkeypatch, _doi_groups(7))

        seen = []
        for offset in range(0, 7, 3):
            page = server.find_duplicates(
                method="doi", limit=3, offset=offset, ctx=dummy_ctx
            )
            seen += re.findall(r"## Group: (doi:\S+)", page)

        assert len(seen) == 7
        assert len(set(seen)) == 7

    def test_pages_are_stable_across_calls(self, monkeypatch, dummy_ctx):
        """The same offset returns the same groups, or paging would skip some."""
        _dup_fake(monkeypatch, _doi_groups(9))

        first = server.find_duplicates(method="doi", limit=4, offset=4, ctx=dummy_ctx)
        second = server.find_duplicates(method="doi", limit=4, offset=4, ctx=dummy_ctx)

        assert re.findall(r"## Group: (doi:\S+)", first) == \
               re.findall(r"## Group: (doi:\S+)", second)

    def test_reports_shown_out_of_total(self, monkeypatch, dummy_ctx):
        """A truncated page says which groups it is showing out of how many."""
        _dup_fake(monkeypatch, _doi_groups(10))

        result = server.find_duplicates(method="doi", limit=4, ctx=dummy_ctx)

        assert "Found 10 duplicate groups" in result
        assert "Showing groups 1-4 of 10" in result
        assert result.count("## Group:") == 4

    def test_truncated_page_names_the_next_offset(self, monkeypatch, dummy_ctx):
        """The footer is actionable: it says how to reach the withheld groups."""
        _dup_fake(monkeypatch, _doi_groups(10))

        result = server.find_duplicates(method="doi", limit=4, ctx=dummy_ctx)

        assert "6 more group(s) not shown" in result
        assert "offset=4" in result

    def test_offset_past_the_end_reports_the_total(self, monkeypatch, dummy_ctx):
        """An empty page is distinguishable from an empty library."""
        _dup_fake(monkeypatch, _doi_groups(3))

        result = server.find_duplicates(
            method="doi", limit=5, offset=99, ctx=dummy_ctx
        )

        assert "Found 3 duplicate groups" in result
        assert "No groups at offset 99" in result
        assert "## Group:" not in result

    def test_header_breaks_down_doi_versus_title_groups(self, monkeypatch, dummy_ctx):
        """Sorted order puts doi: before title:, so say how many of each exist.

        Otherwise a default-limit call on a library whose DOI groups fill the
        page looks like it has no title duplicates at all (#394).
        """
        # Distinct titles inside each DOI pair, so the DOI groups do not also
        # register as title groups and muddy the count.
        items = [
            _make_item("D1A", "Alpha One", doi="10.1000/alpha"),
            _make_item("D1B", "Alpha Two", doi="10.1000/alpha"),
            _make_item("D2A", "Beta One", doi="10.1000/beta"),
            _make_item("D2B", "Beta Two", doi="10.1000/beta"),
            _make_item("T1", "Shared Title No Doi"),
            _make_item("T2", "Shared Title No Doi"),
        ]
        _dup_fake(monkeypatch, items)

        result = server.find_duplicates(method="both", limit=1, ctx=dummy_ctx)

        assert "Found 3 duplicate groups (2 by DOI, 1 by title)" in result

    def test_string_offset_accepted(self, monkeypatch, dummy_ctx):
        """MCP clients hand numbers over as strings."""
        _dup_fake(monkeypatch, _doi_groups(5))

        result = server.find_duplicates(
            method="doi", limit="2", offset="2", ctx=dummy_ctx
        )

        assert "Showing groups 3-4 of 5" in result

    def test_negative_offset_clamps_to_zero(self, monkeypatch, dummy_ctx):
        _dup_fake(monkeypatch, _doi_groups(4))

        result = server.find_duplicates(
            method="doi", limit=2, offset=-5, ctx=dummy_ctx
        )

        assert "Showing groups 1-2 of 4" in result


# ---------------------------------------------------------------------------
# #395: auto / batch merge
# ---------------------------------------------------------------------------

def _auto_fake(monkeypatch, items, children=None):
    """A fake wired for both read and write, as the auto path needs both."""
    fake = FakeZoteroForDuplicates()
    fake._items = list(items)
    fake._children = children or {}
    # _execute_merge re-fetches each child by key, so children have to be
    # findable as items too.
    for kids in fake._children.values():
        for kid in kids:
            if kid not in fake._items:
                fake._items.append(kid)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake, fake))
    return fake


def _child(key, parent, item_type="note", version=10):
    return {"key": key, "version": version,
            "data": {"itemType": item_type, "parentItem": parent}}


def _token_from_plan(plan_text):
    m = re.search(r"plan_token='([0-9a-f]+)'", plan_text)
    assert m, f"no plan_token in plan output:\n{plan_text}"
    return m.group(1)


class TestAutoMergeGating:
    """Auto mode is a two-call operation on purpose."""

    def _pair(self, monkeypatch):
        return _auto_fake(monkeypatch, [
            _make_item("K1", "Same Paper", doi="10.1/x", date_added="2020-01-01"),
            _make_item("K2", "Same Paper", doi="10.1/x", date_added="2024-01-01"),
        ])

    def test_plan_writes_nothing(self, monkeypatch, dummy_ctx):
        fake = self._pair(monkeypatch)

        result = server.merge_duplicates(auto=True, ctx=dummy_ctx)

        assert "Auto-merge plan" in result
        assert "nothing has been changed" in result.lower()
        assert fake.update_calls == []
        assert fake.addto_calls == []
        assert fake.client.patch_calls == []

    def test_plan_names_keeper_and_what_would_be_trashed(self, monkeypatch, dummy_ctx):
        self._pair(monkeypatch)

        result = server.merge_duplicates(auto=True, ctx=dummy_ctx)

        assert "**KEEP** `K1`" in result
        assert "trash `K2`" in result
        assert "Would trash 1 item(s)" in result

    def test_confirm_without_plan_token_is_refused(self, monkeypatch, dummy_ctx):
        """The whole point: confirm=True alone must not be able to trash."""
        fake = self._pair(monkeypatch)

        result = server.merge_duplicates(auto=True, confirm=True, ctx=dummy_ctx)

        assert "will not execute on confirm=True alone" in result
        assert fake.update_calls == []
        assert fake.client.patch_calls == []

    def test_stale_plan_token_is_refused(self, monkeypatch, dummy_ctx):
        fake = self._pair(monkeypatch)

        result = server.merge_duplicates(
            auto=True, confirm=True, plan_token="deadbeef1234", ctx=dummy_ctx
        )

        assert "plan_token mismatch" in result
        assert fake.update_calls == []
        assert fake.client.patch_calls == []

    def test_token_from_a_changed_library_is_refused(self, monkeypatch, dummy_ctx):
        """A plan confirmed after the library moved on must not be applied."""
        fake = self._pair(monkeypatch)
        token = _token_from_plan(server.merge_duplicates(auto=True, ctx=dummy_ctx))

        # A third copy arrives before the confirmation lands.
        fake._items.append(
            _make_item("K3", "Same Paper", doi="10.1/x", date_added="2025-01-01")
        )

        result = server.merge_duplicates(
            auto=True, confirm=True, plan_token=token, ctx=dummy_ctx
        )

        assert "plan_token mismatch" in result
        assert fake.client.patch_calls == []

    def test_plan_then_confirm_executes(self, monkeypatch, dummy_ctx):
        fake = self._pair(monkeypatch)
        token = _token_from_plan(server.merge_duplicates(auto=True, ctx=dummy_ctx))

        result = server.merge_duplicates(
            auto=True, confirm=True, plan_token=token, ctx=dummy_ctx
        )

        assert "Auto-merge complete" in result
        assert "Groups merged: **1**" in result
        trashed = [json.loads(c["content"]) for c in fake.client.patch_calls]
        assert trashed == [{"deleted": 1}]
        assert any("K2" in c["url"] for c in fake.client.patch_calls)
        assert not any("K1" in c["url"] for c in fake.client.patch_calls)

    def test_auto_rejects_explicit_keys(self, monkeypatch, dummy_ctx):
        fake = self._pair(monkeypatch)

        result = server.merge_duplicates(
            auto=True, keeper_key="K1", duplicate_keys=["K2"], ctx=dummy_ctx
        )

        assert "do not" in result.lower()
        assert fake.update_calls == []

    def test_manual_path_still_requires_a_keeper(self, monkeypatch, dummy_ctx):
        fake = self._pair(monkeypatch)

        result = server.merge_duplicates(duplicate_keys=["K2"], ctx=dummy_ctx)

        assert "keeper_key is required" in result
        assert fake.update_calls == []


class TestAutoMergeKeeperHeuristic:
    """most children -> has-an-abstract -> oldest dateAdded -> key."""

    def test_most_children_wins(self, monkeypatch, dummy_ctx):
        _auto_fake(
            monkeypatch,
            [
                # The one with children is NEWER and has no abstract, so only
                # the child count can be what picks it.
                _make_item("FEW", "Paper", doi="10.1/a", abstract="has one",
                           date_added="2019-01-01"),
                _make_item("MANY", "Paper", doi="10.1/a", date_added="2025-01-01"),
            ],
            children={"MANY": [_child("N1", "MANY"), _child("N2", "MANY")]},
        )

        result = server.merge_duplicates(auto=True, ctx=dummy_ctx)

        assert "**KEEP** `MANY`" in result
        assert "2 child item(s)" in result

    def test_abstract_breaks_a_child_count_tie(self, monkeypatch, dummy_ctx):
        _auto_fake(monkeypatch, [
            _make_item("NOABS", "Paper", doi="10.1/b", date_added="2019-01-01"),
            _make_item("ABS", "Paper", doi="10.1/b", abstract="An abstract",
                       date_added="2025-01-01"),
        ])

        result = server.merge_duplicates(auto=True, ctx=dummy_ctx)

        assert "**KEEP** `ABS`" in result
        assert "has abstract" in result

    def test_oldest_date_added_breaks_the_remaining_tie(self, monkeypatch, dummy_ctx):
        _auto_fake(monkeypatch, [
            _make_item("NEW", "Paper", doi="10.1/c", abstract="a",
                       date_added="2025-06-01"),
            _make_item("OLD", "Paper", doi="10.1/c", abstract="a",
                       date_added="2018-02-03"),
        ])

        result = server.merge_duplicates(auto=True, ctx=dummy_ctx)

        assert "**KEEP** `OLD`" in result
        assert "added 2018-02-03" in result

    def test_choice_is_deterministic_when_nothing_discriminates(self, monkeypatch, dummy_ctx):
        """Identical members still have to produce a stable plan token."""
        _auto_fake(monkeypatch, [
            _make_item("BBB", "Paper", doi="10.1/d"),
            _make_item("AAA", "Paper", doi="10.1/d"),
        ])

        first = server.merge_duplicates(auto=True, ctx=dummy_ctx)
        second = server.merge_duplicates(auto=True, ctx=dummy_ctx)

        assert "**KEEP** `AAA`" in first
        assert _token_from_plan(first) == _token_from_plan(second)


class TestAutoMergeSafety:
    """What auto mode declines to touch matters more than what it merges."""

    def test_default_method_is_doi_only(self, monkeypatch, dummy_ctx):
        """Title-only duplicates must not be merged unless title is asked for."""
        fake = _auto_fake(monkeypatch, [
            _make_item("S1", "Identical Title"),
            _make_item("S2", "Identical Title"),
        ])

        result = server.merge_duplicates(auto=True, ctx=dummy_ctx)

        assert "No duplicates found" in result
        assert fake.client.patch_calls == []

    def test_title_method_is_available_on_request(self, monkeypatch, dummy_ctx):
        _auto_fake(monkeypatch, [
            _make_item("S1", "Identical Title"),
            _make_item("S2", "Identical Title"),
        ])

        result = server.merge_duplicates(auto=True, method="title", ctx=dummy_ctx)

        assert "1 group(s) qualify" in result

    def test_mixed_item_types_are_skipped(self, monkeypatch, dummy_ctx):
        """A book and a journal article sharing a DOI are not one record."""
        _auto_fake(monkeypatch, [
            _make_item("M1", "Thing", doi="10.1/m", item_type="journalArticle"),
            _make_item("M2", "Thing", doi="10.1/m", item_type="book"),
        ])

        result = server.merge_duplicates(auto=True, ctx=dummy_ctx)

        assert "0 group(s) qualify" in result
        assert "mixed item types" in result
        assert "book" in result

    def test_conflicting_dois_are_skipped(self, monkeypatch, dummy_ctx):
        """The title false-positive class from #395: two different books each
        with a 'List of Contributors'."""
        _auto_fake(monkeypatch, [
            _make_item("C1", "List of Contributors", doi="10.1/book-one"),
            _make_item("C2", "List of Contributors", doi="10.1/book-two"),
        ])

        result = server.merge_duplicates(auto=True, method="title", ctx=dummy_ctx)

        assert "0 group(s) qualify" in result
        assert "carry different DOIs" in result

    def test_overlapping_groups_are_skipped_not_double_merged(self, monkeypatch, dummy_ctx):
        """With method='both' one pair matches on DOI and on title.

        The second group's items are already trashed by the time it comes up,
        so merging it again would operate on items in the Trash.
        """
        _auto_fake(monkeypatch, [
            _make_item("O1", "Same Title", doi="10.1/o"),
            _make_item("O2", "Same Title", doi="10.1/o"),
        ])

        result = server.merge_duplicates(auto=True, method="both", ctx=dummy_ctx)

        assert "1 group(s) qualify" in result
        assert "overlaps a group already merged" in result

    def test_nothing_qualifies_means_nothing_to_confirm(self, monkeypatch, dummy_ctx):
        _auto_fake(monkeypatch, [
            _make_item("M1", "Thing", doi="10.1/m", item_type="journalArticle"),
            _make_item("M2", "Thing", doi="10.1/m", item_type="book"),
        ])

        result = server.merge_duplicates(auto=True, ctx=dummy_ctx)

        assert "nothing to confirm" in result.lower()
        assert "plan_token" not in result

    def test_children_are_reparented_before_trashing(self, monkeypatch, dummy_ctx):
        """The #387 shape: a child left on an item that then gets trashed."""
        fake = _auto_fake(
            monkeypatch,
            [
                _make_item("KEEP", "Paper", doi="10.1/p", date_added="2019-01-01"),
                _make_item("GONE", "Paper", doi="10.1/p", date_added="2024-01-01"),
            ],
            children={"GONE": [_child("KID", "GONE")]},
        )
        # KEEP is older; GONE has the child, so give KEEP two to keep it winning.
        fake._children["KEEP"] = [_child("K1", "KEEP"), _child("K2", "KEEP")]
        for kid in fake._children["KEEP"]:
            fake._items.append(kid)

        token = _token_from_plan(server.merge_duplicates(auto=True, ctx=dummy_ctx))
        result = server.merge_duplicates(
            auto=True, confirm=True, plan_token=token, ctx=dummy_ctx
        )

        assert "Auto-merge complete" in result
        reparented = [u for u in fake.update_calls if u.get("key") == "KID"]
        assert len(reparented) == 1
        assert reparented[0]["data"]["parentItem"] == "KEEP"
        assert any("GONE" in c["url"] for c in fake.client.patch_calls)

    def test_max_groups_caps_one_call(self, monkeypatch, dummy_ctx):
        _auto_fake(monkeypatch, _doi_groups(5, prefix="Q"))

        result = server.merge_duplicates(auto=True, max_groups=2, ctx=dummy_ctx)

        assert "2 group(s) qualify" in result
        assert "beyond this call's 2-group ceiling" in result
        assert "capped at 2 groups" in result.lower()

    def test_long_skip_list_is_summarised(self, monkeypatch, dummy_ctx):
        """A library with many declined groups must not bury the plan."""
        items = []
        for g in range(30):
            # Same DOI but different item types — declined, 30 times over.
            items.append(_make_item(f"X{g:02d}A", f"Thing {g}", doi=f"10.1/x{g:02d}",
                                    item_type="journalArticle"))
            items.append(_make_item(f"X{g:02d}B", f"Thing {g}", doi=f"10.1/x{g:02d}",
                                    item_type="book"))
        _auto_fake(monkeypatch, items)

        result = server.merge_duplicates(auto=True, ctx=dummy_ctx)

        assert "30 skipped" in result
        assert result.count("mixed item types") == 20
        assert "... and 10 more skipped group(s)" in result
