"""Regression tests for bugs found during integration testing (2026-03-21).

Each test prevents a specific bug from reappearing.
"""

import json

import pytest

from conftest import DummyContext, FakeZotero, _FakeResponse
from zotero_mcp import server


# ---------------------------------------------------------------------------
# Bug 1: manage_collections passed [item_dict] (list) instead of item_dict
# to addto_collection, causing "list indices must be integers or slices, not str"
# ---------------------------------------------------------------------------

class TestManageCollectionsPayloadShape:
    """addto_collection receives a dict (not a list)."""

    def test_addto_receives_dict_not_list(self, monkeypatch):
        received = []

        class FakeZotManage(FakeZotero):
            def item(self, key):
                return {"key": key, "version": 1, "data": {"collections": []}}

            def addto_collection(self, coll_key, payload, **kw):
                received.append(payload)
                return _FakeResponse(204)

        fake = FakeZotManage()
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake, fake))

        ctx = DummyContext()
        server.manage_collections(
            item_keys=["ITEM01"], add_to=["COL01"], ctx=ctx
        )

        assert len(received) == 1
        # Must be a dict, NOT a list wrapping a dict
        assert isinstance(received[0], dict), (
            f"addto_collection received {type(received[0])}, expected dict"
        )
        assert received[0]["key"] == "ITEM01"


# ---------------------------------------------------------------------------
# Bug 2: merge_duplicates used update_item({"deleted": True}) which pyzotero
# rejects. Now uses direct PATCH with {"deleted": 1} for safe trashing.
# ---------------------------------------------------------------------------

class _FakeHttpClient:
    def __init__(self):
        self.patch_calls = []

    def patch(self, url="", headers=None, content=""):
        self.patch_calls.append({"url": url, "headers": headers, "content": content})
        return _FakeResponse(204)


class TestMergeTrashMethod:
    """Merge uses direct PATCH (not update_item) for trashing."""

    def _setup(self, monkeypatch):
        class FakeZotMerge(FakeZotero):
            def __init__(self):
                super().__init__()
                self.update_calls = []
                self.client = _FakeHttpClient()
                self.endpoint = "https://api.zotero.org"
                self.library_type = "users"
                self.library_id = "12345"

            def item(self, key):
                items = {
                    "KEEP": {"key": "KEEP", "version": 1, "data": {
                        "title": "Keeper", "itemType": "journalArticle",
                        "tags": [{"tag": "t1"}], "collections": [],
                    }},
                    "DUP1": {"key": "DUP1", "version": 2, "data": {
                        "title": "Dup", "itemType": "journalArticle",
                        "tags": [{"tag": "t2"}], "collections": [],
                    }},
                }
                return items[key]

            def children(self, key, **kw):
                return []

            def update_item(self, item, **kw):
                self.update_calls.append(item)
                item["version"] = item.get("version", 0) + 100
                return _FakeResponse(204)

            def addto_collection(self, key, payload, **kw):
                return _FakeResponse(204)

        fake = FakeZotMerge()
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake, fake))
        return fake

    def test_trash_uses_direct_patch_not_update_item(self, monkeypatch):
        """Trashing must use client.patch with deleted:1, NOT update_item."""
        fake = self._setup(monkeypatch)
        ctx = DummyContext()

        server.merge_duplicates(
            keeper_key="KEEP", duplicate_keys=["DUP1"], confirm=True, ctx=ctx
        )

        # update_item should NOT have been called with any "deleted" field
        for call in fake.update_calls:
            data = call.get("data", {})
            assert "deleted" not in data, (
                "update_item was called with 'deleted' field — this will fail "
                "with 'Invalid keys present in item'. Must use direct PATCH."
            )

        # Direct PATCH should have been called for DUP1
        assert len(fake.client.patch_calls) >= 1
        patch_contents = [json.loads(c["content"]) for c in fake.client.patch_calls]
        assert any(c.get("deleted") == 1 for c in patch_contents)


# ---------------------------------------------------------------------------
# Bug 3: find_duplicates used zot.everything() which caused
# "cannot pickle '_thread.RLock' object" in MCP contexts.
# Now uses manual pagination.
# ---------------------------------------------------------------------------

class TestFindDuplicatesNoPicle:
    """find_duplicates does NOT call everything()."""

    def test_everything_not_called(self, monkeypatch):
        class FakeZotNoPickle(FakeZotero):
            def everything(self, *args, **kwargs):
                raise RuntimeError(
                    "everything() should not be called — causes RLock pickle error"
                )

            def items(self, **kwargs):
                return [
                    {"key": "A", "data": {"title": "Paper A", "itemType": "journalArticle", "DOI": ""}},
                    {"key": "B", "data": {"title": "Paper B", "itemType": "journalArticle", "DOI": ""}},
                ]

        fake = FakeZotNoPickle()
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)

        ctx = DummyContext()
        # Should complete without hitting everything()
        result = server.find_duplicates(method="both", ctx=ctx)
        assert "Error" not in result or "No duplicates" in result


# ---------------------------------------------------------------------------
# Bug 4: PDF outline tried to import _get_storage_dir as standalone function
# and used a broken local mode path. Now uses zot.dump() for all modes.
# ---------------------------------------------------------------------------

class TestPdfOutlineDownloadMethod:
    """PDF outline always uses zot.dump(), not direct file path access."""

    def test_dump_called_not_direct_path(self, monkeypatch):
        import sys
        import types

        dump_called = []

        class FakeZotDump(FakeZotero):
            def children(self, key, **kw):
                return [{
                    "key": "ATT01",
                    "data": {
                        "itemType": "attachment",
                        "contentType": "application/pdf",
                        "filename": "paper.pdf",
                        "parentItem": key,
                    },
                }]

            def dump(self, key, filename=None, path=None):
                dump_called.append({"key": key, "filename": filename, "path": path})
                # Create a dummy file
                import os
                if path and filename:
                    with open(os.path.join(path, filename), "wb") as f:
                        f.write(b"%PDF-1.4 fake")

        fake = FakeZotDump()
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: True)

        # Mock fitz
        class FakeDoc:
            def get_toc(self):
                return [[1, "Intro", 1]]
            def close(self):
                pass

        fake_fitz = types.ModuleType("fitz")
        fake_fitz.open = lambda *a, **kw: FakeDoc()
        monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

        ctx = DummyContext()
        result = server.get_pdf_outline(item_key="PARENT01", ctx=ctx)

        assert len(dump_called) == 1, "dump() should be called even in local mode"
        assert "Intro" in result


# ---------------------------------------------------------------------------
# Bug 5: batch_update_tags had no tag filter parameter,
# only text search via 'q' parameter.
# ---------------------------------------------------------------------------

class TestBatchUpdateTagsFilter:
    """batch_update_tags accepts a 'tag' parameter for filtering."""

    def test_tag_parameter_exists_in_signature(self):
        import inspect
        sig = inspect.signature(server.batch_update_tags)
        assert "tag" in sig.parameters, (
            "batch_update_tags must have a 'tag' parameter for tag-based filtering"
        )
