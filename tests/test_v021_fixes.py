"""Tests for v0.2.1 fixes: pagination, grandparent resolution,
merge attachment dedup, linked-URL removal, and no-PDF messaging."""

from unittest.mock import MagicMock, patch

import pytest

from conftest import DummyContext, FakeZotero
from zotero_mcp.tools import _helpers
from zotero_mcp.tools.annotations import (
    _batch_resolve_grandparent_titles,
    _batch_resolve_parent_titles,
)


# -------------------------------------------------------------------------
# Fix 1 — Pagination helper
# -------------------------------------------------------------------------


class TestPaginate:
    """Unit tests for _helpers._paginate."""

    def test_paginate_multiple_pages(self):
        """150 items across 2 batches (100 + 50) are all returned."""
        batch1 = [{"key": f"A{i}"} for i in range(100)]
        batch2 = [{"key": f"B{i}"} for i in range(50)]
        mock_method = MagicMock(side_effect=[batch1, batch2])

        result = _helpers._paginate(mock_method)

        assert len(result) == 150
        assert mock_method.call_count == 2
        # First call: start=0, limit=100
        assert mock_method.call_args_list[0][1]["start"] == 0
        assert mock_method.call_args_list[0][1]["limit"] == 100
        # Second call: start=100
        assert mock_method.call_args_list[1][1]["start"] == 100

    def test_paginate_with_max_items(self):
        """max_items=30 caps the returned results even when batch has 100."""
        batch = [{"key": f"X{i}"} for i in range(100)]
        mock_method = MagicMock(return_value=batch)

        result = _helpers._paginate(mock_method, max_items=30)

        assert len(result) == 30

    def test_paginate_empty(self):
        """Empty first batch returns empty list."""
        mock_method = MagicMock(return_value=[])

        result = _helpers._paginate(mock_method)

        assert result == []
        assert mock_method.call_count == 1

    def test_paginate_single_batch(self):
        """50 items (< page size) returned in one call; method called once."""
        batch = [{"key": f"S{i}"} for i in range(50)]
        mock_method = MagicMock(return_value=batch)

        result = _helpers._paginate(mock_method)

        assert len(result) == 50
        assert mock_method.call_count == 1


# -------------------------------------------------------------------------
# Fix 2+5 — Grandparent title resolution
# -------------------------------------------------------------------------


class TestGrandparentResolution:
    """Unit tests for _batch_resolve_grandparent_titles."""

    def _make_zot(self, items_lookup):
        """Build a mock zot whose .items(itemKey=...) returns from a dict."""
        zot = MagicMock()

        def items_side_effect(**kwargs):
            item_key_str = kwargs.get("itemKey", "")
            keys = item_key_str.split(",")
            return [items_lookup[k] for k in keys if k in items_lookup]

        zot.items.side_effect = items_side_effect
        return zot

    def test_grandparent_resolution_paper_title(self):
        """Attachment parent -> paper grandparent returns paper title."""
        ATTACH_KEY = "ATT001"
        PAPER_KEY = "PAPER001"

        items_lookup = {
            ATTACH_KEY: {
                "key": ATTACH_KEY,
                "data": {
                    "itemType": "attachment",
                    "title": "Full Text PDF",
                    "parentItem": PAPER_KEY,
                },
            },
            PAPER_KEY: {
                "key": PAPER_KEY,
                "data": {
                    "itemType": "journalArticle",
                    "title": "Real Paper Title",
                },
            },
        }
        zot = self._make_zot(items_lookup)
        ctx = DummyContext()

        result = _batch_resolve_grandparent_titles(zot, {ATTACH_KEY}, ctx)

        assert result[ATTACH_KEY] == "Real Paper Title"

    def test_grandparent_resolution_orphaned_attachment(self):
        """Attachment with no parentItem falls back to its own title."""
        ATTACH_KEY = "ATT_ORPHAN"

        items_lookup = {
            ATTACH_KEY: {
                "key": ATTACH_KEY,
                "data": {
                    "itemType": "attachment",
                    "title": "Snapshot",
                    # No parentItem
                },
            },
        }
        zot = self._make_zot(items_lookup)
        ctx = DummyContext()

        result = _batch_resolve_grandparent_titles(zot, {ATTACH_KEY}, ctx)

        assert result[ATTACH_KEY] == "Snapshot"

    def test_grandparent_resolution_non_attachment_parent(self):
        """Non-attachment parent (journalArticle) returns its own title."""
        PARENT_KEY = "JA001"

        items_lookup = {
            PARENT_KEY: {
                "key": PARENT_KEY,
                "data": {
                    "itemType": "journalArticle",
                    "title": "My Article Title",
                },
            },
        }
        zot = self._make_zot(items_lookup)
        ctx = DummyContext()

        result = _batch_resolve_grandparent_titles(zot, {PARENT_KEY}, ctx)

        # Not an attachment -> no two-hop -> falls back to own title
        assert result[PARENT_KEY] == "My Article Title"


# -------------------------------------------------------------------------
# Fix 3 — Merge attachment dedup
# -------------------------------------------------------------------------


class TestMergeAttachmentDedup:
    """Unit tests for merge_duplicates attachment deduplication."""

    def _setup_merge(self, keeper_children, dup_children):
        """Set up mocked clients and monkeypatch for merge_duplicates."""
        keeper = {
            "key": "KEEPER",
            "version": 1,
            "data": {
                "title": "Keeper Paper",
                "tags": [],
                "collections": [],
                "itemType": "journalArticle",
            },
        }
        dup_item = {
            "key": "DUP1",
            "version": 1,
            "data": {
                "title": "Duplicate Paper",
                "tags": [],
                "collections": [],
                "itemType": "journalArticle",
            },
        }

        write_zot = MagicMock()
        write_zot.item.side_effect = lambda k: (
            keeper if k == "KEEPER" else
            dup_item if k == "DUP1" else
            # For child re-fetch during execute, return the child itself
            next((c for c in keeper_children + dup_children if c.get("key") == k), {"key": k, "version": 1, "data": {}})
        )
        write_zot.children.side_effect = lambda k: (
            keeper_children if k == "KEEPER" else
            dup_children if k == "DUP1" else []
        )
        write_zot.update_item.return_value = MagicMock(status_code=204, text="")
        write_zot.addto_collection.return_value = MagicMock(status_code=204, text="")
        # trash_items needs to succeed
        write_zot.trash_items = MagicMock(return_value=MagicMock(status_code=204, text=""))

        return write_zot

    @patch("zotero_mcp.tools.write._helpers._get_write_client")
    def test_merge_skips_duplicate_pdf(self, mock_get_client, dummy_ctx):
        """Identical attachment on keeper and duplicate is skipped."""
        keeper_att = {
            "key": "K_ATT",
            "version": 1,
            "data": {
                "itemType": "attachment",
                "contentType": "application/pdf",
                "filename": "paper.pdf",
                "md5": "abc123",
                "url": "",
                "parentItem": "KEEPER",
            },
        }
        dup_att = {
            "key": "D_ATT",
            "version": 1,
            "data": {
                "itemType": "attachment",
                "contentType": "application/pdf",
                "filename": "paper.pdf",
                "md5": "abc123",
                "url": "",
                "parentItem": "DUP1",
            },
        }

        write_zot = self._setup_merge([keeper_att], [dup_att])
        mock_get_client.return_value = (write_zot, write_zot)

        from zotero_mcp.tools.write import merge_duplicates

        result = merge_duplicates("KEEPER", ["DUP1"], confirm=True, ctx=dummy_ctx)

        # The duplicate attachment should NOT have been re-parented
        # update_item should not be called for the dup attachment
        reparent_calls = [
            c for c in write_zot.update_item.call_args_list
            if c[0][0].get("key") == "D_ATT"
        ]
        assert len(reparent_calls) == 0
        assert "D_ATT" not in result or "skipped" in result.lower() or "Merged" in result

    @patch("zotero_mcp.tools.write._helpers._get_write_client")
    def test_merge_keeps_different_pdf(self, mock_get_client, dummy_ctx):
        """Different PDFs on keeper and duplicate are both kept."""
        keeper_att = {
            "key": "K_ATT",
            "version": 1,
            "data": {
                "itemType": "attachment",
                "contentType": "application/pdf",
                "filename": "A.pdf",
                "md5": "aaa111",
                "url": "",
                "parentItem": "KEEPER",
            },
        }
        dup_att = {
            "key": "D_ATT",
            "version": 1,
            "data": {
                "itemType": "attachment",
                "contentType": "application/pdf",
                "filename": "B.pdf",
                "md5": "bbb222",
                "url": "",
                "parentItem": "DUP1",
            },
        }

        write_zot = self._setup_merge([keeper_att], [dup_att])
        mock_get_client.return_value = (write_zot, write_zot)

        from zotero_mcp.tools.write import merge_duplicates

        result = merge_duplicates("KEEPER", ["DUP1"], confirm=True, ctx=dummy_ctx)

        # The different attachment SHOULD have been re-parented
        reparent_calls = [
            c for c in write_zot.update_item.call_args_list
            if isinstance(c[0][0], dict) and c[0][0].get("key") == "D_ATT"
        ]
        assert len(reparent_calls) == 1

    @patch("zotero_mcp.tools.write._helpers._get_write_client")
    def test_merge_dry_run_shows_skipped_count(self, mock_get_client, dummy_ctx):
        """Dry run mentions skipped duplicate attachments."""
        keeper_att = {
            "key": "K_ATT",
            "version": 1,
            "data": {
                "itemType": "attachment",
                "contentType": "application/pdf",
                "filename": "paper.pdf",
                "md5": "abc123",
                "url": "",
                "parentItem": "KEEPER",
            },
        }
        dup_att = {
            "key": "D_ATT",
            "version": 1,
            "data": {
                "itemType": "attachment",
                "contentType": "application/pdf",
                "filename": "paper.pdf",
                "md5": "abc123",
                "url": "",
                "parentItem": "DUP1",
            },
        }

        write_zot = self._setup_merge([keeper_att], [dup_att])
        mock_get_client.return_value = (write_zot, write_zot)

        from zotero_mcp.tools.write import merge_duplicates

        result = merge_duplicates("KEEPER", ["DUP1"], confirm=False, ctx=dummy_ctx)

        assert "duplicate attachment" in result.lower()
        assert "skipped" in result.lower() or "1" in result


# -------------------------------------------------------------------------
# Fix 4 — Linked URL removal / no-PDF messaging
# -------------------------------------------------------------------------


class TestLinkedUrlRemoval:
    """Unit tests for _try_attach_oa_pdf linked-URL changes."""

    def _make_write_zot(self):
        """Create a mock write_zot for attachment tests."""
        write_zot = MagicMock()
        write_zot.create_items.return_value = {"success": {"0": "NEW_ATT"}}
        write_zot.item_template.return_value = {
            "itemType": "attachment",
            "linkMode": "linked_url",
            "url": "",
            "title": "",
            "contentType": "",
            "parentItem": "",
        }
        return write_zot

    @patch("zotero_mcp.tools._helpers._try_unpaywall")
    @patch("zotero_mcp.tools._helpers._try_arxiv_from_crossref")
    @patch("zotero_mcp.tools._helpers._try_semantic_scholar")
    @patch("zotero_mcp.tools._helpers._try_pmc")
    @patch("zotero_mcp.tools._helpers._download_and_attach_pdf")
    @patch("zotero_mcp.tools._helpers._attach_pdf_linked_url")
    def test_auto_mode_reports_url_when_download_fails(
        self, mock_linked, mock_download, mock_pmc, mock_ss, mock_arxiv, mock_unpaywall
    ):
        """In auto mode, if download fails but URL was found, report the URL."""
        # Unpaywall returns a URL, others return None
        mock_unpaywall.return_value = "https://example.com/paper.pdf"
        mock_arxiv.return_value = None
        mock_ss.return_value = None
        mock_pmc.return_value = None
        # Download fails
        mock_download.return_value = False

        write_zot = self._make_write_zot()
        ctx = DummyContext()

        result = _helpers._try_attach_oa_pdf(
            write_zot, "ITEM1", "10.1234/test", ctx,
            crossref_metadata=None, attach_mode="auto"
        )

        # linked_url should NOT be called in auto mode
        mock_linked.assert_not_called()
        # Should report the found URL so user can access manually
        assert "URL was found" in result
        assert "example.com/paper.pdf" in result

    @patch("zotero_mcp.tools._helpers._try_unpaywall")
    @patch("zotero_mcp.tools._helpers._try_arxiv_from_crossref")
    @patch("zotero_mcp.tools._helpers._try_semantic_scholar")
    @patch("zotero_mcp.tools._helpers._try_pmc")
    @patch("zotero_mcp.tools._helpers._attach_pdf_linked_url")
    def test_linked_url_mode_still_works(
        self, mock_linked, mock_pmc, mock_ss, mock_arxiv, mock_unpaywall
    ):
        """In linked_url mode, _attach_pdf_linked_url IS called."""
        mock_unpaywall.return_value = "https://example.com/paper.pdf"
        mock_arxiv.return_value = None
        mock_ss.return_value = None
        mock_pmc.return_value = None
        mock_linked.return_value = True

        write_zot = self._make_write_zot()
        ctx = DummyContext()

        result = _helpers._try_attach_oa_pdf(
            write_zot, "ITEM1", "10.1234/test", ctx,
            crossref_metadata=None, attach_mode="linked_url"
        )

        mock_linked.assert_called_once()
        assert "linked" in result.lower()

    @patch("zotero_mcp.tools._helpers._try_unpaywall")
    @patch("zotero_mcp.tools._helpers._try_arxiv_from_crossref")
    @patch("zotero_mcp.tools._helpers._try_semantic_scholar")
    @patch("zotero_mcp.tools._helpers._try_pmc")
    def test_no_pdf_message_is_clear(
        self, mock_pmc, mock_ss, mock_arxiv, mock_unpaywall
    ):
        """When no PDF is found, message clearly states no open-access PDF was found."""
        mock_unpaywall.return_value = None
        mock_arxiv.return_value = None
        mock_ss.return_value = None
        mock_pmc.return_value = None

        write_zot = self._make_write_zot()
        ctx = DummyContext()

        result = _helpers._try_attach_oa_pdf(
            write_zot, "ITEM1", "10.1234/test", ctx,
            crossref_metadata=None, attach_mode="auto"
        )

        assert "no open-access PDF found" in result
