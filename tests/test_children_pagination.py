"""Regression tests: children() call sites must paginate.

pyzotero's ``children()`` does not paginate — without an explicit ``limit``
the Zotero API returns only its default first page of 25 results, so every
unpaginated call site silently sees at most the first 25 children of an
item. These tests pin each call site to a fake that pages like the real
API, with more than 100 children so ``_paginate``'s ``page_size=100`` must
itself request at least two pages.
"""

from conftest import FakeZotero

from zotero_mcp import client as client_module
from zotero_mcp import server
from zotero_mcp.tools import discovery
from zotero_mcp.tools import write as write_tools


class PagedChildrenZotero(FakeZotero):
    """children() honors start/limit like the real Zotero API (default 25)."""

    def children(self, item_key, start=0, limit=25, **kwargs):
        kids = self._children.get(item_key, [])
        return kids[int(start):int(start) + int(limit)]


def _note(key):
    return {"key": key, "version": 1, "data": {"key": key, "itemType": "note"}}


def _pdf(key, filename="paper.pdf", md5="abc123"):
    return {"key": key, "version": 1, "data": {
        "key": key,
        "itemType": "attachment",
        "contentType": "application/pdf",
        "filename": filename,
        "title": filename,
        "md5": md5,
    }}


def _parent(key, title="Parent Item"):
    return {"key": key, "version": 1, "data": {
        "key": key, "itemType": "journalArticle", "title": title,
    }}


# 130 note children in front of the PDF: past the API's first page of 25
# AND past _paginate's first page of 100.
def _children_with_pdf_last(pdf):
    return [_note(f"N{i:03d}") for i in range(130)] + [pdf]


class TestGetAttachmentDetailsPagination:
    def test_finds_pdf_attachment_past_first_api_page(self):
        zot = PagedChildrenZotero()
        pdf = _pdf("PDF00001")
        zot._children = {"PAR00001": _children_with_pdf_last(pdf)}

        details = client_module.get_attachment_details(zot, _parent("PAR00001"))

        assert details is not None, "PDF attachment past page 1 was not found"
        assert details.key == "PDF00001"
        assert details.content_type == "application/pdf"


class TestGetItemChildrenPagination:
    def test_lists_children_past_first_api_page(self, monkeypatch, dummy_ctx):
        zot = PagedChildrenZotero()
        pdf = _pdf("PDF00001")
        zot._items = [_parent("PAR00001")]
        zot._children = {"PAR00001": _children_with_pdf_last(pdf)}
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: zot)

        result = server.get_item_children(item_key="PAR00001", ctx=dummy_ctx)

        assert "PDF00001" in result, "attachment past page 1 missing from listing"

    def test_batch_variant_lists_children_past_first_api_page(self, monkeypatch, dummy_ctx):
        zot = PagedChildrenZotero()
        pdf = _pdf("PDF00001")
        zot._items = [_parent("PAR00001")]
        zot._children = {"PAR00001": _children_with_pdf_last(pdf)}
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: zot)

        result = server.get_items_children(item_keys=["PAR00001"], ctx=dummy_ctx)

        assert "PDF00001" in result, "attachment past page 1 missing from batch listing"


class TestItemHasPdfPagination:
    def test_detects_pdf_past_first_api_page(self):
        zot = PagedChildrenZotero()
        pdf = _pdf("PDF00001")
        zot._children = {"PAR00001": _children_with_pdf_last(pdf)}

        assert discovery._item_has_pdf(zot, _parent("PAR00001")) is True


class TestGetPdfOutlinePagination:
    def test_finds_pdf_attachment_past_first_api_page(self, monkeypatch, dummy_ctx):
        # The TOC is read out-of-process since #372; stub the outcome.
        monkeypatch.setattr(
            write_tools,
            "_extract_pdf_toc",
            lambda *a, **k: write_tools.TocOutcome("ok", [(1, "Introduction", 1)]),
        )
        monkeypatch.setattr("zotero_mcp.client.get_local_zotero_client", lambda: None)

        zot = PagedChildrenZotero()
        pdf = _pdf("PDF00001")
        zot._items = [_parent("PAR00001")]
        zot._children = {"PAR00001": _children_with_pdf_last(pdf)}
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: zot)

        result = server.get_pdf_outline(item_key="PAR00001", ctx=dummy_ctx)

        assert "No PDF attachment found" not in result
        assert "Introduction" in result


class TestGetAnnotationsPdfFallbackPagination:
    def test_pdf_extraction_scans_attachments_past_first_api_page(self, monkeypatch, dummy_ctx):
        import zotero_mcp.pdfannots_helper as ph

        monkeypatch.delenv("ZOTERO_LOCAL", raising=False)
        monkeypatch.setattr(ph, "ensure_pdfannots_installed", lambda: True)
        monkeypatch.setattr(ph, "extract_annotations_from_pdf", lambda path, tmpdir: [
            {"id": "a1", "type": "highlight", "annotatedText": "MARKER TEXT PAST PAGE ONE",
             "comment": "", "color": "", "page": 3},
        ])

        zot = PagedChildrenZotero()
        pdf = _pdf("PDF00001")
        zot._items = [_parent("PAR00001")]
        zot._children = {"PAR00001": _children_with_pdf_last(pdf)}
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: zot)

        result = server.get_annotations(
            item_key="PAR00001", use_pdf_extraction=True, ctx=dummy_ctx
        )

        assert "MARKER TEXT PAST PAGE ONE" in result, (
            "annotations on a PDF attachment past page 1 were not extracted"
        )
