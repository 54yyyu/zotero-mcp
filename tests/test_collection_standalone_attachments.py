"""Regression tests for #224: standalone (parentless) attachments must appear
in zotero_get_collection_items.

A PDF dragged into Zotero without a parent metadata item is a top-level
"attachment" item. The collection listing previously filtered out every
attachment, so such standalone PDFs vanished from the collection entirely.
"""

from conftest import DummyContext, FakeZotero

from zotero_mcp import server


class FakeZoteroForCollection(FakeZotero):
    """FakeZotero that serves a fixed item set as one collection's contents."""

    def collection(self, key, **kwargs):
        return {"key": key, "data": {"name": "Test Collection"}}

    def collection_items(self, key, start=0, limit=100, **kwargs):
        # Mirror the Zotero API: returns parents AND children mixed together.
        return self._items[start:start + limit]


def _make_zot(monkeypatch, items):
    zot = FakeZoteroForCollection()
    zot._items = items
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: zot)
    return zot


# Item fixtures ------------------------------------------------------------

def _paper(key="PAPER1"):
    return {
        "key": key,
        "data": {"itemType": "journalArticle", "title": "Normal Paper", "date": "2024"},
    }


def _child_pdf(key="CHILD1", parent="PAPER1"):
    return {
        "key": key,
        "data": {
            "itemType": "attachment",
            "contentType": "application/pdf",
            "parentItem": parent,
            "filename": "normal.pdf",
            "title": "normal.pdf",
        },
    }


def _standalone_pdf(key="STANDALONE1", filename="why-language-models-hallucinate.pdf"):
    return {
        "key": key,
        "data": {
            "itemType": "attachment",
            "contentType": "application/pdf",
            "filename": filename,
            "title": filename,
            # NOTE: no "parentItem" — this is what makes it standalone
        },
    }


# Tests --------------------------------------------------------------------

def test_standalone_pdf_appears_in_collection(monkeypatch):
    _make_zot(monkeypatch, [_paper(), _child_pdf(), _standalone_pdf()])

    result = server.get_collection_items(
        collection_key="COLL1", detail="keys_only", ctx=DummyContext()
    )

    assert "STANDALONE1" in result
    assert "why-language-models-hallucinate.pdf" in result


def test_child_attachment_not_listed_as_top_level(monkeypatch):
    # The child PDF belongs under PAPER1 and must NOT appear as its own item
    _make_zot(monkeypatch, [_paper(), _child_pdf(), _standalone_pdf()])

    result = server.get_collection_items(
        collection_key="COLL1", detail="keys_only", ctx=DummyContext()
    )

    assert "CHILD1" not in result


def test_collection_count_includes_standalone(monkeypatch):
    # 1 paper + 1 standalone = 2 top-level items (child excluded)
    _make_zot(monkeypatch, [_paper(), _child_pdf(), _standalone_pdf()])

    result = server.get_collection_items(
        collection_key="COLL1", detail="keys_only", ctx=DummyContext()
    )

    assert "(2 items)" in result


def test_standalone_pdf_flagged_as_pdf(monkeypatch):
    _make_zot(monkeypatch, [_standalone_pdf()])

    result = server.get_collection_items(
        collection_key="COLL1", detail="keys_only", ctx=DummyContext()
    )

    assert "[PDF]" in result


def test_standalone_pdf_uses_filename_when_no_title(monkeypatch):
    no_title = {
        "key": "STANDALONE2",
        "data": {
            "itemType": "attachment",
            "contentType": "application/pdf",
            "filename": "raw-dump.pdf",
        },
    }
    _make_zot(monkeypatch, [no_title])

    result = server.get_collection_items(
        collection_key="COLL1", detail="keys_only", ctx=DummyContext()
    )

    assert "raw-dump.pdf" in result
    assert "Untitled" not in result


def test_normal_papers_still_listed(monkeypatch):
    # Regression guard: ordinary parent items keep working
    _make_zot(monkeypatch, [_paper("PAPER1"), _paper("PAPER2")])

    result = server.get_collection_items(
        collection_key="COLL1", detail="keys_only", ctx=DummyContext()
    )

    assert "PAPER1" in result
    assert "PAPER2" in result
    assert "(2 items)" in result


def _standalone_pdf_no_title(key="STANDALONE_NT", filename="raw-dump.pdf"):
    return {
        "key": key,
        "data": {
            "itemType": "attachment",
            "contentType": "application/pdf",
            "filename": filename,
            # no "title", no "parentItem"
        },
    }


def test_standalone_pdf_shown_in_summary_detail(monkeypatch):
    _make_zot(monkeypatch, [_standalone_pdf_no_title()])

    result = server.get_collection_items(
        collection_key="COLL1", detail="summary", ctx=DummyContext()
    )

    assert "raw-dump.pdf" in result
    assert "Untitled" not in result


def test_standalone_pdf_shown_in_full_detail(monkeypatch):
    _make_zot(monkeypatch, [_standalone_pdf_no_title()])

    result = server.get_collection_items(
        collection_key="COLL1", detail="full", ctx=DummyContext()
    )

    assert "raw-dump.pdf" in result
    assert "Untitled" not in result


def test_standalone_pdf_flagged_as_pdf_in_summary(monkeypatch):
    # The PDF indicator must show in the default detail level too, not only keys_only
    _make_zot(monkeypatch, [_standalone_pdf()])

    result = server.get_collection_items(
        collection_key="COLL1", detail="summary", ctx=DummyContext()
    )

    assert "PDF" in result


def test_standalone_note_not_listed(monkeypatch):
    # #224 is about attachments (PDFs). Standalone notes have no useful title
    # and are out of scope — they must NOT leak in as "Untitled" noise.
    standalone_note = {
        "key": "NOTE1",
        "data": {"itemType": "note", "note": "<p>loose thought</p>"},
    }
    _make_zot(monkeypatch, [_paper(), standalone_note])

    result = server.get_collection_items(
        collection_key="COLL1", detail="keys_only", ctx=DummyContext()
    )

    assert "NOTE1" not in result
    assert "(1 items)" in result
