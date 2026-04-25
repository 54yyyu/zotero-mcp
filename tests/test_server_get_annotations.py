"""Tests for get_annotations, especially descending through attachments
to find annotations (which in Zotero's data model are children of the
PDF attachment, not of the parent paper)."""

from zotero_mcp import server


class DummyContext:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


class FakeZoteroForAnnotations:
    """Fake Zotero client that models the real parent/child hierarchy.

    Papers contain notes and attachments; annotations live under
    attachments, not directly under the paper.
    """

    def __init__(self, parents, children_by_key):
        self._parents = parents
        self._children = children_by_key

    def item(self, key):
        return self._parents.get(key, {"data": {"title": "Unknown"}})

    def children(self, item_key, start=0, limit=100, itemType=None, **_kwargs):
        all_children = self._children.get(item_key, [])
        if itemType:
            all_children = [
                c for c in all_children
                if c.get("data", {}).get("itemType") == itemType
            ]
        return all_children[start:start + limit]


def _annotation(key, parent, text):
    return {
        "key": key,
        "data": {
            "itemType": "annotation",
            "annotationType": "highlight",
            "annotationText": text,
            "annotationComment": "",
            "parentItem": parent,
            "tags": [],
        },
    }


def test_get_annotations_descends_through_attachment(monkeypatch):
    """Paper key → descend through its PDF attachment to find annotations."""
    parents = {
        "PAPER001": {"data": {"title": "A Paper", "itemType": "journalArticle"}},
    }
    # Paper's direct children: one attachment (PDF) and one note.
    # The Zotero web API does NOT return annotations as children of the paper.
    children = {
        "PAPER001": [
            {"key": "ATTACH01", "data": {"itemType": "attachment", "contentType": "application/pdf"}},
            {"key": "NOTE0001", "data": {"itemType": "note", "note": "<p>n</p>"}},
        ],
        "ATTACH01": [
            _annotation("ANNO0001", "ATTACH01", "first highlight"),
            _annotation("ANNO0002", "ATTACH01", "second highlight"),
            _annotation("ANNO0003", "ATTACH01", "third highlight"),
        ],
    }
    fake = FakeZoteroForAnnotations(parents, children)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(item_key="PAPER001", ctx=DummyContext())

    assert "ANNO0001" in result
    assert "ANNO0002" in result
    assert "ANNO0003" in result
    assert "first highlight" in result
    assert "No annotations found" not in result


def test_get_annotations_accepts_attachment_key(monkeypatch):
    """Attachment key → annotations are returned as direct children."""
    parents = {
        "ATTACH01": {"data": {"title": "PDF", "itemType": "attachment"}},
    }
    children = {
        "ATTACH01": [
            _annotation("ANNO0001", "ATTACH01", "highlight text"),
            _annotation("ANNO0002", "ATTACH01", "another one"),
        ],
    }
    fake = FakeZoteroForAnnotations(parents, children)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(item_key="ATTACH01", ctx=DummyContext())

    assert "ANNO0001" in result
    assert "ANNO0002" in result
    assert "highlight text" in result


def test_get_annotations_dedupes_across_paths(monkeypatch):
    """If the API returns the same annotation as both a direct child and
    a nested attachment child, it should appear only once."""
    parents = {
        "PAPER001": {"data": {"title": "Paper", "itemType": "journalArticle"}},
    }
    anno = _annotation("ANNO0001", "ATTACH01", "only once")
    children = {
        # Local API quirk: paper+itemType=annotation returns grandchildren
        "PAPER001": [
            anno,
            {"key": "ATTACH01", "data": {"itemType": "attachment", "contentType": "application/pdf"}},
        ],
        "ATTACH01": [anno],
    }
    fake = FakeZoteroForAnnotations(parents, children)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(item_key="PAPER001", ctx=DummyContext())

    assert result.count("ANNO0001") == 1
    assert result.count("only once") == 1


def test_get_annotations_handles_multiple_attachments(monkeypatch):
    """Paper with two PDF attachments → annotations from both are returned."""
    parents = {
        "PAPER001": {"data": {"title": "Paper", "itemType": "journalArticle"}},
    }
    children = {
        "PAPER001": [
            {"key": "ATTACH01", "data": {"itemType": "attachment", "contentType": "application/pdf"}},
            {"key": "ATTACH02", "data": {"itemType": "attachment", "contentType": "application/pdf"}},
        ],
        "ATTACH01": [_annotation("ANNO0001", "ATTACH01", "from pdf one")],
        "ATTACH02": [_annotation("ANNO0002", "ATTACH02", "from pdf two")],
    }
    fake = FakeZoteroForAnnotations(parents, children)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(item_key="PAPER001", ctx=DummyContext())

    assert "ANNO0001" in result
    assert "ANNO0002" in result


def test_get_annotations_reports_none_when_empty(monkeypatch):
    """Paper with attachments but zero annotations → friendly empty message."""
    parents = {
        "PAPER001": {"data": {"title": "Bare Paper", "itemType": "journalArticle"}},
    }
    children = {
        "PAPER001": [{"key": "ATTACH01", "data": {"itemType": "attachment", "contentType": "application/pdf"}}],
        "ATTACH01": [],
    }
    fake = FakeZoteroForAnnotations(parents, children)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(item_key="PAPER001", ctx=DummyContext())

    assert "No annotations found" in result
    assert "Bare Paper" in result


def test_get_annotations_paginates_through_many_annotations(monkeypatch):
    """Works correctly when attachment has >100 annotations (page boundary)."""
    parents = {
        "PAPER001": {"data": {"title": "Paper", "itemType": "journalArticle"}},
    }
    many = [_annotation(f"A{i:05d}", "ATTACH01", f"hl {i}") for i in range(150)]
    children = {
        "PAPER001": [{"key": "ATTACH01", "data": {"itemType": "attachment", "contentType": "application/pdf"}}],
        "ATTACH01": many,
    }
    fake = FakeZoteroForAnnotations(parents, children)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(item_key="PAPER001", ctx=DummyContext())

    assert "A00000" in result
    assert "A00099" in result
    assert "A00149" in result  # past first page
