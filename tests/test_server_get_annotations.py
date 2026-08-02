"""Tests for get_annotations, especially descending through attachments
to find annotations (which in Zotero's data model are children of the
PDF attachment, not of the parent paper)."""

import json

from zotero_mcp import server
from zotero_mcp.better_bibtex_client import process_annotation
from zotero_mcp.tools.annotations import _annotation_to_record


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


def _annotation(key, parent, text, tags=None):
    return {
        "key": key,
        "data": {
            "itemType": "annotation",
            "annotationType": "highlight",
            "annotationText": text,
            "annotationComment": "",
            "parentItem": parent,
            "tags": [] if tags is None else tags,
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


def test_get_annotations_json_returns_stable_structured_records(monkeypatch):
    """JSON output normalizes Zotero annotation metadata for downstream tools."""
    parents = {
        "PAPER001": {"key": "PAPER001", "data": {
            "title": "A Paper", "itemType": "journalArticle",
        }},
        "ATTACH01": {"key": "ATTACH01", "data": {
            "title": "Full Text PDF", "itemType": "attachment",
            "parentItem": "PAPER001",
        }},
    }
    annotation = _annotation(
        "ANNO0001",
        "ATTACH01",
        "structured highlight",
        tags=[{"tag": "method", "type": 1}, {"tag": "to-read"}],
    )
    annotation["data"].update({
        "annotationComment": "important claim",
        "annotationColor": "#ffd400",
        "annotationPageLabel": "7",
        "annotationPosition": json.dumps({"pageIndex": 6, "rects": []}),
        "dateAdded": "2026-04-10T12:00:00Z",
        "dateModified": "2026-04-11T12:00:00Z",
    })
    children = {
        "PAPER001": [{"key": "ATTACH01", "data": {
            "itemType": "attachment", "contentType": "application/pdf",
        }}],
        "ATTACH01": [annotation],
    }
    fake = FakeZoteroForAnnotations(parents, children)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(
        item_key="PAPER001", format="json", ctx=DummyContext()
    )

    assert json.loads(result) == [{
        "annotation_key": "ANNO0001",
        "item_key": "PAPER001",
        "attachment_key": "ATTACH01",
        "parent_title": "A Paper",
        "attachment_title": "Full Text PDF",
        "type": "highlight",
        "page": "7",
        "page_index": 6,
        "text": "structured highlight",
        "comment": "important claim",
        "color": "#ffd400",
        "color_category": "Yellow",
        "tags": ["method", "to-read"],
        "created": "2026-04-10T12:00:00Z",
        "modified": "2026-04-11T12:00:00Z",
        "source": "zotero",
    }]


def test_get_annotations_json_library_wide_resolves_paper_context(monkeypatch):
    annotation = _annotation("ANNO0001", "ATTACH01", "library highlight")
    parents = {
        "PAPER001": {"key": "PAPER001", "data": {
            "title": "A Paper", "itemType": "journalArticle",
        }},
        "ATTACH01": {"key": "ATTACH01", "data": {
            "title": "Full Text PDF", "itemType": "attachment",
            "parentItem": "PAPER001",
        }},
    }

    class _LibraryFake(FakeZoteroForAnnotations):
        def items(self, start=0, limit=100, itemType=None, itemKey=None, **_kwargs):
            if itemKey:
                keys = itemKey.split(",")
                return [self._parents[key] for key in keys if key in self._parents]
            if itemType == "annotation":
                return [annotation][start:start + limit]
            return []

    fake = _LibraryFake(parents, {})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    records = json.loads(server.get_annotations(format="json", ctx=DummyContext()))

    assert records[0]["item_key"] == "PAPER001"
    assert records[0]["attachment_key"] == "ATTACH01"
    assert records[0]["parent_title"] == "A Paper"
    assert records[0]["attachment_title"] == "Full Text PDF"


def test_get_annotations_json_falls_back_to_caller_context(monkeypatch):
    """An unresolvable attachment must not be passed off as the paper key.

    Trashed/unsynced attachments can't be re-fetched, but the caller already
    told us which paper it asked about — use that instead of a placeholder.
    """
    parents = {
        "PAPER001": {"key": "PAPER001", "data": {
            "title": "Known Good Title", "itemType": "journalArticle",
        }},
    }
    children = {
        "PAPER001": [{"key": "ATTACH01", "data": {
            "itemType": "attachment", "contentType": "application/pdf",
        }}],
        "ATTACH01": [_annotation("ANNO0001", "ATTACH01", "orphaned highlight")],
    }

    class _UnresolvableAttachment(FakeZoteroForAnnotations):
        def item(self, key):
            if key == "ATTACH01":
                raise RuntimeError("404 attachment not found")
            return super().item(key)

        def items(self, itemKey=None, **_kwargs):
            keys = (itemKey or "").split(",")
            return [self._parents[k] for k in keys if k in self._parents]

    fake = _UnresolvableAttachment(parents, children)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    record = json.loads(server.get_annotations(
        item_key="PAPER001", format="json", ctx=DummyContext()
    ))[0]

    assert record["item_key"] == "PAPER001"
    assert record["parent_title"] == "Known Good Title"
    assert record["attachment_key"] == "ATTACH01"


def test_annotation_record_normalizes_page_across_sources():
    """All three sources must describe the same physical page identically.

    ``_pdf_page`` (Better BibTeX, direct PDF extraction) is 1-based while the
    Zotero API's ``pageIndex`` is 0-based, and only Better BibTeX precomputes
    a colour category. The record has to hide both differences, otherwise a
    downstream script silently reads a different page depending on whether
    Zotero desktop happened to be running.
    """
    raw = {
        "key": "ANNO0001",
        "annotationType": "highlight",
        "annotationText": "same highlight",
        "annotationColor": "#ffd400",
        "annotationPageLabel": "7",
        "annotationPosition": json.dumps({"pageIndex": 6, "rects": []}),
    }
    # Built exactly as get_annotations builds them, from one raw annotation.
    processed = process_annotation(raw, {"itemKey": "ATTACH01"})
    by_source = {
        "zotero": dict(raw, itemType="annotation"),
        "better_bibtex": {
            "annotationText": raw["annotationText"],
            "annotationColor": raw["annotationColor"],
            "_pdf_page": processed["page"],
            "_pageLabel": processed["pageLabel"],
            "_from_better_bibtex": True,
        },
        "pdf_extraction": {
            "annotationText": raw["annotationText"],
            "annotationColor": raw["annotationColor"],
            "_pdf_page": 7,  # pdfannots2json is 1-based, and emits no label
            "_from_pdf_extraction": True,
        },
    }

    for source, data in by_source.items():
        record = _annotation_to_record({"key": "ANNO0001", "data": data})
        assert record["source"] == source
        assert record["page"] == "7", source
        assert record["page_index"] == 6, source
        assert record["color_category"] == "Yellow", source


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


def test_get_annotations_json_empty_result_is_an_array(monkeypatch):
    parents = {
        "PAPER001": {"data": {"title": "Bare Paper", "itemType": "journalArticle"}},
    }
    fake = FakeZoteroForAnnotations(parents, {"PAPER001": []})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(
        item_key="PAPER001", format="json", ctx=DummyContext()
    )

    assert json.loads(result) == []


def test_get_annotations_surfaces_tags(monkeypatch):
    """Tags on an annotation are rendered, not dropped (#377)."""
    parents = {
        "ATTACH01": {"data": {"title": "PDF", "itemType": "attachment"}},
    }
    children = {
        "ATTACH01": [
            _annotation(
                "ANNO0001", "ATTACH01", "tagged highlight",
                tags=[{"tag": "method", "type": 1}, {"tag": "to-read"}],
            ),
        ],
    }
    fake = FakeZoteroForAnnotations(parents, children)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(item_key="ATTACH01", ctx=DummyContext())

    assert "**Tags:** `method` `to-read`" in result


def test_get_annotations_tolerates_missing_or_odd_tags(monkeypatch):
    """None / bare-string tags don't crash and render sensibly (#377)."""
    parents = {
        "ATTACH01": {"data": {"title": "PDF", "itemType": "attachment"}},
    }
    children = {
        "ATTACH01": [
            _annotation("ANNO0001", "ATTACH01", "no tags", tags=None),
            _annotation("ANNO0002", "ATTACH01", "string tags", tags=["plain"]),
        ],
    }
    fake = FakeZoteroForAnnotations(parents, children)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(item_key="ATTACH01", ctx=DummyContext())

    assert "Error fetching annotations" not in result
    assert "**Tags:** `plain`" in result
    # The untagged annotation gets no Tags line.
    assert result.count("**Tags:**") == 1


def test_get_annotations_better_bibtex_path_carries_tags(monkeypatch):
    """Better BibTeX annotations keep their tags (bare strings) (#377)."""
    import zotero_mcp.better_bibtex_client as bbt

    class _FakeBBT:
        def is_zotero_running(self):
            return True

        def _make_request(self, method, params=None):
            return [{"citekey": "smith2020", "library": "1"}]

        def get_attachments(self, citekey, library="*"):
            return [{
                "itemKey": "ATTACH01",
                "title": "Full Text PDF",
                "path": "/tmp/smith2020.pdf",
                "annotations": [{
                    "key": "ANNO0001",
                    "annotationType": "highlight",
                    "annotationText": "bbt highlight",
                    "annotationComment": "",
                    "annotationColor": "#ffd400",
                    "annotationPageLabel": "3",
                    "tags": ["method", "to-read"],
                }],
            }]

        def get_annotations_from_attachment(self, attachment):
            return attachment.get("annotations", [])

    parents = {
        "PAPER001": {"data": {
            "title": "A Paper",
            "itemType": "journalArticle",
            "extra": "Citation Key: smith2020",
        }},
    }
    fake = FakeZoteroForAnnotations(parents, {})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr(bbt, "ZoteroBetterBibTexAPI", lambda *a, **k: _FakeBBT())
    monkeypatch.setenv("ZOTERO_LOCAL", "true")

    result = server.get_annotations(item_key="PAPER001", ctx=DummyContext())

    assert "bbt highlight" in result
    assert "**Tags:** `method` `to-read`" in result


def test_get_annotations_pdf_extraction_path_carries_tags(monkeypatch):
    """Directly-extracted PDF annotations keep their tags (#377)."""
    import zotero_mcp.pdfannots_helper as pdfhelper

    class _DumpingZotero(FakeZoteroForAnnotations):
        def dump(self, key, filename=None, path=None):
            with open(f"{path}/{filename}", "wb") as f:
                f.write(b"%PDF-1.4 fake")

    parents = {
        "PAPER001": {"data": {"title": "A Paper", "itemType": "journalArticle"}},
    }
    children = {
        "PAPER001": [
            {"key": "ATTACH01", "data": {
                "itemType": "attachment",
                "contentType": "application/pdf",
                "title": "Full Text PDF",
            }},
        ],
        "ATTACH01": [],
    }
    fake = _DumpingZotero(parents, children)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr(pdfhelper, "ensure_pdfannots_installed", lambda: True)
    monkeypatch.setattr(
        pdfhelper,
        "extract_annotations_from_pdf",
        lambda path, outdir=None, **kw: [{
            "id": "a1",
            "type": "highlight",
            "annotatedText": "pdf highlight",
            "comment": "",
            "color": "#ffd400",
            "page": 2,
            "tags": [{"tag": "from-pdf", "type": 1}],
        }],
    )
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(
        item_key="PAPER001", use_pdf_extraction=True, ctx=DummyContext()
    )

    assert "pdf highlight" in result
    assert "**Tags:** `from-pdf`" in result


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
