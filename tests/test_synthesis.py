"""Tests for the synthesis tool module (digest + bibliography export)."""

from conftest import DummyContext, FakeZotero

import zotero_mcp.client as zotero_client
from zotero_mcp.tools import synthesis

# ---------------------------------------------------------------------------
# synthesize_annotations
# ---------------------------------------------------------------------------


class _DigestZotero(FakeZotero):
    """FakeZotero returning annotation/note fixtures and resolvable parents."""

    def __init__(self):
        super().__init__()
        # parent metadata items (papers) and the attachment between them.
        self._title_map = {
            "PAPER001": "Attention Is All You Need",
            "PAPER002": "Deep Residual Learning",
            "ATTACH01": {"itemType": "attachment", "parentItem": "PAPER001", "title": "Full Text PDF"},
            "ATTACH02": {"itemType": "attachment", "parentItem": "PAPER002", "title": "Full Text PDF"},
        }

    def item(self, item_key):
        entry = self._title_map.get(item_key)
        if isinstance(entry, dict):
            return {"key": item_key, "data": entry}
        if isinstance(entry, str):
            return {"key": item_key, "data": {"itemType": "journalArticle", "title": entry}}
        return {"key": item_key, "data": {"title": f"Item {item_key}"}}

    def items(self, **kwargs):
        item_type = kwargs.get("itemType")
        if item_type == "annotation":
            return [
                {
                    "key": "ANN1",
                    "data": {
                        "itemType": "annotation",
                        "annotationText": "Self-attention scales to long sequences",
                        "annotationComment": "key claim",
                        "parentItem": "ATTACH01",
                    },
                },
                {
                    "key": "ANN2",
                    "data": {
                        "itemType": "annotation",
                        "annotationText": "Residual connections ease optimization",
                        "annotationComment": "",
                        "parentItem": "ATTACH02",
                    },
                },
            ]
        if item_type == "note":
            return [
                {
                    "key": "NOTE1",
                    "data": {
                        "itemType": "note",
                        "note": "<p>Transformers remove recurrence entirely.</p>",
                        "parentItem": "PAPER001",
                    },
                },
            ]
        return []


def test_synthesize_annotations_groups_by_paper(monkeypatch):
    fake = _DigestZotero()
    monkeypatch.setattr(zotero_client, "get_zotero_client", lambda: fake)

    out = synthesis.synthesize_annotations(ctx=DummyContext())

    # Grouped under resolved paper titles, not raw keys.
    assert "## Attention Is All You Need" in out
    assert "## Deep Residual Learning" in out
    # Highlight text surfaced.
    assert "Self-attention scales to long sequences" in out
    assert "Residual connections ease optimization" in out
    # Comment surfaced for the first annotation.
    assert "key claim" in out
    # Note excerpt (HTML stripped) surfaced under its paper.
    assert "Transformers remove recurrence entirely." in out
    # Summary line counts.
    assert "2 papers" in out
    assert "2 highlights" in out
    assert "1 notes" in out


def test_synthesize_annotations_empty(monkeypatch):
    fake = FakeZotero()  # items() returns [] for every itemType
    monkeypatch.setattr(zotero_client, "get_zotero_client", lambda: fake)

    out = synthesis.synthesize_annotations(ctx=DummyContext())
    assert "No annotations or notes found" in out


# ---------------------------------------------------------------------------
# export_bibliography
# ---------------------------------------------------------------------------


class _BibZotero(FakeZotero):
    """FakeZotero honoring a content= kwarg for CSL/bibtex rendering."""

    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def _render(self, content, style):
        if content == "bibtex":
            return "@article{smith2020, title={Title}, author={Smith, J.}}"
        # bib / citation -> list of HTML snippets
        if content == "citation":
            return ['<span class="citation">(Smith, 2020)</span>']
        return ['<div class="csl-entry">Smith, J. (2020). Title. Journal.</div>']

    def items(self, **kwargs):
        self.last_kwargs = kwargs
        content = kwargs.get("content")
        if content:
            return self._render(content, kwargs.get("style"))
        return self._items

    def collection_items(self, key, **kwargs):
        self.last_kwargs = kwargs
        content = kwargs.get("content")
        if content:
            return self._render(content, kwargs.get("style"))
        return super().collection_items(key, **kwargs)


def test_export_bibliography_bib_strips_html(monkeypatch):
    fake = _BibZotero()
    monkeypatch.setattr(zotero_client, "get_zotero_client", lambda: fake)

    out = synthesis.export_bibliography(item_keys=["ABCD1234"], ctx=DummyContext())

    assert "Smith, J. (2020). Title. Journal." in out
    # HTML wrapper stripped.
    assert "csl-entry" not in out
    assert "Bibliography" in out
    # style passed through to the API.
    assert fake.last_kwargs.get("style") == "apa"
    assert fake.last_kwargs.get("content") == "bib"


def test_export_bibliography_style_passthrough(monkeypatch):
    fake = _BibZotero()
    monkeypatch.setattr(zotero_client, "get_zotero_client", lambda: fake)

    out = synthesis.export_bibliography(
        item_keys=["ABCD1234"],
        style="ieee",
        export_format="citation",
        ctx=DummyContext(),
    )

    assert fake.last_kwargs.get("style") == "ieee"
    assert fake.last_kwargs.get("content") == "citation"
    assert "(Smith, 2020)" in out
    assert "ieee" in out


def test_export_bibliography_bibtex_fenced(monkeypatch):
    fake = _BibZotero()
    monkeypatch.setattr(zotero_client, "get_zotero_client", lambda: fake)

    out = synthesis.export_bibliography(
        item_keys=["ABCD1234"],
        export_format="bibtex",
        ctx=DummyContext(),
    )

    assert "@article{smith2020" in out
    assert "```bibtex" in out
    # bibtex ignores style (not passed to the API).
    assert "style" not in fake.last_kwargs


def test_export_bibliography_collection(monkeypatch):
    fake = _BibZotero()
    monkeypatch.setattr(zotero_client, "get_zotero_client", lambda: fake)

    out = synthesis.export_bibliography(
        collection_key="COLL1234",
        export_format="bib",
        ctx=DummyContext(),
    )
    assert "Smith, J. (2020). Title. Journal." in out
    assert fake.last_kwargs.get("content") == "bib"


def test_export_bibliography_api_error(monkeypatch):
    class _ErrZot(FakeZotero):
        def items(self, **kwargs):
            raise RuntimeError("bib not supported in local mode")

    monkeypatch.setattr(zotero_client, "get_zotero_client", lambda: _ErrZot())

    out = synthesis.export_bibliography(item_keys=["ABCD1234"], ctx=DummyContext())
    assert "web API" in out.lower() or "ZOTERO_API_KEY" in out
