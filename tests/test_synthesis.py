"""Tests for the synthesis tool module (digest + bibliography export)."""

import json

import pytest
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


def test_synthesize_annotations_markdown_disambiguates_same_title(monkeypatch):
    """Two distinct papers sharing a title get distinguishable headings."""
    fake = _DigestZotero()
    fake._title_map["PAPER002"] = "Attention Is All You Need"
    monkeypatch.setattr(zotero_client, "get_zotero_client", lambda: fake)

    out = synthesis.synthesize_annotations(ctx=DummyContext())

    assert "## Attention Is All You Need (PAPER001)" in out
    assert "## Attention Is All You Need (PAPER002)" in out
    # Still two papers, and each keeps its own highlight.
    assert "2 papers" in out
    assert "Self-attention scales to long sequences" in out
    assert "Residual connections ease optimization" in out


def test_synthesize_annotations_json_groups_structured_records_by_paper(monkeypatch):
    fake = _DigestZotero()
    monkeypatch.setattr(zotero_client, "get_zotero_client", lambda: fake)

    payload = json.loads(
        synthesis.synthesize_annotations(format="json", ctx=DummyContext())
    )

    assert payload["summary"] == {
        "papers": 2,
        "highlights": 2,
        "notes": 1,
    }
    papers = {paper["item_key"]: paper for paper in payload["papers"]}
    assert papers["PAPER001"]["title"] == "Attention Is All You Need"
    assert papers["PAPER001"]["highlights"][0] == {
        "annotation_key": "ANN1",
        "item_key": "PAPER001",
        "attachment_key": "ATTACH01",
        "parent_title": "Attention Is All You Need",
        "attachment_title": "Full Text PDF",
        "type": None,
        "page": None,
        "page_index": None,
        "text": "Self-attention scales to long sequences",
        "comment": "key claim",
        "color": None,
        "color_category": None,
        "tags": [],
        "created": None,
        "modified": None,
        "source": "zotero",
    }
    assert papers["PAPER001"]["notes"] == [{
        "note_key": "NOTE1",
        "item_key": "PAPER001",
        "parent_title": "Attention Is All You Need",
        "text": "Transformers remove recurrence entirely.",
        "tags": [],
        "created": None,
        "modified": None,
    }]


def test_synthesize_annotations_empty(monkeypatch):
    fake = FakeZotero()  # items() returns [] for every itemType
    monkeypatch.setattr(zotero_client, "get_zotero_client", lambda: fake)

    out = synthesis.synthesize_annotations(ctx=DummyContext())
    assert "No annotations or notes found" in out


def test_synthesize_annotations_json_empty_result_is_structured(monkeypatch):
    fake = FakeZotero()
    monkeypatch.setattr(zotero_client, "get_zotero_client", lambda: fake)

    payload = json.loads(
        synthesis.synthesize_annotations(format="json", ctx=DummyContext())
    )

    assert payload == {
        "summary": {"papers": 0, "highlights": 0, "notes": 0},
        "papers": [],
    }


# ---------------------------------------------------------------------------
# export_bibliography
# ---------------------------------------------------------------------------


class _BibZotero(FakeZotero):
    """Stub matching what Zotero's API actually returns for rendering.

    Both the local and web APIs answer ``include=bib``/``citation`` with JSON
    rows carrying the rendered string, and ``format=bibtex`` with the whole
    .bib file as raw bytes. Neither serves ``content=`` (that implies Atom, and
    the local API 501s on it), so this stub raises if it ever sees one —
    sending ``content=`` is precisely the bug behind #371.
    """

    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def _render(self, kwargs):
        if kwargs.get("content"):
            raise AssertionError(
                "content= implies Atom, which the local API rejects (#371)"
            )
        if kwargs.get("format") == "bibtex":
            return b"@article{smith2020, title={Title}, author={Smith, J.}}"
        include = kwargs.get("include")
        if include == "citation":
            return [{"key": "ABCD1234", "citation": '<span>(Smith, 2020)</span>'}]
        if include == "bib":
            return [
                {
                    "key": "ABCD1234",
                    "bib": '<div class="csl-entry">Smith, J. (2020). Title. Journal.</div>',
                },
                # An attachment: present in the library, nothing to render.
                {"key": "ATTACH01", "bib": ""},
            ]
        return None

    def items(self, **kwargs):
        self.last_kwargs = kwargs
        rendered = self._render(kwargs)
        return self._items if rendered is None else rendered

    def top(self, **kwargs):
        self.last_kwargs = kwargs
        rendered = self._render(kwargs)
        return self._items if rendered is None else rendered

    def collection_items(self, key, **kwargs):
        self.last_kwargs = kwargs
        rendered = self._render(kwargs)
        if rendered is None:
            return super().collection_items(key, **kwargs)
        return rendered


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
    assert fake.last_kwargs.get("include") == "bib"
    # Rows with nothing rendered (the attachment) are dropped, not emitted
    # as blank numbered entries.
    assert "\n2. " not in out


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
    assert fake.last_kwargs.get("include") == "citation"
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
    # bibtex is a whole-file export: format=, not include=, and style is
    # meaningless so it is not sent.
    assert fake.last_kwargs.get("format") == "bibtex"
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
    assert fake.last_kwargs.get("include") == "bib"


def test_export_bibliography_api_error(monkeypatch):
    class _ErrZot(FakeZotero):
        def items(self, **kwargs):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(zotero_client, "get_zotero_client", lambda: _ErrZot())

    out = synthesis.export_bibliography(item_keys=["ABCD1234"], ctx=DummyContext())
    assert "Error rendering bibliography" in out
    # The remedy names both modes now that either can render.
    assert "local" in out.lower()


# ---------------------------------------------------------------------------
# export_bibliography — rendering works without web credentials (#371)
#
# The original fix for #371 concluded the local API had no citation engine and
# routed rendering through the web API, locking local-only users out. The real
# constraint is narrower: the local API rejects *Atom*, and `content=` implies
# Atom. Asking the JSON way (`include=bib`/`citation`, or `format=bibtex`)
# renders fine locally, verified against a live Zotero 7 local API.
# ---------------------------------------------------------------------------


class _AtomRejectingZotero(_BibZotero):
    """Local API stub: renders JSON requests, 501s on Atom ones like Zotero."""

    def _render(self, kwargs):
        if kwargs.get("content"):
            raise RuntimeError("Local API does not support Atom output")
        return super()._render(kwargs)


def test_export_bibliography_renders_in_local_only_mode(monkeypatch):
    """No web credentials at all → still renders, no error (#371)."""
    local = _AtomRejectingZotero()
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setattr(zotero_client, "get_zotero_client", lambda: local)
    monkeypatch.setattr(zotero_client, "get_web_zotero_client", lambda: None)

    out = synthesis.export_bibliography(item_keys=["ABCD1234"], ctx=DummyContext())

    assert "Smith, J. (2020). Title. Journal." in out
    assert "Atom output" not in out
    assert "ZOTERO_API_KEY" not in out


@pytest.mark.parametrize("export_format", ["bib", "citation", "bibtex"])
def test_export_bibliography_never_requests_atom(monkeypatch, export_format):
    """No format may fall back to content=, which is what broke local mode."""
    local = _AtomRejectingZotero()
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setattr(zotero_client, "get_zotero_client", lambda: local)
    monkeypatch.setattr(zotero_client, "get_web_zotero_client", lambda: None)

    out = synthesis.export_bibliography(
        item_keys=["ABCD1234"], export_format=export_format, ctx=DummyContext()
    )

    assert "content" not in (local.last_kwargs or {})
    assert "Atom output" not in out


def test_export_bibliography_local_only_bibtex_is_decoded(monkeypatch):
    """format=bibtex returns raw bytes; they must reach the user as text."""
    local = _AtomRejectingZotero()
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setattr(zotero_client, "get_zotero_client", lambda: local)
    monkeypatch.setattr(zotero_client, "get_web_zotero_client", lambda: None)

    out = synthesis.export_bibliography(
        item_keys=["ABCD1234"], export_format="bibtex", ctx=DummyContext()
    )

    assert "@article{smith2020" in out
    # A bytes repr leaking through would show up as b'...' in the output.
    assert "b'" not in out


class _LooseItemKeyZotero(_AtomRejectingZotero):
    """Local API behaviour: an itemKey filter returns extras alongside the ask.

    Verified against a live Zotero local API — requesting one key came back
    with that item plus three unrelated ones, so the response cannot be
    treated as the selection. The web API filters correctly, so the client-side
    filter is a no-op there.
    """

    def items(self, **kwargs):
        self.last_kwargs = kwargs
        if kwargs.get("format") == "bibtex":
            return b"@article{smith2020, title={Title}}"
        include = kwargs.get("include")
        requested = (kwargs.get("itemKey") or "").split(",")
        rows = [{"key": "UNRELATED", include: "<div>Noise, N. (1999).</div>"}]
        rows += [{"key": k, include: f"<div>Wanted {k}</div>"} for k in requested]
        rows.append({"key": "ALSONOISE", include: "<div>More, M. (1998).</div>"})
        return rows


def test_export_bibliography_item_keys_filters_out_api_extras(monkeypatch):
    """Only the requested keys are rendered, in the order asked for (#371)."""
    local = _LooseItemKeyZotero()
    monkeypatch.setattr(zotero_client, "get_zotero_client", lambda: local)

    out = synthesis.export_bibliography(
        item_keys=["BBBB2222", "AAAA1111"], ctx=DummyContext()
    )

    numbered = [ln for ln in out.splitlines() if ln[:2] in ("1.", "2.", "3.")]
    assert len(numbered) == 2, f"extras leaked into the export: {numbered}"
    assert "Wanted BBBB2222" in numbered[0]
    assert "Wanted AAAA1111" in numbered[1]
    assert "Noise" not in out and "More, M." not in out


def test_export_bibliography_library_wide_uses_top_level_items(monkeypatch):
    """Exporting the library skips attachments/notes, which render as blanks."""
    local = _AtomRejectingZotero()
    monkeypatch.setattr(zotero_client, "get_zotero_client", lambda: local)

    synthesis.export_bibliography(ctx=DummyContext())

    assert local.last_kwargs.get("include") == "bib"
