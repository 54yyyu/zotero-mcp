"""A note item can be read through the tools that claim to read items (#447).

Notes are first-class Zotero items with `itemType: "note"` and their text in
`data.note`. All three read paths refused to show it:

    zotero_get_item_metadata(TDM59NG2) -> "# Untitled" and nothing else
    zotero_get_notes(item_key=TDM59NG2) -> "No notes found for item TDM59NG2."
    zotero_get_item_fulltext(TDM59NG2)  -> "No suitable attachment found"

Between them there was no way to read a note whose key you already held, even
though pyzotero returns its full HTML body for the same key.
"""

import pytest

from zotero_mcp.client import format_item_metadata
from zotero_mcp.tools import annotations as annotation_tools
from zotero_mcp.tools import retrieval as retrieval_tools
from zotero_mcp.utils import note_title

NOTE_KEY = "TDM59NG2"
NOTE_HTML = (
    "<div class=\"zotero-note znv1\"><p>Reviewer objections to the 2019 model</p>"
    "<p>The identification strategy assumes no spillover between treated and "
    "control clinics, which the appendix does not defend.</p></div>"
)


def _note_item(key=NOTE_KEY, note=NOTE_HTML, parent=None):
    data = {
        "key": key,
        "itemType": "note",
        "note": note,
        "tags": [],
        "collections": ["HZJTPVXS"],
    }
    if parent:
        data["parentItem"] = parent
    return {"key": key, "data": data, "meta": {}}


class DummyContext:
    def info(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


class NoteZotero:
    """A library holding one standalone note."""

    def __init__(self, item=None):
        self.item_obj = item if item is not None else _note_item()

    def item(self, key, **kwargs):
        if key == self.item_obj["key"]:
            return self.item_obj
        raise Exception(f"Item not found: {key}")

    def children(self, key, start=0, limit=100, **kwargs):
        return []  # a note has none

    def items(self, start=0, limit=100, **kwargs):
        return [self.item_obj][start:start + limit]


# ---------------------------------------------------------------------------
# note_title
# ---------------------------------------------------------------------------

class TestNoteTitle:
    def test_uses_the_first_line_like_zotero_does(self):
        assert note_title(NOTE_HTML) == "Reviewer objections to the 2019 model"

    def test_empty_note_is_named_not_left_blank(self):
        assert note_title("") == "Untitled Note"
        assert note_title("<p></p>") == "Untitled Note"

    def test_long_first_line_is_elided(self):
        long_line = "x" * 200
        got = note_title(f"<p>{long_line}</p>", max_chars=80)
        assert len(got) <= 81
        assert got.endswith("…")


# ---------------------------------------------------------------------------
# zotero_get_item_metadata
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_note_is_not_rendered_as_untitled(self):
        result = format_item_metadata(_note_item())
        assert result.startswith("# Reviewer objections to the 2019 model")
        assert "# Untitled" not in result

    def test_note_body_is_included(self):
        result = format_item_metadata(_note_item())
        assert "## Note" in result
        assert "no spillover between treated and control clinics" in result
        # Stripped of markup, not dumped as raw HTML.
        assert "<p>" not in result

    def test_ordinary_items_are_unaffected(self):
        article = {
            "key": "ABC",
            "data": {
                "key": "ABC", "itemType": "journalArticle",
                "title": "A Real Title", "abstractNote": "An abstract.",
            },
            "meta": {},
        }
        result = format_item_metadata(article)
        assert result.startswith("# A Real Title")
        assert "## Note" not in result

    def test_untitled_non_note_is_still_untitled(self):
        bare = {"key": "X", "data": {"key": "X", "itemType": "journalArticle"}, "meta": {}}
        assert format_item_metadata(bare).startswith("# Untitled")


# ---------------------------------------------------------------------------
# zotero_get_notes
# ---------------------------------------------------------------------------

class TestGetNotes:
    def test_a_notes_own_key_returns_that_note(self, monkeypatch):
        """The reported call. `children(<note key>)` is empty because a note
        has no children -- which is not the same as the note not existing."""
        zot = NoteZotero()
        monkeypatch.setattr(annotation_tools._client, "get_zotero_client", lambda: zot)

        result = annotation_tools.get_notes(item_key=NOTE_KEY, ctx=DummyContext())

        assert "No notes found" not in result
        assert "no spillover between treated and control clinics" in result

    def test_a_parents_key_still_lists_its_child_notes(self, monkeypatch):
        """The ordinary path must not regress: a real parent still lists the
        notes filed under it, and the fallback never fires."""
        child = _note_item(key="CHILD001", parent="PARENT01")

        class ParentZotero(NoteZotero):
            def children(self, key, start=0, limit=100, **kwargs):
                return [child] if key == "PARENT01" else []

            def item(self, key, **kwargs):
                raise AssertionError("must not need a fallback lookup")

        monkeypatch.setattr(
            annotation_tools._client, "get_zotero_client", lambda: ParentZotero()
        )
        result = annotation_tools.get_notes(item_key="PARENT01", ctx=DummyContext())
        assert "no spillover" in result

    def test_a_non_note_item_with_no_notes_still_reports_none(self, monkeypatch):
        """The fallback must not turn "this paper has no notes" into the paper
        itself being printed as though it were one."""
        article = {
            "key": "PAPER001",
            "data": {"key": "PAPER001", "itemType": "journalArticle", "title": "T"},
            "meta": {},
        }

        class ArticleZotero(NoteZotero):
            def children(self, key, start=0, limit=100, **kwargs):
                return []

            def item(self, key, **kwargs):
                return article

        monkeypatch.setattr(
            annotation_tools._client, "get_zotero_client", lambda: ArticleZotero()
        )
        result = annotation_tools.get_notes(item_key="PAPER001", ctx=DummyContext())
        assert "No notes found for item PAPER001" in result

    def test_an_unknown_key_still_reports_none(self, monkeypatch):
        monkeypatch.setattr(
            annotation_tools._client, "get_zotero_client", lambda: NoteZotero()
        )
        result = annotation_tools.get_notes(item_key="NOSUCHKEY", ctx=DummyContext())
        assert "No notes found" in result


# ---------------------------------------------------------------------------
# zotero_get_item_fulltext
# ---------------------------------------------------------------------------

class TestFulltext:
    def test_note_fulltext_returns_the_note_body(self, monkeypatch):
        zot = NoteZotero()
        monkeypatch.setattr(retrieval_tools._client, "get_zotero_client", lambda: zot)
        monkeypatch.setattr(retrieval_tools._utils, "is_local_mode", lambda: False)

        result = retrieval_tools.get_item_fulltext(
            item_key=NOTE_KEY, ctx=DummyContext()
        )

        assert "No suitable attachment found" not in result
        assert "no spillover between treated and control clinics" in result

    def test_it_does_not_reach_for_an_attachment(self, monkeypatch):
        """A note has no attachment to find, so looking for one can only
        produce a misleading answer."""
        zot = NoteZotero()
        monkeypatch.setattr(retrieval_tools._client, "get_zotero_client", lambda: zot)
        monkeypatch.setattr(retrieval_tools._utils, "is_local_mode", lambda: False)

        def explode(*a, **k):
            raise AssertionError("attachment lookup must not run for a note")

        monkeypatch.setattr(retrieval_tools._client, "get_attachment_details", explode)

        retrieval_tools.get_item_fulltext(item_key=NOTE_KEY, ctx=DummyContext())


@pytest.mark.parametrize("empty", ["", "<p></p>"])
def test_an_empty_note_reads_as_empty_not_as_missing(empty):
    """An empty note is a real state and must be distinguishable from a note
    that could not be read at all."""
    result = format_item_metadata(_note_item(note=empty))
    assert "# Untitled Note" in result
    assert "**Type:** note" in result


class TestHtmlToText:
    """Block boundaries become line breaks; `clean_html` alone loses them."""

    def test_paragraphs_do_not_run_together(self):
        """A paragraph boundary is a blank line, so the text stays readable as
        markdown rather than becoming one block."""
        from zotero_mcp.utils import html_to_text

        assert html_to_text("<p>One</p><p>Two</p>") == "One\n\nTwo"

    def test_paragraph_runs_collapse_to_one_blank_line(self):
        from zotero_mcp.utils import html_to_text

        assert html_to_text("<div><p>One</p></div><div><p>Two</p></div>") == "One\n\nTwo"

    def test_line_breaks_are_preserved(self):
        from zotero_mcp.utils import html_to_text

        assert html_to_text("a<br/>b") == "a\nb"

    def test_list_items_are_separate_lines(self):
        from zotero_mcp.utils import html_to_text

        assert html_to_text("<ul><li>x</li><li>y</li></ul>") == "x\ny"

    def test_inline_markup_does_not_break_a_line(self):
        from zotero_mcp.utils import html_to_text

        assert html_to_text("<p>a <em>b</em> c</p>") == "a b c"

    def test_empty_input(self):
        from zotero_mcp.utils import html_to_text

        assert html_to_text("") == ""
        assert html_to_text(None) == ""
