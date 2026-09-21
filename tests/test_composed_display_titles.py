"""Display titles Zotero *composes* rather than stores.

`item_display_title` resolved a title by reading a field. For four item types
Zotero stores no title at all and builds one in
`Zotero.Item.prototype.updateDisplayTitle`
(chrome/content/zotero/xpcom/data/item.js:903-1035); those items fell through
to the final `or "Untitled"`.

The material case is annotations — 461 in the reporting library, every one
rendering "## Untitled" through `zotero://items/{key}` even though
`annotationText` and `annotationComment` were present and correct. `case`,
`letter` and `interview` are conformance with the same Zotero function.

Every expected string here is Zotero's own, verbatim, including its
inconsistent casing ("Highlight annotation" but "Note Annotation"):

* chrome/locale/en-US/zotero/zotero.properties:378-385 (participants)
* chrome/locale/en-US/zotero/reader.ftl:20-27 (annotation type names)
"""

import pytest

from zotero_mcp.local_db import _ANNOTATION_TYPES
from zotero_mcp.utils import format_item_result, item_display_title


def _annotation(atype="highlight", text="", comment=""):
    return {"key": "ANNO0001", "itemType": "annotation", "annotationType": atype,
            "annotationText": text, "annotationComment": comment,
            "creators": [], "tags": []}


def _case(name="Marbury v. Madison", reporter="", court=""):
    data = {"key": "CASE0001", "itemType": "case", "creators": [], "tags": []}
    if name:
        data["caseName"] = name
    if reporter:
        data["reporter"] = reporter
    if court:
        data["court"] = court
    return data


class TestAnnotations:
    @pytest.mark.parametrize("atype", ["highlight", "underline"])
    def test_quoted_text(self, atype):
        """Only highlight and underline quote their text; Zotero's branch is
        `if (["highlight", "underline"].includes(this.annotationType))`."""
        assert item_display_title(_annotation(atype, text="Attention is all")) == (
            "“Attention is all”"
        )

    def test_text_then_comment_separated_by_one_space(self):
        got = item_display_title(_annotation("highlight", "Some passage", "my note"))
        assert got == "“Some passage” my note"

    @pytest.mark.parametrize("atype", ["text", "note"])
    def test_comment_only_types_are_not_quoted(self, atype):
        """A `text`/`note` annotation is its comment — it never gets quotes,
        because Zotero only quotes for highlight/underline. The reporting
        library's three type-6 items are exactly this shape."""
        assert item_display_title(_annotation(atype, comment="a remark")) == "a remark"

    @pytest.mark.parametrize("atype,expected", [
        ("note", "Note Annotation"),
        ("text", "Text Annotation"),
        ("image", "Image Annotation"),
        ("ink", "Ink Annotation"),
    ])
    def test_empty_annotation_falls_back_to_its_type_name(self, atype, expected):
        """Zotero's casing is inconsistent between these strings; mirror it
        rather than tidy it, so output matches the client verbatim."""
        assert item_display_title(_annotation(atype)) == expected

    @pytest.mark.parametrize("atype", ["highlight", "underline"])
    def test_an_empty_highlight_keeps_its_empty_quotes(self, atype):
        """Zotero closes the quotes *before* testing whether the title is
        empty (item.js:1013-1021), so a highlight with no text is `“”` — two
        characters, so `if (!title.length)` is false and it never reaches the
        type-name fallback the other types get.

        Mirrored deliberately rather than tidied: conformance with the client
        is the whole point of this change, and the case is unreachable in
        practice — 0 of the 445 highlights in the reporting library have empty
        text.
        """
        assert item_display_title(_annotation(atype)) == "“”"

    def test_text_is_capped_at_50_characters_with_an_ellipsis(self):
        text = "x" * 60
        got = item_display_title(_annotation("highlight", text=text))
        assert got == "“" + "x" * 50 + "…”"

    def test_comment_is_capped_independently(self):
        got = item_display_title(_annotation("highlight", "t", "y" * 60))
        assert got == "“t” " + "y" * 50 + "…"

    def test_exactly_50_characters_is_not_truncated(self):
        """`length > maxComponentLength`, not `>=`."""
        got = item_display_title(_annotation("highlight", text="z" * 50))
        assert got == "“" + "z" * 50 + "”"
        assert "…" not in got

    def test_angle_brackets_are_text_not_markup(self):
        title = item_display_title(_annotation(text="p < 0.05 and n > 30 in all"))
        assert title == "“p < 0.05 and n > 30 in all”"

    def test_line_breaks_do_not_reach_the_heading(self):
        assert item_display_title(_annotation(text="line one\nline two")) == "“line one line two”"


class TestAnnotationTypeMap:
    def test_type_6_is_text(self):
        """Zotero.Annotations.ANNOTATION_TYPE_TEXT = 6 (annotations.js:36),
        named 'text' in data/items.js:538. Without it type 6 mapped to "",
        which also renders as `[KEY] : ...` in tools/retrieval.py."""
        assert _ANNOTATION_TYPES[6] == "text"

    def test_the_five_older_types_are_unchanged(self):
        assert _ANNOTATION_TYPES[1] == "highlight"
        assert _ANNOTATION_TYPES[2] == "note"
        assert _ANNOTATION_TYPES[3] == "image"
        assert _ANNOTATION_TYPES[4] == "ink"
        assert _ANNOTATION_TYPES[5] == "underline"


# ---------------------------------------------------------------------------
# Cases — conformance; the reporting library holds none
# ---------------------------------------------------------------------------

class TestCases:
    def test_reporter_is_appended(self):
        assert item_display_title(_case(reporter="5 U.S. 137")) == (
            "Marbury v. Madison (5 U.S. 137)"
        )

    def test_court_is_used_when_there_is_no_reporter(self):
        assert item_display_title(_case(court="Supreme Court")) == (
            "Marbury v. Madison (Supreme Court)"
        )

    def test_reporter_wins_when_both_are_present(self):
        got = item_display_title(_case(reporter="5 U.S. 137", court="Supreme Court"))
        assert got == "Marbury v. Madison (5 U.S. 137)"

    def test_bare_name_when_neither_is_present(self):
        assert item_display_title(_case()) == "Marbury v. Madison"

    def test_the_sqlite_backend_shape_still_gets_its_qualifier(self):
        """The two backends disagree about *where* a case's name lives.

        `row_to_api_item` writes the hydrated value under the BASE key
        (`title`), while the web API returns `caseName`. Resolving `title` ->
        `caseName` therefore finds nothing on a SQL-backed row, and an
        implementation that only consulted the resolved key would drop the
        reporter for every case served by the SQLite backend.
        """
        data = {"key": "C", "itemType": "case", "title": "Marbury v. Madison",
                "reporter": "5 U.S. 137", "creators": [], "tags": []}
        assert item_display_title(data) == "Marbury v. Madison (5 U.S. 137)"


# ---------------------------------------------------------------------------
# Letters and interviews — conformance
# ---------------------------------------------------------------------------

class TestUnaffected:
    def test_ordinary_article(self):
        data = {"itemType": "journalArticle", "title": "Attention Is All You Need"}
        assert item_display_title(data) == "Attention Is All You Need"

    def test_base_field_types_still_resolve(self):
        """#452/#570 must not regress."""
        assert item_display_title(
            {"itemType": "statute", "nameOfAct": "Data Protection Act 2018"}
        ) == "Data Protection Act 2018"

    def test_attachment_filename_fallback(self):
        assert item_display_title(
            {"itemType": "attachment", "filename": "paper.pdf"}
        ) == "paper.pdf"

    def test_a_genuinely_empty_item_is_still_untitled(self):
        assert item_display_title({"itemType": "journalArticle"}) == "Untitled"


class TestRenderedOutput:
    def test_an_annotation_no_longer_renders_as_untitled(self):
        """The reported symptom, at the layer it was reported against."""
        item = {"key": "ZR495LQ2",
                "data": _annotation("highlight", "Sample 7: employees", "-if you ask")}
        rendered = "\n".join(format_item_result(item))
        assert "Untitled" not in rendered
        assert "Sample 7: employees" in rendered
