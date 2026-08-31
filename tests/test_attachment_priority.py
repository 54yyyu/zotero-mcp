"""Tests for configurable attachment priority (#378).

An item can hold several readable files — the publisher PDF, an HTML
snapshot, and (the case that prompted this) a Markdown copy the user
converted themselves with a better tool. The order those are tried in is
configurable, and the default must reproduce the historical PDF > HTML >
rest behaviour exactly so nobody's library changes under them on upgrade.
"""

from pathlib import Path

import pytest
from conftest import FakeZotero, extracted_doc

from zotero_mcp import client as client_module
from zotero_mcp.extract import (
    ATTACHMENT_CATEGORIES,
    DEFAULT_ATTACHMENT_PRIORITY,
    categorize_attachment,
    normalize_attachment_priority,
    pick_by_priority,
)
from zotero_mcp.local_db import LocalZoteroReader
from zotero_mcp.semantic_search import _attachment_priority_changed


class TestCategorizeAttachment:
    @pytest.mark.parametrize(
        "name,ctype,expected",
        [
            ("paper.pdf", "application/pdf", "pdf"),
            ("paper.pdf", None, "pdf"),
            ("no-extension", "application/pdf", "pdf"),
            ("snapshot.html", "text/html", "html"),
            ("snapshot.htm", None, "html"),
            ("page.html", "text/html; charset=utf-8", "html"),
            ("converted.md", None, "markdown"),
            ("converted.markdown", None, "markdown"),
            ("notes", "text/markdown", "markdown"),
            ("transcript.txt", "text/plain", "text"),
            ("captions.vtt", None, "text"),
            ("data.csv", None, "text"),
        ],
    )
    def test_known_kinds(self, name, ctype, expected):
        assert categorize_attachment(Path(name), ctype) == expected

    @pytest.mark.parametrize(
        "name,ctype",
        [
            ("book.epub", "application/epub+zip"),
            ("paper.docx", "application/vnd.openxmlformats-officedocument"
                            ".wordprocessingml.document"),
            ("talk.mp4", "video/mp4"),
        ],
    )
    def test_unreadable_kinds_are_uncategorized(self, name, ctype):
        """None, not "other" — callers decide whether to drop or sweep it."""
        assert categorize_attachment(Path(name), ctype) is None

    def test_never_returns_the_catch_all_label(self):
        for name in ("a.pdf", "a.html", "a.md", "a.txt", "a.bin"):
            assert categorize_attachment(Path(name), None) != "other"

    def test_markdown_is_distinct_from_text(self):
        """The distinction #378 turns on: both are read identically, but only
        a separate label lets markdown be preferred over the PDF."""
        assert categorize_attachment(Path("a.md"), None) == "markdown"
        assert categorize_attachment(Path("a.txt"), None) == "text"


class TestNormalizeAttachmentPriority:
    def test_none_and_empty_give_the_default(self):
        assert normalize_attachment_priority(None) == DEFAULT_ATTACHMENT_PRIORITY
        assert normalize_attachment_priority([]) == DEFAULT_ATTACHMENT_PRIORITY

    def test_case_and_whitespace_are_forgiven(self):
        assert normalize_attachment_priority([" PDF ", "Html"]) == ("pdf", "html")

    def test_duplicates_collapse(self):
        assert normalize_attachment_priority(["pdf", "pdf", "html"]) == ("pdf", "html")

    def test_unknown_entries_are_dropped_not_fatal(self, caplog):
        with caplog.at_level("WARNING", logger="zotero_mcp.extract"):
            assert normalize_attachment_priority(["pdf", "xml-ish", "html"]) == ("pdf", "html")
        assert "xml-ish" in caplog.text

    def test_all_unknown_falls_back_to_the_default(self, caplog):
        with caplog.at_level("WARNING", logger="zotero_mcp.extract"):
            assert normalize_attachment_priority(["nonsense"]) == DEFAULT_ATTACHMENT_PRIORITY

    def test_a_bare_string_is_accepted(self):
        assert normalize_attachment_priority("markdown") == ("markdown",)

    def test_every_documented_category_is_accepted(self):
        assert normalize_attachment_priority(list(ATTACHMENT_CATEGORIES)) == ATTACHMENT_CATEGORIES


class TestPickByPriority:
    # (category, size, value)
    PDF = ("pdf", 100, "paper.pdf")
    BIG_PDF = ("pdf", 900, "paper-full.pdf")
    HTML = ("html", 100, "snapshot.html")
    MD = ("markdown", 10, "converted.md")
    TXT = ("text", 50, "notes.txt")

    def test_default_prefers_pdf(self):
        picked = pick_by_priority([self.MD, self.HTML, self.PDF], DEFAULT_ATTACHMENT_PRIORITY)
        assert picked == "paper.pdf"

    def test_default_falls_to_html_then_the_rest(self):
        assert pick_by_priority([self.MD, self.HTML], DEFAULT_ATTACHMENT_PRIORITY) == "snapshot.html"
        assert pick_by_priority([self.MD], DEFAULT_ATTACHMENT_PRIORITY) == "converted.md"

    def test_default_sweeps_markdown_and_text_into_one_bucket(self):
        """Historical behaviour: neither is named, so the larger simply wins."""
        assert pick_by_priority([self.MD, self.TXT], DEFAULT_ATTACHMENT_PRIORITY) == "notes.txt"

    def test_markdown_first_beats_the_pdf(self):
        """The #378 headline case."""
        picked = pick_by_priority(
            [self.PDF, self.HTML, self.MD], ("markdown", "pdf", "html", "other")
        )
        assert picked == "converted.md"

    def test_largest_wins_within_a_category(self):
        picked = pick_by_priority([self.PDF, self.BIG_PDF], DEFAULT_ATTACHMENT_PRIORITY)
        assert picked == "paper-full.pdf"

    def test_a_category_absent_from_the_list_is_never_chosen(self):
        """Omitting "other" is how a caller opts out of everything else."""
        assert pick_by_priority([self.MD, self.TXT], ("pdf", "html")) is None

    def test_other_excludes_categories_named_later(self):
        """"other" must not swallow markdown when markdown is listed after it,
        or the explicit entry would be unreachable."""
        picked = pick_by_priority([self.MD, self.TXT], ("other", "markdown"))
        assert picked == "notes.txt"

    def test_pdf_only_priority_ignores_everything_else(self):
        assert pick_by_priority([self.MD, self.HTML], ("pdf",)) is None
        assert pick_by_priority([self.MD, self.PDF], ("pdf",)) == "paper.pdf"

    def test_no_candidates(self):
        assert pick_by_priority([], DEFAULT_ATTACHMENT_PRIORITY) is None


class _Reader(LocalZoteroReader):
    """LocalZoteroReader stub: no DB, attachments resolved from a dict."""

    def __init__(self, attachments, resolved, attachment_priority=None):
        self.db_path = "/dev/null"
        self._connection = None
        self.pdf_max_pages = 10
        self.attachment_priority = normalize_attachment_priority(attachment_priority)
        self._attachments = attachments
        self._resolved = resolved

    def _iter_parent_attachments(self, _parent_item_id: int):
        yield from self._attachments

    def _resolve_attachment_path(self, attachment_key, _zotero_path):
        return self._resolved.get(attachment_key)

    def _read_zotero_ft_cache(self, _attachment_key):
        return None

    def _extract_doc_from_file(self, file_path):
        return extracted_doc(f"text of {Path(file_path).name}")


@pytest.fixture
def pdf_and_markdown(tmp_path):
    """An item holding the publisher PDF and a user-converted Markdown copy."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4" + b"x" * 5000)
    md = tmp_path / "paper.md"
    md.write_text("# Paper\n\nclean converted text")
    return (
        [
            ("PDFKEY01", "storage:paper.pdf", "application/pdf"),
            ("MDKEY001", "storage:paper.md", "text/markdown"),
        ],
        {"PDFKEY01": pdf, "MDKEY001": md},
    )


class TestLocalReaderHonorsPriority:
    def test_default_still_reads_the_pdf(self, pdf_and_markdown):
        attachments, resolved = pdf_and_markdown
        reader = _Reader(attachments, resolved)
        text, source = reader._extract_fulltext_for_item(item_id=1)
        assert source == "pdf"
        assert "paper.pdf" in text

    def test_markdown_first_reads_the_markdown(self, pdf_and_markdown):
        """#265 made Markdown readable; this makes it preferred."""
        attachments, resolved = pdf_and_markdown
        reader = _Reader(
            attachments, resolved, attachment_priority=["markdown", "pdf", "html", "other"]
        )
        text, source = reader._extract_fulltext_for_item(item_id=1)
        assert source == "file"
        assert "paper.md" in text

    def test_markdown_first_still_reads_the_pdf_when_no_markdown_exists(self, tmp_path):
        pdf = tmp_path / "only.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        reader = _Reader(
            [("PDFKEY01", "storage:only.pdf", "application/pdf")],
            {"PDFKEY01": pdf},
            attachment_priority=["markdown", "pdf", "html", "other"],
        )
        _text, source = reader._extract_fulltext_for_item(item_id=1)
        assert source == "pdf"


def _attachment(key, filename, ctype, md5="abc123"):
    return {"key": key, "version": 1, "data": {
        "key": key, "itemType": "attachment", "contentType": ctype,
        "filename": filename, "title": filename, "md5": md5,
    }}


def _parent(key="PAR00001"):
    return {"key": key, "version": 1, "data": {
        "key": key, "itemType": "journalArticle", "title": "Parent Item",
    }}


class TestWebApiPathHonorsPriority:
    """``get_attachment_details`` is the Web API twin of the local chooser."""

    CHILDREN = [
        _attachment("PDF00001", "paper.pdf", "application/pdf"),
        _attachment("MD000001", "paper.md", "text/markdown"),
        _attachment("EPUB0001", "book.epub", "application/epub+zip"),
    ]

    def _zot(self):
        zot = FakeZotero()
        zot._children = {"PAR00001": list(self.CHILDREN)}
        return zot

    def test_default_prefers_the_pdf(self):
        details = client_module.get_attachment_details(
            self._zot(), _parent(), priority=DEFAULT_ATTACHMENT_PRIORITY
        )
        assert details.key == "PDF00001"

    def test_markdown_first_prefers_the_markdown(self):
        details = client_module.get_attachment_details(
            self._zot(), _parent(), priority=("markdown", "pdf", "html", "other")
        )
        assert details.key == "MD000001"

    def test_pdf_only_priority_never_returns_a_non_pdf(self):
        """What zotero_read_pdf_pages passes: it renders page ranges, so a
        markdown-first configuration must not hand it a .md file."""
        details = client_module.get_attachment_details(
            self._zot(), _parent(), priority=("pdf",)
        )
        assert details.key == "PDF00001"

    def test_pdf_only_priority_finds_nothing_when_there_is_no_pdf(self):
        zot = FakeZotero()
        zot._children = {"PAR00001": [_attachment("MD000001", "paper.md", "text/markdown")]}
        assert client_module.get_attachment_details(zot, _parent(), priority=("pdf",)) is None

    def test_unparseable_formats_stay_reachable_via_the_catch_all(self):
        """Unlike the local path, an EPUB must still be returned: the caller
        asks Zotero's server-side index for its text, which we cannot parse
        ourselves but Zotero has already indexed."""
        zot = FakeZotero()
        zot._children = {"PAR00001": [_attachment("EPUB0001", "book.epub", "application/epub+zip")]}
        details = client_module.get_attachment_details(
            zot, _parent(), priority=DEFAULT_ATTACHMENT_PRIORITY
        )
        assert details is not None and details.key == "EPUB0001"

    def test_an_attachment_key_bypasses_priority_entirely(self):
        """The documented short-circuit (#378): passing an attachment's own
        key reads exactly that file whatever the priority says."""
        attachment = _attachment("MD000001", "paper.md", "text/markdown")
        details = client_module.get_attachment_details(
            FakeZotero(), attachment, priority=("pdf",)
        )
        assert details.key == "MD000001"
        assert details.content_type == "text/markdown"


class TestPriorityChangeForcesReextraction:
    def test_a_changed_priority_is_detected(self):
        assert _attachment_priority_changed({"attachment_priority": "pdf,html,other"},
                                            "markdown,pdf,html,other")

    def test_an_unchanged_priority_is_not(self):
        assert not _attachment_priority_changed({"attachment_priority": "pdf,html,other"},
                                                "pdf,html,other")

    def test_a_legacy_document_without_the_tag_is_left_alone(self):
        """Documents indexed before this field existed must not all
        re-extract on the first run after upgrading."""
        assert not _attachment_priority_changed({}, "pdf,html,other")
        assert not _attachment_priority_changed({"has_fulltext": True}, "pdf,html,other")
