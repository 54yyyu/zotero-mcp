"""Regression tests for #448: a page-capped read must say it was capped.

``zotero_get_item_fulltext`` stops at ``fulltext_display_max_pages`` — 10 by
default. The text it returned for the first ten pages of a 442-page book was
byte-for-byte indistinguishable from the text it returned for a complete
three-page note: same ``## Full Text`` heading, no marker anywhere. An agent
summarizes and cites the excerpt as if it had read the paper.

The parser already reports ``page_count`` and ``truncated`` on every
extraction; these tests pin that they reach the caller and get rendered.
"""

from pathlib import Path

from conftest import DummyContext, FakeZotero

import zotero_mcp.client as zotero_client
import zotero_mcp.local_db as local_db
import zotero_mcp.utils as zotero_utils
from zotero_mcp.extract import ExtractedDoc
from zotero_mcp.local_db import FulltextExtraction, LocalZoteroReader
from zotero_mcp.tools import retrieval

TARGET = Path("/nonexistent/book.pdf")


def _doc(text="body text", *, page_count=442, truncated=True, source="pdf"):
    """An ExtractedDoc shaped like what the parser returns for a capped read."""
    return ExtractedDoc(
        text=text,
        pages=(text,),
        page_numbers=(0,),
        page_count=page_count,
        source=source,
        truncated=truncated,
    )


class _Reader(LocalZoteroReader):
    """LocalZoteroReader stub with no DB and one resolvable attachment."""

    def __init__(self, *, target=TARGET, cached=None):
        self.db_path = "/dev/null"
        self._connection = None
        self.pdf_max_pages = 10
        self.pdf_timeout = 30
        self.fulltext_cache_enabled = False
        self._target = target
        self._cached = cached

    def _resolve_extraction_target(self, item_id):
        return (self._target, "ATT1") if self._target else None

    def _iter_parent_attachments(self, item_id):
        yield ("ATT1", "storage:book.pdf", "application/pdf")

    def _read_zotero_ft_cache(self, attachment_key):
        return self._cached


# ---------------------------------------------------------------------------
# extract_fulltext_detailed — carrying the parser's page bookkeeping up
# ---------------------------------------------------------------------------


def test_capped_read_carries_page_count_and_truncation(monkeypatch):
    monkeypatch.setattr(local_db, "extract_file", lambda p, **kw: _doc())

    extraction = _Reader().extract_fulltext_detailed(item_id=1)

    assert extraction.source == "pdf"
    assert extraction.text == "body text"
    assert extraction.page_count == 442
    assert extraction.truncated is True


def test_complete_read_is_not_marked_truncated(monkeypatch):
    monkeypatch.setattr(
        local_db, "extract_file", lambda p, **kw: _doc(page_count=3, truncated=False)
    )

    extraction = _Reader().extract_fulltext_detailed(item_id=1)

    assert extraction.truncated is False
    assert retrieval._fulltext_section_heading(False, 3, 10, "ABCD1234") == (
        "## Full Text",
        "",
    )


def test_zotero_ft_cache_fallback_reports_no_page_limits(monkeypatch):
    """Zotero's own cache is flat text with no page bookkeeping, and is not
    page-capped — it must never be reported as truncated."""
    monkeypatch.setattr(local_db, "extract_file", lambda p, **kw: None)

    extraction = _Reader(cached="whole document from Zotero").extract_fulltext_detailed(1)

    assert extraction.source == "zotero-cache"
    assert extraction.page_count is None
    assert extraction.truncated is False


def test_transient_cache_hit_still_returns_an_extraction(monkeypatch):
    """A cache hit short-circuits before the parser, so it carries text and
    source only — and must not crash on the missing page fields."""
    reader = _Reader()
    reader._cache_lookup = lambda target, key: ("cached text", "pdf")
    monkeypatch.setattr(local_db, "extract_file", lambda p, **kw: _doc())

    extraction = reader.extract_fulltext_detailed(item_id=1)

    assert (extraction.text, extraction.source) == ("cached text", "pdf")
    assert extraction.truncated is False


def test_nothing_readable_returns_none(monkeypatch):
    monkeypatch.setattr(local_db, "extract_file", lambda p, **kw: None)

    assert _Reader().extract_fulltext_detailed(item_id=1) is None


def test_indexing_paths_still_get_a_text_source_pair(monkeypatch):
    """The semantic indexer and get_items_with_text unpack a 2-tuple."""
    monkeypatch.setattr(local_db, "extract_file", lambda p, **kw: _doc())

    text, source = _Reader()._extract_fulltext_for_item(item_id=1)

    assert (text, source) == ("body text", "pdf")


# ---------------------------------------------------------------------------
# _fulltext_section_heading — what the caller is told
# ---------------------------------------------------------------------------


def test_heading_names_the_range_and_how_to_read_on():
    heading, notice = retrieval._fulltext_section_heading(True, 442, 10, "ABCD1234")

    assert heading == "## Full Text (pages 1-10 of 442 — TRUNCATED)"
    assert "442" in notice
    assert "fulltext_display_max_pages" in notice
    # Points at the next unread page, not at page 1 again.
    assert "start_page=11" in notice


def test_heading_is_unchanged_when_nothing_was_cut():
    assert retrieval._fulltext_section_heading(False, 3, 10, "ABCD1234") == (
        "## Full Text",
        "",
    )


# ---------------------------------------------------------------------------
# get_item_fulltext — end to end, both capped paths
# ---------------------------------------------------------------------------


class _FakeConfig:
    class semantic_search:  # noqa: N801 — mirrors the config object's shape
        class extraction:
            fulltext_display_max_pages = 10
            attachment_priority = ("pdf", "html", "other")

    def resolve_zotero_db_path(self):
        return "/dev/null"


def _install_local_mode(monkeypatch, extraction):
    """Point get_item_fulltext's local branch at ``extraction``."""

    class _FakeReader:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_item_by_key(self, key):
            return type("I", (), {"item_id": 1})()

        def extract_fulltext_detailed(self, item_id):
            return extraction

    monkeypatch.setattr(zotero_utils, "is_local_mode", lambda: True)
    monkeypatch.setattr(zotero_client, "get_zotero_client", FakeZotero)
    monkeypatch.setattr(retrieval, "load_config", lambda: _FakeConfig())
    monkeypatch.setattr(local_db, "LocalZoteroReader", _FakeReader)


def test_truncated_local_read_says_so_in_the_output(monkeypatch):
    _install_local_mode(
        monkeypatch,
        FulltextExtraction("first ten pages", "pdf", page_count=442, truncated=True),
    )

    out = retrieval.get_item_fulltext(item_key="ABCD1234", ctx=DummyContext())

    assert "## Full Text (pages 1-10 of 442 — TRUNCATED)" in out
    assert "zotero_read_pdf_pages(item_key='ABCD1234', start_page=11)" in out
    assert "first ten pages" in out


def test_complete_local_read_keeps_the_plain_heading(monkeypatch):
    _install_local_mode(
        monkeypatch,
        FulltextExtraction("the whole thing", "pdf", page_count=4, truncated=False),
    )

    out = retrieval.get_item_fulltext(item_key="ABCD1234", ctx=DummyContext())

    assert "## Full Text\n\nthe whole thing" in out
    assert "TRUNCATED" not in out
    assert "zotero_read_pdf_pages" not in out


def test_download_fallback_path_reports_truncation_too(monkeypatch):
    """The last-resort download+parse path applies the same cap, so it must
    report the same way rather than silently returning an excerpt."""
    monkeypatch.setattr(zotero_utils, "is_local_mode", lambda: False)
    monkeypatch.setattr(zotero_client, "get_zotero_client", FakeZotero)
    monkeypatch.setattr(retrieval, "load_config", lambda: _FakeConfig())
    monkeypatch.setattr(
        retrieval._client,
        "get_attachment_details",
        lambda zot, item: type("A", (), {"key": "ATT1", "content_type": "application/pdf", "filename": "p.pdf"})(),
    )
    # Zotero's server-side index has nothing, so we fall through to download.
    monkeypatch.setattr(
        FakeZotero, "fulltext_item", lambda self, key: {}, raising=False
    )

    _path = type("P", (), {"exists": lambda self: True, "name": "p.pdf"})()
    monkeypatch.setattr(
        retrieval._client,
        "download_attachment_file",
        lambda *a, **kw: type("D", (), {"path": _path, "source": "web", "errors": []})(),
    )
    monkeypatch.setattr(retrieval, "extract_file", lambda p, **kw: _doc(page_count=120))

    out = retrieval.get_item_fulltext(item_key="ABCD1234", ctx=DummyContext())

    assert "## Full Text (pages 1-10 of 120 — TRUNCATED)" in out
    assert "start_page=11" in out
