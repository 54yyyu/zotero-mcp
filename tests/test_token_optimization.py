"""Tests for token usage optimization fixes (A through F)."""

import pytest
from typing import Literal

from conftest import DummyContext, FakeZotero, _FakeResponse
from zotero_mcp import server
from zotero_mcp.tools import _helpers


# ---------------------------------------------------------------------------
# Helpers: collection items fixture with children
# ---------------------------------------------------------------------------

def _make_parent(key, title, date="2024", collections=None, abstract=""):
    return {
        "key": key,
        "version": 1,
        "data": {
            "key": key,
            "itemType": "journalArticle",
            "title": title,
            "date": date,
            "creators": [{"firstName": "A", "lastName": "Author", "creatorType": "author"}],
            "abstractNote": abstract,
            "tags": [{"tag": "test"}],
            "collections": collections or ["COL1"],
            "DOI": "",
            "url": "",
        },
    }


def _make_attachment(key, parent_key, content_type="application/pdf", filename="paper.pdf", collections=None):
    return {
        "key": key,
        "version": 1,
        "data": {
            "key": key,
            "itemType": "attachment",
            "parentItem": parent_key,
            "contentType": content_type,
            "filename": filename,
            "title": filename,
            "collections": collections or ["COL1"],
        },
    }


def _make_note(key, parent_key, text="A note", collections=None):
    return {
        "key": key,
        "version": 1,
        "data": {
            "key": key,
            "itemType": "note",
            "parentItem": parent_key,
            "note": f"<p>{text}</p>",
            "collections": collections or ["COL1"],
        },
    }


class CollectionFakeZotero(FakeZotero):
    """Stub that returns parents + children from collection_items."""

    def __init__(self):
        super().__init__()
        self._all_items = []

    def collection(self, key, **kwargs):
        return {"data": {"name": "Test Collection"}}

    def collection_items(self, key, **kwargs):
        return [it for it in self._all_items
                if key in it.get("data", {}).get("collections", [])]


@pytest.fixture
def coll_zot():
    zot = CollectionFakeZotero()
    zot._all_items = [
        _make_parent("P1", "Paper One", abstract="Abstract about depression treatment"),
        _make_parent("P2", "Paper Two", abstract="Abstract about mindfulness"),
        _make_attachment("A1", "P1", "application/pdf", "paper1.pdf"),
        _make_attachment("A2", "P1", "text/html", "snapshot.html"),
        _make_note("N1", "P1", "My note on paper one"),
        _make_attachment("A3", "P2", "application/pdf", "paper2.pdf"),
    ]
    return zot


@pytest.fixture
def dummy_ctx():
    return DummyContext()


# ---------------------------------------------------------------------------
# Test Fix C: detail parameter
# ---------------------------------------------------------------------------

class TestDetailParameter:
    def test_keys_only_minimal(self, monkeypatch, coll_zot, dummy_ctx):
        """keys_only returns just key | title (date) [flags]."""
        monkeypatch.setattr(server, "get_zotero_client", lambda: coll_zot)
        from zotero_mcp.tools.retrieval import get_collection_items
        monkeypatch.setattr("zotero_mcp.tools.retrieval._client.get_zotero_client", lambda: coll_zot)

        result = get_collection_items(collection_key="COL1", detail="keys_only", ctx=dummy_ctx)

        assert "`P1`" in result
        assert "`P2`" in result
        assert "Paper One" in result
        assert "[PDF" in result  # P1 has a PDF
        assert "Notes" in result  # P1 has a note
        # Should NOT contain abstract
        assert "depression treatment" not in result

    def test_summary_no_abstract(self, monkeypatch, coll_zot, dummy_ctx):
        """summary (default) omits abstracts."""
        monkeypatch.setattr("zotero_mcp.tools.retrieval._client.get_zotero_client", lambda: coll_zot)
        from zotero_mcp.tools.retrieval import get_collection_items

        result = get_collection_items(collection_key="COL1", detail="summary", ctx=dummy_ctx)

        assert "Paper One" in result
        assert "Paper Two" in result
        # Abstract should NOT be present
        assert "depression treatment" not in result
        assert "mindfulness" not in result

    def test_full_includes_abstract(self, monkeypatch, coll_zot, dummy_ctx):
        """full mode includes abstracts."""
        monkeypatch.setattr("zotero_mcp.tools.retrieval._client.get_zotero_client", lambda: coll_zot)
        from zotero_mcp.tools.retrieval import get_collection_items

        result = get_collection_items(collection_key="COL1", detail="full", ctx=dummy_ctx)

        assert "Paper One" in result
        assert "depression treatment" in result

    def test_default_is_summary(self, monkeypatch, coll_zot, dummy_ctx):
        """Default detail level should be summary (no abstracts)."""
        monkeypatch.setattr("zotero_mcp.tools.retrieval._client.get_zotero_client", lambda: coll_zot)
        from zotero_mcp.tools.retrieval import get_collection_items

        result = get_collection_items(collection_key="COL1", ctx=dummy_ctx)

        # Should have titles but not abstracts
        assert "Paper One" in result
        assert "depression treatment" not in result


# ---------------------------------------------------------------------------
# Test Fix D: Attachment summary
# ---------------------------------------------------------------------------

class TestAttachmentSummary:
    def test_pdf_indicator_in_keys_only(self, monkeypatch, coll_zot, dummy_ctx):
        """keys_only mode shows [PDF] flag when item has a PDF."""
        monkeypatch.setattr("zotero_mcp.tools.retrieval._client.get_zotero_client", lambda: coll_zot)
        from zotero_mcp.tools.retrieval import get_collection_items

        result = get_collection_items(collection_key="COL1", detail="keys_only", ctx=dummy_ctx)

        # P1 has a PDF attachment
        lines = result.split("\n")
        p1_line = [l for l in lines if "P1" in l][0]
        assert "PDF" in p1_line

    def test_notes_indicator(self, monkeypatch, coll_zot, dummy_ctx):
        """keys_only mode shows [Notes] flag when item has notes."""
        monkeypatch.setattr("zotero_mcp.tools.retrieval._client.get_zotero_client", lambda: coll_zot)
        from zotero_mcp.tools.retrieval import get_collection_items

        result = get_collection_items(collection_key="COL1", detail="keys_only", ctx=dummy_ctx)

        lines = result.split("\n")
        p1_line = [l for l in lines if "P1" in l][0]
        assert "Notes" in p1_line

    def test_attachment_info_in_summary(self, monkeypatch, coll_zot, dummy_ctx):
        """summary mode includes attachment info via extra_fields."""
        monkeypatch.setattr("zotero_mcp.tools.retrieval._client.get_zotero_client", lambda: coll_zot)
        from zotero_mcp.tools.retrieval import get_collection_items

        result = get_collection_items(collection_key="COL1", detail="summary", ctx=dummy_ctx)

        # P1 has PDF + HTML attachment + note
        assert "PDF" in result
        assert "has notes" in result


# ---------------------------------------------------------------------------
# Test Fix E: Batch get_item_children
# ---------------------------------------------------------------------------

class TestBatchChildren:
    def test_multiple_items_grouped(self, monkeypatch, dummy_ctx):
        """Batch children returns results grouped by parent item."""

        class BatchZotero(FakeZotero):
            def items(self, **kwargs):
                item_key = kwargs.get("itemKey", "")
                keys = item_key.split(",") if item_key else []
                return [
                    {"key": k, "data": {"title": f"Paper {k}", "itemType": "journalArticle"}}
                    for k in keys
                ]

            def children(self, key, **kwargs):
                if key == "K1":
                    return [{"key": "C1", "data": {"itemType": "attachment", "contentType": "application/pdf", "filename": "paper.pdf", "linkMode": "imported_file"}}]
                return []

        zot = BatchZotero()
        monkeypatch.setattr("zotero_mcp.tools.retrieval._client.get_zotero_client", lambda: zot)
        from zotero_mcp.tools.retrieval import get_items_children

        result = get_items_children(item_keys=["K1", "K2"], ctx=dummy_ctx)

        assert "Paper K1" in result
        assert "Paper K2" in result
        assert "paper.pdf" in result
        assert "No child items" in result  # K2 has none

    def test_json_string_input(self, monkeypatch, dummy_ctx):
        """Accepts JSON string input for item_keys."""

        class SimpleZotero(FakeZotero):
            def items(self, **kwargs):
                return [{"key": "X1", "data": {"title": "Test", "itemType": "journalArticle"}}]
            def children(self, key, **kwargs):
                return []

        zot = SimpleZotero()
        monkeypatch.setattr("zotero_mcp.tools.retrieval._client.get_zotero_client", lambda: zot)
        from zotero_mcp.tools.retrieval import get_items_children

        result = get_items_children(item_keys='["X1"]', ctx=dummy_ctx)
        assert "Test" in result


# ---------------------------------------------------------------------------
# Test Fix F: Token estimation
# ---------------------------------------------------------------------------

class TestTokenEstimation:
    def test_warning_for_large_response(self):
        """Large text gets a size warning prepended."""
        large_text = "x" * 25000  # ~6K tokens
        result = _helpers._prepend_size_warning(large_text, "Try a lighter query.")
        assert result.startswith("*Response size:")
        assert "Try a lighter query." in result
        assert large_text in result

    def test_no_warning_for_small_response(self):
        """Small text is returned unchanged."""
        small_text = "Hello world"
        result = _helpers._prepend_size_warning(small_text)
        assert result == small_text

    def test_estimate_tokens(self):
        """Token estimation is roughly 4 chars per token."""
        assert _helpers._estimate_tokens("abcd") == 1
        assert _helpers._estimate_tokens("a" * 4000) == 1000


# ---------------------------------------------------------------------------
# Test Fix B: Multi-word collection search
# ---------------------------------------------------------------------------

class TestMultiWordSearch:
    def test_multi_word_matches(self, monkeypatch, dummy_ctx):
        """Multi-word query 'KCL mindfulness' matches 'KCL - Mindfulness'."""

        class SearchZotero(FakeZotero):
            def collections(self, **kwargs):
                return [
                    {"key": "C1", "data": {"name": "KCL - Mindfulness"}},
                    {"key": "C2", "data": {"name": "KCL - Depression"}},
                    {"key": "C3", "data": {"name": "Other Collection"}},
                ]

        zot = SearchZotero()
        monkeypatch.setattr("zotero_mcp.tools.write._client.get_zotero_client", lambda: zot)
        from zotero_mcp.tools.write import search_collections

        result = search_collections(query="KCL mindfulness", ctx=dummy_ctx)

        assert "KCL - Mindfulness" in result
        assert "KCL - Depression" not in result  # doesn't have "mindfulness"
        assert "Other Collection" not in result

    def test_single_word_backward_compatible(self, monkeypatch, dummy_ctx):
        """Single-word query still works as before."""

        class SearchZotero(FakeZotero):
            def collections(self, **kwargs):
                return [
                    {"key": "C1", "data": {"name": "KCL - Mindfulness"}},
                    {"key": "C2", "data": {"name": "Mindfulness Research"}},
                ]

        zot = SearchZotero()
        monkeypatch.setattr("zotero_mcp.tools.write._client.get_zotero_client", lambda: zot)
        from zotero_mcp.tools.write import search_collections

        result = search_collections(query="mindfulness", ctx=dummy_ctx)

        assert "KCL - Mindfulness" in result
        assert "Mindfulness Research" in result


# ---------------------------------------------------------------------------
# Additional edge case and error handling tests
# ---------------------------------------------------------------------------

class TestTokenBoundary:
    """Boundary tests for the 5K token (~20K char) threshold."""

    def test_just_below_threshold_no_warning(self):
        """19999 chars (~4999 tokens) should NOT trigger warning."""
        text = "x" * 19999
        result = _helpers._prepend_size_warning(text, "hint")
        assert result == text  # unchanged

    def test_at_threshold_triggers_warning(self):
        """20000 chars (5000 tokens) should trigger warning."""
        text = "x" * 20000
        result = _helpers._prepend_size_warning(text, "hint")
        assert result.startswith("*Response size:")


class TestCollectionItemsEdgeCases:
    """Edge cases for get_collection_items."""

    def test_empty_collection(self, monkeypatch, dummy_ctx):
        """Empty collection returns a clear message for all detail modes."""

        class EmptyZotero(FakeZotero):
            def collection(self, key, **kwargs):
                return {"data": {"name": "Empty Collection"}}
            def collection_items(self, key, **kwargs):
                return []

        zot = EmptyZotero()
        monkeypatch.setattr("zotero_mcp.tools.retrieval._client.get_zotero_client", lambda: zot)
        from zotero_mcp.tools.retrieval import get_collection_items

        for detail in ["keys_only", "summary", "full"]:
            result = get_collection_items(collection_key="COL1", detail=detail, ctx=dummy_ctx)
            assert "No items found" in result

    def test_truncation_message(self, monkeypatch, coll_zot, dummy_ctx):
        """When limit < total items, truncation message appears."""
        monkeypatch.setattr("zotero_mcp.tools.retrieval._client.get_zotero_client", lambda: coll_zot)
        from zotero_mcp.tools.retrieval import get_collection_items

        result = get_collection_items(collection_key="COL1", detail="summary", limit=1, ctx=dummy_ctx)
        assert "Showing 1 of 2 items" in result


class TestBatchChildrenEdgeCases:
    """Error handling for get_items_children."""

    def test_bad_key_doesnt_abort_batch(self, monkeypatch, dummy_ctx):
        """One bad key doesn't prevent other keys from being processed."""

        class ErrorZotero(FakeZotero):
            def items(self, **kwargs):
                item_key = kwargs.get("itemKey", "")
                keys = item_key.split(",") if item_key else []
                return [
                    {"key": k, "data": {"title": f"Paper {k}", "itemType": "journalArticle"}}
                    for k in keys if k != "BAD"
                ]
            def children(self, key, **kwargs):
                if key == "BAD":
                    raise Exception("Item not found")
                return [{"key": "C1", "data": {"itemType": "note", "note": "<p>A note</p>"}}]

        zot = ErrorZotero()
        monkeypatch.setattr("zotero_mcp.tools.retrieval._client.get_zotero_client", lambda: zot)
        from zotero_mcp.tools.retrieval import get_items_children

        result = get_items_children(item_keys=["GOOD", "BAD"], ctx=dummy_ctx)

        assert "Paper GOOD" in result
        assert "Error fetching children" in result  # BAD key error
        assert "A note" in result  # GOOD key still processed

    def test_empty_keys_error(self, monkeypatch, dummy_ctx):
        """Empty keys returns clear error."""
        monkeypatch.setattr("zotero_mcp.tools.retrieval._client.get_zotero_client", lambda: FakeZotero())
        from zotero_mcp.tools.retrieval import get_items_children

        result = get_items_children(item_keys=[], ctx=dummy_ctx)
        assert "No item keys provided" in result

    def test_comma_separated_input(self, monkeypatch, dummy_ctx):
        """Comma-separated string input is normalized correctly."""

        class SimpleZotero(FakeZotero):
            def items(self, **kwargs):
                item_key = kwargs.get("itemKey", "")
                keys = item_key.split(",") if item_key else []
                return [
                    {"key": k, "data": {"title": f"Paper {k}", "itemType": "journalArticle"}}
                    for k in keys
                ]
            def children(self, key, **kwargs):
                return []

        zot = SimpleZotero()
        monkeypatch.setattr("zotero_mcp.tools.retrieval._client.get_zotero_client", lambda: zot)
        from zotero_mcp.tools.retrieval import get_items_children

        result = get_items_children(item_keys="K1,K2,K3", ctx=dummy_ctx)
        assert "Paper K1" in result
        assert "Paper K2" in result
        assert "Paper K3" in result


class TestBuildAttachmentExtra:
    """Direct unit tests for _build_attachment_extra."""

    def test_none_input(self):
        from zotero_mcp.tools.retrieval import _build_attachment_extra
        assert _build_attachment_extra(None) is None

    def test_empty_dict(self):
        from zotero_mcp.tools.retrieval import _build_attachment_extra
        assert _build_attachment_extra({}) is None

    def test_pdf_only(self):
        from zotero_mcp.tools.retrieval import _build_attachment_extra
        result = _build_attachment_extra({"has_pdf": True, "attachment_count": 1, "has_notes": False})
        assert result is not None
        assert "PDF" in result["Attachments"]

    def test_pluralization(self):
        from zotero_mcp.tools.retrieval import _build_attachment_extra
        result1 = _build_attachment_extra({"has_pdf": False, "attachment_count": 1, "has_notes": False})
        result2 = _build_attachment_extra({"has_pdf": False, "attachment_count": 2, "has_notes": False})
        assert "1 attachment" in result1["Attachments"]
        assert "2 attachments" in result2["Attachments"]


# ---------------------------------------------------------------------------
# Race condition: unknown / not-yet-propagated collection key
# ---------------------------------------------------------------------------

class RacyFakeZotero(FakeZotero):
    """Simulates the Zotero web API's behavior for an unknown or
    not-yet-propagated collection key: ``collection()`` 404s, but
    ``collection_items()`` returns unrelated library-wide items instead
    of erroring. This is the shape that produced the bug in production."""

    def __init__(self, library_items):
        super().__init__()
        self._library_items = library_items

    def collection(self, key, **kwargs):
        raise Exception(f"Code: 404 — Not found: collection {key}")

    def collection_items(self, key, *args, **kwargs):
        return self._library_items


class TestCollectionItemsRace:
    def test_unknown_collection_returns_error_not_library_items(
        self, monkeypatch, dummy_ctx
    ):
        """When ``zot.collection(key)`` fails, the tool must return a
        clear error instead of falling through to ``collection_items()``,
        which may return library-wide items for an unknown key."""
        library = [
            _make_parent("X1", "Unrelated Paper One", collections=["OTHER"]),
            _make_parent("X2", "Unrelated Paper Two", collections=["OTHER"]),
        ]
        zot = RacyFakeZotero(library_items=library)
        monkeypatch.setattr(
            "zotero_mcp.tools.retrieval._client.get_zotero_client", lambda: zot
        )
        from zotero_mcp.tools.retrieval import get_collection_items

        result = get_collection_items(
            collection_key="FRESHKEY", detail="keys_only", ctx=dummy_ctx
        )

        assert "not found" in result.lower() or "not yet accessible" in result.lower()
        assert "FRESHKEY" in result
        assert "Unrelated Paper One" not in result
        assert "Unrelated Paper Two" not in result
        assert "X1" not in result
        assert "X2" not in result
