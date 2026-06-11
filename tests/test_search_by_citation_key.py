"""Tests for the search_by_citation_key tool and helper functions."""

from unittest.mock import patch

from conftest import DummyContext, FakeZotero

# The module reference that search.py uses for client calls.
# Patching this directly avoids module-resolution issues across Python versions.
import zotero_mcp.tools.search as _search_mod
from zotero_mcp.server import (
    _extra_has_citekey,
    search_by_citation_key,
)

# ---------------------------------------------------------------------------
# _extra_has_citekey unit tests
# ---------------------------------------------------------------------------

class TestExtraHasCitekey:
    def test_standard_format(self):
        assert _extra_has_citekey("Citation Key: Smith2024", "Smith2024") is True

    def test_lowercase_variant(self):
        assert _extra_has_citekey("citationkey: Smith2024", "Smith2024") is True

    def test_wrong_key(self):
        assert _extra_has_citekey("Citation Key: Jones2023", "Smith2024") is False

    def test_empty_extra(self):
        assert _extra_has_citekey("", "Smith2024") is False

    def test_multiline_extra_key_on_second_line(self):
        extra = "DOI: 10.1234/example\nCitation Key: Smith2024\nsome other line"
        assert _extra_has_citekey(extra, "Smith2024") is True

    def test_partial_match_rejected(self):
        # "Smith2024x" should not match "Smith2024"
        assert _extra_has_citekey("Citation Key: Smith2024x", "Smith2024") is False


# ---------------------------------------------------------------------------
# Helpers for building fake items
# ---------------------------------------------------------------------------

def _make_item(key="ABC123", title="Test Paper", extra="", citekey=None, **kwargs):
    """Build a minimal Zotero item dict."""
    data = {
        "title": title,
        "itemType": "journalArticle",
        "date": "2024",
        "creators": [{"creatorType": "author", "firstName": "John", "lastName": "Smith"}],
        "extra": extra,
        "tags": [],
        "abstractNote": "",
        "DOI": "",
    }
    data.update(kwargs)
    if citekey and "Citation Key" not in extra:
        data["extra"] = f"Citation Key: {citekey}"
    return {"key": key, "version": 1, "data": data}


class _CitekeyFakeZotero(FakeZotero):
    """FakeZotero with add_parameters support for citation-key tests."""

    def __init__(self):
        super().__init__()
        self._params = {}

    def add_parameters(self, **kwargs):
        self._params.update(kwargs)


# ---------------------------------------------------------------------------
# search_by_citation_key – web/API mode (Strategy B)
# ---------------------------------------------------------------------------

class TestSearchByCitationKeyWebMode:
    """Tests where BBT is not available (non-local mode)."""

    def test_found_via_extra_field(self, monkeypatch):
        fake = _CitekeyFakeZotero()
        fake._items = [
            _make_item(key="ABC123", title="Deep Learning", citekey="Smith2024"),
        ]
        monkeypatch.setattr(_search_mod._utils, "is_local_mode", lambda: False)
        monkeypatch.setattr(_search_mod._client, "get_zotero_client", lambda: fake)

        result = search_by_citation_key("Smith2024", ctx=DummyContext())

        assert "Citation Key: Smith2024" in result
        assert "Deep Learning" in result
        assert "ABC123" in result

    def test_found_via_native_citation_key_field(self, monkeypatch):
        fake = _CitekeyFakeZotero()
        fake._items = [
            _make_item(
                key="ABC123",
                title="Deep Learning",
                citationKey="Smith2024",
            ),
        ]
        monkeypatch.setattr(_search_mod._utils, "is_local_mode", lambda: False)
        monkeypatch.setattr(_search_mod._client, "get_zotero_client", lambda: fake)

        result = search_by_citation_key("Smith2024", ctx=DummyContext())

        assert "Citation Key: Smith2024" in result
        assert "Deep Learning" in result
        assert "ABC123" in result

    def test_no_match(self, monkeypatch):
        fake = _CitekeyFakeZotero()
        fake._items = [
            _make_item(key="XYZ999", title="Other Paper", citekey="Jones2023"),
        ]
        monkeypatch.setattr(_search_mod._utils, "is_local_mode", lambda: False)
        monkeypatch.setattr(_search_mod._client, "get_zotero_client", lambda: fake)

        result = search_by_citation_key("Smith2024", ctx=DummyContext())

        assert "No item found with citation key: 'Smith2024'" in result


# ---------------------------------------------------------------------------
# search_by_citation_key – local mode
# ---------------------------------------------------------------------------

class TestSearchByCitationKeyLocalMode:
    """In local mode the tool now uses the same Extra-field path as web mode.

    The Better BibTeX ``item.search`` JSON-RPC method was removed in #293 —
    it always returned -32601 Method not found, and the exception handler
    silently fell through to the Extra-field path, so the BBT branch only
    added noise without ever succeeding.
    """

    def test_local_mode_finds_via_extra_field(self, monkeypatch):
        fake = _CitekeyFakeZotero()
        fake._items = [
            _make_item(key="ABC123", title="Deep Learning", citekey="Smith2024"),
        ]
        monkeypatch.setattr(_search_mod._utils, "is_local_mode", lambda: True)
        monkeypatch.setattr(_search_mod._client, "get_zotero_client", lambda: fake)

        # BBT should NOT be invoked from this code path anymore.
        with patch(
            "zotero_mcp.better_bibtex_client.ZoteroBetterBibTexAPI",
        ) as MockBBT:
            result = search_by_citation_key("Smith2024", ctx=DummyContext())
            assert MockBBT.call_count == 0, (
                "search_by_citation_key must not call BBT — item.search is "
                "broken in all reported BBT versions (#293)."
            )

        assert "Citation Key: Smith2024" in result
        assert "Deep Learning" in result
        assert "ABC123" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestSearchByCitationKeyEdgeCases:
    def test_empty_citekey(self):
        result = search_by_citation_key("  ", ctx=DummyContext())
        assert "Error: Citation key cannot be empty" in result

    def test_whitespace_stripped(self, monkeypatch):
        fake = _CitekeyFakeZotero()
        fake._items = [
            _make_item(key="ABC123", title="Stripped Key", citekey="Smith2024"),
        ]
        monkeypatch.setattr(_search_mod._utils, "is_local_mode", lambda: False)
        monkeypatch.setattr(_search_mod._client, "get_zotero_client", lambda: fake)

        result = search_by_citation_key("  Smith2024  ", ctx=DummyContext())

        assert "Citation Key: Smith2024" in result
        assert "Stripped Key" in result
