"""Tests for the search_by_citation_key tool and helper functions."""

from unittest.mock import patch, MagicMock

import pytest

from conftest import DummyContext, FakeZotero
from zotero_mcp.server import (
    _extra_has_citekey,
    _format_citekey_result,
    _format_bbt_result,
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

    @patch("zotero_mcp.utils.is_local_mode", return_value=False)
    @patch("zotero_mcp.client.get_zotero_client")
    def test_found_via_extra_field(self, mock_get_client, _mock_local):
        fake = _CitekeyFakeZotero()
        fake._items = [
            _make_item(key="ABC123", title="Deep Learning", citekey="Smith2024"),
        ]
        mock_get_client.return_value = fake

        result = search_by_citation_key("Smith2024", ctx=DummyContext())

        assert "Citation Key: Smith2024" in result
        assert "Deep Learning" in result
        assert "ABC123" in result

    @patch("zotero_mcp.utils.is_local_mode", return_value=False)
    @patch("zotero_mcp.client.get_zotero_client")
    def test_no_match(self, mock_get_client, _mock_local):
        fake = _CitekeyFakeZotero()
        fake._items = [
            _make_item(key="XYZ999", title="Other Paper", citekey="Jones2023"),
        ]
        mock_get_client.return_value = fake

        result = search_by_citation_key("Smith2024", ctx=DummyContext())

        assert "No item found with citation key: 'Smith2024'" in result


# ---------------------------------------------------------------------------
# search_by_citation_key – local mode (Strategy A)
# ---------------------------------------------------------------------------

class TestSearchByCitationKeyLocalMode:
    """Tests where BBT is available (local mode)."""

    @patch("zotero_mcp.utils.is_local_mode", return_value=True)
    @patch("zotero_mcp.client.get_zotero_client")
    def test_bbt_lookup_succeeds(self, mock_get_client, _mock_local):
        # BBT returns a matching result with itemKey
        bbt_instance = MagicMock()
        bbt_instance.is_zotero_running.return_value = True
        bbt_instance._make_request.return_value = [
            {"citekey": "Smith2024", "itemKey": "ABC123", "title": "Deep Learning"}
        ]

        fake = _CitekeyFakeZotero()
        fake._items = [
            _make_item(key="ABC123", title="Deep Learning", citekey="Smith2024"),
        ]
        mock_get_client.return_value = fake

        # Patch the import inside search_by_citation_key
        with patch(
            "zotero_mcp.better_bibtex_client.ZoteroBetterBibTexAPI",
            return_value=bbt_instance,
        ):
            result = search_by_citation_key("Smith2024", ctx=DummyContext())

        assert "Citation Key: Smith2024" in result
        assert "Deep Learning" in result

    @patch("zotero_mcp.utils.is_local_mode", return_value=True)
    @patch("zotero_mcp.client.get_zotero_client")
    def test_bbt_fails_falls_back_to_extra(self, mock_get_client, _mock_local):
        """When BBT raises an exception, Strategy B (Extra field) is used."""
        fake = _CitekeyFakeZotero()
        fake._items = [
            _make_item(key="DEF456", title="Fallback Paper", citekey="Smith2024"),
        ]
        mock_get_client.return_value = fake

        # Make the BBT import succeed but the instance raises
        with patch(
            "zotero_mcp.better_bibtex_client.ZoteroBetterBibTexAPI",
        ) as MockBBT:
            instance = MockBBT.return_value
            instance.is_zotero_running.side_effect = Exception("connection refused")

            result = search_by_citation_key("Smith2024", ctx=DummyContext())

        assert "Citation Key: Smith2024" in result
        assert "Fallback Paper" in result
        assert "DEF456" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestSearchByCitationKeyEdgeCases:
    def test_empty_citekey(self):
        result = search_by_citation_key("  ", ctx=DummyContext())
        assert "Error: Citation key cannot be empty" in result

    @patch("zotero_mcp.utils.is_local_mode", return_value=False)
    @patch("zotero_mcp.client.get_zotero_client")
    def test_whitespace_stripped(self, mock_get_client, _mock_local):
        fake = _CitekeyFakeZotero()
        fake._items = [
            _make_item(key="ABC123", title="Stripped Key", citekey="Smith2024"),
        ]
        mock_get_client.return_value = fake

        result = search_by_citation_key("  Smith2024  ", ctx=DummyContext())

        assert "Citation Key: Smith2024" in result
        assert "Stripped Key" in result
