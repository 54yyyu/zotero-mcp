"""Tests for concise mode feature in zotero-mcp."""

from unittest.mock import MagicMock, patch

import pytest

from zotero_mcp.server import (
    _BIBLIOGRAPHY_RE,
    format_concise_item,
    is_concise_mode,
    trim_bibliography,
)
from zotero_mcp.utils import format_creators

# All @mcp.tool-decorated functions are FunctionTool objects.
# Access the raw callable via `.fn`.
from zotero_mcp import server as _srv

_search_items = _srv.search_items.fn
_search_by_tag = _srv.search_by_tag.fn
_get_collection_items = _srv.get_collection_items.fn
_get_recent = _srv.get_recent.fn
_get_notes = _srv.get_notes.fn
_advanced_search = _srv.advanced_search.fn
_semantic_search = _srv.semantic_search.fn


# ---------------------------------------------------------------------------
# is_concise_mode()
# ---------------------------------------------------------------------------


class TestIsConciseMode:
    """Tests for the is_concise_mode() function."""

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "yes", "Yes", "1"])
    def test_truthy_values(self, monkeypatch: pytest.MonkeyPatch, value: str):
        monkeypatch.setenv("ZOTERO_CONCISE_MODE", value)
        assert is_concise_mode() is True

    @pytest.mark.parametrize("value", ["false", "False", "no", "No", "0", "random"])
    def test_falsy_values(self, monkeypatch: pytest.MonkeyPatch, value: str):
        monkeypatch.setenv("ZOTERO_CONCISE_MODE", value)
        assert is_concise_mode() is False

    def test_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ZOTERO_CONCISE_MODE", raising=False)
        assert is_concise_mode() is False

    def test_empty_string(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ZOTERO_CONCISE_MODE", "")
        assert is_concise_mode() is False


# ---------------------------------------------------------------------------
# format_concise_item()
# ---------------------------------------------------------------------------


class TestFormatConciseItem:
    """Tests for the format_concise_item() function."""

    def test_full_item(self, sample_item: dict):
        result = format_concise_item(1, sample_item)
        assert result == "1. Deep Learning for NLP (2023) - Smith et al. [ABC12345]"

    def test_no_date(self, sample_item: dict):
        sample_item["data"]["date"] = ""
        result = format_concise_item(1, sample_item)
        assert "(20" not in result
        assert "Deep Learning for NLP" in result
        assert "[ABC12345]" in result

    def test_no_creators(self, sample_item: dict):
        sample_item["data"]["creators"] = []
        result = format_concise_item(1, sample_item)
        assert " - " not in result
        assert "1. Deep Learning for NLP (2023) [ABC12345]" == result

    def test_no_key(self, sample_item: dict):
        del sample_item["key"]
        result = format_concise_item(1, sample_item)
        assert "[" not in result
        assert result.startswith("1. Deep Learning for NLP")

    def test_single_creator(self, sample_item_single_author: dict):
        result = format_concise_item(1, sample_item_single_author)
        assert "- Wong" in result
        assert "et al." not in result

    def test_minimal_item(self, sample_item_minimal: dict):
        result = format_concise_item(1, sample_item_minimal)
        assert result == "1. Orphan Paper"

    def test_empty_item(self):
        result = format_concise_item(5, {"data": {}})
        assert result == "5. Untitled"

    def test_non_numeric_date(self, sample_item: dict):
        sample_item["data"]["date"] = "circa 1900"
        result = format_concise_item(1, sample_item)
        # Year extraction requires first 4 chars to be digits
        assert "(circa" not in result


# ---------------------------------------------------------------------------
# _BIBLIOGRAPHY_RE regex
# ---------------------------------------------------------------------------


class TestBibliographyRegex:
    """Tests for the _BIBLIOGRAPHY_RE compiled pattern."""

    @pytest.mark.parametrize(
        "marker",
        [
            # Plain text variants
            "\nReferences\n",
            "\nreferences\n",
            "\nREFERENCES\n",
            "\nReference\n",
            "\nBibliography\n",
            "\nbibliography\n",
            "\nBibliographic References\n",
            "\nWorks Cited\n",
            "\nworks cited\n",
            "\nWork Cited\n",
            "\nCitations\n",
            "\ncitation\n",
            "\nLiterature Cited\n",
            "\nSources\n",
            "\nsource\n",
            "\nReference List\n",
            "\nList of References\n",
            "\nAppendix\n",
            "\nappendix\n",
            "\nAppendices\n",
            "\nSupplementary Material\n",
            "\nSupplement\n",
            # Markdown heading variants
            "\n# References\n",
            "\n## References\n",
            "\n### References\n",
            "\n#### References\n",
            "\n## Bibliography\n",
            "\n## Works Cited\n",
            "\n### Appendix\n",
            # Bold markdown variants
            "\n**References**\n",
            "\n**Bibliography**\n",
            "\n**Works Cited**\n",
            # Bracket variants
            "\n[References]\n",
            "\n[Bibliography]\n",
            # With colon
            "\nReferences:\n",
            "\nBibliography:\n",
            # Numbered prefix
            "\n1. References\n",
            "\n12. Bibliography\n",
            # Combined: heading + numbered
            "\n## 1. References\n",
        ],
    )
    def test_matches_bibliography_markers(self, marker: str):
        assert _BIBLIOGRAPHY_RE.search(marker) is not None, (
            f"Pattern should match: {marker!r}"
        )

    @pytest.mark.parametrize(
        "text",
        [
            # Mid-sentence, no leading newline
            "See the references section for details.",
            # No trailing newline
            "\nReferences",
            # Word inside another word
            "cross-references are important",
        ],
    )
    def test_no_false_positives(self, text: str):
        assert _BIBLIOGRAPHY_RE.search(text) is None, (
            f"Pattern should NOT match: {text!r}"
        )


# ---------------------------------------------------------------------------
# trim_bibliography()
# ---------------------------------------------------------------------------


class TestTrimBibliography:
    """Tests for the trim_bibliography() function."""

    def test_concise_off_returns_unchanged(self, normal_env):
        text = "Main content.\n\nReferences\n\n[1] Foo"
        assert trim_bibliography(text) == text

    def test_concise_on_no_marker(self, concise_env):
        text = "Just some text with no bibliography heading."
        assert trim_bibliography(text) == text

    def test_concise_on_marker_in_last_40_percent(self, concise_env):
        # Build text where the \nReferences\n marker is past the 60% mark
        body = "A" * 700
        bib = "\nReferences\n\n[1] Some reference item\n[2] Another ref"
        text = body + bib
        result = trim_bibliography(text)
        # The original reference entries should be gone
        assert "[1] Some reference" not in result
        assert "[2] Another ref" not in result
        # Trim notice should be appended
        assert "[... Bibliography/References section trimmed" in result
        assert "tokens saved" in result

    def test_concise_on_marker_in_first_60_percent(self, concise_env):
        # Marker appears early — should NOT be trimmed
        bib = "\nReferences\n\nSee references above for details."
        tail = "B" * 1000
        text = bib + tail
        result = trim_bibliography(text)
        assert result == text

    def test_trimmed_message_format(self, concise_env):
        body = "X" * 800
        bib = "\nBibliography\n\n" + "Y" * 200
        text = body + bib
        result = trim_bibliography(text)
        # Verify the appended message
        assert result.endswith("...]")
        assert "concise mode" in result
        assert "tokens saved" in result


# ---------------------------------------------------------------------------
# format_creators(concise=True)
# ---------------------------------------------------------------------------


class TestFormatCreatorsConcise:
    """Tests for format_creators with concise=True."""

    def test_single_author_lastname(self):
        creators = [{"firstName": "John", "lastName": "Smith"}]
        assert format_creators(creators, concise=True) == "Smith"

    def test_multiple_authors_et_al(self):
        creators = [
            {"firstName": "John", "lastName": "Smith"},
            {"firstName": "Jane", "lastName": "Doe"},
        ]
        assert format_creators(creators, concise=True) == "Smith et al."

    def test_name_field_only(self):
        creators = [{"name": "World Health Organization"}]
        result = format_creators(creators, concise=True)
        assert result == "World"

    def test_empty_list(self):
        assert format_creators([], concise=True) == "No authors listed"

    def test_multiple_with_name_field(self):
        creators = [
            {"name": "International Org"},
            {"name": "Another Body"},
        ]
        result = format_creators(creators, concise=True)
        assert "et al." in result


# ---------------------------------------------------------------------------
# Integration tests — tool functions with mocked Zotero client
# ---------------------------------------------------------------------------


def _mock_zotero():
    """Create a patched get_zotero_client returning a MagicMock."""
    return patch("zotero_mcp.server.get_zotero_client")


class TestSearchItemsIntegration:
    """Integration test for the search_items tool."""

    def test_concise_output(self, sample_item, mock_ctx, concise_env):
        with _mock_zotero() as mock_zot:
            zot_instance = MagicMock()
            zot_instance.items.return_value = [sample_item]
            mock_zot.return_value = zot_instance

            result = _search_items(query="deep learning", ctx=mock_ctx)

            assert "1. Deep Learning for NLP (2023)" in result
            assert "Smith et al." in result
            assert "[ABC12345]" in result
            # Abstract snippet (80 chars)
            assert "This paper presents" in result
            # Should NOT have markdown heading format
            assert "## 1." not in result

    def test_normal_output(self, sample_item, mock_ctx, normal_env):
        with _mock_zotero() as mock_zot:
            zot_instance = MagicMock()
            zot_instance.items.return_value = [sample_item]
            mock_zot.return_value = zot_instance

            result = _search_items(query="deep learning", ctx=mock_ctx)

            assert "## 1. Deep Learning for NLP" in result
            assert "**Type:**" in result
            assert "**Item Key:**" in result


class TestSearchByTagIntegration:
    """Integration test for the search_by_tag tool."""

    def test_concise_output(self, sample_item, mock_ctx, concise_env):
        with _mock_zotero() as mock_zot:
            zot_instance = MagicMock()
            zot_instance.items.return_value = [sample_item]
            mock_zot.return_value = zot_instance

            result = _search_by_tag(tag=["deep-learning"], ctx=mock_ctx)

            assert "1. Deep Learning for NLP (2023)" in result
            assert "Smith et al." in result
            assert "## 1." not in result


class TestGetCollectionItemsIntegration:
    """Integration test for the get_collection_items tool."""

    def test_concise_output(self, sample_item, mock_ctx, concise_env):
        with _mock_zotero() as mock_zot:
            zot_instance = MagicMock()
            zot_instance.collection.return_value = {
                "data": {"name": "My Collection"}
            }
            zot_instance.collection_items.return_value = [sample_item]
            mock_zot.return_value = zot_instance

            result = _get_collection_items(
                collection_key="COLL0001", ctx=mock_ctx
            )

            assert "Collection: My Collection" in result
            assert "1. Deep Learning for NLP (2023)" in result
            assert "## 1." not in result


class TestGetRecentIntegration:
    """Integration test for the get_recent tool."""

    def test_concise_output(self, sample_item, mock_ctx, concise_env):
        with _mock_zotero() as mock_zot:
            zot_instance = MagicMock()
            zot_instance.items.return_value = [sample_item]
            mock_zot.return_value = zot_instance

            result = _get_recent(limit=5, ctx=mock_ctx)

            assert "Recent" in result
            assert "1. Deep Learning for NLP (2023)" in result
            assert "## 1." not in result

    def test_normal_output(self, sample_item, mock_ctx, normal_env):
        with _mock_zotero() as mock_zot:
            zot_instance = MagicMock()
            zot_instance.items.return_value = [sample_item]
            mock_zot.return_value = zot_instance

            result = _get_recent(limit=5, ctx=mock_ctx)

            assert "# 5 Most Recently Added Items" in result
            assert "## 1. Deep Learning for NLP" in result
            assert "**Authors:**" in result


class TestGetNotesIntegration:
    """Integration test for the get_notes tool."""

    def test_concise_output(self, sample_notes, mock_ctx, concise_env):
        parent_item = {
            "data": {"title": "Deep Learning for NLP"},
        }
        with _mock_zotero() as mock_zot:
            zot_instance = MagicMock()
            zot_instance.items.return_value = sample_notes
            zot_instance.item.return_value = parent_item
            mock_zot.return_value = zot_instance

            result = _get_notes(ctx=mock_ctx)

            # Concise format: "{i}. Note [KEY] (parent: Title)"
            assert "1. Note [NOTE0001]" in result
            assert "(parent: Deep Learning for NLP)" in result
            # 150-char snippet of cleaned HTML
            assert "Key insight about the methodology." in result
            # Should NOT have full-format headings
            assert "## Note 1" not in result

    def test_normal_output(self, sample_notes, mock_ctx, normal_env):
        parent_item = {
            "data": {"title": "Deep Learning for NLP"},
        }
        with _mock_zotero() as mock_zot:
            zot_instance = MagicMock()
            zot_instance.items.return_value = sample_notes
            zot_instance.item.return_value = parent_item
            mock_zot.return_value = zot_instance

            result = _get_notes(ctx=mock_ctx)

            assert "## Note 1" in result
            assert "**Key:** NOTE0001" in result


class TestSemanticSearchIntegration:
    """Integration test for the semantic_search tool."""

    def test_concise_output(self, sample_item, mock_ctx, concise_env):
        search_results = {
            "results": [
                {
                    "similarity_score": 0.92,
                    "item_key": "ABC12345",
                    "zotero_item": sample_item,
                },
            ],
            "error": None,
        }

        with (
            _mock_zotero(),
            patch(
                "zotero_mcp.semantic_search.create_semantic_search"
            ) as mock_create,
        ):
            mock_search = MagicMock()
            mock_search.search.return_value = search_results
            mock_create.return_value = mock_search

            result = _semantic_search(
                query="deep learning NLP", ctx=mock_ctx
            )

            assert "1. Deep Learning for NLP (2023)" in result
            assert "(sim:0.92)" in result
            assert "## 1." not in result

    def test_normal_output(self, sample_item, mock_ctx, normal_env):
        search_results = {
            "results": [
                {
                    "similarity_score": 0.92,
                    "item_key": "ABC12345",
                    "zotero_item": sample_item,
                },
            ],
            "error": None,
        }

        with (
            _mock_zotero(),
            patch(
                "zotero_mcp.semantic_search.create_semantic_search"
            ) as mock_create,
        ):
            mock_search = MagicMock()
            mock_search.search.return_value = search_results
            mock_create.return_value = mock_search

            result = _semantic_search(
                query="deep learning NLP", ctx=mock_ctx
            )

            assert "## 1. Deep Learning for NLP" in result
            assert "**Similarity Score:** 0.920" in result


class TestAdvancedSearchIntegration:
    """Integration test for the advanced_search tool."""

    def test_concise_output(self, sample_item, mock_ctx, concise_env):
        with _mock_zotero() as mock_zot:
            zot_instance = MagicMock()
            zot_instance.saved_search.return_value = {
                "success": {"0": "SEARCH01"},
            }
            zot_instance.collection_items.return_value = [sample_item]
            mock_zot.return_value = zot_instance

            conditions = [
                {
                    "field": "title",
                    "operation": "contains",
                    "value": "deep learning",
                }
            ]
            result = _advanced_search(
                conditions=conditions, ctx=mock_ctx
            )

            assert "Advanced search:" in result
            assert "1. Deep Learning for NLP (2023)" in result
            # Concise mode should NOT include search criteria section
            assert "## Search Criteria" not in result
            assert "## Results" not in result

    def test_normal_output(self, sample_item, mock_ctx, normal_env):
        with _mock_zotero() as mock_zot:
            zot_instance = MagicMock()
            zot_instance.saved_search.return_value = {
                "success": {"0": "SEARCH01"},
            }
            zot_instance.collection_items.return_value = [sample_item]
            mock_zot.return_value = zot_instance

            conditions = [
                {
                    "field": "title",
                    "operation": "contains",
                    "value": "deep learning",
                }
            ]
            result = _advanced_search(
                conditions=conditions, ctx=mock_ctx
            )

            assert "# Advanced Search Results" in result
            assert "## Search Criteria" in result
            assert "## Results" in result
            assert "### 1. Deep Learning for NLP" in result
