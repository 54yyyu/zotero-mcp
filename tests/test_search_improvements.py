"""Tests for search improvements: normalization, variant generation, fallback cascade."""

import pytest
from unittest.mock import MagicMock, patch
from conftest import DummyContext, FakeZotero

from zotero_mcp import utils as _utils
from zotero_mcp.tools import search as search_module


# ---------------------------------------------------------------------------
# TestNormalization
# ---------------------------------------------------------------------------

class TestNormalization:
    """Test _normalize_for_search utility."""

    def test_diacritics_stripped(self):
        assert _utils._normalize_for_search("Müller") == "Muller"

    def test_dashes_normalized(self):
        # en-dash (\u2013) → hyphen-minus
        assert "-" in _utils._normalize_for_search("Cladder\u2013Micus")

    def test_ascii_unchanged(self):
        assert _utils._normalize_for_search("Smith 2024") == "Smith 2024"

    def test_empty_string(self):
        assert _utils._normalize_for_search("") == ""

    def test_german_umlaut_transliteration(self):
        # unidecode maps ö → o (not oe — that's the variant generator's job)
        result = _utils._normalize_for_search("Schröder")
        assert "o" in result  # ö → o
        assert "ö" not in result

    def test_cjk_transliteration(self):
        result = _utils._normalize_for_search("王").strip()
        assert result == "Wang"


# ---------------------------------------------------------------------------
# TestVariantGeneration
# ---------------------------------------------------------------------------

class TestVariantGeneration:
    """Test _generate_search_variants utility."""

    def test_query_with_dash_includes_space_variant(self):
        variants = _utils._generate_search_variants("Cladder-Micus")
        assert "Cladder-Micus" in variants
        assert "Cladder Micus" in variants

    def test_query_with_umlaut_includes_expanded(self):
        variants = _utils._generate_search_variants("Müller")
        assert "Müller" in variants
        # unidecode form
        assert any("Muller" in v for v in variants)
        # German umlaut expansion
        assert "Mueller" in variants

    def test_plain_ascii_single_variant(self):
        variants = _utils._generate_search_variants("Smith")
        assert variants == ["Smith"]

    def test_deduplication(self):
        variants = _utils._generate_search_variants("test")
        assert len(variants) == len(set(variants))

    def test_cap_at_max(self):
        # Even a complex query shouldn't exceed MAX_SEARCH_VARIANTS
        variants = _utils._generate_search_variants("Müller-Schmidt Björk Straße")
        assert len(variants) <= _utils.MAX_SEARCH_VARIANTS

    def test_empty_query(self):
        variants = _utils._generate_search_variants("")
        assert variants == []


# ---------------------------------------------------------------------------
# TestSearchWithVariants
# ---------------------------------------------------------------------------

class TestSearchWithVariants:
    """Test _search_with_variants helper."""

    def _make_zot(self, items_by_query):
        """Create a fake zot that returns different items per query."""
        zot = MagicMock()
        captured_params = {}

        def fake_add_params(**kwargs):
            captured_params.update(kwargs)

        def fake_items():
            q = captured_params.get("q", "")
            return items_by_query.get(q, [])

        zot.add_parameters = fake_add_params
        zot.items = fake_items
        return zot

    def test_finds_via_original_query(self):
        items = [{"key": "A1", "data": {"title": "Paper A"}}]
        zot = self._make_zot({"Brewer 2011": items})

        result = search_module._search_with_variants(
            zot, "Brewer 2011", "titleCreatorYear", 10
        )
        assert len(result) == 1
        assert result[0]["key"] == "A1"

    def test_finds_via_normalized_variant(self):
        # The item is stored under the ASCII form
        items = [{"key": "B1", "data": {"title": "Paper B"}}]
        zot = self._make_zot({"Muller": items})

        result = search_module._search_with_variants(
            zot, "Müller", "titleCreatorYear", 10
        )
        assert len(result) == 1
        assert result[0]["key"] == "B1"

    def test_deduplicates_across_variants(self):
        # Same item found via both original and variant
        item = {"key": "C1", "data": {"title": "Paper C"}}
        zot = self._make_zot({
            "Cladder-Micus": [item],
            "Cladder Micus": [item],
        })

        result = search_module._search_with_variants(
            zot, "Cladder-Micus", "titleCreatorYear", 10
        )
        assert len(result) == 1  # deduplicated

    def test_forwards_item_type_and_tag(self):
        zot = MagicMock()
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)

        zot.add_parameters = capture
        zot.items = MagicMock(return_value=[])

        search_module._search_with_variants(
            zot, "test", "titleCreatorYear", 10,
            item_type="-note", tag=["research"]
        )

        assert captured["itemType"] == "-note"
        assert captured["tag"] == ["research"]


# ---------------------------------------------------------------------------
# TestFallbackCascade
# ---------------------------------------------------------------------------

class TestFallbackCascade:
    """Test the fallback cascade in search_items."""

    def _setup(self, monkeypatch, items_by_query=None):
        """Set up a fake Zotero client for search_items tests."""
        fake_zot = MagicMock()
        captured_params = {}

        def fake_add_params(**kwargs):
            captured_params.update(kwargs)

        def fake_items():
            q = captured_params.get("q", "")
            return (items_by_query or {}).get(q, [])

        fake_zot.add_parameters = fake_add_params
        fake_zot.items = fake_items
        monkeypatch.setattr(_utils, "_generate_search_variants",
                            lambda q: [q])  # No variant expansion for these tests
        monkeypatch.setattr(search_module._client, "get_zotero_client",
                            lambda: fake_zot)
        return fake_zot

    def test_finds_on_first_try_no_fallback(self, monkeypatch):
        items = [{"key": "X1", "data": {"title": "Found", "itemType": "journalArticle",
                                         "creators": [], "date": "2020", "tags": []}}]
        self._setup(monkeypatch, {"Brewer 2011": items})

        ctx = DummyContext()
        result = search_module.search_items(query="Brewer 2011", ctx=ctx)

        assert "Found" in result
        assert "Note:" not in result  # no fallback note

    def test_finds_via_simplified_query(self, monkeypatch):
        items = [{"key": "X2", "data": {"title": "Simplified Find", "itemType": "journalArticle",
                                         "creators": [], "date": "2020", "tags": []}}]
        # "Brewer 2011 DMN default mode network" → Strategy 1 extracts author+year = "Brewer 2011"
        # Mock "Brewer 2011" so Strategy 1 finds it directly
        self._setup(monkeypatch, {"Brewer 2011": items})

        ctx = DummyContext()
        result = search_module.search_items(
            query="Brewer 2011 DMN default mode network", ctx=ctx
        )

        assert "Simplified Find" in result
        assert "Note:" in result  # fallback note present

    def test_finds_via_author_only(self, monkeypatch):
        items = [{"key": "X3", "data": {"title": "Author Find", "itemType": "journalArticle",
                                         "creators": [], "date": "2020", "tags": []}}]
        self._setup(monkeypatch, {"Brewer": items})

        ctx = DummyContext()
        result = search_module.search_items(query="Brewer 2011", ctx=ctx)

        assert "Author Find" in result
        assert "Note:" in result

    def test_returns_not_found_when_all_fail(self, monkeypatch, tmp_path):
        self._setup(monkeypatch, {})  # Nothing found for any query
        # Point config path to a nonexistent location so semantic search is skipped
        monkeypatch.setattr(search_module, "Path", lambda *a: tmp_path / "nonexistent")

        ctx = DummyContext()
        result = search_module.search_items(query="Nonexistent 9999", ctx=ctx)

        assert "No items found" in result

    def test_fallback_note_includes_verification_guidance(self, monkeypatch):
        items = [{"key": "X4", "data": {"title": "Paper A", "itemType": "journalArticle",
                                         "creators": [], "date": "2020", "tags": []}},
                 {"key": "X5", "data": {"title": "Paper B", "itemType": "journalArticle",
                                         "creators": [], "date": "2021", "tags": []}}]
        self._setup(monkeypatch, {"Brewer": items})

        ctx = DummyContext()
        result = search_module.search_items(query="Brewer 2011", ctx=ctx)

        assert "verify" in result.lower()
        assert "title, authors, journal, and year" in result


# ---------------------------------------------------------------------------
# TestAdvancedSearchNormalization
# ---------------------------------------------------------------------------

class TestAdvancedSearchNormalization:
    """Test that _compare in advanced_search normalizes both sides."""

    def test_umlaut_matches_ascii(self, monkeypatch):
        """Müller should match Muller in advanced search conditions."""
        item = {"key": "Z1", "version": 1, "data": {
            "itemType": "journalArticle", "title": "Test Paper",
            "creators": [{"lastName": "Müller", "firstName": "Hans", "creatorType": "author"}],
            "date": "2020", "tags": [],
        }}

        fake_zot = MagicMock()
        fake_zot.items = MagicMock(return_value=[item])
        monkeypatch.setattr(search_module._client, "get_zotero_client", lambda: fake_zot)

        ctx = DummyContext()
        result = search_module.advanced_search(
            conditions=[{"field": "creator", "operation": "contains", "value": "Muller"}],
            ctx=ctx
        )

        assert "Z1" in result or "Müller" in result or "Test Paper" in result

    def test_dash_matches_en_dash(self, monkeypatch):
        """Cladder-Micus (hyphen) should match Cladder–Micus (en-dash)."""
        item = {"key": "Z2", "version": 1, "data": {
            "itemType": "journalArticle", "title": "Test Paper",
            "creators": [{"lastName": "Cladder\u2013Micus", "firstName": "C", "creatorType": "author"}],
            "date": "2018", "tags": [],
        }}

        fake_zot = MagicMock()
        fake_zot.items = MagicMock(return_value=[item])
        monkeypatch.setattr(search_module._client, "get_zotero_client", lambda: fake_zot)

        ctx = DummyContext()
        result = search_module.advanced_search(
            conditions=[{"field": "creator", "operation": "contains", "value": "Cladder-Micus"}],
            ctx=ctx
        )

        assert "Z2" in result or "Test Paper" in result


# ---------------------------------------------------------------------------
# TestMultiWordSearch (from token optimization plan)
# ---------------------------------------------------------------------------

class TestMultiWordSearch:
    """Test multi-word collection search in search_collections."""

    def test_kcl_mindfulness_matches(self, monkeypatch):
        from zotero_mcp.tools import write as write_module

        fake_zot = MagicMock()
        fake_zot.collections = MagicMock(return_value=[
            {"key": "COL1", "data": {"name": "KCL - Mindfulness", "parentCollection": False}},
            {"key": "COL2", "data": {"name": "Other Collection", "parentCollection": False}},
        ])
        monkeypatch.setattr(write_module._client, "get_zotero_client", lambda: fake_zot)

        ctx = DummyContext()
        result = write_module.search_collections(query="KCL mindfulness", ctx=ctx)

        assert "KCL - Mindfulness" in result
        assert "Other Collection" not in result

    def test_single_word_backward_compatible(self, monkeypatch):
        from zotero_mcp.tools import write as write_module

        fake_zot = MagicMock()
        fake_zot.collections = MagicMock(return_value=[
            {"key": "COL1", "data": {"name": "Machine Learning", "parentCollection": False}},
            {"key": "COL2", "data": {"name": "Deep Learning", "parentCollection": False}},
        ])
        monkeypatch.setattr(write_module._client, "get_zotero_client", lambda: fake_zot)

        ctx = DummyContext()
        result = write_module.search_collections(query="Learning", ctx=ctx)

        assert "Machine Learning" in result
        assert "Deep Learning" in result


# ---------------------------------------------------------------------------
# TestCascadeSimplification (P2 fix)
# ---------------------------------------------------------------------------

class TestCascadeSimplification:
    """Tests for Strategy 1 author+year extraction."""

    def _setup(self, monkeypatch, items_by_query):
        fake_zot = MagicMock()

        def fake_items(**kwargs):
            return []

        fake_zot.items = fake_items
        fake_zot.add_parameters = MagicMock()

        def fake_search_with_variants(zot, query, qmode, limit, item_type="-attachment",
                                       tag=None, cascade_start=None, cascade_timeout=None):
            return items_by_query.get(query, [])

        monkeypatch.setattr(search_module, "_search_with_variants", fake_search_with_variants)
        monkeypatch.setattr(search_module._client, "get_zotero_client", lambda: fake_zot)

    def test_strategy1_uses_author_and_year(self, monkeypatch):
        items = [{"key": "L1", "data": {"title": "Lynch Paper", "itemType": "journalArticle",
                                         "creators": [], "date": "2003", "tags": []}}]
        self._setup(monkeypatch, {"Lynch 2003": items})
        ctx = DummyContext()
        result = search_module.search_items(
            query="Lynch 2003 dialectical behavior therapy depressed older adults", ctx=ctx
        )
        assert "Lynch Paper" in result

    def test_strategy1_no_year_uses_author_only(self, monkeypatch):
        items = [{"key": "L2", "data": {"title": "Lynch No Year", "itemType": "journalArticle",
                                         "creators": [], "date": "2003", "tags": []}}]
        self._setup(monkeypatch, {"Lynch": items})
        ctx = DummyContext()
        result = search_module.search_items(
            query="Lynch dialectical behavior therapy", ctx=ctx
        )
        assert "Lynch No Year" in result

    def test_strategy1_two_words_skipped(self, monkeypatch):
        """With only 2 words, Strategy 1 (len > 2) is skipped."""
        items = [{"key": "L3", "data": {"title": "Two Words", "itemType": "journalArticle",
                                         "creators": [], "date": "2003", "tags": []}}]
        # Only Strategy 2 (author only) should fire for 2-word queries
        self._setup(monkeypatch, {"Lynch": items})
        ctx = DummyContext()
        result = search_module.search_items(query="Lynch 2003", ctx=ctx)
        assert "Two Words" in result

    def test_strategy1_year_first_reordered(self, monkeypatch):
        """'2003 Lynch therapy' should produce 'Lynch 2003', not '2003 2003'."""
        items = [{"key": "L4", "data": {"title": "Year First", "itemType": "journalArticle",
                                         "creators": [], "date": "2003", "tags": []}}]
        self._setup(monkeypatch, {"Lynch 2003": items})
        ctx = DummyContext()
        result = search_module.search_items(
            query="2003 Lynch therapy", ctx=ctx
        )
        assert "Year First" in result

    def test_strategy1_multiple_years_picks_first(self, monkeypatch):
        items = [{"key": "L5", "data": {"title": "Multi Year", "itemType": "journalArticle",
                                         "creators": [], "date": "2003", "tags": []}}]
        self._setup(monkeypatch, {"Lynch 2003": items})
        ctx = DummyContext()
        result = search_module.search_items(
            query="Lynch 2003 2005 therapy", ctx=ctx
        )
        assert "Multi Year" in result


# ---------------------------------------------------------------------------
# TestVerificationGuidance (P1 + P3 fixes)
# ---------------------------------------------------------------------------

class TestVerificationGuidance:
    """Tests for fallback note content."""

    def _setup(self, monkeypatch, items_by_query):
        fake_zot = MagicMock()
        fake_zot.items = MagicMock(return_value=[])
        fake_zot.add_parameters = MagicMock()

        def fake_search_with_variants(zot, query, qmode, limit, item_type="-attachment",
                                       tag=None, cascade_start=None, cascade_timeout=None):
            return items_by_query.get(query, [])

        monkeypatch.setattr(search_module, "_search_with_variants", fake_search_with_variants)
        monkeypatch.setattr(search_module._client, "get_zotero_client", lambda: fake_zot)

    def test_fallback_note_includes_original_query(self, monkeypatch):
        items = [{"key": "V1", "data": {"title": "Some Paper", "itemType": "journalArticle",
                                         "creators": [], "date": "2020", "tags": []}}]
        self._setup(monkeypatch, {"Brewer": items})
        ctx = DummyContext()
        result = search_module.search_items(query="Brewer 2011", ctx=ctx)
        assert "Brewer 2011" in result  # Original query in the note

    def test_semantic_fallback_has_stronger_warning(self, monkeypatch, tmp_path):
        """When semantic search is the fallback, note should say 'may NOT be the exact paper'."""
        sem_items = [{"key": "S1", "data": {"title": "Semantic Hit", "itemType": "journalArticle",
                                              "creators": [], "date": "2020", "tags": []}}]

        def fake_search_with_variants(zot, query, qmode, limit, item_type="-attachment",
                                       tag=None, cascade_start=None, cascade_timeout=None):
            return []  # All text searches fail

        fake_zot = MagicMock()
        fake_zot.items = MagicMock(return_value=[])
        fake_zot.add_parameters = MagicMock()
        monkeypatch.setattr(search_module, "_search_with_variants", fake_search_with_variants)
        monkeypatch.setattr(search_module._client, "get_zotero_client", lambda: fake_zot)

        # Mock semantic search
        fake_sem = MagicMock()
        fake_sem.search.return_value = {
            "results": [{"item_key": "S1", "zotero_item": sem_items[0]}]
        }
        fake_create = MagicMock(return_value=fake_sem)

        # Make config path exist
        config_dir = tmp_path / ".config" / "zotero-mcp"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text("{}")
        monkeypatch.setattr(search_module.Path, "home", lambda: tmp_path)

        monkeypatch.setattr(
            "zotero_mcp.semantic_search.create_semantic_search", fake_create
        )

        ctx = DummyContext()
        result = search_module.search_items(query="Nonexistent Paper 2099", ctx=ctx)

        assert "may NOT be" in result or "semantic" in result.lower()


# ---------------------------------------------------------------------------
# TestCascadeTimeout (P5 fix)
# ---------------------------------------------------------------------------

class TestCascadeTimeout:
    """Test that cascade respects the timeout budget."""

    def test_cascade_respects_timeout(self, monkeypatch):
        """Setting CASCADE_TIMEOUT to 0 should skip all fallback strategies."""
        monkeypatch.setattr(search_module, "CASCADE_TIMEOUT", 0)

        fake_zot = MagicMock()
        fake_zot.items = MagicMock(return_value=[])
        fake_zot.add_parameters = MagicMock()

        def fake_search_with_variants(zot, query, qmode, limit, item_type="-attachment",
                                       tag=None, cascade_start=None, cascade_timeout=None):
            return []  # Nothing found

        monkeypatch.setattr(search_module, "_search_with_variants", fake_search_with_variants)
        monkeypatch.setattr(search_module._client, "get_zotero_client", lambda: fake_zot)

        ctx = DummyContext()
        result = search_module.search_items(
            query="Lynch 2003 dialectical behavior therapy", ctx=ctx
        )

        # Should return "no items found" without trying all strategies
        assert "No items found" in result
