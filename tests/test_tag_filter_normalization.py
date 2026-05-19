"""Regression tests for #237 — tag-filter argument normalization.

Clients produce several wire shapes for the `tag` argument depending on the
MCP runtime path: bare strings, JSON-string lists, lists of strings, lists
of dicts of the form {"tag": "X"} (the shape Zotero uses for stored tags,
which agents sometimes confuse with the filter shape). pyzotero's `tag=`
parameter wants list[str] — the normalizer collapses all inputs to that.
"""

import pytest

from zotero_mcp.tools._helpers import _normalize_tag_filter


class TestNormalizeTagFilter:
    def test_none_returns_empty_list(self):
        assert _normalize_tag_filter(None) == []

    def test_empty_string_returns_empty_list(self):
        assert _normalize_tag_filter("") == []
        assert _normalize_tag_filter("   ") == []

    def test_bare_string(self):
        assert _normalize_tag_filter("FIXME") == ["FIXME"]

    def test_list_of_strings(self):
        assert _normalize_tag_filter(["a", "b"]) == ["a", "b"]

    def test_list_of_dicts_with_tag_key(self):
        """The #237 primary case — LLM sends Zotero's stored-tag dict shape."""
        assert _normalize_tag_filter([{"tag": "FIXME"}]) == ["FIXME"]
        assert _normalize_tag_filter(
            [{"tag": "a"}, {"tag": "b"}]
        ) == ["a", "b"]

    def test_list_of_dicts_with_name_key(self):
        """Accept {'name': 'X'} as a fallback shape some clients emit."""
        assert _normalize_tag_filter([{"name": "FIXME"}]) == ["FIXME"]

    def test_json_string_list_of_strings(self):
        """Pydantic serialization stringifies arrays — this is the exact
        wire shape in the #237 bug report."""
        assert _normalize_tag_filter('["a", "b"]') == ["a", "b"]

    def test_json_string_list_of_dicts(self):
        """The #237 bug's literal failing input: list-of-dicts stringified."""
        assert _normalize_tag_filter('[{"tag": "FIXME"}]') == ["FIXME"]

    def test_json_string_single_dict(self):
        assert _normalize_tag_filter('{"tag": "FIXME"}') == ["FIXME"]

    def test_mixed_list(self):
        """Heterogeneous list: dicts and bare strings interleaved."""
        assert _normalize_tag_filter(
            [{"tag": "a"}, "b", {"name": "c"}]
        ) == ["a", "b", "c"]

    def test_empty_dict_ignored(self):
        assert _normalize_tag_filter([{}]) == []

    def test_dict_with_no_recognized_key_ignored(self):
        assert _normalize_tag_filter([{"foo": "bar"}]) == []

    def test_empty_string_inside_list_ignored(self):
        assert _normalize_tag_filter(["", "real", "   "]) == ["real"]

    def test_whitespace_trimmed(self):
        assert _normalize_tag_filter([" a ", {"tag": " b "}]) == ["a", "b"]

    def test_non_json_string_treated_as_single_tag(self):
        """A bare string that looks like a tag, not JSON."""
        assert _normalize_tag_filter("my-tag") == ["my-tag"]


class _SearchableFake:
    """Minimal pyzotero stub with the methods search_items needs."""

    def __init__(self):
        self.last_params = {}
        self._items_to_return = []

    def add_parameters(self, **kwargs):
        self.last_params = kwargs

    def items(self):
        return list(self._items_to_return)

    def collection(self, key):
        return {"key": key}

    def collection_items(self, *args, **kwargs):
        return list(self._items_to_return)


class TestSearchItemsIntegration:
    """End-to-end: search_items should accept the problematic shapes without
    raising a pydantic validation error AND actually pass the normalized
    list[str] through to pyzotero's ``tag=`` parameter."""

    def _patch(self, monkeypatch, fake):
        monkeypatch.setattr(
            "zotero_mcp.tools.search._client.get_zotero_client",
            lambda: fake,
        )

    def test_search_items_accepts_dict_shape_tag(self, monkeypatch):
        """The exact failing call from the #237 bug report."""
        from zotero_mcp import server
        from conftest import DummyContext

        fake = _SearchableFake()
        self._patch(monkeypatch, fake)

        result = server.search_items(
            query="whatever",
            tag=[{"tag": "FIXME"}],
            ctx=DummyContext(),
        )
        # Normalized to list[str] before hitting pyzotero
        assert fake.last_params.get("tag") == ["FIXME"]
        # No pydantic explosion; search ran to a clean "no results"
        assert "FIXME" in result

    def test_search_items_accepts_json_string_tag(self, monkeypatch):
        """The stringified form the MCP serialization layer produces."""
        from zotero_mcp import server
        from conftest import DummyContext

        fake = _SearchableFake()
        self._patch(monkeypatch, fake)

        result = server.search_items(
            query="whatever",
            tag='[{"tag": "FIXME"}]',
            ctx=DummyContext(),
        )
        assert fake.last_params.get("tag") == ["FIXME"]
        assert "FIXME" in result

    def test_search_items_accepts_canonical_list_of_strings(self, monkeypatch):
        """Regression guard: canonical shape must still work."""
        from zotero_mcp import server
        from conftest import DummyContext

        fake = _SearchableFake()
        self._patch(monkeypatch, fake)

        result = server.search_items(
            query="whatever",
            tag=["a", "b"],
            ctx=DummyContext(),
        )
        assert fake.last_params.get("tag") == ["a", "b"]
        assert "a, b" in result

    def test_search_items_no_tag(self, monkeypatch):
        """Regression guard: tag omitted entirely should not pass tag= to API."""
        from zotero_mcp import server
        from conftest import DummyContext

        fake = _SearchableFake()
        self._patch(monkeypatch, fake)

        server.search_items(query="whatever", ctx=DummyContext())
        assert "tag" not in fake.last_params or not fake.last_params.get("tag")
