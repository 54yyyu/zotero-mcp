"""The semantic fallback of zotero_search_items honours `tag` and `item_type`.

The text searches pass both filters to the backend; the semantic fallback
ignored them, so `tag='a'` listed untagged items and `item_type='book'`
listed articles.
"""

from unittest.mock import MagicMock

import pytest
from conftest import DummyContext

from zotero_mcp.tools import search as search_module


def _item(key, title, item_type="journalArticle", tags=()):
    return {
        "key": key,
        "data": {
            "title": title, "itemType": item_type, "creators": [],
            "date": "2020", "tags": [{"tag": t} for t in tags],
        },
    }


HITS = [
    _item("AAA00001", "Tagged Article", tags=["thesis", "ai"]),
    _item("AAA00002", "Untagged Article"),
    _item("AAA00003", "Tagged Book", item_type="book", tags=["thesis"]),
    _item("AAA00004", "Draft Article", tags=["thesis", "draft"]),
]


@pytest.fixture
def semantic_only(monkeypatch, tmp_path):
    """Every text search finds nothing; the semantic index returns HITS."""
    monkeypatch.setattr(
        search_module, "_search_with_variants", lambda *a, **k: []
    )
    fake_zot = MagicMock()
    monkeypatch.setattr(search_module._client, "get_zotero_client", lambda: fake_zot)
    fake_sem = MagicMock()
    fake_sem.search.side_effect = lambda query, limit, group_id=None: {
        "results": [{"item_key": h["key"], "zotero_item": dict(h)} for h in HITS][:limit]
    }
    monkeypatch.setattr(
        "zotero_mcp.semantic_search.create_semantic_search",
        MagicMock(return_value=fake_sem),
    )
    config_dir = tmp_path / ".config" / "zotero-mcp"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text("{}")
    monkeypatch.setattr(search_module.Path, "home", lambda: tmp_path)
    return fake_sem


def _search(**kwargs):
    return search_module.search_items(query="Nonexistent Paper 2099", ctx=DummyContext(), **kwargs)


def test_fallback_applies_tag_filter(semantic_only):
    result = _search(tag="thesis", limit=10)
    assert "Tagged Article" in result and "Tagged Book" in result
    assert "Untagged Article" not in result


def test_fallback_tag_filter_supports_exclusion_and_or(semantic_only):
    result = _search(tag=["thesis", "-draft"], limit=10)
    assert "Draft Article" not in result and "Tagged Article" in result
    result = _search(tag=["ai OR draft"], limit=10)
    assert "Tagged Article" in result and "Draft Article" in result
    assert "Tagged Book" not in result


def test_fallback_applies_item_type_filter(semantic_only):
    result = _search(item_type="book", limit=10)
    assert "Tagged Book" in result
    assert "Tagged Article" not in result and "Untagged Article" not in result


def test_fallback_filters_leave_limit_intact(semantic_only):
    """The index is asked for more than `limit`, so filtering does not
    shrink the answer below it."""
    result = _search(tag="thesis", limit=2)
    assert result.count("**Item Key:**") == 2
    assert semantic_only.search.call_args.kwargs["limit"] > 2


def test_fallback_with_no_match_reports_none(semantic_only):
    result = _search(tag="nonexistent-tag", limit=10)
    assert result.startswith("No items found matching query")
    assert "nonexistent-tag" in result


def test_fallback_unfiltered_search_is_unchanged(semantic_only):
    result = _search(limit=10)
    for title in ("Tagged Article", "Untagged Article", "Tagged Book"):
        assert title in result
    assert semantic_only.search.call_args.kwargs["limit"] == 10


def test_fallback_wildcard_tag_is_not_guessed_at(semantic_only):
    assert _search(tag="the*", limit=10).startswith("No items found")


def test_fallback_note_follows_the_tag_header(semantic_only):
    lines = _search(tag="thesis", limit=10).splitlines()
    tag_line = next(i for i, l in enumerate(lines) if "with tags: 'thesis'" in l)
    note_line = next(i for i, l in enumerate(lines) if l.startswith("*Note:"))
    assert tag_line < note_line
    assert lines[0].startswith("# Search Results")
