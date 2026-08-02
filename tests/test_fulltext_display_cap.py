"""The inline-read page cap is independent of the indexing page cap.

``pdf_max_pages`` is sized so extraction never becomes the binding limit on
what the semantic index can see — deliberately far above the ~8 pages that
embedding token limits actually admit. ``zotero_get_item_fulltext`` is bound
by something else entirely: an agent's context. If the display cap inherited
the indexing cap, raising the latter would quietly start dumping dozens of
pages of Markdown into a conversation.
"""

import types

from zotero_mcp.tools import retrieval


def _patch_extraction_config(monkeypatch, **fields):
    """Stand in for the on-disk config's ``extraction`` section."""
    extraction = types.SimpleNamespace(
        **{"pdf_max_pages": None, "fulltext_display_max_pages": None, **fields}
    )
    monkeypatch.setattr(
        retrieval,
        "load_config",
        lambda: types.SimpleNamespace(
            semantic_search=types.SimpleNamespace(extraction=extraction)
        ),
    )


class TestFulltextDisplayMaxPages:
    def test_defaults_when_nothing_is_configured(self, monkeypatch):
        _patch_extraction_config(monkeypatch)
        assert retrieval._fulltext_display_max_pages() == retrieval.DEFAULT_FULLTEXT_DISPLAY_MAX

    def test_configured_value_is_honored(self, monkeypatch):
        _patch_extraction_config(monkeypatch, fulltext_display_max_pages=3)
        assert retrieval._fulltext_display_max_pages() == 3

    def test_does_not_inherit_the_indexing_cap(self, monkeypatch):
        """The regression this module exists for."""
        _patch_extraction_config(monkeypatch, pdf_max_pages=200)
        assert retrieval._fulltext_display_max_pages() == retrieval.DEFAULT_FULLTEXT_DISPLAY_MAX

    def test_display_cap_wins_over_the_indexing_cap(self, monkeypatch):
        _patch_extraction_config(
            monkeypatch, pdf_max_pages=200, fulltext_display_max_pages=5
        )
        assert retrieval._fulltext_display_max_pages() == 5

    def test_unreadable_config_falls_back_to_the_default(self, monkeypatch):
        def _boom():
            raise OSError("config unreadable")

        monkeypatch.setattr(retrieval, "load_config", _boom)
        assert retrieval._fulltext_display_max_pages() == retrieval.DEFAULT_FULLTEXT_DISPLAY_MAX

    def test_the_default_stays_modest(self):
        """A guard on intent: this bounds an agent's context window, so it
        must not drift upward alongside the indexing cap."""
        assert retrieval.DEFAULT_FULLTEXT_DISPLAY_MAX <= 15
