"""Headless tests for the Zotero TUI.

These drive the real Textual app via its ``run_test`` pilot against a stub
data layer (no live Zotero needed), verifying the core wiring: initial load,
table population, search dispatch, and write-action dispatch.

Wrapped with ``asyncio.run`` so they don't require the ``pytest-asyncio``
plugin to execute.
"""

import asyncio

import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable, Input  # noqa: E402

from zotero_mcp.tui.app import ZoteroTUI  # noqa: E402
from zotero_mcp.tui.data import Collection, ItemRow  # noqa: E402


def _row(key, item_type, title):
    return ItemRow(
        key=key, item_type=item_type, title=title,
        creators="Doe, J", year="2024", tags=1,
        raw={"key": key, "data": {"itemType": item_type, "title": title}},
    )


class FakeData:
    """Stub implementing the subset of ZoteroData the app calls."""

    def __init__(self):
        self.calls = []

    def set_log_sink(self, sink):
        pass

    def check_connection(self):
        return True, "Connected (fake)"

    def current_library(self):
        return "user:0"

    def collections(self):
        return [Collection("AAA", "Top", None), Collection("BBB", "Child", "AAA")]

    def recent(self, limit=50, collection_key=None):
        self.calls.append(("recent", limit, collection_key))
        return [_row("K1", "journalArticle", "First paper"),
                _row("K2", "preprint", "Second paper")]

    def collection_items(self, key, limit=200):
        self.calls.append(("collection_items", key))
        return [_row("C1", "book", "In collection")]

    def search(self, q, qmode="titleCreatorYear", limit=50):
        self.calls.append(("search", q))
        return [_row("S1", "journalArticle", "Search hit " + q)]

    def search_by_tag(self, tags, limit=50):
        self.calls.append(("tag", tuple(tags)))
        return [_row("T1", "book", "Tagged")]

    def semantic_search(self, q, limit=25):
        self.calls.append(("semantic", q))
        return [_row("M1", "preprint", "Semantic")]

    # detail panes
    def metadata(self, k):
        self.calls.append(("metadata", k))
        return f"# Meta {k}"

    def fulltext(self, k):
        return f"fulltext {k}"

    def annotations(self, k):
        return f"anns {k}"

    def notes(self, k):
        return f"notes {k}"

    def children(self, k):
        return f"children {k}"

    def bibtex(self, k):
        return f"@article{{{k}}}"

    # writes
    def add_by_doi(self, v, collections=None, tags=None):
        self.calls.append(("add_by_doi", v, collections, tags))
        return "Added DOI " + v

    def add_by_url(self, v, collections=None, tags=None):
        self.calls.append(("add_by_url", v, collections, tags))
        return "Added URL " + v

    def add_from_file(self, v, collections=None, tags=None):
        self.calls.append(("add_from_file", v, collections, tags))
        return "Added file " + v

    def update_item(self, key, **kw):
        self.calls.append(("update_item", key, kw))
        return f"Updated {key}"


async def _settle(app, pilot, rows_expected=True):
    for _ in range(20):
        await pilot.pause(0.1)
        await app.workers.wait_for_complete()
        if not rows_expected or app.query_one("#results", DataTable).row_count > 0:
            break


def test_initial_load_and_detail():
    fake = FakeData()
    app = ZoteroTUI(data=fake)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(app, pilot)
            table = app.query_one("#results", DataTable)
            assert table.row_count == 2
            # First row auto-selected -> metadata loaded into the cache.
            assert app.current_item == "K1"
            await app.workers.wait_for_complete()
            await pilot.pause(0.1)
            assert ("K1", "metadata") in app._detail_cache
            # Tree built with the two pseudo-nodes + top collection.
            labels = [str(n.label) for n in app.query_one("#collections").root.children]
            assert any("Recent" in s for s in labels)
            assert any("Top" in s for s in labels)

    asyncio.run(scenario())


def test_search_and_tab_and_action_dispatch():
    fake = FakeData()
    app = ZoteroTUI(data=fake)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(app, pilot)

            # Lazy tab load: switch to BibTeX -> cached for current item.
            app.query_one("#detail").active = "tab-bibtex"
            await pilot.pause(0.1)
            await app.workers.wait_for_complete()
            await pilot.pause(0.1)
            assert ("K1", "bibtex") in app._detail_cache

            # Keyword search dispatch (focus the input first, then submit).
            app.action_focus_search()
            await pilot.pause(0.05)
            app.query_one("#query", Input).value = "genome"
            await pilot.press("enter")
            await _settle(app, pilot)
            assert any(c[0] == "search" and c[1] == "genome" for c in fake.calls)

            # Add-by-DOI write action dispatch.
            await pilot.press("a")
            await pilot.pause(0.2)
            app.screen.query_one("#field-value", Input).value = "10.1/x"
            await pilot.click("#submit")
            await pilot.pause(0.2)
            await app.workers.wait_for_complete()
            await pilot.pause(0.2)
            assert any(c[0] == "add_by_doi" and c[1] == "10.1/x" for c in fake.calls)

    asyncio.run(scenario())
