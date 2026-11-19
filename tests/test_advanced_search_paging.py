"""zotero_advanced_search on the SQLite backend: sort by the real date, report
every match, and hydrate only the page.

Three things were wrong. `sort_by="date"` read `meta.parsedDate`, which a
SQL-hydrated item never carries, and fell back to the first four-digit number
in the display date: a sort by year only. The result was cut to `limit` before
"Found N items" was written, so 13,000 matches read as "Found 50". And every
match was hydrated, creators and tags included, to show that page.
"""

import re

import pytest
from conftest import DummyContext

from _search_corpus import Item, build_sqlite
from zotero_mcp import client as _client
from zotero_mcp import server
from zotero_mcp import utils as _utils
from zotero_mcp.local_db import LocalZoteroReader
from zotero_mcp.tools import search as search_module

ITEMS = [
    Item("JAN20310", title="Sorttest alpha", date="2031-01-05 January 5, 2031",
         creators=[("Zed", "Young")], date_modified="2024-01-03 00:00:00"),
    Item("DEC20310", title="Sorttest charlie", date="2031-12-20 December 20, 2031",
         creators=[("Amy", "Adams")], date_modified="2024-01-01 00:00:00"),
    Item("JUN20310", title="Sorttest bravo", date="2031-06-15 2031-06-15",
         creators=[("Bo", "Miller")], date_modified="2024-01-04 00:00:00"),
    Item("YEAR2030", title="Sorttest delta", date="2030-00-00 2030",
         date_modified="2024-01-02 00:00:00"),
    Item("NODATE00", title="Sorttest echo", date_modified="2024-01-05 00:00:00"),
    Item("OTHER000", title="Unrelated"),
]
MATCH = [{"field": "title", "operation": "contains", "value": "Sorttest"}]


@pytest.fixture
def reader(tmp_path):
    db = tmp_path / "zotero.sqlite"
    build_sqlite(db, ITEMS)
    with LocalZoteroReader(db_path=str(db)) as r:
        yield r


def _keys(page):
    return [item["key"] for item in page.items]


def test_date_sort_orders_within_a_year(reader):
    asc = reader.advanced_search_page(MATCH, sort_by="date")
    assert _keys(asc) == ["NODATE00", "YEAR2030", "JAN20310", "JUN20310", "DEC20310"]
    desc = reader.advanced_search_page(MATCH, sort_by="date", sort_direction="desc", limit=2)
    assert _keys(desc) == ["DEC20310", "JUN20310"]
    assert desc.total == 5 and desc.sorted


def test_default_order_is_newest_modified_first_and_the_page_is_cut(reader):
    page = reader.advanced_search_page(MATCH, limit=2)
    assert _keys(page) == ["NODATE00", "JUN20310"]
    assert page.total == 5


def test_title_and_creator_sorts(reader):
    assert _keys(reader.advanced_search_page(MATCH, sort_by="title", sort_direction="desc")) == [
        "NODATE00", "YEAR2030", "DEC20310", "JUN20310", "JAN20310",
    ]
    # "Adams, Amy" < "Miller, Bo" < "No authors listed" < "Young, Zed", as the
    # client-side sort keys creators.
    by_creator = _keys(reader.advanced_search_page(MATCH, sort_by="creator"))
    assert by_creator[:2] == ["DEC20310", "JUN20310"] and by_creator[-1] == "JAN20310"


def test_sort_on_a_field_no_match_carries_is_reported(reader):
    page = reader.advanced_search_page(MATCH, sort_by="nonexistentField")
    assert not page.sorted
    assert _keys(page) == _keys(reader.advanced_search_page(MATCH))


def test_only_the_page_is_hydrated(reader):
    hydrated = []
    original = reader._hydrate_rows
    reader._hydrate_rows = lambda conn, rows: hydrated.extend(rows) or original(conn, rows)
    reader.advanced_search_page(MATCH, sort_by="date", limit=2)
    assert len(hydrated) == 2


def test_more_matches_than_one_id_slice(tmp_path):
    db = tmp_path / "zotero.sqlite"
    build_sqlite(db, [Item(f"BULK{n:04d}", title="Bulk paper") for n in range(1000)])
    with LocalZoteroReader(db_path=str(db)) as r:
        everything = r.advanced_search_sql(
            [{"field": "title", "operation": "contains", "value": "bulk"}]
        )
    assert len({item["key"] for item in everything}) == 1000


def _tool(monkeypatch, tmp_path, **kwargs):
    db = tmp_path / "zotero.sqlite"
    build_sqlite(db, ITEMS)
    monkeypatch.setattr(_client, "get_active_group_id", lambda: 0)
    monkeypatch.setattr(_utils, "get_search_backend", lambda: "sqlite")
    monkeypatch.setattr(
        search_module, "get_local_zotero_reader", lambda: LocalZoteroReader(db_path=str(db))
    )
    monkeypatch.setattr(_client, "get_zotero_client", lambda: object())
    return server.advanced_search(conditions=MATCH, ctx=DummyContext(), **kwargs)


def test_tool_lists_the_latest_dates_first_and_reports_the_total(monkeypatch, tmp_path):
    out = _tool(monkeypatch, tmp_path, sort_by="date", sort_direction="desc", limit=2)
    assert re.findall(r"\*\*Item Key:\*\* (\w+)", out) == ["DEC20310", "JUN20310"]
    assert "Found 5 items matching the search criteria; showing 2" in out


def test_tool_wording_is_unchanged_when_everything_is_shown(monkeypatch, tmp_path):
    out = _tool(monkeypatch, tmp_path)
    assert "Found 5 items matching the search criteria:" in out


def test_tool_reports_an_unusable_sort_field(monkeypatch, tmp_path):
    out = _tool(monkeypatch, tmp_path, sort_by="nonexistentField")
    assert "Requested sort by `nonexistentField` was not applied" in out
