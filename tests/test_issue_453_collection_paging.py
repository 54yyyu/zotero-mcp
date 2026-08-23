"""A collection larger than 100 items can be enumerated in full (#453).

`zotero_get_collection_items` capped at `_normalize_limit`'s default ceiling
of 100 and then told the caller to "increase the limit parameter to see more",
which is the one thing that could not help: past 100 the parameter did
nothing, and there was no offset. A 158-item collection could not be read.

`zotero_get_recent` shared the cap and was worse -- it truncated silently and
titled the result with the number of items that had been *asked* for.
"""

import pytest

from zotero_mcp.tools import retrieval as retrieval_tools
from zotero_mcp.tools._helpers import _normalize_limit


class DummyContext:
    def info(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _item(key):
    return {
        "key": key,
        "data": {"key": key, "itemType": "journalArticle", "title": f"Paper {key}",
                 "date": "2024"},
        "meta": {},
    }


class BigCollectionZotero:
    """A collection of *size* items, served 100 per request like the real API."""

    def __init__(self, size=158):
        self.items_list = [_item(f"K{i:04d}") for i in range(size)]

    def collection(self, key):
        return {"key": key, "data": {"name": "Banking and payments"}}

    def collection_items(self, key, start=0, limit=100, **kwargs):
        limit = min(limit or 100, 100)  # the API's own hard page size
        return self.items_list[start:start + limit]

    def items(self, start=0, limit=100, **kwargs):
        limit = min(limit or 100, 100)
        return self.items_list[start:start + limit]

    def collections(self, **kwargs):
        return []


@pytest.fixture
def big(monkeypatch):
    zot = BigCollectionZotero()
    monkeypatch.setattr(retrieval_tools._client, "get_zotero_client", lambda: zot)
    return zot


class TestReportedMatrix:
    """The cases from the report, against the 158-item collection it used."""

    def test_limit_101_returns_101_not_100(self, big):
        result = retrieval_tools.get_collection_items(
            collection_key="QS7TQPPA", detail="keys_only", limit=101,
            ctx=DummyContext(),
        )
        assert result.count("- `K") == 101
        assert "Showing items 1-101 of 158" in result

    def test_the_whole_collection_can_be_read_in_one_call(self, big):
        result = retrieval_tools.get_collection_items(
            collection_key="QS7TQPPA", detail="keys_only", limit=200,
            ctx=DummyContext(),
        )
        assert result.count("- `K") == 158
        # Nothing left over, so no paging footer.
        assert "offset=" not in result

    def test_limit_below_100_is_still_honoured(self, big):
        result = retrieval_tools.get_collection_items(
            collection_key="QS7TQPPA", detail="keys_only", limit=3,
            ctx=DummyContext(),
        )
        assert result.count("- `K") == 3

    def test_truncation_message_names_a_followable_next_step(self, big):
        result = retrieval_tools.get_collection_items(
            collection_key="QS7TQPPA", detail="keys_only", limit=50,
            ctx=DummyContext(),
        )
        assert "Showing items 1-50 of 158" in result
        assert "offset=50" in result
        # The advice that could not be followed is gone.
        assert "Increase the limit parameter" not in result

    @pytest.mark.parametrize("detail", ["keys_only", "summary", "full"])
    def test_cap_is_lifted_at_every_detail_level(self, big, detail):
        """The report confirmed the cap was in retrieval, not formatting."""
        result = retrieval_tools.get_collection_items(
            collection_key="QS7TQPPA", detail=detail, limit=123,
            ctx=DummyContext(),
        )
        assert "Showing items 1-123 of 158" in result


class TestPaging:
    def test_offset_returns_the_next_page(self, big):
        page2 = retrieval_tools.get_collection_items(
            collection_key="QS7TQPPA", detail="keys_only", limit=50, offset=50,
            ctx=DummyContext(),
        )
        assert "Showing items 51-100 of 158" in page2
        assert "K0050" in page2
        assert "K0049" not in page2
        assert "offset=100" in page2

    def test_the_pages_together_cover_the_collection_exactly_once(self, big):
        seen = []
        offset = 0
        while True:
            page = retrieval_tools.get_collection_items(
                collection_key="QS7TQPPA", detail="keys_only", limit=50,
                offset=offset, ctx=DummyContext(),
            )
            keys = [ln.split("`")[1] for ln in page.splitlines() if ln.startswith("- `")]
            seen.extend(keys)
            if "offset=" not in page:
                break
            offset = int(page.split("offset=")[1].split(" ")[0].rstrip(".*"))

        assert len(seen) == 158
        assert len(set(seen)) == 158

    def test_numbering_reflects_position_in_the_collection(self, big):
        page2 = retrieval_tools.get_collection_items(
            collection_key="QS7TQPPA", detail="summary", limit=10, offset=100,
            ctx=DummyContext(),
        )
        # Not "## 1." -- the reader must be able to tell where they are.
        assert "## 101." in page2

    def test_offset_past_the_end_says_so(self, big):
        result = retrieval_tools.get_collection_items(
            collection_key="QS7TQPPA", detail="keys_only", limit=50, offset=500,
            ctx=DummyContext(),
        )
        assert "No items at offset 500" in result
        assert "158" in result


class TestGetRecent:
    def test_recent_beyond_100_is_not_silently_truncated(self, big):
        result = retrieval_tools.get_recent(limit=105, ctx=DummyContext())
        assert result.startswith("# 105 Most Recently Added Items")

    def test_heading_reports_what_came_back_not_what_was_asked(self, monkeypatch):
        """Asking for more than the library holds must not claim it returned
        that many -- the report saw "# 100 Most Recently Added Items" for a
        request of 105 that had actually been cut off."""
        zot = BigCollectionZotero(size=12)
        monkeypatch.setattr(retrieval_tools._client, "get_zotero_client", lambda: zot)

        result = retrieval_tools.get_recent(limit=105, ctx=DummyContext())
        assert result.startswith("# 12 Most Recently Added Items")


class TestNormalizeLimit:
    def test_zero_falls_back_to_the_default_not_to_one_item(self):
        """`limit=0` answered with exactly 1 item, which reads as a one-item
        collection rather than as a rejected argument (#453)."""
        assert _normalize_limit(0, default=50) == 50
        assert _normalize_limit(-1, default=50) == 50

    def test_ordinary_values_are_unchanged(self):
        assert _normalize_limit(25, default=50, max_val=100) == 25
        assert _normalize_limit(None, default=50) == 50
        assert _normalize_limit("25", default=50) == 25
        assert _normalize_limit(999, default=50, max_val=100) == 100


class TestPaginateMaxItems:
    """`_paginate(max_items=N)` returns at most N, on every exit path.

    The cap was applied inside the loop, after the `len(batch) < page_size`
    break -- so a run that ended on a short final page returned everything it
    had fetched. Every caller passing max_items to bound a response inherited
    that.
    """

    @staticmethod
    def _method(total):
        pool = list(range(total))

        def fetch(start=0, limit=100, **kwargs):
            return pool[start:start + limit]

        return fetch

    @pytest.mark.parametrize(
        "total,max_items,expected",
        [
            (158, 105, 105),   # ends on a short page -- the regression
            (250, 105, 105),   # ends mid-page
            (200, 105, 105),   # exact multiple of the page size
            (50, 105, 50),     # fewer items than asked for
            (158, None, 158),  # no cap at all
            (158, 200, 158),   # cap above the total
        ],
    )
    def test_cap_is_respected(self, total, max_items, expected):
        from zotero_mcp.utils import _paginate

        got = _paginate(self._method(total), max_items=max_items)
        assert len(got) == expected
        assert got == list(range(expected))
