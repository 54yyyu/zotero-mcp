"""`zotero_advanced_search` no longer reads the whole library (#456).

The client-side path paged the entire library 100 items at a time and
evaluated every condition in Python, with no server-side filter, no early
exit and no bound. On a large library the call hit the client's 300s idle
timeout, and because it holds the process-global Zotero API lock for its whole
duration, every other tool queued behind it reported "Zotero API busy".

The reported repro was `itemType is blogPost` with `limit=3` against a local
library, where the equivalent `curl` returned in under a second.
"""


import pytest

from zotero_mcp.tools import search as search_tools
from zotero_mcp.tools.search import (
    _ADVANCED_SCAN_BUDGET_SECONDS,
    _canonical_item_type,
    _server_side_filters,
)


class DummyContext:
    def info(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _item(key, item_type="journalArticle", **fields):
    data = {"key": key, "itemType": item_type, "title": f"Item {key}"}
    data.update(fields)
    return {"key": key, "data": data, "meta": {}}


class RecordingZotero:
    """Serves a library of *size* items, recording every call it receives."""

    def __init__(self, library, page_size=100):
        self.library = library
        self.page_size = page_size
        self.calls = []

    def items(self, start=0, limit=100, **kwargs):
        self.calls.append({"start": start, "limit": limit, **kwargs})
        pool = self.library
        # Emulate the API's own server-side filters.
        if "itemType" in kwargs:
            pool = [i for i in pool if i["data"]["itemType"] == kwargs["itemType"]]
        if "tag" in kwargs:
            pool = [i for i in pool if kwargs["tag"] in i["data"].get("_tags", [])]
        return pool[start:start + limit]

    def collections(self, **kwargs):
        return []


@pytest.fixture
def patched(monkeypatch):
    def _install(library):
        zot = RecordingZotero(library)
        monkeypatch.setattr(search_tools._client, "get_zotero_client", lambda: zot)
        monkeypatch.setattr(search_tools._utils, "get_search_backend", lambda: "api")
        return zot
    return _install


# ---------------------------------------------------------------------------
# Server-side filter selection
# ---------------------------------------------------------------------------

class TestServerSideFilters:
    def test_item_type_is_pushed_to_the_api(self):
        conds = [{"field": "itemType", "operation": "is", "value": "blogPost"}]
        assert _server_side_filters(conds, "all") == {"itemType": "blogPost"}

    def test_item_type_is_canonicalised_before_being_pushed(self):
        """The API matches exactly; search_semantics matches case-insensitively.
        Pushing the user's casing verbatim would turn a match into no results."""
        conds = [{"field": "itemType", "operation": "is", "value": "blogpost"}]
        assert _server_side_filters(conds, "all") == {"itemType": "blogPost"}

    def test_unknown_item_type_is_not_pushed(self):
        """A typo must fall through to the client-side path, which can still
        match nothing -- rather than to an API filter that guarantees nothing."""
        conds = [{"field": "itemType", "operation": "is", "value": "blogPoast"}]
        assert _server_side_filters(conds, "all") == {}

    def test_any_join_mode_pushes_nothing(self):
        """Under OR a server-side filter would drop items another condition
        matches, silently turning the OR into an AND."""
        conds = [
            {"field": "itemType", "operation": "is", "value": "blogPost"},
            {"field": "title", "operation": "contains", "value": "x"},
        ]
        assert _server_side_filters(conds, "any") == {}

    def test_non_is_operations_are_not_pushed(self):
        conds = [{"field": "itemType", "operation": "isNot", "value": "blogPost"}]
        assert _server_side_filters(conds, "all") == {}

    def test_tag_is_pushed(self):
        conds = [{"field": "tag", "operation": "is", "value": "to-read"}]
        assert _server_side_filters(conds, "all") == {"tag": "to-read"}

    def test_canonical_item_type_rejects_unknown(self):
        assert _canonical_item_type("journalArticle") == "journalArticle"
        assert _canonical_item_type("JOURNALARTICLE") == "journalArticle"
        assert _canonical_item_type("nonsense") is None
        assert _canonical_item_type("") is None


def _advance_clock_past_budget(zot, monkeypatch):
    """Make the scan's clock jump past its budget after the first page.

    `search_tools._time` is the stdlib `time` module itself, so a fake that
    reads `time.monotonic()` would call whatever it just replaced. This one
    holds its own counter and never touches the real clock.
    """
    real_items = zot.items
    clock = {"now": 0.0}
    monkeypatch.setattr(search_tools._time, "monotonic", lambda: clock["now"])

    def slow_items(*a, **k):
        clock["now"] += _ADVANCED_SCAN_BUDGET_SECONDS + 1
        return real_items(*a, **k)

    zot.items = slow_items


# ---------------------------------------------------------------------------
# The reported repro
# ---------------------------------------------------------------------------

class TestReportedRepro:
    def test_item_type_search_does_not_read_the_whole_library(self, patched):
        """#456: itemType is blogPost, limit 3, against a large library."""
        library = [_item(f"A{i}") for i in range(5000)]
        library += [_item(f"B{i}", item_type="blogPost") for i in range(10)]
        zot = patched(library)

        result = search_tools.advanced_search(
            conditions=[{"field": "itemType", "operation": "is", "value": "blogPost"}],
            limit=3,
            ctx=DummyContext(),
        )

        # The API did the filtering, so one page was enough -- not 50.
        assert len(zot.calls) == 1, f"expected 1 request, made {len(zot.calls)}"
        assert zot.calls[0]["itemType"] == "blogPost"
        assert "Found 3 items" in result

    def test_early_exit_once_the_limit_is_met(self, patched):
        """With no sort, results are in library order either way, so nothing
        later in the library can displace the first `limit` matches."""
        library = [_item(f"A{i}", title="match me") for i in range(5000)]
        zot = patched(library)

        search_tools.advanced_search(
            conditions=[{"field": "title", "operation": "contains", "value": "match"}],
            limit=5,
            ctx=DummyContext(),
        )

        assert len(zot.calls) == 1

    def test_a_sort_still_sees_every_match(self, patched):
        """Ordering cannot be decided from a prefix, so the early exit is off
        when sort_by is set -- correctness beats the round trips."""
        library = [_item(f"A{i:04d}", title=f"match me {i:04d}") for i in range(250)]
        zot = patched(library)

        result = search_tools.advanced_search(
            conditions=[{"field": "title", "operation": "contains", "value": "match"}],
            limit=5,
            sort_by="title",
            sort_direction="desc",
            ctx=DummyContext(),
        )

        assert len(zot.calls) == 3  # 250 items over 100-item pages
        # Sorted descending, so the highest-numbered key leads.
        assert "A0249" in result


# ---------------------------------------------------------------------------
# The bound
# ---------------------------------------------------------------------------

class TestScanIsBounded:
    def test_a_slow_walk_returns_partial_results_and_says_so(self, patched, monkeypatch):
        library = [_item(f"A{i}") for i in range(10000)]
        library[0]["data"]["title"] = "needle"
        zot = patched(library)

        # Every page costs more than the whole budget, so the walk is cut
        # after its first page rather than reading 100 of them.
        _advance_clock_past_budget(zot, monkeypatch)

        result = search_tools.advanced_search(
            conditions=[{"field": "title", "operation": "contains", "value": "needle"}],
            limit=50,
            sort_by="title",  # disables the early exit, forcing the budget path
            ctx=DummyContext(),
        )

        assert len(zot.calls) < 100
        assert "partial" in result.lower()
        assert "ZOTERO_SEARCH_BACKEND=sqlite" in result

    def test_empty_result_after_truncation_does_not_claim_the_library_is_empty(
        self, patched, monkeypatch
    ):
        """"No items found" after a truncated scan is a different claim from
        "no items found" after a complete one, and must not be conflated."""
        library = [_item(f"A{i}") for i in range(10000)]
        zot = patched(library)
        _advance_clock_past_budget(zot, monkeypatch)

        result = search_tools.advanced_search(
            conditions=[
                {"field": "title", "operation": "contains", "value": "no-such-thing"}
            ],
            limit=50,
            sort_by="title",
            ctx=DummyContext(),
        )

        assert "portion of the library that was searched" in result

    def test_budget_stays_under_the_api_lock_wait_bound(self):
        """A search that outlives the 45s lock wait turns one slow call into
        "Zotero API busy" on every other tool -- the #456 collateral damage."""
        assert _ADVANCED_SCAN_BUDGET_SECONDS < 45


class TestNoBehaviourChangeOnSmallLibraries:
    def test_complete_scan_carries_no_warning(self, patched):
        library = [_item(f"A{i}") for i in range(10)]
        library[3]["data"]["title"] = "needle here"
        patched(library)

        result = search_tools.advanced_search(
            conditions=[{"field": "title", "operation": "contains", "value": "needle"}],
            limit=50,
            ctx=DummyContext(),
        )

        assert "Found 1 items" in result
        assert "partial" not in result.lower()

    def test_no_match_is_still_the_plain_message(self, patched):
        patched([_item(f"A{i}") for i in range(10)])

        result = search_tools.advanced_search(
            conditions=[{"field": "title", "operation": "contains", "value": "zzz"}],
            limit=50,
            ctx=DummyContext(),
        )

        assert result == "No items found matching the search criteria."
