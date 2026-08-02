from zotero_mcp import server


class DummyContext:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


class FakeZotero:
    def __init__(self, items):
        self._items = items

    def items(self, start=0, limit=100, **_kwargs):
        return self._items[start : start + limit]


def test_advanced_search_filters_items(monkeypatch):
    fake_items = [
        {
            "key": "AAA11111",
            "data": {
                "itemType": "journalArticle",
                "title": "Quantum Networks and Learning",
                "date": "2024",
                "creators": [{"firstName": "Jane", "lastName": "Doe"}],
                "tags": [{"tag": "physics"}],
            },
        },
        {
            "key": "BBB22222",
            "data": {
                "itemType": "journalArticle",
                "title": "Classical Literature Review",
                "date": "2018",
                "creators": [{"firstName": "Alex", "lastName": "Smith"}],
                "tags": [{"tag": "history"}],
            },
        },
        {
            "key": "CCC33333",
            "data": {
                "itemType": "attachment",
                "title": "Ignored Attachment",
                "date": "2024",
                "creators": [],
                "tags": [],
            },
        },
    ]
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: FakeZotero(fake_items))

    result = server.advanced_search(
        conditions=[
            {"field": "title", "operation": "contains", "value": "quantum"},
            {"field": "year", "operation": "isGreaterThan", "value": "2020"},
        ],
        join_mode="all",
        limit=10,
        ctx=DummyContext(),
    )

    assert "Quantum Networks and Learning" in result
    assert "Classical Literature Review" not in result
    assert "Ignored Attachment" not in result


def test_advanced_search_rejects_unknown_operation(monkeypatch):
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: FakeZotero([]))

    result = server.advanced_search(
        conditions=[{"field": "title", "operation": "regex", "value": ".*"}],
        ctx=DummyContext(),
    )

    assert "Unsupported operation" in result


# ---------------------------------------------------------------------------
# Collection conditions (#418)
# ---------------------------------------------------------------------------

def _collection_items():
    """Two items in the target collection, one outside it, one in none."""
    return [
        {
            "key": "AAA11111",
            "data": {
                "itemType": "journalArticle", "title": "In Scope One",
                "date": "2024", "creators": [],
                "tags": [{"tag": "_ai-noted"}],
                "collections": ["MSYFGVKG"],
            },
        },
        {
            "key": "BBB22222",
            "data": {
                "itemType": "journalArticle", "title": "In Scope Two",
                "date": "2023", "creators": [],
                "tags": [{"tag": "_ai-noted"}],
                # Also filed elsewhere — membership is a list, not a scalar.
                "collections": ["OTHERKEY", "MSYFGVKG"],
            },
        },
        {
            "key": "CCC33333",
            "data": {
                "itemType": "journalArticle", "title": "Out Of Scope",
                "date": "2022", "creators": [],
                "tags": [{"tag": "_ai-noted"}],
                "collections": ["OTHERKEY"],
            },
        },
        {
            "key": "DDD44444",
            "data": {
                "itemType": "journalArticle", "title": "Unfiled Item",
                "date": "2021", "creators": [], "tags": [],
                "collections": [],
            },
        },
    ]


def test_collection_condition_matches_membership(monkeypatch):
    """The reporter's case B: a collection condition ANDed with a tag.

    Membership is stored in data["collections"], but extraction read a
    non-existent data["collection"], so every collection condition compared
    against "" and matched nothing.
    """
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client",
                        lambda: FakeZotero(_collection_items()))

    result = server.advanced_search(
        conditions=[
            {"field": "collection", "operation": "is", "value": "MSYFGVKG"},
            {"field": "tag", "operation": "contains", "value": "_ai-noted"},
        ],
        join_mode="all",
        limit=500,
        ctx=DummyContext(),
    )

    assert "In Scope One" in result
    assert "In Scope Two" in result
    assert "Out Of Scope" not in result
    assert "Unfiled Item" not in result


def test_collection_condition_alone(monkeypatch):
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client",
                        lambda: FakeZotero(_collection_items()))

    result = server.advanced_search(
        conditions=[{"field": "collection", "operation": "is", "value": "MSYFGVKG"}],
        ctx=DummyContext(),
    )

    assert "In Scope One" in result and "In Scope Two" in result
    assert "Out Of Scope" not in result


def test_collection_is_not_excludes_members_and_keeps_unfiled(monkeypatch):
    """`isNot` must keep an item that is in no collection at all."""
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client",
                        lambda: FakeZotero(_collection_items()))

    result = server.advanced_search(
        conditions=[{"field": "collection", "operation": "isNot", "value": "MSYFGVKG"}],
        ctx=DummyContext(),
    )

    assert "In Scope One" not in result
    assert "In Scope Two" not in result
    assert "Out Of Scope" in result
    assert "Unfiled Item" in result


def test_collections_plural_is_accepted(monkeypatch):
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client",
                        lambda: FakeZotero(_collection_items()))

    result = server.advanced_search(
        conditions=[{"field": "collections", "operation": "is", "value": "OTHERKEY"}],
        ctx=DummyContext(),
    )

    assert "Out Of Scope" in result
    assert "In Scope Two" in result
    assert "In Scope One" not in result


# ---------------------------------------------------------------------------
# Sorting (#418)
# ---------------------------------------------------------------------------

def _dated(key, title, date, *, added=None, parsed=None):
    data = {
        "itemType": "journalArticle", "title": title, "date": date,
        "creators": [], "tags": [{"tag": "t"}],
    }
    if added:
        data["dateAdded"] = added
    item = {"key": key, "data": data}
    if parsed:
        item["meta"] = {"parsedDate": parsed}
    return item


def _order(result):
    return [
        line.split(". ", 1)[1]
        for line in result.splitlines()
        if line.startswith("## ") and line[3].isdigit()
    ]


def _run_sorted(monkeypatch, items, **kwargs):
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client",
                        lambda: FakeZotero(items))
    return server.advanced_search(
        conditions=[{"field": "tag", "operation": "contains", "value": "t"}],
        limit=10, ctx=DummyContext(), **kwargs,
    )


def test_sort_by_date_uses_parsed_date_not_the_display_string(monkeypatch):
    """data["date"] is a display string, so a lexical sort orders by month name."""
    items = [
        _dated("1", "Jan-2020", "January 1, 2020", parsed="2020-01-01"),
        _dated("2", "Oct-2016", "October 1, 2016", parsed="2016-10-01"),
        _dated("3", "Mar-2024", "March 5, 2024", parsed="2024-03-05"),
    ]
    result = _run_sorted(monkeypatch, items, sort_by="date", sort_direction="desc")
    assert _order(result) == ["Mar-2024", "Jan-2020", "Oct-2016"]


def test_sort_by_date_falls_back_to_the_year_without_parsed_date(monkeypatch):
    """Sparse records carry no meta.parsedDate; sort by year, not month name."""
    items = [
        _dated("1", "Jan-2020", "January 1, 2020"),
        _dated("2", "Oct-2016", "October 1, 2016"),
        _dated("3", "Mar-2024", "March 5, 2024"),
    ]
    result = _run_sorted(monkeypatch, items, sort_by="date", sort_direction="desc")
    assert _order(result) == ["Mar-2024", "Jan-2020", "Oct-2016"]


def test_sort_by_date_added_descending(monkeypatch):
    items = [
        _dated("1", "Old", "2007", added="2007-01-01T00:00:00Z"),
        _dated("2", "Mid", "2015", added="2015-06-01T00:00:00Z"),
        _dated("3", "New", "2024", added="2024-09-09T00:00:00Z"),
    ]
    result = _run_sorted(monkeypatch, items, sort_by="dateAdded", sort_direction="desc")
    assert _order(result) == ["New", "Mid", "Old"]
    assert "was not applied" not in result


def test_unapplied_sort_is_reported_instead_of_silently_ignored(monkeypatch):
    """A field absent from every result sorted every key to "" and returned
    library order as though the sort had been honored."""
    items = [
        _dated("1", "Alpha", "2020"),
        _dated("2", "Beta", "2016"),
        _dated("3", "Gamma", "2024"),
    ]
    result = _run_sorted(monkeypatch, items, sort_by="dateAdded", sort_direction="desc")
    assert "was not applied" in result
    assert "`dateAdded`" in result
    assert _order(result) == ["Alpha", "Beta", "Gamma"]


def test_misspelled_sort_field_is_reported(monkeypatch):
    items = [_dated("1", "Alpha", "2020"), _dated("2", "Beta", "2016")]
    result = _run_sorted(monkeypatch, items, sort_by="notAField")
    assert "was not applied" in result


def test_no_sort_requested_emits_no_note(monkeypatch):
    items = [_dated("1", "Alpha", "2020"), _dated("2", "Beta", "2016")]
    result = _run_sorted(monkeypatch, items)
    assert "was not applied" not in result


def test_partially_present_sort_field_still_sorts(monkeypatch):
    """Only some records carrying the field is not a reason to refuse."""
    items = [
        _dated("1", "NoDate", "2020"),
        _dated("2", "Newer", "2016", added="2024-01-01T00:00:00Z"),
        _dated("3", "Older", "2024", added="2010-01-01T00:00:00Z"),
    ]
    result = _run_sorted(monkeypatch, items, sort_by="dateAdded", sort_direction="desc")
    assert "was not applied" not in result
    assert _order(result) == ["Newer", "Older", "NoDate"]
