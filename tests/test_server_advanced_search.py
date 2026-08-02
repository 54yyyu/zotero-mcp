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
