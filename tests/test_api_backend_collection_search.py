"""ApiBackend's collection-scoped search must send the query to the server.

5df683f moved this branch behind the backend and lost q/qmode on the way, so
a scoped search returned the collection's first N items whatever was asked.
"""

from zotero_mcp.library import ApiBackend


def _item(key, item_type="journalArticle"):
    return {"key": key, "data": {"key": key, "itemType": item_type}}


class RecordingZotero:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def collection_items(self, key, **params):
        self.calls.append(params)
        start = params.get("start", 0)
        return self.pages.get(start, [])


def test_query_and_mode_reach_the_server():
    zot = RecordingZotero({0: [_item("AAAA1111")]})
    ApiBackend(zot).search_items("attention", qmode="everything", collection_keys=["COLL0001"])
    assert zot.calls[0]["q"] == "attention"
    assert zot.calls[0]["qmode"] == "everything"


def test_child_notes_do_not_spend_the_limit():
    """A full first page of note-content matches is neither exhaustion nor
    a filled budget in titleCreatorYear mode (#542)."""
    notes = [_item(f"NOTE{i:04d}", "note") for i in range(100)]
    zot = RecordingZotero({0: notes, 100: [_item("PAPER001")]})
    found = ApiBackend(zot).search_items("attention", limit=5, collection_keys=["COLL0001"])
    assert [i["key"] for i in found] == ["PAPER001"]
