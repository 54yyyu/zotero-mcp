"""SqliteBackend.collection_items must answer like the API does.

The API's /collections/<key>/items includes the child attachments and notes of
the items filed in the collection, and zotero_get_collection_items builds its
per-item PDF/notes summary from them. collectionItems only holds top-level
items, so the SQLite backend returned 17 items where the API returned 52 on a
live library, and the summary went blank.
"""

from zotero_mcp.library import SqliteBackend


def _item(key, item_type="journalArticle", parent=None):
    data = {"key": key, "itemType": item_type}
    if parent:
        data["parentItem"] = parent
    return {"key": key, "version": 1, "data": data}


class _Reader:
    def __init__(self, items, children):
        self._items, self._children = items, children
        self.children_asked = []

    def get_collection_items(self, key, *, include_subcollections=False, group_id=None):
        return None if key == "MISSING0" else list(self._items)

    def get_children_of(self, keys, *, item_type=None, group_id=None):
        self.children_asked.append(list(keys))
        return {k: list(self._children.get(k, [])) for k in keys}


def test_children_of_filed_items_are_included_but_not_annotations():
    reader = _Reader(
        [_item("PAPER001")],
        {"PAPER001": [
            _item("ATT00001", "attachment", "PAPER001"),
            _item("NOTE0001", "note", "PAPER001"),
            _item("ANNO0001", "annotation", "PAPER001"),
        ]},
    )
    got = SqliteBackend(reader, 0).collection_items("COLL0001")
    assert [i["key"] for i in got] == ["PAPER001", "ATT00001", "NOTE0001"]
    assert reader.children_asked == [["PAPER001"]]


def test_missing_collection_stays_none_and_empty_stays_empty():
    reader = _Reader([], {})
    backend = SqliteBackend(reader, 0)
    assert backend.collection_items("MISSING0") is None
    assert backend.collection_items("COLL0001") == []
    assert reader.children_asked == []
