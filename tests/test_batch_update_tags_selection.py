"""Which items batch_update_tags edits, given its selector.

The tag edits used to skip every attachment, wherever it came from, so a
tag that only attachments carried (imported automatic tags, say) could
not be removed by tag or by key. A query alone still leaves attachments
out, now by asking the server, so they no longer use up the limit either.
"""

import copy

import pytest

from zotero_mcp.tools import write


class FakeZot:
    def __init__(self, items):
        self._items = items
        self.params = {}
        self.updated = []

    def add_parameters(self, **kwargs):
        self.params = kwargs

    def items(self, **kwargs):
        return copy.deepcopy(self._items)

    def item(self, key):
        return copy.deepcopy(next(it for it in self._items if it["key"] == key))

    def update_item(self, item):
        self.updated.append(item)
        return True


ATTACHMENT = {
    "key": "ATT00001",
    "data": {
        "itemType": "attachment",
        "tags": [{"tag": "Barrel organ, Hydraulic", "type": 1}],
    },
}


@pytest.fixture
def fake(monkeypatch):
    zot = FakeZot([ATTACHMENT])
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (zot, zot)
    )
    return zot


def test_tag_selection_removes_the_tag_from_an_attachment(fake, dummy_ctx):
    result = write.batch_update_tags(
        tag="Barrel organ, Hydraulic", remove_tags=["Barrel organ, Hydraulic"],
        ctx=dummy_ctx,
    )

    assert "Items updated: 1" in result
    assert [it["key"] for it in fake.updated] == ["ATT00001"]
    assert fake.updated[0]["data"]["tags"] == []
    assert "itemType" not in fake.params


def test_item_keys_reach_an_attachment(fake, dummy_ctx):
    result = write.batch_update_tags(
        item_keys=["ATT00001"], add_tags=["checked"], ctx=dummy_ctx,
    )

    assert "Items updated: 1" in result
    assert {t["tag"] for t in fake.updated[0]["data"]["tags"]} == {
        "Barrel organ, Hydraulic", "checked",
    }


def test_query_only_selection_asks_the_server_to_leave_attachments_out(fake, dummy_ctx):
    write.batch_update_tags(query="organ", add_tags=["x"], ctx=dummy_ctx)

    assert fake.params["q"] == "organ"
    assert fake.params["itemType"] == "-attachment"


def test_combined_batch_selection_follows_the_same_rule(dummy_ctx):
    zot = FakeZot([])
    write._search_item_keys(zot, "organ", None, 50)
    assert zot.params["itemType"] == "-attachment"

    write._search_item_keys(zot, "organ", "to-read", 50)
    assert "itemType" not in zot.params
