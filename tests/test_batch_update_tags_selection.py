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
    """Pages like the API: honours start/limit, records the last request."""

    def __init__(self, items):
        self._items = items
        self.params = {}
        self.requests = 0
        self.updated = []

    def items(self, start=0, limit=100, **kwargs):
        self.params = kwargs
        self.requests += 1
        return copy.deepcopy(self._items[start:start + limit])

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


def _install(monkeypatch, items):
    zot = FakeZot(items)
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (zot, zot)
    )
    return zot


@pytest.fixture
def fake(monkeypatch):
    return _install(monkeypatch, [ATTACHMENT])


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


# ---------------------------------------------------------------------------
# limit: a query/tag selection is paged, and a cut-short one says so
# ---------------------------------------------------------------------------


def _tagged(n):
    return [
        {"key": f"ITEM{i:04d}",
         "data": {"itemType": "journalArticle", "tags": [{"tag": "old"}]}}
        for i in range(n)
    ]


def test_a_limit_past_one_page_is_honoured(monkeypatch, dummy_ctx):
    zot = _install(monkeypatch, _tagged(250))

    result = write.batch_update_tags(
        tag="old", remove_tags=["old"], limit=500, ctx=dummy_ctx,
    )

    assert "Items updated: 250" in result
    assert zot.requests == 3
    assert "More items matched" not in result


def test_a_selection_cut_short_by_limit_says_so(monkeypatch, dummy_ctx):
    zot = _install(monkeypatch, _tagged(3))

    result = write.batch_update_tags(
        tag="old", remove_tags=["old"], limit=2, ctx=dummy_ctx,
    )

    assert "Items updated: 2" in result
    assert len(zot.updated) == 2
    assert "More items matched than limit=2" in result


def test_a_selection_that_exactly_fills_limit_is_not_reported_as_cut(monkeypatch, dummy_ctx):
    _install(monkeypatch, _tagged(2))

    result = write.batch_update_tags(
        tag="old", remove_tags=["old"], limit=2, ctx=dummy_ctx,
    )

    assert "More items matched" not in result


def test_search_item_keys_reports_more_matched():
    keys, more = write._search_item_keys(FakeZot(_tagged(3)), "", "old", 2)
    assert keys == ["ITEM0000", "ITEM0001"]
    assert more is True
