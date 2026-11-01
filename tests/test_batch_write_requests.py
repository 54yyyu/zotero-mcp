"""Request counts for the batch tag / Extra edits.

They used to fetch every explicit key with ``zot.item(key)`` and write every
edited item with its own ``update_item`` PATCH: 1000 items cost 2000 round
trips (3000 in hybrid mode). Keys are now fetched 50 per ``itemKey`` request
and written 50 per ``update_items`` POST, with the per-item path kept for
version conflicts, refused objects and chunks the backend will not batch.
"""

import copy

import pytest
from conftest import DummyContext
from pyzotero.errors import PreConditionFailedError

from zotero_mcp.tools import write


class _Response:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload


class CountingZot:
    """Behaves like a pyzotero client for the calls the batch edits make,
    counting each one as the HTTP request it would be."""

    local = False

    def __init__(self, items, *, child_notes=False, trashed=(), refuse=None,
                 batch_raises=None):
        self._items = {it["key"]: it for it in items}
        self.child_notes = child_notes
        self.trashed = set(trashed)
        self.refuse = refuse or {}  # key -> failure dict, first batch only
        self.batch_raises = batch_raises
        self.requests = 0
        self.item_calls = 0
        self.update_item_calls = []
        self.batches = []
        self.request = None

    def items(self, start=0, limit=100, itemKey=None, **kwargs):
        self.requests += 1
        rows = []
        if itemKey is not None:
            for key in itemKey.split(","):
                it = self._items.get(key)
                if it is None or key in self.trashed:
                    continue
                rows.append(it)
                if self.child_notes:
                    rows.append({"key": key[:4] + "NOTE", "data": {"itemType": "note"}})
        else:
            rows = list(self._items.values())
        return copy.deepcopy(rows[start:start + limit])

    def item(self, key):
        self.requests += 1
        self.item_calls += 1
        return copy.deepcopy(self._items[key])

    def update_item(self, item):
        self.requests += 1
        self.update_item_calls.append(copy.deepcopy(item))
        self._items[item["key"]] = copy.deepcopy(item)
        return True

    def update_items(self, payload):
        assert len(payload) <= 50
        self.requests += 1
        if self.batch_raises is not None:
            raise self.batch_raises
        self.batches.append(copy.deepcopy(payload))
        failed = {}
        for idx, item in enumerate(payload):
            if item["key"] in self.refuse:
                failed[str(idx)] = self.refuse.pop(item["key"])
            else:
                self._items[item["key"]] = copy.deepcopy(item)
        self.request = _Response({"success": {}, "unchanged": {}, "failed": failed})
        return True


def _items(n, item_type="journalArticle"):
    return [
        {
            "key": f"K{i:07d}",
            "version": 1,
            "data": {"key": f"K{i:07d}", "itemType": item_type, "tags": [], "extra": ""},
        }
        for i in range(n)
    ]


def _install(monkeypatch, read_zot, write_zot=None):
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._get_write_client",
        lambda ctx: (read_zot, write_zot or read_zot),
    )


def _keys(n):
    return [f"K{i:07d}" for i in range(n)]


def test_tags_for_1000_keys_take_40_requests_not_2000(monkeypatch):
    zot = CountingZot(_items(1000))
    _install(monkeypatch, zot)

    result = write.batch_update_tags(
        item_keys=_keys(1000), add_tags=["new"], ctx=DummyContext(),
    )

    assert "Items updated: 1000" in result
    assert zot.item_calls == 0 and zot.update_item_calls == []
    assert len(zot.batches) == 20
    assert zot.requests == 40
    assert all(t == [{"tag": "new"}] for t in
               (it["data"]["tags"] for it in zot._items.values()))


def test_extra_for_1000_keys_take_40_requests(monkeypatch):
    zot = CountingZot(_items(1000))
    _install(monkeypatch, zot)

    result = write.batch_update_extra(
        item_keys=_keys(1000), set_keys={"tex.x": "1"}, ctx=DummyContext(),
    )

    assert "Items updated: 1000" in result
    assert zot.requests == 40
    assert zot._items["K0000999"]["data"]["extra"] == "tex.x: 1"


def test_child_notes_page_and_are_dropped(monkeypatch):
    # itemKey answers with the items' child notes too: 50 keys -> 100 rows,
    # a full page, so the fetch pages once more. The notes are not edited.
    zot = CountingZot(_items(50), child_notes=True)
    _install(monkeypatch, zot)

    result = write.batch_update_tags(
        item_keys=_keys(50), add_tags=["new"], ctx=DummyContext(),
    )

    assert "Items processed: 50" in result
    assert "Items updated: 50" in result
    assert [len(b) for b in zot.batches] == [50]


def test_trashed_key_missing_from_the_batch_is_fetched_alone(monkeypatch):
    zot = CountingZot(_items(3), trashed={"K0000001"})
    _install(monkeypatch, zot)

    result = write.batch_update_tags(
        item_keys=_keys(3), add_tags=["new"], ctx=DummyContext(),
    )

    assert "Items updated: 3" in result
    assert zot.item_calls == 1


def test_unknown_key_is_reported_and_left_out(monkeypatch):
    zot = CountingZot(_items(2))
    _install(monkeypatch, zot)
    errors = []
    ctx = DummyContext()
    ctx.error = lambda msg, *a, **k: errors.append(msg)

    result = write.batch_update_extra(
        item_keys=["K0000000", "NOPE0000", "K0000001"], set_keys={"a": "b"}, ctx=ctx,
    )

    assert "Items processed: 3" in result
    assert "Items updated: 2" in result
    assert "Items skipped: 1" in result
    assert any("NOPE0000" in e for e in errors)


def test_version_conflict_in_a_batch_is_retried_per_item(monkeypatch):
    zot = CountingZot(_items(3), refuse={"K0000001": {"key": "K0000001", "code": 412,
                                                      "message": "modified"}})
    _install(monkeypatch, zot)

    result = write.batch_update_tags(
        item_keys=_keys(3), add_tags=["new"], ctx=DummyContext(),
    )

    assert "Items updated: 3" in result
    assert zot.item_calls == 1  # re-fetched for its current version
    assert [c["key"] for c in zot.update_item_calls] == ["K0000001"]


def test_refused_object_is_counted_and_reported(monkeypatch):
    zot = CountingZot(_items(3), refuse={"K0000002": {"key": "K0000002", "code": 400,
                                                      "message": "bad field"}})
    _install(monkeypatch, zot)
    errors = []
    ctx = DummyContext()
    ctx.error = lambda msg, *a, **k: errors.append(msg)

    result = write.batch_update_extra(
        item_keys=_keys(3), set_keys={"a": "b"}, ctx=ctx,
    )

    assert "Items updated: 2" in result
    assert "Items skipped: 1" in result
    assert any("K0000002" in e and "bad field" in e for e in errors)


def test_chunk_the_backend_rejects_is_written_one_at_a_time(monkeypatch):
    zot = CountingZot(_items(3), batch_raises=RuntimeError("405 Method Not Allowed"))
    _install(monkeypatch, zot)

    result = write.batch_update_tags(
        item_keys=_keys(3), add_tags=["new"], ctx=DummyContext(),
    )

    assert "Items updated: 3" in result
    assert len(zot.update_item_calls) == 3


def test_attachments_and_notes_are_skipped_by_extra(monkeypatch):
    items = _items(2) + [
        {"key": "ATT00001", "version": 1,
         "data": {"key": "ATT00001", "itemType": "attachment", "extra": ""}},
    ]
    zot = CountingZot(items)
    _install(monkeypatch, zot)

    result = write.batch_update_extra(
        item_keys=["K0000000", "ATT00001", "K0000001"], set_keys={"a": "b"},
        ctx=DummyContext(),
    )

    assert "Items updated: 2" in result
    assert "Items skipped: 1" in result
    assert zot._items["ATT00001"]["data"]["extra"] == ""


def test_hybrid_fetches_write_side_in_batches(monkeypatch):
    read_zot = CountingZot(_items(100))
    web = _items(100)
    for it in web:
        it["version"] = it["data"]["version"] = 77  # unrelated numbering
    write_zot = CountingZot(web)
    _install(monkeypatch, read_zot, write_zot)

    result = write.batch_update_tags(
        item_keys=_keys(100), add_tags=["new"], ctx=DummyContext(),
    )

    assert "Items updated: 100" in result
    assert read_zot.requests == 2
    assert write_zot.requests == 4  # 2 itemKey fetches + 2 POSTs
    assert all(it["version"] == 77 for b in write_zot.batches for it in b)


def test_hybrid_conflict_still_retries(monkeypatch):
    read_zot = CountingZot(_items(2))
    write_zot = CountingZot(_items(2))
    calls = {"n": 0}
    real_update = write_zot.update_item

    def flaky(item):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PreConditionFailedError("stale")
        return real_update(item)

    write_zot.update_item = flaky
    write_zot.refuse = {"K0000000": {"key": "K0000000", "code": 412, "message": "x"}}
    _install(monkeypatch, read_zot, write_zot)

    result = write.batch_update_tags(
        item_keys=_keys(2), add_tags=["new"], ctx=DummyContext(),
    )

    assert "Items updated: 2" in result
    assert calls["n"] == 2


@pytest.mark.parametrize("n,posts", [(1, 1), (50, 1), (51, 2), (1000, 20)])
def test_query_selection_writes_in_chunks_of_50(monkeypatch, n, posts):
    zot = CountingZot(_items(n))
    _install(monkeypatch, zot)

    write.batch_update_tags(
        query="x", add_tags=["new"], limit=1000, ctx=DummyContext(),
    )

    assert len(zot.batches) == posts
    assert zot.update_item_calls == []
