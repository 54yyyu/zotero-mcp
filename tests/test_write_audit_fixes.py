"""Regressions from the 0.17.3 write-path audit."""

from conftest import DummyContext

from zotero_mcp.tools import _helpers, write


class _VersionedZotero:
    """One item; like the real API, a write with a stale version is refused."""

    def __init__(self, collections=(), tags=(), extra=""):
        self.version = 1
        self.data = {"key": "ITEM0001", "itemType": "journalArticle",
                     "collections": list(collections),
                     "tags": [{"tag": t} for t in tags], "extra": extra}
        self.gets = 0

    def item(self, key):
        self.gets += 1
        return {"key": key, "version": self.version,
                "data": {**self.data, "version": self.version}}

    def addto_collection(self, coll_key, item):
        if item["version"] != self.version:
            raise RuntimeError("412 Precondition Failed")
        self.data["collections"] = item["data"]["collections"] + [coll_key]
        self.version += 1
        return True


def test_filing_into_several_collections_survives_the_version_bump():
    zot = _VersionedZotero(collections=["AAAA1111"])
    failed = _helpers.ensure_collection_membership(
        zot, "ITEM0001", ["AAAA1111", "BBBB2222", "CCCC3333"], DummyContext())
    assert failed == []
    assert zot.data["collections"] == ["AAAA1111", "BBBB2222", "CCCC3333"]
    assert zot.gets == 2  # one read up front, one after the first add


def _hybrid(monkeypatch, write_data):
    """Local reads, web writes: the web copy holds what the desktop has not synced."""
    from test_batch_write_requests import CountingZot, _install, _items

    read_zot = CountingZot(_items(1))
    web = _items(1)
    web[0]["version"] = 77
    web[0]["data"].update(write_data)
    write_zot = CountingZot(web)
    _install(monkeypatch, read_zot, write_zot)
    return write_zot


def test_batch_tag_edit_keeps_tags_only_the_write_side_has(monkeypatch):
    write_zot = _hybrid(monkeypatch, {"tags": [{"tag": "old"}, {"tag": "x", "type": 1}]})

    result = write.batch_update_tags(
        item_keys=["K0000000"], add_tags=["new"], remove_tags=["old"], ctx=DummyContext())

    assert "Items updated: 1" in result
    written, = write_zot.batches[0]
    assert written["data"]["tags"] == [{"tag": "x", "type": 1}, {"tag": "new"}]


def test_batch_extra_edit_keeps_lines_only_the_write_side_has(monkeypatch):
    write_zot = _hybrid(monkeypatch, {"extra": "PMID: 123"})

    write.batch_update_extra(
        item_keys=["K0000000"], set_keys={"tex.x": "1"}, ctx=DummyContext())

    written, = write_zot.batches[0]
    assert written["data"]["extra"].splitlines() == ["PMID: 123", "tex.x: 1"]


# --- zotero_set_item_collections: one read and one write per 50 items ---


def _collections_zot(n, monkeypatch, **kwargs):
    from test_batch_write_requests import CountingZot, _install, _items

    items = _items(n)
    for it in items:
        it["data"]["collections"] = ["OLDCOLL1"]
    zot = CountingZot(items, **kwargs)
    zot.collections = lambda **kw: [] if kw.get("start") else [
        {"key": k, "data": {"name": k, "parentCollection": False}}
        for k in ("OLDCOLL1", "NEWCOLL1")]
    _install(monkeypatch, zot)
    return zot


def test_set_item_collections_batches_reads_and_writes(monkeypatch):
    zot = _collections_zot(100, monkeypatch)

    result = write.manage_collections(
        item_keys=[f"K{i:07d}" for i in range(100)], add_to=["NEWCOLL1"],
        remove_from=["OLDCOLL1"], ctx=DummyContext())

    assert zot.requests == 4  # 2 itemKey reads + 2 POSTs; it was 400
    assert all(it["data"]["collections"] == ["NEWCOLL1"] for it in zot._items.values())
    assert result.count("Added ") == 100 and result.count("Removed ") == 100


def test_set_item_collections_reports_every_item_when_one_fails(monkeypatch):
    """A key that could not be read used to abort the call with a bare
    error, hiding that earlier items had already been filed."""
    zot = _collections_zot(2, monkeypatch, refuse={
        "K0000001": {"key": "K0000001", "code": 400, "message": "bad collection"}})
    real_item = zot.item
    zot.item = lambda key: real_item(key) if key in zot._items else (_ for _ in ()).throw(
        RuntimeError("404 Not Found"))

    result = write.manage_collections(
        item_keys=["K0000000", "MISSING1", "K0000001"], add_to=["NEWCOLL1"],
        ctx=DummyContext())

    assert result.splitlines() == [
        "Added K0000000 to NEWCOLL1",
        "Failed: item MISSING1 not found",
        "Failed to add K0000001 to NEWCOLL1: bad collection",
    ]
    assert zot._items["K0000000"]["data"]["collections"] == ["OLDCOLL1", "NEWCOLL1"]


def test_set_item_collections_skips_the_write_when_nothing_changes(monkeypatch):
    zot = _collections_zot(1, monkeypatch)

    result = write.manage_collections(
        item_keys=["K0000000"], add_to=["OLDCOLL1"], remove_from=["NEWCOLL1"],
        ctx=DummyContext())

    assert result.splitlines() == ["K0000000 already in OLDCOLL1", "K0000000 not in NEWCOLL1"]
    assert zot.batches == [] and zot.update_item_calls == []
