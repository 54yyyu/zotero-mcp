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
