"""The incremental (since-version) sync must not lose a changed item.

Two ways it did: a request that failed for one changed item was skipped while
the sync version still moved past it, and a PDF attached to an item already
indexed changed only the attachment, so its parent was never indexed again.
"""

import json
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from pyzotero.zotero_errors import ResourceNotFoundError
from test_fulltext_web_mode import (
    FakeChromaClient,
    FakeZoteroClient,
    _build_search,
    _paper,
    _pdf,
    _write_config,
)


class KeyFilteringClient(FakeZoteroClient):
    """A client whose ``items`` honours ``itemKey``, as the Zotero API does."""

    def items(self, start=0, limit=100, **kwargs):
        if "itemKey" in kwargs:
            self.calls.append(("items", kwargs["itemKey"]))
            keys = kwargs["itemKey"].split(",")
            return [self.items_by_key[k] for k in keys if k in self.items_by_key]
        return super().items(start=start, limit=limit, **kwargs)


def _saved_version(config_path):
    return json.loads(open(config_path).read())["semantic_search"].get("last_sync_versions", {}).get("0")


def _indexed(chroma):
    return [doc_id for batch in chroma.added for doc_id in batch[2]]


def test_a_failed_item_fetch_keeps_the_sync_version(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, extra={"last_sync_version": 5, "include_fulltext": False})
    zot = KeyFilteringClient()
    zot.load_scenario([_paper("OLD"), _paper("NEW"), _paper("FLAKY")], library_version=8)
    zot.versions_state = {"OLD": 3, "NEW": 8, "FLAKY": 8}
    real_item, real_items = zot.item, zot.items
    outage = {"on": True}

    def flaky_item(key):
        if key == "FLAKY" and outage["on"]:
            raise RuntimeError("503 Service Unavailable")
        return real_item(key)

    def flaky_items(start=0, limit=100, **kwargs):
        if "FLAKY" in kwargs.get("itemKey", "") and outage["on"]:
            raise RuntimeError("503 Service Unavailable")
        return real_items(start=start, limit=limit, **kwargs)

    zot.item, zot.items = flaky_item, flaky_items
    chroma = FakeChromaClient(preloaded_ids=["OLD"])

    stats = _build_search(monkeypatch, zot, chroma, config_path=config_path).update_database()

    assert stats["fetch_failed_items"] == 1
    assert _indexed(chroma) == ["NEW"]
    assert _saved_version(config_path) != 8

    # The next run asks for the same changes again and gets the item.
    outage["on"] = False
    stats = _build_search(monkeypatch, zot, chroma, config_path=config_path).update_database()
    assert "fetch_failed_items" not in stats
    assert "FLAKY" in _indexed(chroma)
    assert _saved_version(config_path) == 8


def test_an_item_gone_from_zotero_does_not_hold_the_sync_version(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, extra={"last_sync_version": 5, "include_fulltext": False})
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("NEW")], library_version=8)
    zot.versions_state = {"NEW": 8, "GONE": 8}
    real_item = zot.item

    def item(key):
        if key == "GONE":
            raise ResourceNotFoundError("404")
        return real_item(key)

    zot.item = item
    chroma = FakeChromaClient()

    stats = _build_search(monkeypatch, zot, chroma, config_path=config_path).update_database()

    assert "fetch_failed_items" not in stats
    assert _indexed(chroma) == ["NEW"]
    assert _saved_version(config_path) == 8


def test_a_pdf_attached_later_reindexes_its_parent(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, extra={"last_sync_version": 5})
    zot = FakeZoteroClient()
    attachment = _pdf("ATT1", parent="P1")
    zot.load_scenario(
        [_paper("P1"), attachment],
        fulltext={"ATT1": {"content": "the full body text"}},
        children={"P1": [attachment]},
        library_version=8,
    )
    zot.versions_state = {"P1": 3, "ATT1": 8}  # only the attachment is new
    chroma = FakeChromaClient(preloaded_ids=["P1"])

    _build_search(monkeypatch, zot, chroma, config_path=config_path).update_database()

    assert _indexed(chroma) == ["P1"]
    assert "the full body text" in chroma.added[0][0][0]
    assert chroma.added[0][1][0]["has_fulltext"] is True
    assert _saved_version(config_path) == 8


def test_text_extracted_later_reindexes_the_parent(monkeypatch, tmp_path):
    """Zotero extracts a PDF's text some time after the file arrives; that
    changes neither the item nor the attachment, only /fulltext?since=."""
    config_path = _write_config(tmp_path, extra={"last_sync_version": 5})
    zot = FakeZoteroClient()
    attachment = _pdf("ATT1", parent="P1")
    zot.load_scenario(
        [_paper("P1"), attachment],
        fulltext={"ATT1": {"content": "text that arrived late"}},
        children={"P1": [attachment]},
        library_version=8,
    )
    zot.versions_state = {"P1": 3, "ATT1": 3}
    zot.new_fulltext = lambda since: {"ATT1": 8} if since < 8 else {}
    chroma = FakeChromaClient(preloaded_ids=["P1"])

    _build_search(monkeypatch, zot, chroma, config_path=config_path).update_database()

    assert _indexed(chroma) == ["P1"]
    assert "text that arrived late" in chroma.added[0][0][0]


def test_an_attachment_change_is_ignored_when_text_is_not_indexed(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, extra={"last_sync_version": 5, "include_fulltext": False})
    zot = FakeZoteroClient()
    attachment = _pdf("ATT1", parent="P1")
    zot.load_scenario([_paper("P1"), attachment], children={"P1": [attachment]}, library_version=8)
    zot.versions_state = {"P1": 3, "ATT1": 8}
    chroma = FakeChromaClient(preloaded_ids=["P1"])

    _build_search(monkeypatch, zot, chroma, config_path=config_path).update_database()

    assert _indexed(chroma) == []


def test_changed_items_are_fetched_fifty_to_a_request(monkeypatch):
    zot = KeyFilteringClient()
    zot.load_scenario([_paper(f"K{i:03d}") for i in range(120)], library_version=9)
    search = _build_search(monkeypatch, zot, FakeChromaClient())
    zot.calls.clear()

    changed, _ = search._get_changed_items_from_api(since_version=0, include_fulltext=False)

    assert [item["key"] for item in changed] == [f"K{i:03d}" for i in range(120)]
    names = [call[0] for call in zot.calls]
    assert names.count("items") == 3
    assert names.count("item") == 0


def test_a_server_that_ignores_item_key_still_gets_the_right_items(monkeypatch):
    """The base double returns its first page whatever ``itemKey`` says."""
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("A"), _paper("B"), _paper("C")], library_version=9)
    zot.versions_state = {"B": 9, "C": 9, "A": 2}
    zot.items_order = ["A"]  # the unfiltered page holds neither changed item
    search = _build_search(monkeypatch, zot, FakeChromaClient())

    changed, _ = search._get_changed_items_from_api(since_version=5, include_fulltext=False)

    assert sorted(item["key"] for item in changed) == ["B", "C"]


def test_a_failed_batch_request_falls_back_to_one_request_per_item(monkeypatch):
    zot = KeyFilteringClient()
    zot.load_scenario([_paper("A"), _paper("B")], library_version=9)
    real_items = zot.items

    def items(start=0, limit=100, **kwargs):
        if "itemKey" in kwargs:
            raise RuntimeError("400 Bad Request")
        return real_items(start=start, limit=limit, **kwargs)

    zot.items = items
    search = _build_search(monkeypatch, zot, FakeChromaClient())

    found, failed = search._fetch_items_by_keys(["A", "B", "A"])

    assert sorted(found) == ["A", "B"]
    assert failed == set()
