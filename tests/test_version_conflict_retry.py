"""Tests for A6: bounded refetch-and-replay retry on HTTP 412 (stale
version) writes.

_handle_write_response only ever checked response status/success; nothing
caught pyzotero's PreConditionFailedError, which update_item raises when
the version sent no longer matches the server's current version. The
codebase already re-fetches the item once before writing (to pick up the
web API's version), but that only closes the common case — a concurrent
writer can still update the item again between that re-fetch and this
write. _update_item_with_version_retry closes that narrower race with a
bounded retry: re-fetch, re-apply the mutation, re-send.
"""

import copy

import pytest
from conftest import DummyContext
from pyzotero.zotero_errors import PreConditionFailedError

from zotero_mcp.tools import _helpers, write


class FakeVersionedZot:
    """Fake write client whose update_item raises PreConditionFailedError
    for the first *fail_times* calls, then succeeds."""

    def __init__(self, item, fail_times=0):
        self._item = item
        self.fetch_count = 0
        self.update_calls = []
        self.fail_times = fail_times

    def item(self, item_key):
        self.fetch_count += 1
        # A fresh copy each time, like a real re-fetch — mutate_fn must not
        # be able to smuggle state across attempts via a shared reference.
        return copy.deepcopy(self._item)

    def update_item(self, item):
        self.update_calls.append(copy.deepcopy(item))
        if len(self.update_calls) <= self.fail_times:
            raise PreConditionFailedError("stale version")
        return True


@pytest.fixture
def dummy_ctx():
    return DummyContext()


class TestUpdateItemWithVersionRetry:
    def test_succeeds_on_first_attempt_without_retry(self, dummy_ctx):
        zot = FakeVersionedZot({"key": "ABC1", "version": 1, "data": {}}, fail_times=0)

        result = _helpers._update_item_with_version_retry(
            zot, "ABC1", lambda it: it["data"].__setitem__("tags", ["x"]), ctx=dummy_ctx,
        )

        assert result is True
        assert zot.fetch_count == 1
        assert len(zot.update_calls) == 1
        assert zot.update_calls[0]["data"]["tags"] == ["x"]

    def test_retries_and_succeeds_after_version_conflicts(self, dummy_ctx):
        zot = FakeVersionedZot({"key": "ABC1", "version": 1, "data": {}}, fail_times=2)

        result = _helpers._update_item_with_version_retry(
            zot, "ABC1", lambda it: it["data"].__setitem__("tags", ["x"]), ctx=dummy_ctx,
        )

        assert result is True
        # Failed twice, succeeded on the 3rd attempt — each attempt re-fetches.
        assert zot.fetch_count == 3
        assert len(zot.update_calls) == 3
        # mutate_fn was re-applied to each freshly-fetched item, not just the first.
        assert all(c["data"]["tags"] == ["x"] for c in zot.update_calls)

    def test_raises_after_exhausting_retries(self, dummy_ctx):
        zot = FakeVersionedZot({"key": "ABC1", "version": 1, "data": {}}, fail_times=99)

        with pytest.raises(PreConditionFailedError):
            _helpers._update_item_with_version_retry(
                zot, "ABC1", lambda it: it["data"].__setitem__("tags", ["x"]), ctx=dummy_ctx,
            )

        assert zot.fetch_count == _helpers._MAX_VERSION_CONFLICT_RETRIES
        assert len(zot.update_calls) == _helpers._MAX_VERSION_CONFLICT_RETRIES

    def test_non_412_errors_propagate_without_retry(self, dummy_ctx):
        class ExplodingZot:
            def __init__(self):
                self.fetch_count = 0

            def item(self, item_key):
                self.fetch_count += 1
                return {"key": item_key, "version": 1, "data": {}}

            def update_item(self, item):
                raise RuntimeError("network is down")

        zot = ExplodingZot()

        with pytest.raises(RuntimeError, match="network is down"):
            _helpers._update_item_with_version_retry(
                zot, "ABC1", lambda it: None, ctx=dummy_ctx,
            )

        assert zot.fetch_count == 1


# ---------------------------------------------------------------------------
# Integration: the hybrid web-API branch (write_zot is not zot) in
# batch_update_tags / batch_update_extra — untested by existing suites,
# which all exercise web-only mode where write_zot IS zot.
# ---------------------------------------------------------------------------

class FakeReadZot:
    """Local-mode read client: search/list only, no update_item."""

    def __init__(self, items):
        self._items = items

    def add_parameters(self, **kwargs):
        pass

    def items(self):
        return self._items

    def item(self, item_key):
        for it in self._items:
            if it["key"] == item_key:
                return copy.deepcopy(it)
        raise KeyError(item_key)


class TestBatchUpdateTagsHybridMode:
    def test_retries_through_to_success_on_transient_conflict(self, monkeypatch, dummy_ctx):
        item = {
            "key": "ITEM1",
            "version": 1,
            "data": {"itemType": "journalArticle", "tags": [{"tag": "old"}]},
        }
        read_zot = FakeReadZot([item])
        write_zot = FakeVersionedZot(item, fail_times=1)

        # batch_update_tags fetches its search/read client directly via
        # client.get_zotero_client(), separately from the write client it
        # gets via _get_write_client() — both need patching so write_zot
        # (used for the actual update) differs from zot (used for search).
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: read_zot)
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client",
            lambda ctx: (read_zot, write_zot),
        )

        result = write.batch_update_tags(
            query="anything", add_tags=["new"], ctx=dummy_ctx,
        )

        assert "Items updated: 1" in result
        assert write_zot.fetch_count == 2
        assert len(write_zot.update_calls) == 2
        final_tags = {t["tag"] for t in write_zot.update_calls[-1]["data"]["tags"]}
        assert final_tags == {"old", "new"}

    def test_reports_skip_when_retries_exhausted(self, monkeypatch, dummy_ctx):
        item = {
            "key": "ITEM1",
            "version": 1,
            "data": {"itemType": "journalArticle", "tags": []},
        }
        read_zot = FakeReadZot([item])
        write_zot = FakeVersionedZot(item, fail_times=99)

        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: read_zot)
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client",
            lambda ctx: (read_zot, write_zot),
        )

        result = write.batch_update_tags(
            query="anything", add_tags=["new"], ctx=dummy_ctx,
        )

        assert "Items updated: 0" in result
        assert "Items skipped: 1" in result


class TestBatchUpdateExtraHybridMode:
    def test_retries_through_to_success_on_transient_conflict(self, monkeypatch, dummy_ctx):
        item = {
            "key": "ITEM1",
            "version": 1,
            "data": {"itemType": "journalArticle", "extra": ""},
        }
        read_zot = FakeReadZot([item])
        write_zot = FakeVersionedZot(item, fail_times=1)

        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: read_zot)
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client",
            lambda ctx: (read_zot, write_zot),
        )

        result = write.batch_update_extra(
            item_keys=["ITEM1"], set_keys={"tex.otscore": "2"}, ctx=dummy_ctx,
        )

        assert "Items updated: 1" in result
        assert write_zot.fetch_count == 2
        assert write_zot.update_calls[-1]["data"]["extra"] == "tex.otscore: 2"
