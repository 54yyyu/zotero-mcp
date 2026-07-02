"""Tests for zotero_attach_file (tools/write.attach_file) and its helpers."""

import pytest
from conftest import DummyContext, FakeZotero

from zotero_mcp.tools import _helpers


@pytest.fixture
def dummy_ctx():
    return DummyContext()


# ---------------------------------------------------------------------------
# _helpers._attachment_filename_exists
# ---------------------------------------------------------------------------


class TestAttachmentFilenameExists:
    def test_true_when_child_has_same_filename(self):
        zot = FakeZotero()
        zot._children["ITEM1"] = [
            {"data": {"itemType": "attachment", "filename": "paper.pdf"}}
        ]
        assert _helpers._attachment_filename_exists(zot, "ITEM1", "paper.pdf")

    def test_false_when_no_children(self):
        zot = FakeZotero()
        assert not _helpers._attachment_filename_exists(zot, "ITEM1", "paper.pdf")

    def test_false_when_different_filename(self):
        zot = FakeZotero()
        zot._children["ITEM1"] = [{"data": {"filename": "other.pdf"}}]
        assert not _helpers._attachment_filename_exists(zot, "ITEM1", "paper.pdf")

    def test_false_when_children_call_raises(self):
        class Boom(FakeZotero):
            def children(self, item_key, **kwargs):
                raise RuntimeError("api down")

        assert not _helpers._attachment_filename_exists(Boom(), "ITEM1", "paper.pdf")

    def test_tolerates_none_data(self):
        zot = FakeZotero()
        zot._children["ITEM1"] = [{"data": None}]
        assert not _helpers._attachment_filename_exists(zot, "ITEM1", "paper.pdf")
