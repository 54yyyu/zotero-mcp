"""Tests for verified file attachment (#403).

``attachment_both()`` reports a client-side rejection by returning the
payload in its ``failure`` list rather than raising, so every call site
that only guarded against exceptions reported "File attached" for a file
that never landed — the root cause behind the silent-attach reports in
#278 / #306 / #399.
"""

import pytest
from conftest import DummyContext, FakeZotero

from zotero_mcp.tools import _helpers


@pytest.fixture
def dummy_ctx():
    return DummyContext()


def _ok_result(key="ATCH0001"):
    return {"success": [{"key": key}], "failure": [], "unchanged": []}


def _failed_result(filename="/abs/path/paper.pdf"):
    """The shape pyzotero returns on the #403 client-side failure.

    The reporter's repro shows the full filesystem path landing in
    ``filename`` and ``md5`` unset on the failed payload.
    """
    return {
        "success": [],
        "failure": [{"filename": filename, "md5": None}],
        "unchanged": [],
    }


class _AttachFake(FakeZotero):
    """FakeZotero with the attach/upload surface ``_attach_and_verify`` uses."""

    local = False

    def __init__(self, attach_result=None):
        super().__init__()
        self._attach_result = attach_result if attach_result is not None else _ok_result()
        self.attach_calls = []
        self.created = []
        self.uploaded = []
        self.deleted = []
        self._items = {}

    def attachment_both(self, files, parentid=None, **kwargs):
        self.attach_calls.append((files, parentid))
        return self._attach_result

    def item_template(self, item_type, linkmode=None, **kwargs):
        return {"itemType": item_type, "linkMode": linkmode, "title": "", "filename": ""}

    def create_items(self, payload):
        self.created.append(payload)
        key = "NEWATCH1"
        self._items[key] = dict(payload[0], key=key, md5=None)
        return {"success": {"0": key}, "successVersions": {"0": 7}, "failed": {}}

    def item(self, key, **kwargs):
        return {"key": key, "version": 7, "data": self._items[key]}

    def upload_attachments(self, attachments, **kwargs):
        self.uploaded.append(attachments)
        for att in attachments:
            self._items[att["key"]]["md5"] = "d41d8cd98f00b204e9800998ecf8427e"
        return {"success": [{"key": a["key"]} for a in attachments], "failure": []}

    def delete_item(self, item, **kwargs):
        self.deleted.append(item)


class TestDescribeAttachFailure:
    def test_none_when_attachment_registered(self):
        assert _helpers._describe_attach_failure(_ok_result()) is None

    def test_none_for_unchanged_entry(self):
        result = {"success": [], "failure": [], "unchanged": [{"key": "ATCH0002"}]}
        assert _helpers._describe_attach_failure(result) is None

    def test_reports_failure_payload(self):
        reason = _helpers._describe_attach_failure(_failed_result())
        assert reason is not None
        assert "rejected" in reason

    def test_reports_empty_result(self):
        # The pre-#403 fakes returned None here and callers read it as success.
        assert _helpers._describe_attach_failure(None) is not None

    def test_reports_result_with_no_key(self):
        result = {"success": [], "failure": [], "unchanged": []}
        assert _helpers._describe_attach_failure(result) is not None


class TestAssertUploadCapable:
    def test_rejects_local_client(self):
        zot = FakeZotero()
        zot.local = True
        with pytest.raises(ValueError, match="local API"):
            _helpers._assert_upload_capable(zot)

    def test_allows_web_client(self):
        zot = FakeZotero()
        zot.local = False
        _helpers._assert_upload_capable(zot)


class TestAttachAndVerify:
    def test_success_passes_through_key(self, dummy_ctx, monkeypatch):
        monkeypatch.setattr(_helpers, "_maybe_upload_to_webdav", lambda *a, **k: "")
        zot = _AttachFake()
        ok, suffix, key = _helpers._attach_and_verify(
            zot, "paper.pdf", "/abs/paper.pdf", "ITEM1", dummy_ctx
        )
        assert ok is True
        assert key == "ATCH0001"
        assert zot.created == []  # no fallback needed

    def test_failed_upload_falls_back_to_two_step(self, dummy_ctx, monkeypatch):
        monkeypatch.setattr(_helpers, "_maybe_upload_to_webdav", lambda *a, **k: "")
        zot = _AttachFake(attach_result=_failed_result())
        ok, suffix, key = _helpers._attach_and_verify(
            zot, "paper.pdf", "/abs/paper.pdf", "ITEM1", dummy_ctx
        )
        assert ok is True
        assert key == "NEWATCH1"
        # The item is created with the basename; only the upload step sees
        # the full path.
        assert zot.created[0][0]["filename"] == "paper.pdf"
        assert zot.uploaded[0][0]["filename"] == "/abs/paper.pdf"

    def test_reports_failure_when_two_step_also_fails(self, dummy_ctx, monkeypatch):
        monkeypatch.setattr(_helpers, "_maybe_upload_to_webdav", lambda *a, **k: "")

        class NoBytes(_AttachFake):
            def upload_attachments(self, attachments, **kwargs):
                # Reports success but never stores md5 — bytes didn't land.
                return {"success": [{"key": a["key"]} for a in attachments], "failure": []}

        zot = NoBytes(attach_result=_failed_result())
        ok, reason, key = _helpers._attach_and_verify(
            zot, "paper.pdf", "/abs/paper.pdf", "ITEM1", dummy_ctx
        )
        assert ok is False
        assert key is None
        assert "md5" in reason

    def test_orphan_shell_deleted_when_upload_fails(self, dummy_ctx, monkeypatch):
        monkeypatch.setattr(_helpers, "_maybe_upload_to_webdav", lambda *a, **k: "")

        class BoomUpload(_AttachFake):
            def upload_attachments(self, attachments, **kwargs):
                raise RuntimeError("network down")

        zot = BoomUpload(attach_result=_failed_result())
        ok, reason, _key = _helpers._attach_and_verify(
            zot, "paper.pdf", "/abs/paper.pdf", "ITEM1", dummy_ctx
        )
        assert ok is False
        assert zot.deleted == [{"key": "NEWATCH1", "version": 7}]

    def test_local_client_fails_fast(self, dummy_ctx):
        zot = _AttachFake()
        zot.local = True
        with pytest.raises(ValueError, match="local API"):
            _helpers._attach_and_verify(
                zot, "paper.pdf", "/abs/paper.pdf", "ITEM1", dummy_ctx
            )
        assert zot.attach_calls == []
