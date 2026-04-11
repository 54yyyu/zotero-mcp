"""Tests for update_annotation and delete_annotation tools."""

import json

from zotero_mcp import server


class DummyContext:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _annotation_item(key, text="highlighted text", comment="my comment",
                     color="#ffd400", anno_type="highlight", parent="ATT0001",
                     version=10):
    return {
        "key": key,
        "version": version,
        "data": {
            "key": key,
            "version": version,
            "itemType": "annotation",
            "parentItem": parent,
            "annotationType": anno_type,
            "annotationText": text,
            "annotationComment": comment,
            "annotationColor": color,
            "tags": [],
        },
    }


# ---------------------------------------------------------------------------
# Fake Zotero clients
# ---------------------------------------------------------------------------

class FakeZoteroForAnnotationUpdate:
    def __init__(self, items):
        self._items = items
        self.updated = []

    def item(self, key):
        if key not in self._items:
            raise KeyError(key)
        return self._items[key]

    def update_item(self, item):
        self.updated.append(item)
        return {"success": True}


class FakePatchResponse:
    def __init__(self, status_code=204, text=""):
        self.status_code = status_code
        self.text = text


class FakeHttpxClient:
    def __init__(self, status_code=204, text=""):
        self._status_code = status_code
        self._text = text
        self.calls = []

    def patch(self, url, headers, content):
        self.calls.append({"url": url, "headers": headers, "content": content})
        return FakePatchResponse(self._status_code, self._text)


class FakeZoteroForAnnotationDelete:
    def __init__(self, items, patch_status=204):
        self._items = items
        self.endpoint = "https://api.zotero.org"
        self.library_type = "users"
        self.library_id = "12345"
        self.client = FakeHttpxClient(status_code=patch_status)

    def item(self, key):
        if key not in self._items:
            raise KeyError(key)
        return self._items[key]


# ---------------------------------------------------------------------------
# update_annotation tests
# ---------------------------------------------------------------------------

def test_update_annotation_comment(monkeypatch):
    anno = _annotation_item("ANNO0001", comment="old comment")
    fake = FakeZoteroForAnnotationUpdate({"ANNO0001": anno})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.update_annotation(
        item_key="ANNO0001", comment="new comment", ctx=DummyContext()
    )

    assert "Successfully updated" in result
    assert fake.updated[0]["data"]["annotationComment"] == "new comment"


def test_update_annotation_color(monkeypatch):
    anno = _annotation_item("ANNO0001", color="#ffd400")
    fake = FakeZoteroForAnnotationUpdate({"ANNO0001": anno})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.update_annotation(
        item_key="ANNO0001", color="#ff6666", ctx=DummyContext()
    )

    assert "Successfully updated" in result
    assert fake.updated[0]["data"]["annotationColor"] == "#ff6666"


def test_update_annotation_text(monkeypatch):
    anno = _annotation_item("ANNO0001", text="old text")
    fake = FakeZoteroForAnnotationUpdate({"ANNO0001": anno})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.update_annotation(
        item_key="ANNO0001", text="new text", ctx=DummyContext()
    )

    assert "Successfully updated" in result
    assert fake.updated[0]["data"]["annotationText"] == "new text"


def test_update_annotation_multiple_fields(monkeypatch):
    anno = _annotation_item("ANNO0001")
    fake = FakeZoteroForAnnotationUpdate({"ANNO0001": anno})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.update_annotation(
        item_key="ANNO0001", comment="updated", color="#a28ae5", ctx=DummyContext()
    )

    assert "Successfully updated" in result
    data = fake.updated[0]["data"]
    assert data["annotationComment"] == "updated"
    assert data["annotationColor"] == "#a28ae5"


def test_update_annotation_no_fields_error(monkeypatch):
    anno = _annotation_item("ANNO0001")
    fake = FakeZoteroForAnnotationUpdate({"ANNO0001": anno})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.update_annotation(
        item_key="ANNO0001", ctx=DummyContext()
    )

    assert "No fields to update" in result
    assert fake.updated == []


def test_update_annotation_rejects_non_annotation(monkeypatch):
    note = {
        "key": "NOTE0001",
        "version": 1,
        "data": {"key": "NOTE0001", "version": 1, "itemType": "note", "note": "<p>x</p>"},
    }
    fake = FakeZoteroForAnnotationUpdate({"NOTE0001": note})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.update_annotation(
        item_key="NOTE0001", comment="x", ctx=DummyContext()
    )

    assert "is not an annotation" in result
    assert fake.updated == []


def test_update_annotation_missing_key(monkeypatch):
    fake = FakeZoteroForAnnotationUpdate({})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.update_annotation(
        item_key="ZZZZZZZZ", comment="x", ctx=DummyContext()
    )

    assert "No item found" in result
    assert fake.updated == []


# ---------------------------------------------------------------------------
# delete_annotation tests
# ---------------------------------------------------------------------------

def test_delete_annotation_trashes_via_patch(monkeypatch):
    anno = _annotation_item("ANNO0001", version=42)
    fake = FakeZoteroForAnnotationDelete({"ANNO0001": anno})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.delete_annotation(item_key="ANNO0001", ctx=DummyContext())

    assert "Successfully trashed" in result
    assert len(fake.client.calls) == 1
    call = fake.client.calls[0]
    assert "ANNO0001" in call["url"]
    assert call["headers"]["If-Unmodified-Since-Version"] == "42"
    assert '"deleted": 1' in call["content"]


def test_delete_annotation_rejects_non_annotation(monkeypatch):
    note = {
        "key": "NOTE0001",
        "version": 1,
        "data": {"key": "NOTE0001", "version": 1, "itemType": "note", "note": ""},
    }
    fake = FakeZoteroForAnnotationDelete({"NOTE0001": note})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.delete_annotation(item_key="NOTE0001", ctx=DummyContext())

    assert "is not an annotation" in result
    assert fake.client.calls == []


def test_delete_annotation_missing_key(monkeypatch):
    fake = FakeZoteroForAnnotationDelete({})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.delete_annotation(item_key="ZZZZZZZZ", ctx=DummyContext())

    assert "No item found" in result
    assert fake.client.calls == []


def test_delete_annotation_http_error(monkeypatch):
    anno = _annotation_item("ANNO0001", version=5)
    fake = FakeZoteroForAnnotationDelete({"ANNO0001": anno}, patch_status=412)
    fake.client._text = "Precondition failed"
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.delete_annotation(item_key="ANNO0001", ctx=DummyContext())

    assert "Failed to trash" in result
    assert "412" in result
