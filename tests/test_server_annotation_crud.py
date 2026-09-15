"""Tests for zotero_update_annotation and zotero_delete_annotation,
and for the `tags` parameter on zotero_create_annotation."""

from zotero_mcp import server


class DummyContext:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


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


def _annotation_item(key, *, text="old text", comment="", color="#ffd400", tags=None):
    return {
        "key": key,
        "version": 1,
        "data": {
            "key": key,
            "version": 1,
            "itemType": "annotation",
            "annotationType": "highlight",
            "parentItem": "ATTACH01",
            "annotationText": text,
            "annotationComment": comment,
            "annotationColor": color,
            "annotationSortIndex": "00000|000000|00000",
            "annotationPosition": "{}",
            "tags": [{"tag": t} for t in (tags or [])],
        },
    }


def _patch_client(monkeypatch, fake):
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)


def test_update_annotation_updates_text_comment_color(monkeypatch):
    fake = FakeZoteroForAnnotationUpdate(
        {"ANNO0001": _annotation_item("ANNO0001", text="old", comment="", color="#ffd400")}
    )
    _patch_client(monkeypatch, fake)

    result = server.update_annotation(
        annotation_key="ANNO0001",
        text="new highlight",
        comment="a thought",
        color="#ff0000",
        ctx=DummyContext(),
    )

    assert "Successfully updated" in result
    data = fake.updated[0]["data"]
    assert data["annotationText"] == "new highlight"
    assert data["annotationComment"] == "a thought"
    assert data["annotationColor"] == "#ff0000"


def test_update_annotation_replaces_tags(monkeypatch):
    fake = FakeZoteroForAnnotationUpdate(
        {"ANNO0001": _annotation_item("ANNO0001", tags=["old1", "old2"])}
    )
    _patch_client(monkeypatch, fake)

    result = server.update_annotation(
        annotation_key="ANNO0001",
        tags=["fresh"],
        ctx=DummyContext(),
    )

    assert "Successfully updated" in result
    assert fake.updated[0]["data"]["tags"] == [{"tag": "fresh"}]


def test_update_annotation_adds_and_removes_tags(monkeypatch):
    fake = FakeZoteroForAnnotationUpdate(
        {"ANNO0001": _annotation_item("ANNO0001", tags=["keep", "drop"])}
    )
    _patch_client(monkeypatch, fake)

    result = server.update_annotation(
        annotation_key="ANNO0001",
        add_tags=["new"],
        remove_tags=["drop"],
        ctx=DummyContext(),
    )

    assert "Successfully updated" in result
    final_tags = {t["tag"] for t in fake.updated[0]["data"]["tags"]}
    assert final_tags == {"keep", "new"}


def test_update_annotation_rejects_tags_with_add_tags(monkeypatch):
    fake = FakeZoteroForAnnotationUpdate({"ANNO0001": _annotation_item("ANNO0001")})
    _patch_client(monkeypatch, fake)

    # Can't use both 'tags' and 'add_tags' in one call.
    result = server.update_annotation(
        annotation_key="ANNO0001",
        tags=["a"],
        add_tags=["b"],
        ctx=DummyContext(),
    )

    assert "Cannot use 'tags'" in result
    assert fake.updated == []


def test_update_annotation_no_changes(monkeypatch):
    fake = FakeZoteroForAnnotationUpdate({"ANNO0001": _annotation_item("ANNO0001")})
    _patch_client(monkeypatch, fake)

    result = server.update_annotation(
        annotation_key="ANNO0001", ctx=DummyContext()
    )

    assert "No changes" in result
    assert fake.updated == []


def test_update_annotation_rejects_non_annotation(monkeypatch):
    note = {
        "key": "NOTE0001",
        "version": 1,
        "data": {"key": "NOTE0001", "version": 1, "itemType": "note", "note": "x"},
    }
    fake = FakeZoteroForAnnotationUpdate({"NOTE0001": note})
    _patch_client(monkeypatch, fake)

    result = server.update_annotation(
        annotation_key="NOTE0001", text="x", ctx=DummyContext()
    )

    assert "is not an annotation" in result
    assert fake.updated == []


def test_update_annotation_missing_key(monkeypatch):
    fake = FakeZoteroForAnnotationUpdate({})
    _patch_client(monkeypatch, fake)

    result = server.update_annotation(
        annotation_key="ZZZZZZZZ", text="x", ctx=DummyContext()
    )

    assert "No item found" in result
    assert fake.updated == []


class FakeZoteroForAnnotationDelete:
    """pyzotero's delete_item returns True, or raises when the write is refused."""

    def __init__(self, items, error=None):
        self._items = items
        self._error = error
        self.deleted = []

    def item(self, key):
        if key not in self._items:
            raise KeyError(key)
        return self._items[key]

    def delete_item(self, payload, last_modified=None):
        if self._error is not None:
            raise self._error
        self.deleted.append(payload)
        return True


def test_delete_annotation_deletes_permanently(monkeypatch):
    """Trashing is not enough for annotations: Zotero's reader keeps drawing a
    trashed annotation and offers no way to remove it, so re-annotating a
    paper left the old and new sets overlapping on the page."""
    item = _annotation_item("ANNO0001")
    fake = FakeZoteroForAnnotationDelete({"ANNO0001": item})
    _patch_client(monkeypatch, fake)

    result = server.delete_annotation(
        annotation_key="ANNO0001", ctx=DummyContext()
    )

    assert result == "Successfully deleted annotation ANNO0001"
    assert fake.deleted == [item]


def test_delete_annotation_rejects_non_annotation(monkeypatch):
    note = {
        "key": "NOTE0001",
        "version": 1,
        "data": {"key": "NOTE0001", "version": 1, "itemType": "note", "note": ""},
    }
    fake = FakeZoteroForAnnotationDelete({"NOTE0001": note})
    _patch_client(monkeypatch, fake)

    result = server.delete_annotation(
        annotation_key="NOTE0001", ctx=DummyContext()
    )

    assert "is not an annotation" in result
    assert fake.deleted == []


def test_delete_annotation_missing_key(monkeypatch):
    fake = FakeZoteroForAnnotationDelete({})
    _patch_client(monkeypatch, fake)

    result = server.delete_annotation(
        annotation_key="ZZZZZZZZ", ctx=DummyContext()
    )

    assert "No item found" in result
    assert fake.deleted == []


def test_delete_annotation_http_error(monkeypatch):
    fake = FakeZoteroForAnnotationDelete(
        {"ANNO0001": _annotation_item("ANNO0001")},
        error=RuntimeError("412 Precondition failed"),
    )
    _patch_client(monkeypatch, fake)

    result = server.delete_annotation(
        annotation_key="ANNO0001", ctx=DummyContext()
    )

    assert result.startswith("Error deleting annotation")
    assert "Precondition failed" in result
    assert fake.deleted == []
