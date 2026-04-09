from zotero_mcp import server


class DummyContext:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


class FakeZoteroForNotes:
    """Fake Zotero client that returns different items based on itemType parameter."""

    def __init__(self, notes, annotations, parent_items):
        self._notes = notes
        self._annotations = annotations
        self._parent_items = parent_items
        self.params = {}

    def add_parameters(self, **kwargs):
        self.params.update(kwargs)

    def items(self, **kwargs):
        # Route based on itemType parameter (set via add_parameters or direct kwarg)
        item_type = kwargs.get("itemType") or self.params.get("itemType")
        if item_type == "annotation":
            return self._annotations
        return self._notes

    def item(self, key):
        return self._parent_items.get(key, {"data": {"title": "Unknown"}})


class FakeZoteroForTags:
    def __init__(self, items):
        self._items = items
        self.updated = []

    def add_parameters(self, **_kwargs):
        return None

    def items(self, **_kwargs):
        return self._items

    def update_item(self, item):
        self.updated.append(item)
        return {"success": True}


def test_search_notes_filters_by_query(monkeypatch):
    """Notes and annotations are filtered by query text; unrelated items excluded."""
    notes = [
        {
            "key": "NOTE0001",
            "data": {
                "itemType": "note",
                "note": "<p>A quantum-computing note.</p>",
                "parentItem": "ITEM0001",
                "tags": [],
            },
        },
        {
            "key": "NOTE0002",
            "data": {
                "itemType": "note",
                "note": "<p>This note is unrelated.</p>",
                "parentItem": "ITEM0002",
                "tags": [],
            },
        },
    ]
    annotations = [
        {
            "key": "ANNO0001",
            "data": {
                "itemType": "annotation",
                "annotationType": "highlight",
                "annotationText": "quantum tunneling effect",
                "annotationComment": "",
                "parentItem": "ITEM0001",
            },
        },
        {
            "key": "ANNO0002",
            "data": {
                "itemType": "annotation",
                "annotationType": "highlight",
                "annotationText": "unrelated topic entirely",
                "annotationComment": "",
                "parentItem": "ITEM0002",
            },
        },
    ]
    parent_items = {
        "ITEM0001": {"data": {"title": "Quantum Book"}},
        "ITEM0002": {"data": {"title": "Other Book"}},
    }
    fake_zot = FakeZoteroForNotes(notes, annotations, parent_items)

    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.search_notes(query="quantum", limit=20, ctx=DummyContext())

    # Quantum note and annotation should appear
    assert "NOTE0001" in result
    assert "ANNO0001" in result
    # Unrelated items should NOT appear
    assert "NOTE0002" not in result
    assert "ANNO0002" not in result


def test_search_notes_note_results_survive_annotation_crash(monkeypatch):
    """If annotation search crashes, note results are still returned."""
    notes = [
        {
            "key": "NOTE0001",
            "data": {
                "itemType": "note",
                "note": "<p>A mindfulness note.</p>",
                "parentItem": "ITEM0001",
                "tags": [],
            },
        },
    ]
    parent_items = {
        "ITEM0001": {"data": {"title": "Mindfulness Paper"}},
    }

    call_count = [0]

    class CrashingAnnotationZot(FakeZoteroForNotes):
        def items(self, **kwargs):
            item_type = kwargs.get("itemType") or self.params.get("itemType")
            if item_type == "annotation":
                raise RuntimeError("Annotation search exploded!")
            return self._notes

    fake_zot = CrashingAnnotationZot(notes, [], parent_items)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.search_notes(query="mindfulness", limit=20, ctx=DummyContext())

    # Note results should still be returned despite annotation crash
    assert "NOTE0001" in result
    assert "mindfulness" in result.lower()


class FakeZoteroForNoteUpdate:
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


def _note_item(key, html, parent="ITEM0001"):
    return {
        "key": key,
        "data": {
            "key": key,
            "version": 1,
            "itemType": "note",
            "parentItem": parent,
            "note": html,
            "tags": [],
        },
    }


class FakeZoteroForGetNotes:
    def __init__(self, notes):
        self._notes = notes

    def children(self, item_key, **kwargs):
        if kwargs.get("start", 0) > 0:
            return []
        return [n for n in self._notes if n["data"].get("parentItem") == item_key]

    def items(self, **kwargs):
        if kwargs.get("start", 0) > 0:
            return []
        return self._notes

    def item(self, key):
        return {"data": {"title": "Parent Paper"}}

    def add_parameters(self, **_kwargs):
        return None


_HTML_NOTE = {
    "key": "NOTE0001",
    "data": {
        "itemType": "note",
        "parentItem": "ITEM0001",
        "note": "<p>Hello <strong>world</strong>.</p>",
        "tags": [],
    },
}


def test_get_notes_strips_html_by_default(monkeypatch):
    fake = FakeZoteroForGetNotes([_HTML_NOTE])
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)

    result = server.get_notes(item_key="ITEM0001", ctx=DummyContext())

    assert "<p>" not in result
    assert "<strong>" not in result
    assert "Hello" in result and "world" in result


def test_get_notes_raw_html_preserves_tags(monkeypatch):
    fake = FakeZoteroForGetNotes([_HTML_NOTE])
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)

    result = server.get_notes(item_key="ITEM0001", raw_html=True, ctx=DummyContext())

    assert "<p>Hello <strong>world</strong>.</p>" in result


def test_search_notes_raw_html_preserves_tags(monkeypatch):
    notes = [
        {
            "key": "NOTE0001",
            "data": {
                "itemType": "note",
                "note": "<p>A <em>quantum</em> note.</p>",
                "parentItem": "ITEM0001",
                "tags": [],
            },
        },
    ]
    parent_items = {"ITEM0001": {"data": {"title": "Quantum Paper"}}}
    fake = FakeZoteroForNotes(notes, [], parent_items)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.search_notes(
        query="quantum", limit=20, raw_html=True, ctx=DummyContext()
    )

    assert "<em>quantum</em>" in result
    # Query matching uses stripped text, so this note is still found.
    assert "NOTE0001" in result


def test_update_note_replaces_content(monkeypatch):
    fake = FakeZoteroForNoteUpdate({"NOTE0001": _note_item("NOTE0001", "<p>old</p>")})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.update_note(
        item_key="NOTE0001", note_text="<p>new</p>", append=False, ctx=DummyContext()
    )

    assert "Successfully updated" in result
    assert fake.updated[0]["data"]["note"] == "<p>new</p>"


def test_update_note_appends_content(monkeypatch):
    fake = FakeZoteroForNoteUpdate({"NOTE0001": _note_item("NOTE0001", "<p>old</p>")})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.update_note(
        item_key="NOTE0001", note_text="<p>more</p>", append=True, ctx=DummyContext()
    )

    assert "Successfully updated" in result
    assert fake.updated[0]["data"]["note"] == "<p>old</p><p>more</p>"


def test_update_note_rejects_non_note(monkeypatch):
    parent = {
        "key": "ITEM0001",
        "data": {"key": "ITEM0001", "version": 1, "itemType": "journalArticle"},
    }
    fake = FakeZoteroForNoteUpdate({"ITEM0001": parent})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.update_note(
        item_key="ITEM0001", note_text="<p>x</p>", append=False, ctx=DummyContext()
    )

    assert "is not a note" in result
    assert fake.updated == []


def test_update_note_missing_key(monkeypatch):
    fake = FakeZoteroForNoteUpdate({})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.update_note(
        item_key="ZZZZZZZZ", note_text="<p>x</p>", append=False, ctx=DummyContext()
    )

    assert "No item found" in result
    assert fake.updated == []


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


class FakeZoteroForDeleteNote:
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


def test_delete_note_trashes_via_patch(monkeypatch):
    note = {
        "key": "NOTE0001",
        "version": 42,
        "data": {"key": "NOTE0001", "version": 42, "itemType": "note", "note": "<p>x</p>"},
    }
    fake = FakeZoteroForDeleteNote({"NOTE0001": note})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.delete_note(item_key="NOTE0001", ctx=DummyContext())

    assert "Successfully trashed" in result
    assert len(fake.client.calls) == 1
    call = fake.client.calls[0]
    assert "NOTE0001" in call["url"]
    assert call["headers"]["If-Unmodified-Since-Version"] == "42"
    assert '"deleted": 1' in call["content"]


def test_delete_note_rejects_non_note(monkeypatch):
    parent = {
        "key": "ITEM0001",
        "version": 1,
        "data": {"key": "ITEM0001", "version": 1, "itemType": "journalArticle"},
    }
    fake = FakeZoteroForDeleteNote({"ITEM0001": parent})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.delete_note(item_key="ITEM0001", ctx=DummyContext())

    assert "is not a note" in result
    assert fake.client.calls == []


def test_delete_note_missing_key(monkeypatch):
    fake = FakeZoteroForDeleteNote({})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.delete_note(item_key="ZZZZZZZZ", ctx=DummyContext())

    assert "No item found" in result
    assert fake.client.calls == []


def test_delete_note_http_error(monkeypatch):
    note = {
        "key": "NOTE0001",
        "version": 5,
        "data": {"key": "NOTE0001", "version": 5, "itemType": "note", "note": ""},
    }
    fake = FakeZoteroForDeleteNote({"NOTE0001": note}, patch_status=412)
    fake.client._text = "Precondition failed"
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)

    result = server.delete_note(item_key="NOTE0001", ctx=DummyContext())

    assert "Failed to trash" in result
    assert "412" in result


def test_batch_update_tags_validates_json_array(monkeypatch):
    items = [
        {
            "key": "ITEM0001",
            "data": {
                "itemType": "journalArticle",
                "tags": [{"tag": "old"}],
            },
        }
    ]
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: FakeZoteroForTags(items))

    result = server.batch_update_tags(
        query="anything",
        add_tags='{"not":"a-list"}',
        remove_tags=None,
        limit=5,
        ctx=DummyContext(),
    )

    assert "must be a list of strings" in result
