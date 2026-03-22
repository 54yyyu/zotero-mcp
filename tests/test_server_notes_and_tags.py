from zotero_mcp import server


class DummyContext:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warn(self, *_args, **_kwargs):
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
