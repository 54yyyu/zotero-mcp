"""Annotation workflow from `zotero-cli`: move boxes, list positions, preview, snap.

Everything here is offline. A synthetic PDF (built with PyMuPDF, one page
rotated) stands in for the attachment and a small fake stands in for the Zotero
client, so no test can reach a real library.

The point of these tests is that boxes are checked against what is *stored*,
read back through the inverse of the writer's coordinate transform, and not
against the coordinates the caller planned.
"""

import json
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from zotero_mcp.tools import annotations

fitz = pytest.importorskip("fitz")

ATTACHMENT = "ATTACH01"


@pytest.fixture
def pdf_path(tmp_path):
    """Page 1 plain, page 2 turned a quarter turn, page 3 with an offset MediaBox."""
    doc = fitz.open()
    for index in range(3):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100), f"Page {index + 1} of the synthetic paper", fontsize=11)
    doc[1].set_rotation(90)
    doc[2].set_mediabox(fitz.Rect(41.9, 46.5, 41.9 + 612, 46.5 + 792))
    path = str(tmp_path / "paper.pdf")
    doc.save(path)
    doc.close()
    return path


class FakeStore:
    """The slice of a Zotero client and library backend these commands use."""

    def __init__(self):
        self.items = {}        # key -> item dict
        self.updated = []      # deep copies of each item sent to update_item
        self.deleted = []
        self.created = []
        self._next = 0

    def add(self, data, key=None):
        self._next += 1
        key = key or f"ANN{self._next:05d}"
        self.items[key] = {"key": key, "version": 1, "data": dict(data)}
        return key

    # pyzotero-style client
    def item(self, key):
        if key not in self.items:
            raise KeyError(key)
        return deepcopy(self.items[key])

    def update_item(self, item, **_kwargs):
        self.updated.append(deepcopy(item))
        self.items[item["key"]] = deepcopy(item)
        return MagicMock(status_code=204, is_success=True)

    def create_items(self, payloads, **_kwargs):
        self.created.extend(deepcopy(payloads))
        success = {}
        for offset, payload in enumerate(payloads):
            success[str(offset)] = self.add(payload)
        return {"success": success, "failed": {}}

    def delete_item(self, item, **_kwargs):
        self.deleted.append(item["key"])
        self.items.pop(item["key"], None)
        return True

    # library backend
    def get_children(self, keys, item_type=None):
        return {key: [deepcopy(i) for i in self.items.values()
                      if i["data"].get("parentItem") == key
                      and (item_type is None or i["data"].get("itemType") == item_type)]
                for key in keys}

    def get_items(self, keys):
        return {k: self.get_item(k) for k in keys}

    def get_item(self, key):
        return deepcopy(self.items.get(key)) if key in self.items else {
            "key": key, "data": {"itemType": "attachment", "contentType": "application/pdf"}}


@pytest.fixture
def store(pdf_path, monkeypatch):
    fake = FakeStore()
    monkeypatch.setattr(annotations, "_fetch_attachment_file",
                        lambda key, tmp, **kw: (pdf_path, "paper.pdf", "pdf", None))
    monkeypatch.setattr(annotations, "_get_note_write_client", lambda _op: (fake, None))
    monkeypatch.setattr(annotations._library, "get_library_backend", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def _no_better_bibtex(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "")


def create_boxes(store, specs):
    results = annotations.create_annotations(ATTACHMENT, specs, ctx=MagicMock())
    assert all(r["ok"] for r in results), results
    return [r["annotation_key"] for r in results]


def update(store, key, **kwargs):
    return annotations.change_annotation(key, ctx=MagicMock(), **kwargs)


# ---------------------------------------------------------------------------
# 1. annotations update --rect / --page
# ---------------------------------------------------------------------------

class TestUpdateInPlace:
    def test_rect_moves_the_box_and_keeps_its_key(self, store):
        (key,) = create_boxes(store, [{"page": 1, "rect": [0.1, 0.1, 0.2, 0.2], "comment": "A"}])
        before = store.items[key]["data"]["annotationPosition"]

        result = update(store, key, rect=[0.5, 0.4, 0.3, 0.1])

        assert "Successfully updated" in result and key in result
        after = store.items[key]["data"]
        assert after["annotationPosition"] != before
        # Same numbers a fresh box with that rect would be stored with.
        (fresh,) = create_boxes(store, [{"page": 1, "rect": [0.5, 0.4, 0.3, 0.1]}])
        for field in ("annotationPosition", "annotationSortIndex", "annotationPageLabel"):
            assert after[field] == store.items[fresh]["data"][field]
        assert after["annotationComment"] == "A"

    def test_rotated_page_is_handled_like_create(self, store):
        (key,) = create_boxes(store, [{"page": 2, "rect": [0.1, 0.1, 0.2, 0.2]}])
        update(store, key, rect=[0.6, 0.3, 0.2, 0.1])
        (fresh,) = create_boxes(store, [{"page": 2, "rect": [0.6, 0.3, 0.2, 0.1]}])
        moved, new = store.items[key]["data"], store.items[fresh]["data"]
        assert moved["annotationPosition"] == new["annotationPosition"]
        assert moved["annotationSortIndex"] == new["annotationSortIndex"]
        # A quarter-turned page is 792 wide and 612 tall as displayed, so the
        # stored rect is not the unrotated one.
        stored = json.loads(moved["annotationPosition"])["rects"][0]
        assert stored != [0.6 * 612, 0.3 * 792, 0.8 * 612, 0.4 * 792]

    def test_page_alone_keeps_the_box_in_place_on_the_new_page(self, store):
        (key,) = create_boxes(store, [{"page": 1, "rect": [0.2, 0.3, 0.4, 0.2]}])
        update(store, key, page=2)
        data = store.items[key]["data"]
        assert json.loads(data["annotationPosition"])["pageIndex"] == 1
        assert data["annotationSortIndex"].startswith("00001|")
        (fresh,) = create_boxes(store, [{"page": 2, "rect": [0.2, 0.3, 0.4, 0.2]}])
        assert data["annotationPosition"] == store.items[fresh]["data"]["annotationPosition"]

    def test_rect_and_page_together(self, store):
        (key,) = create_boxes(store, [{"page": 1, "rect": [0.2, 0.3, 0.4, 0.2]}])
        update(store, key, rect=[0.1, 0.1, 0.1, 0.1], page=3)
        assert json.loads(store.items[key]["data"]["annotationPosition"])["pageIndex"] == 2

    def test_annotation_type_leads_every_annotation_key_sent(self, store):
        """Zotero answers 400 when another annotation* key comes before
        annotationType, and an item read back from the API promises no order.
        Fakes cannot catch it, so the order of the JSON sent is asserted."""
        (key,) = create_boxes(store, [{"page": 1, "rect": [0.1, 0.1, 0.2, 0.2]}])
        data = store.items[key]["data"]
        scrambled = {k: data[k] for k in ("itemType", "annotationPosition", "annotationSortIndex",
                                          "annotationPageLabel", "annotationComment", "annotationType",
                                          "annotationColor", "parentItem", "tags")}
        store.items[key]["data"] = scrambled

        update(store, key, rect=[0.3, 0.3, 0.2, 0.2], comment="moved")
        sent = list(store.updated[-1]["data"])
        annotation_keys = [k for k in sent if k.startswith("annotation")]
        assert annotation_keys[0] == "annotationType", annotation_keys
        assert "annotationPosition" in annotation_keys

    def test_text_only_update_also_sends_the_type_first(self, store):
        key = store.add({"itemType": "annotation", "annotationComment": "x",
                         "annotationType": "highlight", "parentItem": ATTACHMENT, "tags": []})
        update(store, key, comment="y")
        annotation_keys = [k for k in store.updated[-1]["data"] if k.startswith("annotation")]
        assert annotation_keys[0] == "annotationType"

    def test_highlights_cannot_be_moved(self, store):
        key = store.add({"itemType": "annotation", "annotationType": "highlight",
                         "parentItem": ATTACHMENT, "annotationText": "x", "tags": [],
                         "annotationPosition": json.dumps({"pageIndex": 0, "rects": [[0, 0, 1, 1]]})})
        result = update(store, key, rect=[0.1, 0.1, 0.2, 0.2])
        assert result.startswith("Error") and "area (image)" in result
        assert store.updated == []

    @pytest.mark.parametrize("kwargs, message", [
        ({"rect": [0.9, 0.1, 0.5, 0.1]}, "fit within the page width"),
        ({"page": 0}, "positive integer"),
        ({"page": 9}, "out of range"),
    ])
    def test_bad_geometry_is_refused_without_writing(self, store, kwargs, message):
        (key,) = create_boxes(store, [{"page": 1, "rect": [0.1, 0.1, 0.2, 0.2]}])
        result = update(store, key, **kwargs)
        assert result.startswith("Error") and message in result
        assert store.updated == []


# ---------------------------------------------------------------------------
# The inverse transform
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("page_number", [1, 2, 3])
def test_shown_rect_round_trips_through_the_stored_form(pdf_path, page_number):
    from zotero_mcp.pdf_utils import build_area_position_data, zotero_rects_to_shown

    box = [0.12, 0.34, 0.25, 0.15]
    position = build_area_position_data(pdf_path, page_number, *box)
    with fitz.open(pdf_path) as doc:
        (back,) = zotero_rects_to_shown(doc[page_number - 1], position["rects"])
    assert back == pytest.approx(box, abs=1e-3)


# ---------------------------------------------------------------------------
# 2. annotations list --positions
# ---------------------------------------------------------------------------

def listed(store, **kwargs):
    out = annotations.list_annotations(item_key=ATTACHMENT, format="json", ctx=MagicMock(), **kwargs)
    return {r["annotation_key"]: r for r in json.loads(out)}


class TestListPositions:
    BOXES = [
        {"page": 1, "rect": [0.10, 0.10, 0.20, 0.15], "comment": "plain page"},
        {"page": 2, "rect": [0.55, 0.20, 0.30, 0.10], "comment": "rotated page"},
        {"page": 3, "rect": [0.30, 0.60, 0.25, 0.20], "comment": "offset MediaBox"},
    ]

    def test_listed_rect_is_the_convention_create_takes(self, store):
        keys = create_boxes(store, self.BOXES)
        records = listed(store, positions=True)
        for key, spec in zip(keys, self.BOXES):
            position = records[key]["position"]
            assert position["page_index"] == spec["page"] - 1
            assert position["rect"] == pytest.approx(spec["rect"], abs=1e-3)
            assert len(position["rects"]) == 1 and len(position["rects"][0]) == 4

    def test_rects_are_the_stored_pdf_points(self, store):
        (key,) = create_boxes(store, [self.BOXES[0]])
        stored = json.loads(store.items[key]["data"]["annotationPosition"])
        assert listed(store, positions=True)[key]["position"]["rects"] == stored["rects"]

    def test_a_listed_rect_fed_back_to_update_changes_nothing(self, store):
        keys = create_boxes(store, self.BOXES)
        before = {k: store.items[k]["data"]["annotationPosition"] for k in keys}
        for key, record in listed(store, positions=True).items():
            update(store, key, rect=record["position"]["rect"])
        for key in keys:
            old = json.loads(before[key])["rects"][0]
            new = json.loads(store.items[key]["data"]["annotationPosition"])["rects"][0]
            assert new == pytest.approx(old, abs=0.1)

    def test_positions_are_absent_unless_asked_for(self, store):
        (key,) = create_boxes(store, [self.BOXES[0]])
        assert "position" not in listed(store)[key]

    def test_without_the_pdf_the_stored_rects_still_come_back(self, store, monkeypatch):
        (key,) = create_boxes(store, [self.BOXES[0]])
        monkeypatch.setattr(annotations, "_fetch_attachment_file",
                            lambda *a, **k: (None, "", "", "Error: no file\nmore"))
        record = listed(store, positions=True)[key]
        assert record["position"]["rect"] is None and record["position"]["rects"]
        assert record["position_note"] == "Error: no file"
