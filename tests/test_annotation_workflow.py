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


# ---------------------------------------------------------------------------
# 3. annotations preview, and batch --dry-run --render
# ---------------------------------------------------------------------------

SCALE = 1.5
PAGE_SIZE = {1: (612, 792), 2: (792, 612), 3: (612, 792)}   # as displayed; page 2 is turned


def expected_px(page, rect, scale=SCALE):
    width, height = PAGE_SIZE[page]
    x, y, w, h = rect
    return [x * width * scale, y * height * scale, (x + w) * width * scale, (y + h) * height * scale]


def preview(store, tmp_path, name, pages=None):
    images, error = annotations.preview_annotations(
        ATTACHMENT, pages, str(tmp_path / name), SCALE, ctx=MagicMock())
    assert error is None, error
    return {image["page"]: image for image in images}


def pixel(path, x, y):
    return tuple(fitz.Pixmap(path).pixel(int(x), int(y)))[:3]


def centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


class TestPreview:
    RED, GREEN, BLUE = "#ff0000", "#00aa00", "#0000ff"

    def three_boxes(self, store):
        return create_boxes(store, [
            {"page": 1, "rect": [0.10, 0.30, 0.25, 0.15], "comment": "first box", "color": self.RED},
            {"page": 2, "rect": [0.55, 0.40, 0.30, 0.20], "comment": "rotated box", "color": self.GREEN},
            {"page": 3, "rect": [0.30, 0.60, 0.25, 0.20], "comment": "offset box", "color": self.BLUE},
        ])

    def test_boxes_are_drawn_where_they_were_asked_for_on_every_kind_of_page(self, store, tmp_path):
        self.three_boxes(store)
        images = preview(store, tmp_path, "a")
        specs = {1: [0.10, 0.30, 0.25, 0.15], 2: [0.55, 0.40, 0.30, 0.20], 3: [0.30, 0.60, 0.25, 0.20]}
        for page, rect in specs.items():
            (drawn,) = images[page]["boxes"]
            assert drawn == pytest.approx(expected_px(page, rect), abs=2)

    def test_moving_one_box_moves_only_that_box(self, store, tmp_path):
        keys = self.three_boxes(store)
        before = preview(store, tmp_path, "before")

        new_rect = [0.50, 0.10, 0.20, 0.10]
        update(store, keys[1], rect=new_rect)          # the box on the rotated page
        after = preview(store, tmp_path, "after")

        assert after[1]["boxes"] == before[1]["boxes"]
        assert after[3]["boxes"] == before[3]["boxes"]
        assert after[2]["boxes"] != before[2]["boxes"]
        assert after[2]["boxes"][0] == pytest.approx(expected_px(2, new_rect), abs=2)

        # And in the pixels: the old spot is bare page, the new one is green.
        old_x, old_y = centre(before[2]["boxes"][0])
        new_x, new_y = centre(after[2]["boxes"][0])
        assert pixel(after[2]["path"], old_x, old_y) == (255, 255, 255)
        red, green, blue = pixel(after[2]["path"], new_x, new_y)
        assert green > red and green > blue and red < 255
        # The untouched pages are pixel-identical to before.
        for page in (1, 3):
            with open(before[page]["path"], "rb") as a, open(after[page]["path"], "rb") as b:
                assert a.read() == b.read()

    def test_pages_option_limits_the_output(self, store, tmp_path):
        self.three_boxes(store)
        images = preview(store, tmp_path, "some", pages=[2, 3])
        assert sorted(images) == [2, 3]

    def test_a_page_outside_the_pdf_is_an_error(self, store, tmp_path):
        self.three_boxes(store)
        images, error = annotations.preview_annotations(
            ATTACHMENT, [9], str(tmp_path / "x"), SCALE, ctx=MagicMock())
        assert images == [] and "out of range" in error


class TestDryRunRender:
    def plan(self, tmp_path, specs):
        results = annotations.create_annotations(
            ATTACHMENT, specs, ctx=MagicMock(), dry_run=True, render_dir=str(tmp_path / "plan"))
        assert all(r["ok"] for r in results), results
        return results

    def test_the_render_shows_the_planned_rect_when_the_writer_is_right(self, store, tmp_path):
        planned = [0.55, 0.40, 0.30, 0.20]
        (result,) = self.plan(tmp_path, [{"page": 2, "rect": planned, "color": "#ff0000"}])
        box = expected_px(2, planned)
        x, y = centre(box)
        assert pixel(result["rendered"], x, y) != (255, 255, 255)
        assert pixel(result["rendered"], box[0] / 2, box[1] / 2) == (255, 255, 255)
        assert store.created == []                       # a dry run writes nothing

    def test_a_writer_bug_shows_up_because_the_render_reads_the_stored_form(
            self, store, tmp_path, monkeypatch):
        """If the render copied the plan it would look fine whatever the writer
        did. This writer ignores page rotation, as an old bug did."""
        from zotero_mcp import pdf_utils

        def ignores_rotation(page, x0, y0, x1, y1):
            width, height = page.rect.width, page.rect.height
            return pdf_utils._convert_rects_to_zotero(
                [tuple(round(v, 4) for v in (x0 * width, y0 * height, x1 * width, y1 * height))], page)

        monkeypatch.setattr(pdf_utils, "_shown_rect_to_zotero", ignores_rotation)
        planned = [0.55, 0.40, 0.30, 0.20]
        (result,) = self.plan(tmp_path, [{"page": 2, "rect": planned, "color": "#ff0000"}])
        x, y = centre(expected_px(2, planned))
        # Where the plan said the box would be, the render shows bare page.
        assert pixel(result["rendered"], x, y) == (255, 255, 255)

    def test_highlight_matches_are_drawn_too(self, store, tmp_path, pdf_path):
        (result,) = self.plan(tmp_path, [{"page": 1, "text": "synthetic paper", "color": "#ffd400"}])
        with fitz.open(pdf_path) as doc:
            (found,) = doc[0].search_for("synthetic paper")
        # Along the matched words most pixels are yellow-tinted; just past them it is bare page.
        y = (found.y0 + found.y1) / 2 * SCALE
        inside = [pixel(result["rendered"], x, y) for x in range(int(found.x0 * SCALE) + 1, int(found.x1 * SCALE) - 1)]
        yellow = [p for p in inside if p[0] > 200 and p[1] > 180 and p[2] < 200]
        assert len(yellow) > len(inside) / 3
        assert pixel(result["rendered"], (found.x1 + 20) * SCALE, y) == (255, 255, 255)


class TestPreviewCli:
    def test_render_needs_dry_run(self):
        import argparse

        from zotero_mcp import cli_standalone
        from zotero_mcp.cli_json import CliError

        args = argparse.Namespace(verbose=False, json_out=False, subcommand="batch", attachment_key=ATTACHMENT,
                                  file="-", dry_run=False, render="out")
        with pytest.raises(CliError, match="--dry-run"):
            cli_standalone._annotations_batch(args, MagicMock(), MagicMock())

    def test_preview_prints_one_line_per_image(self, store, tmp_path, capsys):
        import argparse

        from zotero_mcp import cli_standalone

        create_boxes(store, [{"page": 1, "rect": [0.1, 0.3, 0.2, 0.1], "comment": "x"}])
        args = argparse.Namespace(verbose=False, json_out=False, subcommand="preview",
                                  attachment_key=ATTACHMENT, pages="all", out=str(tmp_path / "o"), scale=1.5)
        cli_standalone._annotations_preview(args, annotations, MagicMock())
        out = capsys.readouterr().out
        assert "p1:" in out and "1 annotation(s)" in out and (tmp_path / "o" / f"{ATTACHMENT}-p1.png").exists()


# ---------------------------------------------------------------------------
# 4. Panels and `rect: auto`
# ---------------------------------------------------------------------------

#: A 2 x 2 figure: panel letter at the top-left of each, a framed plot under it.
PANELS = {"A": (60, 100), "B": (320, 100), "C": (60, 300), "D": (320, 300)}
PANEL_W, PANEL_H = 220, 160


def panel_pdf(tmp_path, letters="ABCD", rotation=0, name="figure.pdf"):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for letter, (x, y) in PANELS.items():
        page.draw_rect(fitz.Rect(x + 10, y + 14, x + PANEL_W - 10, y + PANEL_H - 10), width=1)
        page.draw_line((x + 10, y + 100), (x + PANEL_W - 10, y + 60), width=2)
        if letter in letters:
            page.insert_text((x, y + 9), letter, fontsize=9, fontname="hebo")
    page.insert_text((60, 480), "Figure 1. A synthetic figure with four panels.", fontsize=9)
    if rotation:
        page.set_rotation(rotation)
    path = str(tmp_path / name)
    doc.save(path)
    doc.close()
    return path


def panel_inner(letter):
    """The framed plot of a panel, normalized on the unrotated 612 x 792 page."""
    x, y = PANELS[letter]
    return [(x + 10) / 612, (y + 14) / 792, (PANEL_W - 20) / 612, (PANEL_H - 24) / 792]


def contains(outer, inner, slack=0.004):
    return (outer[0] <= inner[0] + slack and outer[1] <= inner[1] + slack
            and outer[0] + outer[2] >= inner[0] + inner[2] - slack
            and outer[1] + outer[3] >= inner[1] + inner[3] - slack)


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


class TestPanels:
    def test_panel_letters_split_a_figure_page_into_boxes(self, tmp_path):
        from zotero_mcp import pdf_layout

        layout = pdf_layout.detect_page_regions(panel_pdf(tmp_path), 1, panels=True)
        panels = {r["caption_label"]: r for r in layout["regions"] if r["source"] == "panel"}
        assert sorted(panels) == ["Panel A", "Panel B", "Panel C", "Panel D"]
        for letter in "ABCD":
            box = panels[f"Panel {letter}"]["bbox"]
            assert contains(box, panel_inner(letter))
            for other in "ABCD".replace(letter, ""):
                assert not overlaps(box, panel_inner(other)), (letter, other)
            assert panels[f"Panel {letter}"]["confidence"] in ("high", "medium")

    def test_panels_are_opt_in(self, tmp_path):
        from zotero_mcp import pdf_layout

        layout = pdf_layout.detect_page_regions(panel_pdf(tmp_path), 1)
        assert not [r for r in layout["regions"] if r["source"] == "panel"]

    def test_a_gap_in_the_letters_is_low_confidence(self, tmp_path):
        from zotero_mcp import pdf_layout

        layout = pdf_layout.detect_page_regions(panel_pdf(tmp_path, letters="ACD"), 1, panels=True)
        assert {r["confidence"] for r in layout["regions"] if r["source"] == "panel"} == {"low"}

    def test_stray_bold_letters_on_a_text_page_are_low_confidence(self):
        from zotero_mcp import pdf_layout

        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        for i in range(30):
            page.insert_text((50, 60 + i * 16), f"Body text line {i} of the paper", fontsize=10)
        page.insert_text((50, 700), "A", fontsize=10, fontname="hebo")
        page.insert_text((300, 720), "B", fontsize=10, fontname="hebo")
        layout = pdf_layout.detect_page_regions(doc, 1, panels=True)
        assert {r["confidence"] for r in layout["regions"] if r["source"] == "panel"} == {"low"}

    def test_one_letter_is_not_a_split(self, tmp_path):
        from zotero_mcp import pdf_layout

        layout = pdf_layout.detect_page_regions(panel_pdf(tmp_path, letters="A"), 1, panels=True)
        assert not [r for r in layout["regions"] if r["source"] == "panel"]

    def test_a_landscape_page_stored_turned_gives_the_same_panels_as_when_upright(self):
        """Landscape figure pages are usually portrait pages with /Rotate set, the
        text laid out sideways in the file. Panels are found in the frame the
        reader displays."""
        from zotero_mcp import pdf_layout

        upright = fitz.open()
        page = upright.new_page(width=792, height=612)
        for letter, (x, y) in {"A": (60, 60), "B": (420, 60), "C": (60, 320), "D": (420, 320)}.items():
            page.draw_rect(fitz.Rect(x + 10, y + 14, x + 300, y + 200), width=1)
            page.insert_text((x, y + 9), letter, fontsize=9, fontname="hebo")
        turned = fitz.open()
        sideways = turned.new_page(width=612, height=792)
        sideways.show_pdf_page(sideways.rect, upright, 0, rotate=90)
        sideways.set_rotation(90)

        def panels(doc):
            layout = pdf_layout.detect_page_regions(doc, 1, panels=True)
            return {r["caption_label"]: r["bbox"] for r in layout["regions"] if r["source"] == "panel"}

        want, got = panels(upright), panels(turned)
        assert sorted(got) == ["Panel A", "Panel B", "Panel C", "Panel D"]
        for label, bbox in want.items():
            assert got[label] == pytest.approx(bbox, abs=0.01)


class TestAutoBox:
    def box_of(self, tmp_path, near=None, **kwargs):
        from zotero_mcp import pdf_layout

        with fitz.open(panel_pdf(tmp_path, **kwargs)) as doc:
            return pdf_layout.auto_box(doc, 1, near)

    def test_snaps_to_the_panel_under_the_hint(self, tmp_path):
        near = [0.60, 0.15, 0.30, 0.15]                    # roughly panel B, sloppy
        found = self.box_of(tmp_path, near)
        assert found["source"].startswith("panel") and "Panel B" in found["source"]
        assert contains(found["rect"], panel_inner("B"))
        assert not overlaps(found["rect"], panel_inner("A")) and not overlaps(found["rect"], panel_inner("D"))

    def test_a_hint_that_matches_nothing_is_kept_and_says_so(self, tmp_path):
        found = self.box_of(tmp_path, [0.05, 0.85, 0.2, 0.1])
        assert found["source"] == "hint" and found["rect"] == [0.05, 0.85, 0.2, 0.1]
        assert "overlaps the hint" in found["note"]

    def test_low_confidence_panels_fall_back_to_the_hint(self, tmp_path):
        near = [0.10, 0.12, 0.30, 0.20]
        found = self.box_of(tmp_path, near, letters="ACD")
        assert found["source"] == "hint" and found["rect"] == near
        assert "low-confidence" in found["note"]

    def test_without_a_hint_several_panels_are_an_error(self, tmp_path):
        assert "near hint" in self.box_of(tmp_path)["error"]

    def test_batch_spec_fits_the_box_and_reports_how(self, store, tmp_path, monkeypatch):
        path = panel_pdf(tmp_path)
        monkeypatch.setattr(annotations, "_fetch_attachment_file",
                            lambda key, tmp, **kw: (path, "figure.pdf", "pdf", None))
        results = annotations.create_annotations(ATTACHMENT, [
            {"page": 1, "rect": "auto", "near": [0.60, 0.15, 0.30, 0.15], "comment": "FIG 1B"},
            {"page": 1, "rect": "auto", "near": [0.05, 0.85, 0.2, 0.1], "comment": "nothing here"},
            {"page": 1, "rect": "auto", "comment": "no hint"},
            {"page": 1, "rect": "auto", "near": [2, 2, 2, 2]},
        ], ctx=MagicMock())
        assert [r["ok"] for r in results] == [True, True, False, False]
        assert "Panel B" in results[0]["auto"]["source"]
        stored = json.loads(store.items[results[0]["annotation_key"]]["data"]["annotationPosition"])
        # The box starts at the panel's letter, a little left of its frame.
        assert 315 <= stored["rects"][0][0] <= panel_inner("B")[0] * 612 + 1
        assert results[1]["auto"]["source"] == "hint" and results[1]["auto"]["note"]
        assert "near hint" in results[2]["error"]
        assert "near" in results[3]["error"]

    def test_create_rect_auto_goes_through_the_same_fit(self, store, tmp_path, monkeypatch, capsys):
        import argparse

        from zotero_mcp import cli_standalone

        path = panel_pdf(tmp_path)
        monkeypatch.setattr(annotations, "_fetch_attachment_file",
                            lambda key, tmp, **kw: (path, "figure.pdf", "pdf", None))
        args = argparse.Namespace(verbose=False, json_out=False, subcommand="create", attachment_key=ATTACHMENT,
                                  page=1, rect="auto", near="0.60,0.15,0.30,0.15", text=None, note=None,
                                  comment="FIG 1B", color="purple", tags=None, open=False)
        monkeypatch.setattr(cli_standalone, "setup_zotero_environment", lambda: None)
        cli_standalone.cmd_annotations(args)
        out = capsys.readouterr().out
        assert "Successfully created area annotation" in out and "Panel B" in out
        (created,) = store.created
        assert created["annotationType"] == "image" and created["annotationComment"] == "FIG 1B"

    def test_near_without_auto_is_a_usage_error(self, monkeypatch):
        import argparse

        from zotero_mcp import cli_standalone
        from zotero_mcp.cli_json import CliError

        args = argparse.Namespace(verbose=False, json_out=False, subcommand="create", attachment_key=ATTACHMENT,
                                  page=1, rect="0.1,0.1,0.2,0.2", near="0.1,0.1,0.2,0.2")
        monkeypatch.setattr(cli_standalone, "setup_zotero_environment", lambda: None)
        with pytest.raises(CliError, match="--rect auto"):
            cli_standalone.cmd_annotations(args)

    def test_the_cli_prints_a_fallback_instead_of_hiding_it(self, store, tmp_path, monkeypatch, capsys):
        import argparse

        from zotero_mcp import cli_standalone

        path = panel_pdf(tmp_path)
        monkeypatch.setattr(annotations, "_fetch_attachment_file",
                            lambda key, tmp, **kw: (path, "figure.pdf", "pdf", None))
        specs = tmp_path / "plan.jsonl"
        specs.write_text(json.dumps({"page": 1, "rect": "auto", "near": "0.05,0.85,0.2,0.1"}))
        args = argparse.Namespace(verbose=False, json_out=False, subcommand="batch", attachment_key=ATTACHMENT,
                                  file=str(specs), dry_run=True, render=None)
        cli_standalone._annotations_batch(args, annotations, MagicMock())
        out = capsys.readouterr().out
        assert "auto: hint, low" in out and "kept the near hint" in out


# ---------------------------------------------------------------------------
# 5. Batch safety and the offline dry run
# ---------------------------------------------------------------------------

PLAN = [
    {"page": 1, "rect": [0.10, 0.30, 0.25, 0.15], "comment": "FIG 1A | first panel of the figure, with a long comment"},
    {"page": 2, "rect": [0.55, 0.40, 0.30, 0.20], "comment": "FIG 1B | rotated page"},
    {"page": 1, "text": "synthetic paper", "comment": "PROBLEM. the sentence"},
    {"page": 3, "rect": [0.30, 0.60, 0.25, 0.20]},                       # told apart by its box
]


def write(store, specs=PLAN, **kwargs):
    return annotations.create_annotations(ATTACHMENT, specs, ctx=MagicMock(), **kwargs)


class TestIdempotentBatch:
    def test_rerun_with_skip_creates_nothing_new(self, store):
        first = write(store, existing="skip")
        assert all(r["ok"] and not r.get("skipped") for r in first)
        made = len(store.created)

        second = write(store, existing="skip")
        assert all(r["ok"] for r in second)
        assert [bool(r.get("skipped")) for r in second] == [True, True, True, True]
        assert len(store.created) == made
        assert [r["annotation_key"] for r in second] == [r["annotation_key"] for r in first]

    def test_without_the_option_a_rerun_duplicates(self, store):
        write(store)
        write(store)
        assert len(store.created) == 2 * len(PLAN)

    def test_replace_deletes_the_old_ones_after_creating_the_new(self, store):
        first = write(store, existing="replace")
        second = write(store, existing="replace")
        old = {r["annotation_key"] for r in first}
        assert set(store.deleted) == old
        assert [r.get("replaced") for r in second] == [[r["annotation_key"]] for r in first]
        assert not old & set(store.items)
        assert all(r["ok"] for r in second)

    def test_the_key_is_page_plus_the_first_40_characters_of_comment(self, store):
        write(store, [{"page": 1, "rect": [0.1, 0.1, 0.1, 0.1], "comment": "x" * 40 + " tail one"}])
        again = write(store, [
            {"page": 1, "rect": [0.5, 0.5, 0.1, 0.1], "comment": "x" * 40 + " a different tail"},  # same key
            {"page": 2, "rect": [0.5, 0.5, 0.1, 0.1], "comment": "x" * 40 + " tail one"},          # other page
        ], existing="skip")
        assert [bool(r.get("skipped")) for r in again] == [True, False]

    def test_a_box_and_a_highlight_with_the_same_comment_are_different(self, store):
        write(store, [{"page": 1, "text": "synthetic paper", "comment": "same words"}])
        (again,) = write(store, [{"page": 1, "rect": [0.1, 0.1, 0.1, 0.1], "comment": "same words"}],
                         existing="replace")
        assert again["ok"] and not again.get("replaced") and not store.deleted

    def test_a_comment_less_box_elsewhere_on_the_page_is_new(self, store):
        write(store, [{"page": 3, "rect": [0.30, 0.60, 0.25, 0.20]}])
        (again,) = write(store, [{"page": 3, "rect": [0.30, 0.10, 0.25, 0.20]}], existing="skip")
        assert again["ok"] and not again.get("skipped")

    def test_replace_refuses_when_two_on_file_match_one_entry(self, store):
        """Two boxes whose comments share the first 40 characters: deleting both, or a guessed
        one, would lose an annotation the batch does not recreate."""
        head = "FIG 2 | the knockout reduces expression in"
        write(store, [{"page": 1, "rect": [0.1, 0.1, 0.1, 0.1], "comment": head + " neurons"},
                      {"page": 1, "rect": [0.5, 0.5, 0.1, 0.1], "comment": head + " glia"}])
        before = set(store.items)
        (result,) = write(store, [{"page": 1, "rect": [0.1, 0.1, 0.1, 0.1], "comment": head + " neurons!"}],
                          existing="replace")
        assert not result["ok"] and "unclear" in result["error"]
        assert set(store.items) == before and not store.deleted

    def test_each_one_on_file_is_matched_once(self, store):
        head = "FIG 2 | the knockout reduces expression in"
        plan = [{"page": 1, "rect": [0.1, 0.1, 0.1, 0.1], "comment": head + " neurons"},
                {"page": 1, "rect": [0.5, 0.5, 0.1, 0.1], "comment": head + " glia"}]
        first = write(store, plan)
        second = write(store, plan, existing="replace")
        assert all(r["ok"] and not r.get("warning") for r in second)
        assert sorted(store.deleted) == sorted(r["annotation_key"] for r in first)
        assert len(set(r["replaced"][0] for r in second)) == 2

    def test_replace_never_deletes_what_is_no_longer_an_annotation_here(self, store):
        (first,) = write(store, [{"page": 1, "rect": [0.1, 0.1, 0.1, 0.1], "comment": "moved away"}])
        real_item = store.item

        def moved(key):
            item = real_item(key)
            if key == first["annotation_key"]:
                item["data"]["parentItem"] = "OTHER001"
            return item

        store.item = moved
        (again,) = write(store, [{"page": 1, "rect": [0.1, 0.1, 0.1, 0.1], "comment": "moved away"}],
                         existing="replace")
        assert again["ok"] and "not deleted" in again["warning"] and not store.deleted

    def test_cli_summary_counts_old_annotations_left_behind(self, store, tmp_path, capsys):
        import argparse

        from zotero_mcp import cli_standalone

        plan = tmp_path / "plan.jsonl"
        plan.write_text(json.dumps(PLAN[0]))
        write(store, [PLAN[0]])
        store.delete_item = MagicMock(side_effect=RuntimeError("412 Precondition Failed"))
        args = argparse.Namespace(verbose=False, json_out=False, subcommand="batch", attachment_key=ATTACHMENT,
                                  file=str(plan), dry_run=False, render=None, on_existing="replace")
        cli_standalone._annotations_batch(args, annotations, MagicMock())
        out = capsys.readouterr().out
        assert "WARN" in out and "1 old annotation(s) not deleted" in out

    def test_a_dry_run_reports_what_it_would_do_and_writes_nothing(self, store):
        write(store, existing="skip")
        made, gone = len(store.created), len(store.deleted)
        skip = write(store, dry_run=True, existing="skip")
        replace = write(store, dry_run=True, existing="replace")
        assert [bool(r.get("skipped")) for r in skip] == [True, True, True, True]
        assert [bool(r.get("replaced")) for r in replace] == [True, True, True, True]
        assert len(store.created) == made and len(store.deleted) == gone

    def test_cli_counts_what_was_skipped(self, store, tmp_path, capsys):
        import argparse

        from zotero_mcp import cli_standalone

        plan = tmp_path / "plan.jsonl"
        plan.write_text("\n".join(json.dumps(s) for s in PLAN))
        args = argparse.Namespace(verbose=False, json_out=True, subcommand="batch", attachment_key=ATTACHMENT,
                                  file=str(plan), dry_run=False, render=None, on_existing="skip")
        cli_standalone._annotations_batch(args, annotations, MagicMock())
        capsys.readouterr()
        cli_standalone._annotations_batch(args, annotations, MagicMock())
        data = json.loads(capsys.readouterr().out)["data"]
        assert (data["skipped"], data["failed"]) == (4, 0)


class TestOfflineDryRun:
    def test_pure_matching_reads_the_file_zotero_holds_and_needs_no_client(self, pdf_path, monkeypatch):
        from pathlib import Path

        monkeypatch.setattr(annotations._library, "attachment_path_for", lambda key, **_kw: Path(pdf_path))
        monkeypatch.setattr(annotations, "_download_clients",
                            lambda: pytest.fail("a dry run must not need a Zotero client"))
        monkeypatch.setattr(annotations, "_get_note_write_client",
                            lambda _op: pytest.fail("a dry run must not need write access"))
        monkeypatch.setattr(annotations._utils, "is_local_mode", lambda: False)
        results = annotations.create_annotations(ATTACHMENT, [
            {"page": 1, "text": "synthetic paper"}, {"page": 2, "rect": [0.1, 0.1, 0.2, 0.2]},
        ], ctx=MagicMock(), dry_run=True)
        assert all(r["ok"] for r in results), results
        assert results[0]["matched_text"] == "synthetic paper"

    def test_a_parent_key_is_not_read_as_one_of_its_attachments(self, pdf_path, monkeypatch):
        """The real write refuses a parent key, so the dry run must not pass it on a guessed PDF."""
        from pathlib import Path

        from zotero_mcp import library

        reader = MagicMock()
        reader.resolve_attachment_file.return_value = None
        reader.get_attachment_paths.return_value = [{"resolved_path": Path(pdf_path), "exists": True}]
        monkeypatch.setattr(library, "_sqlite_reader", lambda: reader)
        assert library.attachment_path_for("PARENT01") == Path(pdf_path)
        assert library.attachment_path_for("PARENT01", own_only=True) is None

    def test_a_real_write_still_asks_for_a_client(self, pdf_path, monkeypatch):
        from pathlib import Path

        monkeypatch.setattr(annotations._library, "attachment_path_for", lambda key, **_kw: Path(pdf_path))
        monkeypatch.setattr(annotations, "_get_note_write_client", lambda _op: (None, "Error: no write access"))
        (result,) = annotations.create_annotations(ATTACHMENT, [{"page": 1, "text": "synthetic"}], ctx=MagicMock())
        assert result["error"] == "Error: no write access"

    def test_one_cause_is_said_once_with_its_fix(self, tmp_path, monkeypatch, capsys):
        import argparse

        from zotero_mcp import cli_standalone

        monkeypatch.setattr(annotations._library, "attachment_path_for", lambda key, **_kw: None)
        monkeypatch.setattr(annotations, "_download_clients", lambda: (None, None))
        monkeypatch.setattr(annotations._utils, "is_local_mode", lambda: False)
        plan = tmp_path / "plan.jsonl"
        plan.write_text("\n".join(json.dumps(s) for s in PLAN))
        args = argparse.Namespace(verbose=False, json_out=False, subcommand="batch", attachment_key=ATTACHMENT,
                                  file=str(plan), dry_run=True, render=None, on_existing=None)
        with pytest.raises(SystemExit) as exit_info:
            cli_standalone._annotations_batch(args, annotations, MagicMock())
        out = capsys.readouterr().out
        assert exit_info.value.code == 1
        assert out.count("FAIL") == 1 and "all 4 entries" in out and "Start Zotero" in out


# ---------------------------------------------------------------------------
# 8. Fuzzy text anchors
# ---------------------------------------------------------------------------

@pytest.fixture
def typeset_pdf(tmp_path):
    """Text that keeps what real typesetting leaves in it: a true minus sign, curly
    quotes, an fi ligature, a soft hyphen, and a word broken across two lines."""
    try:
        font = fitz.Font("cjk")      # a built-in font with those glyphs
    except Exception:
        pytest.skip("this PyMuPDF has no built-in font with the typographic glyphs")
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    writer = fitz.TextWriter(page.rect)
    lines = [
        "The change was −5 percent in the “quoted” group and the ﬁnal result.",
        "A soft­hyphen in the middle and a word that is regard-",
        "less of the sample size, which stays fixed.",
        "Row two shows 3 − 2 = 1 for every case.",
    ]
    for index, line in enumerate(lines):
        writer.append((72, 100 + 24 * index), line, font=font, fontsize=10)
    writer.write_text(page)
    path = str(tmp_path / "typeset.pdf")
    doc.save(path)
    doc.close()
    return path


class TestTextNormalisation:
    @pytest.mark.parametrize("wanted, covers", [
        ("change was -5 percent", "change was −5 percent"),               # hyphen-minus for a minus sign
        ("change was −5 percent", "change was −5 percent"),           # and the minus sign itself
        ("3 - 2 = 1 for every", "3 − 2 = 1 for every"),                    # minus between numbers
        ("the \"quoted\" group", "the “quoted” group"),               # straight quotes for curly
        ("and the final result", "and the ﬁnal result"),                   # fi typed as two letters
        ("softhyphen in the middle", "soft­hyphen in the middle"),         # soft hyphen left out
        ("a word that is regardless of the sample", "a word that is regard-less of the sample"),   # broken across lines
        ("word that is regard-\nless of the sample", "word that is regard-less of the sample"),  # break typed as it reads
    ])
    def test_matches_cover_exactly_the_wanted_words(self, typeset_pdf, wanted, covers):
        from zotero_mcp.pdf_utils import find_text_position, normalize_text, text_in_rects

        result = find_text_position(typeset_pdf, 1, wanted)
        assert "error" not in result, result
        def letters(text):
            """Only letters and digits, so the lookalikes and invisible marks do not count."""
            return "".join(c for c in normalize_text(text).lower() if c.isalnum())

        assert letters(text_in_rects(typeset_pdf, 0, result["rects"])) == letters(covers)

    def test_a_highlight_with_a_unicode_minus_is_created(self, typeset_pdf, monkeypatch):
        store = FakeStore()
        monkeypatch.setattr(annotations, "_fetch_attachment_file",
                            lambda key, tmp, **kw: (typeset_pdf, "t.pdf", "pdf", None))
        monkeypatch.setattr(annotations, "_get_note_write_client", lambda _op: (store, None))
        results = annotations.create_annotations(ATTACHMENT, [
            {"page": 1, "text": "change was −5 percent in the"},
            {"page": 1, "text": "word that is regardless of the sample size"},
        ], ctx=MagicMock())
        assert [r["ok"] for r in results] == [True, True], results


class TestNearestCandidates:
    MISSPELT = "the shift was \u22125 percent among the quoted participants"

    def test_a_miss_lists_the_nearest_places_with_page_score_text_and_box(self, typeset_pdf):
        from zotero_mcp.pdf_utils import find_text_position

        result = find_text_position(typeset_pdf, 1, self.MISSPELT)
        assert "error" in result
        top = result["candidates"][0]
        assert top["page"] == 1 and 0.5 < top["score"] < 1
        assert "change was" in top["best_match"]
        assert len(top["rect"]) == 4 and all(0 <= v <= 1 for v in top["rect"])
        assert top["rects"] and all(len(r) == 4 for r in top["rects"])
        assert [c["score"] for c in result["candidates"]] == sorted(
            (c["score"] for c in result["candidates"]), reverse=True)

    def test_the_candidate_box_is_where_the_text_is(self, typeset_pdf):
        from zotero_mcp.pdf_utils import find_text_position

        top = find_text_position(typeset_pdf, 1, self.MISSPELT)["candidates"][0]
        with fitz.open(typeset_pdf) as doc:
            (line,) = doc[0].search_for("The change was")
        x, y, w, h = top["rect"]
        assert x * 612 == pytest.approx(line.x0, abs=3) and abs((y + h / 2) * 792 - (line.y0 + line.y1) / 2) < 8

    def test_nothing_like_the_text_means_no_candidates(self, typeset_pdf):
        from zotero_mcp.pdf_utils import find_text_position

        result = find_text_position(typeset_pdf, 1, "qqqqqq zzzzzz xxxxxxxx wwwww vvvvvv uuuuuu tttttt")
        assert "error" in result and result["candidates"] == []

    def test_the_batch_row_carries_the_candidates_in_json_and_text(self, typeset_pdf, monkeypatch):
        monkeypatch.setattr(annotations, "_fetch_attachment_file",
                            lambda key, tmp, **kw: (typeset_pdf, "t.pdf", "pdf", None))
        (result,) = annotations.create_annotations(
            ATTACHMENT, [{"page": 1, "text": self.MISSPELT}], ctx=MagicMock(), dry_run=True)
        assert not result["ok"]
        assert result["candidates"][0]["page"] == 1
        assert "NEAREST MATCHES" in result["error"] and "p1" in result["error"]
        assert "Could not find text" in result["error"]
