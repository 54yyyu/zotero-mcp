"""Tests for PDF page layout detection (zotero_get_page_layout).

Covers the pure-logic detection helpers in pdf_layout, the
detect_page_regions engine, and the MCP tool layer.
"""

import os
import tempfile

import pytest
from conftest import DummyContext

from zotero_mcp import server
from zotero_mcp.pdf_layout import (
    _associate_captions_with_regions,
    _bbox_iou,
    _merge_candidate_regions,
    _parse_caption_block,
    detect_page_regions,
)


# ---------------------------------------------------------------------------
# Synthetic PDF builders (real PyMuPDF)
# ---------------------------------------------------------------------------

def _new_letter_page(doc):
    import fitz  # noqa: F401

    return doc.new_page(width=612, height=792)


def _gray_image_bytes():
    import fitz

    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 80))
    pix.clear_with(120)
    return pix.tobytes("png")


def _save_pdf(doc, tmpdir, name="test.pdf"):
    path = os.path.join(tmpdir, name)
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# Caption parsing
# ---------------------------------------------------------------------------

class TestParseCaptionBlock:
    def test_figure_caption_with_colon(self):
        result = _parse_caption_block("Figure 3: Mean completion rates across tasks.")
        assert result is not None
        assert result["label"] == "Figure 3"
        assert result["kind"] == "figure"
        assert "Mean completion rates" in result["text"]

    def test_figure_caption_with_period(self):
        result = _parse_caption_block("Figure 3. Mean completion rates across tasks.")
        assert result is not None
        assert result["label"] == "Figure 3"
        assert result["kind"] == "figure"

    def test_fig_abbreviation_with_panel_letter(self):
        result = _parse_caption_block("Fig. 3a: Ablation results.")
        assert result is not None
        assert result["label"] == "Fig. 3a"
        assert result["kind"] == "figure"

    def test_table_caption(self):
        result = _parse_caption_block("Table 1: Hyperparameters used in training.")
        assert result is not None
        assert result["label"] == "Table 1"
        assert result["kind"] == "table"

    def test_supplementary_table_caption(self):
        result = _parse_caption_block("Table S1: Additional results.")
        assert result is not None
        assert result["label"] == "Table S1"
        assert result["kind"] == "table"

    def test_extended_data_figure_caption(self):
        result = _parse_caption_block("Extended Data Fig. 2: Control experiments.")
        assert result is not None
        assert result["kind"] == "figure"
        assert "2" in result["label"]

    def test_in_text_reference_rejected(self):
        # Does not start the block -> not a caption
        assert _parse_caption_block("see Figure 3 for details") is None

    def test_sentence_starting_with_figure_rejected(self):
        # No ./: separator after the number -> in-text sentence, not a caption
        assert _parse_caption_block("Figure 3 shows the results of our experiments") is None

    def test_plain_text_rejected(self):
        assert _parse_caption_block("The quick brown fox jumps over the lazy dog.") is None

    def test_empty_text_rejected(self):
        assert _parse_caption_block("") is None

    def test_caption_is_case_insensitive(self):
        result = _parse_caption_block("FIGURE 2: Uppercase caption.")
        assert result is not None
        assert result["kind"] == "figure"


# ---------------------------------------------------------------------------
# Bounding box IoU
# ---------------------------------------------------------------------------

class TestBboxIou:
    def test_identical_boxes(self):
        box = [0.1, 0.2, 0.5, 0.3]
        assert _bbox_iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert _bbox_iou([0.0, 0.0, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]) == 0.0

    def test_partial_overlap(self):
        # [0,0,0.4,0.4] and [0.2,0.2,0.4,0.4]: intersection 0.2*0.2=0.04,
        # union 0.16+0.16-0.04=0.28
        assert _bbox_iou([0.0, 0.0, 0.4, 0.4], [0.2, 0.2, 0.4, 0.4]) == pytest.approx(
            0.04 / 0.28
        )

    def test_contained_box(self):
        # Small box fully inside large box: intersection = small area
        assert _bbox_iou([0.0, 0.0, 1.0, 1.0], [0.4, 0.4, 0.2, 0.2]) == pytest.approx(
            0.04 / 1.0
        )


# ---------------------------------------------------------------------------
# Region filtering / merging / de-duplication
# ---------------------------------------------------------------------------

def _region(source, x, y, w, h):
    return {"source": source, "bbox": [x, y, w, h]}


class TestMergeCandidateRegions:
    def test_drops_tiny_regions(self):
        # A 0.5%-of-page logo should be filtered out
        regions = [
            _region("image", 0.1, 0.1, 0.5, 0.4),   # real figure (20%)
            _region("image", 0.9, 0.02, 0.05, 0.05),  # tiny logo (0.25%)
        ]
        merged = _merge_candidate_regions(regions)
        assert len(merged) == 1
        assert merged[0]["bbox"][2] == pytest.approx(0.5)

    def test_drops_full_page_background(self):
        regions = [
            _region("drawing", 0.0, 0.0, 1.0, 0.98),  # page-wide background artifact
            _region("image", 0.1, 0.1, 0.5, 0.4),
        ]
        merged = _merge_candidate_regions(regions)
        assert len(merged) == 1
        assert merged[0]["source"] == "image"

    def test_merges_overlapping_fragments(self):
        # Two overlapping fragments of a composite figure -> single merged region
        regions = [
            _region("image", 0.1, 0.1, 0.4, 0.3),
            _region("drawing", 0.3, 0.2, 0.4, 0.3),
        ]
        merged = _merge_candidate_regions(regions)
        assert len(merged) == 1
        assert merged[0]["source"] == "merged"
        # Merged bbox spans both
        x, y, w, h = merged[0]["bbox"]
        assert x == pytest.approx(0.1)
        assert y == pytest.approx(0.1)
        assert x + w == pytest.approx(0.7)
        assert y + h == pytest.approx(0.5)

    def test_merges_vertically_adjacent_fragments(self):
        # Gap below 2% of page height -> merged (e.g. plot + axis labels)
        regions = [
            _region("drawing", 0.1, 0.10, 0.5, 0.20),
            _region("drawing", 0.1, 0.31, 0.5, 0.20),  # 1% gap
        ]
        merged = _merge_candidate_regions(regions)
        assert len(merged) == 1

    def test_keeps_distant_regions_separate(self):
        regions = [
            _region("image", 0.1, 0.05, 0.5, 0.25),
            _region("table", 0.1, 0.60, 0.7, 0.25),
        ]
        merged = _merge_candidate_regions(regions)
        assert len(merged) == 2

    def test_dedupes_same_area_keeping_table(self):
        # find_tables and cluster_drawings often detect the same table;
        # the table source should win
        regions = [
            _region("table", 0.1, 0.1, 0.6, 0.3),
            _region("drawing", 0.11, 0.11, 0.59, 0.29),  # IoU > 0.8 with the table
        ]
        merged = _merge_candidate_regions(regions)
        assert len(merged) == 1
        assert merged[0]["source"] == "table"

    def test_empty_input(self):
        assert _merge_candidate_regions([]) == []


# ---------------------------------------------------------------------------
# Caption <-> region association
# ---------------------------------------------------------------------------

def _caption(label, kind, text, x, y, w, h):
    return {"label": label, "kind": kind, "text": text, "bbox": [x, y, w, h]}


class TestAssociateCaptionsWithRegions:
    def test_figure_caption_below_figure(self):
        regions = [_region("image", 0.1, 0.1, 0.6, 0.3)]
        captions = [_caption("Figure 3", "figure", "Figure 3: Results.", 0.1, 0.42, 0.6, 0.04)]

        result = _associate_captions_with_regions(regions, captions)

        assert result[0]["caption_label"] == "Figure 3"
        assert result[0]["caption_text"] == "Figure 3: Results."
        assert result[0]["confidence"] == "high"

    def test_table_caption_above_table(self):
        regions = [_region("table", 0.1, 0.5, 0.7, 0.3)]
        captions = [_caption("Table 1", "table", "Table 1: Hyperparameters.", 0.1, 0.44, 0.7, 0.04)]

        result = _associate_captions_with_regions(regions, captions)

        assert result[0]["caption_label"] == "Table 1"
        assert result[0]["confidence"] == "high"

    def test_caption_too_far_not_attached(self):
        regions = [_region("image", 0.1, 0.05, 0.5, 0.15)]
        # Caption is 40% of page height below the region -> beyond max distance
        captions = [_caption("Figure 1", "figure", "Figure 1: Far away.", 0.1, 0.6, 0.5, 0.04)]

        result = _associate_captions_with_regions(regions, captions)

        assert result[0]["caption_label"] is None
        assert result[0]["confidence"] == "low"

    def test_two_column_caption_not_cross_attached(self):
        # Region in right column, caption in left column at similar height:
        # no horizontal overlap -> must not attach
        regions = [_region("image", 0.55, 0.1, 0.4, 0.3)]
        captions = [_caption("Figure 2", "figure", "Figure 2: Left column.", 0.05, 0.42, 0.4, 0.04)]

        result = _associate_captions_with_regions(regions, captions)

        assert result[0]["caption_label"] is None

    def test_caption_attaches_to_nearest_of_two_stacked_figures(self):
        # Two stacked figures, caption right below the lower one
        regions = [
            _region("image", 0.1, 0.05, 0.6, 0.25),
            _region("image", 0.1, 0.45, 0.6, 0.25),
        ]
        captions = [_caption("Figure 5", "figure", "Figure 5: Lower figure.", 0.1, 0.72, 0.6, 0.04)]

        result = _associate_captions_with_regions(regions, captions)

        upper, lower = result[0], result[1]
        assert upper["caption_label"] is None
        assert lower["caption_label"] == "Figure 5"

    def test_each_caption_attaches_to_one_region_only(self):
        # One caption between two equally-near figures attaches to exactly one
        regions = [
            _region("image", 0.1, 0.1, 0.6, 0.2),
            _region("image", 0.1, 0.4, 0.6, 0.2),
        ]
        captions = [_caption("Figure 1", "figure", "Figure 1: Between.", 0.1, 0.32, 0.6, 0.05)]

        result = _associate_captions_with_regions(regions, captions)

        attached = [r for r in result if r["caption_label"] == "Figure 1"]
        assert len(attached) == 1

    def test_region_without_caption_gets_low_confidence(self):
        regions = [_region("drawing", 0.1, 0.1, 0.4, 0.2)]

        result = _associate_captions_with_regions(regions, [])

        assert result[0]["caption_label"] is None
        assert result[0]["caption_text"] is None
        assert result[0]["confidence"] == "low"

    def test_no_regions(self):
        captions = [_caption("Figure 1", "figure", "Figure 1: Orphan.", 0.1, 0.5, 0.5, 0.04)]
        assert _associate_captions_with_regions([], captions) == []


# ---------------------------------------------------------------------------
# detect_page_regions (real PyMuPDF on synthetic PDFs)
# ---------------------------------------------------------------------------

class TestDetectPageRegions:
    """Integration tests that build synthetic PDFs with real PyMuPDF.

    Skipped when PyMuPDF is not installed (e.g. CI installs the base
    package without the `pdf` extra). The pure-logic and MCP-tool tests
    above/below do not need PyMuPDF and always run.
    """

    @pytest.fixture(autouse=True)
    def _require_fitz(self):
        pytest.importorskip("fitz")

    def test_detects_raster_image_with_caption(self):
        import fitz

        with tempfile.TemporaryDirectory() as tmpdir:
            doc = fitz.open()
            page = _new_letter_page(doc)
            page.insert_image(fitz.Rect(100, 150, 400, 380), stream=_gray_image_bytes())
            page.insert_text(
                fitz.Point(100, 400), "Figure 1: Synthetic test figure.", fontsize=10
            )
            path = _save_pdf(doc, tmpdir)

            result = detect_page_regions(path, 1)

        assert "error" not in result
        assert result["pageIndex"] == 0
        assert len(result["regions"]) == 1

        region = result["regions"][0]
        assert region["source"] == "image"
        assert region["caption_label"] == "Figure 1"
        assert "Synthetic test figure" in region["caption_text"]
        assert region["confidence"] == "high"

        # bbox covers roughly the inserted rect (normalized to letter page)
        x, y, w, h = region["bbox"]
        assert 0.1 < x < 0.25
        assert 0.15 < y < 0.25
        assert 0.4 < w < 0.55
        assert 0.25 < h < 0.35

    def test_detects_vector_drawing(self):
        import fitz

        with tempfile.TemporaryDirectory() as tmpdir:
            doc = fitz.open()
            page = _new_letter_page(doc)
            page.draw_rect(
                fitz.Rect(100, 450, 300, 560), color=(0, 0, 1), fill=(0.8, 0.8, 0.9)
            )
            page.draw_line(
                fitz.Point(110, 470), fitz.Point(290, 470), color=(1, 0, 0)
            )
            path = _save_pdf(doc, tmpdir)

            result = detect_page_regions(path, 1)

        assert "error" not in result
        assert len(result["regions"]) == 1
        assert result["regions"][0]["source"] in ("drawing", "merged")
        assert result["regions"][0]["caption_label"] is None
        assert result["regions"][0]["confidence"] == "low"

    def test_detects_table_keeping_table_source(self):
        import fitz

        with tempfile.TemporaryDirectory() as tmpdir:
            doc = fitz.open()
            page = _new_letter_page(doc)
            # Lined 2x3 grid with cell text -> detected by both find_tables
            # and cluster_drawings; the table source must win de-duplication
            xs, ys = [400, 460, 520], [450, 480, 510, 540]
            for grid_y in ys:
                page.draw_line(fitz.Point(xs[0], grid_y), fitz.Point(xs[-1], grid_y))
            for grid_x in xs:
                page.draw_line(fitz.Point(grid_x, ys[0]), fitz.Point(grid_x, ys[-1]))
            for row, grid_y in enumerate(ys[:-1]):
                for col, grid_x in enumerate(xs[:-1]):
                    page.insert_text(
                        fitz.Point(grid_x + 5, grid_y + 20), f"R{row}C{col}", fontsize=8
                    )
            page.insert_text(
                fitz.Point(400, 440), "Table 1: Synthetic table.", fontsize=10
            )
            path = _save_pdf(doc, tmpdir)

            result = detect_page_regions(path, 1)

        assert "error" not in result
        table_regions = [r for r in result["regions"] if r["source"] == "table"]
        assert len(table_regions) == 1
        assert table_regions[0]["caption_label"] == "Table 1"

    def test_text_only_page_returns_no_regions(self):
        import fitz

        with tempfile.TemporaryDirectory() as tmpdir:
            doc = fitz.open()
            page = _new_letter_page(doc)
            page.insert_text(
                fitz.Point(72, 100),
                "This page contains nothing but plain paragraph text.",
                fontsize=11,
            )
            path = _save_pdf(doc, tmpdir)

            result = detect_page_regions(path, 1)

        assert "error" not in result
        assert result["regions"] == []

    def test_page_out_of_range_returns_error(self):
        import fitz

        with tempfile.TemporaryDirectory() as tmpdir:
            doc = fitz.open()
            _new_letter_page(doc)
            path = _save_pdf(doc, tmpdir)

            result = detect_page_regions(path, 5)

        assert "error" in result
        assert "out of range" in result["error"]

    def test_full_page_scan_returns_warning(self):
        import fitz

        with tempfile.TemporaryDirectory() as tmpdir:
            doc = fitz.open()
            page = _new_letter_page(doc)
            # Image covering the entire page, no text layer -> scanned page
            page.insert_image(
                fitz.Rect(0, 0, 612, 792),
                stream=_gray_image_bytes(),
                keep_proportion=False,
            )
            path = _save_pdf(doc, tmpdir)

            result = detect_page_regions(path, 1)

        assert "error" not in result
        assert result["regions"] == []
        assert any("scan" in w.lower() for w in result["warnings"])

    def test_missing_cluster_drawings_adds_warning(self, monkeypatch):
        import fitz

        with tempfile.TemporaryDirectory() as tmpdir:
            doc = fitz.open()
            page = _new_letter_page(doc)
            page.insert_image(fitz.Rect(100, 150, 400, 380), stream=_gray_image_bytes())
            path = _save_pdf(doc, tmpdir)

            # Simulate an older PyMuPDF without Page.cluster_drawings
            monkeypatch.delattr(fitz.Page, "cluster_drawings")

            result = detect_page_regions(path, 1)

        assert "error" not in result
        # Raster image still detected
        assert len(result["regions"]) == 1
        assert result["regions"][0]["source"] == "image"
        assert any("pymupdf" in w.lower() for w in result["warnings"])

    def test_regions_have_sequential_ids_and_normalized_bboxes(self):
        import fitz

        with tempfile.TemporaryDirectory() as tmpdir:
            doc = fitz.open()
            page = _new_letter_page(doc)
            page.insert_image(fitz.Rect(100, 100, 350, 280), stream=_gray_image_bytes())
            page.insert_image(fitz.Rect(100, 500, 350, 680), stream=_gray_image_bytes())
            path = _save_pdf(doc, tmpdir)

            result = detect_page_regions(path, 1)

        regions = result["regions"]
        assert [r["region_id"] for r in regions] == [1, 2]
        for region in regions:
            x, y, w, h = region["bbox"]
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0
            assert 0.0 < w <= 1.0
            assert 0.0 < h <= 1.0
            assert x + w <= 1.0 + 1e-6
            assert y + h <= 1.0 + 1e-6
        # Sorted top-to-bottom
        assert regions[0]["bbox"][1] < regions[1]["bbox"][1]

    def test_rotated_page_bboxes_stay_normalized(self):
        import fitz

        with tempfile.TemporaryDirectory() as tmpdir:
            doc = fitz.open()
            page = _new_letter_page(doc)
            page.insert_image(fitz.Rect(100, 150, 400, 380), stream=_gray_image_bytes())
            page.set_rotation(90)
            path = _save_pdf(doc, tmpdir)

            result = detect_page_regions(path, 1)

        assert "error" not in result
        assert len(result["regions"]) == 1
        x, y, w, h = result["regions"][0]["bbox"]
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0
        assert 0.0 < w <= 1.0
        assert 0.0 < h <= 1.0
        assert x + w <= 1.0 + 1e-6
        assert y + h <= 1.0 + 1e-6

    def test_cropped_page_bboxes_stay_normalized(self):
        # Published PDFs often have CropBox != MediaBox; bboxes must be
        # normalized against the visible (cropped) page area
        import fitz

        with tempfile.TemporaryDirectory() as tmpdir:
            doc = fitz.open()
            page = _new_letter_page(doc)
            page.insert_image(fitz.Rect(100, 150, 400, 380), stream=_gray_image_bytes())
            page.insert_text(
                fitz.Point(100, 400), "Figure 1: Cropped page figure.", fontsize=10
            )
            page.set_cropbox(fitz.Rect(50, 50, 562, 742))
            path = _save_pdf(doc, tmpdir)

            result = detect_page_regions(path, 1)

        assert "error" not in result
        assert len(result["regions"]) == 1

        region = result["regions"][0]
        x, y, w, h = region["bbox"]
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0
        assert 0.0 < w <= 1.0
        assert 0.0 < h <= 1.0
        assert x + w <= 1.0 + 1e-6
        assert y + h <= 1.0 + 1e-6
        # Caption association must survive the cropbox shift
        assert region["caption_label"] == "Figure 1"

    def test_invalid_pdf_returns_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "broken.pdf")
            with open(path, "wb") as f:
                f.write(b"not a pdf at all")

            result = detect_page_regions(path, 1)

        assert "error" in result


# ---------------------------------------------------------------------------
# MCP tool: zotero_get_page_layout
# ---------------------------------------------------------------------------

def _pdf_attachment(key="ATTACH01", item_type="attachment", content_type="application/pdf"):
    return {
        "key": key,
        "data": {
            "itemType": item_type,
            "contentType": content_type,
            "filename": "paper.pdf",
            "title": "paper.pdf",
            "parentItem": "PARENT01",
        },
    }


def _layout_result(regions, warnings=None, page_index=6, page_label="7"):
    return {
        "pageIndex": page_index,
        "pageLabel": page_label,
        "regions": regions,
        "warnings": warnings or [],
    }


def _layout_region(region_id=1, source="image", bbox=None, caption_label=None,
                   caption_text=None, confidence="low"):
    return {
        "region_id": region_id,
        "source": source,
        "bbox": bbox or [0.1, 0.2, 0.5, 0.3],
        "caption_label": caption_label,
        "caption_text": caption_text,
        "confidence": confidence,
    }


class TestGetPageLayoutTool:
    def _setup_clients(self, monkeypatch, fake_zot, items=None):
        fake_zot._items = items if items is not None else [_pdf_attachment()]
        monkeypatch.setattr("zotero_mcp.client.get_web_zotero_client", lambda: fake_zot)
        monkeypatch.setattr("zotero_mcp.client.get_local_zotero_client", lambda: None)
        monkeypatch.setattr("zotero_mcp.client.get_active_library", lambda: None)

    def _patch_detection(self, monkeypatch, result):
        monkeypatch.setattr(
            "zotero_mcp.pdf_layout.detect_page_regions",
            lambda _path, _page: result,
        )

    def test_happy_path_renders_regions_table(self, monkeypatch, fake_zot):
        self._setup_clients(monkeypatch, fake_zot)
        self._patch_detection(monkeypatch, _layout_result([
            _layout_region(
                region_id=1,
                source="image",
                bbox=[0.1012, 0.2034, 0.6011, 0.3498],
                caption_label="Figure 3",
                caption_text="Figure 3: Mean completion rates.",
                confidence="high",
            ),
            _layout_region(
                region_id=2,
                source="table",
                bbox=[0.1003, 0.6201, 0.8014, 0.2005],
                caption_label="Table 1",
                caption_text="Table 1: Hyperparameters.",
                confidence="high",
            ),
        ]))

        result = server.get_page_layout(
            attachment_key="ATTACH01", page=7, ctx=DummyContext()
        )

        assert "Error" not in result
        assert "2 region" in result
        # Both regions with their metadata
        assert "Figure 3" in result
        assert "Table 1" in result
        assert "0.1012" in result
        assert "high" in result
        # Ready-to-paste call template references the same attachment and page
        assert "zotero_create_area_annotation" in result
        assert "attachment_key='ATTACH01'" in result
        assert "page=7" in result

    def test_no_regions_returns_guidance(self, monkeypatch, fake_zot):
        self._setup_clients(monkeypatch, fake_zot)
        self._patch_detection(monkeypatch, _layout_result([]))

        result = server.get_page_layout(
            attachment_key="ATTACH01", page=3, ctx=DummyContext()
        )

        assert "No figure/table regions detected" in result
        assert "explicit coordinates" in result

    def test_warnings_are_included(self, monkeypatch, fake_zot):
        self._setup_clients(monkeypatch, fake_zot)
        self._patch_detection(monkeypatch, _layout_result(
            [],
            warnings=["Page appears to be a full-page scan — region detection "
                      "and captions are unavailable."],
        ))

        result = server.get_page_layout(
            attachment_key="ATTACH01", page=1, ctx=DummyContext()
        )

        assert "full-page scan" in result

    def test_detection_error_is_returned(self, monkeypatch, fake_zot):
        self._setup_clients(monkeypatch, fake_zot)
        self._patch_detection(monkeypatch, {"error": "Page 99 out of range (PDF has 12 pages)"})

        result = server.get_page_layout(
            attachment_key="ATTACH01", page=99, ctx=DummyContext()
        )

        assert "Error" in result
        assert "out of range" in result

    def test_rejects_non_pdf_attachment(self, monkeypatch, fake_zot):
        self._setup_clients(
            monkeypatch, fake_zot, items=[_pdf_attachment(content_type="text/html")]
        )

        result = server.get_page_layout(
            attachment_key="ATTACH01", page=1, ctx=DummyContext()
        )

        assert "not a PDF attachment" in result

    def test_rejects_non_attachment_item(self, monkeypatch, fake_zot):
        self._setup_clients(
            monkeypatch, fake_zot, items=[_pdf_attachment(item_type="journalArticle")]
        )

        result = server.get_page_layout(
            attachment_key="ATTACH01", page=1, ctx=DummyContext()
        )

        assert "not an attachment" in result

    def test_rejects_invalid_page_number(self, monkeypatch, fake_zot):
        self._setup_clients(monkeypatch, fake_zot)

        result = server.get_page_layout(
            attachment_key="ATTACH01", page=0, ctx=DummyContext()
        )

        assert "Error" in result
        assert "page" in result.lower()

    def test_requires_a_zotero_client(self, monkeypatch):
        monkeypatch.setattr("zotero_mcp.client.get_web_zotero_client", lambda: None)
        monkeypatch.setattr("zotero_mcp.client.get_local_zotero_client", lambda: None)

        result = server.get_page_layout(
            attachment_key="ATTACH01", page=1, ctx=DummyContext()
        )

        assert "Error" in result

    def test_works_with_local_client_only(self, monkeypatch, fake_zot):
        # Read-only tool must work in local-API mode without web credentials
        # (unlike create_area_annotation, which needs web write access)
        fake_zot._items = [_pdf_attachment()]
        monkeypatch.setattr("zotero_mcp.client.get_web_zotero_client", lambda: None)
        monkeypatch.setattr("zotero_mcp.client.get_local_zotero_client", lambda: fake_zot)
        monkeypatch.setattr("zotero_mcp.client.get_active_library", lambda: None)
        self._patch_detection(monkeypatch, _layout_result([
            _layout_region(
                region_id=1,
                source="image",
                caption_label="Figure 1",
                caption_text="Figure 1: Local mode figure.",
                confidence="high",
            ),
        ]))

        result = server.get_page_layout(
            attachment_key="ATTACH01", page=2, ctx=DummyContext()
        )

        assert "Error" not in result
        assert "1 region" in result
        assert "Figure 1" in result
