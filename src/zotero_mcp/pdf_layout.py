"""
PDF page layout detection for area annotation coordinate grounding.

Detects candidate figure/table regions on a PDF page so that area
annotations (zotero_create_annotation with a `rect`) can be placed on real
detected content instead of guessed coordinates.

Detection sources (all PyMuPDF):
- raster images: page.get_image_info()
- vector graphics: page.cluster_drawings() (PyMuPDF >= 1.24.2)
- tables: page.find_tables()
- captions: text blocks matching "Figure N:" / "Table N:" patterns

Pipeline: collect raw candidates -> filter noise -> merge fragments ->
de-duplicate -> associate captions by spatial scoring.

Known limitation: detection is geometric, not semantic. Detected boxes
cover the graphical core of a figure/table; text labels inside figures and
unruled table header rows may fall outside the box.
"""

from __future__ import annotations

import re

from zotero_mcp.utils import install_hint

# Region filtering / merging thresholds (normalized page units)
LAYOUT_MIN_REGION_AREA = 0.01       # drop regions smaller than 1% of page area
LAYOUT_MAX_REGION_AREA = 0.95       # drop page-wide background artifacts
LAYOUT_MERGE_GAP = 0.02             # merge fragments closer than 2% of page size
LAYOUT_DEDUP_IOU = 0.8              # near-identical regions are duplicates
LAYOUT_CAPTION_MAX_DISTANCE = 0.15  # max caption-to-region vertical gap
LAYOUT_CAPTION_MIN_OVERLAP = 0.3    # min horizontal overlap ratio (column guard)
LAYOUT_RULE_MIN_COUNT = 3           # top, middle and bottom rule of a booktabs table
LAYOUT_RULE_MIN_WIDTH = 0.15        # a table rule spans at least 15% of the page width
LAYOUT_RULE_SPAN_TOLERANCE = 0.02   # rules of one table share left/right edges within 2%
LAYOUT_RULE_MAX_GAP = 0.12          # max vertical gap between consecutive rules of one table

# Source priority when de-duplicating overlapping detections
_LAYOUT_SOURCE_PRIORITY = {"table": 3, "image": 2, "drawing": 1, "merged": 0}

# Real captions start a text block with "Figure N:", "Fig. N.", "Table N:" etc.
# The trailing [.:] separator requirement rejects in-text sentences such as
# "Figure 3 shows the results".
_CAPTION_PATTERN = re.compile(
    r"^(?P<prefix>(?:Extended\s+Data\s+)?(?:Figure|Fig\.?|Table))\s+"
    r"(?P<number>[A-Za-z]?\d+[a-z]?)\s*[.:]\s+",
    re.IGNORECASE,
)


def _parse_caption_block(text: str) -> dict | None:
    """
    Parse a text block as a figure/table caption.

    Args:
        text: Full text of a PDF text block

    Returns:
        {"label": "Figure 3", "kind": "figure" | "table", "text": <block text>}
        or None if the block is not a caption.
    """
    if not text:
        return None

    stripped = text.strip()
    match = _CAPTION_PATTERN.match(stripped)
    if not match:
        return None

    prefix = match.group("prefix")
    kind = "table" if "tab" in prefix.lower() else "figure"
    return {
        "label": f"{prefix} {match.group('number')}",
        "kind": kind,
        "text": stripped,
    }


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection-over-union of two normalized [x, y, width, height] boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _bbox_union(box_a: list[float], box_b: list[float]) -> list[float]:
    """Smallest [x, y, width, height] box containing both boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return [x1, y1, x2 - x1, y2 - y1]


def _boxes_overlap_or_near(
    box_a: list[float],
    box_b: list[float],
    gap: float,
) -> bool:
    """True if two boxes overlap or are within `gap` of each other on both axes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    return not (
        bx > ax + aw + gap
        or bx + bw < ax - gap
        or by > ay + ah + gap
        or by + bh < ay - gap
    )


def _horizontal_overlap_ratio(box_a: list[float], box_b: list[float]) -> float:
    """Overlap of two boxes' x-ranges, relative to the narrower box."""
    ax, _, aw, _ = box_a
    bx, _, bw, _ = box_b

    overlap = min(ax + aw, bx + bw) - max(ax, bx)
    if overlap <= 0:
        return 0.0
    narrower = min(aw, bw)
    if narrower <= 0:
        return 0.0
    return overlap / narrower


def _merge_candidate_regions(
    regions: list[dict],
    *,
    min_area: float = LAYOUT_MIN_REGION_AREA,
    max_area: float = LAYOUT_MAX_REGION_AREA,
    gap: float = LAYOUT_MERGE_GAP,
    dedup_iou: float = LAYOUT_DEDUP_IOU,
) -> list[dict]:
    """
    Filter noise, de-duplicate, and merge fragmented candidate regions.

    Composite figures often appear as multiple raster/vector fragments
    (plot body, axis labels, legend). Overlapping or near-adjacent
    image/drawing fragments are merged into one region. Tables detected by
    find_tables are kept intact (already semantic).

    Args:
        regions: [{"source": "image" | "drawing" | "table", "bbox": [x, y, w, h]}]
                 with bboxes normalized to [0, 1]

    Returns:
        Cleaned region list, sorted top-to-bottom then left-to-right.
        Regions produced by merging fragments get source "merged".
    """
    # 1. Drop noise: tiny logos/icons/rules and page-wide background artifacts
    kept = []
    for region in regions:
        _, _, width, height = region["bbox"]
        area = width * height
        if area < min_area or area > max_area:
            continue
        kept.append({"source": region["source"], "bbox": list(region["bbox"])})

    # 2. De-duplicate near-identical detections, keeping the
    #    highest-priority source (e.g. a table found by both find_tables
    #    and cluster_drawings keeps the "table" source)
    deduped: list[dict] = []
    for region in sorted(
        kept,
        key=lambda r: _LAYOUT_SOURCE_PRIORITY.get(r["source"], 0),
        reverse=True,
    ):
        if any(
            _bbox_iou(region["bbox"], existing["bbox"]) > dedup_iou
            for existing in deduped
        ):
            continue
        deduped.append(region)

    # 3. Merge overlapping / near-adjacent image and drawing fragments
    #    to a fixed point. Tables do not participate in merging.
    fixed = [r for r in deduped if r["source"] == "table"]
    mergeable = [r for r in deduped if r["source"] != "table"]

    while True:
        merged_any = False
        merged: list[dict] = []
        for region in mergeable:
            target = None
            for existing in merged:
                if _boxes_overlap_or_near(region["bbox"], existing["bbox"], gap):
                    target = existing
                    break
            if target is None:
                merged.append(region)
            else:
                target["bbox"] = _bbox_union(target["bbox"], region["bbox"])
                target["source"] = "merged"
                merged_any = True
        mergeable = merged
        if not merged_any:
            break

    result = fixed + mergeable
    result.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return result


def _associate_captions_with_regions(
    regions: list[dict],
    captions: list[dict],
    *,
    max_distance: float = LAYOUT_CAPTION_MAX_DISTANCE,
    min_overlap: float = LAYOUT_CAPTION_MIN_OVERLAP,
) -> list[dict]:
    """
    Attach captions to regions by spatial scoring.

    Score components:
    - vertical proximity (hard cutoff at max_distance)
    - horizontal x-range overlap (hard cutoff at min_overlap — this is the
      guard that keeps two-column layouts from cross-attaching)
    - type prior: figure captions conventionally sit below figures, table
      captions above tables (a score bonus, not a hard rule)

    Each caption attaches to at most one region and vice versa (best score
    wins, greedy assignment).

    Args:
        regions: [{"source": ..., "bbox": [x, y, w, h]}]
        captions: [{"label": ..., "kind": ..., "text": ..., "bbox": [x, y, w, h]}]

    Returns:
        New list of region dicts (same order) augmented with caption_label,
        caption_text, and confidence ("high" | "medium" | "low").
    """
    result = []
    for region in regions:
        augmented = dict(region)
        augmented["caption_label"] = None
        augmented["caption_text"] = None
        augmented["confidence"] = "low"
        result.append(augmented)

    if not result or not captions:
        return result

    # Score every (caption, region) pair that passes the hard cutoffs
    candidates: list[tuple[float, int, int]] = []
    for c_idx, caption in enumerate(captions):
        c_x, c_y, c_w, c_h = caption["bbox"]
        for r_idx, region in enumerate(result):
            r_x, r_y, r_w, r_h = region["bbox"]

            overlap = _horizontal_overlap_ratio(caption["bbox"], region["bbox"])
            if overlap < min_overlap:
                continue

            below_gap = c_y - (r_y + r_h)   # >= 0 when caption is below region
            above_gap = r_y - (c_y + c_h)   # >= 0 when caption is above region
            if below_gap >= 0:
                vertical_gap, position = below_gap, "below"
            elif above_gap >= 0:
                vertical_gap, position = above_gap, "above"
            else:
                vertical_gap, position = 0.0, "overlapping"

            if vertical_gap > max_distance:
                continue

            proximity = 1.0 - (vertical_gap / max_distance)
            prior = 0.0
            if caption["kind"] == "figure" and position == "below":
                prior = 1.0
            elif caption["kind"] == "table" and position == "above":
                prior = 1.0

            score = proximity * 0.5 + overlap * 0.3 + prior * 0.2
            candidates.append((score, c_idx, r_idx))

    # Greedy assignment: best score first, each caption/region used once
    candidates.sort(key=lambda item: item[0], reverse=True)
    used_captions: set[int] = set()
    used_regions: set[int] = set()
    for score, c_idx, r_idx in candidates:
        if c_idx in used_captions or r_idx in used_regions:
            continue
        used_captions.add(c_idx)
        used_regions.add(r_idx)

        caption = captions[c_idx]
        result[r_idx]["caption_label"] = caption["label"]
        result[r_idx]["caption_text"] = caption["text"]

        # Confidence reflects the margin over competing assignments
        competing = [s for s, ci, ri in candidates if ci == c_idx and ri != r_idx]
        if not competing or score - max(competing) >= 0.15:
            result[r_idx]["confidence"] = "high"
        else:
            result[r_idx]["confidence"] = "medium"

    return result


def _ruled_table_boxes(
    drawings: list[dict],
    page_width: float,
    page_height: float,
    *,
    min_rules: int = LAYOUT_RULE_MIN_COUNT,
    min_width: float = LAYOUT_RULE_MIN_WIDTH,
    span_tolerance: float = LAYOUT_RULE_SPAN_TOLERANCE,
    max_gap: float = LAYOUT_RULE_MAX_GAP,
) -> list[tuple[float, float, float, float]]:
    """
    Find tables drawn with horizontal rules only (booktabs style).

    find_tables() needs a grid, and most papers typeset tables with a top,
    middle and bottom rule and no vertical lines at all, so it finds nothing.
    Those rules are still unmistakable: several horizontal lines with the same
    left and right edge, stacked close together. The table is the box from the
    first rule to the last.

    Partial rules (cmidrule under a column group) have a different span and
    are ignored; a lone footnote rule never reaches min_rules.

    Returns:
        (x0, y0, x1, y1) boxes in page coordinates.
    """
    rules = []
    for drawing in drawings:
        rect = drawing.get("rect")
        if rect is None:
            continue
        if rect.height > 2 or rect.width < page_width * min_width:
            continue
        rules.append((rect.x0, rect.x1, (rect.y0 + rect.y1) / 2))
    rules.sort(key=lambda rule: rule[2])

    groups: list[list] = []  # [x0, x1, [y, ...]]
    for x0, x1, y in rules:
        for group in groups:
            same_span = (
                abs(group[0] - x0) <= span_tolerance * page_width
                and abs(group[1] - x1) <= span_tolerance * page_width
            )
            if same_span and y - group[2][-1] <= max_gap * page_height:
                # PDFs often draw a rule twice; count it once.
                if y - group[2][-1] > 0.5:
                    group[2].append(y)
                break
        else:
            groups.append([x0, x1, [y]])

    return [
        (group[0], group[2][0], group[1], group[2][-1])
        for group in groups
        if len(group[2]) >= min_rules
    ]


def _absorb_uncaptioned_panels(
    regions: list[dict],
    captions: list[dict],
    *,
    max_distance: float = LAYOUT_CAPTION_MAX_DISTANCE,
) -> list[dict]:
    """
    Fold side-by-side panels of one figure into the region that won its caption.

    A figure made of two panels ("(left) ... (right) ...") detects as two
    regions too far apart to merge, and caption association gives the caption
    to only one of them. The other is reported as a separate, captionless,
    low-confidence region, so annotating "Figure 2" boxes half of it.

    A captionless region is absorbed when it sits above the same figure
    caption, inside the caption's horizontal extent (the column guard), and
    shares most of its vertical band with the captioned region.
    """
    by_label = {caption["label"]: caption for caption in captions}
    result = [dict(region) for region in regions]
    absorbed: set[int] = set()

    for idx, region in enumerate(result):
        caption = by_label.get(region.get("caption_label"))
        if caption is None or caption["kind"] != "figure":
            continue
        c_x, c_y, c_w, _c_h = caption["bbox"]
        for other_idx, other in enumerate(result):
            if other_idx == idx or other_idx in absorbed or other.get("caption_label"):
                continue
            o_x, o_y, o_w, o_h = other["bbox"]
            gap = c_y - (o_y + o_h)
            if gap < 0 or gap > max_distance:
                continue
            if o_x < c_x - 0.01 or o_x + o_w > c_x + c_w + 0.01:
                continue
            r_x, r_y, r_w, r_h = region["bbox"]
            shared = min(r_y + r_h, o_y + o_h) - max(r_y, o_y)
            if shared < 0.5 * min(r_h, o_h):
                continue
            region["bbox"] = _bbox_union(region["bbox"], other["bbox"])
            region["source"] = "merged"
            absorbed.add(other_idx)

    return [region for idx, region in enumerate(result) if idx not in absorbed]


def detect_page_regions(pdf_path: str, page_num: int) -> dict:
    """
    Detect candidate figure/table regions on a PDF page.

    Provides coordinate grounding for area annotations: instead of guessing
    normalized coordinates, callers can pick from real detected regions.

    Detection sources:
    - raster images: page.get_image_info()
    - vector graphics: page.cluster_drawings() (PyMuPDF >= 1.24.2)
    - tables: page.find_tables()
    - captions: text blocks matching "Figure N:" / "Table N:" patterns

    Args:
        pdf_path: Path to the PDF file
        page_num: 1-indexed page number

    Returns:
        On success:
            {
                "pageIndex": int,        # 0-indexed
                "pageLabel": str,
                "regions": [
                    {
                        "region_id": int,
                        "source": "image" | "drawing" | "table" | "merged",
                        "bbox": [x, y, width, height],   # normalized [0, 1]
                        "caption_label": str | None,
                        "caption_text": str | None,
                        "confidence": "high" | "medium" | "low",
                    },
                    ...
                ],
                "warnings": [str, ...],
            }

        On failure:
            {"error": str}
    """
    try:
        import fitz
    except ImportError:
        raise ImportError(
            f"PDF layout detection requires PyMuPDF. {install_hint('pdf')}"
        )

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return {"error": f"Could not open PDF: {e}"}

    try:
        if not doc.is_pdf:
            return {"error": "File is not a valid PDF"}

        target_index = page_num - 1
        total_pages = len(doc)
        if target_index < 0 or target_index >= total_pages:
            return {
                "error": f"Page {page_num} out of range (PDF has {total_pages} pages)",
            }

        page = doc[target_index]
        page_width = page.rect.width
        page_height = page.rect.height
        if page_width <= 0 or page_height <= 0:
            return {"error": f"Page {page_num} has invalid dimensions"}

        def normalize_bbox(x0: float, y0: float, x1: float, y1: float) -> list[float]:
            """Convert page coordinates to clamped normalized [x, y, w, h]."""
            x = min(max(x0 / page_width, 0.0), 1.0)
            y = min(max(y0 / page_height, 0.0), 1.0)
            w = min(max((x1 - x0) / page_width, 0.0), 1.0 - x)
            h = min(max((y1 - y0) / page_height, 0.0), 1.0 - y)
            return [x, y, w, h]

        warnings: list[str] = []
        raw_regions: list[dict] = []

        # --- Raster images ---
        try:
            for info in page.get_image_info():
                x0, y0, x1, y1 = info["bbox"]
                raw_regions.append(
                    {"source": "image", "bbox": normalize_bbox(x0, y0, x1, y1)}
                )
        except Exception:
            pass

        # --- Vector graphics (clustered) ---
        if hasattr(page, "cluster_drawings"):
            try:
                for rect in page.cluster_drawings():
                    raw_regions.append(
                        {
                            "source": "drawing",
                            "bbox": normalize_bbox(rect.x0, rect.y0, rect.x1, rect.y1),
                        }
                    )
            except Exception:
                pass
        else:
            warnings.append(
                "Vector graphics detection requires pymupdf>=1.24.2; "
                "showing raster images and tables only."
            )

        # --- Tables ---
        try:
            # PyMuPDF prints an advert for pymupdf_layout to stdout on its
            # first find_tables() call, which corrupts `zotero-cli --json`.
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                found_tables = page.find_tables().tables
            for table in found_tables:
                x0, y0, x1, y1 = table.bbox
                raw_regions.append(
                    {"source": "table", "bbox": normalize_bbox(x0, y0, x1, y1)}
                )
        except Exception:
            pass

        # --- Tables drawn with horizontal rules only (find_tables misses these) ---
        try:
            for x0, y0, x1, y1 in _ruled_table_boxes(page.get_drawings(), page_width, page_height):
                raw_regions.append(
                    {"source": "table", "bbox": normalize_bbox(x0, y0, x1, y1)}
                )
        except Exception:
            pass

        # --- Text blocks (for captions and scanned-page detection) ---
        text_blocks: list[dict] = []
        try:
            for block in page.get_text("blocks"):
                x0, y0, x1, y1, block_text, _block_no, block_type = block[:7]
                if block_type == 0 and block_text.strip():
                    text_blocks.append(
                        {
                            "bbox": normalize_bbox(x0, y0, x1, y1),
                            "text": block_text.strip(),
                        }
                    )
        except Exception:
            pass

        # Scanned page: a page-covering image with no text layer is a scan,
        # not an annotatable figure
        full_page_images = [
            r
            for r in raw_regions
            if r["source"] == "image"
            and r["bbox"][2] * r["bbox"][3] > LAYOUT_MAX_REGION_AREA
        ]
        if full_page_images and not text_blocks:
            warnings.append(
                "Page appears to be a full-page scan — region detection and "
                "captions are unavailable."
            )
            return {
                "pageIndex": target_index,
                "pageLabel": _page_label_or_default(page, page_num),
                "regions": [],
                "warnings": warnings,
            }

        # --- Captions ---
        captions: list[dict] = []
        for block in text_blocks:
            parsed = _parse_caption_block(block["text"])
            if parsed:
                captions.append({**parsed, "bbox": block["bbox"]})

        # --- Pipeline: filter/merge/dedupe, then attach captions ---
        regions = _merge_candidate_regions(raw_regions)
        regions = _associate_captions_with_regions(regions, captions)
        regions = _absorb_uncaptioned_panels(regions, captions)

        for idx, region in enumerate(regions, start=1):
            region["region_id"] = idx
            region["bbox"] = [round(value, 4) for value in region["bbox"]]

        return {
            "pageIndex": target_index,
            "pageLabel": _page_label_or_default(page, page_num),
            "regions": regions,
            "warnings": warnings,
        }

    finally:
        doc.close()


def _page_label_or_default(page, page_num: int) -> str:
    """Return the page's printed label, falling back to the 1-indexed number."""
    try:
        label = page.get_label()
        if label:
            return label
    except Exception:
        pass
    return str(page_num)
