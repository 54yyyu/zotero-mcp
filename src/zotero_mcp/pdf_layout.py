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

import os
import re

from zotero_mcp.pdf_utils import page_label, page_range_error
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
LAYOUT_EQUATION_MIN_MATH_SHARE = 0.6  # share of a text block's characters set in math fonts
LAYOUT_EQUATION_LINE_GAP = 0.012    # pieces of one display equation sit closer than this
LAYOUT_EQUATION_PIECE_GAP = 0.02    # side-by-side pieces of one display; below a column gutter
LAYOUT_NESTED_SHARE = 0.9           # a box this much inside a kept box is a fragment of it
LAYOUT_RULE_MIN_ROWS = 2            # text rows a rule-bounded table must hold (else a heading band)
LAYOUT_EQUATION_MIN_WIDTH = 0.05    # an unnumbered display narrower than this is a stray symbol

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


def _share_inside(inner: list[float], outer: list[float]) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer`` (both [x, y, w, h])."""
    x0 = max(inner[0], outer[0])
    y0 = max(inner[1], outer[1])
    x1 = min(inner[0] + inner[2], outer[0] + outer[2])
    y1 = min(inner[1] + inner[3], outer[1] + outer[3])
    area = inner[2] * inner[3]
    if area <= 0 or x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0) / area


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

    # 2. De-duplicate. Boxes are visited largest first, and one that nearly
    #    coincides with a kept box, or lies almost entirely inside it, is a
    #    fragment of that region -- the rule band of one table row, a partial
    #    table inside the drawing that frames it -- not a region of its own.
    #    The outer box is kept, since it holds the whole thing, and takes the
    #    stronger source of the two (a drawing framing a table is a table).
    priority = _LAYOUT_SOURCE_PRIORITY.get
    deduped: list[dict] = []
    for region in sorted(kept, key=lambda r: r["bbox"][2] * r["bbox"][3], reverse=True):
        outer = next(
            (
                existing for existing in deduped
                if _bbox_iou(region["bbox"], existing["bbox"]) > dedup_iou
                or _share_inside(region["bbox"], existing["bbox"]) >= LAYOUT_NESTED_SHARE
            ),
            None,
        )
        if outer is None:
            deduped.append(region)
        elif priority(region["source"], 0) > priority(outer["source"], 0):
            outer["source"] = region["source"]

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
    - type prior: figure captions conventionally sit below figures (a score
      bonus, not a hard rule); table captions sit above or below by venue,
      so they get none

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
            # Figure captions sit below figures almost universally. Table
            # captions go above or below depending on the venue, so for
            # tables proximity alone decides.
            prior = 1.0 if caption["kind"] == "figure" and position == "below" else 0.0

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


# Fonts that set math. TeX's roman fonts (CMR, CMBX) also appear in formulas,
# for digits and operators, but set ordinary text too, so they only count
# towards a line that already carries a real math font.
_MATH_FONT_RE = re.compile(
    r"^(?:[A-Z]{6}\+)?(?:CMMI|CMSY|CMEX|CMBSY|CMMIB|MSBM|MSAM|EUFM|EUSM|RSFS|"
    r"LMMath|LatinModernMath|STIX\w*Math|Cambria-?Math|XITSMath|MTSY|MTEX|MathematicalPi)",
    re.IGNORECASE,
)
_TEX_ROMAN_FONT_RE = re.compile(r"^(?:[A-Z]{6}\+)?(?:CMR|CMBX|LMRoman)", re.IGNORECASE)
_EQUATION_NUMBER_RE = re.compile(r"^\(\d+[a-z]?\)$")


def scan_math(page) -> tuple[list[dict], int]:
    """
    Display equations on a page, and how much inline math the rest carries.

    Text extraction garbles math (a square root becomes "p", subscripts fall
    onto the baseline), so a reader of extracted text needs to know where it
    cannot be trusted. Fonts say so reliably for LaTeX- and Word-produced PDFs:
    a text block set mostly in math fonts is a display equation.

    Pieces of one display (a fraction's denominator, the second line of an
    aligned pair) come out as separate blocks and are joined; an equation
    number "(3)" on the same band labels the result.

    Pages whose font list has no math font are skipped without extracting
    text, which keeps this cheap for prose-only pages.

    Returns:
        (equations, inline_math_chars). Each equation is
        {"bbox": (x0, y0, x1, y1) in page coordinates, "label": "(1)" | None}.
    """
    if not any(_MATH_FONT_RE.match(font[3] or "") for font in page.get_fonts()):
        return [], 0

    boxes: list[list[float]] = []
    numbers: list[tuple[tuple, str]] = []
    inline_math_chars = 0
    for block in page.get_text("dict", flags=0)["blocks"]:
        body = None
        total = mathy = real_math = 0
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if _EQUATION_NUMBER_RE.match(text):
                numbers.append((line["bbox"], text))
                continue
            for span in line["spans"]:
                count = len(span["text"].replace(" ", ""))
                total += count
                if _MATH_FONT_RE.match(span["font"]):
                    real_math += count
                    mathy += count
                elif _TEX_ROMAN_FONT_RE.match(span["font"]):
                    mathy += count
            x0, y0, x1, y1 = line["bbox"]
            body = [x0, y0, x1, y1] if body is None else [
                min(body[0], x0), min(body[1], y0), max(body[2], x1), max(body[3], y1)
            ]
        if body is not None and real_math and total >= 2 and mathy >= LAYOUT_EQUATION_MIN_MATH_SHARE * total:
            boxes.append(body)
        else:
            inline_math_chars += real_math

    # Join pieces to a fixed point. A display splits both ways: a fraction
    # stacks blocks vertically, and a long line breaks into side-by-side
    # blocks. The horizontal allowance is narrower than a column gutter, so
    # equations in the two columns of a page stay apart.
    v_gap = LAYOUT_EQUATION_LINE_GAP * page.rect.height
    h_gap = LAYOUT_EQUATION_PIECE_GAP * page.rect.width
    merged: list[list[float]] = []
    for box in sorted(boxes, key=lambda b: (b[1], b[0])):
        merged.append(box)
        joined = True
        while joined:
            joined = False
            current = merged[-1]
            for idx in range(len(merged) - 1):
                other = merged[idx]
                if (current[1] - other[3] <= v_gap and other[1] - current[3] <= v_gap
                        and current[0] - other[2] <= h_gap and other[0] - current[2] <= h_gap):
                    merged[-1] = [min(current[0], other[0]), min(current[1], other[1]),
                                  max(current[2], other[2]), max(current[3], other[3])]
                    del merged[idx]
                    joined = True
                    break

    # An equation number sits on the display's band, to its right, at the
    # column's margin. Two columns can put displays on one band, so take the
    # nearest number on that side and use each number once.
    equations = []
    used_numbers: set[int] = set()
    for x0, y0, x1, y1 in merged:
        candidates = [
            (max(nx0 - x1, 0.0), idx, text)
            for idx, ((nx0, ny0, _nx1, ny1), text) in enumerate(numbers)
            if idx not in used_numbers and ny0 < y1 + 2 and ny1 > y0 - 2 and nx0 >= x0
        ]
        label = None
        if candidates:
            _distance, idx, label = min(candidates)
            used_numbers.add(idx)
        if label is None and (x1 - x0) < LAYOUT_EQUATION_MIN_WIDTH * page.rect.width:
            continue  # a stray symbol, not a display
        pad = 2.0
        equations.append({"bbox": (x0 - pad, y0 - pad, x1 + pad, y1 + pad), "label": label})
    return equations, inline_math_chars


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


def detect_page_regions(pdf, page_num: int) -> dict:
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
        pdf: Path to the PDF file, or an open PyMuPDF document
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
    if not isinstance(pdf, (str, os.PathLike)):
        return _page_regions(pdf, page_num)

    try:
        import fitz
    except ImportError:
        raise ImportError(
            f"PDF layout detection requires PyMuPDF. {install_hint('pdf')}"
        )
    try:
        doc = fitz.open(pdf)
    except Exception as e:
        return {"error": f"Could not open PDF: {e}"}
    try:
        return _page_regions(doc, page_num)
    finally:
        doc.close()


def _page_regions(doc, page_num: int) -> dict:
    """detect_page_regions on an open document."""
    if not doc.is_pdf:
        return {"error": "File is not a valid PDF"}

    range_error = page_range_error(doc, page_num)
    if range_error:
        return {"error": range_error}
    target_index = page_num - 1

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
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    # find_tables() is the slowest step here and finds only grids, which
    # need vertical strokes. A page whose drawings are all horizontal
    # rules (or that has none) cannot hold one, so it skips the call; its
    # rule-only tables are found below.
    if any(drawing["rect"].height > 2 for drawing in drawings):
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
        ruled = _ruled_table_boxes(drawings, page_width, page_height)
        words = page.get_text("words") if ruled else []
        for x0, y0, x1, y1 in ruled:
            # Rules around a single line are a heading band (an
            # algorithm's title bar, say), not a table.
            rows = {
                round((w[1] + w[3]) / 6)
                for w in words
                if x0 - 2 <= w[0] and w[2] <= x1 + 2 and y0 < (w[1] + w[3]) / 2 < y1
            }
            if len(rows) >= LAYOUT_RULE_MIN_ROWS:
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
            "pageLabel": page_label(page, page_num),
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

    # --- Display equations: added after filtering, since a one-line
    #     equation is far below LAYOUT_MIN_REGION_AREA ---
    try:
        equations, _inline_math = scan_math(page)
    except Exception:
        equations = []
    for equation in equations:
        bbox = normalize_bbox(*equation["bbox"])
        cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
        if any(
            r["bbox"][0] <= cx <= r["bbox"][0] + r["bbox"][2]
            and r["bbox"][1] <= cy <= r["bbox"][1] + r["bbox"][3]
            for r in regions
        ):
            continue  # math inside a table or figure belongs to that region
        label = f"Equation {equation['label']}" if equation["label"] else "Equation"
        regions.append({
            "source": "equation",
            "bbox": bbox,
            "caption_label": label,
            "caption_text": label,
            "confidence": "high" if equation["label"] else "medium",
        })
    regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

    for idx, region in enumerate(regions, start=1):
        region["region_id"] = idx
        region["bbox"] = [round(value, 4) for value in region["bbox"]]

    return {
        "pageIndex": target_index,
        "pageLabel": page_label(page, page_num),
        "regions": regions,
        "warnings": warnings,
    }
