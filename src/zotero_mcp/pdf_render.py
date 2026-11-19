"""Page images with annotation boxes and a coordinate grid drawn on them.

Shared by ``zotero-cli annotations preview``, ``annotations batch --dry-run
--render`` and ``read --format image --grid``. Boxes are given as
``[x, y, w, h]`` normalized to the page as displayed, from the top-left corner.
Callers get them by passing *stored* annotation positions through
``pdf_utils.zotero_rects_to_shown``, the inverse of what the writer does, so a
write-side coordinate bug shows up in the picture instead of being copied into
it.

The page is rendered to a pixmap first and the overlay is drawn on a fresh
blank page that holds that image, so nothing is ever drawn into the source PDF
and a rotated page needs no special handling here.
"""

from __future__ import annotations

import json
import os

from zotero_mcp._pymupdf import fitz
from zotero_mcp.pdf_utils import zotero_rects_to_shown

DEFAULT_COLOR = "#a28ae5"
GRID_STEP = 0.05

#: Annotation types drawn as an outlined box; the rest are filled text rects.
_OUTLINED = {"image", "note"}


def _rgb(color: str | None) -> tuple[float, float, float]:
    """``#rrggbb`` -> 0-1 floats; anything else falls back to purple."""
    text = (color or DEFAULT_COLOR).lstrip("#")
    try:
        return tuple(int(text[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except ValueError:
        return _rgb(DEFAULT_COLOR)


def _latin(text: str) -> str:
    """Text the built-in label font can draw."""
    return text.encode("latin-1", "replace").decode("latin-1").replace("\n", " ")


def render_page(page, *, scale: float = 1.5, clip: list[float] | None = None,
                boxes: list[dict] | None = None, grid: bool = False) -> dict:
    """PNG of one page, or one region of it, with boxes and a grid drawn on top.

    Args:
        page: an open PyMuPDF page
        scale: pixels per PDF point
        clip: ``[x, y, w, h]`` region to render, normalized; the whole page when None
        boxes: dicts with ``rect`` ``[x, y, w, h]`` (normalized, whole-page frame),
            ``color`` (hex), ``label`` and ``kind`` (annotation type)
        grid: draw lines every 0.05, labelled every 0.1, in whole-page coordinates

    Returns:
        ``{"png", "width", "height", "boxes"}``; ``boxes`` are the drawn boxes in
        image pixels as ``[x0, y0, x1, y1]``, in the order given.
    """
    width, height = page.rect.width, page.rect.height
    cx, cy, cw, ch = clip or (0.0, 0.0, 1.0, 1.0)
    region = fitz.Rect(page.rect.x0 + cx * width, page.rect.y0 + cy * height,
                       page.rect.x0 + (cx + cw) * width, page.rect.y0 + (cy + ch) * height)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=region, alpha=False)
    if not grid and not boxes:
        return {"png": pixmap.tobytes("png"), "width": pixmap.width, "height": pixmap.height, "boxes": []}
    sx, sy = pixmap.width / region.width, pixmap.height / region.height

    def to_px(nx: float, ny: float) -> tuple[float, float]:
        return (nx * width - (region.x0 - page.rect.x0)) * sx, (ny * height - (region.y0 - page.rect.y0)) * sy

    canvas = fitz.open()
    sheet = canvas.new_page(width=pixmap.width, height=pixmap.height)
    sheet.insert_image(sheet.rect, pixmap=pixmap)
    font = max(8.0, min(13.0, 7.0 * scale))

    if grid:
        thin, bold = sheet.new_shape(), sheet.new_shape()
        for tick in range(0, int(round(1 / GRID_STEP)) + 1):
            value = tick * GRID_STEP
            major = tick % 2 == 0
            shape = bold if major else thin
            if cx - 1e-9 <= value <= cx + cw + 1e-9:
                x, _ = to_px(value, cy)
                shape.draw_line((x, 0), (x, sheet.rect.height))
                if major:
                    sheet.insert_text((x + 2, font + 1), f"{value:.1f}", fontsize=font, color=(0, 0.35, 0.75))
            if cy - 1e-9 <= value <= cy + ch + 1e-9:
                _, y = to_px(cx, value)
                shape.draw_line((0, y), (sheet.rect.width, y))
                if major:
                    sheet.insert_text((2, y - 2), f"{value:.1f}", fontsize=font, color=(0, 0.35, 0.75))
        thin.finish(color=(0, 0.5, 0.9), width=0.4, stroke_opacity=0.35)
        thin.commit()
        bold.finish(color=(0, 0.5, 0.9), width=0.7, stroke_opacity=0.6)
        bold.commit()

    drawn = []
    for box in boxes or []:
        x, y, w, h = box["rect"]
        (x0, y0), (x1, y1) = to_px(x, y), to_px(x + w, y + h)
        drawn.append([round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)])
        color = _rgb(box.get("color"))
        outlined = box.get("kind", "image") in _OUTLINED
        shape = sheet.new_shape()
        shape.draw_rect(fitz.Rect(x0, y0, x1, y1))
        shape.finish(color=color if outlined else None, fill=color,
                     fill_opacity=0.18 if outlined else 0.35,
                     stroke_opacity=1, width=max(1.5, scale))
        shape.commit()
        label = _latin(box.get("label") or "")
        if label:
            text_width = fitz.get_text_length(label, fontsize=font)
            top = y0 - font - 2 if y0 - font - 2 >= 0 else y0
            plate = fitz.Rect(x0, top, x0 + text_width + 4, top + font + 2)
            sheet.draw_rect(plate, color=None, fill=(1, 1, 1), fill_opacity=0.85)
            sheet.insert_text((x0 + 2, top + font - 1), label, fontsize=font,
                              color=tuple(c * 0.55 for c in color))  # darker, so yellow reads

    result = sheet.get_pixmap(alpha=False)
    png = result.tobytes("png")
    width_px, height_px = result.width, result.height
    canvas.close()
    return {"png": png, "width": width_px, "height": height_px, "boxes": drawn}


def boxes_from_position(page, position, *, color: str | None, label: str, kind: str) -> list[dict]:
    """Boxes for one annotation, from its stored ``annotationPosition``.

    ``position`` is the JSON string Zotero stores, or the parsed dict. A text
    annotation has a rect per line; only the first carries the label.
    """
    if isinstance(position, str):
        try:
            position = json.loads(position)
        except ValueError:
            return []
    rects = (position or {}).get("rects") or []
    return [
        {"rect": rect, "color": color, "label": label if index == 0 else "", "kind": kind}
        for index, rect in enumerate(zotero_rects_to_shown(page, rects))
    ]


def render_annotated_pages(doc, items: list[dict], out_dir: str, stem: str, *,
                           pages: list[int] | None = None, scale: float = 1.5) -> list[dict]:
    """One PNG per page, with every annotation in ``items`` drawn from its stored position.

    ``items`` are dicts with ``page_index``, ``position``, ``color``, ``label``
    and ``kind``. ``pages`` (1-indexed) limits and orders the output; by
    default every page that has an annotation is drawn. Files are named
    ``<stem>-p<page>.png``.

    Returns:
        One ``{"page", "path", "annotations", "width", "height", "boxes"}`` per image.
    """
    by_page: dict[int, list[dict]] = {}
    for item in items:
        by_page.setdefault(item["page_index"], []).append(item)
        # A highlight that runs over a page break keeps the part on the
        # following page under nextPageRects.
        position = item["position"]
        if isinstance(position, str):
            try:
                position = json.loads(position)
            except ValueError:
                position = None
        rest = (position or {}).get("nextPageRects")
        if rest:
            by_page.setdefault(item["page_index"] + 1, []).append(
                {**item, "position": {"rects": rest}, "label": ""})
    wanted = [p - 1 for p in pages] if pages else sorted(by_page)
    os.makedirs(out_dir, exist_ok=True)

    written = []
    for index in wanted:
        if not 0 <= index < len(doc):
            continue
        page = doc[index]
        boxes = []
        for item in by_page.get(index, []):
            boxes.extend(boxes_from_position(
                page, item["position"], color=item.get("color"),
                label=item.get("label", ""), kind=item.get("kind", "image")))
        image = render_page(page, scale=scale, boxes=boxes)
        path = os.path.join(out_dir, f"{stem}-p{index + 1}.png")
        with open(path, "wb") as handle:
            handle.write(image["png"])
        written.append({"page": index + 1, "path": path, "annotations": len(by_page.get(index, [])),
                        "width": image["width"], "height": image["height"], "boxes": image["boxes"]})
    return written
