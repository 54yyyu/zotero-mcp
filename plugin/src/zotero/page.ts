// "This page" in the composer's + popup: the reader's current page as an image chip, copied from the canvas pdf.js
// has already drawn (at the current zoom), on white, at most 1568 px on the long edge (what a model looks at anyway).
import type { ContextChip, ItemHit } from "../types.ts";
import { itemLabel, refOf, viewStats } from "./context.ts";
import { t } from "../i18n.ts";

const MAX_EDGE = 1568;

/** The page hit, first in the + popup while a PDF is open (and the query is empty or asks for a page). */
export function pageHit(reader: any, query: string): ItemHit | null {
  const st = viewStats(reader);
  const att = reader?._item;
  if (!st || !att || (query.trim() && !/^(this )?page/i.test(query.trim()))) return null;
  return { kind: "page", ref: { ...refOf(att), pageIndex: st.pageIndex, pageLabel: st.pageLabel }, title: t("page.this", { page: st.pageLabel }), subtitle: t("page.attach", { item: itemLabel(att.parentItem ?? att) }) };
}

export function pageChip(win: any, reader: any, hit: ItemHit): ContextChip {
  const view = reader?._internalReader?._primaryView;
  const frame = view?._iframeWindow;
  const app = (frame?.wrappedJSObject ?? frame)?.PDFViewerApplication;
  const src = app?.pdfViewer?.getPageView(hit.ref.pageIndex ?? 0)?.canvas;
  if (!src?.width) throw new Error(t("page.notDrawn"));
  const k = Math.min(1, MAX_EDGE / Math.max(src.width, src.height));
  const canvas = win.document.createElementNS("http://www.w3.org/1999/xhtml", "canvas");
  canvas.width = Math.round(src.width * k);
  canvas.height = Math.round(src.height * k);
  const g = canvas.getContext("2d");
  g.fillStyle = "#ffffff";
  g.fillRect(0, 0, canvas.width, canvas.height);
  g.drawImage(src, 0, 0, canvas.width, canvas.height);
  const data = String(canvas.toDataURL("image/png")).split(",")[1] ?? "";
  return {
    id: `page:${hit.ref.attachmentKey}:${hit.ref.pageIndex}`, kind: "area", label: t("page.label", { page: hit.ref.pageLabel ?? "" }), auto: false, pinned: false,
    ref: hit.ref, image: { mime: "image/png", data },
  };
}
