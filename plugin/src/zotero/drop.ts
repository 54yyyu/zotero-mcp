// What was dropped on the composer. Zotero writes two formats (read from its source, not guessed):
//   zotero/item        "12,34": item ids (itemTree, Zotero.Utilities.Internal.onDragItems)
//   zotero/annotation  JSON array of annotations, each with attachmentItemID added by the reader (reader.js)
import type { ContextChip } from "../types.ts";
import type { ContextTracker } from "./context.ts";
import { itemLabel, refOf } from "./context.ts";

const MAX = 10;

export async function dropChips(tracker: ContextTracker, data: DataTransfer): Promise<ContextChip[]> {
  const chips: ContextChip[] = [];

  for (const id of (data.getData("zotero/item") || "").split(",").map(Number).filter(Boolean).slice(0, MAX)) {
    const it = Zotero.Items.get(id);
    if (!it || it.isAnnotation?.() || it.isNote?.()) continue;
    const shown = it.isAttachment() ? it.parentItem ?? it : it;
    chips.push({ id: `item:${shown.key}`, kind: "item", label: itemLabel(shown), auto: false, pinned: true, ref: refOf(it) });
  }

  let dropped: any[] = [];
  try { dropped = JSON.parse(data.getData("zotero/annotation") || "[]"); } catch { /* not ours */ }
  for (const a of dropped.slice(0, MAX)) {
    const att = Zotero.Items.get(a.attachmentItemID);
    if (!att) continue;
    const ann = a.id ? Zotero.Items.getByLibraryAndKey(att.libraryID, a.id) : null;
    if (ann?.isAnnotation?.()) {
      const image = ann.annotationType === "image" ? await tracker.loadImage(ann, att) : null;
      chips.push(tracker.annotationChip(ann, att, { auto: false, image }));
    } else if (a.type !== "image") {
      // Not saved yet (a selection being dragged): the text and page are all there is.
      const label = a.pageLabel || "?";
      chips.push({ id: `annotation:${a.id ?? chips.length}`, kind: "annotation", label: `${a.type ?? "text"} · p.${label}`, auto: false, pinned: true, ref: { ...refOf(att), pageLabel: label }, text: a.text || a.comment || "" });
    }
  }
  return chips;
}
