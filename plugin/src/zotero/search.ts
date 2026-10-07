// `@` mentions: items, collections, and the open reader's annotations that match what was typed.
import type { ContextChip, ItemHit } from "../types.ts";
import { itemLabel, refOf } from "./context.ts";

const MAX_ITEMS = 8;

export async function search(win: any, query: string, activeAttachment: any | null): Promise<ItemHit[]> {
  const q = query.trim();
  const hits: ItemHit[] = [];
  const libs = [...new Set([win.ZoteroPane.getSelectedLibraryID?.() ?? Zotero.Libraries.userLibraryID, Zotero.Libraries.userLibraryID])];

  for (const libraryID of libs) {
    if (hits.length >= MAX_ITEMS) break;
    const s = new Zotero.Search();
    s.libraryID = libraryID;
    if (q) s.addCondition("quicksearch-titleCreatorYear", "contains", q);
    for (const t of ["attachment", "note", "annotation"]) s.addCondition("itemType", "isNot", t);
    const ids: number[] = (await s.search()).slice(0, MAX_ITEMS - hits.length);
    const items = await Zotero.Items.getAsync(ids);
    await Zotero.Items.loadDataTypes(items, ["primaryData", "creators", "itemData"]);
    for (const it of items) hits.push({ kind: "item", ref: refOf(it), title: itemLabel(it), subtitle: it.getDisplayTitle() });
  }

  if (q) {
    const lower = q.toLowerCase();
    for (const libraryID of libs) {
      for (const col of Zotero.Collections.getByLibrary(libraryID, true)) {
        if (hits.filter((h) => h.kind === "collection").length >= 3) break;
        if (col.name.toLowerCase().includes(lower)) hits.push({ kind: "collection", ref: { libraryID, collectionKey: col.key }, title: col.name, subtitle: "Collection" });
      }
    }
    if (activeAttachment) {
      let n = 0;
      for (const ann of activeAttachment.getAnnotations?.() ?? []) {
        const text = `${ann.annotationText ?? ""} ${ann.annotationComment ?? ""}`;
        if (!text.toLowerCase().includes(lower) || n++ >= 3) continue;
        hits.push({ kind: "annotation", ref: { ...refOf(activeAttachment), annotationKey: ann.key, pageLabel: ann.annotationPageLabel }, title: `${ann.annotationType} · p.${ann.annotationPageLabel}`, subtitle: text.trim().slice(0, 80) });
      }
    }
  }
  return hits;
}

/** A hit the user picked becomes a chip that stays until removed. */
export function chipForHit(hit: ItemHit, annotationText?: string): ContextChip {
  const kind = hit.kind === "collection" ? "collection" : hit.kind === "annotation" ? "annotation" : "item";
  return { id: `${kind}:${hit.ref.annotationKey ?? hit.ref.collectionKey ?? hit.ref.itemKey}`, kind, label: hit.title, auto: false, pinned: true, ref: hit.ref, ...(annotationText ? { text: annotationText } : {}) };
}
