// What the user is looking at, as chips. The panel follows focus: the open reader (page, text selection),
// otherwise the items selected in the library.
import type { ContextChip, ZoteroRef } from "../types.ts";

const OBSERVER_ID = "zotero-chat";

type Counts = Record<string, number>;
/** Zotero's annotation types, as the user would name them. */
const ANN_KIND: Record<string, string> = { highlight: "highlight", underline: "highlight", note: "note", text: "note", image: "area", ink: "drawing" };

/** Zotero wraps localized names in invisible bidi isolates ("⁨Bertrand⁩ and ⁨Mullainathan⁩"); they must not reach chips or prompts. */
const stripBidi = (s: string) => s.replace(/[\u200e\u200f\u2066-\u2069]/g, "");

export function itemLabel(item: any): string {
  const who = stripBidi(item.firstCreator || "");
  const year = (item.getField("date", true, true) || "").slice(0, 4);
  if (who && year) return `${who} ${year}`;
  const title = stripBidi(item.getDisplayTitle() || "Untitled");
  return title.length > 40 ? title.slice(0, 39) + "…" : title;
}

function pageRef(reader: any): Partial<ZoteroRef> {
  const st = viewStats(reader);
  return st ? { pageIndex: st.pageIndex, pageLabel: st.pageLabel } : {};
}

export function refOf(item: any, extra: Partial<ZoteroRef> = {}): ZoteroRef {
  const attachment = item.isAttachment() ? item : null;
  const parent = attachment?.parentItem ?? (attachment ? null : item);
  return {
    libraryID: item.libraryID,
    itemKey: (parent ?? item).key,
    ...(attachment ? { attachmentKey: attachment.key } : {}),
    ...extra,
  };
}

export class ContextTracker {
  private listeners = new Set<() => void>();
  private selections = new Map<number, { text: string; pageIndex: number | null; pageLabel: string | null; at: number }>();
  private notifierID: string | null = null;
  private disposers: (() => void)[] = [];
  private images = new Map<string, { mime: "image/png"; data: string }>();
  private rendering = new Set<string>();
  private lastSig = "";
  /** Per attachment id: its annotations counted by kind, in all and per page. Cleared by the item notifier, so a turn costs a lookup. */
  private annIndex = new Map<number, { n: number; all: Counts; pages: Map<number, Counts> }>();
  private onPopup = (event: any) => { this.takeSelection(event); };

  /** A renderTextSelectionPopup event's selection, kept for its reader; returns its chip (null when it has no text). */
  takeSelection(event: any): ContextChip | null {
    const a = event.params?.annotation;
    const att = event.reader?._item;
    if (!a?.text || !att) return null;
    this.selections.set(event.reader.itemID, { text: a.text, pageIndex: a.position?.pageIndex ?? null, pageLabel: a.pageLabel ?? null, at: Date.now() });
    this.changed();
    return this.selectionChip(att);
  }

  private pluginID: string;

  constructor(pluginID: string) {
    this.pluginID = pluginID;
  }

  start(win: any): void {
    Zotero.Reader.registerEventListener("renderTextSelectionPopup", this.onPopup, this.pluginID);
    // An item event may be an annotation added, edited or deleted: the counts are rebuilt on the next ask.
    this.notifierID = Zotero.Notifier.registerObserver({ notify: (_e: string, type: string) => { if (type === "item") this.annIndex.clear(); this.changed(); } }, ["tab", "item"], OBSERVER_ID);
    for (const id of ["zotero-items-tree", "zotero-collections-tree"]) {
      const el = win.document.getElementById(id);
      if (!el) continue;
      const cb = () => this.changed();
      el.addEventListener("select", cb);
      this.disposers.push(() => el.removeEventListener("select", cb));
    }
    // The reader emits no event for page turns or for a selection being cleared; its view stats change quietly.
    const timer = win.setInterval(() => {
      if (!this.listeners.size) return;
      const sig = this.signature(win);
      if (sig !== this.lastSig) { this.lastSig = sig; this.changed(); }
    }, 400);
    this.disposers.push(() => win.clearInterval(timer));
  }

  private signature(win: any): string {
    try {
      const reader = this.activeReader(win);
      if (!reader) return "library";
      const st = viewStats(reader);
      const picked = (reader._internalReader?._state?.selectedAnnotationIDs ?? []).join(",");
      return [reader.itemID, st?.pageIndex, st?.canCopy, picked].join("|");
    } catch {
      return "";
    }
  }

  stop(): void {
    Zotero.Reader.unregisterEventListener("renderTextSelectionPopup", this.onPopup);
    if (this.notifierID) Zotero.Notifier.unregisterObserver(this.notifierID);
    for (const d of this.disposers.splice(0)) d();
    this.listeners.clear();
  }

  on(cb: () => void): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }

  /** Something outside the reader changed what counts as context (a setting): tell listeners to ask again. */
  refresh(): void {
    this.changed();
  }

  private changed(): void {
    for (const cb of [...this.listeners]) cb();
  }

  /** The reader in front, if the front tab is one. */
  activeReader(win: any = Zotero.getMainWindow()): any | null {
    const tabs = win?.Zotero_Tabs;
    if (!tabs || tabs.selectedType !== "reader") return null;
    return Zotero.Reader.getByTabID(tabs.selectedID) ?? null;
  }

  current(win: any = Zotero.getMainWindow()): ContextChip[] {
    if (!win) return [];
    try {
      const reader = this.activeReader(win);
      if (reader?._item) return this.readerChips(reader);
      return this.libraryChips(win);
    } catch {
      // Zotero's panes are not ready yet (startup) or are mid-change; the next select event asks again.
      return [];
    }
  }

  private readerChips(reader: any): ContextChip[] {
    const att = reader._item;
    const parent = att.parentItem ?? att;
    const ref = { ...refOf(att), ...pageRef(reader) };
    const index = this.indexLine(att, ref.pageIndex, parent.key);
    const chips: ContextChip[] = [{
      id: `reader:${att.key}`, kind: "reader", label: itemLabel(parent), auto: true, pinned: false, ref, ...(index ? { text: index } : {}),
    }];
    const sel = viewStats(reader)?.canCopy ? this.selectionChip(att) : null;
    if (sel) chips.push(sel);
    chips.push(...this.annotationChips(reader, att));
    return chips;
  }

  /** The last text selected in this attachment's reader, as a chip. */
  private selectionChip(att: any): ContextChip | null {
    const sel = this.selections.get(att.id);
    if (!sel) return null;
    return {
      id: `selection:${att.key}`, kind: "selection", label: "Text Selection", auto: true, pinned: false,
      ref: { ...refOf(att), ...(sel.pageIndex != null ? { pageIndex: sel.pageIndex } : {}), ...(sel.pageLabel ? { pageLabel: sel.pageLabel } : {}) },
      text: sel.text,
    };
  }

  /**
   * One line about the annotations already in the PDF, never their text: "In this PDF: 14 annotations (9 highlights,
   * 3 notes, 2 areas); on this page: 2 highlights. Read them with ...". Empty when there are none.
   */
  private indexLine(att: any, pageIndex: number | undefined, itemKey: string): string {
    let ix = this.annIndex.get(att.id);
    if (!ix) {
      ix = { n: 0, all: {}, pages: new Map() };
      for (const ann of att.getAnnotations?.() ?? []) {
        const kind = ANN_KIND[ann.annotationType] ?? "annotation";
        let page: number | undefined;
        try { page = JSON.parse(ann.annotationPosition || "{}").pageIndex; } catch { /* no position */ }
        ix.n++;
        ix.all[kind] = (ix.all[kind] ?? 0) + 1;
        if (Number.isInteger(page)) { const c = ix.pages.get(page!) ?? {}; c[kind] = (c[kind] ?? 0) + 1; ix.pages.set(page!, c); }
      }
      this.annIndex.set(att.id, ix);
    }
    if (!ix.n) return "";
    const fmt = (c: Counts) => Object.entries(c).map(([k, n]) => `${n} ${k}${n > 1 ? "s" : ""}`).join(", ");
    const here = pageIndex != null ? ix.pages.get(pageIndex) : undefined;
    return `In this PDF: ${ix.n} annotation${ix.n > 1 ? "s" : ""} (${fmt(ix.all)})${here ? `; on this page: ${fmt(here)}` : ""}. Read them with \`zotero-cli annotations list --item-key ${itemKey}\` if useful.`;
  }

  /** Annotations selected in the reader's sidebar or canvas: a selected area becomes an image chip, the rest text. */
  private annotationChips(reader: any, att: any): ContextChip[] {
    // Copied into a local array first: this one comes from the reader's own window (another JS compartment), and an
    // array method called on it hands our callback's results back wrapped, not flattened.
    const ids: string[] = Array.from(reader._internalReader?._state?.selectedAnnotationIDs ?? []);
    return ids.slice(0, 3).flatMap((id) => {
      const ann = Zotero.Items.getByLibraryAndKey(att.libraryID, id);
      return ann?.isAnnotation?.() ? [this.annotationChip(ann, att, { auto: true, image: this.imageFor(ann, att) })] : [];
    });
  }

  /** One chip for an annotation; an image annotation is an "area". `auto` says whether the panel or the user added it. */
  annotationChip(ann: any, att: any, o: { auto: boolean; image?: { mime: "image/png"; data: string } | null }): ContextChip {
    const label = ann.annotationPageLabel || "?";
    const ref: ZoteroRef = { ...refOf(att), annotationKey: ann.key, pageLabel: label, pageIndex: Number(JSON.parse(ann.annotationPosition || "{}").pageIndex ?? 0) };
    const own = { auto: o.auto, pinned: !o.auto, ref };
    if (ann.annotationType === "image") {
      return { id: `area:${ann.key}`, kind: "area", label: `Selected Area · p.${label}`, ...own, ...(o.image ? { image: o.image } : {}) };
    }
    return { id: `annotation:${ann.key}`, kind: "annotation", label: `${ann.annotationType} · p.${label}`, ...own, text: ann.annotationText || ann.annotationComment || "" };
  }

  /** The area's PNG, rendered by Zotero if it has not been yet. Cached per annotation; null when it cannot be had. */
  async loadImage(ann: any, att: any): Promise<{ mime: "image/png"; data: string } | null> {
    const hit = this.images.get(ann.key);
    if (hit) return hit;
    try {
      const path = Zotero.Annotations.getCacheImagePath(ann);
      if (!(await IOUtils.exists(path))) await Zotero.PDFWorker.renderAttachmentAnnotations(att.id);
      const image = { mime: "image/png" as const, data: toBase64(await IOUtils.read(path)) };
      this.images.set(ann.key, image);
      return image;
    } catch (e) {
      Zotero.debug(`zotero-chat: area image failed: ${e}`);
      return null;
    }
  }

  /** The cached PNG now; the first call also starts the render and announces when it is ready. */
  private imageFor(ann: any, att: any): { mime: "image/png"; data: string } | null {
    const hit = this.images.get(ann.key);
    if (hit) return hit;
    if (!this.rendering.has(ann.key)) {
      this.rendering.add(ann.key);
      void this.loadImage(ann, att).then((img) => { if (img) this.changed(); });
    }
    return null;
  }

  private libraryChips(win: any): ContextChip[] {
    const items: any[] = win.ZoteroPane.getSelectedItems?.() ?? [];
    if (items.length) {
      return items.slice(0, 5).map((it) => ({
        id: `item:${it.key}`, kind: "item" as const, label: itemLabel(it.isAttachment() ? it.parentItem ?? it : it), auto: true, pinned: false, ref: refOf(it),
      }));
    }
    const col = win.ZoteroPane.getSelectedCollections?.()?.[0];
    if (col) return [{ id: `collection:${col.key}`, kind: "collection", label: col.name, auto: true, pinned: false, ref: { libraryID: col.libraryID, collectionKey: col.key } }];
    return [];
  }

}

/** The reader's live view stats: current page, whether text is selected. Null while the reader is still starting. */
export function viewStats(reader: any): { pageIndex: number; pageLabel: string; pagesCount: number; canCopy: boolean } | null {
  const st = reader?._internalReader?._state?.primaryViewStats;
  if (!st || !Number.isInteger(st.pageIndex)) return null;
  return { pageIndex: st.pageIndex, pageLabel: String(st.pageLabel ?? st.pageIndex + 1), pagesCount: st.pagesCount, canCopy: !!st.canCopy };
}

/** Current page of an open reader (0-based); null when it cannot be told. */
export function readerPageIndex(reader: any): number | null {
  return viewStats(reader)?.pageIndex ?? null;
}

function toBase64(bytes: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}
