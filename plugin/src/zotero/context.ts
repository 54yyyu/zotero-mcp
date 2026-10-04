// What the user is looking at, as chips. The panel follows focus: the open reader (page, text selection),
// otherwise the items selected in the library.
import type { ContextChip, ZoteroRef } from "../types.ts";

const OBSERVER_ID = "zotero-chat";

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
  private onPopup = (event: any) => {
    const a = event.params?.annotation;
    if (!a?.text) return;
    this.selections.set(event.reader.itemID, { text: a.text, pageIndex: a.position?.pageIndex ?? null, pageLabel: a.pageLabel ?? null, at: Date.now() });
    this.changed();
  };

  private pluginID: string;

  constructor(pluginID: string) {
    this.pluginID = pluginID;
  }

  start(win: any): void {
    Zotero.Reader.registerEventListener("renderTextSelectionPopup", this.onPopup, this.pluginID);
    this.notifierID = Zotero.Notifier.registerObserver({ notify: () => this.changed() }, ["tab"], OBSERVER_ID);
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
    const chips: ContextChip[] = [{
      id: `reader:${att.key}`, kind: "reader", label: itemLabel(parent), auto: true, pinned: false,
      ref: { ...refOf(att), ...pageRef(reader) },
    }];
    const sel = this.selections.get(att.id);
    if (sel && viewStats(reader)?.canCopy) {
      chips.push({
        id: `selection:${att.key}`, kind: "selection", label: "Text Selection", auto: true, pinned: false,
        ref: { ...refOf(att), ...(sel.pageIndex != null ? { pageIndex: sel.pageIndex } : {}), ...(sel.pageLabel ? { pageLabel: sel.pageLabel } : {}) },
        text: sel.text,
      });
    }
    chips.push(...this.annotationChips(reader, att));
    return chips;
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
