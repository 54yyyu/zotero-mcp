// Jumping to things in Zotero: the zotero:// links the agent writes, and refs from chips.
import type { ZoteroRef } from "../types.ts";
import { findQuote, rangeRects } from "./quote.ts";

type Target = ZoteroRef & { page?: string; quote?: string };

/** zotero://open-pdf/library/items/KEY?page=8&quote=..., zotero://select/groups/123/items/KEY, .../collections/KEY. */
export function parseZoteroURI(uri: string): Target | null {
  const m = /^zotero:\/\/(open-pdf|select)\/(library|groups\/(\d+))\/(items|collections)\/([A-Z0-9]{8})(?:\?(.*))?$/.exec(uri.trim());
  if (!m) return null;
  const [, verb, , groupID, kind, key, query] = m;
  const libraryID = groupID ? Zotero.Groups.getLibraryIDFromGroupID(Number(groupID)) : Zotero.Libraries.userLibraryID;
  if (!libraryID) return null;
  const params = new URLSearchParams(query ?? "");
  const page = params.get("page") ?? undefined;
  const annotationKey = params.get("annotation") ?? undefined;
  const quote = params.get("quote")?.trim().slice(0, 500); // only ever text to search for
  if (kind === "collections") return { libraryID, collectionKey: key! };
  const ref: Target = { libraryID, ...(verb === "open-pdf" ? { attachmentKey: key! } : { itemKey: key! }) };
  if (page) ref.page = page;
  if (annotationKey) ref.annotationKey = annotationKey;
  else if (quote && verb === "open-pdf") ref.quote = quote;
  return ref;
}

export function uriFor(ref: ZoteroRef): string {
  const lib = ref.libraryID === Zotero.Libraries.userLibraryID ? "library" : `groups/${Zotero.Libraries.get(ref.libraryID).groupID}`;
  if (ref.collectionKey) return `zotero://select/${lib}/collections/${ref.collectionKey}`;
  if (ref.attachmentKey) {
    const q = new URLSearchParams();
    const page = ref.pageLabel ?? (ref.pageIndex != null ? String(ref.pageIndex + 1) : "");
    if (page) q.set("page", page);
    if (ref.annotationKey) q.set("annotation", ref.annotationKey);
    return `zotero://open-pdf/${lib}/items/${ref.attachmentKey}${q.size ? "?" + q : ""}`;
  }
  return `zotero://select/${lib}/items/${ref.itemKey}`;
}

export async function openTarget(win: any, target: string | ZoteroRef): Promise<void> {
  // The panel never navigates itself: every link, web ones included, arrives here.
  if (typeof target === "string" && /^https?:\/\//i.test(target)) {
    Zotero.launchURL(target);
    return;
  }
  const uri = typeof target === "string" ? target : uriFor(target);
  const ref = parseZoteroURI(uri);
  if (!ref) throw new Error(`Not a Zotero link: ${uri}`);

  if (ref.collectionKey) {
    const col = Zotero.Collections.getByLibraryAndKey(ref.libraryID, ref.collectionKey);
    if (!col) throw new Error("That collection no longer exists");
    win.Zotero_Tabs.select("zotero-pane");
    await win.ZoteroPane.collectionsView.selectCollection(col.id);
    return;
  }
  if (ref.attachmentKey) {
    const att = Zotero.Items.getByLibraryAndKey(ref.libraryID, ref.attachmentKey);
    if (!att) throw new Error("That attachment no longer exists");
    const location: Record<string, unknown> | undefined = ref.annotationKey ? { annotationID: ref.annotationKey } : ref.page ? { pageLabel: ref.page } : undefined;
    if (!ref.quote) { await Zotero.Reader.open(att.id, location); return; }
    // A quote: an open reader just focuses (no page jump first), then flashes the words; never fails.
    const loaded = () => Zotero.Reader._readers.find((r: any) => r.itemID === att.id && !r._isTabClosed);
    const wasOpen = !!loaded();
    let reader = await Zotero.Reader.open(att.id, wasOpen ? undefined : location);
    for (let i = 0; !reader && i < 50; i++) { await Zotero.Promise.delay(100); reader = loaded(); } // a restored tab loads itself
    if (!reader) return;
    const found = await flashQuote(reader, ref.page, ref.quote).catch((e) => { Zotero.logError(e); return false; });
    if (!found && wasOpen && location) reader.navigate(location);
    return;
  }
  const item = Zotero.Items.getByLibraryAndKey(ref.libraryID, ref.itemKey!);
  if (!item) throw new Error("That item no longer exists");
  win.Zotero_Tabs.select("zotero-pane");
  await win.ZoteroPane.selectItem(item.id);
}

/** Find `quote` on the cited page (or the one before or after) and let the reader flash it for 2 s: the view's
 *  navigate({position}) scrolls there and draws it in the selection color, no annotation is made. False when
 *  the words are not found or the view has no PDF text (EPUB, snapshot, a scan without OCR). */
async function flashQuote(reader: any, pageLabel: string | undefined, quote: string): Promise<boolean> {
  await reader._initPromise;
  const view = reader._internalReader?._lastView;
  if (!view) return false;
  await view.initializedPromise;
  if (!view._pdfPages || typeof view._ensureBasicPageData !== "function") return false;
  if (pageLabel) await view._pageLabelsPromise;
  const labels: string[] = Array.from(view._pageLabels ?? []);
  let page = pageLabel ? labels.indexOf(pageLabel) : 0;
  if (page < 0) page = Math.max(0, (Number(pageLabel) || 1) - 1);
  const count: number = view._iframeWindow?.PDFViewerApplication?.pdfDocument?.numPages ?? page + 1;
  for (const i of [page, page - 1, page + 1]) {
    if (i < 0 || i >= count) continue;
    await view._ensureBasicPageData(i);
    const chars = Array.from<any>(view._pdfPages[i]?.chars ?? []);
    const hit = findQuote(chars, quote);
    if (!hit) continue;
    await reader.navigate({ position: { pageIndex: i, rects: rangeRects(chars, hit[0], hit[1]) } });
    return true;
  }
  return false;
}
