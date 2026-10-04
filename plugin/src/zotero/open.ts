// Jumping to things in Zotero: the zotero:// links the agent writes, and refs from chips.
import type { ZoteroRef } from "../types.ts";

/** zotero://open-pdf/library/items/KEY?page=8, zotero://select/groups/123/items/KEY, .../collections/KEY. */
export function parseZoteroURI(uri: string): (ZoteroRef & { page?: string }) | null {
  const m = /^zotero:\/\/(open-pdf|select)\/(library|groups\/(\d+))\/(items|collections)\/([A-Z0-9]{8})(?:\?(.*))?$/.exec(uri.trim());
  if (!m) return null;
  const [, verb, , groupID, kind, key, query] = m;
  const libraryID = groupID ? Zotero.Groups.getLibraryIDFromGroupID(Number(groupID)) : Zotero.Libraries.userLibraryID;
  if (!libraryID) return null;
  const params = new URLSearchParams(query ?? "");
  const page = params.get("page") ?? undefined;
  const annotationKey = params.get("annotation") ?? undefined;
  if (kind === "collections") return { libraryID, collectionKey: key! };
  const ref: ZoteroRef & { page?: string } = { libraryID, ...(verb === "open-pdf" ? { attachmentKey: key! } : { itemKey: key! }) };
  if (page) ref.page = page;
  if (annotationKey) ref.annotationKey = annotationKey;
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
    const location: Record<string, unknown> = {};
    if (ref.annotationKey) location.annotationID = ref.annotationKey;
    else if (ref.page) location.pageLabel = ref.page;
    await Zotero.Reader.open(att.id, Object.keys(location).length ? location : undefined);
    return;
  }
  const item = Zotero.Items.getByLibraryAndKey(ref.libraryID, ref.itemKey!);
  if (!item) throw new Error("That item no longer exists");
  win.Zotero_Tabs.select("zotero-pane");
  await win.ZoteroPane.selectItem(item.id);
}
