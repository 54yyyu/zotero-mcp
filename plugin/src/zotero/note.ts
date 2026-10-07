// "Save as note" (PanelHost.saveNote): an answer or a diagram as a Zotero note. A diagram's PNG becomes an embedded
// image the way Zotero's own note editor stores one: an attachment of the note (Zotero.Attachments.importEmbeddedImage,
// link mode "embedded image") referenced as <img data-attachment-key="KEY">, which the editor resolves when it opens.
import type { NoteRequest, SavedNote } from "../types.ts";
import { noteHtml } from "../ui/note-html.ts";
import type { ContextTracker } from "./context.ts";
import { uriFor } from "./open.ts";

/** Under the item the user is on (the open reader's parent, or the selected item, or a selected child's parent); else standalone, in the selected collection. */
function target(win: any, context: ContextTracker): { libraryID: number; parent: any | null; collection: any | null } {
  const reader = context.activeReader(win);
  const picked = reader ? reader._item : win.ZoteroPane?.getSelectedItems?.()[0];
  const parent = picked ? (picked.isRegularItem() ? picked : picked.parentItem ?? null) : null;
  if (parent) return { libraryID: parent.libraryID, parent, collection: null };
  const libraryID: number = picked?.libraryID ?? win.ZoteroPane?.getSelectedLibraryID?.() ?? Zotero.Libraries.userLibraryID;
  return { libraryID, parent: null, collection: reader ? null : win.ZoteroPane?.getSelectedCollections?.()?.[0] ?? null };
}

export async function saveNote(win: any, context: ContextTracker, req: NoteRequest): Promise<SavedNote> {
  const t = target(win, context);
  if (!Zotero.Libraries.get(t.libraryID)?.editable) throw new Error("this library is read-only");
  const note = new Zotero.Item("note");
  note.libraryID = t.libraryID;
  if (t.parent) note.parentID = t.parent.id;
  else if (t.collection) note.addToCollection(t.collection.id);
  try {
    const keys: ({ key: string; width: number; height: number } | null)[] = [];
    if (req.images?.some(Boolean)) {
      await note.saveTx(); // an embedded image belongs to its note, so the note must exist first
      for (const img of req.images) {
        if (!img) { keys.push(null); continue; }
        const att = await Zotero.Attachments.importEmbeddedImage({ blob: new win.Blob([img.data], { type: "image/png" }), parentItemID: note.id });
        keys.push({ key: att.key, width: img.width, height: img.height });
      }
    }
    note.setNote(noteHtml(req.markdown, { ...(req.title ? { title: req.title } : {}), image: (i) => keys[i] ?? null }));
    await note.saveTx();
  } catch (e) {
    if (note.id) await note.eraseTx().catch(() => {}); // never leave an empty note behind
    throw e;
  }
  return { noteKey: note.key, ...(t.parent ? { itemKey: t.parent.key } : {}), uri: uriFor({ libraryID: t.libraryID, itemKey: note.key }) };
}
