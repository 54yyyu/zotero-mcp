// The text the agent reads about the user's focus: a <zotero-context> block (brief.ts tells it what that is) plus images.
import type { ContextChip, ZoteroRef } from "../types.ts";
import { CONTEXT_TAG } from "../agent/index.ts";
import { repeatLine } from "../ui/economy.ts";

const libLabel = (ref: ZoteroRef) => (ref.libraryID === Zotero.Libraries.userLibraryID ? "" : ` · group library ${Zotero.Libraries.get(ref.libraryID).groupID}`);
const page = (ref: ZoteroRef) => (ref.pageLabel ? `p.${ref.pageLabel}` : ref.pageIndex != null ? `p.${ref.pageIndex + 1}` : "");
const quote = (s: string) => `"${s.replace(/\s+/g, " ").trim().slice(0, 1500)}"`;

function itemLine(ref: ZoteroRef): string {
  const item = ref.itemKey ? Zotero.Items.getByLibraryAndKey(ref.libraryID, ref.itemKey) : null;
  if (!item) return "";
  const pdfs = (item.isRegularItem() ? item.getAttachments() : [])
    .map((id: number) => Zotero.Items.get(id))
    .filter((a: any) => a?.attachmentContentType === "application/pdf")
    .map((a: any) => a.key);
  return `"${item.getDisplayTitle()}" · item ${item.key}${pdfs.length ? ` · PDF attachment ${pdfs.join(", ")}` : ""}${libLabel(ref)}`;
}

export function describeContext(chips: ContextChip[]): { text: string; images: { mime: string; data: string }[] } {
  const lines: string[] = [];
  const images: { mime: string; data: string }[] = [];
  const focus: string[] = []; // what the user is pointing at now: the live selection and the selected annotations
  for (const c of chips) {
    if (c.repeat) continue; // sent unchanged before: one reminder line below, no text or image again
    switch (c.kind) {
      case "reader": {
        const att = c.ref.attachmentKey ? Zotero.Items.getByLibraryAndKey(c.ref.libraryID, c.ref.attachmentKey) : null;
        const parent = att?.parentItem;
        const title = (parent ?? att)?.getDisplayTitle() ?? c.label;
        lines.push(`Reading in the Zotero reader: "${title}" · ${parent ? `item ${parent.key} · ` : ""}PDF attachment ${c.ref.attachmentKey}${c.ref.pageIndex != null ? ` · on ${page(c.ref)}` : ""}${libLabel(c.ref)}`);
        if (c.text) lines.push(c.text); // the PDF's annotation counts (context.ts indexLine), never their text
        break;
      }
      case "selection":
        focus.push(`- selected text${page(c.ref) ? ` (${page(c.ref)})` : ""}: ${quote(c.text ?? "")}`);
        break;
      case "area":
        focus.push(`- selected area${page(c.ref) ? ` on ${page(c.ref)}` : ""} (annotation ${c.ref.annotationKey})${c.image ? ": the image is attached" : ""}`);
        if (c.image) images.push(c.image);
        break;
      case "annotation":
        focus.push(`- ${c.label.split(" · ")[0]} annotation ${c.ref.annotationKey}${page(c.ref) ? ` (${page(c.ref)})` : ""}: ${quote(c.text ?? "")}`);
        break;
      case "collection": {
        const col = c.ref.collectionKey ? Zotero.Collections.getByLibraryAndKey(c.ref.libraryID, c.ref.collectionKey) : null;
        lines.push(`Collection "${col?.name ?? c.label}" · collection ${c.ref.collectionKey}${libLabel(c.ref)}`);
        break;
      }
      default: {
        const line = itemLine(c.ref);
        if (line) lines.push(`${c.auto ? "Selected in the library" : "Attached item"}: ${line}`);
      }
    }
  }
  if (focus.length) lines.push("You are pointing at:", ...focus);
  const again = repeatLine(chips);
  if (again) lines.push(again);
  const text = lines.length ? `<${CONTEXT_TAG}>\n${lines.join("\n")}\n</${CONTEXT_TAG}>` : "";
  return { text, images };
}
