// "Save as note", the UI half: the request for an answer (its question as the title, a PNG for each diagram) and the
// quiet line that confirms it ("Saved to note · Open") or says why not. The host writes the note (zotero/note.ts).
import type { NoteRequest, SavedNote } from "../types.ts";
import { noteDiagrams } from "./note-html.ts";
import { clip, errMessage, h, icon } from "./dom.ts";

/** An answer's markdown as a note request; diagram.ts is loaded only when the answer has a drawing. */
export async function answerNote(markdown: string, question?: string): Promise<NoteRequest> {
  const srcs = noteDiagrams(markdown);
  const images = srcs.length ? await import("./diagram.ts").then((d) => Promise.all(srcs.map((s) => d.pngOf(s).catch(() => null)))) : null;
  return { markdown, ...(question?.trim() ? { title: clip(question, 100) } : {}), ...(images ? { images } : {}) };
}

export type NoteState = { ok: SavedNote } | { err: string };

/** Runs `save`, and hands back the state for the line. */
export async function trySave(save: () => Promise<SavedNote>): Promise<NoteState> {
  try { return { ok: await save() }; } catch (e) { return { err: errMessage(e) }; }
}

export function noteLine(s: NoteState, open: (uri: string) => void): HTMLElement {
  if ("err" in s) return h("div.noteline.noteline--err", { role: "alert" }, icon("warn"), h("span", null, `Couldn't save the note: ${s.err}`));
  const uri = s.ok.uri;
  return h("div.noteline", { role: "status" }, icon("check"), h("span", null, "Saved to note"),
    h("button.lnk", { type: "button", title: "Show the note in Zotero", onclick: () => open(uri) }, "Open"));
}
