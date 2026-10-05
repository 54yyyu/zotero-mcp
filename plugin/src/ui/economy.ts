// Context economy (DESIGN.md "Context budget"). A chat is one persistent agent session, so what a message's
// <zotero-context> carried once is still in the agent's conversation: a chip goes in full the first time and
// whenever it changes; unchanged, it becomes one short reminder line, and its image is never sent again.
import type { ContextChip, PromptInput, Usage } from "../types.ts";
import type { TranscriptState } from "./transcript.ts";

/** FNV-1a, 32 bit: a fingerprint, not security. */
function fnv(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193);
  return (h >>> 0).toString(36);
}

/** What the agent would read for a chip: same id and same fingerprint means there is nothing new to tell it. */
export function chipHash(c: ContextChip): string {
  const r = c.ref;
  return fnv([c.kind, r.libraryID, r.itemKey, r.attachmentKey, r.pageIndex, r.pageLabel, r.annotationKey, r.collectionKey, c.text ?? "", c.image?.data ?? ""].join("\u0001"));
}

/**
 * Mark the chips this chat already sent unchanged (`repeat`), and record the rest in `sent` (id -> fingerprint).
 * `sent` belongs to one agent session: a new or resumed session, or a compaction, starts it empty.
 */
export function planContext(chips: ContextChip[], sent: Map<string, string>): ContextChip[] {
  // A selected annotation whose text is the live selection says it once, with its key.
  const norm = (t = "") => t.replace(/\s+/g, " ").trim();
  const annotated = new Set(chips.filter((c) => c.kind === "annotation" && c.text).map((c) => norm(c.text)));
  return chips.filter((c) => !(c.kind === "selection" && annotated.has(norm(c.text)))).map((c) => {
    const h = chipHash(c);
    if (sent.get(c.id) === h) return { ...c, repeat: true };
    sent.set(c.id, h);
    return c;
  });
}

const pageOf = (c: ContextChip) => (c.ref.pageLabel ? ` p.${c.ref.pageLabel}` : c.ref.pageIndex != null ? ` p.${c.ref.pageIndex + 1}` : "");
const opening = (s = "") => {
  const words = s.replace(/\s+/g, " ").trim().split(" ");
  return `"${words.slice(0, 8).join(" ")}${words.length > 8 ? "…" : ""}"`;
};

const FOCUS = new Set<ContextChip["kind"]>(["selection", "area", "annotation"]);

/** Short lines naming the chips sent before, so "this selection" still resolves without resending it. */
export function repeatLine(chips: ContextChip[]): string {
  const name = (c: ContextChip) => {
    switch (c.kind) {
      case "reader": return `reading ${c.label}${pageOf(c)}`;
      case "selection": return `selected text${pageOf(c)} ${opening(c.text)}`;
      case "area": return c.ref.annotationKey ? `selected area${pageOf(c)} (annotation ${c.ref.annotationKey})` : `the whole page${pageOf(c)}, as an image`;
      case "annotation": return `annotation ${c.ref.annotationKey}${pageOf(c)} ${opening(c.text)}`;
      case "collection": return `collection ${c.ref.collectionKey} (${c.label})`;
      default: return `item ${c.ref.itemKey} (${c.label})`;
    }
  };
  const rep = chips.filter((c) => c.repeat);
  const open = rep.filter((c) => !FOCUS.has(c.kind)).map(name);
  const focus = rep.filter((c) => FOCUS.has(c.kind)).map(name);
  return [
    ...(open.length ? [`Still open, unchanged: ${open.join("; ")}.`] : []),
    ...(focus.length ? [`Still pointing at, unchanged since you saw it earlier in this chat: ${focus.join("; ")}.`] : []),
  ].join("\n");
}

/** Tokens an image costs a Claude-class model: width x height / 750 after fitting the long edge to 1568 px. */
export function imageTokens(data: string): number {
  try {
    const b = atob(data.slice(0, 32)); // the PNG signature and the IHDR chunk: width and height at bytes 16..23
    const u32 = (o: number) => ((b.charCodeAt(o) << 24) | (b.charCodeAt(o + 1) << 16) | (b.charCodeAt(o + 2) << 8) | b.charCodeAt(o + 3)) >>> 0;
    if (b.slice(12, 16) !== "IHDR") return 1600;
    const w = u32(16), h = u32(20), s = Math.min(1, 1568 / Math.max(w, h, 1));
    return Math.ceil((w * s * (h * s)) / 750);
  } catch {
    return 1600;
  }
}

/** A rough token count of what one prompt sends: text at 4 characters a token, plus its images. */
export function estimateTokens(p: PromptInput): number {
  return Math.ceil(p.text.length / 4) + (p.images ?? []).reduce((n, i) => n + imageTokens(i.data), 0);
}

export interface Fill { used: number; size: number; pct: number }

/**
 * How full the agent's context window is: the last turn that said, or a compaction the user asked for since (it
 * carries the new fill). Null when the backend never said.
 */
export function contextFill(tr: TranscriptState): Fill | null {
  for (let i = tr.messages.length - 1; i >= 0; i--) {
    const m = tr.messages[i]!;
    const c = m.role === "notice" ? m.context
      : m.role === "assistant" && m.usage?.contextSize && m.usage.contextUsed != null ? { used: m.usage.contextUsed, size: m.usage.contextSize } : undefined;
    if (c) return { ...c, pct: Math.min(100, Math.round((c.used / c.size) * 100)) };
  }
  return null;
}

/** What the context popover says about this chat, all read from the transcript (so a reopened chat says the same). */
export function chatStats(tr: TranscriptState): { messages: number; compactions: number; last?: Usage } {
  let messages = 0, compactions = 0, last: Usage | undefined;
  for (const m of tr.messages) {
    if (m.role === "notice") { if (m.compacted) compactions++; continue; }
    messages++;
    if (m.role === "assistant" && (m.usage?.inputTokens || m.usage?.outputTokens)) last = m.usage;
  }
  return { messages, compactions, ...(last ? { last } : {}) };
}

/** What the last message's <zotero-context> carried: "Reader p.8 and selection p.3 in full; item Bell 2017 named only". */
export function sentLine(chips: ContextChip[]): string {
  const name = (c: ContextChip) => c.kind === "reader" ? `reader${pageOf(c)}` : c.kind === "selection" ? `selection${pageOf(c)}`
    : c.kind === "area" ? `area${pageOf(c)}` : c.kind === "annotation" ? `annotation${pageOf(c)}` : `${c.kind} ${c.label.length > 28 ? `${c.label.slice(0, 27)}…` : c.label}`;
  const list = (cs: ContextChip[]) => cs.map(name).join(", ").replace(/, ([^,]*)$/, " and $1");
  const full = list(chips.filter((c) => !c.repeat)), rep = list(chips.filter((c) => c.repeat));
  const s = [full && `${full} in full`, rep && `${rep} named only (unchanged)`].filter(Boolean).join("; ");
  return s && s[0]!.toUpperCase() + s.slice(1);
}
