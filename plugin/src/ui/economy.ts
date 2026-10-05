// Context economy (DESIGN.md "Context budget"). A chat is one persistent agent session, so what a message's
// <zotero-context> carried once is still in the agent's conversation: a chip goes in full the first time and
// whenever it changes; unchanged, it becomes one short reminder line, and its image is never sent again.
import type { ContextChip, PromptInput } from "../types.ts";
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
  return chips.map((c) => {
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

/** One line naming the chips sent before, so "this selection" still resolves without resending it. */
export function repeatLine(chips: ContextChip[]): string {
  const parts = chips.filter((c) => c.repeat).map((c) => {
    switch (c.kind) {
      case "reader": return `reading ${c.label}${pageOf(c)}`;
      case "selection": return `selected text${pageOf(c)} ${opening(c.text)}`;
      case "area": return `selected area${pageOf(c)} (annotation ${c.ref.annotationKey})`;
      case "annotation": return `annotation ${c.ref.annotationKey}${pageOf(c)} ${opening(c.text)}`;
      case "collection": return `collection ${c.ref.collectionKey} (${c.label})`;
      default: return `item ${c.ref.itemKey} (${c.label})`;
    }
  });
  return parts.length ? `Still in focus, unchanged since you saw it earlier in this chat: ${parts.join("; ")}.` : "";
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

/** How full the agent's context window was at the end of the last turn that said; null when the backend never did. */
export function contextFill(tr: TranscriptState): { used: number; size: number; pct: number } | null {
  for (let i = tr.messages.length - 1; i >= 0; i--) {
    const m = tr.messages[i]!;
    if (m.role !== "assistant" || !m.usage?.contextSize || m.usage.contextUsed == null) continue;
    const { contextUsed: used, contextSize: size } = m.usage;
    return { used, size, pct: Math.min(100, Math.round((used / size) * 100)) };
  }
  return null;
}
