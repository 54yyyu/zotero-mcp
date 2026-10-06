// What the agent is doing right now, for the working line under a running answer: a state (which dot pattern) and a
// label a person reads. PURE: `thinkOf` maps the turn's blocks; `Pacer` keeps the label from flickering.
import type { Block } from "./transcript.ts";
import { stepTitle } from "./steptitle.ts";

export type ThinkState = "thinking" | "searching" | "reading" | "writing" | "working" | "waiting";
export interface Think { state: ThinkState; label: string }

const THINKING: Think = { state: "thinking", label: "Thinking" };
const LIBRARY: Think = { state: "searching", label: "Searching your library" };
const WEB: Think = { state: "searching", label: "Searching the web" };
const WORKING: Think = { state: "working", label: "Working" };

/** The file a path names ("~/x/SKILL.md" -> "SKILL.md"), or null when `s` is not a path or a file name. */
function fileName(s: string): string | null {
  const t = s.trim().replace(/^["']|["']$/g, "");
  if (!t.includes("/") && !/^\S+\.\w{1,8}$/.test(t)) return null;
  const base = t.split("/").filter(Boolean).pop() ?? "";
  return base && base.length <= 48 ? base : null;
}

/** zotero-cli calls that only read, by stepTitle's summary. */
const CLI_READS: [RegExp, string][] = [
  [/^Get full text of /, "Reading the full text"],
  [/^Get metadata of /, "Reading the metadata"],
  [/^Get the outline of /, "Reading the outline"],
  [/^List annotations of /, "Reading the annotations"],
  [/^List notes of /, "Reading the notes"],
  [/^Find figures and tables in /, "Finding figures and tables"],
];

/** A running tool step as an activity. The summary is stepTitle's ("Read pages 7–8 of KEY"), never the raw command. */
function toolThink(b: Extract<Block, { type: "tool" }>): Think {
  const s = stepTitle(b.title);
  // the summary, or a search piped into something (which stepTitle leaves as the command)
  if (/^(Search library|Semantic search) for /.test(s) || /\bzotero-cli\s+(?:--\S+\s+)*(?:search|s)\s/.test(b.title)) return LIBRARY;
  const pages = /^Read (pages? \S+) of /.exec(s);
  if (pages) return { state: "reading", label: `Reading ${pages[1]}` };
  const cli = CLI_READS.find(([re]) => re.test(s));
  if (cli) return { state: "reading", label: cli[1] };
  const name = (b.name ?? "").toLowerCase();
  if (b.kind === "fetch" || name.startsWith("web")) return WEB;
  if (b.kind === "search") return { state: "searching", label: "Searching files" };
  if (b.kind === "think") return THINKING;
  if (b.kind === "read") {
    const file = fileName(s.replace(/^Read\s+/, ""));
    return { state: "reading", label: file ? `Reading ${file}` : "Reading" };
  }
  if (b.kind === "edit") {
    const file = fileName(s.replace(/^(Edit|Write)\s+/, ""));
    return { state: "working", label: file ? `Editing ${file}` : "Editing" };
  }
  return WORKING;
}

/**
 * The turn's activity from its blocks: an open permission card waits for the user; otherwise the newest block decides.
 * Nothing yet or a thought: thinking. Text: the answer is being written. A run of tool steps: its most recent running
 * step, else (all finished, the model deciding what next) thinking.
 */
export function thinkOf(blocks: readonly Block[]): Think {
  if (blocks.some((b) => b.type === "permission" && b.resolved === undefined)) return { state: "waiting", label: "Waiting for your OK" };
  const last = blocks[blocks.length - 1];
  if (!last || last.type === "thought") return THINKING;
  if (last.type === "text") return last.text.trim() ? { state: "writing", label: "Writing the answer" } : THINKING;
  for (let i = blocks.length - 1; i >= 0; i--) {
    const b = blocks[i]!;
    if (b.type !== "tool") { if (b.type === "text") break; continue; }
    if (b.status === "running" || b.status === "pending") return toolThink(b);
  }
  return THINKING;
}

/** At most one label change per `gap` ms. A change inside the gap waits; only the newest waiting one is ever shown. */
export class Pacer {
  private shown: Think | null = null;
  private at = -Infinity;
  private next: Think | null = null;
  private timer: unknown = null;
  private clock: { now(): number; set(fn: () => void, ms: number): unknown; clear(t: unknown): void };
  private paint: (t: Think) => void;
  private gap: number;

  constructor(paint: (t: Think) => void, clock: Pacer["clock"], gap = 400) {
    this.paint = paint; this.clock = clock; this.gap = gap;
  }

  push(t: Think): void {
    const same = (a: Think | null) => !!a && a.state === t.state && a.label === t.label;
    if (same(this.shown)) { this.drop(); return; }
    if (same(this.next)) return;
    const wait = this.at + this.gap - this.clock.now();
    if (wait <= 0) { this.drop(); this.show(t); return; }
    this.next = t;
    if (this.timer === null) this.timer = this.clock.set(() => { this.timer = null; const n = this.next; this.next = null; if (n) this.show(n); }, wait);
  }

  /** Show `t` now, whatever is waiting (the turn just started: "Starting Claude Code" must not linger). */
  jump(t: Think): void { this.drop(); this.show(t); }

  /** Cancel what is waiting (the line is going away). */
  drop(): void {
    this.next = null;
    if (this.timer !== null) { this.clock.clear(this.timer); this.timer = null; }
  }

  private show(t: Think): void { this.shown = t; this.at = this.clock.now(); this.paint(t); }
}
