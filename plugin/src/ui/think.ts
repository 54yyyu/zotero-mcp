// What the agent is doing right now, for the working line under a running answer: a state (which dot pattern) and a
// label a person reads. PURE: `thinkOf` maps the turn's blocks; `Pacer` keeps the label from flickering.
import type { Block } from "./transcript.ts";
import { stepOf } from "./steptitle.ts";
import { t } from "../i18n.ts";
import type { Key, Vars } from "../i18n.ts";

export type ThinkState = "thinking" | "searching" | "reading" | "writing" | "working" | "waiting";
export interface Think { state: ThinkState; label: string }

const think = (state: ThinkState, key: Key, vars?: Vars): Think => ({ state, label: t(key, vars) });
const thinking = () => think("thinking", "work.thinking");

/** The file a path names ("~/x/SKILL.md" -> "SKILL.md"), or null when `s` is not a path or a file name. */
function fileName(s: string): string | null {
  const x = s.trim().replace(/^["']|["']$/g, "");
  if (!x.includes("/") && !/^\S+\.\w{1,8}$/.test(x)) return null;
  const base = x.split("/").filter(Boolean).pop() ?? "";
  return base && base.length <= 48 ? base : null;
}

/** zotero-cli calls that only read, by stepOf's summary. */
const CLI_READS: Partial<Record<Key, Key>> = {
  "step.getFulltext": "work.readFulltext", "step.getMetadata": "work.readMetadata", "step.outline": "work.readOutline",
  "step.listAnnotationsOf": "work.readAnnotations", "step.listNotesOf": "work.readNotes", "step.layout": "work.findFigures",
};

/** A running tool step as an activity: by stepOf's summary of a zotero-cli call ("Read pages 7–8 of KEY"), never the raw command. */
function toolThink(b: Extract<Block, { type: "tool" }>): Think {
  const { cmd, step } = stepOf(b.title);
  const [key, vars] = step ?? [null, {}];
  // the summary, or a search piped into something (which stepOf leaves as the command)
  if (key === "step.search" || key === "step.semantic" || /\bzotero-cli\s+(?:--\S+\s+)*(?:search|s)\s/.test(b.title)) return think("searching", "work.library");
  if (key?.startsWith("step.readPage")) return think("reading", key.startsWith("step.readPages") ? "work.readPages" : "work.readPage", vars);
  const cli = key && CLI_READS[key];
  if (cli) return think("reading", cli);
  const name = (b.name ?? "").toLowerCase();
  if (b.kind === "fetch" || name.startsWith("web")) return think("searching", "work.web");
  if (b.kind === "search") return think("searching", "work.files");
  if (b.kind === "think") return thinking();
  if (b.kind === "read") {
    const file = fileName(cmd.replace(/^Read\s+/, ""));
    return file ? think("reading", "work.readFile", { file }) : think("reading", "work.reading");
  }
  if (b.kind === "edit") {
    const file = fileName(cmd.replace(/^(Edit|Write)\s+/, ""));
    return file ? think("working", "work.editFile", { file }) : think("working", "work.editing");
  }
  return think("working", "work.working");
}

/**
 * The turn's activity from its blocks: an open permission card waits for the user; otherwise the newest block decides.
 * Nothing yet or a thought: thinking. Text: the answer is being written. A run of tool steps: its most recent running
 * step, else (all finished, the model deciding what next) thinking.
 */
export function thinkOf(blocks: readonly Block[]): Think {
  if (blocks.some((b) => b.type === "permission" && b.resolved === undefined)) return think("waiting", "work.waiting");
  const last = blocks[blocks.length - 1];
  if (!last || last.type === "thought") return thinking();
  if (last.type === "text") return last.text.trim() ? think("writing", "work.writing") : thinking();
  for (let i = blocks.length - 1; i >= 0; i--) {
    const b = blocks[i]!;
    if (b.type !== "tool") { if (b.type === "text") break; continue; }
    if (b.status === "running" || b.status === "pending") return toolThink(b);
  }
  return thinking();
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
