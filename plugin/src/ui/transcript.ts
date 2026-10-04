// The transcript as data. `applyEvent` is a pure fold: (state, event) -> state, never mutating its input.
// Live streaming and replaying a saved list are the same thing, so a resumed chat looks exactly like it
// did. Unchanged messages keep their object identity, which is how the view patches only what moved.
import type { ChatEvent, ChipSummary, PermissionOption, ToolKind, ToolStatus, Usage } from "../types.ts";

export type PlanEntry = { content: string; status: "pending" | "in_progress" | "completed" };

export type Block =
  | { type: "text"; text: string }
  | { type: "thought"; text: string }
  | { type: "tool"; id: string; title: string; kind: ToolKind; status: ToolStatus; input?: unknown; output?: string; name?: string }
  | { type: "plan"; entries: PlanEntry[] }
  | { type: "permission"; id: string; title: string; kind: ToolKind; input?: unknown; options: PermissionOption[]; resolved?: string; name?: string };

export type Stop = "end_turn" | "cancelled" | "max_tokens" | "refusal" | "error";

export interface UserMessage { role: "user"; id: string; text: string; chips: ChipSummary[] }
export interface AssistantMessage { role: "assistant"; id: string; blocks: Block[]; done: boolean; stop?: Stop; usage?: Usage }
export interface NoticeMessage { role: "notice"; id: string; level: "info" | "warn" | "error"; message: string; hint?: string }
export type Message = UserMessage | AssistantMessage | NoticeMessage;

export interface TranscriptState {
  messages: readonly Message[];
  /** The turn that is streaming now (no `turn_end` yet), if any. */
  running: string | null;
  /** Counts notices, so their ids are the same on replay. */
  notices: number;
}

export const emptyTranscript = (): TranscriptState => ({ messages: [], running: null, notices: 0 });

const replaceAt = <T>(list: readonly T[], i: number, v: T): T[] => {
  const next = list.slice();
  next[i] = v;
  return next;
};

/** Add `block` or update the block with the same `id` (tool, permission) in place. */
function upsert(blocks: Block[], match: (b: Block) => boolean, block: Block): Block[] {
  const i = blocks.findIndex(match);
  return i < 0 ? [...blocks, block] : replaceAt(blocks, i, block);
}

function appendText(blocks: Block[], type: "text" | "thought", delta: string): Block[] {
  const last = blocks[blocks.length - 1];
  if (last && last.type === type) return replaceAt(blocks, blocks.length - 1, { type, text: last.text + delta });
  return [...blocks, { type, text: delta }];
}

/** The assistant message for `turn`, created if the log starts mid-turn (a replay of a partial save). */
function withTurn(state: TranscriptState, turn: string, edit: (m: AssistantMessage) => AssistantMessage): TranscriptState {
  const i = state.messages.findIndex((m) => m.role === "assistant" && m.id === turn);
  if (i < 0) {
    const fresh = edit({ role: "assistant", id: turn, blocks: [], done: false });
    return { ...state, messages: [...state.messages, fresh], running: fresh.done ? state.running : turn };
  }
  const cur = state.messages[i] as AssistantMessage;
  const next = edit(cur);
  return next === cur ? state : { ...state, messages: replaceAt(state.messages, i, next) };
}

export function applyEvent(state: TranscriptState, ev: ChatEvent): TranscriptState {
  switch (ev.t) {
    case "user": {
      const i = state.messages.findIndex((m) => m.role === "user" && m.id === ev.id);
      const msg: UserMessage = { role: "user", id: ev.id, text: ev.text, chips: ev.chips };
      return i < 0 ? { ...state, messages: [...state.messages, msg] } : { ...state, messages: replaceAt(state.messages, i, msg) };
    }
    case "turn_start":
      return { ...withTurn(state, ev.turn, (m) => m), running: ev.turn };
    case "thought":
      return { ...withTurn(state, ev.turn, (m) => ({ ...m, blocks: appendText(m.blocks, "thought", ev.delta) })), running: runningAfter(state, ev.turn) };
    case "text":
      return { ...withTurn(state, ev.turn, (m) => ({ ...m, blocks: appendText(m.blocks, "text", ev.delta) })), running: runningAfter(state, ev.turn) };
    case "tool": {
      const block: Block = { type: "tool", id: ev.id, title: ev.title, kind: ev.kind, status: ev.status, ...(ev.input !== undefined ? { input: ev.input } : {}), ...(ev.output !== undefined ? { output: ev.output } : {}), ...(ev.name ? { name: ev.name } : {}) };
      return { ...withTurn(state, ev.turn, (m) => {
        const prev = m.blocks.find((b) => b.type === "tool" && b.id === ev.id);
        // An update that omits input/output keeps what the first event carried.
        if (prev && prev.type === "tool") {
          if (block.type === "tool") {
            if (block.input === undefined && prev.input !== undefined) block.input = prev.input;
            if (block.output === undefined && prev.output !== undefined) block.output = prev.output;
            if (!block.name && prev.name) block.name = prev.name;
          }
        }
        return { ...m, blocks: upsert(m.blocks, (b) => b.type === "tool" && b.id === ev.id, block) };
      }), running: runningAfter(state, ev.turn) };
    }
    case "plan":
      return { ...withTurn(state, ev.turn, (m) => ({ ...m, blocks: upsert(m.blocks, (b) => b.type === "plan", { type: "plan", entries: ev.entries }) })), running: runningAfter(state, ev.turn) };
    case "permission": {
      const block: Block = { type: "permission", id: ev.id, title: ev.title, kind: ev.kind, options: ev.options, ...(ev.input !== undefined ? { input: ev.input } : {}), ...(ev.resolved !== undefined ? { resolved: ev.resolved } : {}), ...(ev.name ? { name: ev.name } : {}) };
      return { ...withTurn(state, ev.turn, (m) => ({ ...m, blocks: upsert(m.blocks, (b) => b.type === "permission" && b.id === ev.id, block) })), running: runningAfter(state, ev.turn) };
    }
    case "turn_end": {
      const next = withTurn(state, ev.turn, (m) => {
        // A turn that did not finish leaves no tool "running" and no question unanswered on screen.
        const stopped = ev.stop !== "end_turn";
        const blocks = stopped
          ? m.blocks.map((b): Block => (b.type === "tool" && (b.status === "pending" || b.status === "running") ? { ...b, status: "failed" } : b))
          : m.blocks;
        return { ...m, blocks, done: true, stop: ev.stop, ...(ev.usage ? { usage: ev.usage } : {}) };
      });
      return { ...next, running: state.running === ev.turn ? null : state.running };
    }
    case "notice": {
      const n = state.notices + 1;
      const msg: NoticeMessage = { role: "notice", id: `notice-${n}`, level: ev.level, message: ev.message, ...(ev.hint ? { hint: ev.hint } : {}) };
      return { ...state, messages: [...state.messages, msg], notices: n };
    }
  }
}

/** An event for a turn that already ended (a late delta) does not make it "running" again. */
function runningAfter(state: TranscriptState, turn: string): string | null {
  const m = state.messages.find((x) => x.role === "assistant" && x.id === turn) as AssistantMessage | undefined;
  if (m?.done) return state.running;
  return turn;
}

export function replay(events: readonly ChatEvent[], from: TranscriptState = emptyTranscript()): TranscriptState {
  return events.reduce(applyEvent, from);
}

// ───────────────────────────── derived facts ─────────────────────────────

/** The text blocks of an answer, in order. */
export const textsOf = (m: AssistantMessage): string[] => m.blocks.flatMap((b) => (b.type === "text" ? [b.text] : []));

/** The text of an answer, for Copy. */
export const answerText = (m: AssistantMessage): string => textsOf(m).join("\n\n").trim();

/** The user message an answer replies to (the nearest one before it), for Retry. */
export function promptFor(state: TranscriptState, assistantId: string): UserMessage | undefined {
  const i = state.messages.findIndex((m) => m.role === "assistant" && m.id === assistantId);
  for (let k = i - 1; k >= 0; k--) {
    const m = state.messages[k];
    if (m && m.role === "user") return m;
  }
  return undefined;
}
