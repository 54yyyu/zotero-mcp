import { test } from "node:test";
import assert from "node:assert/strict";
import type { ChatEvent } from "../../src/types.ts";
import { answerText, applyEvent, emptyTranscript, promptFor, replay } from "../../src/ui/transcript.ts";
import type { AssistantMessage, TranscriptState } from "../../src/ui/transcript.ts";

const asst = (s: TranscriptState, id: string) => s.messages.find((m) => m.role === "assistant" && m.id === id) as AssistantMessage;

const SCRIPT: ChatEvent[] = [
  { t: "user", id: "u1", text: "Compare studies", chips: [{ id: "c1", kind: "item", label: "Bell 2017", ref: { libraryID: 1, itemKey: "ABCD1234" } }] },
  { t: "turn_start", turn: "t1" },
  { t: "thought", turn: "t1", delta: "Let me " },
  { t: "thought", turn: "t1", delta: "think." },
  { t: "text", turn: "t1", delta: "I will " },
  { t: "text", turn: "t1", delta: "search. " },
  { t: "tool", turn: "t1", id: "x1", title: "Searched library", kind: "search", status: "running", input: { q: "callback" } },
  { t: "tool", turn: "t1", id: "x1", title: "Searched library", kind: "search", status: "done", output: "10 items" },
  { t: "plan", turn: "t1", entries: [{ content: "a", status: "in_progress" }] },
  { t: "plan", turn: "t1", entries: [{ content: "a", status: "completed" }] },
  { t: "permission", turn: "t1", id: "p1", title: "Edit note", kind: "edit", options: [{ id: "ok", name: "Allow once", kind: "allow_once" }] },
  { t: "permission", turn: "t1", id: "p1", title: "Edit note", kind: "edit", options: [{ id: "ok", name: "Allow once", kind: "allow_once" }], resolved: "ok" },
  { t: "text", turn: "t1", delta: "Done " },
  { t: "text", turn: "t1", delta: "[A 2009, p.8](zotero://select/library/items/ABCD1234)." },
  { t: "notice", level: "warn", message: "Slow response" },
  { t: "turn_end", turn: "t1", stop: "end_turn", usage: { inputTokens: 10, outputTokens: 5 } },
  { t: "user", id: "u2", text: "And again", chips: [] },
  { t: "turn_start", turn: "t2" },
  { t: "text", turn: "t2", delta: "Second." },
  { t: "turn_end", turn: "t2", stop: "end_turn" },
];

test("a turn folds into one assistant message with ordered blocks", () => {
  const s = replay(SCRIPT);
  assert.deepEqual(s.messages.map((m) => `${m.role}:${m.id}`), ["user:u1", "assistant:t1", "notice:notice-1", "user:u2", "assistant:t2"]);
  const a = asst(s, "t1");
  assert.deepEqual(a.blocks.map((b) => b.type), ["thought", "text", "tool", "plan", "permission", "text"]);
  assert.equal(a.blocks[0]?.type === "thought" && a.blocks[0].text, "Let me think.");
  assert.equal(a.blocks[1]?.type === "text" && a.blocks[1].text, "I will search. ");
  assert.equal(a.done, true);
  assert.equal(a.stop, "end_turn");
  assert.equal(a.usage?.inputTokens, 10);
  assert.equal(s.running, null);
});

test("tool, plan and permission events upsert by id without moving", () => {
  const s = replay(SCRIPT);
  const a = asst(s, "t1");
  const tool = a.blocks.find((b) => b.type === "tool");
  assert.ok(tool && tool.type === "tool");
  assert.equal(tool.status, "done");
  assert.deepEqual(tool.input, { q: "callback" }, "input from the first event survives an update that omits it");
  assert.equal(tool.output, "10 items");
  assert.equal(a.blocks.filter((b) => b.type === "plan").length, 1);
  const plan = a.blocks.find((b) => b.type === "plan");
  assert.equal(plan?.type === "plan" && plan.entries[0]?.status, "completed");
  const perm = a.blocks.find((b) => b.type === "permission");
  assert.equal(perm?.type === "permission" && perm.resolved, "ok");
});

test("replaying a saved list equals live streaming, event by event", () => {
  let live = emptyTranscript();
  const seen: TranscriptState[] = [];
  for (const ev of SCRIPT) { live = applyEvent(live, ev); seen.push(live); }
  assert.deepEqual(replay(SCRIPT), live);
  // and every prefix: a chat resumed mid-turn looks like it did
  for (let i = 0; i < SCRIPT.length; i++) assert.deepEqual(replay(SCRIPT.slice(0, i + 1)), seen[i]);
});

test("merged deltas (what the panel saves) replay to the same state as the raw stream", () => {
  const merged: ChatEvent[] = [];
  for (const ev of SCRIPT) {
    const last = merged[merged.length - 1];
    if ((ev.t === "text" || ev.t === "thought") && last && last.t === ev.t && last.turn === ev.turn) merged[merged.length - 1] = { ...last, delta: last.delta + ev.delta };
    else merged.push(ev);
  }
  assert.ok(merged.length < SCRIPT.length);
  assert.deepEqual(replay(merged), replay(SCRIPT));
});

test("applyEvent never mutates its input and keeps untouched messages identical", () => {
  let s = replay(SCRIPT.slice(0, 5));
  const frozen = JSON.stringify(s);
  const before = s.messages[0];
  const next = applyEvent(s, { t: "text", turn: "t1", delta: "more" });
  assert.equal(JSON.stringify(s), frozen);
  assert.equal(next.messages[0], before, "the user message is the same object");
  assert.notEqual(next.messages[1], s.messages[1]);
  // a no-op change keeps the whole state's messages' identity for older turns
  s = replay(SCRIPT);
  const later = applyEvent(s, { t: "user", id: "u3", text: "x", chips: [] });
  assert.equal(later.messages[1], s.messages[1]);
});

test("a user event with the same id replaces the message (retry, replay)", () => {
  const s = replay([{ t: "user", id: "u", text: "a", chips: [] }, { t: "user", id: "u", text: "b", chips: [] }]);
  assert.equal(s.messages.length, 1);
  assert.equal(s.messages[0]?.role === "user" && s.messages[0].text, "b");
});

test("a cancelled or failed turn leaves no step running", () => {
  const s = replay([
    { t: "turn_start", turn: "t" },
    { t: "tool", turn: "t", id: "a", title: "A", kind: "read", status: "running" },
    { t: "tool", turn: "t", id: "b", title: "B", kind: "read", status: "done" },
    { t: "turn_end", turn: "t", stop: "cancelled" },
  ]);
  const a = asst(s, "t");
  assert.deepEqual(a.blocks.map((b) => b.type === "tool" && b.status), ["failed", "done"]);
  assert.equal(a.stop, "cancelled");
});

test("running tracks the open turn, and a late delta does not reopen a finished one", () => {
  let s = applyEvent(emptyTranscript(), { t: "turn_start", turn: "t" });
  assert.equal(s.running, "t");
  s = applyEvent(s, { t: "text", turn: "t", delta: "hi" });
  assert.equal(s.running, "t");
  s = applyEvent(s, { t: "turn_end", turn: "t", stop: "end_turn" });
  assert.equal(s.running, null);
  s = applyEvent(s, { t: "text", turn: "t", delta: " late" });
  assert.equal(s.running, null);
  assert.equal(answerText(asst(s, "t")), "hi late");
});

test("a log that starts mid-turn (a partial save) still builds the message", () => {
  const s = replay([{ t: "text", turn: "t9", delta: "tail of an answer" }, { t: "turn_end", turn: "t9", stop: "end_turn" }]);
  assert.equal(asst(s, "t9").done, true);
  assert.equal(answerText(asst(s, "t9")), "tail of an answer");
});

test("notice ids are deterministic and an unresolved permission counts as pending", () => {
  const s = replay([
    { t: "notice", level: "error", message: "a" }, { t: "notice", level: "info", message: "b", hint: "h" },
    { t: "permission", turn: "t", id: "p", title: "X", kind: "execute", options: [] },
  ]);
  assert.deepEqual(s.messages.filter((m) => m.role === "notice").map((m) => m.id), ["notice-1", "notice-2"]);
  const open = asst(s, "t").blocks[0];
  assert.ok(open?.type === "permission" && open.resolved === undefined, "an unanswered question has no `resolved`");
});

test("promptFor finds the user message an answer replies to", () => {
  const s = replay(SCRIPT);
  assert.equal(promptFor(s, "t2")?.id, "u2");
  assert.equal(promptFor(s, "t1")?.id, "u1");
  assert.equal(promptFor(s, "nope"), undefined);
});

test("a tool's own name is kept across updates that omit it", () => {
  const s = replay([
    { t: "turn_start", turn: "t" },
    { t: "tool", turn: "t", id: "a", title: "`zotero-cli search x`", kind: "execute", status: "running", name: "Bash" },
    { t: "tool", turn: "t", id: "a", title: "`zotero-cli search x`", kind: "execute", status: "done", output: "ok" },
    { t: "permission", turn: "t", id: "p", title: "Run it", kind: "execute", options: [], name: "Bash" },
  ]);
  const blocks = asst(s, "t").blocks;
  assert.equal(blocks[0]?.type === "tool" && blocks[0].name, "Bash");
  assert.equal(blocks[1]?.type === "permission" && blocks[1].name, "Bash");
});
