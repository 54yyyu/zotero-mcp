// The conversation (ui/chat.ts) against a real session: AcpAgentSession over the mock agent, through the runtime.
// What is checked here is what happens around a session's edges: a new chat while an answer streams, a saved chat
// reopened when its agent session cannot be, and settings changed while the session is still starting.
import assert from "node:assert/strict";
import { after, describe, it } from "node:test";
import { createRuntime } from "../../src/agent/index.ts";
import type { AgentRuntime, ChatEvent, PanelHost, PanelSettings, SavedSession } from "../../src/types.ts";
import { Chat } from "../../src/ui/chat.ts";
import { env } from "../../src/ui/dom.ts";
import { withDefaults } from "../../src/zotero/defaults.ts";
import { cleanupTemp, fakeBridgeDir, hermeticSpawner, tempDir, waitFor } from "../agent/helpers.ts";

// The saver's timers come from the panel's window; node's own do here.
env.win = globalThis as never;

const chats: Chat[] = [];
after(async () => {
  await Promise.all(chats.map((c) => c.dispose()));
  cleanupTemp();
});

interface Rig {
  chat: Chat;
  /** Save a settings change, as the settings screen does: the host hands out a new object after it. */
  set(patch: Partial<PanelSettings>): void;
  /** Everything saved, per chat id, in order. */
  saved: Map<string, ChatEvent[]>;
  /** What the runtime was asked to start, in order. */
  starts: { backend: string; resumeSessionId?: string }[];
  /** Hold the next start until released: a session that takes a while to come up. */
  hold(): () => void;
}

function rig(): Rig {
  const rt = createRuntime({ spawner: hermeticSpawner(), bridgeDir: fakeBridgeDir(), nodePath: process.execPath });
  let settings = withDefaults({});
  const saved = new Map<string, ChatEvent[]>();
  const starts: Rig["starts"] = [];
  const cwd = tempDir("zmc-chat-");
  let gate: Promise<void> = Promise.resolve();
  const runtime: AgentRuntime = {
    ...rt,
    async start(o) {
      starts.push({ backend: o.backend, ...(o.resumeSessionId ? { resumeSessionId: o.resumeSessionId } : {}) });
      await gate;
      return rt.start({ ...o, env: { ...o.env, MOCK_VARIANT: o.backend === "codex" ? "codex" : "claude" } });
    },
  };
  const host = {
    runtime,
    getSettings: () => settings,
    async setSettings(patch: Partial<PanelSettings>) { settings = { ...settings, ...patch }; },
    prepareSession: async (dir?: string) => ({ cwd: dir || cwd, brief: "", env: {} }),
    paperContext: async () => [],
    paperTitle: () => null,
    describeContext: () => ({ text: "", images: [] }),
    async appendEvent(s: SavedSession, ev: ChatEvent) { saved.set(s.id, [...(saved.get(s.id) ?? []), ev]); },
    loadEvents: async (id: string) => saved.get(id) ?? [],
    async renameSession() {},
  } as unknown as PanelHost;
  const chat = new Chat({
    host, onChange() {}, blockReason: () => null, turnEnded() {}, setupFailed() {}, sessionChanged() {}, shownChanged() {},
    expand: async (text) => text,
  });
  chats.push(chat);
  return { chat, set: (patch) => { settings = { ...settings, ...patch }; }, saved, starts, hold() { let open!: () => void; gate = new Promise((r) => { open = r; }); return open; } };
}

const notices = (events: readonly ChatEvent[], level: string) => events.filter((e) => e.t === "notice" && e.level === level);
const answer = (chat: Chat): string => chat.tr.messages.flatMap((m) => (m.role === "assistant" ? m.blocks.flatMap((b) => (b.type === "text" ? [b.text] : [])) : [])).join("");
const errors = (chat: Chat) => chat.tr.messages.filter((m) => m.role === "notice" && m.level !== "info");
const old = (backend: SavedSession["backend"], agentSessionId: string): SavedSession => ({ id: `saved-${agentSessionId}`, title: "an earlier chat", backend, cwd: "", agentSessionId, updatedAt: 1 });

describe("a new chat while an answer streams", () => {
  it("leaves the chat behind saved as stopped, with no error, and the new chat empty", async () => {
    const { chat, saved } = rig();
    const sending = chat.send("SCENARIO:slow", []);
    await waitFor(() => answer(chat).includes("tick1"), 5000, "the answer to start streaming");
    const id = chat.saved!.id;
    await chat.reset();
    assert.equal(chat.tr.messages.length, 0, "the new chat is empty at once");
    assert.equal(chat.busy, false);
    await sending;
    await chat.closeSession(); // the old agent is gone by now
    const events = saved.get(id)!;
    assert.deepEqual(notices(events, "error"), [], "closing the session is not an error of the chat left behind");
    assert.deepEqual(events.at(-1), { t: "turn_end", turn: (events.at(-1) as { turn: string }).turn, stop: "cancelled" });
    assert.equal(chat.tr.messages.length, 0, "nothing of the old answer reached the new chat");
    assert.equal(chat.busy, false);
  });

  it("does not cancel a message sent in the new chat right away", async () => {
    const { chat } = rig();
    const first = chat.send("SCENARIO:slow", []);
    await waitFor(() => answer(chat).includes("tick1"), 5000, "the answer to start streaming");
    await chat.reset();
    await chat.send("hi", []);
    await first;
    assert.equal(answer(chat), "Hello from mock");
    assert.deepEqual(errors(chat), []);
    assert.equal(chat.busy, false);
  });

  it("a new chat while the session is still starting sends nothing and shows nothing", async () => {
    const r = rig();
    const release = r.hold();
    const sending = r.chat.send("hi", []);
    await waitFor(() => r.starts.length === 1, 5000, "the start");
    await r.chat.reset();
    release();
    await sending;
    assert.equal(r.chat.tr.messages.length, 0);
    assert.equal(r.chat.session, null);
    assert.equal(r.chat.busy, false);
  });
});

describe("a saved chat reopened", () => {
  it("resumes on the agent it was made with, whatever agent is chosen now", async () => {
    const { chat, starts } = rig();
    await chat.load(old("codex", "mock-old"));
    assert.equal(chat.backend, "codex");
    await chat.send("hi", []);
    assert.deepEqual(starts, [{ backend: "codex", resumeSessionId: "mock-old" }]);
    assert.equal(chat.session?.backend, "codex");
    assert.equal(chat.session?.sessionId, "mock-old");
    assert.equal(answer(chat), "Hello from mock");
    assert.deepEqual(errors(chat), []);
  });

  it("goes on in a fresh session, and says so once, when the agent no longer has the conversation", async () => {
    const { chat, starts, saved } = rig();
    const s = old("claude-code", "missing-1");
    await chat.load(s);
    await chat.send("hi", []);
    assert.deepEqual(starts, [{ backend: "claude-code", resumeSessionId: "missing-1" }, { backend: "claude-code" }]);
    assert.equal(answer(chat), "Hello from mock");
    assert.deepEqual(errors(chat), []);
    assert.equal(notices(saved.get(s.id)!, "info").length, 1, "one notice that the agent starts fresh");
    assert.notEqual(chat.saved!.agentSessionId, "missing-1", "the chat now points at the session it runs in");
    await chat.send("hi", []);
    assert.equal(starts.length, 2, "the second message uses the same session");
    assert.equal(notices(saved.get(s.id)!, "info").length, 1);
  });

  it("made with an agent that cannot start now, goes on with the chosen one", async () => {
    const { chat, starts } = rig();
    await chat.load(old("opencode", "mock-old")); // no opencode on this PATH
    await chat.send("hi", []);
    assert.deepEqual(starts.map((x) => x.backend), ["opencode", "claude-code"]);
    assert.equal(chat.session?.backend, "claude-code");
    assert.equal(chat.saved!.backend, "claude-code");
    assert.equal(answer(chat), "Hello from mock");
    assert.deepEqual(errors(chat), []);
  });
});

describe("settings changed while the session starts", () => {
  it("another agent chosen: the session that comes up is the chosen one's", async () => {
    const r = rig();
    const release = r.hold();
    const starting = r.chat.ensureSession();
    await waitFor(() => r.starts.length === 1, 5000, "the start");
    r.set({ backend: "codex" });
    release();
    const sess = await starting;
    assert.equal(sess.backend, "codex");
    assert.deepEqual(r.starts.map((x) => x.backend), ["claude-code", "codex"]);
    await r.chat.send("hi", []);
    assert.equal(r.chat.saved!.backend, "codex", "the chat is saved under the agent that answered");
  });

  it("another model picked: the session that comes up is on it", async () => {
    const r = rig();
    const release = r.hold();
    const starting = r.chat.ensureSession();
    await waitFor(() => r.starts.length === 1, 5000, "the start");
    await r.chat.pick("model", "sonnet");
    release();
    const sess = await starting;
    assert.equal(sess.currentModel(), "sonnet");
  });
});

describe("history cleared with a chat open", () => {
  it("the chat is no longer saved: nothing waiting is written, and leaving it writes nothing", async () => {
    const { chat, saved } = rig();
    const sending = chat.send("SCENARIO:slow", []);
    await waitFor(() => answer(chat).includes("tick1"), 5000, "the answer to start streaming");
    const id = chat.saved!.id;
    const before = saved.get(id)!.length;
    chat.forget();
    await chat.reset();
    await sending;
    await new Promise((r) => setTimeout(r, 700)); // past the saver's own delay
    assert.equal(saved.get(id)!.length, before, "nothing is written after the clear");
    assert.equal(chat.saved, null);
  });
});
