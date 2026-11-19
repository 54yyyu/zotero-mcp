// The chat store's renameSession (zotero/store.ts): a user rename rewrites the index entry and locks it,
// and a session saved before the feature keeps whatever fields it has (no lock) — chat.ts reads that as unlocked.
// store.ts only touches Gecko's IOUtils/PathUtils at call time, so node tests shim them with a Map.
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { ChatEvent, SavedSession } from "../../src/types.ts";

const files = new Map<string, string>();
/** Index writes that did not go through a temporary file (a crash mid-write would leave half a list). */
const direct: string[] = [];
let indexWrites = 0;
(globalThis as Record<string, unknown>).PathUtils = { join: (...p: string[]) => p.join("/") };
(globalThis as Record<string, unknown>).IOUtils = {
  async readUTF8(path: string) { const t = files.get(path); if (t === undefined) throw new Error("no such file"); return t; },
  async writeUTF8(path: string, text: string, o: { mode?: string; tmpPath?: string } = {}) {
    if (path.endsWith("index.json")) { indexWrites++; if (!o.tmpPath) direct.push(path); }
    files.set(path, o.mode === "appendOrCreate" ? (files.get(path) ?? "") + text : text);
  },
  async makeDirectory() {},
  async move(from: string, to: string) { files.set(to, files.get(from)!); files.delete(from); },
  async remove(path: string, o: { recursive?: boolean } = {}) {
    for (const k of [...files.keys()]) if (k === path || (o.recursive && k.startsWith(path + "/"))) files.delete(k);
  },
};

const { createStore } = await import("../../src/zotero/store.ts");

const session = (id: string, title: string): SavedSession => ({ id, title, backend: "claude-code", cwd: "/w", agentSessionId: `agent-${id}`, updatedAt: 1 });
const firstMessage: ChatEvent = { t: "user", id: "u1", text: "the first question", chips: [] };

describe("renameSession", () => {
  it("rewrites the title and locks it in the index", async () => {
    const store = createStore("/chats/a");
    await store.appendEvent(session("abc", "the first question"), firstMessage);
    await store.renameSession("abc", "My paper chat");
    const all = await store.sessions();
    assert.equal(all.length, 1);
    assert.equal(all[0]!.title, "My paper chat");
    assert.equal(all[0]!.titleLocked, true, "a rename is the lock: no automatic source changes the title again");
  });

  it("keeps every other field of the entry", async () => {
    const store = createStore("/chats/b");
    await store.appendEvent({ ...session("keep", "q"), titleSource: "agent" }, firstMessage);
    await store.renameSession("keep", "renamed");
    const s = (await store.sessions())[0]!;
    assert.equal(s.titleSource, "agent");
    assert.equal(s.backend, "claude-code");
    assert.equal(s.cwd, "/w");
    assert.equal(s.agentSessionId, "agent-keep");
  });

  it("leaves a session saved before the feature without the lock field", async () => {
    const store = createStore("/chats/c");
    await store.appendEvent(session("old", "an old title"), firstMessage);
    const s = (await store.sessions())[0]!;
    assert.equal("titleLocked" in s, false, "chat.ts reads a missing lock as unlocked: agent titles may still apply");
  });

  it("renaming a session that does not exist is a no-op", async () => {
    const store = createStore("/chats/d");
    await store.renameSession("nope", "x");
    assert.deepEqual(await store.sessions(), []);
  });
});

describe("the index survives", () => {
  it("is always written through a temporary file", async () => {
    const store = createStore("/chats/atomic");
    await store.appendEvent(session("a", "one"), firstMessage);
    await store.renameSession("a", "renamed");
    await store.deleteSession("a");
    assert.deepEqual(direct, []);
  });

  it("a list cut off mid-write is set aside, not replaced by the chat being saved", async () => {
    const store = createStore("/chats/cut");
    await store.appendEvent(session("a", "one"), firstMessage);
    await store.appendEvent(session("b", "two"), firstMessage);
    const whole = files.get("/chats/cut/index.json")!;
    const half = whole.slice(0, whole.length / 2);
    files.set("/chats/cut/index.json", half);
    await store.appendEvent(session("c", "three"), firstMessage);
    const aside = [...files.keys()].filter((k) => k.startsWith("/chats/cut/index.json.unreadable-"));
    assert.equal(aside.length, 1, "the damaged list is kept");
    assert.equal(files.get(aside[0]!), half);
    assert.deepEqual((await store.sessions()).map((s) => s.id), ["c"]);
  });

  it("is rewritten for a new chat, a sent message, a turn's end and a changed entry, not for every streamed event", async () => {
    const store = createStore("/chats/quiet");
    const s = session("a", "one");
    const writes = async (ev: ChatEvent, of = s) => { const n = indexWrites; await store.appendEvent(of, ev); return indexWrites - n; };
    assert.equal(await writes(firstMessage), 1);
    assert.equal(await writes({ t: "turn_start", turn: "t1" }), 0);
    assert.equal(await writes({ t: "text", turn: "t1", delta: "Hello" }), 0);
    assert.equal(await writes({ t: "text", turn: "t1", delta: " there" }, { ...s, updatedAt: 99 }), 0, "a newer time alone is not news");
    assert.equal(await writes({ t: "text", turn: "t1", delta: "." }, { ...s, title: "The agent's title" }), 1);
    assert.equal(await writes({ t: "turn_end", turn: "t1", stop: "end_turn" }, { ...s, title: "The agent's title" }), 1);
    assert.equal((await store.loadEvents("a")).length, 6, "every event is in the chat's own file");
    assert.equal((await store.sessions())[0]!.title, "The agent's title");
  });

  it("after a rename elsewhere, the next event writes this chat's entry again", async () => {
    const store = createStore("/chats/renamed");
    const s = session("a", "one");
    await store.appendEvent(s, firstMessage);
    await store.renameSession("a", "mine");
    await store.appendEvent({ ...s, title: "mine", titleLocked: true }, { t: "text", turn: "t1", delta: "x" });
    const got = (await store.sessions())[0]!;
    assert.equal(got.title, "mine");
    assert.equal(got.titleLocked, true);
  });
});

describe("clear history", () => {
  it("removes every chat; one saved afterwards starts a new list", async () => {
    const store = createStore("/chats/clear");
    await store.appendEvent(session("a", "one"), firstMessage);
    await store.appendEvent(session("b", "two"), firstMessage);
    await store.clearAll();
    assert.deepEqual(await store.sessions(), []);
    assert.deepEqual(await store.loadEvents("a"), []);
    await store.appendEvent(session("c", "three"), firstMessage);
    assert.deepEqual((await store.sessions()).map((s) => s.id), ["c"]);
  });
});
