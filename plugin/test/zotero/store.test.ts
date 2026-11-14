// The chat store's renameSession (zotero/store.ts): a user rename rewrites the index entry and locks it,
// and a session saved before the feature keeps whatever fields it has (no lock) — chat.ts reads that as unlocked.
// store.ts only touches Gecko's IOUtils/PathUtils at call time, so node tests shim them with a Map.
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { ChatEvent, SavedSession } from "../../src/types.ts";

const files = new Map<string, string>();
(globalThis as Record<string, unknown>).PathUtils = { join: (...p: string[]) => p.join("/") };
(globalThis as Record<string, unknown>).IOUtils = {
  async readUTF8(path: string) { const t = files.get(path); if (t === undefined) throw new Error("no such file"); return t; },
  async writeUTF8(path: string, text: string) { files.set(path, text); },
  async makeDirectory() {},
  async remove() {},
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
