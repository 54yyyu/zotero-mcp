// How a chat gets its name (ui/chat.ts): the first message names it by the setting's fallback chain
// (agent summary is never ready yet, so it falls to the focused paper, then the question), the agent's own
// title replaces that whenever it arrives — but only on an agent-named chat, and never after a user rename.
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { Chat } from "../../src/ui/chat.ts";
import { FakeHost } from "../../src/ui/fake-host.ts";
import { env } from "../../src/ui/dom.ts";
import type { ContextChip } from "../../src/types.ts";

env.win = globalThis as typeof env.win; // the transcript saver's timers

const deps = (host: FakeHost) => ({ host, onChange() {}, blockReason: () => null, turnEnded() {}, setupFailed() {}, sessionChanged() {}, shownChanged() {}, expand: async (t: string) => t });

const READER: ContextChip = { id: "reader:BM2004AB", kind: "reader", label: "Bertrand and Mullainathan 2004", auto: true, pinned: false, ref: { libraryID: 1, itemKey: "BM2004AB", attachmentKey: "ATT00001", pageIndex: 6 } };
const LONG_PAPER: ContextChip = { ...READER, id: "reader:LONGTITL", ref: { libraryID: 1, itemKey: "LONGTITL", attachmentKey: "ATT00002", pageIndex: 0 } };

const chat = (host: FakeHost) => new Chat(deps(host));
const say = (host: FakeHost, c: Chat, text: string, used: ContextChip[] = []) => { host.sim.nextAnswer = "ok"; return c.send(text, used); };

describe("the first name", () => {
  it("agent source: the focused paper's title, snapshotted as an agent-named chat", async () => {
    const host = new FakeHost({ speed: 0, noHistory: true });
    const c = chat(host);
    await say(host, c, "what is the callback gap?", [READER]);
    assert.equal(c.saved!.title, "Are Emily and Greg More Employable than Lakisha and Jamal?");
    assert.equal(c.saved!.titleSource, "agent", "the setting's value rides on the chat, so later setting changes never re-rule it");
    assert.equal(c.saved!.titleLocked, undefined);
  });

  it("agent source with no paper: the question text", async () => {
    const host = new FakeHost({ speed: 0, noHistory: true });
    const c = chat(host);
    await say(host, c, "which of my papers cite Pager 2003?", []);
    assert.equal(c.saved!.title, "which of my papers cite Pager 2003?");
    assert.equal(c.saved!.titleSource, "agent");
  });

  it("an item chip (the library selection, no reader open) names the chat too", async () => {
    const host = new FakeHost({ speed: 0, noHistory: true });
    const c = chat(host);
    await say(host, c, "what did they find?", [{ id: "item:BM2004AB", kind: "item", label: "Bertrand and Mullainathan 2004", auto: true, pinned: false, ref: { libraryID: 1, itemKey: "BM2004AB" } }]);
    assert.equal(c.saved!.title, "Are Emily and Greg More Employable than Lakisha and Jamal?");
  });

  it("paper source with no paper: the question text", async () => {
    const host = new FakeHost({ speed: 0, noHistory: true });
    await host.setSettings({ titleSource: "paper" });
    const c = chat(host);
    await say(host, c, "summarise what is open", []);
    assert.equal(c.saved!.title, "summarise what is open");
    assert.equal(c.saved!.titleSource, "paper");
  });

  it("question source ignores the paper entirely", async () => {
    const host = new FakeHost({ speed: 0, noHistory: true });
    await host.setSettings({ titleSource: "question" });
    const c = chat(host);
    await say(host, c, "explain table 4", [READER]);
    assert.equal(c.saved!.title, "explain table 4");
  });

  it("a long paper title clips at 80 characters, a long question at 60", async () => {
    const host = new FakeHost({ speed: 0, noHistory: true });
    const c = chat(host);
    await say(host, c, "read this", [LONG_PAPER]);
    assert.equal(c.saved!.title.length, 80);
    assert.ok(c.saved!.title.endsWith("…"));
    const host2 = new FakeHost({ speed: 0, noHistory: true });
    const c2 = chat(host2);
    const long = "question ".repeat(12).trim();
    await say(host2, c2, long, []);
    assert.equal(c2.saved!.title.length, 60);
    assert.ok(c2.saved!.title.endsWith("…"));
  });
});

describe("the agent's own title", () => {
  it("arrives mid-turn and replaces the fallback name", async () => {
    const host = new FakeHost({ speed: 0, noHistory: true });
    const c = chat(host);
    await say(host, c, "retitle: what does section 2 argue?", [READER]);
    assert.equal(c.saved!.title, "The audit-study question, summarised by the agent");
    assert.equal(c.saved!.titleLocked, undefined, "an agent title does not lock anything");
  });

  it("arrives just after the turn ends (a list-only backend, pulled) and lands the same way", async () => {
    const host = new FakeHost({ speed: 0, noHistory: true });
    const c = chat(host);
    await say(host, c, "listtitle: what does section 2 argue?", [READER]);
    await new Promise((r) => setTimeout(r, 20));
    assert.equal(c.saved!.title, "Pulled from the session list", "same rules as a pushed title");
    const host2 = new FakeHost({ speed: 0, noHistory: true });
    const c2 = chat(host2);
    await say(host2, c2, "hello", []);
    await c2.rename("Kept by the user");
    await say(host2, c2, "listtitle again", []);
    await new Promise((r) => setTimeout(r, 20));
    assert.equal(c2.saved!.title, "Kept by the user", "a locked chat ignores a pulled title too");
  });

  it("is ignored on a chat named under another source", async () => {
    const host = new FakeHost({ speed: 0, noHistory: true });
    await host.setSettings({ titleSource: "paper" });
    const c = chat(host);
    await say(host, c, "retitle: what does section 2 argue?", [READER]);
    assert.equal(c.saved!.title, "Are Emily and Greg More Employable than Lakisha and Jamal?");
  });

  it("never overwrites a user rename", async () => {
    const host = new FakeHost({ speed: 0, noHistory: true });
    const c = chat(host);
    await say(host, c, "hello", []);
    await c.rename("My own name for this chat");
    assert.equal(c.saved!.title, "My own name for this chat");
    assert.equal(c.saved!.titleLocked, true);
    await say(host, c, "retitle again", []);
    assert.equal(c.saved!.title, "My own name for this chat", "locked: the agent's title changes nothing");
  });

  it("a rename to blank changes nothing", async () => {
    const host = new FakeHost({ speed: 0, noHistory: true });
    const c = chat(host);
    await say(host, c, "hello", []);
    await c.rename("   ");
    assert.equal(c.saved!.title, "hello");
    assert.equal(c.saved!.titleLocked, undefined);
  });

  it("a rename reaches the store", async () => {
    const host = new FakeHost({ speed: 0, noHistory: true });
    const c = chat(host);
    await say(host, c, "hello", []);
    await c.rename("Renamed from the test");
    const all = await host.sessions();
    const saved = all.find((s) => s.id === c.saved!.id);
    assert.equal(saved!.title, "Renamed from the test");
    assert.equal(saved!.titleLocked, true);
  });
});

describe("a resumed pre-feature chat", () => {
  it("has no source snapshot and accepts agent titles", async () => {
    const host = new FakeHost({ speed: 0, noHistory: true });
    const c = chat(host);
    // A chat saved before the feature: no titleSource on the saved record.
    await say(host, c, "an old chat", []);
    c.saved = { ...c.saved!, titleSource: undefined };
    await say(host, c, "retitle: continue", []);
    assert.equal(c.saved!.title, "The audit-study question, summarised by the agent", "a missing snapshot reads as agent-named");
  });
});
