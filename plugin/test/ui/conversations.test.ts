import assert from "node:assert/strict";
import { test } from "node:test";
import { Chat } from "../../src/ui/chat.ts";
import { Conversations } from "../../src/ui/conversations.ts";
import { env } from "../../src/ui/dom.ts";
import { FakeHost } from "../../src/ui/fake-host.ts";
import { withDefaults } from "../../src/zotero/defaults.ts";
import { waitFor } from "../agent/helpers.ts";

env.win = globalThis as never;

function setup(limit: number) {
  const host = new FakeHost({ speed: 2, noHistory: true, context: "none" });
  void host.setSettings({ backgroundChatLimit: limit });
  const stack: Conversations = new Conversations({
    makeChat: (onChange) => new Chat({
      host, onChange, beforeStart: () => stack.waitForClose(),
      blockReason: () => null, turnEnded() {}, setupFailed() {}, sessionChanged() {}, shownChanged() {},
      expand: async (text) => text,
    }),
    limit: () => host.getSettings().backgroundChatLimit,
    foregroundChanged() {}, backgroundChanged() {},
  });
  host.onHistoryClear(() => stack.clear());
  return { host, stack };
}

const answer = (chat: Chat) => chat.tr.messages.flatMap((msg) => msg.role === "assistant"
  ? msg.blocks.flatMap((block) => block.type === "text" ? [block.text] : []) : []).join("");
const streaming = (chat: Chat) => waitFor(() => !!answer(chat), 5000, "the answer to start streaming");
const stopOf = async (host: FakeHost, id: string) => (await host.loadEvents(id)).findLast((event) => event.t === "turn_end")?.stop;

test("background chat settings default to zero and accept only nonnegative integer counts", () => {
  assert.equal(withDefaults({}).backgroundChatLimit, 0);
  assert.equal(withDefaults({ backgroundChatLimit: 2 }).backgroundChatLimit, 2);
  for (const invalid of [-1, 1.5, Infinity, NaN, "2"]) {
    assert.equal(withDefaults({ backgroundChatLimit: invalid as number }).backgroundChatLimit, 0);
  }
});

test("a running chat keeps answering offscreen and can be reopened and stopped", async (t) => {
  const { host, stack } = setup(1);
  t.after(() => stack.dispose());
  const first = stack.current;
  host.sim.nextAnswer = "first answer ".repeat(600);
  const sending = first.send("first question", []);
  await streaming(first);
  const entry = first.saved!;
  const session = first.session;
  const before = answer(first);

  stack.newChat();
  assert.equal(stack.isBackground(entry.id), true);
  assert.deepEqual(stack.current.tr.messages, []);
  host.sim.nextAnswer = "Only the second answer";
  await stack.current.send("second question", []);
  assert.equal(answer(stack.current), "Only the second answer");
  assert.ok(answer(first).length > before.length, "the first answer advanced in the background");

  assert.equal(await stack.open(entry), true);
  assert.equal(stack.current, first);
  assert.equal(first.session, session);
  assert.equal(stack.isBackground(entry.id), false);
  await first.stop();
  await sending;
  assert.equal(await stopOf(host, entry.id), "cancelled");
});

test("a completed background answer is saved and reopening it needs only the transcript", async (t) => {
  const { host, stack } = setup(1);
  t.after(() => stack.dispose());
  const first = stack.current;
  const text = "Finished in the background. ".repeat(20);
  host.sim.nextAnswer = text;
  const sending = first.send("write a summary", []);
  await streaming(first);
  const entry = first.saved!;
  stack.newChat();

  await sending;
  assert.equal(await stopOf(host, entry.id), "end_turn");
  assert.equal(stack.isBackground(entry.id), false);
  assert.equal(first.session, null, "the finished background agent was released");
  assert.equal(await stack.open(entry), true);
  assert.equal(answer(stack.current), text);
  assert.equal(stack.current.session, null);
});

test("reaching or lowering the limit stops the appropriate turns", async (t) => {
  const { host, stack } = setup(1);
  t.after(() => stack.dispose());
  host.sim.nextAnswer = "first answer ".repeat(600);
  const first = stack.current;
  const firstSend = first.send("first", []);
  await streaming(first);
  stack.newChat();

  host.sim.nextAnswer = "second answer ".repeat(600);
  const second = stack.current;
  const secondSend = second.send("second", []);
  await streaming(second);
  stack.newChat();
  await secondSend;
  assert.equal(await stopOf(host, second.saved!.id), "cancelled");
  assert.equal(stack.isBackground(first.saved!.id), true);

  await host.setSettings({ backgroundChatLimit: 0 });
  stack.trim();
  await firstSend;
  assert.equal(await stopOf(host, first.saved!.id), "cancelled");
  assert.equal(stack.isBackground(first.saved!.id), false);
});

test("deleting or clearing history stops live chats before the transcript is removed", async (t) => {
  const { host, stack } = setup(2);
  t.after(() => stack.dispose());
  const send = async () => {
    const chat = stack.current;
    host.sim.nextAnswer = "answer ".repeat(600);
    const sending = chat.send("question", []);
    await streaming(chat);
    return { id: chat.saved!.id, chat, sending };
  };
  const first = await send();
  stack.newChat();
  const second = await send();
  stack.newChat();
  const third = await send();

  assert.equal(stack.remove(first.id), false);
  await host.deleteSession(first.id);
  await first.sending;
  assert.deepEqual(await host.loadEvents(first.id), []);
  assert.equal(stack.isBackground(first.id), false);

  await host.clearHistory();
  await Promise.all([second.sending, third.sending]);
  assert.deepEqual(await host.sessions(), []);
  assert.equal(stack.isBackground(second.id), false);
  assert.equal(stack.current.saved, null);
});
