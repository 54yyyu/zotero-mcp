// Opt-in, COSTS MODEL TOKENS on the user's subscription: a couple of tiny real prompts against Claude Code.
//   ZMC_LIVE=1 [ZMC_BRIDGE_DIR=...] node --test test/live/chat.live.ts
// Not part of `npm test`. Do not run casually.
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, describe, it } from "node:test";
import { buildBrief, createRuntime } from "../../src/agent/index.ts";
import type { ChatEvent } from "../../src/types.ts";
import { createNodeSpawner } from "../node-spawner.ts";

const on = process.env["ZMC_LIVE"] === "1";
const bridgeDir = process.env["ZMC_BRIDGE_DIR"] ?? join(tmpdir(), "zmc-live-bridges");
const cwds: string[] = [];
after(() => cwds.forEach((d) => rmSync(d, { recursive: true, force: true })));

describe("real Claude Code, one tiny prompt", { skip: on ? false : "set ZMC_LIVE=1 (spends tokens)" }, () => {
  const runtime = createRuntime({ spawner: createNodeSpawner(), bridgeDir, initTimeoutMs: 60_000 });

  it("answers, streams, ends the turn; a second session resumes it without replaying history", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "zmc-live-ws-"));
    cwds.push(cwd);
    const base = { backend: "claude-code" as const, cwd, brief: buildBrief(), model: "haiku", mode: "default" };
    const first = await runtime.start(base);
    const events: ChatEvent[] = [];
    first.on((e) => events.push(e));
    try {
      await first.prompt({ text: "Reply with exactly the word: pong. Do not use any tools." });
      const text = events.filter((e) => e.t === "text").map((e) => (e as { delta: string }).delta).join("");
      assert.match(text, /pong/i);
      assert.equal(events[0]!.t, "turn_start");
      assert.equal((events.at(-1) as { stop: string }).stop, "end_turn");
      // Summarise now: the bridge's /compact, silently; one notice with a smaller fill (measured 24k -> 2.7k on 2026-10-05)
      const before = (events.at(-1) as { usage?: { contextUsed?: number } }).usage?.contextUsed ?? Infinity;
      assert.equal(first.canCompact, true, "claude-agent-acp advertises /compact");
      events.length = 0;
      await first.compact();
      assert.deepEqual(events.map((e) => e.t), ["notice"], JSON.stringify(events));
      const n = events[0] as unknown as Extract<ChatEvent, { t: "notice" }>;
      assert.ok(n.compacted && n.context && n.context.used < before, `compacted, smaller: ${JSON.stringify(n)} before ${before}`);
    } finally {
      await first.close();
    }

    const second = await runtime.start({ ...base, resumeSessionId: first.sessionId });
    const replayed: ChatEvent[] = [];
    second.on((e) => replayed.push(e));
    try {
      await new Promise((r) => setTimeout(r, 500));
      assert.equal(replayed.length, 0, "session/load history is not emitted as live events");
      await second.prompt({ text: "What single word did I ask you to reply with? Answer in one word, no tools." });
      assert.match((replayed as ChatEvent[]).filter((e) => e.t === "text").map((e) => (e as { delta: string }).delta).join(""), /pong/i);
    } finally {
      await second.close();
    }
  });
});
