// Opt-in, COSTS A FEW MODEL TOKENS: `/name rest` as the panel sends it (ui/skills-model.ts invocationText), to each real
// agent, with a tiny skill in its folder laid out as zotero/skills.ts puts it there. The skill holds a token the agent can
// only know by reading the file. ZMC_LIVE_COMPACT=1 also asks each agent for its own /compact afterwards.
//   ZMC_LIVE=1 [ZMC_LIVE_BACKENDS=claude-code,codex,pi] [ZMC_BRIDGE_DIR=...] node --test test/live/skills.live.ts
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, describe, it } from "node:test";
import { DRAWING_GUIDE, FORMAT_GUIDE, buildBrief, createRuntime } from "../../src/agent/index.ts";
import type { BackendId, ChatEvent, StartOpts } from "../../src/types.ts";
import { AGENT_SKILL_DIRS, invocationText } from "../../src/ui/skills-model.ts";
import { createNodeSpawner } from "../node-spawner.ts";

const on = process.env["ZMC_LIVE"] === "1";
const bridgeDir = process.env["ZMC_BRIDGE_DIR"] ?? join(tmpdir(), "zmc-live-bridges");
const backends = (process.env["ZMC_LIVE_BACKENDS"] ?? "claude-code,codex,pi").split(",") as BackendId[];
const SETUP: Record<BackendId, Partial<StartOpts>> = {
  "claude-code": { model: "haiku", auth: "subscription" },
  codex: { model: "gpt-6-luna[low]", auth: "subscription", mode: "read-only" },
  pi: { model: process.env["ZMC_LIVE_PI_MODEL"] ?? "qwen38/qwen38-27b", auth: "api-key" },
};
const SKILL = "---\nname: token-check\ndescription: Checks that skills are followed. Use only when asked to use the token-check skill.\n---\n\n" +
  "When this skill is used, reply with exactly one line: the token PINEAPPLE-7, a space, then the extra words the user wrote, in capital letters. Nothing else.\n";
const dirs: string[] = [];
after(() => dirs.forEach((d) => rmSync(d, { recursive: true, force: true })));

for (const backend of backends) {
  describe(`skill invocation on ${backend}`, { skip: on ? false : "set ZMC_LIVE=1 (spends tokens)" }, () => {
    it("reads the skill named in a plain message and follows it", async () => {
      const runtime = createRuntime({ spawner: createNodeSpawner(), bridgeDir, initTimeoutMs: 90_000 });
      const ready = (await runtime.detect()).find((s) => s.id === backend);
      if (!ready?.available) { console.log(`SKIP ${backend}: ${ready?.reason}`); return; }
      const cwd = mkdtempSync(join(tmpdir(), "zmc-live-skill-"));
      dirs.push(cwd);
      for (const d of AGENT_SKILL_DIRS) { mkdirSync(join(cwd, d, "token-check"), { recursive: true }); writeFileSync(join(cwd, d, "token-check", "SKILL.md"), SKILL); }
      const session = await runtime.start({ backend, cwd, brief: `${buildBrief()}\n\n${DRAWING_GUIDE}\n\n${FORMAT_GUIDE}`, ...SETUP[backend] } as StartOpts);
      const events: ChatEvent[] = [];
      session.on((e) => {
        events.push(e);
        // a read inside its own folder should not need asking; if it does, allow that one step and say so
        if (e.t === "permission" && !e.resolved) session.respondPermission(e.id, e.options.find((o) => o.kind === "allow_once")?.id ?? null);
      });
      try {
        const t0 = Date.now();
        await session.prompt({ text: invocationText("token-check", "kiwi mango") });
        const said = events.filter((e) => e.t === "text").map((e) => (e as { delta: string }).delta).join("").trim();
        const tools = events.filter((e) => e.t === "tool" && e.status === "done").map((e) => `${(e as { name?: string }).name ?? ""} ${(e as { title: string }).title}`.trim());
        const report = { backend, model: session.currentModel(), ms: Date.now() - t0, said: said.slice(0, 120), tools: [...new Set(tools)].slice(0, 6), asked: events.filter((e) => e.t === "permission").length, usage: events.filter((e) => e.t === "turn_end").map((e) => (e as { usage?: unknown }).usage) };
        console.log(JSON.stringify(report));
        assert.match(said, /PINEAPPLE-7\s+KIWI MANGO/, "followed the skill's instruction");
        if (process.env["ZMC_LIVE_COMPACT"] === "1") {
          const before = events.length;
          await session.compact();
          const notes = events.slice(before).filter((e) => e.t === "notice").map((e) => `${(e as { level: string }).level}: ${(e as { message: string }).message}${(e as { compacted?: boolean }).compacted ? " [compacted]" : ""}`);
          console.log(JSON.stringify({ backend, compact: notes }));
        }
      } finally {
        await session.close();
      }
    });
  });
}
