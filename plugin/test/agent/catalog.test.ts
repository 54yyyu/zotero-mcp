// catalogOf: what a chat's own session says its backend offers, so the pickers need no agent started just to ask.
import assert from "node:assert/strict";
import { after, test } from "node:test";
import { catalogOf, createRuntime } from "../../src/agent/index.ts";
import type { StartOpts } from "../../src/types.ts";
import { cleanupTemp, fakeBridgeDir, hermeticSpawner, tempDir } from "./helpers.ts";

after(cleanupTemp);

const rt = createRuntime({ spawner: hermeticSpawner(), bridgeDir: fakeBridgeDir(), nodePath: process.execPath });
const start = (o: Partial<StartOpts> = {}) => rt.start({ backend: "claude-code", cwd: tempDir("zmc-cat-"), brief: "", ...o });

test("a session started with nothing chosen says the same as a probe", async () => {
  const probe = await rt.catalog("claude-code");
  const s = await start();
  try {
    assert.deepEqual(catalogOf(s, {}), probe);
  } finally { await s.close(); }
});

test("a session on the user's own model, mode and effort gives the lists, and keeps the defaults known before", async () => {
  const probe = await rt.catalog("claude-code");
  const o = { model: "sonnet", mode: "plan", effort: "low" };
  const s = await start(o);
  try {
    assert.equal(s.currentMode(), "plan");
    const c = catalogOf(s, o, probe);
    assert.deepEqual(c.models, probe.models);
    assert.deepEqual(c.modes, probe.modes);
    assert.equal(c.model, probe.model, "the agent's own default model, not the user's choice");
    assert.equal(c.mode, probe.mode);
    assert.equal(c.effort, probe.effort);
    const alone = catalogOf(s, o);
    assert.equal(alone.mode, undefined, "with nothing known before, the user's choice is not passed off as the agent's default");
    assert.equal(alone.effort, undefined);
  } finally { await s.close(); }
});

test("a resumed session does not know the agent's defaults", async () => {
  const o = { resumeSessionId: "mock-old" };
  const s = await start(o);
  try {
    const c = catalogOf(s, o);
    assert.ok(c.models.length > 0);
    assert.equal(c.model, undefined);
    assert.equal(c.mode, undefined);
  } finally { await s.close(); }
});
