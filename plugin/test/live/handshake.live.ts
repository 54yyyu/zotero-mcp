// Free, opt-in: initialize + session/new (+ the explicit model and mode) against the REAL bridges. No prompt,
// so no model tokens. Needs the bridges installed (the runtime installs the pinned ones into the bridge dir
// on first use: ~600 MB for all three; reuse a dir with ZMC_BRIDGE_DIR).
//
//   ZMC_LIVE_HANDSHAKE=1 [ZMC_BRIDGE_DIR=/tmp/zmc-live-bridges] node --test test/live/handshake.live.ts
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, describe, it } from "node:test";
import { BACKEND_IDS, backendOf } from "../../src/agent/backends.ts";
import { buildBrief, createRuntime } from "../../src/agent/index.ts";
import { createNodeSpawner } from "../node-spawner.ts";

const on = process.env["ZMC_LIVE_HANDSHAKE"] === "1" || process.env["ZMC_LIVE"] === "1";
const bridgeDir = process.env["ZMC_BRIDGE_DIR"] ?? join(tmpdir(), "zmc-live-bridges");
const cwds: string[] = [];
after(() => cwds.forEach((d) => rmSync(d, { recursive: true, force: true })));

describe("real bridge handshakes (no prompt)", { skip: on ? false : "set ZMC_LIVE_HANDSHAKE=1" }, () => {
  const runtime = createRuntime({ spawner: createNodeSpawner(), bridgeDir, onProgress: (m) => console.log(m), initTimeoutMs: 60_000 });

  it("detect() says what is runnable here", async () => {
    const all = await runtime.detect();
    console.log(JSON.stringify(all, null, 1));
    assert.deepEqual(all.map((s) => s.id), BACKEND_IDS);
  });

  for (const id of BACKEND_IDS) {
    it(`${id}: session/new, models, modes, account`, async (t) => {
      const status = (await runtime.detect()).find((s) => s.id === id)!;
      if (!status.available) return t.skip(status.reason);
      const cwd = mkdtempSync(join(tmpdir(), "zmc-live-ws-"));
      cwds.push(cwd);
      const session = await runtime.start({ backend: id, cwd, brief: buildBrief() });
      try {
        console.log(id, { sessionId: session.sessionId, account: session.account, images: session.supportsImages, model: session.currentModel(), models: session.models().length, mode: session.currentMode(), modes: session.modes().map((m) => m.id), effort: session.currentEffort(), efforts: session.efforts().map((e) => e.id) });
        assert.ok(session.sessionId);
        assert.ok(session.models().length > 0, "the bridge lists models");
        assert.ok(session.currentModel(), "and says which one is current");
        if (backendOf(id).permissionModes) assert.ok(session.modes().length > 0 && session.currentMode());
        else assert.deepEqual(session.modes(), []);
        assert.ok(session.efforts().length > 0 && session.currentEffort(), "and an effort / thinking level list");
        // model + mode choice round-trip without a prompt
        const other = session.models().find((m) => m.id !== session.currentModel());
        if (other) {
          await session.setModel(other.id);
          assert.equal(session.currentModel(), other.id);
        }
        const mode = session.modes().find((m) => m.id !== session.currentMode());
        if (mode) {
          await session.setMode(mode.id);
          assert.equal(session.currentMode(), mode.id);
        }
        // (a model with no reasoning, e.g. some of pi's, offers no levels at all: then there is nothing to set)
        const eff = session.efforts().find((e) => e.id !== session.currentEffort());
        if (eff) {
          await session.setEffort(eff.id);
          assert.equal(session.currentEffort(), eff.id);
        }
      } finally {
        await session.close();
      }
    });
  }

  for (const id of BACKEND_IDS) {
    it(`${id}: runtime.catalog() reads the lists from a short-lived bridge and caches them`, async (t) => {
      const status = (await runtime.detect()).find((s) => s.id === id)!;
      if (!status.available) return t.skip(status.reason);
      const c = await runtime.catalog(id);
      console.log(id, "catalog", { models: c.models.map((m) => m.id), modes: c.modes.map((m) => m.id), efforts: c.efforts.map((e) => e.id), model: c.model, mode: c.mode, effort: c.effort });
      assert.ok(c.models.length > 0 && c.efforts.length > 0);
      assert.strictEqual(await runtime.catalog(id), c);
    });
  }

  it("claude-code: an explicit model (full id) and mode are applied at start", async (t) => {
    const status = (await runtime.detect()).find((s) => s.id === "claude-code")!;
    if (!status.available) return t.skip(status.reason);
    const cwd = mkdtempSync(join(tmpdir(), "zmc-live-ws-"));
    cwds.push(cwd);
    const session = await runtime.start({ backend: "claude-code", cwd, brief: buildBrief(), model: "claude-sonnet-5-5", mode: "default" });
    try {
      assert.equal(session.currentModel(), "sonnet", "the bridge lands the full id on its alias option");
      assert.equal(session.currentMode(), "default");
      assert.match(session.account ?? "", /Claude/);
    } finally {
      await session.close();
    }
  });
});
