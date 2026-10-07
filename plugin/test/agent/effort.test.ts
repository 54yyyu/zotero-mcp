import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { after, describe, it } from "node:test";
import { createRuntime } from "../../src/agent/index.ts";
import type { AgentSession, Spawner, StartOpts } from "../../src/types.ts";
import { createNodeSpawner } from "../node-spawner.ts";
import { alive, cleanupTemp, collect, fakeBridgeDir, hermeticSpawner, tempDir, textOf, waitFor } from "./helpers.ts";

after(cleanupTemp);

const open: AgentSession[] = [];
after(async () => {
  await Promise.all(open.map((s) => s.close()));
});

async function start(opts: Partial<StartOpts> & { variant?: string } = {}): Promise<AgentSession> {
  const rt = createRuntime({ spawner: hermeticSpawner(), bridgeDir: fakeBridgeDir(), nodePath: process.execPath });
  const { variant, ...rest } = opts;
  const s = await rt.start({ backend: "claude-code", cwd: tempDir("zmc-ws-"), brief: "x", ...rest, env: { MOCK_VARIANT: variant ?? "claude", ...(opts.env ?? {}) } });
  open.push(s);
  return s;
}

/** What the mock bridge itself believes (model, mode, effort). */
async function bridgeState(s: AgentSession): Promise<{ model: string; mode: string; effort: string }> {
  const { events } = collect(s);
  await s.prompt({ text: "SCENARIO:whoami" });
  return JSON.parse(textOf(events));
}

const ids = (l: { id: string }[]) => l.map((x) => x.id);

describe("efforts: claude (config option `effort`)", () => {
  it("lists the levels (without the 'default' alias) and the current one", async () => {
    const s = await start();
    assert.deepEqual(ids(s.efforts()), ["low", "medium", "high", "xhigh", "max"]);
    assert.equal(s.currentEffort(), "medium");
  });

  it("setEffort sends set_config_option and rejects a level the bridge refuses", async () => {
    const s = await start();
    await s.setEffort("low");
    assert.equal(s.currentEffort(), "low");
    assert.equal((await bridgeState(s)).effort, "low");
    await assert.rejects(s.setEffort("turbo"));
    assert.equal(s.currentEffort(), "low");
  });

  it("keeps the effort across a model change even when the bridge resets it; a model with no effort has none", async () => {
    const s = await start();
    await s.setEffort("low");
    await s.setModel("sonnet"); // the mock resets the effort to "high" like the real bridge did
    assert.equal(s.currentEffort(), "low");
    assert.equal((await bridgeState(s)).effort, "low");
    await s.setModel("haiku"); // no effort option at all
    assert.deepEqual(s.efforts(), []);
    assert.equal(s.currentEffort(), undefined);
    await assert.rejects(s.setEffort("low"), /no effort/);
  });

  it("start({effort}) applies a known level and skips an unknown one", async () => {
    assert.equal((await start({ effort: "xhigh" })).currentEffort(), "xhigh");
    const s = await start({ effort: "turbo" });
    assert.equal(s.currentEffort(), "medium", "unknown level skipped, start still succeeds");
    const t = await start({ model: "sonnet", effort: "max" });
    assert.equal(t.currentEffort(), "max", "applied after the model, which would have reset it");
  });
});

describe("efforts: pi (config option `thought_level`)", () => {
  it("lists thinking levels and sets one", async () => {
    const s = await start({ backend: "pi", variant: "pi" });
    assert.deepEqual(ids(s.efforts()), ["off", "low", "high"]);
    assert.equal(s.currentEffort(), "high");
    await s.setEffort("low");
    assert.equal((await bridgeState(s)).effort, "low");
    assert.deepEqual(s.modes(), [], "pi's `modes` stay hidden; they are the same thing");
  });
  it("start({effort})", async () => {
    assert.equal((await start({ backend: "pi", variant: "pi", effort: "off" })).currentEffort(), "off");
  });
});

describe("efforts: codex (level is a suffix on the model id)", () => {
  const codex = (o: Partial<StartOpts> = {}) => start({ backend: "codex", variant: "codex", ...o });

  it("models are the unique bases, efforts the suffixes found, current from the bridge's id", async () => {
    const s = await codex();
    assert.deepEqual(s.models().map((m) => [m.id, m.name]), [["gpt-6-sol", "6 Sol"], ["gpt-6-luna", "6 Luna"], ["gpt-5.5", "5.5"]]);
    assert.deepEqual(ids(s.efforts()), ["low", "medium", "high", "max"]);
    assert.equal(s.currentModel(), "gpt-6-sol");
    assert.equal(s.currentEffort(), "medium");
  });

  it("setEffort and setModel recombine into the full id the bridge wants", async () => {
    const s = await codex();
    await s.setEffort("high");
    assert.equal((await bridgeState(s)).model, "gpt-6-sol[high]");
    await s.setModel("gpt-6-luna");
    assert.equal((await bridgeState(s)).model, "gpt-6-luna[high]", "effort kept when the new model has it");
    assert.equal(s.currentEffort(), "high");
    await s.setEffort("max");
    await s.setModel("gpt-5.5"); // no max there: falls back to its default (medium)
    assert.equal((await bridgeState(s)).model, "gpt-5.5[medium]");
    assert.equal(s.currentModel(), "gpt-5.5");
    assert.equal(s.currentEffort(), "medium");
    await assert.rejects(s.setEffort("max"), /no effort level "max"/);
    await s.setModel("gpt-6-luna[max]"); // a full id is accepted too
    assert.equal(s.currentModel(), "gpt-6-luna");
    assert.equal(s.currentEffort(), "max");
  });

  it("a default model missing from the bridge's own list is still shown, with the one level it is on", async () => {
    const s = await codex({ env: { MOCK_CODEX_HIDE_DEFAULT: "1" } });
    assert.deepEqual(ids(s.models()), ["gpt-6-sol", "gpt-6-luna", "gpt-5.5"]);
    assert.equal(s.currentModel(), "gpt-6-sol");
    assert.equal(s.currentEffort(), "medium");
    await s.setModel("gpt-6-luna");
    assert.equal(s.currentModel(), "gpt-6-luna");
  });

  it("start({model, effort}) applies both, skipping a level the model lacks", async () => {
    const a = await codex({ model: "gpt-6-luna", effort: "max" });
    assert.equal((await bridgeState(a)).model, "gpt-6-luna[max]");
    const b = await codex({ model: "gpt-5.5", effort: "max" });
    assert.equal((await bridgeState(b)).model, "gpt-5.5[medium]", "5.5 has no max: model applied, effort skipped");
    const c = await codex({ effort: "low" });
    assert.equal((await bridgeState(c)).model, "gpt-6-sol[low]");
  });
});

describe("start({auth})", () => {
  async function envOf(backend: "claude-code" | "codex", auth: StartOpts["auth"], extra: Record<string, string> = {}): Promise<Record<string, string>> {
    const dump = join(tempDir(), "dump.json");
    const rt = createRuntime({
      spawner: hermeticSpawner({ ANTHROPIC_API_KEY: "sk-a", ANTHROPIC_AUTH_TOKEN: "tok", OPENAI_API_KEY: "sk-o", CODEX_API_KEY: "sk-c", MOCK_DUMP: dump, MOCK_VARIANT: backend === "codex" ? "codex" : "claude" }),
      bridgeDir: fakeBridgeDir(),
      nodePath: process.execPath,
    });
    const s = await rt.start({ backend, cwd: tempDir("zmc-ws-"), brief: "x", ...(auth ? { auth } : {}), env: extra });
    open.push(s);
    return (JSON.parse(readFileSync(dump, "utf8")) as { env: Record<string, string> }).env;
  }
  it("subscription removes the backend's API-key variables; api-key (or nothing) leaves them; explicit env wins", async () => {
    assert.equal((await envOf("claude-code", "subscription"))["ANTHROPIC_API_KEY"], undefined);
    assert.equal((await envOf("claude-code", "api-key"))["ANTHROPIC_API_KEY"], "sk-a");
    assert.equal((await envOf("claude-code", undefined))["ANTHROPIC_API_KEY"], "sk-a");
    assert.equal((await envOf("claude-code", "subscription", { ANTHROPIC_API_KEY: "sk-explicit" }))["ANTHROPIC_API_KEY"], "sk-explicit");
    const sub = await envOf("claude-code", "subscription");
    assert.equal(sub["ANTHROPIC_AUTH_TOKEN"], undefined);
    const codexSub = await envOf("codex", "subscription");
    assert.deepEqual([codexSub["OPENAI_API_KEY"], codexSub["CODEX_API_KEY"]], [undefined, undefined]);
    assert.equal((await envOf("codex", "api-key"))["OPENAI_API_KEY"], "sk-o");
  });
});

describe("runtime.catalog()", () => {
  function counting(env: Record<string, string>): { spawner: Spawner; spawns: { cwd?: string }[] } {
    const base = createNodeSpawner({ graceMs: 500, baseEnv: env });
    const spawns: { cwd?: string }[] = [];
    return {
      spawner: { ...base, spawn: (c, a, o) => (spawns.push({ ...(o.cwd ? { cwd: o.cwd } : {}) }), base.spawn(c, a, o)) },
      spawns,
    };
  }
  const baseEnv = () => ({ PATH: `${process.execPath.replace(/\/node$/, "")}:/usr/bin:/bin`, HOME: tempDir("zmc-home-") });

  it("returns models, modes, efforts and the current values, per backend", async () => {
    const dump = join(tempDir(), "dump.json");
    const { spawner } = counting({ ...baseEnv(), MOCK_VARIANT: "codex", MOCK_DUMP: dump });
    const rt = createRuntime({ spawner, bridgeDir: fakeBridgeDir(), nodePath: process.execPath });
    const c = await rt.catalog("codex");
    assert.deepEqual(ids(c.models), ["gpt-6-sol", "gpt-6-luna", "gpt-5.5"]);
    assert.deepEqual(ids(c.modes), ["read-only", "agent"]);
    assert.deepEqual(ids(c.efforts), ["low", "medium", "high", "max"]);
    assert.deepEqual([c.model, c.mode, c.effort], ["gpt-6-sol", "agent", "medium"]);
  });

  it("is cached, shared by concurrent callers, leaves no process or temp dir, and a failure is not cached", async () => {
    const dump = join(tempDir(), "dump.json");
    const env: Record<string, string> = { ...baseEnv(), MOCK_VARIANT: "claude", MOCK_DUMP: dump, MOCK_INIT: "hang" };
    const { spawner, spawns } = counting(env);
    const rt = createRuntime({ spawner, bridgeDir: fakeBridgeDir(), nodePath: process.execPath, initTimeoutMs: 300 });

    // failure first: not remembered
    const [f1, f2] = await Promise.allSettled([rt.catalog("claude-code"), rt.catalog("claude-code")]);
    assert.equal(f1.status, "rejected");
    assert.equal(f2.status, "rejected");
    assert.equal(spawns.length, 1, "concurrent callers share one bridge");
    const hung = (JSON.parse(readFileSync(dump, "utf8")) as { pid: number }).pid;
    await waitFor(() => !alive(hung), 3000, "failed probe's bridge to be gone");
    assert.ok(!existsSync(spawns[0]!.cwd!), "temp workspace removed after a failure");

    delete env["MOCK_INIT"];
    const [a, b] = await Promise.all([rt.catalog("claude-code"), rt.catalog("claude-code")]);
    assert.equal(spawns.length, 2, "the failure was retried, once, for both callers");
    assert.strictEqual(a, b);
    assert.deepEqual(ids(a.models), ["opus", "sonnet", "haiku"]);
    assert.deepEqual(ids(a.efforts), ["low", "medium", "high", "xhigh", "max"]);
    assert.deepEqual([a.model, a.mode, a.effort], ["opus", "auto", "medium"]);

    const pid = (JSON.parse(readFileSync(dump, "utf8")) as { pid: number }).pid;
    assert.equal(alive(pid), false, "no process left behind");
    assert.ok(!existsSync(spawns[1]!.cwd!), "temp workspace removed");

    const again = await rt.catalog("claude-code");
    assert.strictEqual(again, a);
    assert.equal(spawns.length, 2, "cached: no new bridge");
  });
});
