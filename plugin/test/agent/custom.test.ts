import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { createRuntime } from "../../src/agent/runtime.ts";
import { readCustomAgent } from "../../src/custom-agent.ts";
import { resumeCommand } from "../../src/agent/backends.ts";
import { createNodeSpawner } from "../node-spawner.ts";

const mock = fileURLToPath(new URL("../mock-agent.mjs", import.meta.url));

for (const variant of ["claude", "codex"]) test(`a custom ${variant} endpoint preserves argv/env and uses advertised models and modes`, async () => {
  const cwd = mkdtempSync(join(tmpdir(), "zmc-custom-"));
  const dump = join(cwd, "dump.json");
  const config = { label: "My ACP agent", command: process.execPath, args: [mock, "a b", "literal&argument"], env: { MOCK_DUMP: dump, MOCK_VARIANT: variant } };
  const calls: string[] = [];
  const native = createNodeSpawner({ baseEnv: { HOME: cwd, PATH: "", ...(process.env.SystemRoot ? { SystemRoot: process.env.SystemRoot } : {}) } });
  const spawner = { ...native, run: async (...args: Parameters<typeof native.run>) => { calls.push(args[0]); return native.run(...args); } };
  const runtime = createRuntime({ spawner, bridgeDir: join(cwd, "never-install"), customAgent: () => config });
  try {
    const session = await runtime.start({ backend: "custom", cwd, brief: "Use zotero-cli for the library." });
    try {
      assert.equal(session.backend, "custom");
      assert.ok(session.models().length);
      assert.ok(session.modes().length);
      const model = session.models()[1]!;
      await session.setModel(model.id);
      assert.equal(session.currentModel(), model.id);
      const mode = session.modes().at(-1)!;
      await session.setMode(mode.id);
      assert.equal(session.currentMode(), mode.id);
      const events: string[] = [];
      session.on((event) => { if (event.t === "text") events.push(event.delta); });
      await session.prompt({ text: "SCENARIO:brief" });
      assert.match(events.join(""), /Use zotero-cli for the library\./);
      const seen = JSON.parse(readFileSync(dump, "utf8"));
      assert.deepEqual(seen.argv.slice(-2), ["a b", "literal&argument"]);
      assert.equal(seen.cwd, realpathSync(cwd)); // macOS: the tmpdir /var is /private/var to the child
      assert.equal(seen.env.MOCK_VARIANT, variant);
      assert.deepEqual(calls, [], "direct launch does not probe a vendor, Node or npm");
    } finally { await session.close(); }
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("an unconfigured custom backend explains how to configure it", async () => {
  const spawner = { baseEnv: async () => ({}), spawn: async () => { throw new Error("must not spawn"); }, run: async () => { throw new Error("must not run"); } };
  const runtime = createRuntime({ spawner, bridgeDir: "/unused", customAgent: () => undefined });
  await assert.rejects(runtime.start({ backend: "custom", cwd: "/unused", brief: "" }), /custom agent.*command/i);
});

test("custom arguments and environment are validated before launch", () => {
  const base = { label: " Agent ", command: " agent ", args: ["--acp", "", "a b"], env: { MODE: "" } };
  assert.deepEqual(readCustomAgent(base), { ...base, label: "Agent", command: "agent" });
  for (const invalid of [
    { command: "" }, { command: "agent\0" }, { command: "agent", args: "--acp" },
    { command: "agent", args: [3] }, { command: "agent", args: ["x\0"] },
    { command: "agent", env: [] }, { command: "agent", env: { A: 3 } },
    { command: "agent", env: { "bad=name": "x" } }, { command: "agent", env: { A: "x\0" } },
  ]) assert.throws(() => readCustomAgent(invalid));
  assert.deepEqual(readCustomAgent({ command: "agent" }), { label: "", command: "agent", args: [], env: {} }); // the name falls back where it is drawn, in the panel's language
});

test("custom agents do not get unverified translation restrictions or resume commands", async () => {
  let spawned = false;
  const spawner = { baseEnv: async () => ({}), spawn: async () => { spawned = true; throw new Error("unexpected"); }, run: async () => { throw new Error("unexpected"); } };
  const runtime = createRuntime({ spawner, bridgeDir: "/unused" });
  await assert.rejects(runtime.start({ backend: "custom", cwd: "/unused", brief: "", locked: true }), /Translate is not available/);
  await assert.rejects(runtime.start({ backend: "custom", cwd: "/unused", brief: "", resumeSessionId: "old" }), /Start a new chat/);
  assert.equal(resumeCommand("custom", "/unused", "old"), null);
  assert.equal(spawned, false);
});

test("custom detection checks executables without launching an agent or probing its login", async () => {
  const calls: string[][] = [];
  const spawner = {
    windows: true, baseEnv: async () => ({ SystemRoot: "C:\\Windows", PATH: "" }),
    spawn: async () => { throw new Error("detection must not spawn an agent"); },
    run: async (command: string, args: string[]) => {
      calls.push([command, ...args]);
      return args[0] === "my-acp" ? { code: 0, stdout: "C:\\Agents\\my-acp.exe\r\n", stderr: "" } : { code: 1, stdout: "", stderr: "" };
    },
  };
  let config = { label: "Native ACP", command: "my-acp", args: ["--acp"], env: {} };
  const runtime = createRuntime({ spawner, bridgeDir: "C:\\unused", customAgent: () => config });
  assert.deepEqual((await runtime.detect()).find(x => x.id === "custom"), { id: "custom", label: "Native ACP", available: true });
  config = { ...config, command: "C:\\missing.exe" };
  assert.equal((await runtime.detect()).find(x => x.id === "custom")?.available, false);
  assert.ok(calls.every(([command]) => command?.endsWith("where.exe")));
});

test("a failed custom handshake closes its process", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "zmc-custom-failure-"));
  const native = createNodeSpawner({ baseEnv: { HOME: cwd, PATH: "" } });
  let closed = false;
  const spawner = { ...native, spawn: async (...args: Parameters<typeof native.spawn>) => {
    const proc = await native.spawn(...args);
    return { ...proc, kill: async () => { closed = true; await proc.kill(); } };
  } };
  const runtime = createRuntime({ spawner, bridgeDir: cwd, initTimeoutMs: 40,
    customAgent: () => ({ label: "Hanging agent", command: process.execPath, args: [mock], env: { MOCK_INIT: "hang" } }) });
  try { await assert.rejects(runtime.start({ backend: "custom", cwd, brief: "" }), /did not answer initialize/i); assert.equal(closed, true); }
  finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("an ACP endpoint without optional offers has no invented model, mode or effort", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "zmc-custom-empty-"));
  const spawner = createNodeSpawner({ baseEnv: { HOME: cwd, PATH: "" } });
  const script = `require('node:readline').createInterface({input:process.stdin}).on('line', line => {
    const m=JSON.parse(line); if(m.id===undefined)return;
    const result=m.method==='initialize'?{protocolVersion:1,agentCapabilities:{}}:{sessionId:'empty'};
    console.log(JSON.stringify({jsonrpc:'2.0',id:m.id,result}));
  });`;
  const runtime = createRuntime({ spawner, bridgeDir: cwd, customAgent: () => ({ label: "Empty ACP", command: process.execPath, args: ["-e", script], env: {} }) });
  try {
    const session = await runtime.start({ backend: "custom", cwd, brief: "" });
    try { assert.deepEqual([session.models(), session.modes(), session.efforts()], [[], [], []]); assert.equal(session.supportsImages, false); }
    finally { await session.close(); }
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});
