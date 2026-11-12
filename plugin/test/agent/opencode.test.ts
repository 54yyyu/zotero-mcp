// OpenCode: a CLI that is its own ACP agent (`opencode acp`), so no bridge, no npm and no Node. The fake `opencode` on PATH
// runs test/mock-agent.mjs as MOCK_VARIANT=opencode, the shape measured on opencode 1.19.5 (backends.ts).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { after, describe, it } from "node:test";
import { BACKENDS, BACKEND_IDS, findCli, isNative, probeLogin, resumeCommand } from "../../src/agent/backends.ts";
import { OPENCODE_LOCKED, createRuntime } from "../../src/agent/runtime.ts";
import { shimTarget } from "../../src/agent/windows.ts";
import type { Spawner } from "../../src/types.ts";
import { createNodeSpawner } from "../node-spawner.ts";
import { cleanupTemp, collect, fakeBin, tempDir, textOf } from "./helpers.ts";

after(cleanupTemp);

const MOCK = fileURLToPath(new URL("../mock-agent.mjs", import.meta.url));

/** A spawner whose PATH holds only a fake `opencode` (and the system dirs): no node, no npm. Every `run` is recorded. */
function setup(env: Record<string, string> = {}) {
  const dir = tempDir();
  const dump = `${dir}/dump.json`;
  const bin = fakeBin({ opencode: `exec ${JSON.stringify(process.execPath)} ${JSON.stringify(MOCK)} "$@"` });
  const native = createNodeSpawner({ graceMs: 500, baseEnv: { PATH: `${bin}:/usr/bin:/bin`, HOME: tempDir("zmc-home-"), MOCK_VARIANT: "opencode", MOCK_DUMP: dump, ...env } });
  const ran: string[][] = [];
  const spawner: Spawner = { ...native, run: async (c, a, o) => { ran.push([c, ...a]); return native.run(c, a, o); } };
  const runtime = createRuntime({ spawner, bridgeDir: `${dir}/never-installed` });
  return { dir, dump, bin, runtime, ran, spawner };
}

describe("opencode backend", () => {
  it("is a built-in backend run as `opencode acp`, with no bridge package", () => {
    assert.ok(BACKEND_IDS.includes("opencode"));
    const spec = BACKENDS.opencode;
    assert.ok(isNative(spec) && !("pkg" in spec));
    assert.deepEqual(spec.acpArgs, ["acp"]);
    assert.deepEqual([spec.modelVia, spec.effortVia, spec.briefVia, spec.permissionModes, spec.compacts, spec.apiKeyVars], ["config_option", "config_option", "first-prompt", true, false, []]);
  });

  it("detect: ready when the CLI is on PATH, even with no Node; else the install hint", async () => {
    const { runtime, ran } = setup();
    const st = (await runtime.detect()).find((s) => s.id === "opencode");
    assert.deepEqual(st, { id: "opencode", label: "OpenCode", available: true });
    assert.ok(!ran.some(([c, ...a]) => /opencode$/.test(c!) && a[0] === "auth"), "no login probe: OpenCode chats on its free models with no credentials");

    const empty = createRuntime({ spawner: createNodeSpawner({ baseEnv: { PATH: "/usr/bin:/bin", HOME: tempDir() } }), bridgeDir: tempDir() });
    const missing = (await empty.detect()).find((s) => s.id === "opencode");
    assert.equal(missing?.available, false);
    assert.match(missing?.reason ?? "", /npm i -g opencode-ai/);
  });

  it("probeLogin says nothing for OpenCode (it cannot tell signed-out from free models)", async () => {
    const spawner = { baseEnv: async () => ({}), spawn: async () => { throw new Error("no"); }, run: async () => { throw new Error("must not run"); } };
    assert.deepEqual(await probeLogin(spawner, {}, "opencode", "/x/opencode"), {});
  });

  it("starts `opencode acp` directly: models, build/plan modes, effort per model, brief on the first prompt", async () => {
    const { dir, dump, bin, runtime, ran } = setup();
    const s = await runtime.start({ backend: "opencode", cwd: dir, brief: "Use zotero-cli for the library." });
    try {
      const seen = JSON.parse(readFileSync(dump, "utf8"));
      assert.deepEqual(seen.argv, ["acp"]);
      assert.ok(!ran.some(([c]) => /\b(npm|node)$/.test(c!)), "no bridge install, no node lookup: " + JSON.stringify(ran));
      assert.ok(ran.some((r) => r.join(" ").includes("command -v")), "the CLI was looked up on PATH");
      assert.equal(seen.env.OPENCODE_CONFIG_CONTENT, undefined, "a chat is not locked");
      assert.deepEqual(s.models().map((m) => m.id), ["opencode/big-pickle", "opencode/deepseek-v4.1-flash-free"]);
      assert.equal(s.currentModel(), "opencode/big-pickle");
      assert.deepEqual(s.modes().map((m) => m.id), ["build", "plan"]);
      assert.equal(s.currentMode(), "build");
      assert.deepEqual(s.efforts(), [], "big-pickle has no variants");
      assert.equal(s.canCompact, false);

      await s.setModel("opencode/deepseek-v4.1-flash-free");
      assert.equal(s.currentModel(), "opencode/deepseek-v4.1-flash-free");
      assert.deepEqual(s.efforts().map((e) => e.id), ["low", "medium", "high"], "its variants; OpenCode's \"default\" is not a choice");
      await s.setEffort("high");
      assert.equal(s.currentEffort(), "high");
      await s.setMode("plan");
      assert.equal(s.currentMode(), "plan");

      const c = collect(s);
      await s.prompt({ text: "SCENARIO:brief" });
      assert.match(textOf(c.events), /Use zotero-cli for the library\./);
      assert.ok(bin);
    } finally {
      await s.close();
    }
  });

  it("a chat's saved model, mode and effort are applied at start", async () => {
    const { dir, runtime } = setup();
    const s = await runtime.start({ backend: "opencode", cwd: dir, brief: "", model: "opencode/deepseek-v4.1-flash-free", mode: "plan", effort: "medium" });
    try {
      assert.deepEqual([s.currentModel(), s.currentMode(), s.currentEffort()], ["opencode/deepseek-v4.1-flash-free", "plan", "medium"]);
    } finally {
      await s.close();
    }
  });

  it("locked (Translate): every tool but read/glob/grep is denied, nothing outside its folder, no Claude Code files", async () => {
    const { dir, dump, runtime } = setup();
    const s = await runtime.start({ backend: "opencode", cwd: dir, brief: "BRIEF", locked: true });
    await s.close();
    const env = JSON.parse(readFileSync(dump, "utf8")).env;
    assert.equal(env.OPENCODE_CONFIG_CONTENT, OPENCODE_LOCKED);
    const perm = JSON.parse(env.OPENCODE_CONFIG_CONTENT).permission;
    for (const k of ["edit", "bash", "webfetch", "websearch", "task", "skill", "external_directory"]) assert.equal(perm[k], "deny", k);
    for (const k of ["read", "glob", "grep", "*"]) assert.equal(perm[k], undefined, `${k}: denying every tool makes Zen's free models refuse the request`);
    assert.equal(env.OPENCODE_DISABLE_CLAUDE_CODE, "1");
  });

  it("a missing CLI at start says how to install it", async () => {
    const runtime = createRuntime({ spawner: createNodeSpawner({ baseEnv: { PATH: "/usr/bin:/bin", HOME: tempDir() } }), bridgeDir: tempDir() });
    await assert.rejects(runtime.start({ backend: "opencode", cwd: tempDir(), brief: "" }), /npm i -g opencode-ai/);
  });

  it("the terminal resume command is opencode --session", () => {
    assert.equal(resumeCommand("opencode", "/Users/o'neil/Zotero Agent", "ses_abc"), `cd '/Users/o'\\''neil/Zotero Agent' && opencode --session ses_abc`);
  });

  it("Windows: where.exe finds npm's opencode.cmd, and its shim is read through to the real opencode.exe", async () => {
    const npm = "C:\\Users\\me\\AppData\\Roaming\\npm";
    const spawner: Spawner = {
      windows: true,
      baseEnv: async () => ({}),
      spawn: async () => { throw new Error("not used"); },
      run: async (command, args) => ({ code: 0, stdout: command.endsWith("where.exe") && args[0] === "opencode" ? `${npm}\\opencode\r\n${npm}\\opencode.cmd\r\n${npm}\\opencode.ps1\r\n` : "", stderr: "" }),
    };
    assert.equal(await findCli(spawner, { PATH: npm, SystemRoot: "C:\\Windows" }, BACKENDS.opencode), `${npm}\\opencode.cmd`);
    // What npm (cmd-shim) writes for opencode-ai's bin, which is the native binary itself (bin/opencode.exe), not a JS file.
    const shim = '@ECHO off\r\nGOTO start\r\n:find_dp0\r\nSET dp0=%~dp0\r\nEXIT /b\r\n:start\r\nSETLOCAL\r\nCALL :find_dp0\r\n"%dp0%\\node_modules\\opencode-ai\\bin\\opencode.exe"   %*\r\n';
    assert.equal(shimTarget(shim), "node_modules\\opencode-ai\\bin\\opencode.exe");
  });
});
