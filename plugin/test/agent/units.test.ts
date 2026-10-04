import assert from "node:assert/strict";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { after, describe, it, test } from "node:test";
import { BACKENDS } from "../../src/agent/backends.ts";
import { buildBrief, withBrief } from "../../src/agent/brief.ts";
import { ensureBridge, locateBridge } from "../../src/agent/bridges.ts";
import { cleanEnv, stripApiKeys } from "../../src/agent/env.ts";
import { CONTEXT_TAG, createRuntime, findBinary, prepareWorkspace } from "../../src/agent/index.ts";
import { JsonRpcPeer } from "../../src/agent/jsonrpc.ts";
import type { Proc, Spawner } from "../../src/types.ts";
import { cleanupTemp, fakeBin, fakeBridgeDir, hermeticSpawner, tempDir } from "./helpers.ts";

after(cleanupTemp);

describe("jsonrpc", () => {
  function fakeProc(): { proc: Proc; written: string[]; feed: (line: string) => void; die: (code: number | null) => void } {
    const written: string[] = [];
    let onLine: (l: string) => void = () => {};
    let die!: (code: number | null) => void;
    const exited = new Promise<number | null>((r) => (die = r));
    const proc: Proc = { pid: 1, write: (d) => void written.push(d), onStdoutLine: (cb) => (onLine = cb), onStderr: () => {}, exited, kill: async () => {} };
    return { proc, written, feed: (l) => onLine(l), die };
  }

  it("matches responses to requests by id (as strings) and rejects on an error frame", async () => {
    const f = fakeProc();
    const peer = new JsonRpcPeer(f.proc);
    const a = peer.request("a");
    const b = peer.request("b");
    const ids = f.written.map((w) => JSON.parse(w).id);
    f.feed(JSON.stringify({ jsonrpc: "2.0", id: String(ids[1]), error: { code: -32601, message: "nope" } }));
    f.feed(JSON.stringify({ jsonrpc: "2.0", id: ids[0], result: { ok: 1 } }));
    assert.deepEqual(await a, { ok: 1 });
    await assert.rejects(b, /nope/);
  });

  it("answers every inbound request, with method-not-found when nobody handles it", async () => {
    const f = fakeProc();
    new JsonRpcPeer(f.proc);
    f.feed(JSON.stringify({ jsonrpc: "2.0", id: 7, method: "fs/read_text_file", params: {} }));
    await new Promise((r) => setTimeout(r, 10));
    const reply = JSON.parse(f.written[0]!);
    assert.equal(reply.id, 7);
    assert.equal(reply.error.code, -32601);
  });

  it("once the process is gone, pending requests reject and new ones reject at once", async () => {
    const f = fakeProc();
    const peer = new JsonRpcPeer(f.proc);
    const pending = peer.request("x");
    f.die(3);
    await assert.rejects(pending, /code 3/);
    await assert.rejects(peer.request("y"), /code 3/);
    assert.ok(peer.dead);
  });

  it("ignores garbage lines", async () => {
    const f = fakeProc();
    const peer = new JsonRpcPeer(f.proc);
    const p = peer.request("x");
    f.feed("not json at all");
    f.feed("");
    f.feed(JSON.stringify({ jsonrpc: "2.0", id: JSON.parse(f.written[0]!).id, result: 1 }));
    assert.equal(await p, 1);
  });
});

describe("brief", () => {
  it("is short and says the four things", () => {
    const b = buildBrief();
    assert.ok(b.split(/\s+/).length < 250, `${b.split(/\s+/).length} words`);
    assert.match(b, /Zotero/);
    assert.match(b, /zotero-cli/);
    assert.ok(b.includes(`<${CONTEXT_TAG}>`));
    assert.ok(b.includes("zotero://open-pdf/library/items/ATTKEY?page=8"));
    assert.ok(b.includes("zotero://open-pdf/groups/"));
    assert.ok(b.includes("zotero://select/library/items/"));
  });
  it("withBrief wraps the brief ahead of the text", () => {
    assert.equal(withBrief("B", "hi"), "<zotero-panel-brief>\nB\n</zotero-panel-brief>\n\nhi");
  });
});

describe("env", () => {
  it("cleanEnv drops parent-session variables and keeps the user's own config", () => {
    const out = cleanEnv({ CLAUDECODE: "1", CLAUDE_CODE_SESSION_ID: "x", CLAUDE_CODE_ENTRYPOINT: "cli", CLAUDE_CODE_SSE_PORT: "1", CLAUDE_PID: "1", CLAUDE_EFFORT: "x", CLAUDE_CODE_USE_VERTEX: "1", HOME: "/h", PATH: "/bin" });
    assert.deepEqual(out, { CLAUDE_CODE_USE_VERTEX: "1", HOME: "/h", PATH: "/bin" });
  });
  it("findBinary resolves through PATH, and probes extra dirs", async () => {
    const bin = fakeBin({ mytool: "exit 0" });
    const home = fakeBin({ other: "exit 0" });
    const sp = hermeticSpawner({}, [bin]);
    const env = await sp.baseEnv();
    assert.equal(await findBinary(sp, env, "mytool"), join(bin, "mytool"));
    assert.equal(await findBinary(sp, env, "other"), null);
    assert.equal(await findBinary(sp, env, "other", [home]), join(home, "other"));
    assert.equal(await findBinary(sp, env, "definitely-not-a-binary-xyz"), null);
  });
});

/** A Spawner that records every run() and plays a tiny fake npm / node -e. */
function scriptedSpawner(state: { installed: boolean; npmCode?: number; npmStderr?: string; version?: string; npmMissing?: boolean }) {
  const runs: { command: string; args: string[] }[] = [];
  const spawner: Spawner = {
    baseEnv: async () => ({ PATH: "/fake/bin", HOME: "/fake/home" }),
    spawn: async () => {
      throw new Error("not used");
    },
    run: async (command, args, opts) => {
      runs.push({ command, args });
      if (command === "/bin/sh") {
        const name = opts.env["NAME"];
        return name === "npm" && state.npmMissing ? { code: 1, stdout: "", stderr: "" } : { code: 0, stdout: `/fake/bin/${name}\n`, stderr: "" };
      }
      if (args[0] === "-e") {
        return { code: 0, stdout: state.installed ? JSON.stringify({ version: state.version ?? BACKENDS["claude-code"].version, entry: "/fake/dir/node_modules/p/dist/index.js" }) + "\n" : "", stderr: "" };
      }
      if (command === "/fake/bin/npm") {
        if (state.npmCode) return { code: state.npmCode, stdout: "", stderr: state.npmStderr ?? "boom" };
        state.installed = true;
        return { code: 0, stdout: "", stderr: "" };
      }
      return { code: 0, stdout: "", stderr: "" };
    },
  };
  return { spawner, runs, installs: () => runs.filter((r) => r.command.endsWith("npm") && r.args[0] === "install") };
}

describe("bridges", () => {
  const spec = BACKENDS["claude-code"];
  const base = { env: { PATH: "/fake/bin" }, node: "/fake/bin/node", bridgeDir: "/fake/dir", spec };

  it("installs once with the pinned version and the login PATH, then is idempotent", async () => {
    const s = scriptedSpawner({ installed: false });
    const a = await ensureBridge({ ...base, spawner: s.spawner });
    const b = await ensureBridge({ ...base, spawner: s.spawner });
    assert.equal(a.entry, "/fake/dir/node_modules/p/dist/index.js");
    assert.deepEqual(a, b);
    assert.equal(s.installs().length, 1);
    assert.deepEqual(s.installs()[0]!.args.slice(0, 4), ["install", "--prefix", "/fake/dir", "@agentclientprotocol/claude-agent-acp@0.85.1"]);
  });

  it("concurrent callers share one install", async () => {
    const s = scriptedSpawner({ installed: false });
    await Promise.all([ensureBridge({ ...base, bridgeDir: "/fake/dir2", spawner: s.spawner }), ensureBridge({ ...base, bridgeDir: "/fake/dir2", spawner: s.spawner })]);
    assert.equal(s.installs().length, 1);
  });

  it("does nothing when the pinned version is already there", async () => {
    const s = scriptedSpawner({ installed: true });
    await ensureBridge({ ...base, bridgeDir: "/fake/dir3", spawner: s.spawner });
    assert.equal(s.installs().length, 0);
  });

  it("reinstalls when another version is installed", async () => {
    const s = scriptedSpawner({ installed: true, version: "0.1.0" });
    assert.equal(await locateBridge(s.spawner, {}, "/fake/bin/node", "/fake/dir", spec), null);
    s.runs.length = 0;
    // the fake keeps reporting 0.1.0, so the post-install check must fail loudly
    await assert.rejects(ensureBridge({ ...base, bridgeDir: "/fake/dir4", spawner: s.spawner }), /entry point was not found/);
    assert.equal(s.installs().length, 1);
  });

  it("surfaces npm's failure and a missing npm", async () => {
    await assert.rejects(ensureBridge({ ...base, bridgeDir: "/fake/dir5", spawner: scriptedSpawner({ installed: false, npmCode: 1, npmStderr: "ENOTFOUND registry" }).spawner }), /exit 1[\s\S]*ENOTFOUND/);
    await assert.rejects(ensureBridge({ ...base, bridgeDir: "/fake/dir6", spawner: scriptedSpawner({ installed: false, npmMissing: true }).spawner }), /npm was not found/);
  });

  it("locates a real on-disk install with the real node", async () => {
    const dir = fakeBridgeDir(["claude-code"]);
    const sp = hermeticSpawner();
    const found = await locateBridge(sp, await sp.baseEnv(), process.execPath, dir, spec);
    assert.ok(found && existsSync(found.entry));
    assert.equal(await locateBridge(sp, await sp.baseEnv(), process.execPath, dir, BACKENDS.codex), null);
  });

  it("pins are the versions measured", () => {
    assert.deepEqual(
      Object.values(BACKENDS).map((b) => `${b.pkg}@${b.version}`),
      ["@agentclientprotocol/claude-agent-acp@0.85.1", "@agentclientprotocol/codex-acp@2.1.1", "pi-acp@0.0.34"],
    );
  });
});

describe("detect()", () => {
  const AUTH = JSON.stringify({ loggedIn: true, authMethod: "claude.ai", subscriptionType: "max" });

  it("reports availability, accounts and human reasons", async () => {
    const bin = fakeBin({
      claude: `[ "$1" = auth ] && echo '${AUTH}'`,
      codex: `echo "Not logged in" ; exit 1`,
      npm: "exit 0",
    });
    const rt = createRuntime({ spawner: hermeticSpawner({}, [bin]), bridgeDir: fakeBridgeDir(["claude-code"]), nodePath: process.execPath });
    const st = Object.fromEntries((await rt.detect()).map((s) => [s.id, s]));
    assert.equal(st["claude-code"]!.available, true);
    assert.equal(st["claude-code"]!.account, "Claude Max");
    assert.equal(st["codex"]!.available, false);
    assert.match(st["codex"]!.reason!, /not signed in/i);
    assert.equal(st["pi"]!.available, false);
    assert.match(st["pi"]!.reason!, /Install pi/);
  });

  it("claude missing, and a logged-out claude", async () => {
    const none = createRuntime({ spawner: hermeticSpawner(), bridgeDir: fakeBridgeDir(), nodePath: process.execPath });
    const a = (await none.detect()).find((s) => s.id === "claude-code")!;
    assert.equal(a.available, false);
    assert.match(a.reason!, /Install Claude Code/);
    const bin = fakeBin({ claude: `echo '{"loggedIn":false}'; exit 1` });
    const out = createRuntime({ spawner: hermeticSpawner({}, [bin]), bridgeDir: fakeBridgeDir(), nodePath: process.execPath });
    const b = (await out.detect()).find((s) => s.id === "claude-code")!;
    assert.equal(b.available, false);
    assert.match(b.reason!, /not signed in/i);
  });

  it("no node: every backend is unavailable with the same explanation", async () => {
    const rt = createRuntime({ spawner: hermeticSpawner({ PATH: "/nonexistent" }), bridgeDir: tempDir() });
    const all = await rt.detect();
    assert.equal(all.length, 3);
    assert.ok(all.every((s) => !s.available && /node was not found/.test(s.reason!)));
  });

  it("an uninstalled bridge needs npm", async () => {
    const bin = fakeBin({ claude: `echo '${AUTH}'` });
    const sp = hermeticSpawner({ PATH: `${bin}:/usr/bin:/bin` });
    // node is given by path, npm is not on this PATH
    const rt = createRuntime({ spawner: sp, bridgeDir: tempDir(), nodePath: process.execPath });
    const c = (await rt.detect()).find((s) => s.id === "claude-code")!;
    // npm sits next to the real node, which is put first on PATH for the lookup
    assert.equal(c.available, true, c.reason ?? "");
  });
});

describe("workspace", () => {
  function recording(results: (cmd: string, args: string[], env: Record<string, string>) => { code: number; stdout: string; stderr?: string }) {
    const calls: { command: string; args: string[]; cwd?: string }[] = [];
    const spawner: Spawner = {
      baseEnv: async () => ({ PATH: "/fake/bin" }),
      spawn: async () => {
        throw new Error("no");
      },
      run: async (command, args, o) => {
        calls.push({ command, args, ...(o.cwd ? { cwd: o.cwd } : {}) });
        const r = results(command, args, o.env);
        return { code: r.code, stdout: r.stdout, stderr: r.stderr ?? "" };
      },
    };
    return { spawner, calls };
  }

  it("makes the dir, then runs zotero-mcp install-skill for claude and agents with --root", async () => {
    const { spawner, calls } = recording((cmd, args) => {
      if (cmd === "/bin/sh" && args[1]!.includes("command -v")) return { code: 0, stdout: "/fake/bin/zotero-mcp\n" };
      if (cmd === "/bin/sh") return { code: 0, stdout: "" };
      return { code: 0, stdout: "Installed the zotero-cli skill into 2 target(s)" };
    });
    const dir = await prepareWorkspace(spawner, "/data/ws");
    assert.equal(dir, "/data/ws");
    assert.deepEqual(calls[0]!.args.slice(-2), ["sh", "/data/ws"]);
    assert.ok(calls[0]!.args[1]!.includes("mkdir -p"));
    const install = calls.find((c) => c.command === "/fake/bin/zotero-mcp")!;
    assert.deepEqual(install.args, ["install-skill", "--target", "claude", "--target", "agents", "--root", "/data/ws"]);
  });

  it("the panel's own folder gets --force, so a newer skill replaces an old copy; any other folder does not", async () => {
    const mk = () => recording((cmd, args) => (cmd === "/bin/sh" && args[1]!.includes("command -v") ? { code: 0, stdout: "/fake/bin/zotero-mcp\n" } : { code: 0, stdout: "" }));
    const own = mk();
    await prepareWorkspace(own.spawner, "/data/ws", { refresh: true });
    assert.ok(own.calls.find((c) => c.args.includes("install-skill"))!.args.includes("--force"));
    const theirs = mk();
    await prepareWorkspace(theirs.spawner, "/data/ws");
    assert.ok(!theirs.calls.find((c) => c.args.includes("install-skill"))!.args.includes("--force"));
  });

  it("prefers the zotero-mcp beside the given zotero-cli, falls back to uv tool run", async () => {
    const bin = fakeBin({ "zotero-mcp": "exit 0", "zotero-cli": "exit 0" });
    const { spawner, calls } = recording((cmd, args) => {
      if (cmd === "/bin/sh" && args[1]!.includes('[ -x "$1" ]')) return { code: existsSync(args[3]!) ? 0 : 1, stdout: existsSync(args[3]!) ? "ok\n" : "" };
      if (cmd === "/bin/sh" && args[1]!.includes("command -v")) return { code: 1, stdout: "" };
      if (cmd === "/bin/sh") return { code: 0, stdout: "" };
      return { code: 0, stdout: "Installed the zotero-cli skill" };
    });
    await prepareWorkspace(spawner, "/data/ws", { zoteroCli: join(bin, "zotero-cli") });
    assert.equal(calls.find((c) => c.args.includes("install-skill"))!.command, join(bin, "zotero-mcp"));

    const second = recording((cmd, args, env) => {
      if (cmd === "/bin/sh" && args[1]!.includes("command -v")) return env["NAME"] === "uv" ? { code: 0, stdout: "/fake/bin/uv\n" } : { code: 1, stdout: "" };
      return { code: 0, stdout: "Installed the zotero-cli skill" };
    });
    await prepareWorkspace(second.spawner, "/data/ws");
    const uvCall = second.calls.find((c) => c.args.includes("install-skill"))!;
    assert.deepEqual(uvCall.args.slice(0, 4), ["tool", "run", "--from", "zotero-mcp-server"]);
  });

  it("an absent installer is not fatal", async () => {
    const { spawner } = recording((cmd, args) => {
      if (cmd === "/bin/sh" && args[1]!.includes("command -v")) return { code: 1, stdout: "" };
      return { code: 0, stdout: "" };
    });
    assert.equal(await prepareWorkspace(spawner, "/data/ws"), "/data/ws");
  });

  it("a failing mkdir throws", async () => {
    const { spawner } = recording(() => ({ code: 1, stdout: "", stderr: "Permission denied" }));
    await assert.rejects(prepareWorkspace(spawner, "/nope/ws"), /could not create the workspace/);
  });

  // The real installer, into a temp dir only (project-scope targets; nothing under ~).
  it("with the real zotero-mcp: installs, is idempotent, never clobbers an edited skill", async (t) => {
    const loginish = { PATH: `${process.env["HOME"]}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin` };
    const real = await findBinary(hermeticSpawner(loginish), loginish, "zotero-mcp");
    if (!real) return t.skip("zotero-mcp not installed");
    const sp = hermeticSpawner({}, [real.replace(/\/zotero-mcp$/, "")]);
    const dir = join(tempDir("zmc-realws-"), "ws");
    await prepareWorkspace(sp, dir);
    const skill = join(dir, ".claude", "skills", "zotero-cli", "SKILL.md");
    assert.ok(existsSync(skill), "claude skill installed");
    assert.ok(existsSync(join(dir, "AGENTS.md")), "AGENTS.md pointer installed");
    const before = readFileSync(skill, "utf8");
    await prepareWorkspace(sp, dir);
    assert.equal(readFileSync(skill, "utf8"), before);
    writeFileSync(skill, before + "\nMY OWN NOTE\n");
    await prepareWorkspace(sp, dir);
    assert.ok(readFileSync(skill, "utf8").includes("MY OWN NOTE"), "an edited skill is left alone");
  });
});

test("stripApiKeys: a subscription session loses the backend's API-key variables and nothing else", () => {
  const login = { PATH: "/bin", ANTHROPIC_API_KEY: "sk-a", ANTHROPIC_AUTH_TOKEN: "t", OPENAI_API_KEY: "sk-o", CODEX_API_KEY: "c", ANTHROPIC_BASE_URL: "https://proxy" };
  assert.deepEqual(stripApiKeys(login, BACKENDS["claude-code"].apiKeyVars), { PATH: "/bin", OPENAI_API_KEY: "sk-o", CODEX_API_KEY: "c", ANTHROPIC_BASE_URL: "https://proxy" });
  assert.deepEqual(stripApiKeys(login, BACKENDS.codex.apiKeyVars), { PATH: "/bin", ANTHROPIC_API_KEY: "sk-a", ANTHROPIC_AUTH_TOKEN: "t", ANTHROPIC_BASE_URL: "https://proxy" });
  assert.deepEqual(stripApiKeys(login, BACKENDS.pi.apiKeyVars), login, "pi runs on provider keys; none are removed");
  assert.notEqual(stripApiKeys(login, BACKENDS.pi.apiKeyVars), login, "returns a copy");
});
