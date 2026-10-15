// Windows: no /bin/sh anywhere on the detection and launch path (#700). A scripted Spawner plays where.exe and node.
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { BACKENDS } from "../../src/agent/backends.ts";
import { dirnameOf, findBinary, withPathFirst } from "../../src/agent/env.ts";
import { createRuntime, prepareWorkspace } from "../../src/agent/index.ts";
import { cmdLine, shimTarget } from "../../src/agent/windows.ts";
import type { Spawner } from "../../src/types.ts";

const ENV = { PATH: "C:\\Windows\\System32;C:\\Program Files\\nodejs", HOME: "C:\\Users\\me", SystemRoot: "C:\\Windows" };
const NODE = "C:\\Program Files\\nodejs\\node.exe";
const NPM_DIR = "C:\\Users\\me\\AppData\\Roaming\\npm";
const WHERE = "C:\\Windows\\System32\\where.exe";

/** What where.exe prints for each name, given the PATH it is run with (extension-less npm shim first, as npm leaves it). */
const FOUND: Record<string, string[]> = {
  node: [NODE],
  npm: ["C:\\Program Files\\nodejs\\npm", "C:\\Program Files\\nodejs\\npm.cmd"],
  codex: [`${NPM_DIR}\\codex`, `${NPM_DIR}\\codex.cmd`, `${NPM_DIR}\\codex.ps1`],
};

function windows(extra: (command: string, args: string[]) => { code: number; stdout: string } | undefined = () => undefined) {
  const calls: { command: string; args: string[]; env: Record<string, string> }[] = [];
  const spawner: Spawner = {
    windows: true,
    baseEnv: async () => ENV,
    spawn: async () => {
      throw new Error("not used");
    },
    run: async (command, args, o) => {
      calls.push({ command, args, env: o.env });
      if (command === "/bin/sh") throw new Error("File at path \"/bin/sh\" does not exist, or is not executable.");
      const mine = extra(command, args);
      if (mine) return { ...mine, stderr: "" };
      if (command === WHERE) {
        const hits = FOUND[args[0]!] ?? [];
        return { code: hits.length ? 0 : 1, stdout: hits.map((h) => h + "\r\n").join(""), stderr: "" };
      }
      return { code: 0, stdout: "", stderr: "" };
    },
  };
  return { spawner, calls };
}

describe("windows: finding binaries", () => {
  it("findBinary asks where.exe and takes the first real executable, not npm's extension-less shim", async () => {
    const { spawner, calls } = windows();
    assert.equal(await findBinary(spawner, ENV, "codex"), `${NPM_DIR}\\codex.cmd`);
    assert.equal(await findBinary(spawner, ENV, "node"), NODE);
    assert.equal(calls[0]!.command, WHERE);
    assert.deepEqual(calls[0]!.args, ["codex"]);
  });

  it("extra directories are searched too, with backslashes; a miss is null", async () => {
    const { spawner, calls } = windows();
    assert.equal(await findBinary(spawner, ENV, "claude", ["C:\\Users\\me/.local/bin"]), null);
    assert.equal(calls[0]!.env["PATH"], `${ENV.PATH};C:\\Users\\me\\.local\\bin`);
  });

  it("a Spawner that cannot start where.exe gives null, not a throw", async () => {
    const { spawner } = windows();
    spawner.run = async () => {
      throw new Error("nope");
    };
    assert.equal(await findBinary(spawner, ENV, "node"), null);
  });

  it("Windows paths: dirnameOf and PATH ordering", () => {
    assert.equal(dirnameOf(NODE), "C:\\Program Files\\nodejs");
    assert.equal(dirnameOf("/usr/bin/node"), "/usr/bin");
    assert.equal(withPathFirst({ PATH: "C:\\a;C:\\b" }, "C:\\b")["PATH"], "C:\\b;C:\\a");
    assert.equal(withPathFirst({ PATH: "/a:/b" }, "/c")["PATH"], "/c:/a:/b");
  });
});

describe("windows: npm .cmd shims", () => {
  const CODEX = [
    "@ECHO off",
    "GOTO start",
    ":find_dp0",
    "SET dp0=%~dp0",
    "EXIT /b",
    ":start",
    "SETLOCAL",
    "CALL :find_dp0",
    'IF EXIST "%dp0%\\node.exe" (',
    '  SET "_prog=%dp0%\\node.exe"',
    ") ELSE (",
    '  SET "_prog=node"',
    "  SET PATHEXT=%PATHEXT:;.JS;=;%",
    ")",
    'endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "%_prog%"  "%dp0%\\node_modules\\@openai\\codex\\bin\\codex.js" %*',
  ].join("\r\n");

  it("shimTarget finds the script behind the shim, and skips the node.exe the shim probes", () => {
    assert.equal(shimTarget(CODEX), "node_modules\\@openai\\codex\\bin\\codex.js");
    assert.equal(shimTarget('@"%dp0%\\node_modules\\@anthropic-ai\\claude-code\\bin\\claude.exe"   %*'), "node_modules\\@anthropic-ai\\claude-code\\bin\\claude.exe");
    assert.equal(shimTarget("@echo off\r\nsome-other-tool %*"), null);
  });

  it("anything else goes through cmd.exe", () => {
    assert.deepEqual(cmdLine("C:\\Windows\\System32\\cmd.exe", "C:\\x\\a.bat", ["b"]), ["C:\\Windows\\System32\\cmd.exe", ["/d", "/c", "C:\\x\\a.bat", "b"]]);
  });
});

describe("windows: detect and workspace never touch /bin/sh", () => {
  it("detect() finds a .cmd CLI, probes its login through it, and explains the CLIs that are missing", async () => {
    const entry = JSON.stringify({ version: BACKENDS["codex"].version, entry: "C:\\data\\bridges\\node_modules\\p\\index.js" });
    const { spawner, calls } = windows((command, args) => {
      if (command === `${NPM_DIR}\\codex.cmd`) return { code: 0, stdout: "Logged in using ChatGPT\n" };
      if (args[0] === "-e") return { code: 0, stdout: entry + "\n" };
      return undefined;
    });
    const st = Object.fromEntries((await createRuntime({ spawner, bridgeDir: "C:\\data\\bridges" }).detect()).map((s) => [s.id, s]));
    assert.equal(st["codex"]!.available, true, st["codex"]!.reason ?? "");
    assert.equal(st["codex"]!.account, "ChatGPT");
    assert.match(st["claude-code"]!.reason!, /Install Claude Code/);
    assert.ok(calls.some((c) => c.command === `${NPM_DIR}\\codex.cmd` && c.args.join(" ") === "login status"));
    assert.ok(calls.every((c) => c.command !== "/bin/sh"));
  });

  it("prepareWorkspace makes the folder with node, not mkdir -p", async () => {
    const { spawner, calls } = windows();
    assert.equal(await prepareWorkspace(spawner, "C:\\data\\ws"), "C:\\data\\ws");
    const mk = calls.find((c) => c.command === NODE)!;
    assert.deepEqual(mk.args.slice(0, 1), ["-e"]);
    assert.ok(mk.args[1]!.includes("mkdirSync"));
    assert.equal(mk.args[2], "C:\\data\\ws");
    assert.ok(calls.every((c) => c.command !== "/bin/sh"));
  });
});
