// The one-click zotero-cli install: which installer runs when. A fake Spawner stands in for the machine.
import assert from "node:assert/strict";
import { test } from "node:test";
import type { Proc, Spawner } from "../src/types.ts";
import { installCli } from "../src/zotero/doctor.ts";

/** A machine with the given binaries; spawn() records each command and applies its effect. */
function machine(have: Record<string, string>, outcome: (cmd: string, args: string[]) => number = () => 0) {
  const ran: { cmd: string; args: string[]; env: Record<string, string> }[] = [];
  const spawner: Spawner = {
    baseEnv: async () => ({ HOME: "/Users/me", PATH: "/usr/bin:/bin" }),
    // findBinary(): command -v NAME on PATH, then the extra folders it is given
    async run(_command, args, opts) {
      const name = opts.env["NAME"]!;
      const dirs = args.slice(3);
      const hit = have[name] ?? dirs.map((d) => `${d}/${name}`).find((p) => Object.values(have).includes(p));
      return { code: hit ? 0 : 1, stdout: hit ? hit + "\n" : "", stderr: "" };
    },
    async spawn(cmd, args, opts): Promise<Proc> {
      ran.push({ cmd, args, env: opts.env });
      const code = outcome(cmd, args);
      if (code === 0 && args.join(" ").includes("astral.sh/uv/install.sh")) have["uv"] = "/Users/me/.local/bin/uv";
      return {
        pid: 1, write() {}, onStdoutLine(cb) { cb("output line"); }, onStderr() {}, exited: Promise.resolve(code), kill: async () => {},
      };
    },
  };
  return { spawner, ran };
}
const drain = async (it: AsyncIterable<string>) => { const out: string[] = []; for await (const l of it) out.push(l); return out; };

test("with uv installed, only `uv tool install` runs", async () => {
  const { spawner, ran } = machine({ uv: "/Users/me/.local/bin/uv" });
  const lines = await drain(installCli(spawner)());
  assert.deepEqual(ran.map((r) => [r.cmd.split("/").pop(), r.args.slice(0, 2).join(" ")]), [["uv", "tool install"]]);
  assert.ok(lines.includes("Installed."));
});

test("on a machine with nothing, uv is fetched first (leaving the shell profile alone), then used", async () => {
  const { spawner, ran } = machine({});
  const lines = await drain(installCli(spawner)());
  assert.equal(ran.length, 2);
  assert.equal(ran[0]!.cmd, "/bin/sh");
  assert.match(ran[0]!.args.join(" "), /astral\.sh\/uv\/install\.sh/);
  assert.equal(ran[0]!.env["UV_NO_MODIFY_PATH"], "1", "the user's dotfiles are not edited");
  assert.equal(ran[1]!.cmd, "/Users/me/.local/bin/uv", "uv is found in ~/.local/bin even though the login PATH lacks it");
  assert.deepEqual(ran[1]!.args, ["tool", "install", "--upgrade", "zotero-mcp-server"]);
  assert.ok(lines.some((l) => /uv tool update-shell|terminal/.test(l)), "tells how to get zotero-cli into a terminal");
});

test("if uv cannot be fetched, pipx and then pip are tried; with no installer at all it says what to do", async () => {
  const failUv = (cmd: string, args: string[]) => (args.join(" ").includes("astral.sh") ? 1 : 0);
  const { spawner, ran } = machine({ pipx: "/usr/local/bin/pipx", python3: "/usr/bin/python3" }, failUv);
  await drain(installCli(spawner)());
  assert.deepEqual(ran.map((r) => r.cmd.split("/").pop()), ["sh", "pipx"], "stops at the first installer that works");

  const none = machine({}, () => 1);
  const lines = await drain(installCli(none.spawner)());
  assert.ok(lines.some((l) => /No installer found/.test(l)));
});

test("an installer that never finishes is killed, and the stream ends instead of spinning forever", async () => {
  let killed = false;
  let end!: (code: number | null) => void;
  const spawner: Spawner = {
    baseEnv: async () => ({ HOME: "/Users/me", PATH: "/usr/bin:/bin" }),
    run: async (_c, _a, opts) => (opts.env["NAME"] === "uv" ? { code: 0, stdout: "/usr/bin/uv\n", stderr: "" } : { code: 1, stdout: "", stderr: "" }),
    spawn: async () => ({
      pid: 1, write() {}, onStdoutLine() {}, onStderr() {}, kill: async () => { killed = true; end(null); },
      exited: new Promise<number | null>((r) => { end = r; }),
    }),
  };
  const lines = await drain(installCli(spawner, 30)());
  assert.ok(killed, "the stuck process is killed");
  assert.ok(lines.some((l) => /Gave up waiting/.test(l)));
  assert.ok(!lines.includes("Installed."));
});
