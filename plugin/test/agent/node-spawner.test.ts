import assert from "node:assert/strict";
import { after, describe, it } from "node:test";
import { createNodeSpawner, loginEnv } from "../node-spawner.ts";
import { alive, cleanupTemp, fakeBin } from "./helpers.ts";

after(cleanupTemp);

describe("node spawner", () => {
  it("baseEnv: the login shell's PATH wins, rc-file noise is ignored", async () => {
    const shellDir = fakeBin({
      fakeshell: 'echo "Welcome! last login: today"; echo "PATH=garbage-in-noise"; PATH="/custom/login/bin:$PATH" MY_LOGIN_VAR="a b\nc" /bin/sh -c "$2"',
    });
    const saved = process.env["SHELL"];
    process.env["SHELL"] = `${shellDir}/fakeshell`;
    try {
      const env = await loginEnv();
      assert.ok(env["PATH"]!.startsWith("/custom/login/bin:"), env["PATH"] ?? "");
      assert.equal(env["MY_LOGIN_VAR"], "a b\nc");
      assert.ok(env["HOME"], "process env is merged underneath");
    } finally {
      if (saved === undefined) delete process.env["SHELL"];
      else process.env["SHELL"] = saved;
    }
  });

  it("run() captures output and exit code, and times out", async () => {
    const sp = createNodeSpawner({ baseEnv: { PATH: "/usr/bin:/bin" } });
    const env = await sp.baseEnv();
    assert.deepEqual(await sp.run("/bin/sh", ["-c", "echo out; echo err >&2; exit 4"], { env }), { code: 4, stdout: "out\n", stderr: "err\n" });
    const slow = await sp.run("/bin/sh", ["-c", "sleep 30"], { env, timeoutMs: 100 });
    assert.equal(slow.code, null);
    await assert.rejects(sp.run("/nonexistent/binary", [], { env }), /ENOENT/);
  });

  it("spawn() rejects when the command cannot start; kill() takes the whole group", async () => {
    const sp = createNodeSpawner({ baseEnv: { PATH: "/usr/bin:/bin" }, graceMs: 300 });
    const env = await sp.baseEnv();
    await assert.rejects(sp.spawn("/nonexistent/binary", [], { env }), /ENOENT/);
    const lines: string[] = [];
    const proc = await sp.spawn("/bin/sh", ["-c", "sleep 300 & echo $!; wait"], { env });
    proc.onStdoutLine((l) => lines.push(l));
    while (lines.length === 0) await new Promise((r) => setTimeout(r, 10));
    const grandchild = Number(lines[0]);
    assert.ok(alive(proc.pid) && alive(grandchild));
    await proc.kill();
    assert.equal(alive(proc.pid), false);
    assert.equal(alive(grandchild), false);
    assert.equal(await proc.exited, null);
  });
});
