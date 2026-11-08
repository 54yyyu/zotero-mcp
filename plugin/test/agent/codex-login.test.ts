import assert from "node:assert/strict";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { after, describe, it } from "node:test";
import { probeLogin } from "../../src/agent/backends.ts";
import { createRuntime } from "../../src/agent/runtime.ts";
import type { Spawner } from "../../src/types.ts";
import { cleanupTemp, fakeBin, fakeBridgeDir, hermeticSpawner } from "./helpers.ts";

after(cleanupTemp);

/** A nested dependency, as npm may install it, with no platform-specific vendor path. */
function bundledCli(dir: string, body: string): void {
  const pkg = join(dir, "node_modules", "@agentclientprotocol", "codex-acp", "node_modules", "@openai", "codex");
  mkdirSync(join(pkg, "bin"), { recursive: true });
  writeFileSync(join(pkg, "package.json"), JSON.stringify({ name: "@openai/codex", bin: { codex: "bin/codex.js" } }));
  writeFileSync(join(pkg, "bin", "codex.js"), body);
}

const SIGNED_IN = 'console.error("Logged in using ChatGPT");';
const SIGNED_OUT = 'console.error("Not logged in"); process.exitCode = 1;';

async function detect(dir: string, systemCli = "kill -KILL $$", npm = "exit 0") {
  const bin = fakeBin({ codex: systemCli, npm });
  const spawner = hermeticSpawner({}, [bin]);
  const runtime = createRuntime({ spawner, bridgeDir: dir, nodePath: process.execPath });
  return (await runtime.detect()).find((s) => s.id === "codex")!;
}

describe("Codex login detection", () => {
  it("uses the bridge's Codex when the system CLI is killed before reporting login", async () => {
    const dir = fakeBridgeDir(["codex"]);
    bundledCli(dir, SIGNED_IN);
    const status = await detect(dir);
    assert.equal(status.available, true, status.reason ?? "");
    assert.equal(status.account, "ChatGPT");
  });

  it("does not probe the bundled CLI when the system CLI is already signed in", async () => {
    const dir = fakeBridgeDir(["codex"]);
    const marker = join(dir, "unexpected-probe");
    bundledCli(dir, `require("fs").writeFileSync(${JSON.stringify(marker)}, "called"); ${SIGNED_OUT}`);
    assert.equal((await detect(dir, 'echo "Logged in using ChatGPT" >&2')).available, true);
    assert.equal(existsSync(marker), false);
  });

  it("keeps a genuinely signed-out user unavailable", async () => {
    const dir = fakeBridgeDir(["codex"]);
    bundledCli(dir, SIGNED_IN);
    const status = await detect(dir, 'echo "Not logged in" >&2; exit 1');
    assert.equal(status.available, false);
    assert.match(status.reason!, /not signed in/i);
  });

  it("reports the failed system probe when the bridge has no Codex dependency", async () => {
    const status = await detect(fakeBridgeDir(["codex"]));
    assert.equal(status.available, false);
    assert.match(status.reason!, /login check.*terminated/i);
    assert.doesNotMatch(status.reason!, /not signed in/i);
  });

  it("does not turn two terminated probes into a successful login", async () => {
    const dir = fakeBridgeDir(["codex"]);
    bundledCli(dir, 'process.kill(process.pid, "SIGKILL");');
    const status = await detect(dir);
    assert.equal(status.available, false);
    assert.match(status.reason!, /login check.*terminated/i);
  });

  it("does not treat a signed-out fallback as successful", async () => {
    const dir = fakeBridgeDir(["codex"]);
    bundledCli(dir, SIGNED_OUT);
    const status = await detect(dir);
    assert.equal(status.available, false);
    assert.match(status.reason!, /login check.*terminated/i);
  });

  it("recognizes both Gecko's negative exit code and Node's null signal result", async () => {
    for (const code of [-9, null]) {
      const spawner: Spawner = {
        baseEnv: async () => ({}),
        spawn: async () => { throw new Error("not used"); },
        run: async () => ({ code, stdout: "", stderr: "" }),
      };
      const result = await probeLogin(spawner, {}, "codex", "/codex");
      assert.equal(result.loggedIn, false);
      assert.match(result.reason!, /login check.*terminated/i);
    }
  });

  it("does not install a bridge just to check login", async () => {
    const dir = fakeBridgeDir([]);
    const marker = join(dir, "unexpected-install");
    const status = await detect(dir, "kill -KILL $$", `touch '${marker}'; exit 1`);
    assert.equal(status.available, false);
    assert.match(status.reason!, /login check.*terminated/i);
    assert.equal(existsSync(marker), false);
  });
});
