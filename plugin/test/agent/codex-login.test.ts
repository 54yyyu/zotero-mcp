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
function bundledCli(dir: string, body: string, version?: string): void {
  const pkg = join(dir, "node_modules", "@agentclientprotocol", "codex-acp", "node_modules", "@openai", "codex");
  mkdirSync(join(pkg, "bin"), { recursive: true });
  writeFileSync(join(pkg, "package.json"), JSON.stringify({ name: "@openai/codex", ...(version ? { version } : {}), bin: { codex: "bin/codex.js" } }));
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

  it("asks only the bridge's Codex, which chats run, when it answers", async () => {
    const dir = fakeBridgeDir(["codex"]);
    const marker = join(dir, "unexpected-probe");
    bundledCli(dir, SIGNED_IN);
    assert.equal((await detect(dir, `touch '${marker}'; echo "Not logged in" >&2; exit 1`)).available, true);
    assert.equal(existsSync(marker), false);
  });

  it("keeps a genuinely signed-out user unavailable", async () => {
    const dir = fakeBridgeDir(["codex"]);
    bundledCli(dir, SIGNED_OUT);
    const status = await detect(dir, 'echo "Logged in using ChatGPT" >&2');
    assert.equal(status.available, false);
    assert.match(status.reason!, /not signed in/i);
  });

  it("reports a config.toml the bridge's Codex cannot read, though the newer system Codex can (#811)", async () => {
    const dir = fakeBridgeDir(["codex"]);
    bundledCli(dir, 'console.error("Error loading configuration: /home/u/.codex/config.toml:39:1: invalid type: map, expected a boolean\\nin `features`\\n"); process.exitCode = 1;');
    const status = await detect(dir, 'echo "Logged in using ChatGPT" >&2');
    assert.equal(status.available, false);
    assert.match(status.reason!, /can't read your Codex configuration: \/home\/u\/\.codex\/config\.toml:39:1: invalid type: map, expected a boolean in `features`\. /);
  });

  // A user's own Codex that reports its version, then answers login status signed in.
  const own = (version: string, marker?: string) =>
    `case "$1" in --version) echo "codex-cli ${version}";; *) ${marker ? `touch '${marker}'; ` : ""}echo "Logged in using ChatGPT" >&2;; esac`;
  const CONFIG_ERROR = 'console.error("Error loading configuration: /home/u/.codex/config.toml:1:1: invalid type: map, expected a boolean"); process.exitCode = 1;';

  it("asks the user's own Codex when it is newer than the bridge's, since chats will run it (#811)", async () => {
    const dir = fakeBridgeDir(["codex"]);
    bundledCli(dir, CONFIG_ERROR, "0.159.3");
    const status = await detect(dir, own("0.167.0"));
    assert.equal(status.available, true, status.reason ?? "");
    assert.equal(status.account, "ChatGPT");
  });

  it("keeps the bridge's Codex when the user's own is older", async () => {
    const dir = fakeBridgeDir(["codex"]);
    const marker = join(dir, "own-probed");
    bundledCli(dir, CONFIG_ERROR, "0.159.3");
    const status = await detect(dir, own("0.150.1", marker));
    assert.equal(status.available, false);
    assert.match(status.reason!, /can't read your Codex configuration/);
    assert.equal(existsSync(marker), false, "the older Codex is not asked once the bundled one answered");
  });

  it("starts Codex chats with CODEX_PATH set to the user's newer Codex, and not to an older one (#811)", async () => {
    for (const [version, expected] of [["0.167.0", true], ["0.150.1", false]] as const) {
      const dir = fakeBridgeDir(["codex"]);
      bundledCli(dir, SIGNED_IN, "0.159.3");
      const bin = fakeBin({ codex: own(version), npm: "exit 0" });
      const base = hermeticSpawner({}, [bin]);
      const envs: Record<string, string>[] = [];
      const spawner: Spawner = { ...base, spawn: (command, args, opts) => { envs.push(opts.env); return base.spawn(command, args, opts); } };
      const runtime = createRuntime({ spawner, bridgeDir: dir, nodePath: process.execPath });
      const session = await runtime.start({ backend: "codex", cwd: dir, brief: "" } as never);
      await session.close();
      const bridgeEnv = envs.at(-1)!;
      if (expected) assert.equal(bridgeEnv.CODEX_PATH, join(bin, "codex"), `codex ${version}`);
      else assert.equal(bridgeEnv.CODEX_PATH, undefined, `codex ${version}`);
    }
  });

  it("asks the system Codex when the bridge is not installed yet", async () => {
    const status = await detect(fakeBridgeDir([]), 'echo "Logged in using ChatGPT" >&2');
    assert.equal(status.available, true, status.reason ?? "");
    assert.equal(status.account, "ChatGPT");
  });

  it("falls back to the system Codex when the OS kills the bridge's one", async () => {
    const dir = fakeBridgeDir(["codex"]);
    bundledCli(dir, 'process.kill(process.pid, "SIGKILL");');
    const status = await detect(dir, 'echo "Logged in using ChatGPT" >&2');
    assert.equal(status.available, true, status.reason ?? "");
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
    bundledCli(dir, 'process.kill(process.pid, "SIGKILL");');
    const status = await detect(dir, 'echo "Not logged in" >&2; exit 1');
    assert.equal(status.available, false);
    assert.match(status.reason!, /login check.*terminated/i);
  });

  it("counts a configured API key as signed in (#785)", async () => {
    const status = await detect(fakeBridgeDir(["codex"]), 'echo "API key configured - sk-proj-***"');
    assert.equal(status.available, true, status.reason ?? "");
    assert.equal(status.account, "API key");
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
