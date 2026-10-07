// Shared bits for the agent tests: a fake "installed bridge" that runs the mock agent, a hermetic
// Spawner, and event collection.
import { chmodSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { AgentSession, ChatEvent, Spawner } from "../../src/types.ts";
import { BACKENDS } from "../../src/agent/backends.ts";
import { createNodeSpawner } from "../node-spawner.ts";

const MOCK_AGENT = join(dirname(fileURLToPath(import.meta.url)), "..", "mock-agent.mjs");

const cleanups: (() => void)[] = [];
export function cleanupTemp(): void {
  while (cleanups.length) cleanups.pop()!();
}

export function tempDir(prefix = "zmc-test-"): string {
  const d = mkdtempSync(join(tmpdir(), prefix));
  cleanups.push(() => rmSync(d, { recursive: true, force: true }));
  return d;
}

/**
 * A bridgeDir in which the pinned package for `ids` is "installed" and its bin runs the mock agent,
 * so the runtime never needs npm.
 */
export function fakeBridgeDir(ids: (keyof typeof BACKENDS)[] = ["claude-code", "codex", "pi"]): string {
  const dir = tempDir("zmc-bridges-");
  for (const id of ids) {
    const spec = BACKENDS[id];
    const pkgDir = join(dir, "node_modules", ...spec.pkg.split("/"));
    mkdirSync(join(pkgDir, "dist"), { recursive: true });
    writeFileSync(join(pkgDir, "package.json"), JSON.stringify({ name: spec.pkg, version: spec.version, type: "module", bin: { bridge: "dist/index.js" } }));
    writeFileSync(join(pkgDir, "dist", "index.js"), `import ${JSON.stringify("file://" + MOCK_AGENT)};\n`);
  }
  return dir;
}

/** A directory of fake executables, each a shell script with the given body. */
export function fakeBin(scripts: Record<string, string>): string {
  const dir = tempDir("zmc-bin-");
  for (const [name, body] of Object.entries(scripts)) {
    const p = join(dir, name);
    writeFileSync(p, `#!/bin/sh\n${body}\n`);
    chmodSync(p, 0o755);
  }
  return dir;
}

/**
 * A hermetic Spawner: PATH is `extraPath`, a dir holding only a `node` symlink, and the system's;
 * nothing from the user's shell (so a real `pi` or `codex` next to node never leaks in).
 */
export function hermeticSpawner(extraEnv: Record<string, string> = {}, extraPath: string[] = []): Spawner {
  const nodeOnly = tempDir("zmc-node-");
  symlinkSync(process.execPath, join(nodeOnly, "node"));
  const path = [...extraPath, nodeOnly, "/usr/bin", "/bin"].join(":");
  return createNodeSpawner({ graceMs: 500, baseEnv: { PATH: path, HOME: tempDir("zmc-home-"), ...extraEnv } });
}

export function collect(session: AgentSession): { events: ChatEvent[]; stop: () => void } {
  const events: ChatEvent[] = [];
  const stop = session.on((e) => events.push(e));
  return { events, stop };
}

export function textOf(events: ChatEvent[]): string {
  return events.filter((e): e is Extract<ChatEvent, { t: "text" }> => e.t === "text").map((e) => e.delta).join("");
}

export function waitFor<T>(fn: () => T | undefined | false, ms = 5000, what = "condition"): Promise<T> {
  const start = Date.now();
  return new Promise<T>((resolve, reject) => {
    const tick = () => {
      const v = fn();
      if (v) return resolve(v as T);
      if (Date.now() - start > ms) return reject(new Error(`timed out waiting for ${what}`));
      setTimeout(tick, 10);
    };
    tick();
  });
}

export function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
