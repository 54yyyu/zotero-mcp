// A Spawner (see src/types.ts) backed by node:child_process, for tests and the dev harness.
// Mirrors what the Gecko one must do: the child leads its own process group (kill() signals the
// whole group), baseEnv() is the login-shell environment, spawn failures reject.
import { spawn as nodeSpawn, type ChildProcess } from "node:child_process";
import type { Proc, Spawner } from "../src/types.ts";

const BEGIN = "__ZMC_ENV_BEGIN__";
const END = "__ZMC_ENV_END__";
const SKIP = new Set(["_", "SHLVL", "PWD", "OLDPWD"]);

/**
 * The login shell's environment (`$SHELL -lc`), merged over this process's. rc-file noise (banners,
 * `echo`s in .zshrc) is ignored: only what sits between the markers is read. `env -0` keeps values
 * with newlines intact.
 */
export async function loginEnv(timeoutMs = 10_000): Promise<Record<string, string>> {
  const base: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) if (v !== undefined) base[k] = v;
  const shell = process.env["SHELL"] || "/bin/sh";
  const out = await new Promise<string>((resolve) => {
    let text = "";
    let child: ChildProcess;
    try {
      child = nodeSpawn(shell, ["-lc", `printf '%s' ${BEGIN}; env -0; printf '%s' ${END}`], { stdio: ["ignore", "pipe", "ignore"], env: base });
    } catch {
      return resolve("");
    }
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      resolve("");
    }, timeoutMs);
    child.stdout!.setEncoding("utf8");
    child.stdout!.on("data", (d: string) => (text += d));
    child.on("error", () => {
      clearTimeout(timer);
      resolve("");
    });
    child.on("close", () => {
      clearTimeout(timer);
      resolve(text);
    });
  });
  const a = out.indexOf(BEGIN);
  const b = out.lastIndexOf(END);
  if (a < 0 || b < a) return base;
  const login: Record<string, string> = {};
  for (const entry of out.slice(a + BEGIN.length, b).split("\0")) {
    const eq = entry.indexOf("=");
    if (eq <= 0) continue;
    const key = entry.slice(0, eq);
    if (!SKIP.has(key)) login[key] = entry.slice(eq + 1);
  }
  return { ...base, ...login };
}

export interface NodeSpawnerOpts {
  /** SIGTERM-to-SIGKILL grace in ms (default 2000). */
  graceMs?: number;
  /** Use this environment instead of resolving the login shell's (fast, hermetic tests). */
  baseEnv?: Record<string, string>;
}

export function createNodeSpawner(opts: NodeSpawnerOpts = {}): Spawner {
  const graceMs = opts.graceMs ?? 2000;
  let cached: Promise<Record<string, string>> | undefined;

  return {
    baseEnv(): Promise<Record<string, string>> {
      if (opts.baseEnv) return Promise.resolve({ ...opts.baseEnv });
      cached ??= loginEnv();
      return cached.then((e) => ({ ...e }));
    },

    spawn(command, args, o): Promise<Proc> {
      return new Promise<Proc>((resolve, reject) => {
        const child = nodeSpawn(command, args, { cwd: o.cwd, env: o.env, detached: true, stdio: ["pipe", "pipe", "pipe"] });
        child.once("error", (e) => reject(e));
        child.once("spawn", () => resolve(wrap(child, graceMs)));
      });
    },

    run(command, args, o) {
      return new Promise((resolve, reject) => {
        const child = nodeSpawn(command, args, { cwd: o.cwd, env: o.env, detached: true, stdio: ["ignore", "pipe", "pipe"] });
        let stdout = "";
        let stderr = "";
        let timedOut = false;
        const timer = o.timeoutMs
          ? setTimeout(() => {
              timedOut = true;
              signalGroup(child, "SIGKILL");
            }, o.timeoutMs)
          : undefined;
        child.stdout!.setEncoding("utf8");
        child.stderr!.setEncoding("utf8");
        child.stdout!.on("data", (d: string) => (stdout += d));
        child.stderr!.on("data", (d: string) => (stderr += d));
        child.once("error", (e) => {
          if (timer) clearTimeout(timer);
          reject(e);
        });
        child.once("close", (code) => {
          if (timer) clearTimeout(timer);
          resolve({ code: timedOut ? null : code, stdout, stderr: timedOut ? stderr + `\ntimed out after ${o.timeoutMs} ms` : stderr });
        });
      });
    },
  };
}

function signalGroup(child: ChildProcess, signal: NodeJS.Signals): void {
  const pid = child.pid;
  if (pid === undefined) return;
  try {
    process.kill(-pid, signal);
  } catch {
    try {
      process.kill(pid, signal);
    } catch {
      // already gone
    }
  }
}

function wrap(child: ChildProcess, graceMs: number): Proc {
  let outBuf = "";
  const lineCbs: ((line: string) => void)[] = [];
  const errCbs: ((chunk: string) => void)[] = [];
  child.stdout!.setEncoding("utf8");
  child.stderr!.setEncoding("utf8");
  child.stdout!.on("data", (chunk: string) => {
    outBuf += chunk;
    let i = outBuf.indexOf("\n");
    while (i >= 0) {
      const line = outBuf.slice(0, i);
      outBuf = outBuf.slice(i + 1);
      for (const cb of lineCbs) cb(line);
      i = outBuf.indexOf("\n");
    }
  });
  child.stderr!.on("data", (chunk: string) => {
    for (const cb of errCbs) cb(chunk);
  });
  // Writing into a pipe whose reader died raises EPIPE asynchronously; unhandled it kills the host.
  child.stdin!.on("error", () => {});

  let gone = false;
  const exited = new Promise<number | null>((resolve) => {
    child.once("exit", (code) => {
      gone = true;
      // Let stdout drain so the last frames are read before the exit is announced.
      const t = setTimeout(() => resolve(code), 200);
      child.once("close", () => {
        clearTimeout(t);
        resolve(code);
      });
    });
    child.once("error", () => resolve(null));
  });

  return {
    pid: child.pid!,
    write(data: string) {
      if (!child.stdin!.destroyed) child.stdin!.write(data);
    },
    onStdoutLine(cb) {
      lineCbs.push(cb);
    },
    onStderr(cb) {
      errCbs.push(cb);
    },
    exited,
    async kill() {
      if (!gone) {
        signalGroup(child, "SIGTERM");
        const timer = new Promise<boolean>((r) => setTimeout(() => r(false), graceMs));
        const ok = await Promise.race([exited.then(() => true), timer]);
        if (!ok) signalGroup(child, "SIGKILL");
        await exited;
      }
      // A grandchild shares the group: make sure none outlives the bridge.
      try {
        process.kill(-child.pid!, "SIGKILL");
      } catch {
        // group already empty
      }
      child.stdin!.destroy();
      child.stdout!.destroy();
      child.stderr!.destroy();
      child.unref();
    },
  };
}
