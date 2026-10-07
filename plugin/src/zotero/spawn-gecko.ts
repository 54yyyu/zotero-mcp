// Spawner for Gecko: Subprocess.sys.mjs underneath. agent/ is handed this instead of node:child_process.
import type { Proc, Spawner } from "../types.ts";

const FALLBACK_PATH = ["/opt/homebrew/bin", "/usr/local/bin", "~/.local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"];

function Subprocess(): any {
  return ChromeUtils.importESModule("resource://gre/modules/Subprocess.sys.mjs").Subprocess;
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/** Output of a short helper command; never throws. */
async function quick(command: string, args: string[]): Promise<string> {
  try {
    const p = await Subprocess().call({ command, arguments: args, environmentAppend: true, stderr: "pipe" });
    let out = "", c: string;
    while ((c = await p.stdout.readString())) out += c;
    await p.wait();
    return out;
  } catch {
    return "";
  }
}

/** pid plus every descendant, children first: a bridge's grandchildren must not outlive it. */
async function tree(pid: number): Promise<number[]> {
  const kids = (await quick("/usr/bin/pgrep", ["-P", String(pid)])).split("\n").map(Number).filter(Boolean);
  const below = await Promise.all(kids.map(tree));
  return [...below.flat(), pid];
}

function parseEnv(text: string): Record<string, string> {
  const env: Record<string, string> = {};
  for (const line of text.split("\n")) {
    const m = /^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/.exec(line);
    if (m) env[m[1]!] = m[2]!;
  }
  return env;
}

class GeckoProc implements Proc {
  readonly pid: number;
  readonly exited: Promise<number | null>;
  private chain: Promise<unknown> = Promise.resolve();
  private lineCb: ((l: string) => void) | null = null;
  private errCb: ((c: string) => void) | null = null;
  private lines: string[] = [];
  private errs: string[] = [];
  private gone = false;

  private p: any;

  constructor(p: any) {
    this.p = p;
    this.pid = p.pid;
    this.exited = p.wait().then((r: any) => { this.gone = true; return typeof r.exitCode === "number" && r.exitCode >= 0 ? r.exitCode : null; });
    this.pump(p.stdout, (line) => (this.lineCb ? this.lineCb(line) : this.lines.push(line)), true);
    this.pump(p.stderr, (chunk) => (this.errCb ? this.errCb(chunk) : this.errs.push(chunk)), false);
  }

  private async pump(stream: any, emit: (s: string) => void, byLine: boolean) {
    let buf = "", chunk: string;
    try {
      while ((chunk = await stream.readString())) {
        if (!byLine) { emit(chunk); continue; }
        buf += chunk;
        let i: number;
        while ((i = buf.indexOf("\n")) >= 0) { emit(buf.slice(0, i)); buf = buf.slice(i + 1); }
      }
    } catch { /* the pipe closed under us */ }
    if (byLine && buf) emit(buf);
  }

  write(data: string): void {
    this.chain = this.chain.then(() => this.p.stdin.write(data)).catch(() => {});
  }

  onStdoutLine(cb: (line: string) => void): void {
    this.lineCb = cb;
    for (const l of this.lines.splice(0)) cb(l);
  }

  onStderr(cb: (chunk: string) => void): void {
    this.errCb = cb;
    for (const c of this.errs.splice(0)) cb(c);
  }

  async kill(): Promise<void> {
    if (this.gone) return;
    const pids = await tree(this.pid);
    await quick("/bin/kill", ["-TERM", ...pids.map(String)]);
    for (let i = 0; i < 20 && !this.gone; i++) await sleep(150);
    if (!this.gone) await quick("/bin/kill", ["-KILL", ...pids.map(String)]);
    await Promise.race([this.exited, sleep(2000)]);
  }
}

let baseEnvCache: Promise<Record<string, string>> | null = null;

async function loadBaseEnv(): Promise<Record<string, string>> {
  const process = parseEnv(await quick("/usr/bin/env", []));
  const shell = Services.env.get("SHELL") || "/bin/zsh";
  // -i as well as -l: nvm and friends are usually set up in .zshrc, which a login shell alone skips.
  let login: Record<string, string> = {};
  try {
    const p = await Subprocess().call({
      command: shell,
      arguments: ["-lic", 'printf "\\n__ZMC_ENV__\\n"; /usr/bin/env'],
      environmentAppend: true,
      stderr: "pipe",
    });
    let out = "", c: string;
    const read = (async () => { while ((c = await p.stdout.readString())) out += c; })();
    const done = await Promise.race([read.then(() => true), sleep(10000).then(() => false)]);
    if (!done) await p.kill(0);
    const i = out.lastIndexOf("__ZMC_ENV__");
    if (i >= 0) login = parseEnv(out.slice(i + "__ZMC_ENV__".length));
  } catch { /* fall through to the process env */ }
  const home = process.HOME || "";
  const merged = { ...process, ...login };
  if (!login.PATH) merged.PATH = [process.PATH, ...FALLBACK_PATH.map((d) => d.replace("~", home))].filter(Boolean).join(":");
  return merged;
}

export function createGeckoSpawner(): Spawner {
  async function launch(command: string, args: string[], opts: { cwd?: string; env: Record<string, string> }) {
    const sp = Subprocess();
    const exe = command.includes("/") ? command : await sp.pathSearch(command, opts.env);
    return sp.call({ command: exe, arguments: args, environment: opts.env, environmentAppend: false, workdir: opts.cwd, stderr: "pipe" });
  }
  return {
    async spawn(command, args, opts) {
      return new GeckoProc(await launch(command, args, opts));
    },
    async run(command, args, opts) {
      const proc = new GeckoProc(await launch(command, args, opts));
      let stdout = "", stderr = "";
      proc.onStdoutLine((l) => { stdout += l + "\n"; });
      proc.onStderr((c) => { stderr += c; });
      const timer = opts.timeoutMs ? setTimeout(() => { void proc.kill(); }, opts.timeoutMs) : null;
      const code = await proc.exited;
      if (timer) clearTimeout(timer);
      await sleep(30); // let the pumps flush the last chunk
      return { code, stdout, stderr };
    },
    baseEnv() {
      return (baseEnvCache ??= loadBaseEnv());
    },
  };
}
