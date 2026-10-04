// "Is everything the chat needs in place?" Each check says what is wrong and, where we can, fixes it.
import type { AgentRuntime, DoctorCheck, PanelSettings, Spawner } from "../types.ts";
import { findBinary } from "../agent/index.ts";

const port = () => Number(Zotero.Prefs.get("httpServer.port")) || 23119;

/**
 * HTTP status of Zotero's local server as an outside process sees it, which is what zotero-cli is. (Asking from inside
 * Zotero's own window is refused by the platform, and would not catch the server that has gone quiet.) 0 = no answer.
 */
async function httpStatus(spawner: Spawner, env: Record<string, string>, path: string, headers: string[] = []): Promise<number> {
  const args = ["-s", "-m", "4", "-o", "/dev/null", "-w", "%{http_code}", ...headers.flatMap((h) => ["-H", h]), `http://127.0.0.1:${port()}${path}`];
  const r = await spawner.run("/usr/bin/curl", args, { env, timeoutMs: 8000 }).catch(() => null);
  return Number(r?.stdout.trim()) || 0;
}

/** Run a command and yield its output line by line; returns its exit code. A command still running after `maxMs` is killed: a stalled download must not leave the button spinning forever. */
async function* stream(spawner: Spawner, env: Record<string, string>, cmd: string, args: string[], maxMs = 10 * 60_000): AsyncGenerator<string, number | null> {
  yield `$ ${cmd.split("/").pop()} ${args.join(" ")}`;
  const proc = await spawner.spawn(cmd, args, { env });
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; void proc.kill(); }, maxMs);
  const lines: string[] = [];
  let wake: (() => void) | null = null;
  const push = (s: string) => { lines.push(...s.split("\n").filter(Boolean)); wake?.(); };
  proc.onStdoutLine(push);
  proc.onStderr(push);
  let done = false;
  void proc.exited.then(() => { done = true; wake?.(); });
  while (!done || lines.length) {
    if (lines.length) yield lines.shift()!;
    else await new Promise<void>((r) => { wake = r; });
  }
  clearTimeout(timer);
  if (timedOut) yield "Gave up waiting. Check your network and try again.";
  return proc.exited;
}

const localBin = (env: Record<string, string>): string[] => (env["HOME"] ? [`${env["HOME"]}/.local/bin`] : []);

/** zotero-cli, also in ~/.local/bin: where uv puts it, and a Dock-launched Zotero's login PATH may not list that. */
export const findCli = (spawner: Spawner, env: Record<string, string>) => findBinary(spawner, env, "zotero-cli", localBin(env));

/**
 * Install zotero-cli. uv first, because it brings its own Python (macOS ships 3.9, and zotero-mcp needs 3.10); on a machine
 * with no uv it is fetched from astral.sh first, without touching the user's shell profile. pipx and pip are the fallbacks.
 */
export function installCli(spawner: Spawner, maxMs?: number): () => AsyncIterable<string> {
  return async function* () {
    const env = await spawner.baseEnv();
    let uv = await findBinary(spawner, env, "uv", localBin(env));
    if (!uv) {
      yield "uv, the installer zotero-cli uses, is not installed. Fetching it from astral.sh first.";
      const code = yield* stream(spawner, { ...env, UV_NO_MODIFY_PATH: "1" }, "/bin/sh", ["-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], maxMs);
      uv = code === 0 ? await findBinary(spawner, env, "uv", localBin(env)) : null;
    }
    const attempts: [string, string[]][] = [];
    if (uv) attempts.push([uv, ["tool", "install", "--upgrade", "zotero-mcp-server"]]);
    const pipx = await findBinary(spawner, env, "pipx");
    if (pipx) attempts.push([pipx, ["install", "--force", "zotero-mcp-server"]]);
    const py = await findBinary(spawner, env, "python3");
    if (py) attempts.push([py, ["-m", "pip", "install", "--user", "--upgrade", "zotero-mcp-server"]]);
    if (!attempts.length) {
      yield "No installer found. Install uv (https://docs.astral.sh/uv/), then try again.";
      return;
    }
    for (const [cmd, args] of attempts) {
      const code = yield* stream(spawner, env, cmd, args, maxMs);
      if (code === 0) {
        yield "Installed.";
        if (cmd === uv && !(env["PATH"] ?? "").split(":").some((p) => localBin(env).includes(p))) {
          yield "The chat can use it now. To use zotero-cli in a terminal too, run `uv tool update-shell` and reopen the terminal.";
        }
        return;
      }
      yield `Failed (exit ${code}).`;
    }
  };
}

export function createDoctor(deps: { win: any; spawner: Spawner; runtime: AgentRuntime; settings: () => PanelSettings }) {
  const { win, spawner, runtime } = deps;
  return async function doctor(): Promise<DoctorCheck[]> {
    const checks: DoctorCheck[] = [];
    const env = await spawner.baseEnv();

    // 1. Zotero's local server: a long-running Zotero can stop answering on its port while the app looks fine.
    const ping = await httpStatus(spawner, env, "/connector/ping");
    if (!Zotero.Prefs.get("httpServer.enabled")) {
      checks.push({
        id: "zotero-api", ok: false, label: "Zotero's local server is turned off", detail: "zotero-cli reaches your library through it. Turn it on, then restart Zotero.",
        fix: { label: "Turn it on", run: async function* () { Zotero.Prefs.set("httpServer.enabled", true); Zotero.Prefs.set("httpServer.localAPI.enabled", true); yield "Enabled. Restart Zotero to apply."; } },
      });
    } else if (!ping) {
      checks.push({ id: "zotero-api", ok: false, label: "Zotero's local server is not answering", detail: "Zotero stops serving its local port after running for a long time. Quit and reopen Zotero, then try again." });
    } else if (!Zotero.Prefs.get("httpServer.localAPI.enabled")) {
      checks.push({
        id: "zotero-api", ok: false, label: "Zotero's local API is off", detail: "zotero-cli reads your library through it.",
        fix: { label: "Turn it on", run: async function* () { Zotero.Prefs.set("httpServer.localAPI.enabled", true); yield "Local API enabled."; } },
      });
    } else {
      const api = await httpStatus(spawner, env, "/api/users/0/items?limit=1", ["Zotero-API-Version: 3"]);
      // pyzotero, which zotero-cli is built on, only ever talks to 23119.
      checks.push(api === 200 && port() !== 23119
        ? { id: "zotero-api", ok: false, label: `Zotero's local server is on port ${port()}, not 23119`, detail: "zotero-cli can only reach Zotero on port 23119. Change it back in Zotero's settings (Advanced) and restart Zotero." }
        : api === 200
        ? { id: "zotero-api", ok: true, label: "Zotero's local API is reachable" }
        : { id: "zotero-api", ok: false, label: "Zotero's local API does not answer", detail: `The server is up but /api answered ${api || "nothing"}. Restart Zotero.` });
    }

    // 2. zotero-cli
    const cli = await findCli(spawner, env);
    if (cli) {
      checks.push({ id: "cli", ok: true, label: "zotero-cli is installed", detail: cli });
    } else {
      checks.push({ id: "cli", ok: false, label: "zotero-cli is not installed", detail: "The agent uses it to work with your library. One click installs it, and uv (its installer) if you do not have that.", fix: { label: "Install zotero-cli", run: installCli(spawner) } });
    }

    // 3. node: the bridges are Node programs
    const node = await findBinary(spawner, env, "node");
    checks.push(node
      ? { id: "node", ok: true, label: "Node.js found", detail: node }
      : { id: "node", ok: false, label: "Node.js not found", detail: "The agent bridges run on Node.js. Install it from https://nodejs.org, then reopen Zotero." });

    // 4. the chosen agent
    const chosen = deps.settings().backend;
    const status = (await runtime.detect()).find((s) => s.id === chosen);
    checks.push(status?.available
      ? { id: "backend", ok: true, label: `${status.label} is ready`, ...(status.account ? { detail: status.account } : {}) }
      : { id: "backend", ok: false, label: `${status?.label ?? chosen} is not available`, ...(status?.reason ? { detail: status.reason } : {}) });

    // 5. writes: the CLI reports how it can write; without a grant, the fix runs the authorization (Zotero asks the user)
    const mcp = await findBinary(spawner, env, "zotero-mcp");
    const st = mcp ? await spawner.run(mcp, ["authorize-local", "--status"], { env, timeoutMs: 20000 }).catch(() => null) : null;
    const mode = st && /^Write mode:\s+(\w+)/m.exec(st.stdout)?.[1];
    if (mcp && mode === "none") {
      checks.push({ id: "write-access", ok: false, label: "Writes to your library are not authorized", detail: "Reading works. To let the agent edit, Zotero will ask you to allow it; choose Always Allow.",
        fix: { label: "Authorize writes", run: async function* () { yield* stream(spawner, env, mcp, ["authorize-local"]); } } });
    } else {
      checks.push({ id: "write-access", ok: true, label: mode ? `Writes are set up (${mode})` : "Changes to your library need your OK in Zotero", ...(mode ? {} : { detail: "The first time the agent edits your library, Zotero asks you to allow it." }) });
    }
    return checks;
  };
}
