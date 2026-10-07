// createRuntime: detect which backends can run here, and start a session on one.
// Process spawning is injected (Spawner); this file has no node:*, DOM or Zotero imports.
import type { AgentRuntime, AgentSession, BackendId, BackendStatus, Catalog, Spawner, StartOpts } from "../types.ts";
import { AcpClient } from "./acp.ts";
import { BACKEND_IDS, backendOf, findCli, probeLogin, type BackendSpec } from "./backends.ts";
import { ensureBridge, locateBridge } from "./bridges.ts";
import { bridgeEnv, dirnameOf, findBinary, stripApiKeys, withPathFirst } from "./env.ts";
import { AcpAgentSession } from "./session.ts";

export interface RuntimeOpts {
  spawner: Spawner;
  /** Where bridges are installed (`npm install --prefix`): the plugin's data dir. */
  bridgeDir: string;
  /** Use this `node` instead of the one on the login PATH. */
  nodePath?: string;
  /** Bound on the handshake requests (ms). Default 30 000. */
  initTimeoutMs?: number;
  /** Progress text while a bridge installs ("Installing @agentclientprotocol/claude-agent-acp@0.85.1…"). */
  onProgress?: (message: string) => void;
}

const NODE_MISSING = "node was not found on your login PATH. Install Node.js (https://nodejs.org), then reopen Zotero.";

/**
 * Remove a probe's temp workspace, and the empty project folder Claude Code makes for it under ~/.claude/projects
 * (its name is the real path with every non-alphanumeric turned into "-"). rmdir only removes empty folders, so a
 * folder holding anything is never touched.
 */
export const CATALOG_CLEANUP = 'real=$(cd "$1" && pwd -P); d="$HOME/.claude/projects/$(printf %s "$real" | sed "s/[^A-Za-z0-9]/-/g")"; rmdir "$d/memory" "$d" 2>/dev/null; rm -rf "$1"';

/**
 * What locks a bridge down (StartOpts.locked), beyond session/new's _meta (session.ts). Claude: no thinking, and the model
 * from the start (the bridge then skips its own setModel; a switch after session/new rebuilds Claude Code, measured 2.25 s).
 * Codex: config overrides (codex-acp hands CODEX_CONFIG to thread/start). pi: pi-acp starts PI_ACP_PI_COMMAND, here a
 * script in the session's folder that runs the user's pi without tools, extensions, skills, templates or context files, and
 * keeps no session. (codex-acp cannot make a thread ephemeral: Codex keeps its usual session file.)
 */
async function lockedEnv(spawner: Spawner, env: Record<string, string>, spec: BackendSpec, { cwd, model }: StartOpts): Promise<Record<string, string>> {
  if (spec.id === "claude-code") return { MAX_THINKING_TOKENS: "0", ...(model ? { ANTHROPIC_MODEL: model } : {}) };
  if (spec.id === "codex") return { CODEX_CONFIG: JSON.stringify({ sandbox_mode: "read-only", approval_policy: "never", web_search: "disabled" }) };
  const pi = await findCli(spawner, env, spec);
  if (!pi) throw new Error(spec.cli!.hint);
  const script = `${cwd}/pi-locked.sh`;
  const q = (s: string) => `'${s.replace(/'/g, `'\\''`)}'`;
  const body = `#!/bin/sh\nexec ${q(pi)} --no-tools --no-extensions --no-skills --no-prompt-templates --no-context-files --no-session "$@"\n`;
  const r = await spawner.run("/bin/sh", ["-c", 'printf "%s" "$2" > "$1" && chmod 700 "$1"', "sh", script, body], { env, timeoutMs: 10_000 });
  if (r.code !== 0) throw new Error(`could not prepare pi for translation: ${r.stderr.trim() || "write failed"}`);
  return { PI_ACP_PI_COMMAND: script };
}

export function createRuntime(opts: RuntimeOpts): AgentRuntime {
  const { spawner, bridgeDir } = opts;

  async function resolveNode(env: Record<string, string>): Promise<string | null> {
    return opts.nodePath ?? (await findBinary(spawner, env, "node"));
  }

  const catalogs = new Map<BackendId, Promise<Catalog>>();

  /** A short-lived session in a temp workspace, read and closed. Nothing is left running; Claude is asked not to keep the session, the others cannot be. */
  async function probeCatalog(id: BackendId): Promise<Catalog> {
    const env = await spawner.baseEnv();
    const made = await spawner.run("/bin/sh", ["-c", 'mktemp -d "${TMPDIR:-/tmp}/zotero-chat-catalog.XXXXXX"'], { env, timeoutMs: 10_000 });
    const cwd = made.stdout.trim();
    if (made.code !== 0 || !cwd) throw new Error("could not create a temporary workspace to read the agent's options");
    try {
      const session = await runtime.start({ backend: id, cwd, brief: "", ephemeral: true });
      try {
        return {
          models: session.models(),
          modes: session.modes(),
          efforts: session.efforts(),
          ...(session.currentModel() ? { model: session.currentModel()! } : {}),
          ...(session.currentMode() ? { mode: session.currentMode()! } : {}),
          ...(session.currentEffort() ? { effort: session.currentEffort()! } : {}),
        };
      } finally {
        await session.close();
      }
    } finally {
      await spawner.run("/bin/sh", ["-c", CATALOG_CLEANUP, "sh", cwd], { env, timeoutMs: 10_000 }).catch(() => undefined);
    }
  }

  const runtime: AgentRuntime = {
    catalog(id: BackendId): Promise<Catalog> {
      let p = catalogs.get(id);
      if (!p) {
        p = probeCatalog(id);
        catalogs.set(id, p);
        // A failure is not remembered: the next call tries again.
        p.catch(() => {
          if (catalogs.get(id) === p) catalogs.delete(id);
        });
      }
      return p;
    },

    async detect(): Promise<BackendStatus[]> {
      const env = await spawner.baseEnv();
      const node = await resolveNode(env);
      const nodeEnv = node ? withPathFirst(env, dirnameOf(node)) : env;
      const npm = node ? await findBinary(spawner, nodeEnv, "npm") : null;
      return Promise.all(
        BACKEND_IDS.map(async (id): Promise<BackendStatus> => {
          const spec = backendOf(id);
          const base = { id, label: spec.label };
          if (!node) return { ...base, available: false, reason: NODE_MISSING };
          const cli = await findCli(spawner, env, spec);
          if (spec.cli && !cli) return { ...base, available: false, reason: spec.cli.hint };
          const installed = await locateBridge(spawner, nodeEnv, node, bridgeDir, spec).catch(() => null);
          if (!installed && !npm) return { ...base, available: false, reason: `npm was not found, so the ${spec.label} bridge cannot be installed. Install Node.js (it includes npm).` };
          const login = cli ? await probeLogin(spawner, env, id, cli) : {};
          if (login.loggedIn === false) return { ...base, available: false, reason: `${spec.label} is not signed in. ${spec.cli?.hint ?? ""}`.trim() };
          return { ...base, available: true, ...(login.account ? { account: login.account } : {}) };
        }),
      );
    },

    async start(start: StartOpts): Promise<AgentSession> {
      const spec = backendOf(start.backend);
      const login = await spawner.baseEnv();
      const keyed = start.auth === "subscription" ? stripApiKeys(login, spec.apiKeyVars) : login;
      const env = (start.path ?? []).reduceRight((e, dir) => withPathFirst(e, dir), keyed);
      const node = await resolveNode(env);
      if (!node) throw new Error(NODE_MISSING);
      const [bridge, claude] = await Promise.all([
        ensureBridge({ spawner, env, node, bridgeDir, spec, ...(opts.onProgress ? { onProgress: opts.onProgress } : {}) }),
        spec.claudeMeta ? findCli(spawner, withPathFirst(env, dirnameOf(node)), spec) : null,
      ]);
      const extra = { ...start.env, ...(start.locked ? await lockedEnv(spawner, env, spec, start) : {}) };
      const childEnv = bridgeEnv(env, { claudeExecutable: claude, node, extra });
      const client = await AcpClient.spawn({
        spawner,
        command: node,
        args: [bridge.entry],
        cwd: start.cwd,
        env: childEnv,
        ...(opts.initTimeoutMs !== undefined ? { initTimeoutMs: opts.initTimeoutMs } : {}),
      });
      return AcpAgentSession.open(client, spec, start);
    },
  };
  return runtime;
}
