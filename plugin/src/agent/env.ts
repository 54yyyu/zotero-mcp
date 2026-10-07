// The environment a bridge runs in, and how we find binaries without node:fs (Gecko has none):
// everything goes through the injected Spawner, with the login-shell PATH it gives us.
import type { Spawner } from "../types.ts";

/** Session markers a parent Claude Code sets; left in the child's env, session/new refuses to start. */
const PARENT_SESSION_VARS = [
  "CLAUDECODE",
  "CLAUDE_CODE_ENTRYPOINT",
  "CLAUDE_CODE_SESSION_ID",
  "CLAUDE_CODE_CHILD_SESSION",
  "CLAUDE_CODE_SESSION_ATTENDED",
  "CLAUDE_CODE_MESSAGING_SOCKET",
  "CLAUDE_CODE_MESSAGING_TOKEN",
  "CLAUDE_CODE_SSE_PORT",
  "CLAUDE_CODE_EXECPATH",
  "CLAUDE_PID",
  "CLAUDE_EFFORT",
];

/** Anything else that is clearly about the parent's session (not the user's own config such as USE_BEDROCK). */
const PARENT_SESSION_PREFIXES = ["CLAUDE_CODE_SESSION", "CLAUDE_CODE_MESSAGING", "CLAUDE_AGENT_SDK"];

export function cleanEnv(base: Record<string, string>): Record<string, string> {
  const env: Record<string, string> = { ...base };
  for (const name of PARENT_SESSION_VARS) delete env[name];
  for (const key of Object.keys(env)) if (PARENT_SESSION_PREFIXES.some((p) => key.startsWith(p))) delete env[key];
  return env;
}

/** The env for a subscription session: the backend's API-key variables (BackendSpec.apiKeyVars) removed. */
export function stripApiKeys(env: Record<string, string>, names: string[]): Record<string, string> {
  const out = { ...env };
  for (const name of names) delete out[name];
  return out;
}

/** Put `dir` first on PATH (so a chosen `node` wins over another one). */
export function withPathFirst(env: Record<string, string>, dir: string | undefined): Record<string, string> {
  if (!dir) return env;
  const parts = (env["PATH"] ?? "").split(":").filter(Boolean);
  return { ...env, PATH: [dir, ...parts.filter((p) => p !== dir)].join(":") };
}

export function dirnameOf(path: string): string {
  const i = path.lastIndexOf("/");
  return i < 0 ? "." : i === 0 ? "/" : path.slice(0, i);
}

/**
 * Resolve `name` on the login PATH to an absolute path (`command -v` in /bin/sh), or null.
 * `extraDirs` are probed too (`~/.local/bin` for claude, where its installer puts it).
 */
export async function findBinary(spawner: Spawner, env: Record<string, string>, name: string, extraDirs: string[] = []): Promise<string | null> {
  const script =
    'p=$(command -v "$NAME" 2>/dev/null); if [ -n "$p" ] && [ -x "$p" ]; then echo "$p"; exit 0; fi; ' +
    'for d in "$@"; do if [ -x "$d/$NAME" ]; then echo "$d/$NAME"; exit 0; fi; done; exit 1';
  try {
    const r = await spawner.run("/bin/sh", ["-c", script, "sh", ...extraDirs], { env: { ...env, NAME: name }, timeoutMs: 10_000 });
    if (r.code !== 0) return null;
    const found = r.stdout.trim().split("\n").pop()?.trim();
    return found ? found : null;
  } catch {
    return null;
  }
}

/** The environment for a bridge process: clean, with CLAUDE_CODE_EXECUTABLE set when known, then the caller's extras. */
export function bridgeEnv(base: Record<string, string>, opts: { claudeExecutable?: string | null; node?: string; extra?: Record<string, string> } = {}): Record<string, string> {
  let env = cleanEnv(base);
  env = withPathFirst(env, opts.node ? dirnameOf(opts.node) : undefined);
  if (opts.claudeExecutable && !env["CLAUDE_CODE_EXECUTABLE"]) env["CLAUDE_CODE_EXECUTABLE"] = opts.claudeExecutable;
  return { ...env, ...(opts.extra ?? {}) };
}
