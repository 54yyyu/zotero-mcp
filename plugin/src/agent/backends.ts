// Which coding agent answers: a small table of ACP bridges, one entry per vendor. An entry is a
// record of measurement (handshakes run against the real bridges, 2026-10-04), not a wish list.
// Everything vendor-specific lives here; acp.ts and session.ts speak plain ACP and branch only
// through these fields.
//
// Measured shapes (session/new response):
//   claude-agent-acp 0.85.1  configOptions: mode(category mode) / model(category model, values are
//                            aliases: default, opus, sonnet, haiku, claude-opus-5, ...) / effort / fast;
//                            `modes` (default, acceptEdits, plan, auto, bypassPermissions);
//                            no `models`. session/set_model is -32601; session/set_config_option
//                            {configId:"model", value} works and accepts a full id ("claude-sonnet-5-5"
//                            lands on the "sonnet" option). session/set_mode works.
//   codex-acp 2.1.1          `models` {availableModels:[{modelId:"gpt-5.5[medium]"}], currentModelId},
//                            `modes` (read-only, workspace-write, agent, agent-full-access), and
//                            configOptions too. session/set_model {modelId} and session/set_mode work.
//   pi-acp 0.0.34            configOptions: model (hundreds of "provider/model") / thought_level;
//                            `modes` are pi's thinking levels, NOT permissions (ignored here);
//                            session/set_config_option works. No permission frames at all.
import type { BackendId, Spawner } from "../types.ts";
import { findBinary } from "./env.ts";

export interface BackendSpec {
  id: BackendId;
  label: string;
  /** npm package of the ACP bridge, and the version we pin (checked with `npm view`). */
  pkg: string;
  version: string;
  /** The CLI the user must have on their login PATH, when the backend needs one. */
  cli?: { name: string; hint: string };
  /** How the model is chosen after session/new: a config option, or session/set_model. */
  modelVia: "config_option" | "set_model";
  /** Where the reasoning effort lives: a config option (`effort` / `thought_level`), or a `[level]` suffix on model ids (codex). */
  effortVia: "config_option" | "model-suffix";
  /** Env vars that make this backend bill an API key instead of the user's subscription. */
  apiKeyVars: string[];
  /** The bridge's `modes` are permission modes we may set (pi's are thinking levels). */
  permissionModes: boolean;
  /** `_meta.systemPrompt.append` is honoured (Claude), or the brief rides on the first prompt. */
  briefVia: "system-prompt" | "first-prompt";
  /** Tool names in `_meta.claudeCode.toolName` (Claude only). */
  claudeMeta: boolean;
  /** The terminal command that continues a session by its id, run from the folder the session ran in. */
  resume: (sessionId: string) => string;
}

export const BACKENDS: Record<BackendId, BackendSpec> = {
  "claude-code": {
    id: "claude-code",
    label: "Claude Code",
    pkg: "@agentclientprotocol/claude-agent-acp",
    version: "0.85.1",
    cli: { name: "claude", hint: "Install Claude Code (https://claude.com/claude-code), then run `claude` once to sign in." },
    modelVia: "config_option",
    effortVia: "config_option",
    apiKeyVars: ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
    permissionModes: true,
    briefVia: "system-prompt",
    claudeMeta: true,
    resume: (id) => `claude --resume ${id}`,
  },
  codex: {
    id: "codex",
    label: "Codex",
    pkg: "@agentclientprotocol/codex-acp",
    version: "2.1.1",
    cli: { name: "codex", hint: "Install Codex (npm i -g @openai/codex), then run `codex login`." },
    modelVia: "set_model",
    effortVia: "model-suffix",
    apiKeyVars: ["OPENAI_API_KEY", "CODEX_API_KEY"],
    permissionModes: true,
    briefVia: "first-prompt",
    claudeMeta: false,
    resume: (id) => `codex resume ${id}`,
  },
  pi: {
    id: "pi",
    label: "pi",
    pkg: "pi-acp",
    version: "0.0.34",
    cli: { name: "pi", hint: "Install pi (npm i -g @earendil-works/pi-coding-agent) and configure a provider." },
    modelVia: "config_option",
    effortVia: "config_option",
    apiKeyVars: [],
    permissionModes: false,
    briefVia: "first-prompt",
    claudeMeta: false,
    resume: (id) => `pi --session ${id}`,
  },
};

export const BACKEND_IDS: BackendId[] = ["claude-code", "codex", "pi"];

export function backendOf(id: BackendId): BackendSpec {
  const b = BACKENDS[id];
  if (!b) throw new Error(`unknown backend "${id}"`);
  return b;
}

/** Login state of a backend's own CLI, when it can say (free: no model call). */
interface LoginProbe {
  /** undefined = the CLI cannot tell. */
  loggedIn?: boolean;
  account?: string;
}

/** `claude auth status` prints JSON; `codex login status` prints a sentence. pi has no login of its own. */
export async function probeLogin(spawner: Spawner, env: Record<string, string>, id: BackendId, cliPath: string): Promise<LoginProbe> {
  try {
    if (id === "claude-code") {
      const r = await spawner.run(cliPath, ["auth", "status"], { env, timeoutMs: 15_000 });
      const s = JSON.parse(r.stdout) as { loggedIn?: boolean; subscriptionType?: string; authMethod?: string };
      if (typeof s.loggedIn !== "boolean") return {};
      const plan = s.subscriptionType ? `Claude ${s.subscriptionType[0]!.toUpperCase()}${s.subscriptionType.slice(1)}` : undefined;
      return s.loggedIn ? { loggedIn: true, ...(plan ? { account: plan } : s.authMethod ? { account: s.authMethod } : {}) } : { loggedIn: false };
    }
    if (id === "codex") {
      const r = await spawner.run(cliPath, ["login", "status"], { env, timeoutMs: 15_000 });
      const text = `${r.stdout}\n${r.stderr}`;
      if (r.code === 0 && /logged in/i.test(text)) {
        const via = /using (.+)/i.exec(text)?.[1]?.trim();
        return { loggedIn: true, ...(via ? { account: via } : {}) };
      }
      return { loggedIn: false };
    }
  } catch {
    // an unreadable answer is "cannot tell", not "logged out"
  }
  return {};
}

/** The backend's own CLI (`claude` also from ~/.local/bin, where its installer puts it): the user's, which may be newer than the SDK the bridge bundles. */
export async function findCli(spawner: Spawner, env: Record<string, string>, spec: BackendSpec): Promise<string | null> {
  if (!spec.cli) return null;
  const home = env["HOME"];
  return findBinary(spawner, env, spec.cli.name, spec.id === "claude-code" && home ? [`${home}/.local/bin`] : []);
}

/** `cd "<folder>" && <resume command>`: what to paste in a terminal to continue a chat started in the panel. */
export function resumeCommand(backend: BackendId, cwd: string, sessionId: string): string {
  const quoted = `'${cwd.replace(/'/g, `'\\''`)}'`;
  return `cd ${quoted} && ${BACKENDS[backend].resume(sessionId)}`;
}
