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
import type { BackendId, BuiltinBackendId, Spawner } from "../types.ts";
import { findBinary } from "./env.ts";
import { t } from "../i18n.ts";
import type { Key } from "../i18n.ts";

export interface BackendSpec {
  id: BackendId;
  label: string;
  /** The CLI the user must have on their login PATH, when the backend needs one. */
  /** `hint`: how to install it, a message key (shown in the user's language). */
  cli?: { name: string; hint: Key };
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
  /** Its advertised `/compact` was verified live (summarises on demand, the chat goes on): Claude, and Codex (2026-10-05, test/live/skills.live.ts). pi-acp lists one too, but reported no summary. */
  compacts: boolean;
}

export interface BridgeSpec extends BackendSpec {
  /** npm package of the ACP bridge, and the pinned version. */
  pkg: string;
  version: string;
  resume: (sessionId: string) => string;
}

// An unmeasured endpoint: prefer the wire IDs in ACP's models block, falling back to config options.
// Only advertised choices are shown; no vendor login probe, compaction claim or terminal resume command.
const CUSTOM: BackendSpec = {
  id: "custom", label: "Custom agent", modelVia: "set_model", effortVia: "config_option",
  apiKeyVars: [], permissionModes: true, briefVia: "first-prompt", claudeMeta: false, compacts: false,
};

export const BACKENDS: Record<BuiltinBackendId, BridgeSpec> = {
  "claude-code": {
    id: "claude-code",
    label: "Claude Code",
    pkg: "@agentclientprotocol/claude-agent-acp",
    version: "0.85.1",
    cli: { name: "claude", hint: "agent.install.claude" },
    modelVia: "config_option",
    effortVia: "config_option",
    apiKeyVars: ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
    permissionModes: true,
    briefVia: "system-prompt",
    claudeMeta: true,
    compacts: true,
    resume: (id) => `claude --resume ${id}`,
  },
  codex: {
    id: "codex",
    label: "Codex",
    pkg: "@agentclientprotocol/codex-acp",
    version: "2.1.1",
    cli: { name: "codex", hint: "agent.install.codex" },
    modelVia: "set_model",
    effortVia: "model-suffix",
    apiKeyVars: ["OPENAI_API_KEY", "CODEX_API_KEY"],
    permissionModes: true,
    briefVia: "first-prompt",
    claudeMeta: false,
    compacts: true,
    resume: (id) => `codex resume ${id}`,
  },
  pi: {
    id: "pi",
    label: "pi",
    pkg: "pi-acp",
    version: "0.0.34",
    cli: { name: "pi", hint: "agent.install.pi" },
    modelVia: "config_option",
    effortVia: "config_option",
    apiKeyVars: [],
    permissionModes: false,
    briefVia: "first-prompt",
    claudeMeta: false,
    compacts: false,
    resume: (id) => `pi --session ${id}`,
  },
};

export const BACKEND_IDS: BuiltinBackendId[] = ["claude-code", "codex", "pi"];

export function backendOf(id: BackendId): BackendSpec {
  if (id === "custom") return CUSTOM;
  const b = BACKENDS[id];
  if (!b) throw new Error(`unknown backend "${id}"`);
  return b;
}

/** Login state of a backend's own CLI, when it can say (free: no model call). */
interface LoginProbe {
  /** undefined = the CLI cannot tell. */
  loggedIn?: boolean;
  account?: string;
  /** A failed check is not evidence that the user is signed out. */
  reason?: string;
  /** The CLI could not read its own config file: chats with it fail too, whatever the login (#811). */
  configError?: boolean;
}

/** `claude auth status` prints JSON; `codex login status` prints a sentence. pi has no login of its own. */
export async function probeLogin(spawner: Spawner, env: Record<string, string>, id: BackendId, cliPath: string, cliArgs: string[] = []): Promise<LoginProbe> {
  try {
    if (id === "claude-code") {
      const r = await spawner.run(cliPath, [...cliArgs, "auth", "status"], { env, timeoutMs: 15_000 });
      const s = JSON.parse(r.stdout) as { loggedIn?: boolean; subscriptionType?: string; authMethod?: string };
      if (typeof s.loggedIn !== "boolean") return {};
      const plan = s.subscriptionType ? `Claude ${s.subscriptionType[0]!.toUpperCase()}${s.subscriptionType.slice(1)}` : undefined;
      return s.loggedIn ? { loggedIn: true, ...(plan ? { account: plan } : s.authMethod ? { account: s.authMethod } : {}) } : { loggedIn: false };
    }
    if (id === "codex") {
      const r = await spawner.run(cliPath, [...cliArgs, "login", "status"], { env, timeoutMs: 15_000 });
      if (r.code === null || r.code < 0) {
        return { loggedIn: false, reason: t("agent.codexLoginCheck", { code: r.code ?? "?" }) };
      }
      const text = `${r.stdout}\n${r.stderr}`;
      const config = /Error loading config[^:]*:\s*([\s\S]+)/i.exec(text)?.[1]?.replace(/\s+/g, " ").trim().replace(/\.$/, "");
      if (r.code !== 0 && config) return { loggedIn: false, configError: true, reason: t("agent.codexConfig", { error: config }) };
      // Newer Codex reports an API key as "API key configured - sk-…" rather than "Logged in using an API key" (#785).
      if (r.code === 0 && /api key configured/i.test(text)) return { loggedIn: true, account: "API key" };
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
export function resumeCommand(backend: BackendId, cwd: string, sessionId: string): string | null {
  if (backend === "custom") return null;
  const quoted = `'${cwd.replace(/'/g, `'\\''`)}'`;
  return `cd ${quoted} && ${BACKENDS[backend].resume(sessionId)}`;
}
