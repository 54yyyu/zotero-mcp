import type { CustomAgentConfig, CustomAgents, CustomBackendId } from "./types.ts";
import { t } from "./i18n.ts";

export const isCustomBackend = (id: unknown): id is CustomBackendId =>
  typeof id === "string" && /^custom(?:-[a-z0-9-]+)?$/.test(id);

export const customAgentIds = (agents: CustomAgents): CustomBackendId[] =>
  Object.keys(agents).filter(isCustomBackend);

/** Changing the display name or the order of environment keys does not change how the agent runs. */
export function customLaunchKey(config?: CustomAgentConfig): string | undefined {
  if (!config) return undefined;
  return JSON.stringify([config.command, config.args, Object.entries(config.env ?? {}).sort(([a], [b]) => a.localeCompare(b))]);
}

/** Validate both saved preferences and the settings form before launching a process. */
export function readCustomAgent(value: unknown): CustomAgentConfig {
  const v = value as Partial<CustomAgentConfig> | undefined;
  if (!v || typeof v.command !== "string" || !v.command.trim() || v.command.includes("\0")) {
    throw new Error(t("agent.custom.noCommand"));
  }
  const args = v.args ?? [];
  const env = v.env ?? {};
  if (!Array.isArray(args) || args.some((a) => typeof a !== "string" || a.includes("\0"))) {
    throw new Error(t("agent.custom.badArgs"));
  }
  if (typeof env !== "object" || env === null || Array.isArray(env) || Object.entries(env).some(
    ([k, x]) => !/^[A-Za-z_][A-Za-z0-9_()]*$/.test(k) || typeof x !== "string" || x.includes("\0"),
  )) throw new Error(t("agent.custom.badEnv"));
  return { label: typeof v.label === "string" ? v.label.trim() : "", command: v.command.trim(), args: [...args], env: { ...env } };
}
