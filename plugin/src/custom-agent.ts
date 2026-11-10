import type { CustomAgentConfig } from "./types.ts";

/** Validate both saved preferences and the settings form before launching a process. */
export function readCustomAgent(value: unknown): CustomAgentConfig {
  const v = value as Partial<CustomAgentConfig> | undefined;
  if (!v || typeof v.command !== "string" || !v.command.trim() || v.command.includes("\0")) {
    throw new Error("Set the custom agent's command in Settings > Agent first.");
  }
  const args = v.args ?? [];
  const env = v.env ?? {};
  if (!Array.isArray(args) || args.some((a) => typeof a !== "string" || a.includes("\0"))) {
    throw new Error("Custom agent arguments must be a JSON array of strings.");
  }
  if (typeof env !== "object" || env === null || Array.isArray(env) || Object.entries(env).some(
    ([k, x]) => !/^[A-Za-z_][A-Za-z0-9_()]*$/.test(k) || typeof x !== "string" || x.includes("\0"),
  )) throw new Error("Custom agent environment must be a JSON object of variable names and string values.");
  return { label: typeof v.label === "string" && v.label.trim() ? v.label.trim() : "Custom agent", command: v.command.trim(), args: [...args], env: { ...env } };
}
