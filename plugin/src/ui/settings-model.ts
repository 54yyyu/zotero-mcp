// What the settings screen changes, as pure functions from the current settings to a patch for
// `host.setSettings`. No DOM, so the save paths are unit-tested against the fake host.
import type { BackendId, Catalog, ModeOption, PanelSettings, PromptEntry } from "../types.ts";

/** Friendly one-liners for the permission modes we know; anything else shows the catalog's own words. */
const MODE_HELP: Record<string, string> = {
  default: "Asks before editing files or running commands.",
  acceptEdits: "Edits files without asking; still asks before running commands.",
  plan: "Reads and plans. Changes nothing.",
  auto: "Lets the agent judge what needs asking.",
  bypassPermissions: "Never asks. Use with care.",
  "read-only": "Reads files; asks before anything else.",
  "workspace-write": "Edits inside the workspace; asks before going outside it.",
  agent: "Works on its own inside the workspace; asks for the rest.",
  "agent-full-access": "No sandbox and no questions. Use with care.",
};
export const modeHelp = (m: ModeOption): string => MODE_HELP[m.id] ?? m.description ?? "";

export type PerBackend = "model" | "mode" | "effort";
export type FlagKey = "followFocus" | "attachSelection" | "attachAreas" | "enterToSend" | "showThinking" | "expandTools" | "openAtStart";

/** Model, mode and effort are saved per backend: ids from one backend mean nothing to another. */
export const setPerBackend = (s: PanelSettings, key: PerBackend, backend: BackendId, id: string): Partial<PanelSettings> =>
  ({ [key]: { ...s[key], [backend]: id } }) as Partial<PanelSettings>;

/** The value in force: the saved choice, else the backend's own default from its catalog ("" when unknown). */
export const effective = (s: PanelSettings, cat: Catalog | undefined, key: PerBackend, backend: BackendId): string => s[key][backend] || cat?.[key] || "";

export const setFolder = (path: string): Partial<PanelSettings> => ({ chatFolder: path });

/** A long path shortened in the middle ("/Users/me/\u2026/Zotero Chat"), keeping the start and the last folder. */
export function shortPath(p: string, max = 34): string {
  if (p.length <= max) return p;
  const parts = p.split("/");
  const last = parts[parts.length - 1] || parts[parts.length - 2] || p;
  const head = parts.slice(0, 3).join("/");
  const mid = `${head}/\u2026/${last}`;
  if (mid.length <= max) return mid;
  return last.length >= max ? `\u2026${last.slice(-(max - 1))}` : `\u2026/${last}`;
}

export const setAuth = (s: PanelSettings, backend: BackendId, mode: "subscription" | "api-key"): Partial<PanelSettings> => ({ auth: { ...s.auth, [backend]: mode } });
export const setFlag = (key: FlagKey, on: boolean): Partial<PanelSettings> => ({ [key]: on }) as Partial<PanelSettings>;

// prompts: every edit returns a new list; a slot belongs to one prompt at a time
export const editPrompt = (list: PromptEntry[], i: number, patch: Partial<Pick<PromptEntry, "title" | "text">>): PromptEntry[] => list.map((p, k) => (k === i ? { ...p, ...patch } : p));
export const setSlot = (list: PromptEntry[], i: number, slot: number): PromptEntry[] =>
  list.map((p, k) => {
    const { slot: _old, ...rest } = p;
    return k === i ? (slot ? { ...rest, slot } : rest) : slot && p.slot === slot ? rest : p;
  });
export const addPrompt = (list: PromptEntry[], id: string): PromptEntry[] => [...list, { id, title: "", text: "" }];
export const removePrompt = (list: PromptEntry[], i: number): PromptEntry[] => list.filter((_, k) => k !== i);

/** The shortcut reference: [what, keys] for this platform. */
export const shortcuts = (mac: boolean, enterToSend: boolean): [string, string][] => [
  ["Show or hide the panel", mac ? "⌘⌥L" : "Ctrl+Alt+L"],
  ["Run custom prompt 1 to 4", mac ? "⌘⌃ 1–4" : "Ctrl+Alt+1–4"],
  ["Send", enterToSend ? "Enter" : mac ? "⌘Enter" : "Ctrl+Enter"],
  ["New line", enterToSend ? "Shift+Enter" : "Enter"],
  ["Stop the answer", "Esc"],
  ["Add an item or collection", "@"],
];
