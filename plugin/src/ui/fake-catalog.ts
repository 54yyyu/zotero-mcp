// What each fake backend offers (names and ids as the real bridges report them) and the default settings.
import type { BackendId, Catalog, PanelSettings } from "../types.ts";
import { DEFAULT_APPEARANCE } from "./appearance.ts";

const level = (id: string, name: string, description?: string) => ({ id, name, ...(description ? { description } : {}) });

export const CATALOGS: Record<BackendId, Catalog> = {
  "claude-code": {
    models: [
      level("opus", "Claude Opus", "Most capable, slower"),
      level("sonnet", "Claude Sonnet", "Balanced for everyday work"),
      level("haiku", "Claude Haiku", "Fastest for light tasks"),
    ],
    modes: [
      level("default", "Ask first"),
      level("acceptEdits", "Auto-edit"),
      level("plan", "Plan"),
      level("auto", "Auto"),
      level("bypassPermissions", "Allow all"),
    ],
    efforts: [level("low", "Low"), level("medium", "Medium"), level("high", "High"), level("xhigh", "Extra high"), level("max", "Max")],
    model: "sonnet", mode: "default", effort: "medium",
  },
  codex: {
    models: [level("gpt-5-codex", "GPT-5 Codex", "Tuned for coding agents"), level("gpt-5", "GPT-5")],
    modes: [
      level("read-only", "Read only", "Codex's own words: read files, ask for anything else"),
      level("workspace-write", "Workspace write"),
      level("agent", "Agent"),
      level("agent-full-access", "Full access"),
    ],
    efforts: [level("low", "Low"), level("medium", "Medium"), level("high", "High"), level("max", "Max")],
    model: "gpt-5-codex", mode: "agent", effort: "medium",
  },
  pi: {
    models: [level("provider-default", "Provider default"), level("small", "Small and fast")],
    modes: [],
    efforts: [level("off", "Off"), level("minimal", "Minimal"), level("low", "Low"), level("medium", "Medium"), level("high", "High"), level("xhigh", "Extra high")],
    model: "provider-default", effort: "off",
  },
};

export function defaultSettings(): PanelSettings {
  return {
    backend: "claude-code",
    model: { "claude-code": "", codex: "", pi: "" },
    mode: { "claude-code": "", codex: "", pi: "" },
    effort: { "claude-code": "", codex: "", pi: "" },
    auth: { "claude-code": "subscription", codex: "subscription", pi: "subscription" },
    prompts: [
      { id: "p1", title: "Detailed summary", text: "Write a detailed summary of this paper.", slot: 1 },
      { id: "p2", title: "Short summary", text: "Summarize this paper in five sentences.", slot: 2 },
      { id: "p3", title: "Propose testable hypotheses", text: "Propose testable hypotheses that follow from this paper.", slot: 3 },
      { id: "p4", title: "Compare key findings to other studies", text: "Compare the key findings to other studies in my library.", slot: 4 },
    ],
    followFocus: true, attachSelection: true, attachAreas: true,
    enterToSend: true, showThinking: true, expandTools: false, showUsage: false, openAtStart: false, welcomed: true, chatFolder: "",
    appearance: { ...DEFAULT_APPEARANCE },
  };
}
