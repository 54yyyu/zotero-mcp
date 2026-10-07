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
      level("opus-4-1", "Claude Opus 4.1", "Previous generation"),
      level("sonnet-4", "Claude Sonnet 4", "Previous generation"),
      level("haiku-3-5", "Claude Haiku 3.5", "Oldest, cheapest"),
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

/**
 * pi as it really reports itself when OpenRouter is configured: every OpenRouter model ("openrouter/Vendor: Model"), sorted
 * by id, then the user's own providers from ~/.pi/agent/models.json last; its default (pi's built-in one for OpenRouter)
 * sits in the middle. 418 models, like the real list measured on 2026-10-05 (preview: ?pi=big).
 */
export function bigPiCatalog(): Catalog {
  const vendors: [string, string, string[]][] = [
    ["anthropic", "Anthropic", ["Claude 3 Haiku", "Claude Haiku 4.5", "Claude Opus 4", "Claude Opus 4.1", "Claude Opus 4.5", "Claude Opus 4.6", "Claude Opus 4.8", "Claude Opus 5", "Claude Sonnet 4", "Claude Sonnet 4.5", "Claude Sonnet 5", "Claude Fable 5"]],
    ["deepseek", "DeepSeek", ["DeepSeek V3", "DeepSeek V3.1", "DeepSeek V3.2", "DeepSeek V4", "DeepSeek V4 Flash", "DeepSeek V4 Pro", "DeepSeek R1"]],
    ["google", "Google", ["Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 3 Flash", "Gemini 3 Pro", "Gemini 3.1 Pro", "Gemma 3 27B"]],
    ["meta-llama", "Meta", ["Llama 3.3 70B Instruct", "Llama 4 Maverick", "Llama 4 Scout"]],
    ["mistralai", "Mistral", ["Mistral Large", "Mistral Medium 3", "Devstral Medium", "Codestral"]],
    ["moonshotai", "MoonshotAI", ["Kimi K2", "Kimi K2 Thinking", "Kimi K2.5", "Kimi K2.6", "Kimi K3"]],
    ["openai", "OpenAI", ["GPT-4.1", "GPT-4.1 Mini", "GPT-4o", "GPT-5", "GPT-5 Mini", "GPT-5.4", "GPT-5.5", "o3", "o4 Mini", "gpt-oss-120b"]],
    ["qwen", "Qwen", ["Qwen3 235B A22B", "Qwen3 Coder", "Qwen3.5 Plus", "Qwen3.6 Max"]],
    ["x-ai", "xAI", ["Grok 4", "Grok 4.6", "Grok Code Fast"]],
    ["z-ai", "Z.ai", ["GLM 4.5", "GLM 4.6", "GLM 4.7", "GLM 5", "GLM 5.1", "GLM 5.2", "GLM 5.3"]],
  ];
  const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9.]+/g, "-");
  const models: Catalog["models"] = [];
  for (const [vid, vname, names] of vendors) for (const n of names) models.push(level(`openrouter/${vid}/${slug(n)}`, `openrouter/${vname}: ${n}`));
  // fill to OpenRouter's size with plausible smaller vendors
  for (let i = 0; models.length < 416; i++) models.push(level(`openrouter/vendor-${String.fromCharCode(97 + (i % 26))}/model-${i}`, `openrouter/Vendor ${String.fromCharCode(65 + (i % 26))}: Model ${i}`));
  models.sort((a, b) => (a.id < b.id ? -1 : 1));
  models.push(level("my-cluster/deepseek-v4-flash", "my-cluster/DeepSeek V4 Flash (4xH100)"), level("ollama/qwen3-8b", "ollama/Qwen3 8B"));
  return { models, modes: [], efforts: CATALOGS.pi.efforts, model: "openrouter/moonshotai/kimi-k2.6", effort: "medium" };
}

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
    translate: true, translateTo: "en", translateModel: { "claude-code": "", codex: "", pi: "" },
    appearance: { ...DEFAULT_APPEARANCE },
  };
}
