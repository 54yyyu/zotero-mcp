import type { BackendId, PanelSettings, PromptEntry } from "../types.ts";
import { DEFAULT_APPEARANCE, readAppearance } from "../ui/appearance.ts";
import { LANGUAGES, languageForLocale } from "../ui/settings-model.ts";

export const DEFAULT_PROMPTS: PromptEntry[] = [
  { id: "summary-detailed", slot: 1, title: "Detailed summary", text: "Give a detailed summary of this paper: the question, the data and method, the main results, and the limitations. Cite the pages." },
  { id: "summary-short", slot: 2, title: "Short summary", text: "Summarize this paper in five sentences, then list its three most important claims with page citations." },
  { id: "hypotheses", slot: 3, title: "Propose testable hypotheses", text: "Based on this paper, propose testable hypotheses that follow from its findings, and say what data would test each." },
  { id: "compare", slot: 4, title: "Compare key findings to other studies", text: "Compare this paper's key findings with related studies in my library, and with recent work outside it. Cite everything." },
];

const each = <T>(v: T): Record<BackendId, T> => ({ "claude-code": v, codex: v, pi: v });

export const DEFAULT_SETTINGS: PanelSettings = {
  backend: "claude-code",
  // "" = the backend's own default; the pickers list what each backend really offers.
  model: each(""),
  mode: each(""),
  effort: each(""),
  auth: { "claude-code": "subscription", codex: "subscription", pi: "api-key" },
  prompts: DEFAULT_PROMPTS,
  skills: {},
  followFocus: true,
  attachSelection: true,
  attachAreas: true,
  enterToSend: true,
  showThinking: true,
  expandTools: false,
  showUsage: false,
  openAtStart: false,
  welcomed: false,
  chatFolder: "",
  translate: true,
  translateTo: "en", // withDefaults puts Zotero's own language here
  translateModel: each(""),
  appearance: DEFAULT_APPEARANCE,
};

/**
 * Saved settings laid over the defaults; the per-backend maps merge key by key so a new backend never has holes, and the
 * appearance is validated field by field. The translate target is a known language, else the one `locale` (Zotero's UI
 * language) reads.
 */
export function withDefaults(saved: Partial<PanelSettings>, locale?: string): PanelSettings {
  const d = DEFAULT_SETTINGS;
  return {
    ...d, ...saved,
    model: { ...d.model, ...saved.model }, mode: { ...d.mode, ...saved.mode },
    effort: { ...d.effort, ...saved.effort }, auth: { ...d.auth, ...saved.auth },
    skills: saved.skills && typeof saved.skills === "object" ? saved.skills : {},
    translate: typeof saved.translate === "boolean" ? saved.translate : d.translate,
    translateTo: LANGUAGES.some((l) => l.id === saved.translateTo) ? saved.translateTo! : languageForLocale(locale),
    translateModel: { ...d.translateModel, ...saved.translateModel },
    appearance: readAppearance(saved.appearance),
  };
}
