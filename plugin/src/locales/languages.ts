// Which interface language is in force, without the catalogs: plugin.js (read at Zotero's startup) needs the answer for
// its two words, and must not pay for every catalog to get it.

/** The interface languages, each named as it names itself; `beta` ones await a native speaker's review. */
export const UI_LANGUAGES: { id: string; label: string; beta?: boolean }[] = [
  { id: "en", label: "English" },
  { id: "zh-Hans", label: "简体中文" },
  { id: "zh-Hant", label: "繁體中文" },
  { id: "ja", label: "日本語", beta: true },
  { id: "de", label: "Deutsch", beta: true },
  { id: "fr", label: "Français", beta: true },
  { id: "es", label: "Español", beta: true },
  { id: "pt-BR", label: "Português (Brasil)", beta: true },
];

const known = (id: string | undefined): id is string => !!id && UI_LANGUAGES.some((l) => l.id === id);

/** The interface language a Zotero locale ("zh-CN", "pt-PT", "de-AT") reads: English for one we have no catalog for. */
export function uiLanguageFor(locale: string | undefined): string {
  const [lang = "", region = ""] = (locale ?? "").replace("_", "-").toLowerCase().split("-");
  if (lang === "zh") return /^(tw|hk|mo|hant)$/.test(region) ? "zh-Hant" : "zh-Hans";
  if (lang === "pt") return "pt-BR";
  return known(lang) ? lang : "en";
}

/** The "Interface language" setting ("" follows Zotero) over Zotero's locale. */
export const resolveLanguage = (setting: string | undefined, locale: string | undefined): string => (known(setting) ? setting : uiLanguageFor(locale));

/** The words plugin.js shows before panel.js is ever read: the toolbar button's tooltip and the reader's Translate button. */
export const STARTUP_WORDS: Record<string, { chat: string; translate: string }> = {
  en: { chat: "Chat", translate: "Translate" },
  "zh-Hans": { chat: "聊天", translate: "翻译" },
  "zh-Hant": { chat: "聊天", translate: "翻譯" },
  ja: { chat: "チャット", translate: "翻訳" },
  de: { chat: "Chat", translate: "Übersetzen" },
  fr: { chat: "Discussion", translate: "Traduire" },
  es: { chat: "Chat", translate: "Traducir" },
  "pt-BR": { chat: "Chat", translate: "Traduzir" },
};

export const startupWord = (word: "chat" | "translate", setting: string | undefined, locale: string | undefined): string =>
  STARTUP_WORDS[resolveLanguage(setting, locale)]![word];
