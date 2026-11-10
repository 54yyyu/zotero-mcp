// The panel's own words in the user's language. `t("key", { n, name })` looks the key up in the language in force
// (Zotero's, or the "Interface language" setting), falling back to English for a key a catalog lacks. A message with a
// count is an object of plural forms (Intl.PluralRules picks one by `n`); `{name}` is replaced by `vars.name`.
// Pure strings: ui/, zotero/ and agent/ may all import it. What the agent reads (the brief, the context block) stays
// English; only what a person reads is translated. Catalogs, glossary and how to add a language: src/locales/README.md.
// (Which language is in force is src/locales/languages.ts, which plugin.js reads without the catalogs.)
import en from "./locales/en.ts";
import de from "./locales/de.ts";
import es from "./locales/es.ts";
import fr from "./locales/fr.ts";
import ja from "./locales/ja.ts";
import ptBR from "./locales/pt-BR.ts";
import zhHans from "./locales/zh-Hans.ts";
import zhHant from "./locales/zh-Hant.ts";

export { UI_LANGUAGES, resolveLanguage, uiLanguageFor } from "./locales/languages.ts";

export type Plural = Partial<Record<Intl.LDMLPluralRule, string>> & { other: string };
export type Msg = string | Plural;
export type Key = keyof typeof en;
export type Catalog = Record<Key, Msg>;
export type Vars = Record<string, string | number>;

const CATALOGS: Record<string, Catalog> = { en, "zh-Hans": zhHans, "zh-Hant": zhHant, ja, de, fr, es, "pt-BR": ptBR };

let current = "en";
const rules = new Map<string, Intl.PluralRules>();

/** Switch the language; true when it changed (the caller redraws). */
export function setLanguage(id: string): boolean {
  const next = id in CATALOGS ? id : "en";
  if (next === current) return false;
  current = next;
  return true;
}

export const language = (): string => current;

/** The message for `key` in `lang` (the language in force by default), with `{var}`s filled in and the plural form for `vars.n`. */
export function t(key: Key, vars?: Vars, lang = current): string {
  let m: Msg = CATALOGS[lang]?.[key] ?? en[key];
  if (typeof m !== "string") {
    let pr = rules.get(lang);
    if (!pr) rules.set(lang, (pr = new Intl.PluralRules(lang)));
    m = m[pr.select(Number(vars?.n ?? 0))] ?? m.other;
  }
  return vars ? m.replace(/\{(\w+)\}/g, (all, k: string) => (k in vars ? String(vars[k]) : all)) : m;
}
