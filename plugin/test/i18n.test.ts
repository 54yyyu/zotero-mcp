// The UI's languages: every catalog has exactly the English keys, the same placeholders and valid plural forms; the
// language in force follows the setting, else Zotero's locale; t() fills in, picks plurals and falls back to English.
import assert from "node:assert/strict";
import { test } from "node:test";
import en from "../src/locales/en.ts";
import { STARTUP_WORDS, startupWord } from "../src/locales/languages.ts";
import { UI_LANGUAGES, language, resolveLanguage, setLanguage, t, uiLanguageFor } from "../src/i18n.ts";
import type { Catalog, Key, Msg } from "../src/i18n.ts";
import { withDefaults } from "../src/zotero/defaults.ts";

const catalogs: Record<string, Catalog> = Object.fromEntries(await Promise.all(UI_LANGUAGES.map(async (l) => [l.id, (await import(`../src/locales/${l.id}.ts`)).default as Catalog])));
const holes = (m: string) => [...m.matchAll(/\{(\w+)\}/g)].map((x) => x[1]).sort();
const forms = (m: Msg): string[] => (typeof m === "string" ? [m] : Object.values(m));

test("every catalog has exactly the English keys", () => {
  const keys = Object.keys(en).sort();
  for (const [id, cat] of Object.entries(catalogs)) assert.deepEqual(Object.keys(cat).sort(), keys, id);
});

test("every message keeps the English placeholders and `code`, and plurals use the language's own categories", () => {
  for (const [id, cat] of Object.entries(catalogs)) {
    const cats = new Intl.PluralRules(id).resolvedOptions().pluralCategories;
    for (const k of Object.keys(en) as Key[]) {
      const src = en[k], msg = cat[k];
      assert.equal(typeof msg, typeof src, `${id} ${k}: plural in one and not the other`);
      const want = holes(typeof src === "string" ? src : src.other);
      if (typeof msg === "string") {
        assert.ok(msg.trim(), `${id} ${k} is empty`);
        assert.deepEqual(holes(msg), want, `${id} ${k}`);
        assert.deepEqual(msg.match(/`[^`]+`/g) ?? [], (src as string).match(/`[^`]+`/g) ?? [], `${id} ${k}: commands stay as they are`);
      } else {
        assert.ok(msg.other, `${id} ${k} has no "other"`);
        for (const [form, text] of Object.entries(msg)) {
          assert.ok(cats.includes(form as Intl.LDMLPluralRule), `${id} ${k}: ${form} is not a plural category of ${id}`);
          // "one" may leave the number out ("once"); every other form names it
          if (form === "one") assert.ok(holes(text!).every((x) => want.includes(x)), `${id} ${k}.one`);
          else assert.deepEqual(holes(text!), want, `${id} ${k}.${form}`);
        }
      }
    }
  }
});

test("no catalog left a message in English by accident (beyond words that are the same in both)", () => {
  // Names, commands and short shared words may match; a long sentence identical to the English is a missed string.
  for (const [id, cat] of Object.entries(catalogs)) {
    if (id === "en") continue;
    const same = (Object.keys(en) as Key[]).filter((k) => forms(en[k]).some((s, i) => s.length > 24 && /[a-z]{3} [a-z]{3}/.test(s) && forms(cat[k])[i] === s));
    assert.deepEqual(same, [], id);
  }
});

test("Zotero's locale picks the catalog; the setting overrides it; anything else is English", () => {
  const cases: [string | undefined, string][] = [
    ["zh-CN", "zh-Hans"], ["zh-SG", "zh-Hans"], ["zh", "zh-Hans"], ["zh-TW", "zh-Hant"], ["zh-HK", "zh-Hant"], ["zh-MO", "zh-Hant"],
    ["ja-JP", "ja"], ["ja", "ja"], ["de-DE", "de"], ["de-AT", "de"], ["fr-FR", "fr"], ["fr-CA", "fr"], ["es-ES", "es"], ["es-419", "es"],
    ["pt-BR", "pt-BR"], ["pt-PT", "pt-BR"], ["en-US", "en"], ["en-GB", "en"], ["ko-KR", "en"], ["it-IT", "en"], ["", "en"], [undefined, "en"], ["zh_TW", "zh-Hant"],
  ];
  for (const [locale, want] of cases) assert.equal(uiLanguageFor(locale), want, String(locale));
  assert.equal(resolveLanguage("", "de-DE"), "de");
  assert.equal(resolveLanguage("ja", "de-DE"), "ja");
  assert.equal(resolveLanguage("ko", "fr-FR"), "fr", "a language we have no catalog for is ignored");
  // the settings: "" follows Zotero, a known language is kept, anything else is dropped
  assert.equal(withDefaults({}).uiLanguage, "");
  assert.equal(withDefaults({ uiLanguage: "zh-Hant" }).uiLanguage, "zh-Hant");
  assert.equal(withDefaults({ uiLanguage: "klingon" }).uiLanguage, "");
});

test("t() fills in, picks the plural form, and falls back to English", () => {
  assert.equal(t("status.attention", { n: 1 }, "en"), "1 thing needs attention");
  assert.equal(t("status.attention", { n: 3 }, "en"), "3 things need attention");
  assert.equal(t("ring.times", { n: 1 }, "en"), "once");
  assert.equal(t("ring.times", { n: 2 }, "en"), "2 times");
  assert.equal(t("history.deleteAsk", { title: "X" }, "xx"), "Delete X?", "an unknown language reads English");
  assert.equal(t("welcome.count", { n: 2 }, "en"), "2 of {total}", "a missing value leaves its placeholder visible");
  setLanguage("zh-Hans");
  try {
    assert.equal(language(), "zh-Hans");
    assert.equal(t("common.cancel"), "取消", "Zotero's own word");
    assert.equal(t("settings.title"), "设置");
    assert.match(t("status.attention", { n: 3 }), /3/);
  } finally {
    setLanguage("en");
  }
  assert.equal(setLanguage("en"), false, "no change, no redraw");
});

test("a new user's prompts are in the panel's language; saved prompts stay as written", () => {
  assert.equal(withDefaults({}).prompts[0]!.title, "Detailed summary");
  const zh = withDefaults({}, "zh-CN").prompts;
  assert.equal(zh.length, 4);
  assert.equal(zh[0]!.title, t("prompt.detailed.title", undefined, "zh-Hans"));
  assert.notEqual(zh[0]!.title, "Detailed summary");
  assert.equal(withDefaults({ uiLanguage: "de" }, "zh-CN").prompts[0]!.title, t("prompt.detailed.title", undefined, "de"));
  const mine = [{ id: "x", title: "Mine", text: "Mine" }];
  assert.deepEqual(withDefaults({ prompts: mine }, "zh-CN").prompts, mine);
});

test("plugin.js has its two startup words in every language", () => {
  assert.deepEqual(Object.keys(STARTUP_WORDS).sort(), UI_LANGUAGES.map((l) => l.id).sort());
  assert.equal(startupWord("translate", "", "de-DE"), "Übersetzen");
  assert.equal(startupWord("chat", "zh-Hans", "en-US"), STARTUP_WORDS["zh-Hans"]!.chat);
});
