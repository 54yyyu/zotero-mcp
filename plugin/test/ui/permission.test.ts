// A permission card's buttons (ui/parts.ts optionName): in English the bridge's own words; in another language ours, by
// kind, unless two options share a kind: then only the bridge's words tell them apart.
import assert from "node:assert/strict";
import { after, test } from "node:test";
import { setLanguage } from "../../src/i18n.ts";
import { optionName } from "../../src/ui/parts.ts";

after(() => { setLanguage("en"); });

const once = { id: "a", name: "Yes", kind: "allow_once" };
const command = { id: "b", name: "Yes, and don't ask again for this command", kind: "allow_always" };
const sessionWide = { id: "c", name: "Yes, and don't ask again this session", kind: "allow_always" };
const no = { id: "d", name: "No, keep planning", kind: "reject_once" };

test("in English the bridge's own words", () => {
  setLanguage("en");
  assert.deepEqual([once, command, no].map((o, _i, all) => optionName(o, all)), [once.name, command.name, no.name]);
});

test("in another language: ours by kind, but two options of one kind keep the bridge's words", () => {
  setLanguage("de");
  const plain = [once, command, no];
  const names = plain.map((o) => optionName(o, plain));
  assert.ok(names.every((n, i) => n !== plain[i]!.name), "translated: " + names.join(" | "));
  const twin = [once, command, sessionWide, no];
  const shown = twin.map((o) => optionName(o, twin));
  assert.equal(new Set(shown).size, 4, "no two buttons read the same: " + shown.join(" | "));
  assert.deepEqual([shown[1], shown[2]], [command.name, sessionWide.name]);
  assert.equal(shown[0], names[0], "the kinds that are alone are still translated");
});
