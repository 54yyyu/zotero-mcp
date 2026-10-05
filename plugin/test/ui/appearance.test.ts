import { test } from "node:test";
import assert from "node:assert/strict";
import { ACCENTS, BACKGROUNDS, DEFAULT_APPEARANCE, accentVars, contrast, readAppearance, reach, textOn } from "../../src/ui/appearance.ts";
import { withDefaults } from "../../src/zotero/defaults.ts";

test("a missing or broken appearance reads as the default: glass on, our red, no background", () => {
  for (const raw of [undefined, null, "glass", 3, [], {}]) assert.deepEqual(readAppearance(raw), DEFAULT_APPEARANCE);
  assert.deepEqual(withDefaults({}).appearance, DEFAULT_APPEARANCE);
  assert.equal(DEFAULT_APPEARANCE.glass, true);
  assert.equal(DEFAULT_APPEARANCE.accent, "");
});

test("each field is checked on its own: one bad value never resets the others", () => {
  const a = readAppearance({ glass: "yes", accent: "#2563C9", background: "dawn", imageVisibility: 250, imageBlur: -4, textSize: "huge", density: "compact" });
  assert.deepEqual(a, { ...DEFAULT_APPEARANCE, accent: "#2563c9", background: "dawn", imageVisibility: 100, imageBlur: 0, density: "compact" });
  assert.equal(readAppearance({ imageVisibility: 33.6 }).imageVisibility, 34);
  assert.equal(readAppearance({ imageVisibility: Number.NaN }).imageVisibility, DEFAULT_APPEARANCE.imageVisibility);
  assert.equal(readAppearance({ glass: false }).glass, false);
});

test("accent: only #rrggbb, and our red is stored as the default", () => {
  for (const bad of ["red", "#fff", "#12345g", "url(x)", "#1234567", 0x2563c9]) assert.equal(readAppearance({ accent: bad }).accent, "", String(bad));
  assert.equal(readAppearance({ accent: "#CC2936" }).accent, "");
});

test("background: none, a preset, or the picture only when one is saved", () => {
  for (const b of BACKGROUNDS) assert.equal(readAppearance({ background: b.id }).background, b.id);
  assert.equal(readAppearance({ background: "plaid" }).background, "none");
  assert.equal(readAppearance({ background: "image" }).background, "none", "no picture saved");
  assert.deepEqual([readAppearance({ background: "image", image: "kyoto.jpg" }).background, readAppearance({ image: "x".repeat(500) }).image.length], ["image", 120]);
});

test("contrast is WCAG's", () => {
  assert.equal(Math.round(contrast("#000000", "#ffffff")), 21);
  assert.equal(contrast("#777777", "#777777"), 1);
  assert.ok(Math.abs(contrast("#cc2936", "#ffffff") - 5.33) < 0.05);
  assert.equal(textOn("#cc2936"), "#ffffff");
  assert.equal(textOn("#ffd43b"), "#16181d", "dark text on a light yellow");
  assert.equal(reach("#cc2936", "#ffffff", 4.5, "#000000"), "#cc2936", "already enough: unchanged");
});

test("any accent stays readable: links 4.5:1 on the paper, text on the fill picks the better of white and ink", () => {
  const customs = ["#ffd43b", "#ffffff", "#000000", "#0d0d0d", "#e9ecef", "#ff6b6b", "#20c997", "#4dabf7", "#7950f2", "#868e96"];
  for (const hex of [...ACCENTS.map((a) => a.id), ...customs]) {
    const v = accentVars(hex);
    assert.ok(contrast(v["--accent-ink-l"]!, "#f5f6f7") >= 4.5, `${hex} light ink ${v["--accent-ink-l"]}`);
    assert.ok(contrast(v["--accent-ink-d"]!, "#262a31") >= 4.5, `${hex} dark ink on a dark glass card ${v["--accent-ink-d"]}`);
    assert.ok(contrast(v["--accent-d"]!, "#131519") >= 3 || ACCENTS.some((a) => a.id === hex), `${hex} dark fill stays visible`);
    for (const t of ["l", "d"] as const) {
      const fill = v[`--accent-${t}`]!, on = v[`--on-accent-${t}`]!;
      assert.ok(contrast(fill, on) >= Math.max(contrast(fill, "#ffffff"), contrast(fill, "#16181d")) - 1e-9, `${hex} on-accent ${t}`);
    }
  }
  assert.deepEqual([accentVars("")["--accent-l"], accentVars("")["--accent-d"]], ["#cc2936", "#ff7b86"], "the default is our red, in both themes");
});
