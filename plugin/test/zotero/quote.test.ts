import { test } from "node:test";
import assert from "node:assert/strict";
import { findQuote, fold, rangeRects } from "../../src/zotero/quote.ts";
import type { PdfChar } from "../../src/zotero/quote.ts";

/** A page the way the reader stores it: one char per glyph, spaces as spaceAfter, lines 12 units apart. */
function page(lines: string[], rotation = 0): PdfChar[] {
  const out: PdfChar[] = [];
  lines.forEach((line, li) => {
    const y1 = 700 - li * 12, y2 = y1 + 10;
    const glyphs = [...line].map((c, x) => ({ c, x })).filter((g) => g.c !== " ");
    const lastX = glyphs.at(-1)!.x;
    for (const g of glyphs) {
      out.push({ c: g.c, rect: [72 + g.x * 5, y1 + 1, 77 + g.x * 5, y2 - 1], inlineRect: [72, y1, 77 + lastX * 5, y2], rotation, lineBreakAfter: g.x === lastX });
    }
  });
  return out;
}
const text = (chars: PdfChar[], [a, b]: [number, number]) => chars.slice(a, b + 1).map((c) => c.c).join("");

const PAGE = page([
  "Applicants with white-sounding names re-",
  "ceived 50 percent more callbacks for “ﬁrst” inter-",
  "views. This gap is uniform across occupations.",
]);

test("fold drops spacing, punctuation, case, ligatures, accents and quote styles", () => {
  assert.equal(fold("“Ｆirst”  ﬁeld-work, Café!"), "firstfieldworkcafe");
});

test("a quote across a hyphenated line break, with curly quotes and a ligature on the page", () => {
  const hit = findQuote(PAGE, `white-sounding names received 50 percent more callbacks for "first" interviews`);
  assert.ok(hit);
  assert.equal(text(PAGE, hit), "white-soundingnamesre-ceived50percentmorecallbacksfor“ﬁrst”inter-views");
});

test("falls back to the quote's first words, then its last words", () => {
  const first = findQuote(PAGE, "Applicants with white-sounding names received 50 percent more ... and a made up ending nobody wrote here");
  assert.ok(first);
  assert.ok(text(PAGE, first).startsWith("Applicants"));
  const last = findQuote(PAGE, "a beginning the agent made up entirely, inter- views. This gap is uniform across occupations.");
  assert.ok(last);
  assert.equal(text(PAGE, last), "inter-views.Thisgapisuniformacrossoccupations");
});

test("no match, or a quote too short to be specific, is null", () => {
  assert.equal(findQuote(PAGE, "results were not significant at all in this study"), null);
  assert.equal(findQuote(PAGE, "the gap"), null);
  assert.equal(findQuote([], "anything at all that is long enough"), null);
});

test("ignorable chars are skipped", () => {
  const chars = page(["abc defghij klmnop qrstu"]);
  chars.splice(3, 0, { c: "Z", rect: [0, 0, 1, 1], ignorable: true });
  assert.ok(findQuote(chars, "abc defghij klmnop qrstu"));
});

test("one rect per line: the line's height, first to last char of the run", () => {
  const hit = findQuote(PAGE, "names received 50 percent more callbacks")!;
  const rects = rangeRects(PAGE, hit[0], hit[1]);
  assert.equal(rects.length, 2);
  const [l1, l2] = rects as [number[], number[]];
  assert.deepEqual([l1[1], l1[3]], [700, 710]); // line 1 box
  assert.deepEqual([l2[1], l2[3]], [688, 698]); // line 2 box
  assert.equal(l1[0], 72 + "Applicants with white-sounding ".length * 5); // starts at "names"
  assert.equal(l2[0], 72); // line 2 starts at its margin
  assert.equal(l2[2], 77 + "ceived 50 percent more callback".length * 5); // ends at the last "s"
});

test("vertical text takes the line box across x", () => {
  const chars = page(["abcdefghijklmnopqrstuvwxyz"], 90);
  const r = rangeRects(chars, 2, 5);
  assert.deepEqual(r, [[72, chars[2]!.rect[1], 77 + 25 * 5, chars[5]!.rect[3]]]);
});
