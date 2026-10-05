// Diagrams: the pure half (parse, sanitize, theme, fit, export). The DOM half is checked in behavior.mjs and in Zotero.
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { PALETTE, exportPalette, fitSvg, idPrefix, paletteName, parseSvg, prepareSvg, sanitizeSvg, serializeSvg } from "../../src/ui/diagram-svg.ts";
import type { SvgEl, SvgNode } from "../../src/ui/diagram-svg.ts";
import { CAUSAL, HOSTILE, MATRIX, PIPELINE } from "../../src/ui/fake-diagrams.ts";
import { mdToTree, walk } from "../../src/ui/markdown.ts";
import { DRAWING_GUIDE } from "../../src/agent/brief.ts";

const all = (n: SvgEl): SvgEl[] => [n, ...n.kids.flatMap((k: SvgNode) => (typeof k === "string" ? [] : all(k)))];
const texts = (n: SvgEl): string => n.kids.map((k) => (typeof k === "string" ? k : texts(k))).join("");
const clean = (src: string) => { const t = parseSvg(src); return t && sanitizeSvg(t, "p"); };

describe("diagram parser", () => {
  it("reads attributes in every quoting style, self-closing tags, CDATA and entities", () => {
    const t = parseSvg(`<svg viewBox='0 0 10 10'><rect x=1 y="2" width='3'/><text>a &amp; b &lt; c <![CDATA[<raw>]]></text></svg>`) as SvgEl;
    assert.equal(t.tag, "svg");
    assert.equal(t.attrs.viewBox, "0 0 10 10");
    const rect = t.kids[0] as SvgEl;
    assert.deepEqual(rect.attrs, { x: "1", y: "2", width: "3" });
    assert.equal(texts(t), "a & b < c <raw>");
  });
  it("is lenient: a bare &, an unclosed tag and a stray close tag still give the drawing", () => {
    const t = clean(`<svg viewBox="0 0 10 10"><g><rect width="2" height="2"/></span><text>R&D</text>`) as SvgEl;
    assert.ok(t);
    assert.deepEqual(all(t).map((e) => e.tag), ["svg", "g", "rect", "text"]);
    assert.equal(texts(t), "R&D");
  });
  it("finds the svg after leading prose and returns null when there is none", () => {
    assert.equal(parseSvg("Here: <svg><circle r='1'/></svg>")?.tag, "svg");
    assert.equal(parseSvg("<div>no drawing</div>"), null);
    assert.equal(parseSvg(""), null);
  });
  it("never expands a DTD entity and survives huge or deep input quickly", () => {
    const t0 = Date.now();
    const lol = `<!DOCTYPE x [<!ENTITY a "AAAAAAAAAA"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;">]><svg><text>&b;</text><rect/></svg>`;
    assert.equal(texts(parseSvg(lol) as SvgEl), "&b;");
    const deep = "<svg>" + "<g>".repeat(5000) + "<rect/>" + "</g>".repeat(5000) + "</svg>";
    assert.ok(parseSvg(deep));
    const wide = "<svg>" + "<rect/>".repeat(20000) + "</svg>";
    assert.ok(all(parseSvg(wide) as SvgEl).length <= 6001);
    assert.equal(parseSvg("<svg>" + "x".repeat(400_000) + "</svg>"), null);
    assert.ok(Date.now() - t0 < 1500, `${Date.now() - t0} ms`);
  });
});

describe("diagram sanitizer", () => {
  const out = clean(HOSTILE) as SvgEl;
  const els = all(out);
  const attrs = els.flatMap((e) => Object.entries(e.attrs));

  it("keeps only drawing elements: no script, style, foreignObject, image, a, animate", () => {
    assert.deepEqual([...new Set(els.map((e) => e.tag))].sort(), ["circle", "rect", "svg", "text"]);
    assert.ok(!texts(out).includes("pwned"), "no script text leaks out as text");
    assert.ok(!texts(out).includes("click me"), "a link's content goes with it");
  });
  it("drops event handlers, classes, fonts, external urls and every non-# href", () => {
    for (const [k, v] of attrs) {
      assert.ok(!/^on/i.test(k), k);
      assert.ok(!["class", "style", "font-family", "filter"].includes(k), k);
      assert.ok(!/javascript:|https?:|evil/i.test(v), `${k}=${v}`);
      if (k === "href") assert.match(v, /^#/);
    }
    const rect = els.find((e) => e.tag === "rect") as SvgEl;
    assert.equal(rect.attrs.fill, undefined, "url(javascript:) fill is dropped, not kept");
  });
  it("turns a safe style declaration into a themed attribute", () => {
    const circle = els.find((e) => e.tag === "circle") as SvgEl;
    assert.equal(circle.attrs.fill, "var(--dg-accent)");
    assert.equal(circle.attrs.stroke, "var(--dg-red)");
  });
  it("namespaces ids and rewrites local references; foreign references go", () => {
    const t = clean(`<svg><defs><marker id="m"><path d="M0 0"/></marker></defs><path d="M0 0" marker-end="url(#m)" clip-path="url(#nope)"/><use href="#m"/><use xlink:href="#m"/><path id="bad id" d="M1 1" fill="url('#m')"/></svg>`) as SvgEl;
    const e = all(t);
    const [, main, bad] = e.filter((x) => x.tag === "path");
    assert.equal(e.find((x) => x.tag === "marker")?.attrs.id, "p-m");
    assert.equal(main?.attrs["marker-end"], "url(#p-m)");
    assert.equal(main?.attrs["clip-path"], "url(#p-nope)", "an unknown local id is still namespaced, so it can never reach the panel");
    assert.deepEqual(e.filter((x) => x.tag === "use").map((x) => x.attrs.href), ["#p-m", "#p-m"]);
    assert.equal(bad?.attrs.id, undefined);
    assert.equal(bad?.attrs.fill, "url(#p-m)");
    assert.notEqual(idPrefix(PIPELINE), idPrefix(MATRIX));
  });
  it("rejects data: and javascript: wherever they hide, even in otherwise allowed attributes", () => {
    for (const bad of ["javascript:1", " JaVaScRiPt:1", "data:text/html,x", "url(data:image/png;base64,AA)", "expression(alert(1))", "url( 'https://x' )", "url(#a) url(https://x)"]) {
      const t = clean(`<svg><rect width="1" height="1" fill="${bad}" mask="${bad}" transform="${bad}"/></svg>`) as SvgEl;
      assert.deepEqual((t.kids[0] as SvgEl).attrs, { width: "1", height: "1" }, bad);
    }
  });
  it("is null for something with no drawing in it", () => {
    assert.equal(clean("<svg><script>x</script></svg>"), null);
    assert.equal(clean("<svg><title>only a title</title></svg>"), null);
    assert.equal(prepareSvg("not svg at all"), null);
  });
});

describe("diagram colours", () => {
  it("palette names are kept, case-insensitively", () => {
    for (const n of PALETTE) assert.equal(paletteName(n.toUpperCase(), "fill"), n);
  });
  it("greys map to ink, muted, line and surface by lightness", () => {
    assert.equal(paletteName("#000", "fill"), "ink");
    assert.equal(paletteName("#333333", "stroke"), "ink");
    assert.equal(paletteName("#666", "fill"), "muted");
    assert.equal(paletteName("#888888", "stroke"), "muted");
    assert.equal(paletteName("#bbbbbb", "stroke"), "line");
    assert.equal(paletteName("#f5f5f5", "fill"), "surface");
    assert.equal(paletteName("white", "stroke"), "line");
    assert.equal(paletteName("rgb(240, 240, 240)", "fill"), "surface");
  });
  it("hues map to the nearest named colour; light tints become -soft fills", () => {
    const cases: [string, string, string][] = [
      ["#d62728", "fill", "red"], ["red", "stroke", "red"], ["#ff7f0e", "fill", "orange"], ["gold", "stroke", "orange"],
      ["#2ca02c", "stroke", "green"], ["#17becf", "fill", "teal"], ["#1f77b4", "stroke", "accent"], ["steelblue", "fill", "accent"],
      ["#9467bd", "fill", "violet"], ["magenta", "stroke", "violet"], ["#cfe2ff", "fill", "accent-soft"], ["#cfe2ff", "stroke", "accent"],
      ["hsl(140, 60%, 90%)", "fill", "green-soft"], ["rgba(255,0,0,0.3)", "fill", "red"], ["#fde2e2", "fill", "red-soft"],
    ];
    for (const [v, prop, want] of cases) assert.equal(paletteName(v, prop), want, `${v} ${prop}`);
  });
  it("anything else is not a colour", () => {
    for (const v of ["", "inherit", "url(#a)", "var(--x)", "#12", "blah"]) assert.equal(paletteName(v, "fill"), null, v);
  });
  it("the samples end up with palette colours only (the stray #888888 included)", () => {
    for (const src of [PIPELINE, MATRIX, CAUSAL]) {
      const p = prepareSvg(src);
      assert.ok(p);
      for (const e of all(p.svg)) for (const k of ["fill", "stroke", "stop-color", "color"]) {
        const v = e.attrs[k];
        if (v !== undefined) assert.match(v, /^(var\(--dg-[a-z]+(-soft)?\)|none|currentColor|url\(#[\w.-]+\))$/, `${k}=${v}`);
      }
    }
    assert.ok(JSON.stringify(prepareSvg(CAUSAL)).includes("var(--dg-muted)"));
  });
});

describe("diagram fit", () => {
  const fit = (src: string) => { const t = clean(src) as SvgEl; return { t, f: fitSvg(t) }; };
  it("keeps a good viewBox, drops width and height", () => {
    const { t, f } = fit(`<svg viewBox="0 0 360 200" width="720" height="400"><circle r="3"/></svg>`);
    assert.deepEqual(f.box, [0, 0, 360, 200]);
    assert.equal(t.attrs.width, undefined);
    assert.equal(t.attrs.height, undefined);
  });
  it("makes a viewBox from width and height, or a default", () => {
    assert.deepEqual(fit(`<svg width="300" height="150"><circle r="3"/></svg>`).f.box, [0, 0, 300, 150]);
    assert.deepEqual(fit(`<svg width="100%" height="100%"><circle r="3"/></svg>`).f.box, [0, 0, 360, 240]);
    assert.deepEqual(fit(`<svg viewBox="0 0 0 10"><circle r="3"/></svg>`).f.box, [0, 0, 360, 240]);
    assert.equal(fit(`<svg viewBox="0,0,200,100"><circle r="3"/></svg>`).t.attrs.viewBox, "0 0 200 100");
  });
  it("drops a full-size background rect (first child, or first in the first group), keeps real boxes", () => {
    const tags = (t: SvgEl) => all(t).map((e) => e.tag).join(" ");
    assert.equal(tags(fit(`<svg viewBox="0 0 100 50"><rect width="100" height="50" fill="white"/><circle r="3"/></svg>`).t), "svg circle");
    assert.equal(tags(fit(`<svg viewBox="0 0 100 50"><title>t</title><rect width="100%" height="100%"/><circle r="3"/></svg>`).t), "svg title circle");
    assert.equal(tags(fit(`<svg viewBox="0 0 100 50"><g><rect x="0" y="0" width="100" height="50"/><circle r="3"/></g></svg>`).t), "svg g circle");
    assert.equal(tags(fit(`<svg viewBox="0 0 100 50"><rect x="10" y="10" width="40" height="20"/><circle r="3"/></svg>`).t), "svg rect circle");
    assert.equal(tags(fit(`<svg viewBox="0 0 100 50"><circle r="3"/><rect width="100" height="50"/></svg>`).t), "svg circle rect", "only a leading rect is a background");
  });
});

describe("diagram export", () => {
  it("bakes the light palette into a standalone file", () => {
    const p = prepareSvg(PIPELINE);
    assert.ok(p);
    const pal = exportPalette();
    const s = serializeSvg(p.svg, p.fit, pal);
    assert.match(s, /^<svg [^>]*xmlns="http:\/\/www.w3.org\/2000\/svg"/);
    assert.match(s, /width="360" height="132"/);
    assert.ok(!s.includes("var("), "no CSS variables left");
    assert.ok(s.includes(`fill="${pal["accent-soft"]}"`) && s.includes(`stroke="${pal.accent}"`));
    assert.ok(s.includes("font-family="));
    assert.ok(s.includes(">From question to cited answer</title>"));
  });
  it("escapes text and attribute values, and takes a custom accent", () => {
    const p = prepareSvg(`<svg viewBox="0 0 10 10"><text aria-label='a"b'>x &lt; y &amp; "z"</text></svg>`);
    assert.ok(p);
    const s = serializeSvg(p.svg, p.fit, exportPalette("#AA0000"));
    assert.ok(s.includes('aria-label="a&quot;b"'));
    assert.ok(s.includes('>x &lt; y &amp; "z"</text>'));
    assert.equal(exportPalette("#AA0000").accent, "#aa0000");
    assert.equal(exportPalette("not a colour").accent, exportPalette().accent);
  });
  it("soft fills are opaque tints of their hue", () => {
    const pal = exportPalette();
    for (const n of PALETTE) assert.match(pal[n] as string, /^#[0-9a-f]{6}$/, n);
  });
});

describe("svg fences in markdown", () => {
  it("a closed ```svg block is a diagram; an unclosed one is open (still streaming)", () => {
    const closed = [...walk(mdToTree("a\n\n```svg\n<svg/>\n```\n\nb"))].find((e) => e.tag === "diagram");
    assert.deepEqual(closed, { tag: "diagram", kids: ["<svg/>"] });
    const open = [...walk(mdToTree("a\n\n```svg\n<svg><rect"))].find((e) => e.tag === "diagram");
    assert.equal(open?.attrs?.open, "1");
    assert.equal([...walk(mdToTree("```xml\n<svg/>\n```"))].find((e) => e.tag === "diagram"), undefined);
  });
});

describe("the drawing guide", () => {
  it("is short and names exactly the palette the panel themes", () => {
    assert.ok(DRAWING_GUIDE.split(/\s+/).length <= 90, `${DRAWING_GUIDE.split(/\s+/).length} words`);
    assert.ok(DRAWING_GUIDE.includes("```svg"));
    for (const n of PALETTE.filter((x) => !x.endsWith("-soft"))) assert.ok(new RegExp(`\\b${n}\\b`).test(DRAWING_GUIDE), n);
    assert.ok(DRAWING_GUIDE.includes("-soft"));
  });
});
