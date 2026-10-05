import { test } from "node:test";
import assert from "node:assert/strict";
import { ALLOWED_ATTRS, ALLOWED_CLASSES, ALLOWED_TAGS, MAX_MD, collectSources, citeLabel, citeTitle, decodeEntities, mdToTree, parseZoteroUri, safeHref, safeImageSrc, walk } from "../../src/ui/markdown.ts";
import type { MdNode } from "../../src/ui/markdown.ts";
import { HOSTILE_SHAPES, XSS_CORPUS } from "./corpus.ts";

const DATA_IMG = /^data:image\/(?:png|jpeg);base64,[A-Za-z0-9+/]+={0,2}$/;

/** Every rule the policy states, checked on a tree. Returns the first violation or null. */
function violation(tree: MdNode[]): string | null {
  for (const el of walk(tree)) {
    if (!ALLOWED_TAGS.has(el.tag)) return `tag <${el.tag}>`;
    for (const [k, v] of Object.entries(el.attrs ?? {})) {
      if (!ALLOWED_ATTRS.has(k)) return `attribute ${k} on <${el.tag}>`;
      if (k.toLowerCase().startsWith("on")) return `event handler ${k}`;
      if (k === "href") {
        if (el.tag !== "a" && el.tag !== "cite") return `href on <${el.tag}>`;
        if (!/^(https?|zotero):\/\/\S+$/i.test(v)) return `href ${v}`;
        if (/[\s"'<>`]/.test(v.replace(/%[0-9a-f]{2}/gi, "")) && /["'<>`]/.test(v)) { /* quotes are inert in setAttribute; noted */ }
      }
      if (k === "src" && !(el.tag === "img" && DATA_IMG.test(v))) return `src ${v.slice(0, 40)} on <${el.tag}>`;
      if (k === "class") for (const c of v.split(/\s+/)) if (!ALLOWED_CLASSES.has(c)) return `class ${c}`;
    }
    if (el.tag === "img" && !el.attrs?.src) return "img without src";
  }
  return null;
}

function depth(nodes: MdNode[]): number {
  let d = 0;
  for (const n of nodes) if (typeof n !== "string") d = Math.max(d, 1 + depth(n.kids ?? []));
  return d;
}

test("the XSS corpus renders to a tree that obeys the policy", () => {
  for (const src of XSS_CORPUS) {
    const tree = mdToTree(src);
    assert.equal(violation(tree), null, `corpus entry: ${JSON.stringify(src.slice(0, 80))}`);
  }
});

test("raw HTML becomes text, never elements", () => {
  for (const src of [`<script>alert(1)</script>`, `<img src=x onerror=alert(1)>`, `a <b>bold</b> c`, `<iframe src=x></iframe>`]) {
    const tree = mdToTree(src);
    const tags = [...walk(tree)].map((e) => e.tag);
    for (const bad of ["script", "img", "b", "iframe"]) assert.ok(!tags.includes(bad), `${bad} in ${src}`);
  }
  const text = JSON.stringify(mdToTree(`<script>alert(1)</script>`));
  assert.ok(text.includes("<script>alert(1)</script>"), "the markup survives as visible text");
});

test("links: only http(s) and zotero survive; the rest show their text", () => {
  const dead = [
    "javascript:alert(1)", "JaVaScRiPt:alert(1)", "data:text/html,x", "vbscript:x", "file:///etc/passwd", "about:config",
    "chrome://x/y", "resource://x/y", "blob:https://x/y", "//evil.example", "/relative", "#frag", "mailto:a@b.c", "java\tscript:1", " javascript:1",
  ];
  for (const href of dead) assert.equal(safeHref(href), null, href);
  for (const href of ["https://a.example/p?q=1#h", "http://a.example", "zotero://open-pdf/library/items/ABCD1234?page=3", "HTTPS://A.EXAMPLE"]) assert.equal(safeHref(href), href.trim(), href);
  const t = mdToTree("[plain words](javascript:alert(1))");
  assert.equal([...walk(t)].filter((e) => e.tag === "a" || e.tag === "cite").length, 0);
  assert.ok(JSON.stringify(t).includes("plain words"));
});

test("images: only inline png/jpeg data URLs; everything else is its alt text", () => {
  assert.ok(safeImageSrc("data:image/png;base64,iVBORw0KGgo="));
  assert.ok(safeImageSrc("data:image/jpeg;base64,/9j/4AAQ"));
  for (const s of ["https://a.example/x.png", "data:image/svg+xml;base64,AAAA", "data:image/gif;base64,AAAA", "data:image/png;base64,AA\"onerror=1", "javascript:1", "data:image/png;base64,"]) assert.equal(safeImageSrc(s), null, s);
  const t = mdToTree("![alt words](https://a.example/x.png) ![ok](data:image/png;base64,iVBORw0KGgo=)");
  assert.deepEqual([...walk(t)].filter((e) => e.tag === "img").map((e) => e.attrs?.alt), ["ok"]);
  assert.ok(JSON.stringify(t).includes("alt words"));
});

test("zotero links become citation nodes; others become anchors", () => {
  const t = mdToTree("[Pager et al. 2009, p.8](zotero://open-pdf/library/items/ABCD1234?page=8) and [site](https://a.example)");
  const els = [...walk(t)];
  assert.equal(els.filter((e) => e.tag === "cite").length, 1);
  assert.equal(els.filter((e) => e.tag === "a").length, 1);
});

test("parseZoteroUri reads libraries, groups, pages and annotations", () => {
  assert.deepEqual(parseZoteroUri("zotero://open-pdf/library/items/abcd1234?page=8"), { libraryID: 1, itemKey: "ABCD1234", page: 8, action: "open-pdf" });
  assert.deepEqual(parseZoteroUri("zotero://open-pdf/groups/55/items/ABCD1234?page=2&annotation=ANNO1234"), { libraryID: 55, itemKey: "ABCD1234", page: 2, annotationKey: "ANNO1234", action: "open-pdf" });
  assert.deepEqual(parseZoteroUri("zotero://select/library/items/ABCD1234"), { libraryID: 1, itemKey: "ABCD1234", action: "select" });
  assert.equal(parseZoteroUri("zotero://select/library/collections/ABCD1234"), null);
  assert.equal(parseZoteroUri("zotero://open-pdf/library/items/short"), null);
  assert.equal(parseZoteroUri("https://example.com"), null);
});

test("collectSources lists each cited item once, with its pages", () => {
  const a = "see [Pager et al. 2009, p.8](zotero://open-pdf/library/items/PAGER009?page=8) and [Pager et al. 2009, p.9](zotero://open-pdf/library/items/PAGER009?page=9)";
  const b = "also [Quillian 2017](zotero://select/library/items/QUIL2017) and [again](zotero://open-pdf/library/items/PAGER009?page=8) [x](https://a.example)";
  const s = collectSources([a, b]);
  assert.deepEqual(s.map((x) => [x.itemKey, x.label, x.pages]), [["PAGER009", "Pager et al. 2009", [8, 9]], ["QUIL2017", "Quillian 2017", []]]);
  assert.equal(citeLabel("Smith 2020, pp. 3-5"), "Smith 2020");
  assert.equal(citeLabel("p.8"), "p.8");
});

test("math: $..$, $$..$$, \\(..\\), \\[..\\] and ```math; currency stays text", () => {
  const maths = (src: string) => [...walk(mdToTree(src))].filter((e) => e.tag === "math").map((e) => [e.kids?.[0], e.attrs?.display]);
  assert.deepEqual(maths("inline $x^2$ here"), [["x^2", "0"]]);
  assert.deepEqual(maths("$$\\frac{a}{b}$$"), [["\\frac{a}{b}", "1"]]);
  assert.deepEqual(maths("\\(a+b\\) and \\[c\\]"), [["a+b", "0"], ["c", "1"]]);
  assert.deepEqual(maths("```math\nE=mc^2\n```"), [["E=mc^2", "1"]]);
  assert.deepEqual(maths("it costs $5 and $10 total"), []);
  assert.deepEqual(maths("an unclosed $$ stays text while streaming"), []);
});

test("entities and escapes decode to text, once", () => {
  assert.equal(decodeEntities("a &amp; b &lt;c&gt; &#65; &#x42; &nbsp;|&unknown; &#0; &#55296;"), "a & b <c> A B \u00a0|&unknown; &#0; &#55296;");
  const t = JSON.stringify(mdToTree("a &amp; b \\<c> `x&amp;y`"));
  assert.ok(t.includes("a & b "));
  assert.ok(t.includes("x&amp;y"), "code is verbatim");
});

test("structure: headings, lists, tables, code, task items, quotes", () => {
  const t = mdToTree("# H\n\n- [x] done\n- [ ] todo\n\n3. a\n4. b\n\n> quote\n\n| a | b |\n|:--|--:|\n| 1 | 2 |\n\n```py\nprint(1)\n```\n\n---");
  const tags = [...walk(t)].map((e) => e.tag);
  for (const want of ["h1", "ul", "li", "ol", "blockquote", "tablewrap", "table", "thead", "tbody", "th", "td", "codeblock", "hr"]) assert.ok(tags.includes(want), want);
  const ol = [...walk(t)].find((e) => e.tag === "ol");
  assert.equal(ol?.attrs?.start, "3");
  const cb = [...walk(t)].find((e) => e.tag === "codeblock");
  assert.equal(cb?.attrs?.lang, "py");
  assert.equal(violation(t), null);
});

test("hostile shapes finish in time, stay shallow and keep the policy", () => {
  for (const { name, text, ms } of HOSTILE_SHAPES) {
    const t0 = performance.now();
    const tree = mdToTree(text);
    const dt = performance.now() - t0;
    assert.ok(dt < ms, `${name}: ${Math.round(dt)} ms (limit ${ms})`);
    assert.ok(depth(tree) < 120, `${name}: depth ${depth(tree)}`);
    assert.equal(violation(tree), null, name);
  }
});

test("input past the cap is shown as plain preformatted text", () => {
  const t = mdToTree("a ".repeat(MAX_MD));
  const last = t[t.length - 1];
  assert.ok(last && typeof last !== "string" && last.tag === "pre");
});

test("punctuation after a citation chip is glued to it", () => {
  const t = mdToTree("see [A 2009, p.8](zotero://open-pdf/library/items/ABCD1234?page=8). Then [B](zotero://select/library/items/ABCD1235), and [C](zotero://select/library/items/ABCD1236) more");
  const wraps = [...walk(t)].filter((e) => e.tag === "span" && e.attrs?.class === "md-nobr");
  assert.deepEqual(wraps.map((w) => w.kids?.[1]), [".", ",", undefined].slice(0, 2));
  assert.ok(wraps.every((w) => typeof w.kids?.[0] !== "string" && (w.kids?.[0] as { tag: string }).tag === "cite"));
  assert.equal(violation(t), null);
  assert.equal(collectSources(["see [A 2009, p.8](zotero://open-pdf/library/items/ABCD1234?page=8)."]).length, 1);
});

const Q = "zotero://open-pdf/library/items/ABCD1234?page=8&quote=white%20applicants%20were%20called%20back%20more";
test("a quote rides on the citation link: parsed as plain text, shown as the chip's tooltip, the label unchanged", () => {
  assert.deepEqual(parseZoteroUri(Q), { libraryID: 1, itemKey: "ABCD1234", page: 8, quote: "white applicants were called back more", action: "open-pdf" });
  assert.equal(parseZoteroUri("zotero://open-pdf/library/items/ABCD1234?page=8&quote=a+b%22%3Cscript%3E")?.quote, 'a b"<script>');
  assert.equal(citeTitle(Q), "“white applicants were called back more”");
  assert.equal(citeTitle("zotero://open-pdf/library/items/ABCD1234?page=8"), "open-pdf/library/items/ABCD1234?page=8");
  const cite = [...walk(mdToTree(`See [Pager 2009, p.8](${Q}).`))].find((n) => n.tag === "cite");
  assert.deepEqual(cite, { tag: "cite", attrs: { href: Q }, kids: ["Pager 2009, p.8"] });
});
test("a quote cannot turn a link into anything else", () => {
  for (const bad of ["zotero://open-pdf/library/items/ABCD1234?page=8&quote=a b", "zotero://open-pdf/library/items/ABCD1234?page=8&quote=a\tb", "javascript:alert(1)//zotero://x?quote=1"]) {
    assert.equal(safeHref(bad), null, bad);
  }
  const tree = mdToTree("[x](zotero://open-pdf/library/items/ABCD1234?page=8&quote=%22%3E%3Cimg%20src%3Dx%20onerror%3Dalert(1)%3E)");
  assert.equal(violation(tree), null);
});
test("sources: one row per item, pages once, the fold's link without the quote", () => {
  const [s] = collectSources([`[P 2009, p.8](${Q}) and [P 2009, p.8](zotero://open-pdf/library/items/ABCD1234?page=8&quote=another%20passage%20on%20the%20same%20page) and [P 2009, p.9](zotero://open-pdf/library/items/ABCD1234?quote=x%20y&page=9)`]);
  assert.equal(s?.href, "zotero://open-pdf/library/items/ABCD1234?page=8");
  assert.deepEqual(s?.pages, [8, 9]);
  assert.equal(collectSources(["[a](zotero://open-pdf/library/items/ABCD1234?quote=x%20y&page=9)"])[0]?.href, "zotero://open-pdf/library/items/ABCD1234?page=9");
  assert.equal(collectSources(["[a](zotero://open-pdf/library/items/ABCD1234?quote=x%20y)"])[0]?.href, "zotero://open-pdf/library/items/ABCD1234");
});

const sentenceLink = (label: string, q = "") => `[${label}](zotero://open-pdf/library/items/ABCD1234?page=4${q})`;

test("a sentence-like citation chip loses the quotation marks the model puts around it; a short label keeps its own", () => {
  const long = JSON.stringify(mdToTree(`In other words, "${sentenceLink("IDH-A tumors are associated with more microglia")}". Next.`));
  assert.ok(!long.includes('\\"'), long);
  assert.match(long, /In other words, /);
  const short = JSON.stringify(mdToTree(`He said "${sentenceLink("p.4")}" once.`));
  assert.ok(short.includes('\\"'), short);
});

test("the tooltip of a long chip label is the label; with a quote it is the quote", () => {
  const long = "IDH-A tumors are associated with more microglia/macrophages";
  assert.equal(citeTitle("zotero://open-pdf/library/items/ABCD1234?page=4", long), long);
  assert.equal(citeTitle("zotero://open-pdf/library/items/ABCD1234?page=4&quote=white%20names", long), "“white names”");
});
