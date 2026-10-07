import { test } from "node:test";
import assert from "node:assert/strict";
import { noteDiagrams, noteHtml } from "../../src/ui/note-html.ts";
import { XSS_CORPUS } from "./corpus.ts";

/** Every tag and attribute the note may contain; anything else in the output is a leak. */
const TAGS = new Set(["div", "p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "blockquote", "em", "strong", "del", "u", "sub", "sup", "code", "pre", "span", "a", "br", "hr", "img", "table", "thead", "tbody", "tr", "th", "td"]);
const ATTRS: Record<string, RegExp> = {
  "div data-schema-version": /^(8|9)$/, "a href": /^(https?|zotero):\/\/[^\s"]+$/, "span class": /^math$/, "pre class": /^math$/,
  "span style": /^(color|background-color): #[0-9a-f]{3,8}$/,
  "ol start": /^\d+$/, "img data-attachment-key": /^[A-Z0-9]{8}$/, "img width": /^\d+$/, "img height": /^\d+$/,
};

/** The first thing in `html` outside the note's allow-list, or null. Text is escaped, so every "<" opens a real tag. */
function leak(html: string): string | null {
  const re = /<(\/?)([a-zA-Z0-9]+)((?:\s[a-z-]+="[^"<>]*")*)\s*>|</g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(html))) {
    if (m[0] === "<") return `a stray "<" at ${m.index}`;
    const tag = m[2] as string;
    if (!TAGS.has(tag)) return `<${tag}>`;
    for (const a of (m[3] ?? "").matchAll(/\s([a-z-]+)="([^"]*)"/g)) {
      const rule = ATTRS[`${tag} ${a[1]}`];
      if (!rule || !rule.test(a[2] as string)) return `${tag} ${a[1]}="${a[2]}"`;
    }
  }
  return null;
}

const body = (html: string) => html.replace(/^<div data-schema-version="\d">|<\/div>$/g, "");

test("headings, lists, emphasis, code and links come out as the note editor's own tags", () => {
  const html = noteHtml("## Result\n\n**Bold** and *it* and ~~gone~~ and `code`.\n\n1. one\n2. two\n3. three\n\n- a\n- b\n\n> quoted\n\n---\n\n```bash\nzotero-cli search \"x\" <y>\n```\n\nSee [OSF](https://osf.io/x).");
  assert.equal(leak(html), null);
  assert.match(html, /^<div data-schema-version="8">/, "no math: schema 8, as Zotero itself writes");
  assert.match(html, /<h2>Result<\/h2>/);
  assert.match(html, /<p><strong>Bold<\/strong> and <em>it<\/em> and <del>gone<\/del> and <code>code<\/code>\.<\/p>/);
  assert.match(html, /<ol><li>one<\/li><li>two<\/li>/);
  assert.match(html, /<ul><li>a<\/li><li>b<\/li><\/ul>/);
  assert.match(html, /<blockquote><p>quoted<\/p><\/blockquote><hr>/);
  assert.match(html, /<pre>zotero-cli search &quot;x&quot; &lt;y&gt;<\/pre>/, "code is escaped text");
  assert.match(html, /<a href="https:\/\/osf.io\/x">OSF<\/a>/);
});

test("citation chips become zotero:// links that stay clickable in the note", () => {
  const href = "zotero://open-pdf/library/items/ABCD1234?page=8&quote=callbacks%20for%20interviews";
  const html = noteHtml(`It persists [Bertrand and Mullainathan 2004, p.8](${href}).`);
  assert.equal(leak(html), null);
  assert.match(html, /<a href="zotero:\/\/open-pdf\/library\/items\/ABCD1234\?page=8&amp;quote=callbacks%20for%20interviews">Bertrand and Mullainathan 2004, p\.8<\/a>\./);
});

test("math becomes the editor's math nodes: inline span, display pre, schema 9", () => {
  const html = noteHtml("The ratio $R = c_w / c_b$ is\n\n$$\\Delta = \\ln\\left(\\frac{c_w}{c_b}\\right) < 1$$\n\nand \\(x\\) too, with \\[y^2\\] inline.\n\n```math\na \\& b\n```");
  assert.equal(leak(html), null);
  assert.match(html, /^<div data-schema-version="9">/);
  assert.match(html, /<span class="math">\$R = c_w \/ c_b\$<\/span>/);
  assert.match(html, /<pre class="math">\$\$\\Delta = \\ln\\left\(\\frac\{c_w\}\{c_b\}\\right\) &lt; 1\$\$<\/pre>/, "a display formula is a block, its < escaped");
  assert.match(html, /<span class="math">\$x\$<\/span>/);
  assert.match(html, /<span class="math">\$\\displaystyle y\^2\$<\/span>/, "display math inside a sentence stays inline");
  assert.match(html, /<pre class="math">\$\$a \\&amp; b\$\$<\/pre>/, "a ```math fence is a math block");
  assert.ok(!/<p><pre/.test(html), "no block inside a paragraph");
});

test("tables keep their cells", () => {
  const html = noteHtml("| Study | Ratio |\n|---|---:|\n| Pager | 1.44 |\n| Quillian | $1.36$ |");
  assert.equal(leak(html), null);
  assert.match(html, /<table><thead><tr><th>Study<\/th><th>Ratio<\/th><\/tr><\/thead><tbody><tr><td>Pager<\/td><td>1\.44<\/td><\/tr><tr><td>Quillian<\/td><td><span class="math">\$1\.36\$<\/span><\/td><\/tr><\/tbody><\/table>/);
});

test("a title is the note's first line; diagrams are embedded images, or code without one", () => {
  const svg = '<svg viewBox="0 0 10 10"><rect width="5" height="5"/></svg>';
  const md = `Intro\n\n\`\`\`svg\n${svg}\n\`\`\`\n\nMiddle\n\n\`\`\`svg\n${svg.replace("5", "6")}\n\`\`\``;
  assert.deepEqual(noteDiagrams(md), [svg, svg.replace("5", "6")]);
  const html = noteHtml(md, { title: "  What is   the gap?\n", image: (i) => (i === 0 ? { key: "IMGKEY01", width: 450.4, height: 300 } : null) });
  assert.equal(leak(html), null);
  assert.match(html, /^<div data-schema-version="8"><h1>What is the gap\?<\/h1><p>Intro<\/p>/);
  assert.match(html, /<p><img data-attachment-key="IMGKEY01" width="450" height="300"><\/p><p>Middle<\/p>/);
  assert.match(html, /<pre>&lt;svg viewBox=&quot;0 0 10 10&quot;&gt;&lt;rect width=&quot;6&quot;/, "no image: the SVG source as code, escaped");
  // a fence cut off mid-drawing is not a drawing, so it takes no image slot
  assert.deepEqual(noteDiagrams("```svg\n<svg><rect/>"), []);
  assert.match(noteHtml("```svg\n<svg><rect/>", { image: () => ({ key: "IMGKEY01", width: 1, height: 1 }) }), /<pre>&lt;svg&gt;/);
});

test("hostile markdown: only allowed tags and attributes, no script, no foreign links", () => {
  for (const src of XSS_CORPUS) {
    const html = noteHtml(src, { title: '<img src=x onerror="alert(1)">' });
    assert.equal(leak(html), null, `${src}\n=> ${html}`);
    assert.ok(!/<(script|iframe|style|svg|object)/i.test(html), `${src}\n=> ${html}`); // leak() checked every real tag's attributes
  }
  // the title and an image key are attacker-controlled strings too
  const t = noteHtml("x", { title: '"><script>alert(1)</script>', image: () => null });
  assert.equal(leak(t), null);
  const k = noteHtml("```svg\n<svg viewBox=\"0 0 1 1\"><rect width=\"1\" height=\"1\"/></svg>\n```", { image: () => ({ key: 'K" onerror="x', width: NaN, height: 1e9 }) });
  assert.match(k, /data-attachment-key="K&quot; onerror=&quot;x" width="1" height="4000"/);
  // control characters vanish; a huge reply is still one well-formed note
  assert.equal(body(noteHtml("a\u0000b\u0007c")), "<p>abc</p>");
  assert.equal(leak(noteHtml("*a ".repeat(30000))), null);
});

test("the formatting markdown lacks: underline, strike, sub/sup, text and highlight colours, as the editor's own marks", () => {
  const html = noteHtml('H<sub>2</sub>O, x<sup>2</sup>, <u>under</u>, <s>old</s>, <mark>key</mark>, <span style="color: red">red</span>, <span style="background-color: blue; color: #123456">both</span>, <span style="color: grey">grey</span>.');
  assert.equal(leak(html), null, html);
  assert.equal(body(html), '<p>H<sub>2</sub>O, x<sup>2</sup>, <u>under</u>, <del>old</del>, <span style="background-color: #ffd40080">key</span>, <span style="color: #ff2020">red</span>, <span style="color: #123456"><span style="background-color: #2ea8e580">both</span></span>, <span style="color: #7e8386">grey</span>.</p>');
});

test("the formatting subset is strict: other styles, attributes and tags never reach the note", () => {
  for (const src of [
    '<span style="color: url(javascript:alert(1))">a</span>', '<span style="color: expression(alert(1))">a</span>',
    '<span style="position: fixed; color: red" onclick="alert(1)" class="x">a</span>', "<span style='background: var(--x)'>a</span>",
    '<u onmouseover="alert(1)">a</u>', '<sup style="color: red">a</sup>', '<font color="red">a</font>', '<span style="color: red">unclosed',
  ]) {
    const html = noteHtml(src);
    assert.equal(leak(html), null, `${src}\n=> ${html}`);
    assert.ok(!/url\(|expression|onclick|onmouseover|position|var\(|<font/.test(html.replace(/&lt;[^]*?&gt;/g, "")), `${src}\n=> ${html}`);
  }
  assert.equal(body(noteHtml('<span style="position: fixed; color: red" onclick="x">a</span>')), '<p><span style="color: #ff2020">a</span></p>', "only the colour survives");
  assert.equal(body(noteHtml('<span style="color: url(x)">a</span>')), "<p>a</p>", "no colour we allow: just the text");
});

test("the agent's formatting guide names exactly what the panel and the note accept", async () => {
  const { FORMAT_GUIDE } = await import("../../src/agent/brief.ts");
  const { TEXT_COLORS, mdToTree, walk } = await import("../../src/ui/markdown.ts");
  assert.ok(FORMAT_GUIDE.split(/\s+/).length <= 60, `${FORMAT_GUIDE.split(/\s+/).length} words`);
  for (const name of Object.keys(TEXT_COLORS)) assert.match(FORMAT_GUIDE, new RegExp(`\\b${name}\\b`), name);
  for (const tag of FORMAT_GUIDE.match(/<(u|s|sub|sup|mark)>/g) ?? []) {
    const t = tag.slice(1, -1);
    const tree = mdToTree(`a <${t}>b</${t}> c`);
    assert.ok([...walk(tree)].some((n) => n.tag !== "p"), `${tag} renders as formatting, not text`);
  }
  assert.ok([...walk(mdToTree('<span style="color: red">x</span>'))].some((n) => n.attrs?.color === "#ff2020"));
});
