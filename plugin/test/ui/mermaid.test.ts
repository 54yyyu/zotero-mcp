// Mermaid blocks (#823): the common flowchart and sequence syntax becomes a drawing that goes through the same sanitizer and
// palette as a ```svg one; anything else stays code. Pure: no DOM.
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { mermaidToSvg, textWidth } from "../../src/ui/mermaid.ts";
import { PALETTE, prepareSvg } from "../../src/ui/diagram-svg.ts";
import type { SvgEl, SvgNode } from "../../src/ui/diagram-svg.ts";
import { MERMAID_FLOW, MERMAID_SEQ } from "../../src/ui/fake-diagrams.ts";
import { mdToTree } from "../../src/ui/markdown.ts";
import { noteDiagrams, noteHtml } from "../../src/ui/note-html.ts";

const all = (n: SvgEl): SvgEl[] => [n, ...n.kids.flatMap((k: SvgNode) => (typeof k === "string" ? [] : all(k)))];
const texts = (n: SvgEl): string[] => all(n).filter((e) => e.tag === "tspan" || e.tag === "text").flatMap((e) => e.kids.filter((k): k is string => typeof k === "string"));
const drawn = (src: string) => {
  const svg = mermaidToSvg(src);
  assert.ok(svg, "drawn");
  const p = prepareSvg(svg!);
  assert.ok(p, "survives the sanitizer");
  return p!;
};

describe("mermaid flowcharts", () => {
  it("nodes, shapes, labels and edges come out as a themed drawing", () => {
    const p = drawn(MERMAID_FLOW);
    const t = texts(p.svg);
    for (const w of ["Research question", "Enough papers", "in the library?", "yes", "no", "Write up", "Cited answer"]) assert.ok(t.includes(w), w);
    const els = all(p.svg);
    assert.ok(els.some((e) => e.tag === "polygon"), "the decision is a diamond");
    assert.equal(els.filter((e) => e.tag === "path" && (e.attrs["marker-end"] || e.attrs["marker-start"])).length, 6, "an arrow per link (the loop back's head at its start)");
    // every colour is a palette variable, never a literal
    for (const e of els) for (const k of ["fill", "stroke"]) {
      const v = e.attrs[k];
      if (v && v !== "none") assert.match(v, /^var\(--dg-[a-z-]+\)$/, `${e.tag} ${k}=${v}`);
    }
    // style A fill:#e7f0ff,stroke:#3b5bdb: a soft accent fill with an accent outline
    assert.ok(els.some((e) => e.tag === "rect" && e.attrs.fill === "var(--dg-accent-soft)" && e.attrs.stroke === "var(--dg-accent)"));
    const [, , w, h] = p.fit.box;
    assert.ok(w > 150 && w < 600 && h > 200, `a readable size: ${w}x${h}`);
  });

  it("reads the directions, chains, &, pipe and inline labels, dotted/thick/headless links and the node shapes", () => {
    const lr = drawn("graph LR\n  A[Start] --> B(Round) & C([Stadium])\n  B -->|yes| D{{Hex}}\n  C -. maybe .-> E[(DB)]\n  D ==> F((Circle))\n  E --- F\n  F --x G>Flag]\n  G <--> H[[Sub]]\n  H ~~~ A");
    const [, , w, h] = lr.fit.box;
    assert.ok(w > h, "LR runs sideways");
    const t = texts(lr.svg);
    for (const s of ["Start", "Round", "Stadium", "yes", "Hex", "maybe", "DB", "Circle", "Flag", "Sub"]) assert.ok(t.includes(s), s);
    const paths = all(lr.svg).filter((e) => e.tag === "path" && e.attrs.stroke === "var(--dg-muted)" && e.attrs.d?.startsWith("M") && !e.attrs["stroke-linejoin"]);
    assert.ok(paths.some((e) => e.attrs["stroke-dasharray"]), "a dotted link");
    assert.ok(paths.some((e) => e.attrs["stroke-width"] === "2.5"), "a thick link");
    assert.ok(paths.some((e) => e.attrs["marker-start"] && e.attrs["marker-end"]), "a two-headed link");
    const td = drawn("flowchart BT\n  a --> b");
    assert.ok(td.fit.box[3] > td.fit.box[2] * 0.8, "BT runs up and down");
  });

  it("cycles and self-loops are drawn, not looped forever", () => {
    const p = drawn("flowchart TD\nA --> B --> C --> A\nC --> C");
    assert.equal(texts(p.svg).filter((x) => /^[ABC]$/.test(x)).length, 3);
  });

  it("labels: <br>, quotes, #entities, markdown strings and CJK, wrapped to fit", () => {
    const p = drawn('flowchart TD\n  A["Quoted (with) [brackets]"] --> B[two<br/>lines]\n  B --> C["`**bold** text`"]\n  C --> D[#quot;hi#quot; #35;1]\n  D --> E[一个很长很长很长很长很长很长很长很长很长的中文标签]');
    const t = texts(p.svg);
    for (const s of ["Quoted (with) [brackets]", "two", "lines", "bold text", '"hi" #1']) assert.ok(t.includes(s), s);
    assert.ok(t.filter((s) => /^[一-鿿]+$/.test(s)).length >= 2, "the long CJK label wraps");
    assert.ok(textWidth("中文") > textWidth("ab") * 1.5);
  });

  it("a label cannot inject markup", () => {
    const svg = mermaidToSvg('flowchart TD\n  A["<script>alert(1)</script><img src=x onerror=alert(1)>"] --> B["</text><a href=javascript:1>x</a>"]')!;
    const p = prepareSvg(svg)!;
    assert.ok(!all(p.svg).some((e) => /script|img|^a$/i.test(e.tag)));
    assert.ok(!/onerror|javascript/.test(JSON.stringify(p.svg.attrs)));
  });

  it("subgraphs are boxes with their title around their nodes", () => {
    const p = drawn("flowchart LR\n  subgraph one [First part]\n    a1 --> a2\n  end\n  subgraph two\n    b1 --> b2\n  end\n  a2 --> b1");
    const t = texts(p.svg);
    assert.ok(t.includes("First part") && t.includes("two"));
    assert.equal(all(p.svg).filter((e) => e.tag === "rect" && e.attrs.rx === "10").length, 2);
  });

  it("directives, comments, frontmatter titles and click/linkStyle lines are read past", () => {
    const p = drawn("---\ntitle: Pipeline\n---\n%%{init: {'theme':'dark'}}%%\nflowchart TD\n  %% a comment\n  A --> B\n  click A href \"https://example.com\"\n  linkStyle 0 stroke:red\n  classDef hot fill:#f96\n  class B hot\n  A:::hot");
    assert.equal(p.title, "Pipeline");
    assert.ok(all(p.svg).some((e) => e.attrs.fill === "var(--dg-orange-soft)"), "classDef colours map to the palette");
  });

  it("anything it cannot read is null, so the block stays code", () => {
    for (const bad of ["", "pie title X\n  \"a\" : 1", "classDiagram\n  A <|-- B", "gantt\n  title x", "flowchart TD\n  A --> ", "flowchart TD\n  A --> B\n  end", "flowchart TD\n  subgraph x\n  A", "flowchart TD\n  A -->> B",
      "sequenceDiagram\n  A->>B: hi\n  end", "sequenceDiagram\n  loop x\n  A->>B: hi", "sequenceDiagram\n  this is not a message", "flowchart TD\n" + "A-->B\n".repeat(400)]) {
      assert.equal(mermaidToSvg(bad), null, bad.slice(0, 40));
    }
  });
});

describe("mermaid sequence diagrams", () => {
  it("participants, messages, a loop and a note", () => {
    const p = drawn(MERMAID_SEQ);
    const t = texts(p.svg);
    for (const s of ["You", "Panel", "Agent", "Ask about the open paper", "loop", "[until it can cite]", "zotero-cli search / read", "Answer with page links"]) assert.ok(t.some((x) => x.includes(s)), s);
    assert.ok(all(p.svg).some((e) => e.tag === "rect" && e.attrs.fill === "var(--dg-orange-soft)"), "the note");
    assert.ok(all(p.svg).some((e) => e.attrs["stroke-dasharray"] === "4 3" && e.attrs["marker-end"]), "a dashed reply");
  });

  it("alt/else, opt, autonumber, actors, -x and -) arrows, notes left and right", () => {
    const p = drawn("sequenceDiagram\n  autonumber\n  actor Alice\n  participant J as John Doe\n  Alice->>+J: Hello\n  alt sick\n    J-->>-Alice: Not good\n  else well\n    J-->>Alice: Fine\n  end\n  opt extra\n    J->>Alice: Thanks\n  end\n  Note left of Alice: L\n  Note right of J: R\n  J-xAlice: bye\n  J-)Alice: later\n  activate J\n  deactivate J");
    const t = texts(p.svg);
    for (const s of ["Alice", "John Doe", "1. Hello", "2. Not good", "3. Fine", "alt", "[sick]", "[well]", "opt", "L", "R", "5. bye", "6. later"]) assert.ok(t.includes(s), s);
    assert.ok(all(p.svg).some((e) => e.attrs["marker-end"]?.includes("mmx")), "a cross head");
  });
});

describe("mermaid in markdown and notes", () => {
  it("a ```mermaid block is a diagram, open while its fence is not closed", () => {
    assert.deepEqual(mdToTree("```mermaid\ngraph TD\nA-->B\n```"), [{ tag: "diagram", attrs: { lang: "mermaid" }, kids: ["graph TD\nA-->B"] }]);
    assert.deepEqual(mdToTree("```mermaid\ngraph TD\nA-->"), [{ tag: "diagram", attrs: { open: "1", lang: "mermaid" }, kids: ["graph TD\nA-->"] }]);
    assert.deepEqual(mdToTree("```svg\n<svg/>\n```"), [{ tag: "diagram", kids: ["<svg/>"] }], "svg blocks unchanged");
  });

  it("notes: a closed mermaid block takes an image slot, with its language", () => {
    const md = "A\n\n```mermaid\ngraph TD\nA-->B\n```\n\n```svg\n<svg><rect/></svg>\n```\n\n```mermaid\ngraph TD\nC-->";
    assert.deepEqual(noteDiagrams(md), [{ src: "graph TD\nA-->B", lang: "mermaid" }, { src: "<svg><rect/></svg>", lang: "svg" }]);
    const html = noteHtml(md, { image: (i) => (i === 0 ? { key: "IMGKEY01", width: 300, height: 200 } : null) });
    assert.match(html, /<img data-attachment-key="IMGKEY01"/);
    assert.match(html, /<pre>graph TD\nC--&gt;<\/pre>/, "the cut-off block stays code");
  });

  it("the palette it uses is the guide's", () => {
    const used = new Set<string>();
    for (const src of [MERMAID_FLOW, MERMAID_SEQ]) for (const m of mermaidToSvg(src)!.matchAll(/(?:fill|stroke)="([a-z-]+)"/g)) if (m[1] !== "none") used.add(m[1]!);
    for (const u of used) assert.ok(PALETTE.includes(u), u);
  });

  it("is fast: a 60-node flowchart in well under a frame", () => {
    const src = "flowchart TD\n" + Array.from({ length: 60 }, (_, i) => `n${i} --> n${(i * 7 + 3) % 60}\nn${i} --> n${(i + 1) % 60}`).join("\n");
    const t0 = performance.now();
    assert.ok(mermaidToSvg(src));
    assert.ok(performance.now() - t0 < 100, `${performance.now() - t0} ms`);
  });
});
