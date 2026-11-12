// Mermaid, the common part (#823): a ```mermaid flowchart or sequence diagram becomes SVG text drawn with the palette names
// of the drawing guide, which then goes through diagram-svg.ts like the agent's own ```svg (sanitized, themed, exported,
// saved to notes). Pure, no DOM. Anything else (another diagram type, a line it cannot read) is null: the block stays code.
// Bundling the mermaid library instead would add 5.5 MB (1.6 MB gzipped) to a 0.98 MB panel; this adds 22 kB (9 kB gzipped).
//
//   flowchart / graph  TD TB BT LR RL; node shapes [] () ([]) [[]] [()] (()) {} {{}} [//] [\\] >]; edges --> --- -.-> ==>
//                      ~~~, heads > o x and <, labels |text| and -- text -->, chains and &; subgraph ... end; style,
//                      classDef, class and :::name colours (mapped to the palette). click, linkStyle and the like are ignored.
//   sequenceDiagram    participant / actor (as alias), ->> -->> -> --> -x --x -) --), Note left of / right of / over,
//                      loop alt else opt par and critical option break rect ... end, autonumber, activate (ignored), title.
import { paletteName } from "./diagram-svg.ts";

/** Beyond this the block shows as code: a diagram a reader can follow is far smaller. */
const MAX_NODES = 150;
const MAX_EDGES = 300;
const MAX_SRC = 30_000;

// ───────────────────────────── text ─────────────────────────────

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const r1 = (n: number) => Math.round(n * 10) / 10;

/** Width of `s` at `size` px in the panel's sans-serif, estimated (no DOM to measure with): wide CJK, narrow i/l/punctuation. */
export function textWidth(s: string, size = 12): number {
  let w = 0;
  for (const ch of s) {
    const c = ch.codePointAt(0)!;
    if (c >= 0x2e80 && c <= 0xffdc && !(c >= 0xfe00 && c <= 0xfe0f)) w += 1;
    else if (/[iljtfr.,:;'|!()[\] ]/.test(ch)) w += 0.32;
    else if (/[mwMW@%]/.test(ch)) w += 0.85;
    else if (/[A-Z0-9#&]/.test(ch)) w += 0.64;
    else w += 0.54;
  }
  return w * size;
}

/** Mermaid's label text: quotes off, #entities and <br> decoded, markdown emphasis and other tags dropped; lines. */
function labelLines(raw: string): string[] {
  let s = raw.trim();
  if (/^"[^]*"$/.test(s)) s = s.slice(1, -1);
  if (/^`[^]*`$/.test(s)) s = s.slice(1, -1); // a markdown string
  s = s.replace(/#(\d{1,6});/g, (_m, d: string) => String.fromCodePoint(Number(d)))
    .replace(/#(quot|amp|lt|gt|nbsp);/g, (_m, n: string) => ({ quot: '"', amp: "&", lt: "<", gt: ">", nbsp: " " })[n] ?? "");
  s = s.replace(/<br\s*\/?>|\\n/gi, "\n").replace(/<[^>]*>/g, "").replace(/\*\*(.+?)\*\*|__(.+?)__/g, "$1$2").replace(/(^|[^*])\*([^*\n]+)\*/g, "$1$2");
  return s.split("\n").map((l) => l.trim()).filter((l, i, a) => l || a.length === 1);
}

/** Lines wrapped at `max` px: by words, or by character where there are no spaces (CJK). */
function wrap(lines: string[], max: number, size = 12): string[] {
  const out: string[] = [];
  for (const line of lines) {
    if (textWidth(line, size) <= max) { out.push(line); continue; }
    let cur = "";
    for (const tok of line.match(/[⺀-ￜ]|[^\s⺀-ￜ]+\s*|\s+/g) ?? [line]) {
      if (cur && textWidth((cur + tok).trimEnd(), size) > max) { out.push(cur.trimEnd()); cur = tok.trimStart(); }
      else cur += tok;
    }
    if (cur.trim()) out.push(cur.trimEnd());
  }
  return out.length ? out : [""];
}

const LH = 15; // line height for 12px
/** <text> lines centred on (x, y). */
function textEl(x: number, y: number, lines: string[], o: { size?: number; fill?: string; bold?: boolean } = {}): string {
  const size = o.size ?? 12, lh = size >= 12 ? LH : 14;
  const y0 = y - ((lines.length - 1) * lh) / 2 + size / 3;
  const spans = lines.map((l, i) => `<tspan x="${r1(x)}" y="${r1(y0 + i * lh)}">${esc(l)}</tspan>`).join("");
  return `<text text-anchor="middle" fill="${o.fill ?? "ink"}" font-size="${size}"${o.bold ? ` font-weight="600"` : ""}>${spans}</text>`;
}

const MARKERS = `<defs>
<marker id="mma" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M1.5 1.5 8.5 5 1.5 8.5" fill="none" stroke="muted" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></marker>
<marker id="mmo" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><circle cx="5" cy="5" r="3.5" fill="surface" stroke="muted" stroke-width="1.5"/></marker>
<marker id="mmx" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M2 2 8 8M8 2 2 8" stroke="muted" stroke-width="1.6" stroke-linecap="round"/></marker>
</defs>`;
const MARK: Record<string, string> = { ">": "mma", o: "mmo", x: "mmx" };

function svgDoc(box: [number, number, number, number], title: string, body: string): string {
  const [x, y, w, h] = box.map(r1) as [number, number, number, number];
  return `<svg viewBox="${x} ${y} ${w} ${h}" xmlns="http://www.w3.org/2000/svg">` +
    (title ? `<title>${esc(title)}</title>` : "") + MARKERS + body + "</svg>";
}

/** The diagram type's first line, after frontmatter (its `title:` kept), `%%{init}%%` and comments. */
function preamble(src: string): { title: string; lines: string[] } | null {
  if (src.length > MAX_SRC) return null;
  let text = src.replace(/\r\n?/g, "\n");
  let title = "";
  const fm = /^\s*---\n([^]*?)\n---\s*\n/.exec(text);
  if (fm) { title = /^title:\s*(.+)$/m.exec(fm[1]!)?.[1]?.trim().replace(/^["']|["']$/g, "") ?? ""; text = text.slice(fm[0].length); }
  const lines = text.replace(/%%\{[^]*?\}%%/g, "").split("\n").map((l) => l.replace(/%%.*$/, "").trim()).filter(Boolean);
  return lines.length ? { title, lines } : null;
}

/** SVG text for a Mermaid flowchart or sequence diagram, or null when it is neither or cannot be read. */
export function mermaidToSvg(src: string): string | null {
  const p = preamble(src);
  if (!p) return null;
  const head = p.lines[0]!;
  try {
    const fc = /^(?:flowchart|graph)(?:-elk)?(?:\s+(TB|TD|BT|RL|LR))?\s*;?$/i.exec(head);
    if (fc) return flowchart(p.lines.slice(1), ((fc[1] ?? "TB").toUpperCase().replace("TD", "TB")) as Dir, p.title);
    if (/^sequenceDiagram\s*$/.test(head)) return sequence(p.lines.slice(1), p.title);
  } catch {
    // a construct this reader does not know: the block stays code
  }
  return null;
}

// ───────────────────────────── flowchart: parse ─────────────────────────────

type Dir = "TB" | "BT" | "LR" | "RL";
type Shape = "rect" | "round" | "stadium" | "sub" | "cyl" | "circle" | "dcircle" | "diamond" | "hex" | "para" | "trap" | "asym";
interface Paint { fill?: string; stroke?: string }
interface FNode { id: string; lines: string[]; shape: Shape; group: number; paint: Paint; classes: string[]; w: number; h: number }
interface FEdge { from: string; to: string; label: string[]; line: "solid" | "dotted" | "thick" | "none"; head: string; tail: string }
interface Group { id: string; title: string; parent: number }

class Unreadable extends Error {}
const fail = (what: string): never => { throw new Unreadable(what); };

// Opening bracket -> [closing, shape]; longest first.
const SHAPES: [string, string, Shape][] = [
  ["(((", ")))", "dcircle"], ["((", "))", "circle"], ["([", "])", "stadium"], ["[[", "]]", "sub"], ["[(", ")]", "cyl"],
  ["{{", "}}", "hex"], ["[/", "/]", "para"], ["[\\", "\\]", "para"], ["[/", "\\]", "trap"], ["[\\", "/]", "trap"],
  ["[", "]", "rect"], ["(", ")", "round"], ["{", "}", "diamond"], [">", "]", "asym"],
];
const ID = /[\p{L}\p{N}_][\p{L}\p{N}_.$]*/uy;
const TEXT_LINK = /\s*([<ox])?(--|-\.|==)\s+(?![->.=])(.+?)\s*(-{2,}|\.+-|={2,})(>|[ox](?![\p{L}\p{N}_]))?/uy;
const PLAIN_LINK = /\s*([<ox])?(-{2,}|-\.+-|={2,}|~{3,})(>|[ox](?![\p{L}\p{N}_]))?(?:\s*\|([^|]*)\|)?/uy;

/** Colours from `fill:#f96,stroke:#333`: fills become soft tints, strokes palette hues. */
function paintOf(decls: string): Paint {
  const out: Paint = {};
  for (const d of decls.split(/[,;]/)) {
    const [k, v] = d.split(":").map((x) => x.trim());
    if (!k || !v) continue;
    if (k === "fill") {
      const n = paletteName(v, "fill");
      if (n) out.fill = /^(ink|muted|line|surface)$/.test(n) ? "surface" : n.endsWith("-soft") ? n : `${n}-soft`;
    } else if (k === "stroke") {
      const n = paletteName(v, "stroke");
      if (n) out.stroke = n.replace(/-soft$/, "");
    }
  }
  if (out.fill && !out.stroke && out.fill !== "surface") out.stroke = out.fill.replace(/-soft$/, "");
  return out;
}

function parseFlowchart(lines: string[]) {
  const nodes = new Map<string, FNode>();
  const edges: FEdge[] = [];
  const groups: Group[] = [];
  const classDefs = new Map<string, Paint>();
  const stack: number[] = [];
  const cur = () => stack[stack.length - 1] ?? -1;

  const node = (id: string): FNode => {
    let n = nodes.get(id);
    if (n && n.group < 0 && cur() >= 0) n.group = cur(); // first defined outside, used inside: it belongs to the subgraph
    if (!n) {
      if (nodes.size >= MAX_NODES) fail("too many nodes");
      n = { id, lines: [id], shape: "rect", group: cur(), paint: {}, classes: [], w: 0, h: 0 };
      nodes.set(id, n);
    }
    return n;
  };

  // One statement: node groups joined by links. `s` is consumed with a cursor.
  const statement = (s: string) => {
    let i = 0;
    const ws = () => { while (s[i] === " " || s[i] === "\t") i++; };
    const ref = (): string => {
      ws();
      ID.lastIndex = i;
      const m = ID.exec(s);
      if (!m) fail(`expected a node at "${s.slice(i, i + 20)}"`);
      i = ID.lastIndex;
      const n = node(m![0]);
      // A bracket right after the id (no space): the shape and its label.
      for (const [open, close, shape] of SHAPES) {
        if (!s.startsWith(open, i)) continue;
        let j = i + open.length;
        let text: string;
        if (s[j] === '"') {
          const q = s.indexOf('"', j + 1);
          if (q < 0) fail("unclosed quote");
          text = s.slice(j, q + 1);
          j = q + 1;
          if (!s.startsWith(close, j)) continue;
        } else {
          const k = s.indexOf(close, j);
          if (k < 0) continue;
          text = s.slice(j, k);
          j = k;
        }
        n.lines = labelLines(text);
        n.shape = shape;
        i = j + close.length;
        break;
      }
      if (s.startsWith("@{", i)) { // the v11 shape syntax: its label, any shape drawn as a box
        const k = s.indexOf("}", i);
        if (k < 0) fail("unclosed @{");
        const label = /label:\s*("[^"]*"|[^,}]+)/.exec(s.slice(i, k))?.[1];
        if (label) n.lines = labelLines(label);
        i = k + 1;
      }
      const cls = /^:::([\w-]+)/.exec(s.slice(i));
      if (cls) { n.classes.push(cls[1]!); i += cls[0].length; }
      return n.id;
    };
    const many = (): string[] => {
      const ids = [ref()];
      for (;;) { ws(); if (s[i] !== "&") return ids; i++; ids.push(ref()); }
    };
    let left = many();
    for (;;) {
      ws();
      if (i >= s.length) return;
      let m: RegExpExecArray | null;
      let label: string[] = [], op: string, head: string | undefined, tail: string | undefined;
      TEXT_LINK.lastIndex = i;
      if ((m = TEXT_LINK.exec(s))) {
        tail = m[1]; head = m[5];
        label = labelLines(m[3]!);
        op = m[2]! + m[4]!;
        i = TEXT_LINK.lastIndex;
      } else {
        PLAIN_LINK.lastIndex = i;
        m = PLAIN_LINK.exec(s);
        if (!m) fail(`cannot read "${s.slice(i, i + 20)}"`);
        tail = m![1]; op = m![2]!; head = m![3];
        if (m![4] !== undefined) label = labelLines(m![4]);
        i = PLAIN_LINK.lastIndex;
      }
      const line = op.startsWith("~") ? "none" : op.includes(".") ? "dotted" : op.includes("=") ? "thick" : "solid";
      const right = many();
      for (const a of left) for (const b of right) {
        if (edges.length >= MAX_EDGES) fail("too many edges");
        edges.push({ from: a, to: b, label, line, head: head ?? "", tail: tail ? (tail === "<" ? ">" : tail) : "" });
      }
      left = right;
    }
  };

  for (const raw of lines) {
    for (const st of raw.split(/(?<!#\w+);(?=(?:[^"]*"[^"]*")*[^"]*$)/).map((x) => x.trim()).filter(Boolean)) {
      let m: RegExpExecArray | null;
      if ((m = /^subgraph\s+(.+)$/.exec(st))) {
        const spec = m[1]!.trim();
        const t = /^([\p{L}\p{N}_.-]+)\s*\[(.*)\]$/u.exec(spec);
        groups.push({ id: t ? t[1]! : spec, title: labelLines(t ? t[2]! : spec).join(" "), parent: cur() });
        stack.push(groups.length - 1);
      } else if (st === "end") {
        if (!stack.length) fail("end without subgraph");
        stack.pop();
      } else if ((m = /^style\s+(\S+)\s+(.+)$/.exec(st))) {
        Object.assign(node(m[1]!).paint, paintOf(m[2]!));
      } else if ((m = /^classDef\s+(\S+)\s+(.+)$/.exec(st))) {
        for (const name of m[1]!.split(",")) classDefs.set(name, paintOf(m[2]!));
      } else if ((m = /^class\s+(\S+)\s+([\w-]+)$/.exec(st))) {
        for (const id of m[1]!.split(",")) node(id).classes.push(m[2]!);
      } else if (/^(direction|click|linkStyle|accTitle|accDescr|call|href)\b/.test(st)) {
        // no drawing in them
      } else statement(st);
    }
  }
  if (!nodes.size) fail("no nodes");
  if (stack.length) fail("subgraph without end");
  for (const n of nodes.values()) for (const c of n.classes) n.paint = { ...classDefs.get(c), ...n.paint };
  // A subgraph used as a node in a link is not drawn as a node of its own.
  return { nodes, edges, groups };
}

// ───────────────────────────── flowchart: layout ─────────────────────────────
// Layered (Sugiyama-style): ranks by longest path with cycles broken, a dummy node per rank an edge crosses (an edge label is
// one), a few barycentre sweeps for the order, then positions pulled toward their neighbours. Computed with ranks running
// down (TB); LR, BT and RL map it at the end.

interface LNode { id: string; rank: number; order: number; across: number; acrossSize: number; alongSize: number; real?: FNode; label?: string[]; group: number }

function sizeNode(n: FNode, horizontal: boolean): void {
  const lines = wrap(n.lines, horizontal ? 130 : 160);
  n.lines = lines;
  const tw = Math.max(...lines.map((l) => textWidth(l))), th = lines.length * LH;
  let w = Math.max(56, tw + 26), h = Math.max(34, th + 16);
  switch (n.shape) {
    case "diamond": w = Math.max(64, tw * 1.25 + 34); h = Math.max(48, th * 1.6 + 26); break;
    case "circle": case "dcircle": w = h = Math.max(48, Math.hypot(tw, th) + 14); break;
    case "hex": case "para": case "trap": case "asym": w += 22; break;
    case "stadium": w += 12; break;
    case "cyl": h += 10; break;
  }
  n.w = Math.ceil(w); n.h = Math.ceil(h);
}

function flowchart(lines: string[], dir: Dir, title: string): string {
  const { nodes, edges, groups } = parseFlowchart(lines);
  const horizontal = dir === "LR" || dir === "RL";
  for (const n of nodes.values()) sizeNode(n, horizontal);
  const ids = [...nodes.keys()];

  // 1. break cycles: a depth-first walk in declaration order; an edge back into the walk's path is reversed for layout.
  const out = new Map<string, FEdge[]>(ids.map((id) => [id, []]));
  for (const e of edges) if (e.from !== e.to) out.get(e.from)!.push(e);
  const state = new Map<string, 1 | 2>();
  const back = new Set<FEdge>();
  for (const root of ids) {
    if (state.has(root)) continue;
    const stack: [string, number][] = [[root, 0]];
    state.set(root, 1);
    while (stack.length) {
      const top = stack[stack.length - 1]!;
      const list = out.get(top[0])!;
      if (top[1] >= list.length) { state.set(top[0], 2); stack.pop(); continue; }
      const e = list[top[1]++]!;
      const s = state.get(e.to);
      if (s === 1) back.add(e);
      else if (!s) { state.set(e.to, 1); stack.push([e.to, 0]); }
    }
  }
  const dag = edges.filter((e) => e.from !== e.to).map((e) => (back.has(e) ? { e, u: e.to, v: e.from } : { e, u: e.from, v: e.to }));

  // 2. ranks: longest path from the sources (Kahn order).
  const rank = new Map<string, number>(ids.map((id) => [id, 0]));
  const indeg = new Map<string, number>(ids.map((id) => [id, 0]));
  for (const d of dag) indeg.set(d.v, indeg.get(d.v)! + 1);
  const queue = ids.filter((id) => !indeg.get(id));
  while (queue.length) {
    const u = queue.shift()!;
    for (const d of dag) if (d.u === u) {
      rank.set(d.v, Math.max(rank.get(d.v)!, rank.get(u)! + 1));
      indeg.set(d.v, indeg.get(d.v)! - 1);
      if (!indeg.get(d.v)) queue.push(d.v);
    }
  }
  // An edge label sits on a rank of its own between its ends, so every rank is doubled when any edge has one.
  const k = edges.some((e) => e.label.some(Boolean)) ? 2 : 1;
  const L = new Map<string, LNode>();
  const across = (n: FNode) => (horizontal ? n.h : n.w), along = (n: FNode) => (horizontal ? n.w : n.h);
  for (const n of nodes.values()) L.set(n.id, { id: n.id, rank: rank.get(n.id)! * k, order: 0, across: 0, acrossSize: across(n), alongSize: along(n), real: n, group: n.group });

  // 3. chains: an edge through a dummy at every rank between its ends; a label is the middle one.
  interface Chain { e: FEdge; ids: string[]; reversed: boolean; labelAt: number }
  const chains: Chain[] = [];
  let dummies = 0;
  for (const d of dag) {
    const a = L.get(d.u)!, b = L.get(d.v)!;
    const path = [a.id];
    const mid = Math.floor((a.rank + b.rank) / 2);
    let labelAt = -1;
    for (let r = a.rank + 1; r < b.rank; r++) {
      const id = `\u0000${dummies++}`;
      const lab = r === mid && d.e.label.some(Boolean) ? wrap(d.e.label, 140, 11) : undefined;
      const lw = lab ? Math.max(...lab.map((l) => textWidth(l, 11))) + 8 : 0, lh = lab ? lab.length * 14 + 4 : 0;
      L.set(id, { id, rank: r, order: 0, across: 0, acrossSize: lab ? (horizontal ? lh : lw) : 8, alongSize: lab ? (horizontal ? lw : lh) : 0, group: a.group === b.group ? a.group : -1, ...(lab ? { label: lab } : {}) });
      if (lab) labelAt = path.length;
      path.push(id);
    }
    path.push(b.id);
    chains.push({ e: d.e, ids: path, reversed: d.u !== d.e.from, labelAt });
  }

  // 4. order within ranks: first appearance, then barycentre sweeps; members of a subgraph kept together.
  const maxRank = Math.max(0, ...[...L.values()].map((n) => n.rank));
  const ranks: LNode[][] = Array.from({ length: maxRank + 1 }, () => []);
  for (const n of L.values()) ranks[n.rank]!.push(n);
  ranks.forEach((r) => r.forEach((n, i) => (n.order = i)));
  const up = new Map<string, string[]>(), down = new Map<string, string[]>();
  for (const c of chains) for (let i = 1; i < c.ids.length; i++) {
    (down.get(c.ids[i - 1]!) ?? down.set(c.ids[i - 1]!, []).get(c.ids[i - 1]!)!).push(c.ids[i]!);
    (up.get(c.ids[i]!) ?? up.set(c.ids[i]!, []).get(c.ids[i]!)!).push(c.ids[i - 1]!);
  }
  const groupOf = (n: LNode) => n.group;
  const sortRank = (r: LNode[], nb: Map<string, string[]>) => {
    const bary = new Map<LNode, number>();
    for (const n of r) {
      const ns = (nb.get(n.id) ?? []).map((id) => L.get(id)!.order);
      bary.set(n, ns.length ? ns.reduce((a, b) => a + b, 0) / ns.length : n.order);
    }
    r.sort((a, b) => bary.get(a)! - bary.get(b)! || a.order - b.order);
    // keep a subgraph's members contiguous, at the place of its first member
    const firstAt = new Map<number, number>();
    r.forEach((n, i) => { if (!firstAt.has(groupOf(n))) firstAt.set(groupOf(n), i); });
    const at = new Map(r.map((n, i) => [n, i]));
    r.sort((a, b) => firstAt.get(groupOf(a))! - firstAt.get(groupOf(b))! || at.get(a)! - at.get(b)!);
    r.forEach((n, i) => (n.order = i));
  };
  for (let it = 0; it < 4; it++) {
    for (let i = 1; i < ranks.length; i++) sortRank(ranks[i]!, up);
    for (let i = ranks.length - 2; i >= 0; i--) sortRank(ranks[i]!, down);
  }

  // 5. positions across the ranks: packed, then pulled toward the neighbours' centres (both pushes averaged: never overlapping).
  const gapAcross = horizontal ? 18 : 26;
  const sep = (a: LNode, b: LNode) => (a.acrossSize + b.acrossSize) / 2 + (a.real && b.real ? gapAcross : a.real || b.real ? gapAcross / 2 + 2 : 6) + (a.group !== b.group ? 16 : 0);
  for (const r of ranks) { let x = 0; r.forEach((n, i) => { x = i ? x + sep(r[i - 1]!, n) : n.acrossSize / 2; n.across = x; }); }
  const place = (r: LNode[], nb: Map<string, string[]>) => {
    const want = r.map((n) => { const ns = (nb.get(n.id) ?? []).map((id) => L.get(id)!.across); return ns.length ? ns.reduce((a, b) => a + b, 0) / ns.length : n.across; });
    const right = [...want], left = [...want];
    for (let i = 1; i < r.length; i++) right[i] = Math.max(right[i]!, right[i - 1]! + sep(r[i - 1]!, r[i]!));
    for (let i = r.length - 2; i >= 0; i--) left[i] = Math.min(left[i]!, left[i + 1]! - sep(r[i]!, r[i + 1]!));
    r.forEach((n, i) => (n.across = (right[i]! + left[i]!) / 2));
  };
  for (let it = 0; it < 6; it++) {
    for (let i = 1; i < ranks.length; i++) place(ranks[i]!, up);
    for (let i = ranks.length - 2; i >= 0; i--) place(ranks[i]!, down);
  }

  // 6. positions along: each rank as deep as its deepest node, a gap between ranks (more where subgraph titles sit).
  const gapAlong = (horizontal ? 52 : 44) / k + (groups.length ? 10 : 0);
  const pos: number[] = [];
  let y = 0;
  ranks.forEach((r, i) => {
    const depth = Math.max(0, ...r.map((n) => n.alongSize));
    const prev = i ? Math.max(0, ...ranks[i - 1]!.map((n) => n.alongSize)) : 0;
    y = i ? y + prev / 2 + gapAlong + depth / 2 : depth / 2;
    pos.push(y);
  });
  const total = y + Math.max(0, ...(ranks[ranks.length - 1] ?? []).map((n) => n.alongSize)) / 2;
  const flip = dir === "BT" || dir === "RL";
  /** Final x, y of an (across, along) point. */
  const at = (a: number, r: number): [number, number] => { const rr = flip ? total - r : r; return horizontal ? [rr, a] : [a, rr]; };
  const center = (n: LNode) => at(n.across, pos[n.rank]!);

  // 7. draw: subgraph boxes, edges, nodes.
  const body: string[] = [];
  const ext = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity };
  const grow = (x0: number, y0: number, x1: number, y1: number) => { ext.x0 = Math.min(ext.x0, x0); ext.y0 = Math.min(ext.y0, y0); ext.x1 = Math.max(ext.x1, x1); ext.y1 = Math.max(ext.y1, y1); };
  for (const n of nodes.values()) { const [x, yy] = center(L.get(n.id)!); grow(x - n.w / 2, yy - n.h / 2, x + n.w / 2, yy + n.h / 2); }

  // subgraphs, innermost last so their boxes sit inside: the box around their nodes (and inner boxes), the title on top
  const boxes = new Map<number, [number, number, number, number]>();
  const depthOf = (g: number): number => (g < 0 ? 0 : 1 + depthOf(groups[g]!.parent));
  const order = groups.map((_, i) => i).sort((a, b) => depthOf(b) - depthOf(a));
  for (const g of order) {
    let b: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
    for (const n of nodes.values()) if (n.group === g) { const [x, yy] = center(L.get(n.id)!); b = [Math.min(b[0], x - n.w / 2), Math.min(b[1], yy - n.h / 2), Math.max(b[2], x + n.w / 2), Math.max(b[3], yy + n.h / 2)]; }
    for (const [h, ib] of boxes) if (groups[h]!.parent === g) b = [Math.min(b[0], ib[0]), Math.min(b[1], ib[1]), Math.max(b[2], ib[2]), Math.max(b[3], ib[3])];
    if (!Number.isFinite(b[0])) continue;
    const tw = textWidth(groups[g]!.title, 11) + 16;
    b = [b[0] - 12, b[1] - 30, Math.max(b[2] + 12, b[0] - 12 + tw), b[3] + 12];
    boxes.set(g, b);
    grow(...b);
  }
  for (const g of [...order].reverse()) {
    const b = boxes.get(g);
    if (!b) continue;
    body.push(`<rect x="${r1(b[0])}" y="${r1(b[1])}" width="${r1(b[2] - b[0])}" height="${r1(b[3] - b[1])}" rx="10" fill="surface" stroke="line" stroke-width="1"/>`);
    body.push(`<text x="${r1(b[0] + 10)}" y="${r1(b[1] + 17)}" font-size="11" font-weight="600" fill="muted">${esc(groups[g]!.title)}</text>`);
  }

  // edges: from the side of a node facing the next rank, spread over it; S-curves between ranks; back edges loop round the side
  const portSide = new Map<string, { out: Chain[]; in: Chain[] }>();
  for (const c of chains) {
    (portSide.get(c.ids[0]!) ?? portSide.set(c.ids[0]!, { out: [], in: [] }).get(c.ids[0]!)!).out.push(c);
    (portSide.get(c.ids[c.ids.length - 1]!) ?? portSide.set(c.ids[c.ids.length - 1]!, { out: [], in: [] }).get(c.ids[c.ids.length - 1]!)!).in.push(c);
  }
  /** Where a chain leaves (end 0) or enters (end 1) node `id`, in (across, along). */
  const port = (c: Chain, end: 0 | 1): [number, number] => {
    const id = end ? c.ids[c.ids.length - 1]! : c.ids[0]!;
    const n = L.get(id)!;
    const list = end ? portSide.get(id)!.in : portSide.get(id)!.out;
    const sorted = [...list].sort((p, q) => L.get(end ? p.ids[p.ids.length - 2]! : p.ids[1]!)!.across - L.get(end ? q.ids[q.ids.length - 2]! : q.ids[1]!)!.across);
    const i = sorted.indexOf(c);
    const fs = n.real!.shape;
    const room = /circle/.test(fs) ? 0 : /diamond|hex/.test(fs) ? 0.3 : 0.6;
    const spread = list.length > 1 ? Math.min(n.acrossSize * room, (list.length - 1) * 14) : 0;
    const a = n.across + (list.length > 1 ? -spread / 2 + (spread * i) / (list.length - 1) : 0);
    const half = n.alongSize / 2;
    return [a, pos[n.rank]! + (end ? -half : half)];
  };
  const P = (a: number, r: number) => { const [x, yy] = at(a, r); grow(x - 2, yy - 2, x + 2, yy + 2); return `${r1(x)} ${r1(yy)}`; };
  const curve = (a0: number, r0: number, a1: number, r1_: number) => { const m = (r0 + r1_) / 2; return `C${P(a0, m)} ${P(a1, m)} ${P(a1, r1_)}`; };
  const labels: string[] = [];
  for (const c of chains) {
    if (c.e.line === "none") { if (c.labelAt > 0) labels.push(edgeLabel(L.get(c.ids[c.labelAt]!)!)); continue; }
    // in layout direction: start at the first id's exit, end at the last id's entry
    let [a, r] = port(c, 0);
    let d = `M${P(a, r)}`;
    for (let i = 1; i < c.ids.length; i++) {
      const n = L.get(c.ids[i]!)!;
      const last = i === c.ids.length - 1;
      const [na, nr] = last ? port(c, 1) : n.label ? [n.across, pos[n.rank]! - n.alongSize / 2] : [n.across, pos[n.rank]!];
      d += curve(a, r, na, nr);
      [a, r] = [na, nr];
      if (n.label && !last) { // the line stops at the label and goes on below it
        labels.push(edgeLabel(n));
        r = pos[n.rank]! + n.alongSize / 2;
        d += `M${P(a, r)}`;
      }
    }
    // marker-end is at the layout end; a reversed (back) edge's head belongs at its start
    const [mEnd, mStart] = c.reversed ? [c.e.tail, c.e.head] : [c.e.head, c.e.tail];
    const dash = c.e.line === "dotted" ? ` stroke-dasharray="3 4"` : "";
    const width = c.e.line === "thick" ? 2.5 : 1.5;
    body.push(`<path d="${d}" fill="none" stroke="muted" stroke-width="${width}" stroke-linecap="round"${dash}${mEnd ? ` marker-end="url(#${MARK[mEnd]})"` : ""}${mStart ? ` marker-start="url(#${MARK[mStart]})"` : ""}/>`);
  }
  function edgeLabel(n: LNode): string {
    const [x, yy] = center(n);
    return textEl(x, yy, n.label!, { size: 11, fill: "muted" });
  }
  // self-loops: a small arc on the node's far side
  for (const e of edges) if (e.from === e.to && e.line !== "none") {
    const n = nodes.get(e.from)!;
    const [x, yy] = center(L.get(n.id)!);
    const x0 = x + n.w / 2, s = 14;
    body.push(`<path d="M${r1(x0)} ${r1(yy - 7)}C${r1(x0 + 30)} ${r1(yy - 7 - s)} ${r1(x0 + 30)} ${r1(yy + 7 + s)} ${r1(x0 + 1)} ${r1(yy + 7)}" fill="none" stroke="muted" stroke-width="1.5"${e.head ? ` marker-end="url(#${MARK[e.head]})"` : ""}/>`);
    grow(x0, yy - 20, x0 + 26, yy + 20);
    if (e.label.some(Boolean)) {
      const lab = wrap(e.label, 120, 11);
      const lw = Math.max(...lab.map((l) => textWidth(l, 11)));
      labels.push(textEl(x0 + 30 + lw / 2, yy, lab, { size: 11, fill: "muted" }));
      grow(x0, yy - 10, x0 + 34 + lw, yy + 10);
    }
  }
  body.push(...labels);
  for (const n of nodes.values()) body.push(shape(n, ...center(L.get(n.id)!)));

  const pad = 6;
  return svgDoc([ext.x0 - pad, ext.y0 - pad, ext.x1 - ext.x0 + 2 * pad, ext.y1 - ext.y0 + 2 * pad], title, body.join(""));
}

/** A node's outline and label, centred on (x, y). */
function shape(n: FNode, x: number, y: number): string {
  const { w, h } = n;
  const fill = n.paint.fill ?? "surface", stroke = n.paint.stroke ?? "line";
  const st = `fill="${fill}" stroke="${stroke}" stroke-width="1.5"`;
  const l = x - w / 2, t = y - h / 2, R = x + w / 2, B = y + h / 2;
  const poly = (pts: [number, number][]) => `<polygon points="${pts.map(([a, b]) => `${r1(a)},${r1(b)}`).join(" ")}" ${st} stroke-linejoin="round"/>`;
  let outline: string;
  switch (n.shape) {
    case "round": outline = `<rect x="${r1(l)}" y="${r1(t)}" width="${w}" height="${h}" rx="14" ${st}/>`; break;
    case "stadium": outline = `<rect x="${r1(l)}" y="${r1(t)}" width="${w}" height="${h}" rx="${r1(h / 2)}" ${st}/>`; break;
    case "circle": outline = `<circle cx="${r1(x)}" cy="${r1(y)}" r="${r1(w / 2)}" ${st}/>`; break;
    case "dcircle": outline = `<circle cx="${r1(x)}" cy="${r1(y)}" r="${r1(w / 2)}" ${st}/><circle cx="${r1(x)}" cy="${r1(y)}" r="${r1(w / 2 - 4)}" fill="none" stroke="${stroke}" stroke-width="1.5"/>`; break;
    case "diamond": outline = poly([[x, t], [R, y], [x, B], [l, y]]); break;
    case "hex": outline = poly([[l + 12, t], [R - 12, t], [R, y], [R - 12, B], [l + 12, B], [l, y]]); break;
    case "para": outline = poly([[l + 10, t], [R, t], [R - 10, B], [l, B]]); break;
    case "trap": outline = poly([[l + 10, t], [R - 10, t], [R, B], [l, B]]); break;
    case "asym": outline = poly([[l, t], [R, t], [R, B], [l, B], [l + 12, y]]); break;
    case "sub": outline = `<rect x="${r1(l)}" y="${r1(t)}" width="${w}" height="${h}" rx="4" ${st}/><path d="M${r1(l + 7)} ${r1(t)}V${r1(B)}M${r1(R - 7)} ${r1(t)}V${r1(B)}" stroke="${stroke}" stroke-width="1.5"/>`; break;
    case "cyl": {
      const ry = 5;
      outline = `<path d="M${r1(l)} ${r1(t + ry)}A${r1(w / 2)} ${ry} 0 0 1 ${r1(R)} ${r1(t + ry)}V${r1(B - ry)}A${r1(w / 2)} ${ry} 0 0 1 ${r1(l)} ${r1(B - ry)}Z" ${st}/><path d="M${r1(l)} ${r1(t + ry)}A${r1(w / 2)} ${ry} 0 0 0 ${r1(R)} ${r1(t + ry)}" fill="none" stroke="${stroke}" stroke-width="1.5"/>`;
      break;
    }
    default: outline = `<rect x="${r1(l)}" y="${r1(t)}" width="${w}" height="${h}" rx="8" ${st}/>`;
  }
  return outline + textEl(x, n.shape === "cyl" ? y + 3 : y, n.lines);
}

// ───────────────────────────── sequence diagrams ─────────────────────────────

interface Actor { id: string; lines: string[]; w: number; x: number }
type SeqItem =
  | { t: "msg"; from: number; to: number; text: string[]; dashed: boolean; head: string; both: boolean }
  | { t: "note"; at: number[]; side: "left" | "right" | "over"; text: string[] }
  | { t: "open"; kind: string; label: string }
  | { t: "else"; label: string }
  | { t: "end" };

const ARROW = /^(.+?)\s*(<<-->>|<<->>|-->>|->>|--x|-x|--\)|-\)|-->|->)\s*[+-]?\s*(.+?)\s*:\s*(.*)$/;

function sequence(lines: string[], title0: string): string {
  let title = title0;
  const actors: Actor[] = [];
  const items: SeqItem[] = [];
  let auto = 0, depth = 0;
  const actor = (raw: string, alias?: string): number => {
    const id = raw.trim();
    if (!id) fail("empty participant");
    let i = actors.findIndex((a) => a.id === id);
    if (i < 0) {
      if (actors.length >= 40) fail("too many participants");
      actors.push({ id, lines: wrap(labelLines(alias ?? id), 140), w: 0, x: 0 });
      i = actors.length - 1;
    } else if (alias) actors[i]!.lines = wrap(labelLines(alias), 140);
    return i;
  };
  for (const line of lines) {
    let m: RegExpExecArray | null;
    if ((m = /^(?:create\s+)?(?:participant|actor)\s+(.+?)(?:\s+as\s+(.+))?$/.exec(line))) actor(m[1]!, m[2]);
    else if ((m = /^title:?\s+(.+)$/.exec(line))) title = m[1]!.trim();
    else if ((m = /^autonumber\b/.exec(line))) auto = 1;
    else if (/^(activate|deactivate|destroy|links?|properties|details)\b/.test(line)) { /* nothing to draw */ }
    else if ((m = /^note\s+(left of|right of|over)\s+([^:]+?)\s*:\s*(.*)$/i.exec(line))) {
      const at = m[2]!.split(",").map((s) => actor(s));
      items.push({ t: "note", at, side: m[1]!.toLowerCase().startsWith("left") ? "left" : m[1]!.toLowerCase().startsWith("right") ? "right" : "over", text: wrap(labelLines(m[3]!), 160) });
    } else if ((m = /^(loop|alt|opt|par|critical|break|rect)\b\s*(.*)$/.exec(line))) {
      items.push({ t: "open", kind: m[1]!, label: m[1] === "rect" ? "" : labelLines(m[2] ?? "").join(" ") });
      depth++;
    } else if ((m = /^box\b/.exec(line))) { items.push({ t: "open", kind: "box", label: "" }); depth++; }
    else if ((m = /^(else|and|option)\b\s*(.*)$/.exec(line))) items.push({ t: "else", label: labelLines(m[2] ?? "").join(" ") });
    else if (line === "end") { if (!depth--) fail("end without block"); items.push({ t: "end" }); }
    else if ((m = ARROW.exec(line))) {
      const op = m[2]!;
      const from = actor(m[1]!), to = actor(m[3]!);
      const text = labelLines(m[4]!);
      if (auto) text[0] = `${auto++}. ${text[0] ?? ""}`;
      items.push({ t: "msg", from, to, text: wrap(text, 220, 11.5), dashed: op.includes("--"), head: op.endsWith(">>") || op.endsWith(")") ? ">" : op.endsWith("x") ? "x" : "", both: op.startsWith("<<") });
      if (items.length > MAX_EDGES) fail("too many messages");
    } else fail(`cannot read "${line.slice(0, 30)}"`);
  }
  if (!actors.length) fail("no participants");

  // columns: each actor's box, then gaps wide enough for the messages and notes between neighbours
  for (const a of actors) a.w = Math.max(72, Math.max(...a.lines.map((l) => textWidth(l))) + 24);
  const gap = actors.map((a, i) => (i ? (actors[i - 1]!.w + a.w) / 2 + 28 : 0));
  const need = (a: number, b: number, w: number) => { // columns a < b at least w apart
    const have = gap.slice(a + 1, b + 1).reduce((s, g) => s + g, 0);
    if (have < w) gap[b] = gap[b]! + (w - have);
  };
  let rightPad = 0, leftPad = 0;
  for (const it of items) {
    if (it.t === "msg") {
      const w = Math.max(...it.text.map((l) => textWidth(l, 11.5))) + 28;
      if (it.from === it.to) { if (it.from + 1 < actors.length) need(it.from, it.from + 1, w + 30); else rightPad = Math.max(rightPad, w + 20); }
      else need(Math.min(it.from, it.to), Math.max(it.from, it.to), w);
    } else if (it.t === "note") {
      const w = Math.max(...it.text.map((l) => textWidth(l, 11.5))) + 24;
      const a = Math.min(...it.at), b = Math.max(...it.at);
      if (it.side === "over") { if (a !== b) need(a, b, w - 40); }
      else if (it.side === "right") { if (a + 1 < actors.length) need(a, a + 1, w + 30); else rightPad = Math.max(rightPad, w + 16); }
      else if (a > 0) need(a - 1, a, w + 30); else leftPad = Math.max(leftPad, w + 16);
    }
  }
  let x = leftPad + actors[0]!.w / 2;
  actors.forEach((a, i) => { x += i ? gap[i]! : 0; a.x = x; });
  const headH = Math.max(...actors.map((a) => a.lines.length)) * LH + 16;
  const body: string[] = [];
  let y = headH + 18;
  let minX = -4, maxX = 0;
  const blocks: { kind: string; label: string; y: number; lo: number; hi: number; right: number; elses: [number, string][]; depth: number }[] = [];
  const drawnBlocks: string[] = [];
  const touch = (cols: number[]) => { for (const b of blocks) { b.lo = Math.min(b.lo, ...cols); b.hi = Math.max(b.hi, ...cols); } };
  for (const it of items) {
    if (it.t === "msg") {
      const a = actors[it.from]!, b = actors[it.to]!;
      const th = it.text.length * 14;
      touch([it.from, it.to]);
      const tw = Math.max(...it.text.map((l) => textWidth(l, 11.5)));
      const mid = it.from === it.to ? a.x + 34 + tw / 2 : (a.x + b.x) / 2;
      if (it.from === it.to) for (const bl of blocks) bl.right = Math.max(bl.right, a.x + 42 + tw);
      body.push(textEl(mid, y + th / 2 - 3, it.text, { size: 11.5 }));
      y += th + 4;
      const dash = it.dashed ? ` stroke-dasharray="4 3"` : "";
      const mk = `${it.head ? ` marker-end="url(#${MARK[it.head]})"` : ""}${it.both ? ` marker-start="url(#mma)"` : ""}`;
      if (it.from === it.to) {
        body.push(`<path d="M${r1(a.x)} ${r1(y)}H${r1(a.x + 28)}V${r1(y + 16)}H${r1(a.x + 2)}" fill="none" stroke="muted" stroke-width="1.5" stroke-linejoin="round"${dash}${mk}/>`);
        y += 16;
      } else {
        const dir = Math.sign(b.x - a.x);
        body.push(`<path d="M${r1(a.x)} ${r1(y)}H${r1(b.x - dir)}" fill="none" stroke="muted" stroke-width="1.5"${dash}${mk}/>`);
      }
      y += 22;
    } else if (it.t === "note") {
      const cols = it.at.map((i) => actors[i]!);
      const tw = Math.max(...it.text.map((l) => textWidth(l, 11.5))) + 20;
      let x0: number, x1: number;
      if (it.side === "over") { x0 = Math.min(...cols.map((c) => c.x)) - 20; x1 = Math.max(...cols.map((c) => c.x)) + 20; if (x1 - x0 < tw) { const c = (x0 + x1) / 2; x0 = c - tw / 2; x1 = c + tw / 2; } }
      else if (it.side === "right") { x0 = cols[0]!.x + 10; x1 = x0 + tw; }
      else { x1 = cols[0]!.x - 10; x0 = x1 - tw; }
      const h = it.text.length * LH + 12;
      touch(it.at);
      body.push(`<rect x="${r1(x0)}" y="${r1(y)}" width="${r1(x1 - x0)}" height="${h}" rx="6" fill="orange-soft" stroke="orange" stroke-width="1"/>`);
      body.push(textEl((x0 + x1) / 2, y + h / 2, it.text, { size: 11.5 }));
      minX = Math.min(minX, x0 - 4);
      y += h + 16;
    } else if (it.t === "open") {
      blocks.push({ kind: it.kind, label: it.label, y, lo: Infinity, hi: -Infinity, right: -Infinity, elses: [], depth: blocks.length });
      if (it.kind !== "box" && it.kind !== "rect") y += 26;
    } else if (it.t === "else") {
      const b = blocks[blocks.length - 1];
      if (b) { b.elses.push([y, it.label]); y += 22; }
    } else {
      const b = blocks.pop()!;
      if (b.kind === "box") continue;
      const lo = Number.isFinite(b.lo) ? b.lo : 0, hi = Number.isFinite(b.hi) ? b.hi : actors.length - 1;
      // nested frames sit inside the one around them
      const x0 = actors[lo]!.x - actors[lo]!.w / 2 - 12 + 6 * blocks.length;
      const x1 = Math.max(actors[hi]!.x + actors[hi]!.w / 2 + 12 - 6 * blocks.length, b.right + 6, x0 + textWidth(b.kind, 11) + textWidth(b.label, 11) + 40);
      minX = Math.min(minX, x0 - 4);
      maxX = Math.max(maxX, x1 + 4);
      const y1 = y - 6;
      const parts = [`<rect x="${r1(x0)}" y="${r1(b.y)}" width="${r1(x1 - x0)}" height="${r1(y1 - b.y)}" rx="6" fill="${b.kind === "rect" ? "accent-soft" : "none"}" stroke="line" stroke-width="1"/>`];
      if (b.kind !== "rect") {
        const kw = textWidth(b.kind, 11) + 14;
        parts.push(`<path d="M${r1(x0)} ${r1(b.y + 20)}H${r1(x0 + kw)}l6 -6V${r1(b.y)}" fill="surface" stroke="line" stroke-width="1"/>`);
        parts.push(`<text x="${r1(x0 + 7)}" y="${r1(b.y + 14)}" font-size="11" font-weight="600" fill="ink">${esc(b.kind)}</text>`);
        if (b.label) parts.push(`<text x="${r1(x0 + kw + 12)}" y="${r1(b.y + 14)}" font-size="11" fill="muted">[${esc(b.label)}]</text>`);
      }
      for (const [ey, label] of b.elses) {
        parts.push(`<path d="M${r1(x0)} ${r1(ey)}H${r1(x1)}" stroke="line" stroke-width="1" stroke-dasharray="4 3"/>`);
        if (label) parts.push(`<text x="${r1(x0 + 8)}" y="${r1(ey + 14)}" font-size="11" fill="muted">[${esc(label)}]</text>`);
      }
      drawnBlocks.push(parts.join(""));
      // the enclosing block covers these columns too
      if (blocks.length) { touch([lo, hi]); blocks[blocks.length - 1]!.right = Math.max(blocks[blocks.length - 1]!.right, x1 + 6); }
      y += 8;
    }
  }
  if (blocks.length) fail("unclosed block");
  const bottom = y;
  const heads = actors.map((a) => {
    return `<path d="M${r1(a.x)} ${headH}V${r1(bottom)}" stroke="line" stroke-width="1" stroke-dasharray="3 4"/>` +
      `<rect x="${r1(a.x - a.w / 2)}" y="0" width="${r1(a.w)}" height="${headH}" rx="8" fill="surface" stroke="line" stroke-width="1.5"/>` +
      textEl(a.x, headH / 2, a.lines, { bold: true });
  });
  const x1 = Math.max(maxX, Math.max(...actors.map((a) => a.x + a.w / 2)) + rightPad + 4);
  return svgDoc([minX, -4, x1 - minX, bottom + 10], title, heads.join("") + drawnBlocks.join("") + body.join(""));
}
