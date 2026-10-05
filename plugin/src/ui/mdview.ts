// Markdown tree to DOM, and the streaming view. DOM nodes are created with createElement and set with
// textContent/setAttribute: never innerHTML, never a parsed string. The allow-lists from markdown.ts are
// enforced again here, so a bug in the tree builder still cannot create a <script> or an onerror.
import type { Token } from "marked";
import type { NoteRequest, SavedNote } from "../types.ts";
import { ALLOWED_ATTRS, ALLOWED_CLASSES, ALLOWED_TAGS, HEX_COLOR, MAX_MD, blockNodes, citeTitle, lexBlocks, safeHref, safeImageSrc, settledBlock } from "./markdown.ts";
import type { MdNode } from "./markdown.ts";
import { copyText, edgeFade, env, flashCheck, h, icon } from "./dom.ts";

/** `open` gets a link or citation chip that was clicked (http(s) or zotero: only ever reaches it); `saveFile` is the host's file picker
 *  (diagram export), `saveNote` the host's note writer (a diagram's "Add to a note"). */
interface MdHooks {
  open(href: string): void;
  saveFile?(name: string, data: Uint8Array | string, mime: string): Promise<string | null>;
  saveNote?(note: NoteRequest): Promise<SavedNote>;
}
/** What a block's rendering needs besides its tree: is it the block still streaming in, and the hooks. */
interface RenderCtx { live: boolean; hooks: MdHooks }

// ───────────────────────────── math (lazy) ─────────────────────────────

interface KatexLike { render(tex: string, el: HTMLElement, opts: Record<string, unknown>): void }
let katexP: Promise<KatexLike | null> | null = null;
const mathCache = new Map<string, Node>();
const MATH_CACHE_MAX = 300;

function loadKatex(): Promise<KatexLike | null> {
  // The dynamic import keeps katex out of first paint; MathML output needs no stylesheet or fonts in Gecko.
  katexP ??= import("katex").then((m) => ((m as { default?: KatexLike }).default ?? (m as unknown as KatexLike))).catch(() => null);
  return katexP;
}

const mathKey = (tex: string, display: boolean) => (display ? "D" : "I") + tex;

/** Render `tex` into `host`; leaves the source text there if katex is missing or the TeX is bad. */
async function renderMath(host: HTMLElement, tex: string, display: boolean): Promise<void> {
  const key = mathKey(tex, display);
  const hit = mathCache.get(key);
  if (hit) { host.replaceChildren(hit.cloneNode(true)); host.classList.remove("math--raw"); if (display) fitMath(host); return; }
  const katex = await loadKatex();
  if (!katex || (!host.isConnected && !host.parentNode)) return;
  // katex builds its nodes with the free variable `document`; a plugin scope may not have one.
  const g = globalThis as { document?: Document };
  g.document ??= env.doc;
  try {
    katex.render(tex, host, { displayMode: display, throwOnError: true, strict: "ignore", trust: false, output: "mathml", maxSize: 50, maxExpand: 500 });
    host.classList.remove("math--raw");
    if (display) fitMath(host);
    if (mathCache.size >= MATH_CACHE_MAX) mathCache.delete(mathCache.keys().next().value as string);
    mathCache.set(key, host.cloneNode(true).firstChild as Node);
  } catch (e) {
    host.textContent = display ? tex : `$${tex}$`;
    host.classList.remove("math--raw");
    host.classList.add("math--error");
    host.title = String((e as Error)?.message ?? e).replace(/^KaTeX parse error: /, "");
  }
}

/** Below this a wide formula stops shrinking and scrolls sideways instead (a soft fade on the side that has more). */
const MIN_MATH_SCALE = 0.72;
let mathRO: ResizeObserver | null = null;

/** A display formula wider than the panel: smaller first, down to 72%, then scrolling; never clipped. Again on resize. */
function fitMath(host: HTMLElement): void {
  if (!host.isConnected) { env.win.requestAnimationFrame?.(() => { if (host.isConnected) fitMath(host); }); return; }
  const RO = env.win.ResizeObserver;
  if (RO && !host.dataset.fit) {
    host.dataset.fit = "1";
    mathRO ??= new RO((entries) => { for (const e of entries) fitMath(e.target as HTMLElement); });
    mathRO.observe(host);
    host.addEventListener("scroll", () => edgeFade(host), { passive: true });
  }
  const cw = host.clientWidth;
  if (!cw) return;
  host.style.fontSize = "";
  let scale = 1;
  // Glyph spacing does not scale exactly with the size: measure again after each step (twice is enough).
  for (let i = 0; i < 2 && host.scrollWidth > cw + 1 && scale > MIN_MATH_SCALE; i++) {
    scale = Math.max(MIN_MATH_SCALE, Math.floor((scale * cw * 100) / host.scrollWidth) / 100);
    host.style.fontSize = `${Math.round(scale * 100)}%`;
  }
  edgeFade(host);
}

// ───────────────────────────── diagrams (lazy) ─────────────────────────────
// diagram.ts runs only once a ```svg block shows up (esbuild bundles the import as a lazily initialised module).

type DiagramMod = typeof import("./diagram.ts");
let diagramMod: DiagramMod | null = null;
let diagramP: Promise<DiagramMod | null> | null = null;
const loadDiagram = (): Promise<DiagramMod | null> => (diagramP ??= import("./diagram.ts").then((m) => (diagramMod = m)).catch(() => null));

/** A ```svg block: "Drawing…" while it streams, the figure once its fence closes, the code if it is no drawing. */
function diagram(src: string, closed: boolean, rc: RenderCtx): HTMLElement {
  const card = h("div");
  const asCode = () => { card.className = ""; card.removeAttribute("role"); card.replaceChildren(codeBlock(src, "svg")); };
  const draw = (m: DiagramMod | null) => {
    if (!m) return asCode();
    if (!closed) return rc.live ? m.pending(card) : asCode(); // cut off mid-drawing: show what came, as code
    const { saveFile, saveNote, open } = rc.hooks;
    if (!m.fill(card, src, { codeBlock, open, ...(saveFile ? { saveFile } : {}), ...(saveNote ? { saveNote } : {}) })) asCode();
  };
  if (diagramMod) draw(diagramMod);
  else { card.className = "dg dg--pending"; void loadDiagram().then(draw); }
  return card;
}

// ───────────────────────────── tree to DOM ─────────────────────────────

function textOf(n: MdNode): string {
  return typeof n === "string" ? n : (n.kids ?? []).map(textOf).join("");
}

function toDom(node: MdNode, rc: RenderCtx): Node {
  if (typeof node === "string") return env.doc.createTextNode(node);
  const { tag, attrs = {}, kids = [] } = node;
  const sub = (n: MdNode) => toDom(n, rc);
  switch (tag) {
    case "codeblock": return codeBlock(textOf(node), attrs.lang);
    case "diagram": return diagram(textOf(node), attrs.open !== "1", rc);
    case "math": {
      const display = attrs.display === "1";
      const tex = textOf(node);
      const host = h(display ? "div.math.math--display.math--raw" : "span.math.math--raw", null, tex);
      void renderMath(host, tex, display);
      if (!display) return host;
      // Copy TeX: the formula's source, from a quiet button that shows on hover or focus.
      const copy = h("button.iconbtn.iconbtn--sm.math__copy", { type: "button", "aria-label": "Copy TeX", title: "Copy TeX" }, icon("copy"));
      copy.addEventListener("click", () => void copyText(tex).then((ok) => ok && flashCheck(copy)));
      return h("div.mathblock", null, host, copy);
    }
    case "cite": {
      const href = safeHref(attrs.href);
      if (!href) return env.doc.createTextNode(textOf(node));
      return h("button.cite", { type: "button", dataset: { href }, title: citeTitle(href) }, h("span.cite__t", null, icon("item"), textOf(node)));
    }
    case "tablewrap": return h("div.md-table", null, ...kids.map(sub));
    case "img": {
      const src = safeImageSrc(attrs.src);
      return src ? h("img", { src, alt: attrs.alt ?? "" }) : env.doc.createTextNode(attrs.alt ?? "");
    }
    case "a": {
      const href = safeHref(attrs.href);
      if (!href) return h("span", null, ...kids.map(sub));
      return h("a", { href, ...(attrs.title ? { title: attrs.title } : {}), rel: "noopener noreferrer" }, ...kids.map(sub));
    }
  }
  if (!ALLOWED_TAGS.has(tag)) return h("span", null, ...kids.map(sub));
  const out = h(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (!ALLOWED_ATTRS.has(k) || k === "href" || k === "src") continue;
    if (k === "class") {
      const ok = v.split(/\s+/).filter((c) => ALLOWED_CLASSES.has(c));
      if (ok.length) out.className = ok.join(" ");
    } else if (k === "start" || k === "colspan") { if (/^\d{1,6}$/.test(v)) out.setAttribute(k, v); }
    else if (k === "align") { if (v === "left" || v === "center" || v === "right") out.setAttribute("align", v); }
    else if (k === "color" || k === "bg") { if (tag === "span" && HEX_COLOR.test(v)) out.style.setProperty(k === "bg" ? "background-color" : "color", v); }
    else out.setAttribute(k, v);
  }
  for (const kid of kids) out.appendChild(sub(kid));
  return out;
}

function codeBlock(code: string, lang?: string): HTMLElement {
  const copy = h("button.iconbtn.iconbtn--sm.code__copy", { type: "button", "aria-label": "Copy code", title: "Copy" }, icon("copy"));
  copy.addEventListener("click", () => void copyText(code).then((ok) => ok && flashCheck(copy)));
  return h("div.code", null,
    h("div.code__head", null, h("span.code__lang", null, lang ?? "text"), copy),
    h("pre", null, h("code", null, code)));
}

// ───────────────────────────── the streaming view ─────────────────────────────

interface BlockView { start: number; key: string; nodes: Node[]; live: boolean }

/**
 * One reply's Markdown. `set(text, streaming)` lexes only from the start of the last block (the blocks
 * before it are final) and replaces only that block's nodes, so a long answer costs the same per token
 * at the end as at the start.
 */
export class MdView {
  readonly el: HTMLElement;
  private blocks: BlockView[] = [];
  private text = "";
  private refDefs = false;
  private tailEl: HTMLElement | null = null;

  private hooks: MdHooks;

  constructor(hooks: MdHooks) {
    this.hooks = hooks;
    this.el = h("div.md");
    this.el.addEventListener("click", (e) => {
      const t = e.target as Element | null;
      const cite = t?.closest?.("button.cite") as HTMLElement | null;
      if (cite?.dataset.href) { e.preventDefault(); this.hooks.open(cite.dataset.href); return; }
      const a = t?.closest?.("a[href]") as HTMLAnchorElement | null;
      if (a) {
        e.preventDefault(); // never navigate the (privileged) window; the host decides what a link opens
        const href = safeHref(a.getAttribute("href"));
        if (href) this.hooks.open(href);
      }
    });
  }

  set(text: string, streaming: boolean): void {
    if (text.includes("\r")) text = text.replace(/\r\n?/g, "\n");
    this.el.classList.toggle("md--streaming", streaming);
    if (text === this.text) return;
    const prev = this.text;
    this.text = text;

    let tail = "";
    if (text.length > MAX_MD) { tail = text.slice(MAX_MD); text = text.slice(0, MAX_MD); }

    const last = this.blocks[this.blocks.length - 1];
    let from = 0;
    if (last && !this.refDefs && samePrefix(prev, text, last.start)) {
      if (REF_DEF.test(text.slice(last.start))) this.refDefs = true;
      else from = last.start;
    } else this.refDefs = REF_DEF.test(text);

    const tokens = lexBlocks(text.slice(from));
    const fresh: { tok: Token; start: number; key: string }[] = [];
    let off = from;
    for (const t of tokens) {
      if (t.type !== "space") fresh.push({ tok: t, start: off, key: t.raw.trimEnd() });
      off += t.raw.length;
    }
    // Tokens that do not add up to the text (a lexer quirk) cannot be diffed: start over next time.
    if (off !== text.length) for (const f of fresh) f.start = 0;

    const old = this.blocks.splice(from ? this.blocks.length - 1 : 0);
    const n = fresh.length;
    let i = 0;
    while (i < n && i < old.length && (old[i] as BlockView).key === (fresh[i] as { key: string }).key && !((old[i] as BlockView).live && !(streaming && i === n - 1))) i++;
    for (const b of old.splice(i)) for (const nd of b.nodes) (nd as ChildNode).remove();

    // Insert new blocks before the tail (the capped remainder), after the kept ones.
    for (; i < n; i++) {
      const f = fresh[i] as { tok: Token; start: number; key: string };
      const live = streaming && i === n - 1;
      // The block still arriving shows only what is settled: never a half-written formula or table as raw text.
      const nodes = (live ? settledBlock(f.tok) : [f.tok]).flatMap((t) => blockNodes(t)).map((nd) => toDom(nd, { live, hooks: this.hooks }));
      for (const nd of nodes) this.el.insertBefore(nd, this.tailEl);
      old.push({ start: f.start, key: f.key, nodes, live });
    }
    this.blocks.push(...old);
    this.setTail(tail);
  }

  private setTail(tail: string): void {
    if (!tail) { this.tailEl?.remove(); this.tailEl = null; return; }
    if (!this.tailEl) { this.tailEl = h("pre.md-tail"); this.el.appendChild(this.tailEl); }
    this.tailEl.textContent = tail;
  }
}

const REF_DEF = /^ {0,3}\[[^\]^][^\]]*\]:/m; // a link definition: earlier blocks depend on later text

// Same [0, n)? Head + last 512 chars only: a full compare per token is quadratic.
function samePrefix(a: string, b: string, n: number): boolean {
  if (a.length < n || b.length < n) return false;
  const hd = Math.min(n, 128), t = Math.max(hd, n - 512);
  return a.startsWith(b.slice(0, hd)) && a.slice(t, n) === b.slice(t, n);
}

/** Plain text in a code-ish box (tool input and output): a string, JSON for objects, capped. */
export function pretty(v: unknown, max = 6000): { text: string; trimmed: boolean } {
  let s: string;
  if (typeof v === "string") s = v;
  else { try { s = JSON.stringify(v, null, 2) ?? String(v); } catch { s = String(v); } }
  return s.length > max ? { text: s.slice(0, max), trimmed: true } : { text: s, trimmed: false };
}
