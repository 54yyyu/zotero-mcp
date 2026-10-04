// Markdown tree to DOM, and the streaming view. DOM nodes are created with createElement and set with
// textContent/setAttribute: never innerHTML, never a parsed string. The allow-lists from markdown.ts are
// enforced again here, so a bug in the tree builder still cannot create a <script> or an onerror.
import type { Token } from "marked";
import { ALLOWED_ATTRS, ALLOWED_CLASSES, ALLOWED_TAGS, MAX_MD, blockNodes, lexBlocks, safeHref, safeImageSrc } from "./markdown.ts";
import type { MdNode } from "./markdown.ts";
import { copyText, env, flashCheck, h, icon } from "./dom.ts";

/** `open` gets a link or citation chip that was clicked (http(s) or zotero: only ever reaches it). */
interface MdHooks { open(href: string): void }

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
  if (hit) { host.replaceChildren(hit.cloneNode(true)); host.classList.remove("math--raw"); return; }
  const katex = await loadKatex();
  if (!katex || (!host.isConnected && !host.parentNode)) return;
  // katex builds its nodes with the free variable `document`; a plugin scope may not have one.
  const g = globalThis as { document?: Document };
  g.document ??= env.doc;
  try {
    katex.render(tex, host, { displayMode: display, throwOnError: true, strict: "ignore", trust: false, output: "mathml", maxSize: 50, maxExpand: 500 });
    host.classList.remove("math--raw");
    if (mathCache.size >= MATH_CACHE_MAX) mathCache.delete(mathCache.keys().next().value as string);
    mathCache.set(key, host.cloneNode(true).firstChild as Node);
  } catch (e) {
    host.textContent = display ? tex : `$${tex}$`;
    host.classList.remove("math--raw");
    host.classList.add("math--error");
    host.title = String((e as Error)?.message ?? e).replace(/^KaTeX parse error: /, "");
  }
}

// ───────────────────────────── tree to DOM ─────────────────────────────

function textOf(n: MdNode): string {
  return typeof n === "string" ? n : (n.kids ?? []).map(textOf).join("");
}

function toDom(node: MdNode): Node {
  if (typeof node === "string") return env.doc.createTextNode(node);
  const { tag, attrs = {}, kids = [] } = node;
  switch (tag) {
    case "codeblock": return codeBlock(textOf(node), attrs.lang);
    case "math": {
      const display = attrs.display === "1";
      const tex = textOf(node);
      const host = h(display ? "div.math.math--display.math--raw" : "span.math.math--raw", null, tex);
      void renderMath(host, tex, display);
      return host;
    }
    case "cite": {
      const href = safeHref(attrs.href);
      if (!href) return env.doc.createTextNode(textOf(node));
      return h("button.cite", { type: "button", dataset: { href }, title: href.replace(/^zotero:\/\//, "") }, h("span.cite__t", null, icon("item"), textOf(node)));
    }
    case "tablewrap": return h("div.md-table", null, ...kids.map(toDom));
    case "img": {
      const src = safeImageSrc(attrs.src);
      return src ? h("img", { src, alt: attrs.alt ?? "" }) : env.doc.createTextNode(attrs.alt ?? "");
    }
    case "a": {
      const href = safeHref(attrs.href);
      if (!href) return h("span", null, ...kids.map(toDom));
      return h("a", { href, ...(attrs.title ? { title: attrs.title } : {}), rel: "noopener noreferrer" }, ...kids.map(toDom));
    }
  }
  if (!ALLOWED_TAGS.has(tag)) return h("span", null, ...kids.map(toDom));
  const out = h(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (!ALLOWED_ATTRS.has(k) || k === "href" || k === "src") continue;
    if (k === "class") {
      const ok = v.split(/\s+/).filter((c) => ALLOWED_CLASSES.has(c));
      if (ok.length) out.className = ok.join(" ");
    } else if (k === "start" || k === "colspan") { if (/^\d{1,6}$/.test(v)) out.setAttribute(k, v); }
    else if (k === "align") { if (v === "left" || v === "center" || v === "right") out.setAttribute("align", v); }
    else out.setAttribute(k, v);
  }
  for (const kid of kids) out.appendChild(toDom(kid));
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
      const nodes = blockNodes(f.tok).map(toDom);
      for (const nd of nodes) this.el.insertBefore(nd, this.tailEl);
      old.push({ start: f.start, key: f.key, nodes, live: streaming && i === n - 1 });
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
