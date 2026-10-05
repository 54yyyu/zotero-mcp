// Markdown to a small node tree. PURE: no DOM, so it runs under `node --test` and the XSS corpus can
// inspect exactly what the agent's text can become. mdview.ts turns the tree into DOM nodes with
// createElement; there is no HTML string anywhere between the agent's text and the screen.
//
// Policy (the panel runs in a privileged window and the agent reads the web):
//   links    http(s): and zotero: only; anything else renders as its plain text
//   images   data:image/png|jpeg;base64 only; anything else renders as its alt text
//   raw HTML renders as escaped text (it is a text node, so there is nothing to escape), except the formatting
//            subset Zotero notes have and markdown lacks: <u> <s> <del> <sub> <sup> <mark> and <span style="color: …;
//            background-color: …"> with a palette name or a hex colour (FORMAT_TAGS); any other attribute is dropped
//   tags and attributes come from the allow-lists below, enforced again by the DOM builder
import { Marked } from "marked";
import type { Token, Tokens } from "marked";

export type MdNode = string | MdEl;
export interface MdEl { tag: string; attrs?: Record<string, string>; kids?: MdNode[] }

/** Element tags the tree may contain. `codeblock`, `math`, `cite`, `tablewrap`, `diagram` are ours; the DOM builder knows them. */
export const ALLOWED_TAGS: ReadonlySet<string> = new Set([
  "p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "blockquote", "pre", "code", "em", "strong", "del",
  "a", "br", "hr", "table", "thead", "tbody", "tr", "th", "td", "img", "span",
  "codeblock", "math", "cite", "tablewrap", "diagram", "u", "sub", "sup",
]);
/** `color` and `bg` (on span only) hold a validated hex colour, nothing else. */
export const ALLOWED_ATTRS: ReadonlySet<string> = new Set(["href", "src", "alt", "title", "start", "align", "lang", "display", "class", "colspan", "open", "color", "bg"]);
/** The only class names a tree may carry (set by this file, never from agent text). */
export const ALLOWED_CLASSES: ReadonlySet<string> = new Set(["md-raw", "md-task", "md-task--on", "md-nobr"]);

const MAX_DEPTH = 24;
/** Beyond this the rest of a reply is shown as plain text: a megabyte of markdown is not a reply. */
export const MAX_MD = 400_000;
const MAX_IMG = 2_000_000;
const CHUNK = 8000;
const BUDGET_MS = 600;

// ───────────────────────────── urls ─────────────────────────────

/** `href` if it is http(s) or zotero, else null. Control characters anywhere reject it (browsers ignore tabs inside a scheme). */
export function safeHref(href: string | null | undefined): string | null {
  if (!href) return null;
  const u = href.trim();
  // eslint-disable-next-line no-control-regex
  if (/[\x00-\x20\x7f-\x9f]/.test(u) || u.includes(String.fromCharCode(0x2028)) || u.includes(String.fromCharCode(0x2029))) return null;
  if (/^https?:\/\/[^/?#\s]/i.test(u) || /^zotero:\/\/[^\s]/i.test(u)) return u;
  return null;
}

export function safeImageSrc(src: string | null | undefined): string | null {
  if (!src) return null;
  const u = src.trim();
  if (u.length > MAX_IMG) return null;
  return /^data:image\/(?:png|jpeg);base64,[A-Za-z0-9+/]+={0,2}$/.test(u) ? u : null;
}

export interface ZoteroLink {
  /** 1 = My Library; a group's id otherwise. */
  libraryID: number;
  itemKey: string;
  page?: number;
  annotationKey?: string;
  /** The cited words (`&quote=`), which the reader flashes on the page. Only ever text to search for. */
  quote?: string;
  /** open-pdf (a page of an attachment) or select (an item in the library). */
  action: "open-pdf" | "select" | "open-note" | "other";
}

/** zotero://open-pdf/library/items/KEY?page=8, zotero://select/groups/123/items/KEY, ... */
export function parseZoteroUri(uri: string): ZoteroLink | null {
  const m = /^zotero:\/\/([a-z-]+)\/(library|groups\/(\d+))\/items\/([A-Za-z0-9]{8})(?:[/?#]|$)/.exec(uri.trim());
  if (!m) return null;
  const action = m[1] === "open-pdf" || m[1] === "select" || m[1] === "open-note" ? m[1] : "other";
  const q = /\?([^#]*)/.exec(uri)?.[1] ?? "";
  const params = new URLSearchParams(q);
  const pg = Number(params.get("page"));
  const ann = params.get("annotation");
  const quote = params.get("quote")?.trim().slice(0, 500);
  return {
    libraryID: m[3] ? Number(m[3]) : 1,
    itemKey: (m[4] as string).toUpperCase(),
    ...(Number.isFinite(pg) && pg > 0 ? { page: pg } : {}),
    ...(ann && /^[A-Za-z0-9]{8}$/.test(ann) ? { annotationKey: ann } : {}),
    ...(quote ? { quote } : {}),
    action,
  };
}

/** A citation chip's tooltip: the quoted words when the link carries them, else where it points. */
export function citeTitle(href: string): string {
  const q = parseZoteroUri(href)?.quote;
  return q ? `“${q}”` : href.replace(/^zotero:\/\//, "");
}

/** The link without its `quote`: the sources fold opens the page, it lists several. */
const withoutQuote = (href: string): string => href.replace(/([?&])quote=[^&#]*&?/, "$1").replace(/[?&]$/, "");

// ───────────────────────────── entities ─────────────────────────────

const NAMED: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", ndash: "–", mdash: "—", hellip: "…" };
/** marked leaves entities in text tokens as written (HTML output would have decoded them); decode the few we know. */
export function decodeEntities(s: string): string {
  if (!s.includes("&")) return s;
  return s.replace(/&(?:#(\d{1,7})|#x([0-9a-f]{1,6})|([a-z]{2,8}));/gi, (all, dec, hex, name) => {
    if (name) return NAMED[String(name).toLowerCase()] ?? all;
    const cp = dec ? Number(dec) : parseInt(hex, 16);
    if (!(cp >= 32 && cp <= 0x10ffff) || (cp >= 0xd800 && cp <= 0xdfff)) return all;
    return String.fromCodePoint(cp);
  });
}

// ───────────────────────────── colours and the formatting subset ─────────────────────────────

/** Zotero's note editor palette (its text colours, and its highlight colours at 50%): the names the agent may use. */
export const TEXT_COLORS: Readonly<Record<string, string>> = {
  red: "#ff2020", orange: "#ff7700", yellow: "#ffcb00", green: "#4eb31c", purple: "#7953e3", magenta: "#eb52f7", blue: "#05a2ef", gray: "#7e8386",
};
export const HIGHLIGHT_COLORS: Readonly<Record<string, string>> = {
  red: "#ff666680", orange: "#f1983780", yellow: "#ffd40080", green: "#5fb23680", purple: "#a28ae580", magenta: "#e56eee80", blue: "#2ea8e580", gray: "#aaaaaa80",
};
export const HEX_COLOR = /^#(?:[0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})$/i;

/** A palette name or a hex colour as hex; anything else (url(), expressions, var()) is null. */
export function safeColor(v: string | undefined, palette: Readonly<Record<string, string>>): string | null {
  const c = (v ?? "").trim().toLowerCase().replace(/^grey$/, "gray");
  return palette[c] ?? (HEX_COLOR.test(c) ? c : null);
}

const FORMAT_TAGS = "u|s|del|strike|sub|sup|mark|span";

/** The colours of `<span style="…">`: only color and background(-color), only palette names or hex. */
function spanColors(attrs: string): { color?: string; bg?: string } {
  const style = /\bstyle\s*=\s*(?:"([^"]*)"|'([^']*)')/i.exec(attrs);
  const out: { color?: string; bg?: string } = {};
  for (const decl of (style?.[1] ?? style?.[2] ?? "").split(";")) {
    const m = /^\s*(color|background-color|background)\s*:\s*([^;]+?)\s*$/i.exec(decl);
    if (!m) continue;
    const prop = (m[1] as string).toLowerCase();
    const c = safeColor(m[2], prop === "color" ? TEXT_COLORS : HIGHLIGHT_COLORS);
    if (c) out[prop === "color" ? "color" : "bg"] = c;
  }
  return out;
}

interface FormatToken { type: "fmt"; raw: string; tag: string; color?: string; bg?: string; tokens: Token[] }

// ───────────────────────────── marked with math ─────────────────────────────
// `$x$` must hug its content and not be followed by a digit, so "costs $5 and $10" stays text.

const RE = {
  dollar: /^\$(?=[^\s$])((?:\\[\s\S]|[^\\$\n]|\n(?!\s*\n)){0,2000}?[^\s\\$])\$(?!\d)/,
  paren: /^\\\(((?:\\[^)]|[^\\]){1,2000}?)\\\)/,
  bracket: /^\\\[((?:\\[^\]]|[^\\]){1,2000}?)\\\]/,
  // the two alternatives of each group must not overlap, or an unclosed formula backtracks exponentially
  display: /^\$\$((?:\\[\s\S]|[^\\$]){1,4000}?)\$\$/,
  block: /^ {0,3}(?:\$\$((?:[^$]|\$(?!\$)){1,4000}?)\$\$|\\\[([\s\S]{1,4000}?)\\\])[ \t]*(?:\n+|$)/,
};

interface MathToken { type: "mathBlock" | "mathInline"; raw: string; tex: string; display: boolean }

const marked = new Marked({
  gfm: true,
  breaks: false,
  extensions: [
    {
      // <u>…</u>, <sub>…</sub>, <span style="color: red">…</span>: one element with its closing tag, markdown inside.
      // Unclosed, or any other tag, stays text.
      name: "fmt",
      level: "inline",
      start: (src: string) => src.match(new RegExp(`<(?:${FORMAT_TAGS})[\\s>]`, "i"))?.index,
      tokenizer(src: string) {
        const m = new RegExp(`^<(${FORMAT_TAGS})((?:\\s+[a-z-]+\\s*=\\s*(?:"[^"]*"|'[^']*'))*)\\s*>`, "i").exec(src);
        if (!m) return undefined;
        const tag = (m[1] as string).toLowerCase();
        const re = new RegExp(`<(/?)${tag}(?:\\s[^>]*)?>`, "gi");
        re.lastIndex = m[0].length;
        let depth = 1, x: RegExpExecArray | null = null;
        while (depth && (x = re.exec(src))) depth += x[1] ? -1 : 1;
        if (depth || !x) return undefined;
        const end = re.lastIndex;
        const inner = src.slice(m[0].length, x.index);
        const colors = tag === "span" ? spanColors(m[2] ?? "") : tag === "mark" ? { bg: HIGHLIGHT_COLORS.yellow as string } : {};
        return { type: "fmt", raw: src.slice(0, end), tag, ...colors, tokens: this.lexer.inlineTokens(inner) } satisfies FormatToken;
      },
    },
    {
      name: "mathBlock",
      level: "block",
      start: (src: string) => src.match(/^ {0,3}(\$\$|\\\[)/m)?.index,
      tokenizer(src: string) {
        const m = RE.block.exec(src);
        const tex = m && (m[1] ?? m[2]);
        if (m && tex?.trim()) return { type: "mathBlock", raw: m[0], tex, display: true } satisfies MathToken;
        return undefined;
      },
    },
    {
      name: "mathInline",
      level: "inline",
      start: (src: string) => src.match(/\$|\\[([]/)?.index,
      tokenizer(src: string) {
        let m: RegExpExecArray | null;
        if (src.startsWith("$$")) {
          if ((m = RE.display.exec(src)) && (m[1] as string).trim()) return { type: "mathInline", raw: m[0], tex: m[1] as string, display: true } satisfies MathToken;
          return { type: "text", raw: "$$", text: "$$" }; // unclosed: text
        }
        if ((m = RE.dollar.exec(src) || RE.paren.exec(src))) return { type: "mathInline", raw: m[0], tex: m[1] as string, display: false } satisfies MathToken;
        if ((m = RE.bracket.exec(src))) return { type: "mathInline", raw: m[0], tex: m[1] as string, display: true } satisfies MathToken;
        return undefined;
      },
    },
  ],
});

const plainParagraph = (src: string): Token => ({ type: "paragraph", raw: src, text: src, tokens: [{ type: "text", raw: src, text: src }] }) as Tokens.Paragraph;

/** Pieces of at most CHUNK characters that concatenate back to `src`, cut at blank lines where possible. */
function chunks(src: string): string[] {
  if (src.length <= CHUNK) return [src];
  const out: string[] = [];
  let pos = 0;
  while (pos < src.length) {
    let end = Math.min(pos + CHUNK, src.length);
    if (end < src.length) {
      const win = src.slice(pos, end);
      let cut = win.lastIndexOf("\n\n");
      if (cut > 0) { cut += 2; while (win[cut] === "\n") cut++; }
      else if ((cut = win.lastIndexOf("\n")) > 0) cut += 1;
      else if ((cut = win.lastIndexOf(" ")) > 0) cut += 1;
      end = pos + (cut > 0 ? cut : win.length);
    }
    out.push(src.slice(pos, end));
    pos = end;
  }
  return out;
}

/**
 * Block tokens of `src`. marked's emphasis matching is quadratic on hostile input (a few thousand
 * unmatched delimiters take seconds), so text is lexed in pieces of CHUNK characters under a time
 * budget; what does not fit is shown as plain paragraphs. Normal replies are one piece. A throw from
 * marked also becomes a plain paragraph. The tokens' raw text always adds up to `src`.
 */
export function lexBlocks(src: string): Token[] {
  const out: Token[] = [];
  const t0 = Date.now();
  for (const piece of chunks(src)) {
    if (Date.now() - t0 > BUDGET_MS) { out.push(plainParagraph(piece)); continue; }
    try {
      out.push(...marked.lexer(piece));
    } catch {
      out.push(plainParagraph(piece));
    }
  }
  return out;
}

// ───────────────────────────── the block still streaming ─────────────────────────────

/**
 * Where an unclosed formula starts in a block's markdown (`$$`, `\[`, `\(`, or a `$` that hugs a non-digit), or -1.
 * Code spans and escapes are skipped; a `$` before a digit is money, not math, and a blank line ends an inline `$`.
 */
export function openMathAt(s: string): number {
  let open = -1, close = "";
  for (let i = 0; i < s.length; i++) {
    const c = s[i], nx = s[i + 1] ?? "";
    if (open < 0 && c === "`") { // a code span: its text is code, whatever it holds
      const run = /^`+/.exec(s.slice(i))?.[0] ?? "`";
      const end = s.indexOf(run, i + run.length);
      if (end < 0) return -1;
      i = end + run.length - 1;
    } else if (c === "\\") {
      if (open < 0 && (nx === "[" || nx === "(")) { open = i; close = nx === "[" ? "]" : ")"; }
      else if (open >= 0 && close === nx) open = -1;
      i++;
    } else if (c === "$") {
      const dbl = nx === "$";
      if (open < 0) { if (dbl || !/[\s\d]/.test(nx || " ")) { open = i; close = dbl ? "$$" : "$"; } }
      else if (close === "$$" ? dbl : close === "$" && !dbl && !/\s/.test(s[i - 1] ?? " ") && !/\d/.test(nx)) open = -1;
      if (dbl) i++;
    } else if (close === "$" && open >= 0 && c === "\n" && /^\n[ \t]*\n/.test(s.slice(i, i + 40))) open = -1;
  }
  return open;
}

/** Where a formatting tag (<u>, <sup>, <span style=…>) opens without its closing tag yet, or -1. */
function openTagAt(s: string): number {
  const stack: { tag: string; at: number }[] = [];
  for (const m of s.matchAll(new RegExp(`<(/?)(${FORMAT_TAGS})(?:\\s[^>]*)?>`, "gi"))) {
    const tag = (m[2] as string).toLowerCase();
    if (!m[1]) stack.push({ tag, at: m.index });
    else { const i = stack.map((x) => x.tag).lastIndexOf(tag); if (i >= 0) stack.splice(i); }
  }
  return stack[0]?.at ?? -1;
}

const closedFence = (raw: string) => /\n {0,3}(?:`{3,}|~{3,})[ \t]*\s*$/.test(raw);
const HAS_INLINE = new Set(["paragraph", "heading", "list", "blockquote", "table", "text"]);

/**
 * The block still streaming in, as it may be shown now: never a half-written formula (the block stops before it
 * until it closes), never a ```math fence before its end, never a table's header row as a line of pipes before
 * its delimiter row arrives. [] means nothing to show yet.
 */
export function settledBlock(t: Token): Token[] {
  if (t.type === "code") return (t as Tokens.Code).lang?.trim().toLowerCase() === "math" && !closedFence(t.raw) ? [] : [t];
  if (!HAS_INLINE.has(t.type)) return [t];
  if (t.type === "paragraph" && /^ {0,3}\|/.test(t.raw)) return [];
  const cuts = [openMathAt(t.raw), openTagAt(t.raw)].filter((c) => c >= 0);
  if (!cuts.length) return [t];
  const cut = Math.min(...cuts);
  const head = t.raw.slice(0, cut);
  return head.trim() ? lexBlocks(head).filter((x) => x.type !== "space") : [];
}

// ───────────────────────────── tokens to nodes ─────────────────────────────

const el = (tag: string, kids?: MdNode[], attrs?: Record<string, string>): MdEl => {
  const e: MdEl = { tag };
  if (attrs) e.attrs = attrs;
  if (kids) e.kids = kids;
  return e;
};

const plain = (nodes: MdNode[]): string => nodes.map((n) => (typeof n === "string" ? n : plain(n.kids ?? []))).join("");

/** Punctuation right after a citation chip stays glued to it, so a lone "." never starts a line. */
function glueCites(nodes: MdNode[]): MdNode[] {
  const out: MdNode[] = [];
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i] as MdNode;
    const next = nodes[i + 1];
    const m = typeof n !== "string" && n.tag === "cite" && typeof next === "string" ? /^[.,;:!?)\]]+/.exec(next) : null;
    if (m && typeof next === "string") {
      out.push(el("span", [n, m[0]], { class: "md-nobr" }));
      if (next.length > m[0].length) out.push(next.slice(m[0].length));
      i++;
    } else out.push(n);
  }
  return out;
}

function inline(tokens: Token[] | undefined, depth: number): MdNode[] {
  return glueCites(inlineRaw(tokens, depth));
}

function inlineRaw(tokens: Token[] | undefined, depth: number): MdNode[] {
  const out: MdNode[] = [];
  for (const t of tokens ?? []) {
    if (depth > MAX_DEPTH) { out.push(t.raw); continue; }
    switch (t.type) {
      case "text": {
        const tt = t as Tokens.Text;
        if (tt.tokens?.length) out.push(...inline(tt.tokens, depth + 1));
        else out.push(decodeEntities(tt.text));
        break;
      }
      case "escape": out.push((t as Tokens.Escape).text); break;
      case "strong": out.push(el("strong", inline((t as Tokens.Strong).tokens, depth + 1))); break;
      case "em": out.push(el("em", inline((t as Tokens.Em).tokens, depth + 1))); break;
      case "del": out.push(el("del", inline((t as Tokens.Del).tokens, depth + 1))); break;
      case "codespan": out.push(el("code", [(t as Tokens.Codespan).text])); break;
      case "br": out.push(el("br")); break;
      case "checkbox": break; // handled by the list item
      case "html": out.push(t.raw); break; // raw HTML is text
      case "fmt": {
        const f = t as unknown as FormatToken;
        const kids = inline(f.tokens, depth + 1);
        const tag = f.tag === "s" || f.tag === "strike" ? "del" : f.tag;
        if (tag === "u" || tag === "del" || tag === "sub" || tag === "sup") out.push(el(tag, kids));
        else if (f.color || f.bg) out.push(el("span", kids, { ...(f.color ? { color: f.color } : {}), ...(f.bg ? { bg: f.bg } : {}) }));
        else out.push(...kids); // a span without a colour we allow: its content
        break;
      }
      case "mathInline": {
        const m = t as unknown as MathToken;
        out.push(el("math", [m.tex.trim()], { display: m.display ? "1" : "0" }));
        break;
      }
      case "link": {
        const l = t as Tokens.Link;
        const kids = inline(l.tokens, depth + 1);
        const href = safeHref(l.href);
        if (!href) { out.push(...kids); break; }
        const attrs: Record<string, string> = { href };
        if (l.title) attrs.title = l.title;
        if (/^zotero:/i.test(href)) out.push(el("cite", [plain(kids) || href], attrs));
        else out.push(el("a", kids, attrs));
        break;
      }
      case "image": {
        const i = t as Tokens.Image;
        const src = safeImageSrc(i.href);
        if (src) out.push(el("img", undefined, { src, alt: i.text || "" }));
        else out.push(i.text || "");
        break;
      }
      default:
        out.push(decodeEntities((t as { text?: string }).text ?? t.raw));
    }
  }
  return out;
}

function blocks(tokens: Token[] | undefined, depth: number): MdNode[] {
  const out: MdNode[] = [];
  for (const t of tokens ?? []) out.push(...blockNodes(t, depth));
  return out;
}

/** One block token as nodes (marked's `space` and `def` produce none). */
export function blockNodes(t: Token, depth = 0): MdNode[] {
  if (depth > MAX_DEPTH) return [el("p", [t.raw])];
  switch (t.type) {
    case "space":
    case "def":
      return [];
    case "paragraph":
      return [el("p", inline((t as Tokens.Paragraph).tokens, depth + 1))];
    case "text": { // a tight list item's text
      const tt = t as Tokens.Text;
      return tt.tokens?.length ? [el("p", inline(tt.tokens, depth + 1))] : [el("p", [decodeEntities(tt.text)])];
    }
    case "heading": {
      const hh = t as Tokens.Heading;
      return [el(`h${Math.min(6, Math.max(1, hh.depth))}`, inline(hh.tokens, depth + 1))];
    }
    case "hr": return [el("hr")];
    case "blockquote": return [el("blockquote", blocks((t as Tokens.Blockquote).tokens, depth + 1))];
    case "code": {
      const c = t as Tokens.Code;
      const lang = (c.lang ?? "").trim().split(/\s+/)[0] ?? "";
      if (lang.toLowerCase() === "math") return [el("math", [c.text.trim()], { display: "1" })];
      // A ```svg block is a diagram (diagram.ts). `open`: its closing fence has not arrived yet (still streaming).
      if (lang.toLowerCase() === "svg") return [el("diagram", [c.text], /\n {0,3}(?:`{3,}|~{3,})[ \t]*\s*$/.test(c.raw) ? undefined : { open: "1" })];
      return [el("codeblock", [c.text], /^[\w+#.-]{1,24}$/.test(lang) ? { lang } : {})];
    }
    case "mathBlock": return [el("math", [(t as unknown as MathToken).tex.trim()], { display: "1" })];
    case "html": return [el("p", [(t as Tokens.HTML).text.trim()], { class: "md-raw" })];
    case "list": {
      const l = t as Tokens.List;
      const attrs: Record<string, string> = {};
      if (l.ordered && typeof l.start === "number" && l.start !== 1 && l.start >= 0 && l.start < 1e6) attrs.start = String(l.start);
      const items = l.items.map((it) => {
        const kids = blocks(it.tokens.filter((x) => x.type !== "checkbox"), depth + 1);
        // A tight item's paragraph becomes inline content: <li>text</li>, not <li><p>text</p></li>.
        const flat: MdNode[] = [];
        for (const k of kids) {
          if (!l.loose && typeof k !== "string" && k.tag === "p" && !k.attrs) flat.push(...(k.kids ?? []));
          else flat.push(k);
        }
        if (it.task) flat.unshift(el("span", [it.checked ? "☑" : "☐"], { class: it.checked ? "md-task md-task--on" : "md-task" }));
        return el("li", flat);
      });
      return [el(l.ordered ? "ol" : "ul", items, attrs)];
    }
    case "table": {
      const tb = t as Tokens.Table;
      const cell = (c: Tokens.TableCell, tag: string, i: number) => {
        const a = tb.align[i];
        return el(tag, inline(c.tokens, depth + 1), a === "left" || a === "center" || a === "right" ? { align: a } : undefined);
      };
      return [el("tablewrap", [el("table", [
        el("thead", [el("tr", tb.header.map((c, i) => cell(c, "th", i)))]),
        el("tbody", tb.rows.map((row) => el("tr", row.map((c, i) => cell(c, "td", i))))),
      ])])];
    }
    default:
      return t.raw.trim() ? [el("p", [decodeEntities((t as { text?: string }).text ?? t.raw)])] : [];
  }
}

/** The whole document as a tree (tests, the sources fold). The view lexes incrementally instead. */
export function mdToTree(src: string): MdNode[] {
  if (src.length > MAX_MD) return [...mdToTree(src.slice(0, MAX_MD)), el("pre", [src.slice(MAX_MD)])];
  return blocks(lexBlocks(src.replace(/\r\n?/g, "\n")), 0);
}

/** Every element node, depth first. */
export function* walk(nodes: MdNode[]): Generator<MdEl> {
  for (const n of nodes) {
    if (typeof n === "string") continue;
    yield n;
    if (n.kids) yield* walk(n.kids);
  }
}

// ───────────────────────────── citations ─────────────────────────────

export interface CitedSource {
  key: string; // "libraryID/ITEMKEY"
  libraryID: number;
  itemKey: string;
  /** The first chip's text: "Pager et al. 2009". */
  label: string;
  /** The first link, which a click opens. */
  href: string;
  pages: number[];
}

/** Each cited item once, in order of first citation, with the pages cited. */
export function collectSources(texts: string[]): CitedSource[] {
  const map = new Map<string, CitedSource>();
  for (const text of texts) {
    for (const n of walk(mdToTree(text))) {
      if (n.tag !== "cite") continue;
      const href = n.attrs?.href ?? "";
      const z = parseZoteroUri(href);
      if (!z) continue;
      const key = `${z.libraryID}/${z.itemKey}`;
      let s = map.get(key);
      if (!s) {
        s = { key, libraryID: z.libraryID, itemKey: z.itemKey, label: citeLabel(plain(n.kids ?? [])), href: withoutQuote(href), pages: [] };
        map.set(key, s);
      }
      if (z.page && !s.pages.includes(z.page)) s.pages.push(z.page);
    }
  }
  return [...map.values()];
}

/** "Pager et al. 2009, p.8" shows as "Pager et al. 2009" in the fold: the pages are listed beside it. */
export function citeLabel(text: string): string {
  return text.replace(/,?\s*(?:pp?\.?|pages?)\s*\d[\d–—,\s-]*$/i, "").trim() || text;
}
