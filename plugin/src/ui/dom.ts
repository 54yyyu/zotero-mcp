// DOM helpers for the panel. Nothing here touches Zotero or node. The document and window come from the
// mount point (`initEnv`), not from globals: a plugin script in a Zotero window may not have `document`
// or `setTimeout` as free variables.

const XHTML = "http://www.w3.org/1999/xhtml";
const SVG = "http://www.w3.org/2000/svg";

export const env: { doc: Document; win: Window & typeof globalThis } = {
  doc: (globalThis as { document?: Document }).document as Document,
  win: (globalThis as { window?: Window }).window as Window & typeof globalThis,
};

export function initEnv(root: ShadowRoot): void {
  env.doc = root.ownerDocument;
  env.win = (root.ownerDocument.defaultView ?? env.win) as Window & typeof globalThis;
}

export type Kid = Node | string | number | null | undefined | false | Kid[];
type Props = Record<string, unknown>;

const isNode = (k: unknown): k is Node => typeof k === "object" && k !== null && "nodeType" in k;

/**
 * h("div.card.card--x", { onclick, dataset, ...attrs }, ...children)
 * No `html` prop on purpose: the panel never parses markup, it builds nodes. Event handlers only
 * as functions (an `onclick="..."` string would be script).
 */
export function h(spec: string, props?: Props | null, ...kids: Kid[]): HTMLElement {
  const [name, ...classes] = spec.split(".");
  const el = env.doc.createElementNS(XHTML, name || "div") as HTMLElement;
  if (classes.length) el.className = classes.join(" ");
  if (props) {
    for (const [k, v] of Object.entries(props)) {
      if (v === undefined || v === null || v === false) continue;
      if (k.startsWith("on")) {
        if (typeof v === "function") el.addEventListener(k.slice(2), v as EventListener);
      } else if (k === "dataset") Object.assign(el.dataset, v as Record<string, string>);
      else if (k === "text") el.textContent = String(v);
      else if (k === "value") (el as HTMLInputElement).value = String(v);
      else if (k === "checked") (el as HTMLInputElement).checked = true;
      else if (v === true) el.setAttribute(k, "");
      else el.setAttribute(k, String(v));
    }
  }
  append(el, kids);
  return el;
}

export function append(el: Node, kids: Kid[]): void {
  for (const k of kids) {
    if (k === null || k === undefined || k === false) continue;
    if (Array.isArray(k)) append(el, k);
    else if (isNode(k)) el.appendChild(k);
    else el.appendChild(env.doc.createTextNode(String(k)));
  }
}

export function clear(el: Node): void {
  while (el.firstChild) el.removeChild(el.firstChild);
}

export function setKids(el: Node, ...kids: Kid[]): void {
  clear(el);
  append(el, kids);
}

// ───────────────────────────── icons ─────────────────────────────
// Parley's set: a 16-unit box, 1.5 stroke, round caps, currentColor. A string is a path; an array is
// [tag, attrs] for the odd circle or rect. Built with createElementNS, never parsed.

type Shape = string | [string, Record<string, string>];
const c = (cx: number, cy: number, r: number, fill = false): Shape =>
  ["circle", { cx: String(cx), cy: String(cy), r: String(r), ...(fill ? { fill: "currentColor", stroke: "none" } : {}) }];
const r = (x: number, y: number, w: number, hh: number, rx: number, fill = false): Shape =>
  ["rect", { x: String(x), y: String(y), width: String(w), height: String(hh), rx: String(rx), ...(fill ? { fill: "currentColor", stroke: "none" } : {}) }];

const ICONS = {
  close: ["m4 4 8 8M12 4l-8 8"],
  plus: ["M8 3.2v9.6M3.2 8h9.6"],
  history: [c(8, 8, 5.6), "M8 4.8V8l2.2 1.6"],
  gear: [c(8, 8, 2.1), "M8 1.6l.9 1.6 1.8-.4.5 1.8 1.8.5-.4 1.8 1.2 1.4-1.2 1.4.4 1.8-1.8.5-.5 1.8-1.8-.4L8 14.4l-.9-1.6-1.8.4-.5-1.8-1.8-.5.4-1.8L2.2 7.7l1.2-1.4-.4-1.8 1.8-.5.5-1.8 1.8.4Z"],
  send: ["M8 13V3M3.5 7.5 8 3l4.5 4.5"],
  stop: [r(4.2, 4.2, 7.6, 7.6, 1.6, true)],
  chev: ["M6 3.2 10.8 8 6 12.8"],
  chevDown: ["M3.2 6 8 10.8 12.8 6"],
  back: ["M10 3.2 5.2 8 10 12.8"],
  copy: [r(5.4, 5.4, 8, 8, 1.6), "M10.6 5.4V4a1.4 1.4 0 0 0-1.4-1.4H4a1.4 1.4 0 0 0-1.4 1.4v5.2A1.4 1.4 0 0 0 4 10.6h1.4"],
  check: ["m3.5 8.5 3 3 6-7"],
  retry: ["M13.2 8a5.2 5.2 0 1 1-1.6-3.8", "M13.4 2.6v3h-3"],
  bookmark: ["M4.6 2.4h6.8v11.2L8 10.9l-3.4 2.7Z"],
  search: [c(7.1, 7.1, 4.5), "M10.4 10.4 13.6 13.6"],
  item: ["M3.4 2.4h5.4l3.8 3.8v7.4H3.4Z", "M8.6 2.4v3.9h3.8", "M5.8 9h4.4M5.8 11.2h3"],
  file: ["M3.4 2.4h5.4l3.8 3.8v7.4H3.4Z", "M8.6 2.4v3.9h3.8"],
  folder: ["M2 5.2a1.2 1.2 0 0 1 1.2-1.2h2.7l1.3 1.5h4.6A1.2 1.2 0 0 1 14 6.7v5.1a1.2 1.2 0 0 1-1.2 1.2H3.2A1.2 1.2 0 0 1 2 11.8Z"],
  highlight: ["M3 13.4h10", "M5.2 10.6 10.6 5.2l2.2 2.2-5.4 5.4H5.2Z"],
  selection: ["M2.6 3.8h10.8M2.6 8h10.8M2.6 12.2h6.4"],
  area: ["M4.6 1.8v9.6h9.6", "M1.8 4.6h9.6v9.6"],
  warn: ["M8 2.2 14.2 13H1.8z", "M8 6.5v3M8 11.3v.01"],
  info: [c(8, 8, 6), "M8 7.3v3.9", "M8 4.9v.5"],
  down: ["M8 3v10M3.8 8.8 8 13l4.2-4.2"],
  terminal: ["m2.6 4.6 3.4 3.4-3.4 3.4M7.6 12h5.8"],
  globe: [c(8, 8, 5.6), "M2.4 8h11.2", "M8 2.4c1.5 1.7 2.3 3.6 2.3 5.6S9.5 11.9 8 13.6C6.5 11.9 5.7 10 5.7 8S6.5 4.1 8 2.4Z"],
  pencil: ["M10.4 2.2 13.8 5.6 5.6 13.8 1.8 14.2 2.2 10.4Z", "M8.9 3.7 12.3 7.1"],
  trash: ["M2.6 4.2h10.8M6.1 4.2V2.6h3.8v1.6", "M3.9 4.2l.6 9h7l.6-9", "M6.6 6.6v4.2M9.4 6.6v4.2"],
  move: ["M2 8h12M10.4 4.4 14 8l-3.6 3.6"],
  sparkle: ["M6.4 2.2 7.5 5.3 10.6 6.4 7.5 7.5 6.4 10.6 5.3 7.5 2.2 6.4 5.3 5.3Z", "M11.6 9.2l.6 1.6 1.6.6-1.6.6-.6 1.6-.6-1.6-1.6-.6 1.6-.6Z"],
  dot: [c(8, 8, 2, true)],
  external: ["M9 2.6h4.4V7M13.4 2.6 7.4 8.6", "M12 9.6v2.8a1 1 0 0 1-1 1H3.6a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h2.8"],
  shield: ["M8 1.8 13 3.6v4.1c0 3-2.1 5.2-5 6.5-2.9-1.3-5-3.5-5-6.5V3.6Z"],
  code: ["M5.6 4.4 2 8l3.6 3.6M10.4 4.4 14 8l-3.6 3.6"],
  download: ["M8 2.6v7.6M4.8 7.2 8 10.4l3.2-3.2", "M2.8 13.2h10.4"],
  note: ["M3 2.6h10v7.2l-3.6 3.6H3Z", "M13 9.8H9.4v3.6", "M5.4 5.6h5.2M5.4 8h3.2"],
  bulb: ["M8 1.8a4.2 4.2 0 0 0-2.4 7.65c.5.36.8.9.8 1.5v.45h3.2v-.45c0-.6.3-1.14.8-1.5A4.2 4.2 0 0 0 8 1.8Z", "M6.6 13.8h2.8"],
  list: ["M5.6 4h8M5.6 8h8M5.6 12h8", c(2.6, 4, 0.7, true), c(2.6, 8, 0.7, true), c(2.6, 12, 0.7, true)],
} satisfies Record<string, Shape[]>;
export type IconName = keyof typeof ICONS;

/** The icon for a context chip's kind and a search hit's kind. */
export const CHIP_ICON: Record<string, IconName> = {
  item: "item", reader: "item", selection: "selection", area: "area", page: "file", annotation: "highlight", collection: "folder", note: "file",
};

/** An SVG element with attributes (and children). */
export function svg(tag: string, attrs: Record<string, string>, ...kids: Node[]): SVGElement {
  const n = env.doc.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  for (const k of kids) n.appendChild(k);
  return n;
}

export function icon(name: IconName, cls?: string): SVGElement {
  return svg("svg", { viewBox: "0 0 16 16", fill: "none", stroke: "currentColor", "stroke-width": "1.5", "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true", ...(cls ? { class: cls } : {}) },
    ...(ICONS[name] as Shape[]).map((s) => (typeof s === "string" ? svg("path", { d: s }) : svg(s[0], s[1]))));
}

// ───────────────────────────── small things ─────────────────────────────

let uid = 0;
export const nextId = (prefix: string): string => `${prefix}${(++uid).toString(36)}`;

export function isMac(): boolean {
  const nav = env.win?.navigator;
  return /Mac|iPhone|iPad/i.test(nav?.platform || nav?.userAgent || "");
}

/** The shortcut badge for a custom prompt slot. */
export function slotLabel(slot: number): string {
  return isMac() ? `⌘⌃${slot}` : `Ctrl+Alt+${slot}`;
}

export function relTime(ms: number, now = Date.now()): string {
  const s = Math.max(0, Math.round((now - ms) / 1000));
  if (s < 45) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const hr = Math.round(m / 60);
  if (hr < 24) return `${hr} h ago`;
  const d = Math.round(hr / 24);
  if (d < 7) return `${d} d ago`;
  return new Date(ms).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/** Today / Yesterday / This week / "September" / "September 2025": the history's groups. */
export function dayGroup(ms: number, now = new Date()): string {
  const d = new Date(ms);
  const day = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const days = Math.round((day(now) - day(d)) / 864e5);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return "This week";
  return d.toLocaleDateString(undefined, { month: "long", ...(d.getFullYear() === now.getFullYear() ? {} : { year: "numeric" }) });
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await env.win.navigator.clipboard.writeText(text);
    return true;
  } catch { /* fall through to the old way */ }
  try {
    const ta = h("textarea", { "aria-hidden": "true" }) as HTMLTextAreaElement;
    ta.style.cssText = "position:fixed;top:-100px";
    ta.value = text;
    env.doc.body?.appendChild(ta);
    ta.select();
    const ok = env.doc.execCommand("copy");
    ta.remove();
    return ok;
  } catch {
    return false;
  }
}

/** A check mark in an icon button for a moment (Copy done), then the copy icon again. */
export function flashCheck(btn: HTMLElement): void {
  btn.replaceChildren(icon("check"));
  env.win.setTimeout(() => btn.replaceChildren(icon("copy")), 1200);
}

/** A sideways scroller's `data-fade` (l, r, lr): the side that has more content fades (styles-chat.ts draws it). */
export function edgeFade(el: HTMLElement): void {
  const { scrollLeft: l, scrollWidth: sw, clientWidth: cw } = el;
  const f = (l > 1 ? "l" : "") + (l + cw < sw - 1 ? "r" : "");
  if (f) el.dataset.fade = f; else delete el.dataset.fade;
}

/** Truncate for a one-line label; the full text belongs in a tooltip. */
export function clip(s: string, n: number): string {
  const t = s.replace(/\s+/g, " ").trim();
  return t.length > n ? t.slice(0, n - 1) + "…" : t;
}

export function fmtTokens(n: number): string {
  return n >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : n >= 10000 ? `${Math.round(n / 1000)}k` : n >= 1000 ? `${+(n / 1000).toFixed(1)}k` : String(n);
}

/** Claude's bridge titles a Bash call "`zotero-cli search x`": show it without the wrapping backticks. */
export const unwrapTicks = (s: string): string => s.replace(/^`([^`]*)`$/, "$1");

export const errMessage = (e: unknown): string => (e instanceof Error ? e.message : String(e));
