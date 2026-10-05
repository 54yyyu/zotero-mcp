// Diagrams, the DOM half: a ```svg block as a themed figure card with a quiet toolbar (Source, Copy as PNG,
// Save as PNG, Save as SVG, Add to a note). mdview.ts imports this module dynamically on the first ```svg block, so a chat
// without diagrams never runs it; its stylesheet is added to the panel's shadow root on first use too.
// Nodes come from diagram-svg.ts's sanitized tree, built with createElementNS: no markup is ever parsed.
import type { NoteImage, NoteRequest, SavedNote } from "../types.ts";
import type { SvgEl } from "./diagram-svg.ts";
import { HUES, HUE_DARK, HUE_LIGHT, SOFT_DARK, SOFT_LIGHT, exportPalette, prepareSvg, serializeSvg } from "./diagram-svg.ts";
import { copyText, edgeFade, env, h, icon } from "./dom.ts";
import type { IconName } from "./dom.ts";
import { noteLine, trySave } from "./notes.ts";

export interface DiagramOpts {
  /** The host's file picker (PanelHost.saveFile); absent, the Save buttons are not shown. */
  saveFile?: (name: string, data: Uint8Array | string, mime: string) => Promise<string | null>;
  /** PanelHost.saveNote, and what opens the saved note; absent, "Add to a note" is not shown. */
  saveNote?: (note: NoteRequest) => Promise<SavedNote>;
  open?: (uri: string) => void;
  /** The panel's code block, for Source. */
  codeBlock(code: string, lang: string): HTMLElement;
}

// ───────────────────────────── styles (once per shadow root) ─────────────────────────────

const soft = (k: number) => HUES.map((x) => `--dg-${x}-soft: color-mix(in srgb, var(--dg-${x}) ${Math.round(k * 100)}%, transparent);`).join(" ");
const hues = (m: Record<string, string>) => HUES.map((x) => `--dg-${x}: ${m[x]};`).join(" ");
/** Below this scale a drawing scrolls sideways instead of shrinking: 12px labels stay about 11px. */
const MIN_SCALE = 0.9;

const CSS = `
.zmc .dg {
  --dg-ink: var(--ink); --dg-muted: var(--ink-muted); --dg-line: color-mix(in srgb, var(--ink) 30%, transparent);
  --dg-surface: color-mix(in srgb, var(--ink) 5%, transparent);
  ${hues(HUE_LIGHT)} ${soft(SOFT_LIGHT)}
  position: relative; min-width: 0; padding: var(--s1) var(--s2) var(--s3); border: 1px solid var(--rule); border-radius: var(--r2); background: var(--paper-raised);
}
.zmc[data-theme="dark"] .dg { ${hues(HUE_DARK)} ${soft(SOFT_DARK)} --dg-surface: color-mix(in srgb, var(--ink) 7%, transparent); background: color-mix(in srgb, var(--ink) 3%, var(--paper)); }
/* the user's own accent only when they chose one (appearance sets data-accent="custom"); else the calm blue above */
.zmc[data-accent="custom"] .dg, .zmc[data-accent="custom"][data-theme="dark"] .dg { --dg-accent: var(--accent); }
.zmc .dg__head { display: flex; align-items: center; gap: var(--s2); height: 28px; margin-bottom: var(--s1); }
.zmc .dg__cap { flex: 1; min-width: 0; padding-left: var(--s1); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--ink-faint); font-size: var(--fs-1); }
.zmc .dg__fig { overflow-x: auto; overflow-y: hidden; scrollbar-width: thin; scrollbar-color: var(--rule-strong) transparent; }
.zmc .dg__fig svg { display: block; width: 100%; height: auto; max-height: 560px; margin: 0 auto; overflow: visible; fill: var(--dg-ink); font-family: var(--font); }
.zmc .dg__fig :is(text, tspan) { font-family: var(--font); }
.zmc .dg__bar { flex: none; display: flex; align-items: center; gap: 1px; opacity: 0; transition: opacity var(--ease); }
.zmc .dg:hover .dg__bar, .zmc .dg__bar:focus-within, .zmc .dg--src .dg__bar { opacity: 1; }
@media (hover: none) { .zmc .dg__bar { opacity: 1; } }
@container zmc (max-width: 380px) { .zmc .dg__bar { opacity: 1; margin-left: auto; } .zmc .dg__cap { display: none; } } /* narrow: the strip is the toolbar */
.zmc .dg__bar .iconbtn { width: 24px; height: 24px; border-radius: var(--r0); }
.zmc .dg__bar .iconbtn svg { width: 14px; height: 14px; }
.zmc .dg__btn { display: inline-flex; align-items: center; gap: 3px; height: 24px; padding: 0 6px 0 4px; border: 0; border-radius: var(--r0); background: none; color: var(--ink-muted); font: 500 var(--fs-1)/1 var(--font); letter-spacing: 0.02em; }
.zmc .dg__btn svg { width: 12px; height: 12px; }
.zmc .dg__btn:hover { color: var(--ink); background: var(--tint-hover); }
.zmc .dg__sep { width: 1px; height: 14px; margin: 0 2px; background: var(--rule); }
.zmc .dg__src { margin-top: var(--s3); }
.zmc .dg--pending { display: flex; align-items: center; justify-content: center; gap: var(--s2); min-height: 96px; border-style: dashed; background: none; color: var(--ink-faint); font-size: var(--fs-2); }
.zmc .dg--pending::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; animation: zmc-pulse 1.4s ease-in-out infinite; }
`;

const styled = new WeakSet<Node>();
/** Adds the diagram stylesheet to the shadow root `el` is in (once). */
export function ensureStyles(el: Node): void {
  const root = el.getRootNode?.();
  if (!root || root.nodeType !== 11 || styled.has(root)) return; // 11: a shadow root (not yet attached: next time)
  styled.add(root);
  (root as ShadowRoot).appendChild(h("style", { text: CSS }));
}

// ───────────────────────────── tree to DOM ─────────────────────────────

const SVG_NS = "http://www.w3.org/2000/svg";
const PAINT = /^var\(--dg-[a-z]+(?:-soft)?\)$/;

function build(n: SvgEl): SVGElement {
  const out = env.doc.createElementNS(SVG_NS, n.tag) as SVGElement;
  for (const [k, v] of Object.entries(n.attrs)) {
    if (PAINT.test(v)) out.style.setProperty(k, v); // a CSS variable only works as a property, not as an attribute
    else out.setAttribute(k, v);
  }
  for (const k of n.kids) out.appendChild(typeof k === "string" ? env.doc.createTextNode(k) : build(k));
  return out;
}

const cache = new Map<string, ReturnType<typeof prepareSvg>>();
function prepared(src: string): ReturnType<typeof prepareSvg> {
  if (cache.has(src)) return cache.get(src) ?? null;
  const p = prepareSvg(src);
  if (cache.size >= 40) cache.delete(cache.keys().next().value as string);
  cache.set(src, p);
  return p;
}

/**
 * Fills `card` (an empty div) with the drawing and its toolbar. False when the text is not a drawing we
 * can show; the caller then shows it as code.
 */
export function fill(card: HTMLElement, src: string, opts: DiagramOpts): boolean {
  const p = prepared(src);
  if (!p) return false;
  const { svg: tree, fit, title } = p;
  const fig = build(tree);
  fig.setAttribute("role", "img");
  fig.setAttribute("aria-label", tree.attrs["aria-label"] || title || "Diagram");
  // Between 0.9x and 1.3x its own size: labels never billboard-size in a wide panel, never below ~11px in a narrow one
  // (there it scrolls sideways, with a soft fade on the side that has more).
  fig.style.maxWidth = `${Math.round(fit.box[2] * 1.3)}px`;
  fig.style.minWidth = `${Math.round(fit.box[2] * MIN_SCALE)}px`;
  const figWrap = h("div.dg__fig", null, fig);
  const edges = () => edgeFade(figWrap);
  figWrap.addEventListener("scroll", edges, { passive: true });
  const RO = env.win.ResizeObserver;
  if (RO) new RO(edges).observe(figWrap);

  let srcEl: HTMLElement | null = null;
  const btn = (label: string, ic: IconName, run: (b: HTMLElement) => unknown) => {
    const b = h("button.iconbtn", { type: "button", "aria-label": label, title: label }, icon(ic));
    b.addEventListener("click", () => void run(b));
    return b;
  };
  const sourceBtn = btn("Show the SVG source", "code", (b) => {
    const on = !srcEl;
    if (on) { srcEl = h("div.dg__src", null, opts.codeBlock(src, "svg")); card.appendChild(srcEl); }
    else { srcEl?.remove(); srcEl = null; }
    b.setAttribute("aria-pressed", String(on));
    card.classList.toggle("dg--src", on);
  });
  sourceBtn.setAttribute("aria-pressed", "false");
  const name = fileName(title);
  // Files are light whatever the panel's theme, with the same accent rule: the blue, or the user's own (light) accent.
  const accent = () => {
    const app = card.closest(".zmc");
    if (app?.getAttribute("data-accent") !== "custom") return undefined;
    const cs = env.win.getComputedStyle(card);
    return cs.getPropertyValue("--accent-l").trim() || (app.getAttribute("data-theme") === "dark" ? undefined : cs.getPropertyValue("--accent").trim()) || undefined;
  };
  const exported = () => serializeSvg(tree, fit, exportPalette(accent()));
  const copyBtn = btn("Copy as an image", "copy", async (b) => done(b, await copyPng(exported(), fit.box[2], fit.box[3])));
  const saveBtn = (fmt: "PNG" | "SVG") => {
    const b = h("button.dg__btn", { type: "button", title: `Save as ${fmt}`, "aria-label": `Save as ${fmt}` }, icon("download"), fmt);
    b.addEventListener("click", () => void (async () => {
      if (!opts.saveFile) return;
      try {
        const data = fmt === "SVG" ? `<?xml version="1.0" encoding="UTF-8"?>\n${exported()}\n` : await toPng(exported(), fit.box[2], fit.box[3]);
        const path = await opts.saveFile(`${name}.${fmt.toLowerCase()}`, data, fmt === "SVG" ? "image/svg+xml" : "image/png");
        if (path) done(b, true);
      } catch { done(b, false); }
    })());
    return b;
  };
  // A note under the item the user is on: the title, then the drawing as an embedded PNG; the line under the card says where it went.
  let noteEl: HTMLElement | null = null;
  const noteBtn = (save: NonNullable<DiagramOpts["saveNote"]>) => btn("Add to a note", "note", async (b) => {
    b.setAttribute("disabled", "");
    const st = await trySave(async () => save({ ...(title ? { title } : {}), markdown: fence(src), images: [await pngOf(src, accent())] }));
    b.removeAttribute("disabled");
    if ("ok" in st) done(b, true);
    noteEl?.remove();
    noteEl = noteLine(st, (uri) => opts.open?.(uri));
    card.insertBefore(noteEl, srcEl);
  });
  const bar = h("div.dg__bar", { role: "toolbar", "aria-label": "Diagram" }, sourceBtn, copyBtn,
    opts.saveFile ? [h("span.dg__sep"), saveBtn("PNG"), saveBtn("SVG")] : null,
    opts.saveNote ? [h("span.dg__sep"), noteBtn(opts.saveNote)] : null);
  card.className = "dg";
  card.replaceChildren(h("div.dg__head", null, h("span.dg__cap", { "aria-hidden": "true" }, title), bar), figWrap);
  ensureStyles(card);
  env.win.queueMicrotask?.(() => ensureStyles(card)); // the card is in the shadow root by now
  return true;
}

/** A quiet "Drawing…" placeholder, never a half-drawn diagram. */
export function pending(card: HTMLElement): void {
  card.className = "dg dg--pending";
  card.setAttribute("role", "status");
  card.replaceChildren("Drawing…");
  ensureStyles(card);
  env.win.queueMicrotask?.(() => ensureStyles(card));
}

/** The source back in a fence longer than any run of backticks inside it. */
const fence = (src: string) => {
  const t = "`".repeat(Math.max(3, ...[...src.matchAll(/`+/g)].map((m) => m[0].length + 1)));
  return `${t}svg\n${src}\n${t}`;
};

/** The drawing as a PNG for a note (light palette, on white), shown at its own width up to 640 px. Null when it is no drawing. */
export async function pngOf(src: string, accent?: string): Promise<NoteImage | null> {
  const p = prepared(src);
  if (!p) return null;
  const [, , w, hh] = p.fit.box;
  const width = Math.min(640, Math.round(w));
  return { data: await toPng(serializeSvg(p.svg, p.fit, exportPalette(accent)), w, hh), width, height: Math.round((hh * width) / w) };
}

const fileName = (title: string) => title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 48) || "diagram";

/** A check mark for a moment on success; the button's own content again after. */
function done(b: HTMLElement, ok: boolean): void {
  if (!ok) { b.title = "That did not work"; return; }
  const kids = [...b.childNodes];
  b.replaceChildren(icon("check"));
  env.win.setTimeout(() => b.replaceChildren(...kids), 1200);
}

// ───────────────────────────── PNG ─────────────────────────────

/** The exported SVG drawn on white, at least 1400 px wide (2x at least). */
async function pngBlob(svgText: string, w: number, hh: number): Promise<Blob> {
  const img = new env.win.Image();
  await new Promise<void>((res, rej) => {
    img.onload = () => res();
    img.onerror = () => rej(new Error("the drawing could not be drawn"));
    img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgText)}`;
  });
  const scale = Math.max(2, 1400 / w);
  const canvas = env.doc.createElementNS("http://www.w3.org/1999/xhtml", "canvas") as HTMLCanvasElement;
  canvas.width = Math.round(w * scale);
  canvas.height = Math.round(hh * scale);
  const g = canvas.getContext("2d");
  if (!g) throw new Error("no canvas");
  g.fillStyle = "#ffffff";
  g.fillRect(0, 0, canvas.width, canvas.height);
  g.drawImage(img, 0, 0, canvas.width, canvas.height);
  return new Promise((res, rej) => canvas.toBlob((b) => (b ? res(b) : rej(new Error("no PNG"))), "image/png"));
}

export async function toPng(svgText: string, w: number, hh: number): Promise<Uint8Array> {
  return new Uint8Array(await (await pngBlob(svgText, w, hh)).arrayBuffer());
}

/** The PNG on the clipboard; the SVG text if the clipboard takes no images. */
async function copyPng(svgText: string, w: number, hh: number): Promise<boolean> {
  try {
    const CI = (env.win as unknown as { ClipboardItem?: typeof ClipboardItem }).ClipboardItem;
    if (!CI) throw new Error("no ClipboardItem");
    await env.win.navigator.clipboard.write([new CI({ "image/png": pngBlob(svgText, w, hh) })]);
    return true;
  } catch {
    return copyText(svgText);
  }
}
