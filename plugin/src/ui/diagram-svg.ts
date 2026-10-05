// Diagrams, the pure half: the agent's ```svg text to a small, safe element tree. No DOM, so it runs under
// `node --test` and hostile input can be checked exactly; diagram.ts builds the nodes with createElementNS.
//
// Policy (the panel runs in a privileged window and the agent reads the web):
//   parse     our own lenient tokenizer: never a DOMParser, never an entity from a DTD (a DOCTYPE is skipped)
//   elements  drawing elements only; script, style, image, foreignObject, a, animate, ... go with everything inside them
//   attrs     geometry and presentation only; no on*, no class, no style (its safe declarations become attributes),
//             href only to "#id", url() only to "#id", ids namespaced per diagram so they cannot reach the panel's own
//   colours   only palette NAMES survive (they become the panel's CSS variables); a literal colour is mapped to the
//             nearest name by lightness and hue, so a diagram always follows the theme and the user's accent
import { decodeEntities } from "./markdown.ts";

export interface SvgEl { tag: string; attrs: Record<string, string>; kids: SvgNode[] }
export type SvgNode = SvgEl | string;

// ───────────────────────────── palette ─────────────────────────────

/** The colour names the agent is told to use (brief.ts DRAWING_GUIDE). `-soft` are area fills. */
export const HUES = ["accent", "teal", "violet", "orange", "red", "green"] as const;
export const NEUTRALS = ["ink", "muted", "line", "surface"] as const;
export const PALETTE: readonly string[] = [...NEUTRALS, ...HUES, ...HUES.map((h) => `${h}-soft`)];
const PALETTE_SET = new Set(PALETTE);

type Hue = (typeof HUES)[number];
/** Light and dark values of the hues; ink, muted, line and surface come from the panel's own tokens. */
export const HUE_LIGHT: Record<Hue, string> = { accent: "#2f6fd6", teal: "#0f8b98", violet: "#6c55d4", orange: "#d2612b", red: "#c9373f", green: "#2c9154" };
export const HUE_DARK: Record<Hue, string> = { accent: "#7aa7ff", teal: "#45c2cc", violet: "#a897f5", orange: "#f39a62", red: "#f27d80", green: "#5dcb8a" };
/** How strong a `-soft` fill is (over the card), light and dark. */
export const SOFT_LIGHT = 0.14;
export const SOFT_DARK = 0.24;

/** The palette of an exported file: light, on white, every value a plain hex so any tool reads it. */
export function exportPalette(accent = HUE_LIGHT.accent): Record<string, string> {
  const hues: Record<string, string> = { ...HUE_LIGHT, accent: /^#[0-9a-f]{6}$/i.test(accent) ? accent.toLowerCase() : HUE_LIGHT.accent };
  const out: Record<string, string> = { ink: "#1b1e24", muted: "#5f646c", line: "#b4bac2", surface: "#f1f2f4" };
  for (const h of HUES) { out[h] = hues[h] as string; out[`${h}-soft`] = mixWhite(hues[h] as string, SOFT_LIGHT); }
  return out;
}

function mixWhite(hex: string, a: number): string {
  const [r, g, b] = rgbOf(hex) as [number, number, number];
  return "#" + [r, g, b].map((c) => Math.round(255 - (255 - c) * a).toString(16).padStart(2, "0")).join("");
}

const NAMED_COLORS: Record<string, string> = {
  black: "#000000", white: "#ffffff", gray: "#808080", grey: "#808080", silver: "#c0c0c0", lightgray: "#d3d3d3", lightgrey: "#d3d3d3",
  darkgray: "#a9a9a9", darkgrey: "#a9a9a9", dimgray: "#696969", whitesmoke: "#f5f5f5", gainsboro: "#dcdcdc", red: "#ff0000",
  crimson: "#dc143c", tomato: "#ff6347", orange: "#ffa500", gold: "#ffd700", yellow: "#ffff00", green: "#008000", lime: "#00ff00",
  seagreen: "#2e8b57", teal: "#008080", cyan: "#00ffff", aqua: "#00ffff", turquoise: "#40e0d0", blue: "#0000ff", navy: "#000080",
  steelblue: "#4682b4", royalblue: "#4169e1", dodgerblue: "#1e90ff", skyblue: "#87ceeb", lightblue: "#add8e6", purple: "#800080",
  violet: "#ee82ee", indigo: "#4b0082", magenta: "#ff00ff", fuchsia: "#ff00ff", pink: "#ffc0cb", brown: "#a52a2a", coral: "#ff7f50",
  salmon: "#fa8072", lightyellow: "#ffffe0", lightgreen: "#90ee90", lavender: "#e6e6fa", beige: "#f5f5dc", ivory: "#fffff0",
};

/** [r, g, b] 0..255 of a hex, rgb(), hsl() or CSS name; null for anything else. */
export function rgbOf(value: string): [number, number, number] | null {
  const v = NAMED_COLORS[value] ?? value;
  let m = /^#([0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})$/i.exec(v);
  if (m) {
    const x = (m[1] as string).length <= 4 ? [...(m[1] as string)].map((c) => c + c).join("") : (m[1] as string);
    return [0, 2, 4].map((i) => parseInt(x.slice(i, i + 2), 16)) as [number, number, number];
  }
  m = /^rgba?\(\s*([\d.]+)(%?)[\s,]+([\d.]+)(%?)[\s,]+([\d.]+)(%?)/i.exec(v);
  if (m) return [1, 3, 5].map((i) => Math.min(255, Number(m![i]) * (m![i + 1] ? 2.55 : 1))) as [number, number, number];
  m = /^hsla?\(\s*([\d.]+)(?:deg)?[\s,]+([\d.]+)%[\s,]+([\d.]+)%/i.exec(v);
  if (m) {
    const hh = (Number(m[1]) % 360) / 360, s = Math.min(1, Number(m[2]) / 100), l = Math.min(1, Number(m[3]) / 100);
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s, p = 2 * l - q;
    const f = (t: number) => { t = (t + 1) % 1; return t < 1 / 6 ? p + (q - p) * 6 * t : t < 1 / 2 ? q : t < 2 / 3 ? p + (q - p) * (2 / 3 - t) * 6 : p; };
    return [f(hh + 1 / 3), f(hh), f(hh - 1 / 3)].map((c) => Math.round(c * 255)) as [number, number, number];
  }
  return null;
}

/**
 * The palette name for a colour value on property `prop`: a name is itself; a literal colour maps by
 * lightness and hue (greys to ink/muted/line/surface, light tints to a `-soft` fill); null when it is
 * not a colour at all.
 */
export function paletteName(value: string, prop: string): string | null {
  const v = value.trim().toLowerCase();
  if (PALETTE_SET.has(v)) return v;
  const rgb = rgbOf(v);
  if (!rgb) return null;
  const [r, g, b] = rgb.map((c) => c / 255) as [number, number, number];
  const max = Math.max(r, g, b), min = Math.min(r, g, b), l = (max + min) / 2, d = max - min;
  const sat = d === 0 ? 0 : d / (1 - Math.abs(2 * l - 1));
  const isFill = prop !== "stroke";
  if (sat < 0.18 || d < 0.06) {
    if (l < 0.3) return "ink";
    if (l > 0.85) return isFill ? "surface" : "line";
    return l > 0.62 ? (isFill && l > 0.75 ? "surface" : "line") : "muted";
  }
  let hue = max === r ? ((g - b) / d + 6) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
  hue *= 60;
  const name = hue < 15 || hue >= 335 ? "red" : hue < 70 ? "orange" : hue < 165 ? "green" : hue < 195 ? "teal" : hue < 255 ? "accent" : "violet";
  return l > 0.78 && isFill ? `${name}-soft` : name;
}

// ───────────────────────────── parse ─────────────────────────────

/** Beyond this the block shows as code: a diagram is not a megabyte. */
export const MAX_SVG = 300_000;
const MAX_NODES = 6000;
const MAX_DEPTH = 40;

/**
 * Lenient tokenizer: tags, attributes, text, CDATA. Comments, processing instructions and DOCTYPEs are
 * skipped, so no entity a document declares is ever expanded. An unclosed tag closes at the end; a stray
 * close tag is ignored. Returns the first <svg> element, or null.
 */
export function parseSvg(src: string): SvgEl | null {
  if (src.length > MAX_SVG) return null;
  const root: SvgEl = { tag: "#root", attrs: {}, kids: [] };
  const stack: SvgEl[] = [root];
  let count = 0;
  let i = 0;
  const top = () => stack[stack.length - 1] as SvgEl;
  const text = (s: string) => { if (s && stack.length <= MAX_DEPTH) top().kids.push(decodeEntities(s)); };
  const skipTo = (end: string, from: number) => { const j = src.indexOf(end, from); return j < 0 ? src.length : j + end.length; };
  while (i < src.length) {
    const lt = src.indexOf("<", i);
    if (lt < 0) { text(src.slice(i)); break; }
    if (lt > i) text(src.slice(i, lt));
    i = lt;
    if (src.startsWith("<!--", i)) { i = skipTo("-->", i + 4); continue; }
    if (src.startsWith("<![CDATA[", i)) { const j = src.indexOf("]]>", i); text(src.slice(i + 9, j < 0 ? src.length : j)); i = j < 0 ? src.length : j + 3; continue; }
    if (src.startsWith("<!", i)) { // DOCTYPE, maybe with an internal subset: skipped whole
      const gt = src.indexOf(">", i), br = src.indexOf("[", i);
      i = br >= 0 && (gt < 0 || br < gt) ? skipTo("]>", br) : gt < 0 ? src.length : gt + 1;
      continue;
    }
    if (src.startsWith("<?", i)) { i = skipTo("?>", i + 2); continue; }
    if (src[i + 1] === "/") {
      const m = /^<\/\s*([A-Za-z][\w:.-]*)[^>]*>?/.exec(src.slice(i, i + 200));
      if (!m) { text("<"); i++; continue; }
      const name = (m[1] as string).toLowerCase();
      for (let k = stack.length - 1; k > 0; k--) if ((stack[k] as SvgEl).tag.toLowerCase() === name) { stack.length = k; break; }
      i += m[0].length;
      continue;
    }
    const nm = /^<([A-Za-z][\w:.-]*)/.exec(src.slice(i, i + 100));
    if (!nm) { text("<"); i++; continue; }
    i += nm[0].length;
    const attrs: Record<string, string> = {};
    const re = /\s*(?:(\/?>)|([^\s=/>"']+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>"']+)))?|(\S))/y;
    let selfClose = false;
    for (;;) {
      re.lastIndex = i;
      const a = re.exec(src);
      if (!a) { i = src.length; break; }
      i = re.lastIndex;
      if (a[1]) { selfClose = a[1] === "/>"; break; }
      if (a[2]) { const k = a[2]; if (!(k in attrs)) attrs[k] = decodeEntities(a[3] ?? a[4] ?? a[5] ?? ""); }
      // a[6]: a stray character (a lone "/" or quote): skip it
    }
    if (++count > MAX_NODES) break;
    const el: SvgEl = { tag: nm[1] as string, attrs, kids: [] };
    if (stack.length <= MAX_DEPTH) top().kids.push(el);
    if (!selfClose) stack.push(el);
  }
  const find = (n: SvgEl): SvgEl | null => {
    for (const k of n.kids) if (typeof k !== "string") { if (k.tag.toLowerCase() === "svg") return k; const f = find(k); if (f) return f; }
    return null;
  };
  return find(root);
}

// ───────────────────────────── sanitize ─────────────────────────────

const ELEMENTS = ["svg", "g", "defs", "symbol", "use", "title", "desc", "path", "rect", "circle", "ellipse", "line", "polyline",
  "polygon", "text", "tspan", "textPath", "marker", "linearGradient", "radialGradient", "stop", "clipPath", "mask", "pattern"];
const ELEMENT = new Map(ELEMENTS.map((e) => [e.toLowerCase(), e]));
const GRAPHIC = new Set(["path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text", "use"]);
const TEXT_PARENTS = new Set(["text", "tspan", "textPath", "title", "desc"]);

/** Colour properties: palette names or a mapped literal; never a free value. */
export const COLOR_PROPS = ["fill", "stroke", "stop-color", "color"] as const;
const ATTRS = [
  "x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry", "fx", "fy", "width", "height", "d", "points", "transform", "viewBox",
  "preserveAspectRatio", "dx", "dy", "rotate", "textLength", "lengthAdjust", "startOffset", "offset", "pathLength",
  "markerWidth", "markerHeight", "refX", "refY", "orient", "markerUnits", "gradientUnits", "gradientTransform", "spreadMethod",
  "patternUnits", "patternContentUnits", "patternTransform", "clipPathUnits", "maskUnits", "maskContentUnits",
  ...COLOR_PROPS, "fill-opacity", "fill-rule", "stroke-width", "stroke-opacity", "stroke-linecap", "stroke-linejoin", "stroke-dasharray",
  "stroke-dashoffset", "stroke-miterlimit", "opacity", "stop-opacity", "font-size", "font-weight", "font-style", "text-anchor",
  "dominant-baseline", "alignment-baseline", "baseline-shift", "letter-spacing", "marker-start", "marker-mid", "marker-end",
  "clip-path", "clip-rule", "mask", "visibility", "display", "vector-effect", "paint-order", "text-decoration", "id", "href", "aria-label",
];
const ATTR = new Map(ATTRS.map((a) => [a.toLowerCase(), a]));
/** What a `style` attribute may set (as presentation attributes). */
const STYLE_PROPS = new Set(ATTRS.filter((a) => /-|^(fill|stroke|opacity|color|display|visibility|mask)$/.test(a) && a !== "aria-label"));
const ID = /^[A-Za-z_][\w.-]{0,63}$/;
const BAD_VALUE = /javascript:|vbscript:|data:|expression\s*\(|@import|\\/i;

/** A short stable prefix for one diagram's ids, so two diagrams (or the panel) never share one. */
export function idPrefix(src: string): string {
  let h = 5381;
  for (let i = 0; i < src.length; i++) h = ((h * 33) ^ src.charCodeAt(i)) >>> 0;
  return `dg${h.toString(36)}`;
}

/** `url(#a)` -> `url(#p-a)`; null if any url() points elsewhere. */
function localUrls(v: string, p: string): string | null {
  let ok = true;
  const out = v.replace(/url\(\s*(['"]?)([^)'"]*)\1\s*\)/gi, (_all, _q, ref: string) => {
    const id = ref.trim().startsWith("#") ? ref.trim().slice(1) : "";
    if (!ID.test(id)) { ok = false; return ""; }
    return `url(#${p}-${id})`;
  });
  return ok && !/url\s*\(/i.test(out.replace(/url\(#[\w.-]+\)/g, "")) ? out : null;
}

function cleanAttr(name: string, raw: string, p: string): [string, string] | null {
  const key = ATTR.get(name.replace(/^xlink:/i, "").toLowerCase());
  if (!key) return null;
  const v = raw.trim();
  if (v.length > (key === "d" || key === "points" ? 100_000 : 400) || BAD_VALUE.test(v)) return null;
  if (key === "id") return ID.test(v) ? ["id", `${p}-${v}`] : null;
  if (key === "href") return v.startsWith("#") && ID.test(v.slice(1)) ? ["href", `#${p}-${v.slice(1)}`] : null;
  if ((COLOR_PROPS as readonly string[]).includes(key)) {
    const lv = v.toLowerCase();
    if (lv === "none" || lv === "transparent") return [key, "none"];
    if (lv === "currentcolor") return [key, "currentColor"];
    if (/^url\(/i.test(v)) { const u = localUrls(v, p); return u ? [key, u] : null; }
    const name2 = paletteName(v, key);
    return name2 ? [key, `var(--dg-${name2})`] : null;
  }
  if (/url\(/i.test(v)) { const u = localUrls(v, p); return u ? [key, u] : null; }
  return [key, v];
}

/** The declarations of a style attribute that are presentation properties we allow, as attributes. */
function styleAttrs(style: string): [string, string][] {
  const out: [string, string][] = [];
  for (const decl of style.split(";")) {
    const c = decl.indexOf(":");
    if (c < 0) continue;
    const k = decl.slice(0, c).trim().toLowerCase(), v = decl.slice(c + 1).replace(/!important/i, "").trim();
    if (STYLE_PROPS.has(k) && v) out.push([k, v]);
  }
  return out;
}

/** The tree with only what the policy allows; null if no drawing is left. `p` namespaces ids (idPrefix). */
export function sanitizeSvg(tree: SvgEl, p: string): SvgEl | null {
  const walk = (n: SvgEl): SvgEl | null => {
    const tag = ELEMENT.get(n.tag.toLowerCase());
    if (!tag) return null;
    const attrs: Record<string, string> = {};
    const style = Object.entries(n.attrs).find(([k]) => k.toLowerCase() === "style")?.[1];
    // explicit attributes win over the style attribute's, as later CSS would not; good enough for drawings
    for (const [k, v] of [...(style ? styleAttrs(style) : []), ...Object.entries(n.attrs)]) {
      if (/^on/i.test(k)) continue;
      const kv = cleanAttr(k, v, p);
      if (kv) attrs[kv[0]] = kv[1];
    }
    const kids: SvgNode[] = [];
    for (const k of n.kids) {
      if (typeof k === "string") { if (TEXT_PARENTS.has(tag) && k.trim()) kids.push(k.replace(/\s+/g, " ")); }
      else { const c = walk(k); if (c) kids.push(c); }
    }
    if (tag === "use" && !attrs.href) return null; // pointed outside the drawing
    return { tag, attrs, kids };
  };
  const out = tree.tag.toLowerCase() === "svg" ? walk(tree) : null;
  const draws = (n: SvgEl): boolean => n.kids.some((k) => typeof k !== "string" && (GRAPHIC.has(k.tag) || (k.tag !== "defs" && draws(k))));
  return out && draws(out) ? out : null;
}

// ───────────────────────────── fit ─────────────────────────────

export interface Fit { box: [number, number, number, number] }
const num = (v: string | undefined) => (v === undefined ? NaN : parseFloat(v));

/**
 * Sizes the drawing by its viewBox (from width/height if it has none, 360x240 if neither), drops its
 * width, height and a full-size background rectangle, so it sits on the card's own surface.
 */
export function fitSvg(svg: SvgEl): Fit {
  let box = (svg.attrs.viewBox ?? "").trim().split(/[\s,]+/).map(Number);
  if (box.length !== 4 || box.some((n) => !Number.isFinite(n)) || (box[2] as number) <= 0 || (box[3] as number) <= 0) {
    const w = num(svg.attrs.width), hh = num(svg.attrs.height);
    box = w > 0 && hh > 0 && !/%/.test(`${svg.attrs.width}${svg.attrs.height}`) ? [0, 0, w, hh] : [0, 0, 360, 240];
  }
  const [bx, by, bw, bh] = box as [number, number, number, number];
  svg.attrs.viewBox = box.join(" ");
  delete svg.attrs.width; delete svg.attrs.height;
  const els = (n: SvgEl) => n.kids.filter((k): k is SvgEl => typeof k !== "string");
  const first = els(svg).find((k) => k.tag !== "title" && k.tag !== "desc" && k.tag !== "defs");
  const cand = first?.tag === "rect" ? { parent: svg, rect: first } : first?.tag === "g" && els(first)[0]?.tag === "rect" ? { parent: first, rect: els(first)[0] as SvgEl } : null;
  if (cand) {
    const a = cand.rect.attrs;
    const full = (v: string | undefined, total: number) => (v ?? "").endsWith("%") ? parseFloat(v as string) >= 99 : num(v) >= total * 0.97;
    const at = (v: string | undefined, o: number) => Math.abs((num(v) || 0) - o) <= Math.max(2, bw * 0.01);
    if (full(a.width, bw) && full(a.height, bh) && at(a.x, bx) && at(a.y, by)) cand.parent.kids.splice(cand.parent.kids.indexOf(cand.rect), 1);
  }
  return { box: [bx, by, bw, bh] };
}

/** Parse, sanitize, fit: the agent's text to a safe tree and its box, or null (shown as code instead). */
export function prepareSvg(src: string): { svg: SvgEl; fit: Fit; title: string } | null {
  const tree = parseSvg(src);
  const svg = tree && sanitizeSvg(tree, idPrefix(src));
  if (!svg) return null;
  const fit = fitSvg(svg);
  const t = svg.kids.find((k): k is SvgEl => typeof k !== "string" && k.tag === "title");
  return { svg, fit, title: t ? t.kids.join("").trim().slice(0, 120) : "" };
}

// ───────────────────────────── export ─────────────────────────────

const escAttr = (s: string) => s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const escText = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
export const EXPORT_FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif";

/** A standalone SVG file: palette names baked in as hex, the font and default ink on the root, a pixel size. */
export function serializeSvg(svg: SvgEl, fit: Fit, palette: Record<string, string>): string {
  const color = (v: string) => v.replace(/var\(--dg-([a-z-]+)\)/g, (all, n: string) => palette[n] ?? all);
  const el = (n: SvgEl, root: boolean): string => {
    const attrs = { ...n.attrs };
    if (root) {
      Object.assign(attrs, { xmlns: "http://www.w3.org/2000/svg", width: String(fit.box[2]), height: String(fit.box[3]) });
      attrs["font-family"] = EXPORT_FONT;
      attrs.fill ??= "var(--dg-ink)";
      attrs["font-size"] ??= "12";
    }
    const a = Object.entries(attrs).map(([k, v]) => ` ${k}="${escAttr(color(v))}"`).join("");
    const kids = n.kids.map((k) => (typeof k === "string" ? escText(k) : el(k, false))).join("");
    return `<${n.tag}${a}${kids ? `>${kids}</${n.tag}>` : "/>"}`;
  };
  return el(svg, true);
}
