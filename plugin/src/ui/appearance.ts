// How the panel looks: glass, accent colour, background, text size, density. The pure part (validation, the accent's
// colours by WCAG contrast) is unit-tested; `Look` writes the choice onto the panel's root as data attributes and CSS
// variables, so a change shows at once and nothing re-renders. The CSS that reads them is styles-look.ts.
import type { Appearance, PanelHost } from "../types.ts";

export const DEFAULT_APPEARANCE: Appearance = {
  glass: true, accent: "", background: "none", image: "", imageVisibility: 50, imageBlur: 0, textSize: "default", density: "comfortable",
};

/** The swatches, each a light and a dark colour. The first, Mono (our ink), is the default and is saved as "". */
export const ACCENTS = [
  { id: "", name: "Mono", light: "#16181d", dark: "#e6e8eb" },
  { id: "#cc2936", name: "Red", light: "#cc2936", dark: "#ff7b86" },
  { id: "#d9480f", name: "Orange", light: "#d9480f", dark: "#ff922b" },
  { id: "#2b8a3e", name: "Green", light: "#2b8a3e", dark: "#69db7c" },
  { id: "#0b7285", name: "Teal", light: "#0b7285", dark: "#66d9e8" },
  { id: "#2563c9", name: "Blue", light: "#2563c9", dark: "#79a8ff" },
  { id: "#6741d9", name: "Violet", light: "#6741d9", dark: "#b197fc" },
];

/** Drawn by CSS (styles-look.ts), so they cost no bytes and follow light and dark: the glow (three faint washes of colour)
 * and soft gradients. "none" (Plain, the default) is just the paper. */
export const BACKGROUNDS = [
  { id: "glow", name: "Glow" },
  { id: "mist", name: "Mist" },
  { id: "dawn", name: "Dawn" },
  { id: "sage", name: "Sage" },
  { id: "dusk", name: "Dusk" },
  { id: "sand", name: "Sand" },
];

const HEX = /^#[0-9a-f]{6}$/i;

/** Saved appearance (any shape, from prefs) to a complete, valid one: each field checked on its own, so one bad value never resets the rest. */
export function readAppearance(raw: unknown): Appearance {
  const s = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const d = DEFAULT_APPEARANCE;
  const num = (v: unknown, min: number, max: number, def: number) => (typeof v === "number" && Number.isFinite(v) ? Math.min(max, Math.max(min, Math.round(v))) : def);
  const oneOf = <T extends string>(v: unknown, list: readonly T[], def: T): T => (list.includes(v as T) ? (v as T) : def);
  const image = typeof s.image === "string" ? s.image.slice(0, 120) : "";
  const accent = typeof s.accent === "string" && HEX.test(s.accent) ? s.accent.toLowerCase() : "";
  return {
    glass: typeof s.glass === "boolean" ? s.glass : d.glass,
    accent,
    background: oneOf(s.background, ["none", ...BACKGROUNDS.map((b) => b.id), ...(image ? ["image"] : [])], "none"),
    image,
    imageVisibility: num(s.imageVisibility, 0, 100, d.imageVisibility),
    imageBlur: num(s.imageBlur, 0, 20, d.imageBlur),
    textSize: oneOf(s.textSize, ["small", "default", "large"] as const, d.textSize),
    density: oneOf(s.density, ["compact", "comfortable"] as const, d.density),
  };
}

// ───────────────────────────── colour ─────────────────────────────

type RGB = [number, number, number];
const rgb = (hex: string): RGB => [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
const toHex = (c: number[]): string => `#${c.map((v) => Math.round(Math.min(255, Math.max(0, v))).toString(16).padStart(2, "0")).join("")}`;

/** WCAG relative luminance. */
export function luminance(hex: string): number {
  const lin = (v: number) => { const c = v / 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const [r, g, b] = rgb(hex);
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

export function contrast(a: string, b: string): number {
  const la = luminance(a), lb = luminance(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

/** `hex` moved toward `to` in 5% steps until it reaches `ratio` against `bg` (unchanged if it already does). */
export function reach(hex: string, bg: string, ratio: number, to: string): string {
  const a = rgb(hex), b = rgb(to);
  for (let t = 0; t <= 1.0001; t += 0.05) {
    const c = toHex([0, 1, 2].map((i) => a[i]! + (b[i]! - a[i]!) * t));
    if (contrast(c, bg) >= ratio) return c;
  }
  return to;
}

// What accent text sits on, the hardest case in each theme: the light paper (cards are lighter still), and in dark the
// lightest surface, a glass card (#262a31), which is lighter than the paper.
const PAPER = { light: "#f5f6f7", dark: "#262a31" };
const INK = "#16181d";

/** Text on an accent fill: white or our ink, whichever reads better. */
export const textOn = (fill: string): string => (contrast(fill, "#ffffff") >= contrast(fill, INK) ? "#ffffff" : INK);

/**
 * The CSS variables for an accent, per theme: the fill (`--accent`, the swatch's own colour; a custom one is lifted in dark
 * mode only as far as it needs to stay visible, 3:1), the text-safe version (`--accent-ink`, 4.5:1 on the paper, for links
 * and labels), and the colour of text on the fill (`--on-accent`).
 */
export function accentVars(accent: string): Record<string, string> {
  const preset = ACCENTS.find((a) => a.id === accent);
  const light = preset?.light ?? accent;
  const dark = preset?.dark ?? reach(accent, PAPER.dark, 3, "#ffffff");
  return {
    "--accent-l": light, "--accent-d": dark,
    "--accent-ink-l": reach(light, PAPER.light, 4.5, "#000000"), "--accent-ink-d": reach(dark, PAPER.dark, 4.5, "#ffffff"),
    "--on-accent-l": textOn(light), "--on-accent-d": textOn(dark),
  };
}

// ───────────────────────────── applying it ─────────────────────────────

/** Puts an appearance on the panel's root element. Owns the background picture, read from the host once, when first shown. */
export class Look {
  private el: HTMLElement;
  private host: PanelHost;
  private image: string | null = null;
  private loading = false;

  constructor(el: HTMLElement, host: PanelHost) {
    this.el = el;
    this.host = host;
  }

  /** The saved look, or `a`: a change being previewed (a slider being dragged) before it is saved. */
  apply(a: Appearance = this.host.getSettings().appearance): void {
    const { el } = this;
    const wantsImage = a.background === "image" && !!a.image;
    Object.assign(el.dataset, {
      glass: a.glass ? "on" : "off", size: a.textSize, density: a.density,
      bg: wantsImage ? (this.image ? "image" : "none") : a.background,
    });
    const vars: Record<string, string> = {
      ...accentVars(a.accent),
      // The veil is the panel's paper laid over the picture: at full visibility 30% of it stays, so text keeps its contrast.
      "--bg-veil": `${Math.round(100 - a.imageVisibility * 0.7)}%`,
      "--bg-blur": `${a.imageBlur}px`,
    };
    for (const [k, v] of Object.entries(vars)) el.style.setProperty(k, v);
    // The contract with the diagrams: data-accent="custom" only when the user chose an accent other than mono.
    if (a.accent) el.dataset.accent = "custom"; else delete el.dataset.accent;
    if (wantsImage && !this.image && !this.loading) {
      this.loading = true;
      this.host.loadImage().then((url) => { if (url) this.setImage(url); }, () => {}).finally(() => { this.loading = false; });
    }
  }

  /** A picture just chosen (or null: removed). */
  setImage(url: string | null): void {
    this.image = url;
    // A data: URL in a custom property, read by .zmc::before: Zotero loads data: images from a stylesheet, not file: ones.
    if (url) this.el.style.setProperty("--bg-img", `url("${url}")`); else this.el.style.removeProperty("--bg-img");
    this.apply();
  }
}
