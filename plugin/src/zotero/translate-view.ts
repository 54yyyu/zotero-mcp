// The translation inside the reader's own text selection popup, under its buttons (Zotero's documented use of
// renderTextSelectionPopup: append a container, fill it later). The popup widens for it and is kept pointing at the
// selection as it grows; it closes with the selection, as it always does. Styled with the reader's own tokens, so it
// follows its light and dark themes; the bar on the left is the panel's accent (quiet ink when it is Mono).
import type { PanelSettings } from "../types.ts";
import { LANGUAGES, language } from "../ui/settings-model.ts";
import { translationRequest } from "../agent/index.ts";
import type { Translator } from "./translate.ts";

const WIDTH = 320;

const CSS = `
.zmc-tr{margin-top:7px;padding-top:9px;border-top:1px solid var(--fill-quinary);display:flex;flex-direction:column;gap:6px;text-align:start;animation:zmc-tr-in .16s ease-out}
.zmc-tr__body{position:relative;padding-inline-start:11px}
.zmc-tr__body::before{content:"";position:absolute;inset-block:3px;inset-inline-start:0;width:2px;border-radius:2px;background:var(--zmc-tr-accent)}
.zmc-tr__scroll{max-height:216px;overflow-y:auto;overscroll-behavior:contain;scrollbar-width:thin;scrollbar-color:var(--fill-quarternary) transparent;padding-inline-end:2px}
.zmc-tr__scroll[data-more]{mask-image:linear-gradient(to bottom,#000 calc(100% - 26px),transparent)}
.zmc-tr__text{font-size:13.5px;line-height:1.55;color:var(--fill-primary);white-space:pre-wrap;overflow-wrap:anywhere;user-select:text;-moz-user-select:text;cursor:text;outline:none}
.zmc-tr__text:empty{display:none}
.zmc-tr__text span{animation:zmc-tr-in .24s ease-out}
.zmc-tr__err{font-size:12.5px;line-height:1.5;color:var(--fill-secondary);user-select:text;-moz-user-select:text}
.zmc-tr__err b{color:var(--fill-primary);font-weight:600;display:block}
.zmc-tr__sk{display:grid;gap:8px;padding-block:5px 3px}
.zmc-tr__sk i{display:block;height:7px;border-radius:4px;background:linear-gradient(90deg,var(--fill-quinary) 30%,var(--fill-quarternary) 50%,var(--fill-quinary) 70%);background-size:300% 100%;animation:zmc-tr-sh 1.3s ease-in-out infinite}
.zmc-tr__sk i:last-child{width:58%}
.zmc-tr__foot{display:flex;align-items:center;gap:4px;min-height:22px;padding-inline-start:11px}
.zmc-tr__lang{position:relative;display:inline-flex;align-items:center;gap:3px;height:20px;padding-inline:5px 4px;margin-inline-start:-5px;border-radius:6px;font-size:11.5px;color:var(--fill-secondary)}
.zmc-tr__lang:hover{background:var(--fill-quinary);color:var(--fill-primary)}
.zmc-tr__lang svg{color:var(--fill-tertiary)}
.zmc-tr__lang select{position:absolute;inset:0;width:100%;opacity:0;cursor:pointer;font:inherit}
.zmc-tr__lang:has(select:focus-visible){outline:2px solid var(--color-accent)}
.zmc-tr__note{flex:1;min-width:0;font-size:11px;color:var(--fill-tertiary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.zmc-tr__act{appearance:none;-moz-appearance:none;border:0;background:transparent;color:var(--fill-secondary);font:inherit;font-size:11.5px;display:inline-flex;align-items:center;gap:4px;height:20px;padding:0 6px;border-radius:6px;cursor:pointer;margin-inline-end:-4px}
.zmc-tr__act:hover{background:var(--fill-quinary);color:var(--fill-primary)}
.zmc-tr__act[hidden]{display:none}
.zmc-tr__act:focus-visible{outline:2px solid var(--color-accent);outline-offset:0}
@keyframes zmc-tr-in{from{opacity:0}}
@keyframes zmc-tr-sh{from{background-position:100% 0}to{background-position:0 0}}
@media (prefers-reduced-motion:reduce){.zmc-tr,.zmc-tr__text span{animation:none}.zmc-tr__sk i{animation-duration:3s}}
`;

const SVG = "http://www.w3.org/2000/svg";
const ICON = {
  chevron: "M2.5 4.5 6 8l3.5-3.5",
  copy: "M4.5 4.5V3a1 1 0 0 1 1-1h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1H9M2.5 4.5h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1h-5a1 1 0 0 1-1-1v-5a1 1 0 0 1 1-1Z",
  check: "M2.5 6.5 5 9l4.5-5.5",
};

/** What one press shows: the result block in this popup, and how to take it away. `text`: the selection it translates. */
interface Shown { text: string; reader: any; box: Element; el: HTMLElement; popup: HTMLElement | null; button: HTMLElement; closed: boolean }

export function createPopupTranslate(t: Translator, o: { settings(): PanelSettings; copy(text: string): void }) {
  let shown: Shown | null = null;

  const close = (s: Shown) => {
    s.closed = true;
    t.cancel();
    s.el.remove();
    s.button.setAttribute("aria-expanded", "false");
    s.popup?.style.removeProperty("width");
    s.popup?.style.removeProperty("max-width");
    if (shown === s) shown = null;
  };

  /** Widen the popup for the result and keep it at the selection. */
  const attach = (s: Shown) => {
    s.popup = s.box.closest(".selection-popup") as HTMLElement | null;
    if (s.popup) { keepAnchored(s.popup); s.popup.style.width = `${WIDTH}px`; s.popup.style.maxWidth = "calc(100vw - 40px)"; }
    s.button.setAttribute("aria-expanded", "true");
    s.box.append(s.el);
    // Focus is in the popup now, where the reader does not take Esc: clear the selection as Esc in the PDF does.
    s.box.addEventListener("keydown", (e: Event) => {
      if ((e as KeyboardEvent).key !== "Escape") return;
      e.preventDefault();
      try { const v = s.reader._internalReader._lastView; v.clearSelection(); v._render?.(); v.focus(); } catch { close(s); }
    });
  };

  /** The Translate button was pressed in this popup: show the translation under it, or take it away again. */
  function toggle(event: any, box: Element, button: HTMLElement): void {
    if (shown?.box === box) return close(shown);
    if (shown) close(shown);
    const doc: Document = event.doc;
    if (!doc.getElementById("zmc-tr-css")) doc.head.append(Object.assign(doc.createElement("style"), { id: "zmc-tr-css", textContent: CSS }));
    const el = node(doc, "div", "zmc-tr", { role: "region", "aria-label": "Translation" });
    // The panel's accent; Mono is a quiet ink line.
    el.style.setProperty("--zmc-tr-accent", o.settings().appearance.accent || "color-mix(in srgb, var(--fill-primary) 32%, transparent)");
    const s: Shown = { text: String(event.params.annotation.text), reader: event.reader, box, el, popup: null, button, closed: false };
    shown = s;
    attach(s);
    run(s, s.text, o.settings().translateTo);
  }

  /**
   * The reader rebuilt its popup's plugin sections for the same selection (it does when its own state changes): the
   * translation moves into the new section, still streaming. Synchronous, inside Zotero's render, so no word is lost.
   */
  function revive(event: any, box: Element, button: HTMLElement): void {
    const s = shown;
    if (!s || s.closed || s.el.ownerDocument !== event.doc || s.text !== String(event.params.annotation.text)) return;
    Object.assign(s, { box, button, reader: event.reader });
    attach(s);
  }

  function run(s: Shown, source: string, to: string): void {
    const doc = s.el.ownerDocument;
    const { text: request, truncated } = translationRequest(source);
    const out = node(doc, "div", "zmc-tr__text", { lang: to, dir: "auto", tabindex: "0" });
    const sk = node(doc, "div", "zmc-tr__sk", { "aria-label": "Translating" }, node(doc, "i"), node(doc, "i"));
    const scroll = node(doc, "div", "zmc-tr__scroll", null, sk, out);
    const select = node(doc, "select", "", { "aria-label": "Translate into" }, ...LANGUAGES.map((l) => node(doc, "option", "", { value: l.id }, l.label))) as HTMLSelectElement;
    select.value = to;
    const chip = node(doc, "label", "zmc-tr__lang", null, node(doc, "span", "", null, language(to).label), icon(doc, ICON.chevron), select);
    select.addEventListener("change", () => run(s, source, select.value)); // this popup only; the setting stays
    const copy = node(doc, "button", "zmc-tr__act", { type: "button", title: "Copy the translation", hidden: "" }, icon(doc, ICON.copy), "Copy") as HTMLButtonElement;
    const note = node(doc, "span", "zmc-tr__note", null, truncated ? "First 6,000 characters" : "");
    s.el.replaceChildren(node(doc, "div", "zmc-tr__body", null, scroll), node(doc, "div", "zmc-tr__foot", null, chip, note, copy));
    s.el.setAttribute("aria-busy", "true");
    scroll.addEventListener("scroll", () => fade(scroll), { passive: true });

    let all = "";
    copy.addEventListener("click", () => {
      o.copy(all.trim());
      copy.replaceChildren(icon(doc, ICON.check), "Copied");
      doc.defaultView!.setTimeout(() => copy.replaceChildren(icon(doc, ICON.copy), "Copy"), 1400);
    });
    const gone = () => s.closed || !s.el.isConnected;
    t.translate({
      text: request, to,
      onText(delta) {
        if (gone()) return t.cancel(); // the popup closed with its selection
        const piece = all ? delta : delta.trimStart();
        all += delta;
        if (!piece) return;
        sk.remove();
        out.append(node(doc, "span", "", null, piece));
        fade(scroll);
      },
    }).then(
      () => {
        if (gone() || select.value !== to) return;
        out.replaceChildren(all.trim()); // one text node: selecting and copying see plain text
        copy.hidden = false;
        s.el.removeAttribute("aria-busy");
        fade(scroll);
      },
      (e: unknown) => {
        if (gone() || select.value !== to) return;
        sk.remove();
        out.remove();
        s.el.removeAttribute("aria-busy");
        scroll.append(node(doc, "div", "zmc-tr__err", { role: "alert" }, node(doc, "b", "", null, "Couldn't translate"), e instanceof Error ? e.message : String(e)));
        const retry = node(doc, "button", "zmc-tr__act", { type: "button" }, "Try again");
        retry.addEventListener("click", () => run(s, source, select.value));
        copy.replaceWith(retry);
      },
    );
  }

  return { toggle, revive };
}

/** A soft fade at the bottom edge while more text is below. */
function fade(scroll: HTMLElement): void {
  const more = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight > 2;
  if (more) scroll.setAttribute("data-more", ""); else scroll.removeAttribute("data-more");
}

/**
 * The reader places its popup once, when it renders (ViewPopup's translate(left, top)), not when its content grows. So as
 * the popup widens and grows, move it the way the reader would have placed it: centred on the same point, the arrow still
 * at the selection, kept inside the view.
 */
const anchored = new WeakSet<HTMLElement>();
function keepAnchored(popup: HTMLElement): void {
  if (anchored.has(popup)) return;
  anchored.add(popup);
  let w = popup.offsetWidth, h = popup.offsetHeight;
  new (popup.ownerDocument.defaultView as any).ResizeObserver(() => {
    const nw = popup.offsetWidth, nh = popup.offsetHeight, dw = nw - w, dh = nh - h;
    w = nw; h = nh;
    const m = /translate\(([-\d.]+)px,\s*([-\d.]+)px\)/.exec(popup.style.transform);
    if ((!dw && !dh) || !m) return;
    const side = /page-popup-(top|bottom|left|right)-center/.exec(popup.className)?.[1];
    let left = Number(m[1]), top = Number(m[2]);
    if (side === "top" || side === "bottom") left -= dw / 2;
    else if (side === "left") left -= dw;
    if (side === "top") top -= dh;
    else if (side === "left" || side === "right") top -= dh / 2;
    const view = popup.parentElement!.getBoundingClientRect(), pad = 20;
    left = Math.max(pad, Math.min(left, view.width - nw - pad));
    top = Math.max(pad, Math.min(top, view.height - nh - pad));
    popup.style.transform = `translate(${left}px, ${top}px)`;
  }).observe(popup);
}

function node(doc: Document, tag: string, cls = "", attrs: Record<string, string> | null = null, ...kids: (Node | string)[]): HTMLElement {
  const el = doc.createElement(tag);
  if (cls) el.className = cls;
  for (const [k, v] of Object.entries(attrs ?? {})) el.setAttribute(k, v);
  el.append(...kids);
  return el;
}

function icon(doc: Document, d: string): SVGElement {
  const svg = doc.createElementNS(SVG, "svg");
  for (const [k, v] of Object.entries({ width: "12", height: "12", viewBox: "0 0 12 12", fill: "none", stroke: "currentColor", "stroke-width": "1.3", "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" })) svg.setAttribute(k, v);
  const p = doc.createElementNS(SVG, "path");
  p.setAttribute("d", d);
  svg.append(p);
  return svg;
}
