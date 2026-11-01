// Callout cards, the DOM half (markdown.ts builds the `callout` and `ref` nodes). A card is a statement or an aside:
// "DEFINITION 1 — Title" over its body. Statements are numbered across the whole chat in reading order, and a
// `\ref{label}` becomes a link to the card with that label: a click scrolls there and flashes it. The numbering is a pass
// over the feed (linkCallouts), run only when a card or a reference was drawn since the last pass.
import { CALLOUT_KINDS } from "./markdown.ts";
import { env, h } from "./dom.ts";

let dirty = false;
/** True once after a card or a reference was drawn: the feed numbers and links them again. */
export const takeCalloutsDirty = (): boolean => { const d = dirty; dirty = false; return d; };

/** `title` and `body` are already DOM nodes. An unknown kind (the tree allows none) is a plain group. */
export function calloutEl(kind: string, label: string | undefined, title: Node[] | null, body: Node[]): HTMLElement {
  dirty = true;
  const spec = CALLOUT_KINDS[kind];
  const name = h("span.cb__kind", null, spec?.label ?? kind, spec?.numbered ? h("span.cb__num") : null);
  const head = h("div.cb__head", null, name, title ? h("span.cb__title", null, " — ", ...title) : null);
  return h(`div.cb.cb--${spec?.tone ?? "rem"}`, { role: "group", dataset: { kind, ...(label ? { label } : {}) } }, head, h("div.cb__body", null, ...body));
}

export function refEl(label: string): HTMLElement {
  dirty = true;
  return h("button.cbref", { type: "button", dataset: { ref: label } }, label);
}

const cards = (root: ParentNode): HTMLElement[] => [...root.querySelectorAll<HTMLElement>(".cb[data-kind]")];

/** Numbers the statements ("Definition 1", "Theorem 2", …) and points every reference at its card. */
export function linkCallouts(root: ParentNode): void {
  const byLabel = new Map<string, HTMLElement>();
  let n = 0;
  for (const c of cards(root)) {
    const num = c.querySelector<HTMLElement>(":scope > .cb__head .cb__num");
    if (!num) continue;
    const text = ` ${++n}`;
    if (num.textContent !== text) num.textContent = text;
    c.dataset.number = String(n);
    const label = c.dataset.label;
    if (label && !byLabel.has(label)) byLabel.set(label, c);
  }
  for (const r of root.querySelectorAll<HTMLElement>(".cbref")) {
    const c = byLabel.get(r.dataset.ref ?? "");
    r.classList.toggle("cbref--missing", !c);
    if (!c) { r.title = "Not defined in this chat"; continue; }
    const kind = CALLOUT_KINDS[c.dataset.kind ?? ""]?.label ?? "";
    const text = `${kind} ${c.dataset.number}`;
    if (r.textContent !== text) r.textContent = text;
    const t = c.querySelector(":scope > .cb__head .cb__title")?.textContent?.replace(/^\s*— /, "");
    const body = (c.querySelector(".cb__body")?.textContent ?? "").replace(/\s+/g, " ").trim();
    r.title = `${kind} ${c.dataset.number}${t ? ` — ${t}` : ""}\n${body.length > 200 ? `${body.slice(0, 200)}…` : body}`;
  }
}

/** A click on a reference: scroll its card into view and flash it. */
export function jumpToCallout(ref: HTMLElement): void {
  const root = ref.closest(".feed__inner");
  const box = root && cards(root).find((c) => c.dataset.label === ref.dataset.ref);
  if (!box) return;
  const reduce = env.win.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  box.scrollIntoView({ block: "center", behavior: reduce ? "auto" : "smooth" });
  box.classList.remove("cb--flash");
  void box.offsetWidth; // restart the animation
  box.classList.add("cb--flash");
  env.win.setTimeout(() => box.classList.remove("cb--flash"), 1800);
}
