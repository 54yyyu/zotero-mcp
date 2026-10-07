// The settings screen's controls, shared by its cards: segmented choice, radio list, switch, select, a labelled
// field, a card and its rows, swatches, a slider, and a two-step confirm.
import { append, env, h } from "./dom.ts";
import type { Kid } from "./dom.ts";

export interface Opt { id: string; label: string; disabled?: boolean; title?: string }

export function seg(name: string, options: Opt[], value: string, onPick: (id: string) => void): HTMLElement {
  const el = h("div.seg", { role: "radiogroup", "aria-label": name });
  append(el, options.map((o) => h(`button.seg__opt${o.id === value ? ".seg__opt--on" : ""}`, {
    type: "button", role: "radio", "aria-checked": String(o.id === value), tabindex: o.id === value ? "0" : "-1", disabled: o.disabled ? true : null, title: o.title, dataset: { fid: `${name}:${o.id}` },
    onclick: () => onPick(o.id),
    onkeydown: (e: KeyboardEvent) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      e.preventDefault();
      const live = options.filter((x) => !x.disabled);
      const i = live.findIndex((x) => x.id === value);
      const n = live[(i + (e.key === "ArrowRight" ? 1 : live.length - 1)) % live.length];
      if (n) onPick(n.id);
    },
  }, o.label)));
  return el;
}

/** A vertical list of choices with a one-line explanation each. */
export function radios(name: string, items: { id: string; label: string; help?: string }[], value: string, onPick: (id: string) => void): HTMLElement {
  return h("div.radios", { role: "radiogroup", "aria-label": name }, items.map((m) => h(`button.radio${m.id === value ? ".radio--on" : ""}`, {
    type: "button", role: "radio", "aria-checked": String(m.id === value), dataset: { fid: `${name}:${m.id}` }, onclick: () => onPick(m.id),
  }, h("span.radio__dot"), h("span.radio__tx", null, h("span.radio__t", null, m.label), m.help ? h("span.radio__d", null, m.help) : null))));
}

/** A label, its explanation and a switch on the right. */
export function switchRow(label: string, help: string, on: boolean, onChange: (on: boolean) => void): HTMLElement {
  const input = h("input", { type: "checkbox", role: "switch", checked: on, dataset: { fid: `switch:${label}` }, onchange: () => onChange((input as HTMLInputElement).checked) });
  return h("label.swrow", null,
    h("span.swrow__tx", null, h("span.swrow__t", null, label), help ? h("span.swrow__d", null, help) : null),
    h("span.switch", null, input, h("span.switch__track", null, h("span.switch__thumb"))));
}

/** A native select in our frame; `options[].id === ""` is the "default" entry. */
export function selectField(label: string, options: { id: string; label: string }[], value: string, onChange: (id: string) => void): HTMLElement {
  const sel = h("select.input", { "aria-label": label, dataset: { fid: `select:${label}` }, onchange: () => onChange((sel as HTMLSelectElement).value) },
    options.map((o) => h("option", { value: o.id, selected: o.id === value ? true : null }, o.label)));
  return h("div.field", null, h("span.field__l", null, label), h("div.select", null, sel));
}

/** A card: a title, one line on what it is for, then its rows. */
export function section(title: string, desc: string, ...kids: Kid[]): HTMLElement {
  return h("section.sec", { "aria-label": title },
    h("header.sec__head", null, h("h3.sec__t", null, title), h("p.sec__d", null, desc)),
    h("div.sec__body", null, ...kids));
}

/** A titled group inside a card (sign-in inside Agent, the shortcuts inside About). */
export const sub = (title: string, ...kids: Kid[]): HTMLElement => h("div.sec__sub", { role: "group", "aria-label": title }, h("span.sec__subt", null, title), ...kids);

export const hint = (text: string): HTMLElement => h("p.sec__hint", null, text);

/** A row: the label on the left, its control on the right (below it when the card is narrow), a hint under both. */
export function row(label: string, control: Kid, help?: string): HTMLElement {
  return h("div.row", null, h("div.row__main", null, h("span.row__l", null, label), h("div.row__ctl", null, control)), help ? h("p.row__hint", null, help) : null);
}

/** Round colour swatches (or any small tiles) as one radio group; `extra` is appended inside it (the custom colour). */
export function swatches(name: string, cls: string, items: { id: string; label: string; style?: string; mod?: string }[], value: string, onPick: (id: string) => void, extra?: Kid): HTMLElement {
  const pick = (dir: number) => {
    const i = items.findIndex((x) => x.id === value);
    const n = items[(i + dir + items.length) % items.length];
    if (n) onPick(n.id);
  };
  const none = !items.some((x) => x.id === value);
  return h("div.sws", { role: "radiogroup", "aria-label": name }, items.map((o, i) => h(`button.${cls}${o.mod ? `.${cls}--${o.mod}` : ""}`, {
    type: "button", role: "radio", "aria-checked": String(o.id === value), "aria-label": o.label, title: o.label, style: o.style,
    tabindex: o.id === value || (none && i === 0) ? "0" : "-1", dataset: { fid: `${name}:${o.id}` }, onclick: () => onPick(o.id),
    onkeydown: (e: KeyboardEvent) => { if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); pick(e.key === "ArrowRight" ? 1 : -1); } },
  })), extra);
}

/** A labelled range with its value shown: `onInput` runs while dragging (a live preview), `onChange` on release (the save). */
export function slider(label: string, o: { value: number; min: number; max: number; unit: string; help?: string; onInput(v: number): void; onChange(v: number): void }): HTMLElement {
  const out = h("span.row__v", null, `${o.value}${o.unit}`);
  const input = h("input", {
    type: "range", min: String(o.min), max: String(o.max), step: "1", value: String(o.value), "aria-label": label, dataset: { fid: `range:${label}` },
    oninput: () => { const v = Number((input as HTMLInputElement).value); out.textContent = `${v}${o.unit}`; o.onInput(v); },
    onchange: () => o.onChange(Number((input as HTMLInputElement).value)),
  });
  return h("div.row", null, h("div.row__main", null, h("span.row__l", null, label), out), input, o.help ? h("p.row__hint", null, o.help) : null);
}

/** A button that asks "sure?" before it runs. */
export function confirmAction(o: { label: string; ask: string; yes: string; danger?: boolean; run: () => Promise<string | void> }): HTMLElement {
  const box = h("div.confirm");
  const idle = (msg = "") => {
    box.replaceChildren(h("button.btn.btn--sm", { type: "button", onclick: ask }, o.label), msg ? h("span.confirm__msg", { role: "status" }, msg) : "");
  };
  const ask = () => {
    const yes = h(`button.btn.btn--sm${o.danger ? ".btn--danger" : ".btn--solid"}`, { type: "button", onclick: go }, o.yes);
    box.replaceChildren(h("span.confirm__q", { role: "alert" }, o.ask), yes, h("button.btn.btn--sm", { type: "button", onclick: () => idle() }, "Cancel"));
    (box.querySelector("button.btn--sm:last-child") as HTMLElement | null)?.focus();
  };
  const go = async () => {
    box.replaceChildren(h("span.confirm__msg", null, "Working…"));
    try { idle((await o.run()) || "Done."); } catch (e) { idle(`Failed: ${e instanceof Error ? e.message : String(e)}`); }
    env.win.setTimeout(() => { const m = box.querySelector(".confirm__msg"); if (m) m.textContent = ""; }, 4000);
  };
  idle();
  return box;
}
