// The settings screen's controls, shared by its sections: segmented choice, radio list, switch, select,
// a labelled field, a section, and a two-step confirm.
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

export function section(title: string, ...kids: Kid[]): HTMLElement {
  return h("section.sec", { "aria-label": title }, h("h3.sec__t", null, title), ...kids);
}

export const hint = (text: string): HTMLElement => h("p.sec__hint", null, text);

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
