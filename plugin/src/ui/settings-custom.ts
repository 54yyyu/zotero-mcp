import type { PanelSettings } from "../types.ts";
import { readCustomAgent } from "../custom-agent.ts";
import { errMessage, h } from "./dom.ts";
import { hint, sub } from "./settings-parts.ts";
import { t } from "../i18n.ts";

/** Keep unfinished JSON while a health or catalog answer rebuilds the settings page. */
export function customAgentEditor(get: () => PanelSettings, save: (patch: Partial<PanelSettings>) => Promise<void>): () => HTMLElement {
  let source: string | undefined;
  let draft = { label: "", command: "", args: "[]", env: "{}" };
  let error = "";
  return () => {
    const s = get();
    const current = JSON.stringify(s.customAgent);
    if (source !== current) {
      source = current;
      error = "";
      draft = { label: s.customAgent?.label ?? "", command: s.customAgent?.command ?? "", args: JSON.stringify(s.customAgent?.args ?? []), env: JSON.stringify(s.customAgent?.env ?? {}, null, 2) };
    }
    const status = h("p.sec__hint", { role: "alert" }, error);
    const field = (key: keyof typeof draft, label: string, rows?: number) => {
      const input = h(rows ? "textarea.input" : "input.input", {
        value: draft[key], rows, type: rows ? undefined : "text", spellcheck: "false", autocomplete: "off",
        "aria-label": label, dataset: { fid: `custom:${key}` }, oninput: () => { draft[key] = (input as HTMLInputElement).value; },
      });
      return h("label.field", null, h("span.field__l", null, label), input);
    };
    return sub(t("settings.custom.title"),
      hint(t("settings.custom.hint")),
      h("div.fields", null, field("label", t("settings.custom.name")), field("command", t("settings.custom.command")), field("args", t("settings.custom.args"), 2), field("env", t("settings.custom.env"), 3)),
      hint(t("settings.custom.envHint")),
      h("button.btn.btn--sm", { type: "button", onclick: async () => {
        try {
          const customAgent = readCustomAgent({ label: draft.label, command: draft.command, args: JSON.parse(draft.args), env: JSON.parse(draft.env) });
          await save({ customAgent, model: { ...s.model, custom: "" }, mode: { ...s.mode, custom: "" }, effort: { ...s.effort, custom: "" } });
        } catch (e) { error = errMessage(e); status.textContent = error; }
      } }, t("settings.custom.save")), status,
      hint(t("settings.custom.limits")));
  };
}
