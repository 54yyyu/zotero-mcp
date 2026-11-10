import type { PanelSettings } from "../types.ts";
import { readCustomAgent } from "../custom-agent.ts";
import { errMessage, h } from "./dom.ts";
import { hint, sub } from "./settings-parts.ts";

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
    return sub("Custom ACP agent",
      hint("Launch an installed stdio ACP agent. Put only the executable in Command; arguments are separate JSON strings. Sign in with its own CLI first."),
      h("div.fields", null, field("label", "Custom agent name"), field("command", "Custom agent command"), field("args", "Arguments (JSON array)", 2), field("env", "Environment (JSON object)", 3)),
      hint("Inherits your login environment. These extra variables are saved in Zotero preferences, not the keychain; prefer the agent's own credential storage."),
      h("button.btn.btn--sm", { type: "button", onclick: async () => {
        try {
          const customAgent = readCustomAgent({ label: draft.label, command: draft.command, args: JSON.parse(draft.args), env: JSON.parse(draft.env) });
          await save({ customAgent, model: { ...s.model, custom: "" }, mode: { ...s.mode, custom: "" }, effort: { ...s.effort, custom: "" } });
        } catch (e) { error = errMessage(e); status.textContent = error; }
      } }, "Save custom agent"), status,
      hint("Applies to new chats. Saved custom chats can be read locally; continuing them, a terminal resume command, and Translate are not supported yet."));
  };
}
