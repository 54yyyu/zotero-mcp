import type { CustomBackendId, PanelSettings } from "../types.ts";
import { customAgentIds, customLaunchKey, readCustomAgent } from "../custom-agent.ts";
import { env, errMessage, h } from "./dom.ts";
import { confirmAction, hint, sub } from "./settings-parts.ts";
import { t } from "../i18n.ts";

function clearChoices(s: PanelSettings, id: CustomBackendId): Pick<PanelSettings, "model" | "mode" | "effort"> {
  const model = { ...s.model }, mode = { ...s.mode }, effort = { ...s.effort };
  delete model[id];
  delete mode[id];
  delete effort[id];
  return { model, mode, effort };
}

/** Keep unfinished JSON while a health or catalog answer rebuilds the settings page. */
export function customAgentEditor(get: () => PanelSettings, save: (patch: Partial<PanelSettings>) => Promise<void>) {
  let source: string | undefined;
  let draft = { label: "", command: "", args: "[]", env: "{}" };
  let error = "";
  return (id?: CustomBackendId, cancel?: () => void): HTMLElement => {
    const config = id ? get().customAgents[id] : undefined;
    const current = JSON.stringify([id, config]);
    if (source !== current) {
      source = current;
      error = "";
      draft = {
        label: config ? config.label || t("agent.custom.tab") : "",
        command: config?.command ?? "",
        args: JSON.stringify(config?.args ?? []),
        env: JSON.stringify(config?.env ?? {}, null, 2),
      };
    }
    const status = h("p.sec__hint", { role: "alert" }, error);
    const field = (key: keyof typeof draft, label: string, rows?: number) => {
      const input = h(rows ? "textarea.input" : "input.input", {
        value: draft[key], rows, type: rows ? undefined : "text", spellcheck: "false", autocomplete: "off",
        "aria-label": label, dataset: { fid: `custom:${key}` }, oninput: () => { draft[key] = (input as HTMLInputElement).value; },
      });
      return h("label.field", null, h("span.field__l", null, label), input);
    };
    const saveButton = h("button.btn.btn--sm.btn--solid", { type: "button", onclick: async () => {
      saveButton.disabled = true;
      error = "";
      status.textContent = "";
      try {
        if (!draft.label.trim()) throw new Error(t("settings.custom.missingName"));
        const agent = readCustomAgent({ label: draft.label, command: draft.command, args: JSON.parse(draft.args), env: JSON.parse(draft.env) });
        const s = get();
        if (id && !s.customAgents[id]) return;
        const key = id ?? `custom-${env.win.crypto.randomUUID()}`;
        const changed = customLaunchKey(s.customAgents[key]) !== customLaunchKey(agent);
        await save({
          ...(id ? {} : { backend: key }),
          customAgents: { ...s.customAgents, [key]: agent },
          ...(changed ? clearChoices(s, key) : {}),
        });
      } catch (e) {
        error = errMessage(e);
        status.textContent = error;
      } finally {
        saveButton.disabled = false;
      }
    } }, t("settings.custom.save")) as HTMLButtonElement;

    const remove = id && config ? confirmAction({
      label: t("common.delete"), ask: t("settings.custom.deleteAsk", { name: config.label || t("agent.custom.tab") }),
      yes: t("common.delete"), danger: true,
      async run() {
        const s = get();
        const customAgents = { ...s.customAgents };
        delete customAgents[id];
        await save({
          ...(s.backend === id ? { backend: customAgentIds(customAgents)[0] ?? "claude-code" } : {}),
          customAgents, ...clearChoices(s, id),
        });
      },
    }) : null;

    return sub(t(config ? "settings.custom.title" : "settings.custom.new"),
      hint(t("settings.custom.hint")),
      h("div.fields", null, field("label", t("settings.custom.name")), field("command", t("settings.custom.command")), field("args", t("settings.custom.args"), 2), field("env", t("settings.custom.env"), 3)),
      hint(t("settings.custom.envHint")),
      h("div.custom__actions", null, saveButton,
        cancel ? h("button.btn.btn--sm", { type: "button", onclick: () => { source = undefined; cancel(); } }, t("common.cancel")) : remove),
      status,
      hint(t("settings.custom.limits")));
  };
}
