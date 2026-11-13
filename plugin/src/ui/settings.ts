// The settings screen, as stacked cards: agent (backend, sign-in, model, effort, permissions from the backend's
// catalog), appearance (settings-look.ts), context, chat, translate, skills and prompts (settings-skills.ts), chat folder, data, about. Every change saves at once
// through `save(patch)`; the patches are built in settings-model.ts.
import type { Appearance, BackendId, BackendStatus, Catalog, PanelSettings, SettingsHost } from "../types.ts";
import type { Look } from "./appearance.ts";
import { clear, env, errMessage, h, icon, isMac } from "./dom.ts";
import { LANGUAGES, effective, modeHelp, ownProviders, setAuth, setFlag, setFolder, setPerBackend, shortPath, shortcuts } from "./settings-model.ts";
import type { FlagKey } from "./settings-model.ts";
import { confirmAction, hint, radios, row, section, seg, select, selectField, sub, switchRow } from "./settings-parts.ts";
import { appearanceCard } from "./settings-look.ts";
import { skillsCard } from "./settings-skills.ts";
import { customAgentEditor } from "./settings-custom.ts";
import { BACKEND_LABEL, BACKENDS, frame } from "./views.ts";
import { UI_LANGUAGES, t, uiLanguageFor } from "../i18n.ts";

type CatState = { state: "loading" } | { state: "ok"; catalog: Catalog } | { state: "error"; error: string };

export interface SettingsDeps {
  /** The panel's way back to the chat; Zotero's Settings pane has none (and no header: Zotero shows its own title). */
  back?: () => void;
  statuses(): BackendStatus[] | null;
  refreshStatuses(): void;
  /** Something that other parts of the panel show has changed. */
  changed(): void;
  /** Applies the appearance (saved, or a draft while a slider is dragged). */
  look: Look;
  /** "Create with the agent": a new chat starting with /create-skill (the panel only; Zotero's Settings pane has no chat). */
  createSkill?: () => void;
  /**
   * Whether reading this backend's catalog costs nothing now (the open panel already read it). Reading one starts the agent
   * for a moment, so where this says no (Zotero's Settings pane) it waits for a button. Omitted: always read.
   */
  catalogReady?: (b: BackendId) => boolean;
}

export function settingsView(host: SettingsHost, o: SettingsDeps): { el: HTMLElement; render(): void } {
  const body = h("div.set");
  const cats = new Map<BackendId, CatState>();
  const keys: Partial<Record<BackendId, boolean>> = {};
  let keyBusy = false;
  let keyMsg = "";
  let keyDraft = ""; // typed but not saved yet: survives the page being rebuilt by a late status or catalog answer
  let saved = "";
  let timer: number | undefined;
  let customSource = JSON.stringify(host.getSettings().customAgent);

  /** The "Saved" pill: kept across re-renders, gone after a moment. */
  const flash = (text = t("settings.saved")) => {
    const show = (t: string) => { saved = t; const e = body.querySelector(".set__saved span"); if (e) e.textContent = t; };
    show(text);
    env.win.clearTimeout(timer);
    timer = env.win.setTimeout(() => show(""), 1600);
  };
  const save = async (patch: Partial<PanelSettings>) => {
    try { await host.setSettings(patch); o.changed(); if (patch.customAgent) o.refreshStatuses(); flash(); } catch (e) {
      if (patch.customAgent) throw e;
      keyMsg = t("settings.saveFailed", { error: errMessage(e) });
    }
    render();
  };
  // The look needs no health check or catalog reload, only the restyle.
  const saveLook = async (patch: Partial<Appearance>) => {
    try { await host.setSettings({ appearance: { ...host.getSettings().appearance, ...patch } }); o.look.apply(); flash(); } catch (e) { keyMsg = t("settings.saveFailed", { error: errMessage(e) }); }
    render();
  };
  /** The interface language: Zotero's (named, so the user sees what Automatic means), or one of ours. */
  const language = () => {
    const auto = UI_LANGUAGES.find((l) => l.id === uiLanguageFor(host.locale()))!.label;
    return row(t("settings.language"), select(t("settings.language"),
      [{ id: "", label: t("settings.language.auto", { name: auto }) }, ...UI_LANGUAGES.map((l) => ({ id: l.id, label: l.beta ? t("settings.language.beta", { name: l.label }) : l.label }))],
      host.getSettings().uiLanguage, (id) => void save({ uiLanguage: id })));
  };
  const appearance = appearanceCard(host, o.look, saveLook, () => render(), language);
  const skills = skillsCard(host, { save, rerender: () => render(), ...(o.createSkill ? { createSkill: o.createSkill } : {}) });
  const customEditor = customAgentEditor(() => host.getSettings(), save);
  const flag = (key: FlagKey, label: string, help: string) => switchRow(label, help, host.getSettings()[key], (on) => void save(setFlag(key, on)));

  /** Read the backend's catalog once; the runtime caches it, this remembers the answer for the screen. */
  function loadCatalog(b: BackendId, force = false) {
    if (!force && cats.has(b)) return;
    cats.set(b, { state: "loading" });
    const source = JSON.stringify(host.getSettings().customAgent);
    host.runtime.catalog(b).then(
      (catalog) => { if (b !== "custom" || source === JSON.stringify(host.getSettings().customAgent)) cats.set(b, { state: "ok", catalog }); },
      (e) => { if (b !== "custom" || source === JSON.stringify(host.getSettings().customAgent)) cats.set(b, { state: "error", error: errMessage(e) }); },
    ).finally(render);
  }

  // ───────────── sections ─────────────

  function agent(s: PanelSettings): HTMLElement {
    const b = s.backend;
    const sts = o.statuses();
    const cur = sts?.find((x) => x.id === b);
    const cat = cats.get(b);
    const parts: HTMLElement[] = [];
    const name = (b === "custom" && s.customAgent?.label) || BACKEND_LABEL[b];
    if (!cat && (b === "custom" || (o.catalogReady && !o.catalogReady(b)))) {
      parts.push(h("div.field", null, hint(t("settings.agent.readHint", { agent: name })),
        h("div", null, h("button.btn.btn--sm", { type: "button", onclick: () => loadCatalog(b, true) }, t("settings.agent.show", { agent: name })))));
    } else if (!cat || cat.state === "loading") {
      parts.push(h("div.sk-group", { "aria-busy": "true", "aria-label": t("settings.agent.reading", { agent: name }) }, h("div.sk.sk--field"), h("div.sk.sk--field"), h("div.sk.sk--block")));
    } else if (cat.state === "error") {
      parts.push(h("div.inlineerr", { role: "alert" }, icon("warn"),
        h("div.inlineerr__tx", null, h("div.inlineerr__t", null, t("settings.agent.readFailed", { agent: name })), h("div.inlineerr__d", null, cat.error)),
        h("button.btn.btn--sm", { type: "button", onclick: () => { loadCatalog(b, true); render(); } }, t("common.tryAgain"))));
    } else {
      const c = cat.catalog;
      // The agent's own default is the Default option (as in the composer's dropdown), not listed twice; a choice saved
      // before that still shows.
      const pick = (key: "model" | "effort", label: string, list: { id: string; name: string }[]) => {
        const def = list.find((x) => x.id === c[key]);
        const rest = list.filter((x) => x !== def || s[key][b] === x.id);
        return selectField(label, [{ id: "", label: def ? t("settings.defaultNamed", { name: def.name }) : t("common.default") }, ...rest.map((x) => ({ id: x.id, label: x.name }))], s[key][b] ?? "", (id) => void save(setPerBackend(s, key, b, id)));
      };
      if (c.models.length) parts.push(pick("model", t("settings.model"), c.models));
      if (c.efforts.length) parts.push(pick("effort", t("settings.effort"), c.efforts));
      if (c.modes.length) {
        parts.push(h("div.field", null, h("span.field__l", null, b === "custom" ? t("settings.agentMode") : t("settings.permissions")),
          radios(b === "custom" ? t("settings.agentMode") : t("settings.permissionMode"), c.modes.map((m) => ({ id: m.id, label: m.name, help: b === "custom" ? m.description ?? "" : modeHelp(m) })), effective(s, c, "mode", b), (id) => void save(setPerBackend(s, "mode", b, id)))));
      }
    }
    return section(t("common.agent"), t("settings.agent.desc"),
      seg(t("common.agent"), BACKENDS.map((id) => ({ id, label: BACKEND_LABEL[id], title: sts?.find((x) => x.id === id && !x.available)?.reason })), b, (id) => void save({ backend: id as BackendId })),
      hint(sts ? (cur?.available ? t("settings.agent.ready", { agent: cur.label }) : t("settings.agent.notReady", { agent: cur?.label ?? BACKEND_LABEL[b], reason: cur?.reason ?? t("health.notFound") })) : t("settings.agent.looking")),
      b === "custom" ? customEditor() : signIn(s),
      parts.length ? h("div.fields", null, ...parts) : null,
      hint(t("settings.agent.nextChat")));
  }

  function signIn(s: PanelSettings): HTMLElement {
    const b = s.backend;
    if (ownProviders(b)) return sub(t("settings.signIn"), hint(t(b === "pi" ? "settings.signIn.pi" : "settings.signIn.opencode")));
    const mode = s.auth[b] ?? "subscription";
    const st = o.statuses()?.find((x) => x.id === b);
    const input = h("input.input", { type: "password", autocomplete: "off", spellcheck: "false", placeholder: keys[b] ? t("settings.key.replace") : t("settings.key.paste"), "aria-label": t("settings.key.label", { agent: BACKEND_LABEL[b] }), value: keyDraft, oninput: () => { keyDraft = input.value; } }) as HTMLInputElement;
    const saveKey = async () => {
      const v = input.value.trim();
      if (!v) return;
      keyBusy = true; render();
      try { await host.setApiKey(b, v); keys[b] = true; keyDraft = ""; keyMsg = t("settings.key.saved"); } catch (e) { keyMsg = t("settings.key.saveFailed", { error: errMessage(e) }); }
      keyBusy = false; o.changed(); render();
    };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); void saveKey(); } });
    return sub(t("settings.signIn"),
      seg(t("settings.signInWith"), [{ id: "subscription", label: t("settings.subscription") }, { id: "api-key", label: t("settings.apiKey") }], mode, (id) => void save(setAuth(s, b, id as "subscription" | "api-key"))),
      mode === "subscription"
        ? [hint(st?.account ? t("settings.signedInAs", { account: st.account }) : st?.available === false ? t("settings.notSignedIn", { reason: st.reason ?? t("settings.unknown") }) : t("settings.usesAccount")),
          hint(t("settings.subscriptionHint"))]
        : h("div.keyrow", null,
          h("div.keyrow__f", null, input, h("button.btn.btn--sm.btn--solid", { type: "button", disabled: keyBusy ? true : null, onclick: () => void saveKey() }, t("common.save"))),
          keys[b] ? h("div.keyrow__s", null, h("span.keyrow__ok", null, icon("check"), t("settings.key.inKeychain")),
            h("button.lnk", { type: "button", onclick: async () => { try { await host.setApiKey(b, null); keys[b] = false; keyMsg = t("settings.key.removed"); } catch (e) { keyMsg = errMessage(e); } o.changed(); render(); } }, t("common.remove"))) : null,
          hint(t("settings.key.where")),
          keyMsg ? h("p.sec__hint", { role: "status" }, keyMsg) : null));
  }

  /** The reader's Translate button. Its model list is the catalog this screen already has; none known, only the default. */
  function translate(s: PanelSettings): HTMLElement {
    const b = s.backend;
    if (b === "custom") return section(t("settings.translate"), t("agent.custom.noTranslate"));
    const cat = cats.get(b);
    const models = cat?.state === "ok" ? cat.catalog.models : [];
    const chosen = s.translateModel[b] ?? "";
    const list = chosen && !models.some((m) => m.id === chosen) ? [...models, { id: chosen, name: chosen }] : models;
    return section(t("settings.translate"), t("settings.translate.desc"),
      flag("translate", t("settings.translate.show"), t("settings.translate.showHint")),
      h("div.fields", null,
        selectField(t("settings.translate.into"), LANGUAGES.map((l) => ({ id: l.id, label: l.label })), s.translateTo, (id) => void save({ translateTo: id })),
        selectField(t("settings.model"), [{ id: "", label: t("settings.translate.fastest") }, ...list.map((m) => ({ id: m.id, label: m.name }))], chosen, (id) => void save(setPerBackend(s, "translateModel", b, id)))),
      hint(t("settings.translate.runsOn", { agent: BACKEND_LABEL[b] })));
  }

  function folder(s: PanelSettings): HTMLElement {
    const path = host.about().workspace;
    const choose = async () => {
      try { const p = await host.chooseFolder(s.chatFolder || path); if (p) await save(setFolder(p)); } catch (e) { keyMsg = errMessage(e); render(); }
    };
    return section(t("settings.folder"), t("settings.folder.desc"),
      h("div.folder", { title: path }, icon("folder"), h("span.folder__p", null, shortPath(path))),
      h("div.folder__acts", null,
        h("button.btn.btn--sm", { type: "button", onclick: () => void choose() }, t("settings.folder.choose")),
        h("button.btn.btn--sm", { type: "button", disabled: s.chatFolder ? null : true, onclick: () => void save(setFolder("")) }, t("settings.folder.useDefault")),
        h("button.btn.btn--sm", { type: "button", onclick: () => void host.revealWorkspace().catch((e) => { keyMsg = errMessage(e); render(); }) }, t("common.open"))),
      hint(t("settings.folder.hint")),
      keyMsg ? h("p.sec__hint", { role: "status" }, keyMsg) : null);
  }

  function data(): HTMLElement {
    return section(t("settings.data"), t("settings.data.desc"),
      h("div.datarow", null,
        confirmAction({ label: t("settings.data.clear"), ask: t("settings.data.clearAsk"), yes: t("settings.data.clearYes"), danger: true, run: async () => { await host.clearHistory(); o.changed(); return t("settings.data.cleared"); } }),
        confirmAction({ label: t("settings.data.reset"), ask: t("settings.data.resetAsk"), yes: t("settings.data.resetYes"), danger: true, run: async () => { await host.resetSettings(); cats.clear(); o.look.setImage(null); o.changed(); render(); return t("settings.data.wasReset"); } })),
      hint(t("settings.data.hint")));
  }

  function about(s: PanelSettings): HTMLElement {
    return section(t("settings.about"), t("settings.about.desc"),
      h("p.about", null, `Zotero Agent ${host.about().version}`),
      sub(t("settings.shortcuts"), h("dl.keys", null, shortcuts(isMac(), s.enterToSend).flatMap(([what, k]) => [h("dt", null, what), h("dd", null, h("kbd", null, k))]))));
  }

  function render() {
    const s = host.getSettings();
    const source = JSON.stringify(s.customAgent);
    if (source !== customSource) { customSource = source; cats.delete("custom"); }
    if (s.backend !== "custom" && (!o.catalogReady || o.catalogReady(s.backend))) loadCatalog(s.backend);
    // Rebuilding the page must not drop the keyboard user's place: refocus the control that had focus.
    const focused = (body.getRootNode() as ShadowRoot).activeElement as HTMLElement | null;
    const fid = focused && body.contains(focused) ? focused.dataset.fid : undefined;
    // ...and its scroll position (WebKit drops it when the page is emptied and refilled).
    const scroller = body.parentElement;
    const top = scroller?.scrollTop ?? 0;
    clear(body);
    body.append(
      agent(s), appearance(),
      section(t("settings.context"), t("settings.context.desc"),
        flag("followFocus", t("settings.context.follow"), t("settings.context.followHint")),
        flag("attachSelection", t("settings.context.selection"), t("settings.context.selectionHint")),
        flag("attachAreas", t("settings.context.areas"), t("settings.context.areasHint")),
      ),
      section(t("settings.chat"), t("settings.chat.desc"),
        flag("enterToSend", t("settings.chat.enter"), s.enterToSend ? t("settings.chat.enterOn") : t("settings.chat.enterOff")),
        flag("showThinking", t("settings.chat.thinking"), t("settings.chat.thinkingHint")),
        flag("expandTools", t("settings.chat.tools"), t("settings.chat.toolsHint")),
        flag("showUsage", t("settings.chat.usage"), t("settings.chat.usageHint")),
        flag("openAtStart", t("settings.chat.openAtStart"), ""),
        h("div.fields", null,
          selectField(t("settings.chat.titleSource"), [
            { id: "agent", label: t("settings.chat.titleAgent") },
            { id: "paper", label: t("settings.chat.titlePaper") },
            { id: "question", label: t("settings.chat.titleQuestion") },
          ], s.titleSource, (id) => void save({ titleSource: id as PanelSettings["titleSource"] })),
          hint(t("settings.chat.titleSourceHint")),
        ),
      ),
      translate(s), skills(), folder(s), data(), about(s),
      h("div.set__saved", { role: "status", "aria-live": "polite" }, h("span", null, saved)));
    if (scroller) scroller.scrollTop = top;
    if (fid) (body.querySelector(`[data-fid="${CSS.escape(fid)}"]`) as HTMLElement | null)?.focus({ preventScroll: true });
  }

  const el = o.back ? frame(t("settings.title"), o.back, body) : body;
  o.refreshStatuses();
  render();
  void Promise.all(BACKENDS.filter((b) => b !== "custom").map((b) => host.hasApiKey(b).then((v) => (keys[b] = v), () => (keys[b] = false)))).then(render);
  return { el, render };
}
