// The settings screen, as stacked cards: agent (backend, sign-in, model, effort, permissions from the backend's
// catalog), appearance (settings-look.ts), context, chat, prompts, chat folder, data, about. Every change saves at once
// through `save(patch)`; the patches are built in settings-model.ts.
import type { Appearance, BackendId, BackendStatus, Catalog, PanelHost, PanelSettings } from "../types.ts";
import type { Look } from "./appearance.ts";
import { clear, env, errMessage, h, icon, isMac, nextId } from "./dom.ts";
import { addPrompt, editPrompt, effective, modeHelp, removePrompt, setAuth, setFlag, setFolder, setPerBackend, setSlot, shortPath, shortcuts } from "./settings-model.ts";
import type { FlagKey } from "./settings-model.ts";
import { confirmAction, hint, radios, section, seg, selectField, sub, switchRow } from "./settings-parts.ts";
import { appearanceCard } from "./settings-look.ts";
import { slotLabel } from "./dom.ts";
import { BACKEND_LABEL, BACKENDS, frame } from "./views.ts";

type CatState = { state: "loading" } | { state: "ok"; catalog: Catalog } | { state: "error"; error: string };

export interface SettingsDeps {
  back(): void;
  statuses(): BackendStatus[] | null;
  refreshStatuses(): void;
  /** Something that other parts of the panel show has changed. */
  changed(): void;
  /** Applies the appearance (saved, or a draft while a slider is dragged). */
  look: Look;
  focusPrompts: boolean;
}

export function settingsView(host: PanelHost, o: SettingsDeps): { el: HTMLElement; render(): void } {
  const body = h("div.set");
  const cats = new Map<BackendId, CatState>();
  const keys: Partial<Record<BackendId, boolean>> = {};
  let keyBusy = false;
  let keyMsg = "";
  let keyDraft = ""; // typed but not saved yet: survives the page being rebuilt by a late status or catalog answer
  let saved = "";
  let timer: number | undefined;

  /** The "Saved" pill: kept across re-renders, gone after a moment. */
  const flash = (text = "Saved") => {
    const show = (t: string) => { saved = t; const e = body.querySelector(".set__saved span"); if (e) e.textContent = t; };
    show(text);
    env.win.clearTimeout(timer);
    timer = env.win.setTimeout(() => show(""), 1600);
  };
  const save = async (patch: Partial<PanelSettings>) => {
    try { await host.setSettings(patch); o.changed(); flash(); } catch (e) { keyMsg = `Couldn't save: ${errMessage(e)}`; }
    render();
  };
  // The look needs no health check or catalog reload, only the restyle.
  const saveLook = async (patch: Partial<Appearance>) => {
    try { await host.setSettings({ appearance: { ...host.getSettings().appearance, ...patch } }); o.look.apply(); flash(); } catch (e) { keyMsg = `Couldn't save: ${errMessage(e)}`; }
    render();
  };
  const appearance = appearanceCard(host, o.look, saveLook, () => render());
  const flag = (key: FlagKey, label: string, help: string) => switchRow(label, help, host.getSettings()[key], (on) => void save(setFlag(key, on)));

  /** Read the backend's catalog once; the runtime caches it, this remembers the answer for the screen. */
  function loadCatalog(b: BackendId, force = false) {
    if (!force && cats.has(b)) return;
    cats.set(b, { state: "loading" });
    host.runtime.catalog(b).then(
      (catalog) => cats.set(b, { state: "ok", catalog }),
      (e) => cats.set(b, { state: "error", error: errMessage(e) }),
    ).finally(render);
  }

  // ───────────── sections ─────────────

  function agent(s: PanelSettings): HTMLElement {
    const b = s.backend;
    const sts = o.statuses();
    const cur = sts?.find((x) => x.id === b);
    const cat = cats.get(b);
    const parts: HTMLElement[] = [];
    if (!cat || cat.state === "loading") {
      parts.push(h("div.sk-group", { "aria-busy": "true", "aria-label": `Reading ${BACKEND_LABEL[b]}'s options` }, h("div.sk.sk--field"), h("div.sk.sk--field"), h("div.sk.sk--block")));
    } else if (cat.state === "error") {
      parts.push(h("div.inlineerr", { role: "alert" }, icon("warn"),
        h("div.inlineerr__tx", null, h("div.inlineerr__t", null, `Couldn't read ${BACKEND_LABEL[b]}'s options`), h("div.inlineerr__d", null, cat.error)),
        h("button.btn.btn--sm", { type: "button", onclick: () => { loadCatalog(b, true); render(); } }, "Try again")));
    } else {
      const c = cat.catalog;
      const pick = (key: "model" | "effort", label: string, list: { id: string; name: string }[]) => {
        const def = list.find((x) => x.id === c[key])?.name ?? "";
        return selectField(label, [{ id: "", label: def ? `Default (${def})` : "Default" }, ...list.map((x) => ({ id: x.id, label: x.name }))], s[key][b], (id) => void save(setPerBackend(s, key, b, id)));
      };
      if (c.models.length) parts.push(pick("model", "Model", c.models));
      if (c.efforts.length) parts.push(pick("effort", "Effort", c.efforts));
      if (c.modes.length) {
        parts.push(h("div.field", null, h("span.field__l", null, "Permissions"),
          radios("Permission mode", c.modes.map((m) => ({ id: m.id, label: m.name, help: modeHelp(m) })), effective(s, c, "mode", b), (id) => void save(setPerBackend(s, "mode", b, id)))));
      }
    }
    return section("Agent", "Who you chat with, how it signs in, and what it may do.",
      seg("Agent", BACKENDS.map((id) => ({ id, label: BACKEND_LABEL[id], title: sts?.find((x) => x.id === id && !x.available)?.reason })), b, (id) => void save({ backend: id as BackendId })),
      hint(sts ? (cur?.available ? `${cur.label} is ready.` : `${cur?.label ?? BACKEND_LABEL[b]} isn't ready: ${cur?.reason ?? "not found"}.`) : "Looking for agents…"),
      signIn(s),
      parts.length ? h("div.fields", null, ...parts) : null,
      hint("Backend, model, effort and permissions apply to your next new chat; the pickers under the message box change the current one."));
  }

  function signIn(s: PanelSettings): HTMLElement {
    const b = s.backend;
    if (b === "pi") return sub("Sign-in", hint("pi has no subscription to sign in to. It runs on the provider keys already in your shell or in pi's own settings, so there is nothing to set up here."));
    const mode = s.auth[b] ?? "subscription";
    const st = o.statuses()?.find((x) => x.id === b);
    const input = h("input.input", { type: "password", autocomplete: "off", spellcheck: "false", placeholder: keys[b] ? "A key is saved. Paste a new one to replace it." : "Paste your API key", "aria-label": `${BACKEND_LABEL[b]} API key`, value: keyDraft, oninput: () => { keyDraft = input.value; } }) as HTMLInputElement;
    const saveKey = async () => {
      const v = input.value.trim();
      if (!v) return;
      keyBusy = true; render();
      try { await host.setApiKey(b, v); keys[b] = true; keyDraft = ""; keyMsg = "Saved to your keychain."; } catch (e) { keyMsg = `Couldn't save the key: ${errMessage(e)}`; }
      keyBusy = false; o.changed(); render();
    };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); void saveKey(); } });
    return sub("Sign-in",
      seg("Sign in with", [{ id: "subscription", label: "Subscription" }, { id: "api-key", label: "API key" }], mode, (id) => void save(setAuth(s, b, id as "subscription" | "api-key"))),
      mode === "subscription"
        ? [hint(st?.account ? `Signed in as ${st.account}.` : st?.available === false ? `Not signed in: ${st.reason ?? "unknown"}.` : "Uses the account you are signed in to in the agent's own app."),
          hint("Chats run on your subscription. An API key exported in your shell for other tools is ignored here, so it can't bill you by accident.")]
        : h("div.keyrow", null,
          h("div.keyrow__f", null, input, h("button.btn.btn--sm.btn--solid", { type: "button", disabled: keyBusy ? true : null, onclick: () => void saveKey() }, "Save")),
          keys[b] ? h("div.keyrow__s", null, h("span.keyrow__ok", null, icon("check"), "Key saved in your keychain"),
            h("button.lnk", { type: "button", onclick: async () => { try { await host.setApiKey(b, null); keys[b] = false; keyMsg = "Key removed."; } catch (e) { keyMsg = errMessage(e); } o.changed(); render(); } }, "Remove")) : null,
          hint("Stored in your OS keychain, never in Zotero's preferences."),
          keyMsg ? h("p.sec__hint", { role: "status" }, keyMsg) : null));
  }

  function prompts(s: PanelSettings): HTMLElement {
    const list = s.prompts;
    const rows = list.map((p, i) => {
      const ta = h("textarea.input.pe__text", { rows: "2", placeholder: "What to ask", "aria-label": `Prompt ${i + 1} text`, onchange: () => void save({ prompts: editPrompt(list, i, { text: (ta as HTMLTextAreaElement).value }) }) }) as HTMLTextAreaElement;
      ta.value = p.text;
      return h("div.pe", null,
        h("div.pe__row", null,
          h("input.input.input--sm", { type: "text", value: p.title, placeholder: "Title", "aria-label": `Prompt ${i + 1} title`, onchange: (e: Event) => void save({ prompts: editPrompt(list, i, { title: (e.target as HTMLInputElement).value }) }) }),
          h("select.input.input--sm.pe__slot", { "aria-label": `Prompt ${i + 1} shortcut`, onchange: (e: Event) => void save({ prompts: setSlot(list, i, Number((e.target as HTMLSelectElement).value)) }) },
            h("option", { value: "0" }, "No shortcut"), [1, 2, 3, 4].map((n) => h("option", { value: String(n), selected: p.slot === n ? true : null }, slotLabel(n)))),
          h("button.iconbtn.iconbtn--sm", { type: "button", "aria-label": `Delete prompt ${p.title || i + 1}`, title: "Delete", onclick: () => void save({ prompts: removePrompt(list, i) }) }, icon("trash"))),
        ta);
    });
    const sec = section("Custom prompts", "Shown on a new chat. A prompt with a shortcut runs from anywhere in Zotero.", ...rows,
      h("button.btn.btn--sm", { type: "button", onclick: () => void save({ prompts: addPrompt(list, nextId("p") + Date.now().toString(36)) }) }, icon("plus"), "Add a prompt"));
    sec.id = "prompts";
    return sec;
  }

  function folder(s: PanelSettings): HTMLElement {
    const path = host.about().workspace;
    const choose = async () => {
      try { const p = await host.chooseFolder(s.chatFolder || path); if (p) await save(setFolder(p)); } catch (e) { keyMsg = errMessage(e); render(); }
    };
    return section("Chat folder", "Where new chats run, and where the zotero-cli skill is installed.",
      h("div.folder", { title: path }, icon("folder"), h("span.folder__p", null, shortPath(path))),
      h("div.folder__acts", null,
        h("button.btn.btn--sm", { type: "button", onclick: () => void choose() }, "Choose\u2026"),
        h("button.btn.btn--sm", { type: "button", disabled: s.chatFolder ? null : true, onclick: () => void save(setFolder("")) }, "Use default"),
        h("button.btn.btn--sm", { type: "button", onclick: () => void host.revealWorkspace().catch((e) => { keyMsg = errMessage(e); render(); }) }, "Open")),
      hint("Every chat remembers its own folder, so changing this never breaks an old chat. If you pick an existing project, the skill is added under .claude/skills and AGENTS.md (only the block we mark is changed)."),
      keyMsg ? h("p.sec__hint", { role: "status" }, keyMsg) : null);
  }

  function data(): HTMLElement {
    return section("Data", "Your saved chats and settings.",
      h("div.datarow", null,
        confirmAction({ label: "Clear all history", ask: "Delete every saved chat?", yes: "Delete all", danger: true, run: async () => { await host.clearHistory(); o.changed(); return "History cleared."; } }),
        confirmAction({ label: "Reset settings", ask: "Reset every setting?", yes: "Reset", danger: true, run: async () => { await host.resetSettings(); cats.clear(); o.look.setImage(null); o.changed(); render(); return "Settings reset."; } })),
      hint("Reset keeps your API keys and your chats."));
  }

  function about(s: PanelSettings): HTMLElement {
    return section("About", "The version, and the keys that work everywhere.",
      h("p.about", null, `Zotero chat ${host.about().version}`),
      sub("Shortcuts", h("dl.keys", null, shortcuts(isMac(), s.enterToSend).flatMap(([what, k]) => [h("dt", null, what), h("dd", null, h("kbd", null, k))]))));
  }

  function render() {
    const s = host.getSettings();
    loadCatalog(s.backend);
    // Rebuilding the page must not drop the keyboard user's place: refocus the control that had focus.
    const focused = (body.getRootNode() as ShadowRoot).activeElement as HTMLElement | null;
    const fid = focused && body.contains(focused) ? focused.dataset.fid : undefined;
    clear(body);
    body.append(
      agent(s), appearance(),
      section("Context", "What goes with each message.",
        flag("followFocus", "Follow what I'm reading", "Attach the open paper, or the item selected in your library, to each message."),
        flag("attachSelection", "Attach selected text", "Text you select in the reader goes with the next message."),
        flag("attachAreas", "Attach selected areas as images", "Sends the picture of the area to the agent's provider."),
      ),
      section("Chat", "How you send, and what each answer shows.",
        flag("enterToSend", "Press Enter to send", s.enterToSend ? "Shift+Enter adds a line." : "Enter adds a line; Cmd/Ctrl+Enter sends."),
        flag("showThinking", "Show the agent's thinking", "A collapsed Thinking row above each answer."),
        flag("expandTools", "Expand tool steps", "Show each step's input and output without a click."),
        flag("showUsage", "Show tokens and cost", "A small line under each answer. Off by default."),
        flag("openAtStart", "Open the panel when Zotero starts", ""),
      ),
      prompts(s), folder(s), data(), about(s),
      h("div.set__saved", { role: "status", "aria-live": "polite" }, h("span", null, saved)));
    if (fid) (body.querySelector(`[data-fid="${CSS.escape(fid)}"]`) as HTMLElement | null)?.focus({ preventScroll: true });
    if (o.focusPrompts) { o.focusPrompts = false; env.win.setTimeout(() => body.querySelector("#prompts")?.scrollIntoView({ block: "start" }), 0); }
  }

  const el = frame("Settings", o.back, body);
  o.refreshStatuses();
  render();
  void Promise.all(BACKENDS.map((b) => host.hasApiKey(b).then((v) => (keys[b] = v), () => (keys[b] = false)))).then(render);
  return { el, render };
}
