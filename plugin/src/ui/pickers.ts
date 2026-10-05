// The composer's two pickers. The model button shows the model and, softly, the effort level; it opens one dropdown
// right above itself: the agent (Claude Code, Codex, pi, each with its status dot), that agent's models in its own
// catalog's order (the first few, the rest under "More models"), and an effort slider. The mode button opens the
// permission modes. An empty list hides its part (pi has no modes, some backends no effort).
import type { BackendId, ModeOption, ModelOption } from "../types.ts";
import { env, errMessage, h, icon, setKids } from "./dom.ts";

export interface AgentChoice { id: BackendId; label: string; available?: boolean | undefined; reason?: string | undefined }
export interface Choices {
  backend: BackendId;
  agents: AgentChoice[];
  models: ModelOption[]; model?: string | undefined;
  efforts: ModeOption[]; effort?: string | undefined;
  /** The agent's own default level (its catalog's), marked Recommended. */
  defaultEffort?: string | undefined;
  modes: ModeOption[]; mode?: string | undefined;
}
export type PickKind = "model" | "effort" | "mode";

interface PickerOpts {
  /** The backend's choices; may start the agent or wait for its catalog. */
  load(): Promise<Choices>;
  pick(kind: PickKind, id: string): void;
  /** New chats use this agent from now on; the current chat is replaced by a new one if it has messages. */
  switchAgent(id: BackendId): Promise<void>;
  /** The chat has messages, so switching the agent asks first. */
  hasMessages(): boolean;
  checkSetup(): void;
}

/** Models listed before "More models". The current one is listed too, wherever it sits in the catalog. */
const TOP = 4;
const MODE_TITLE = "Permission mode: how much the agent asks before acting";
const nameOf = (list: { id: string; name: string }[], id: string | undefined, fallback: string) => list.find((x) => x.id === id)?.name ?? (id || fallback);

export class Pickers {
  readonly buttons: { model: HTMLButtonElement; mode: HTMLButtonElement };
  private host: HTMLElement;
  private opts: PickerOpts;
  private choices: Choices = { backend: "claude-code", agents: [], models: [], efforts: [], modes: [] };
  private menu: HTMLElement | null = null;
  private anchor: HTMLElement | null = null;
  /** The open dropdown's parts: the agent row, a line for a question or a reason, the models and effort. */
  private agentsEl: HTMLElement | null = null;
  private noteEl: HTMLElement | null = null;
  private bodyEl: HTMLElement | null = null;
  /** Lands an effort chosen with the keys when the dropdown closes before the pause. */
  private flush: (() => void) | null = null;
  private gen = 0;

  constructor(host: HTMLElement, opts: PickerOpts) {
    this.host = host;
    this.opts = opts;
    this.buttons = {
      model: h("button.pick.pick--model", { type: "button", "aria-haspopup": "dialog", "aria-expanded": "false", title: "Agent, model and effort", onclick: () => void this.openModel() }) as HTMLButtonElement,
      mode: h("button.pick.pick--mode", { type: "button", "aria-haspopup": "menu", "aria-expanded": "false", title: MODE_TITLE, onclick: () => void this.openMode() }) as HTMLButtonElement,
    };
    this.set(this.choices);
  }

  /** Show these values (and the agents' status in an open dropdown). */
  set(c: Choices): void {
    this.choices = c;
    const model = nameOf(c.models, c.model, "Default model");
    const effort = c.efforts.length && c.effort ? nameOf(c.efforts, c.effort, "") : "";
    const mode = nameOf(c.modes, c.mode, "Mode");
    setKids(this.buttons.model, h("span.pick__t", null, model), effort ? h("span.pick__e", null, effort) : null, icon("chevDown", "pick__chev"));
    setKids(this.buttons.mode, icon("shield"), h("span.pick__t", null, mode));
    this.buttons.mode.hidden = c.modes.length === 0;
    this.buttons.mode.className = `pick pick--mode pick--m-${c.mode ?? "default"}`;
    this.buttons.model.setAttribute("aria-label", `Model: ${model}${effort ? `, effort ${effort}` : ""}. Change the agent, model or effort`);
    this.buttons.mode.setAttribute("aria-label", `Permission mode: ${mode}`);
    if (this.agentsEl) this.paintAgents();
  }

  get isOpen(): boolean { return this.menu !== null; }

  /** True when the event path is inside the open menu or its button (an outside press closes it). */
  inside(path: EventTarget[]): boolean {
    return !!this.menu && (path.includes(this.menu) || (!!this.anchor && path.includes(this.anchor)));
  }

  close(): void {
    this.flush?.();
    this.flush = null;
    this.menu?.remove();
    this.menu = this.anchor = this.agentsEl = this.noteEl = this.bodyEl = null;
    this.gen++;
    for (const b of Object.values(this.buttons)) b.setAttribute("aria-expanded", "false");
  }

  /** A menu above `anchor`, or null when `anchor`'s own menu was open (the press closed it). */
  private mount(anchor: HTMLElement, sel: string, attrs: Record<string, string>): HTMLElement | null {
    if (this.menu && this.anchor === anchor) { this.close(); anchor.focus(); return null; }
    this.close();
    this.anchor = anchor;
    anchor.setAttribute("aria-expanded", "true");
    this.menu = h(sel, attrs);
    this.host.appendChild(this.menu);
    return this.menu;
  }

  /** The menu opens straight above its own button, like a dropdown; it grows upward and never past the top. */
  private place(): void {
    const { menu, anchor } = this;
    if (!menu || !anchor) return;
    menu.style.bottom = `${this.host.offsetHeight - anchor.offsetTop + 4}px`;
    menu.style.maxHeight = `${Math.max(120, Math.min(440, anchor.getBoundingClientRect().top - 12))}px`;
    if (anchor === this.buttons.model) menu.style.left = `${Math.max(0, Math.min(anchor.offsetLeft, this.host.clientWidth - menu.offsetWidth))}px`;
  }

  private focused(): HTMLElement | null { return (this.host.getRootNode() as ShadowRoot).activeElement as HTMLElement | null; }

  // ───────────────────────────── agent, model and effort ─────────────────────────────

  private async openModel(): Promise<void> {
    const menu = this.mount(this.buttons.model, "div.menu.mdd", { role: "dialog", "aria-label": "Agent, model and effort" });
    if (!menu) return;
    this.agentsEl = h("div.mdd__agents", { role: "radiogroup", "aria-label": "Agent" });
    this.noteEl = h("div.mdd__note", { hidden: true, role: "status" });
    this.bodyEl = h("div.mdd__body");
    menu.append(this.agentsEl, this.noteEl, this.bodyEl);
    menu.addEventListener("keydown", (e) => this.keyModel(e as KeyboardEvent));
    this.paintAgents();
    await this.fill(true);
  }

  private paintAgents(): void {
    const el = this.agentsEl;
    if (!el) return;
    const c = this.choices;
    const had = el.contains(this.focused()) ? this.focused()?.dataset.id : undefined;
    setKids(el, c.agents.map((a) => {
      const on = a.id === c.backend;
      return h(`button.mdd__agent${on ? ".mdd__agent--on" : ""}`, {
        type: "button", role: "radio", "aria-checked": String(on), tabindex: on ? "0" : "-1",
        title: a.available === false ? `${a.label} isn't ready: ${a.reason ?? "not found"}` : a.available ? `${a.label} is ready` : a.label,
        dataset: { id: a.id, state: a.available === undefined ? "" : a.available ? "ok" : "bad", stop: "" },
        onclick: () => this.chooseAgent(a),
      }, h("span.mdd__dot", { "aria-hidden": "true" }), h("span.mdd__an", null, a.label));
    }));
    if (had) (el.querySelector(`[data-id="${had}"]`) as HTMLElement | null)?.focus();
  }

  /** A question or a reason under the agents, with its buttons; null clears it. */
  private note(text: string | null, actions: [string, () => void, boolean?][] = []): void {
    const el = this.noteEl;
    if (!el) return;
    el.hidden = !text;
    setKids(el, text ? [h("span.mdd__q", null, text), h("span.mdd__acts", null, actions.map(([label, run, solid]) =>
      h(`button.btn.btn--sm${solid ? ".btn--solid" : ".btn--quiet"}`, { type: "button", dataset: { stop: "" }, onclick: run }, label)))] : null);
    this.place();
    if (actions.length) (el.querySelector("button") as HTMLElement | null)?.focus();
  }

  private focusAgent(id: BackendId): void { (this.agentsEl?.querySelector(`[data-id="${id}"]`) as HTMLElement | null)?.focus(); }

  /** Another agent: at once on an empty chat; a chat with messages is never cut without asking. */
  private chooseAgent(a: AgentChoice): void {
    if (a.id === this.choices.backend) { this.note(null); return; }
    if (a.available === false) {
      this.note(`${a.label} isn't ready: ${a.reason ?? "not found"}.`, [["Check setup", () => { this.close(); this.opts.checkSetup(); }]]);
      return;
    }
    if (!this.opts.hasMessages()) { void this.switchTo(a.id); return; }
    this.note(`Switching starts a new chat with ${a.label}.`, [
      ["Start new chat", () => void this.switchTo(a.id), true],
      ["Cancel", () => { this.note(null); this.focusAgent(this.choices.backend); }],
    ]);
  }

  private async switchTo(id: BackendId): Promise<void> {
    const menu = this.menu;
    this.note(null);
    this.set({ ...this.choices, backend: id });
    this.focusAgent(id);
    if (this.bodyEl) setKids(this.bodyEl, h("div.menu__note", null, "Loading…"));
    try {
      await this.opts.switchAgent(id);
    } catch (e) {
      if (this.menu === menu && this.bodyEl) setKids(this.bodyEl, h("div.menu__note.menu__note--bad", null, `Couldn't switch: ${errMessage(e)}`));
      return;
    }
    if (this.menu !== menu) return;
    await this.fill(false);
    this.focusAgent(id); // a new chat moved the focus to the message box
  }

  /** The models and effort of the chosen agent: Loading, then the lists, or what went wrong and a way to look. */
  private async fill(focusCurrent: boolean): Promise<void> {
    const { menu, bodyEl: body } = this;
    if (!menu || !body) return;
    const gen = ++this.gen;
    this.flush?.();
    this.flush = null;
    setKids(body, h("div.menu__note", null, "Loading…"));
    this.place();
    if (focusCurrent) this.focusAgent(this.choices.backend);
    let c: Choices;
    try {
      c = await this.opts.load();
    } catch (e) {
      if (gen !== this.gen) return;
      setKids(body, h("div.menu__note.menu__note--bad", null, `Couldn't load the choices: ${errMessage(e)}`),
        h("button.menu__item", { type: "button", dataset: { stop: "" }, onclick: () => { this.close(); this.opts.checkSetup(); } }, "Check setup"));
      this.place();
      return;
    }
    if (gen !== this.gen) return;
    this.set(c);
    setKids(body, this.models(c), c.efforts.length ? this.effort(c) : null,
      !c.models.length && !c.efforts.length ? h("div.menu__note", null, "This agent doesn't offer a choice here.") : null);
    this.place();
    if (focusCurrent) (body.querySelector('[aria-checked="true"]') as HTMLElement | null)?.focus();
  }

  private models(c: Choices): HTMLElement | null {
    if (!c.models.length) return null;
    const at = c.models.findIndex((m) => m.id === c.model);
    const shown = (i: number) => i < TOP || i === at;
    const item = (o: ModelOption) => h("button.menu__item", {
      type: "button", role: "menuitemradio", "aria-checked": String(o.id === c.model), dataset: { stop: "" },
      onclick: () => { this.close(); this.buttons.model.focus(); this.opts.pick("model", o.id); },
    },
      h("span.menu__check", null, o.id === c.model ? icon("check") : null),
      h("span.menu__tx", null, h("span.menu__t", null, o.name), o.description ? h("span.menu__d", null, o.description) : null));
    const list = h("div.mdd__models", { role: "menu", "aria-label": "Models" }, c.models.filter((_, i) => shown(i)).map(item));
    const rest = c.models.filter((_, i) => !shown(i));
    if (rest.length) {
      const extra = h("div.mdd__extra", { hidden: true }, rest.map(item));
      const more = h("button.menu__item.mdd__more", {
        type: "button", role: "menuitem", "aria-expanded": "false", dataset: { stop: "" },
        onclick: () => {
          extra.hidden = !extra.hidden;
          more.setAttribute("aria-expanded", String(!extra.hidden));
          this.place();
          if (!extra.hidden) extra.scrollIntoView?.({ block: "nearest" });
        },
      }, h("span.mdd__chev", null, icon("chevDown")), h("span.menu__tx", null, h("span.menu__t", null, "More models")), h("span.mdd__n", null, String(rest.length)));
      list.append(more, extra);
    }
    return list;
  }

  /**
   * The effort levels as a stepped slider (after Claude's): drag, click a stop, or Left and Right; the level is named on
   * the right. A change is applied when the pointer is released, or a moment after the last key press.
   */
  private effort(c: Choices): HTMLElement {
    const levels = c.efforts;
    const n = levels.length;
    const idx = (id: string | undefined) => levels.findIndex((l) => l.id === id);
    let at = Math.max(0, idx(c.effort) >= 0 ? idx(c.effort) : idx(c.defaultEffort));
    let applied = at;
    let timer = 0;
    const cur = h("span.eff__cur");
    const desc = h("div.eff__desc");
    const fill = h("span.eff__fill");
    const stops = levels.map((l) => h(`span.eff__stop${l.id === c.defaultEffort ? ".eff__stop--rec" : ""}`, { title: l.id === c.defaultEffort ? `${l.name} (recommended)` : l.name }));
    const track = h("div.eff__track", { role: "slider", tabindex: "0", "aria-label": "Effort", "aria-valuemin": "0", "aria-valuemax": String(n - 1), style: `--n:${n}`, dataset: { stop: "" } }, h("span.eff__line"), fill, ...stops);
    const paint = () => {
      const l = levels[at];
      cur.textContent = l?.name ?? "";
      const rec = !!l && l.id === c.defaultEffort;
      setKids(desc, rec ? h("span.eff__rec", null, "Recommended") : null, l?.description ?? null);
      desc.hidden = !rec && !l?.description;
      track.setAttribute("aria-valuenow", String(at));
      track.setAttribute("aria-valuetext", `${l?.name ?? ""}${rec ? ", recommended" : ""}`);
      fill.style.setProperty("--at", String(at));
      stops.forEach((s, i) => s.classList.toggle("eff__stop--on", i === at));
    };
    const apply = () => {
      env.win.clearTimeout(timer);
      timer = 0;
      if (at === applied) return;
      applied = at;
      this.opts.pick("effort", levels[at]!.id);
    };
    const move = (i: number) => { at = Math.max(0, Math.min(n - 1, i)); paint(); };
    const nearest = (x: number): number => {
      const r = track.getBoundingClientRect();
      const stop = stops[0]?.getBoundingClientRect().width || 20;
      return Math.round(((x - r.left - stop / 2) / Math.max(1, r.width - stop)) * (n - 1));
    };
    let dragging = false;
    track.addEventListener("pointerdown", (e) => {
      dragging = true;
      track.setPointerCapture?.(e.pointerId);
      move(nearest(e.clientX));
      track.focus({ preventScroll: true });
      e.preventDefault();
    });
    track.addEventListener("pointermove", (e) => { if (dragging) move(nearest(e.clientX)); });
    const release = () => { if (dragging) { dragging = false; apply(); } };
    track.addEventListener("pointerup", release);
    track.addEventListener("pointercancel", release);
    // Left and Right step (Up and Down move through the dropdown); Enter applies and closes.
    track.addEventListener("keydown", (e) => {
      const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (step || e.key === "Home" || e.key === "End") {
        e.preventDefault();
        e.stopPropagation();
        move(step ? at + step : e.key === "Home" ? 0 : n - 1);
        env.win.clearTimeout(timer);
        timer = env.win.setTimeout(apply, 250);
      } else if (e.key === "Enter") {
        e.preventDefault();
        e.stopPropagation();
        apply();
        this.close();
        this.buttons.model.focus();
      }
    });
    this.flush = apply;
    paint();
    return h("div.mdd__eff", { role: "group", "aria-label": "Effort" }, h("span.mdd__el", null, "Effort"), track, cur, desc);
  }

  private keyModel(e: KeyboardEvent): void {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); this.close(); this.buttons.model.focus(); return; }
    const active = this.focused();
    if ((e.key === "ArrowLeft" || e.key === "ArrowRight") && active?.classList.contains("mdd__agent")) {
      // The agents are one stop: Left and Right move between them, Enter or Space chooses.
      e.preventDefault();
      const tabs = [...(this.agentsEl?.querySelectorAll("button") ?? [])] as HTMLElement[];
      const next = tabs[(tabs.indexOf(active) + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      for (const t of tabs) t.tabIndex = t === next ? 0 : -1;
      next?.focus();
      return;
    }
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    e.preventDefault();
    const stops = [...(this.menu?.querySelectorAll("[data-stop]") ?? [])].filter((x) => (x as HTMLElement).tabIndex >= 0 && x.getClientRects().length > 0) as HTMLElement[];
    const k = stops.indexOf(active as HTMLElement);
    stops[k < 0 ? 0 : (k + (e.key === "ArrowDown" ? 1 : stops.length - 1)) % stops.length]?.focus();
  }

  // ───────────────────────────── permission mode ─────────────────────────────

  private async openMode(): Promise<void> {
    const anchor = this.buttons.mode;
    const menu = this.mount(anchor, "div.menu.menu--mode", { role: "menu", "aria-label": "Permission mode" });
    if (!menu) return;
    menu.addEventListener("keydown", (e) => this.keyMode(e as KeyboardEvent));
    setKids(menu, h("div.menu__note", null, "Loading…"));
    this.place();
    let c: Choices;
    try {
      c = await this.opts.load();
    } catch (e) {
      if (this.menu !== menu) return;
      setKids(menu, h("div.menu__note.menu__note--bad", null, `Couldn't load the choices: ${errMessage(e)}`),
        h("button.menu__item", { type: "button", onclick: () => { this.close(); this.opts.checkSetup(); } }, "Check setup"));
      return;
    }
    if (this.menu !== menu) return;
    this.set(c);
    if (!c.modes.length) { setKids(menu, h("div.menu__note", null, "This agent doesn't offer a choice here.")); return; }
    setKids(menu, c.modes.map((o) => h("button.menu__item", {
      type: "button", role: "menuitemradio", "aria-checked": String(o.id === c.mode),
      onclick: () => { this.close(); anchor.focus(); this.opts.pick("mode", o.id); },
    },
      h("span.menu__check", null, o.id === c.mode ? icon("check") : null),
      h("span.menu__tx", null, h("span.menu__t", null, o.name), o.description ? h("span.menu__d", null, o.description) : null))));
    this.place();
    ((menu.querySelector('[aria-checked="true"]') ?? menu.querySelector("button")) as HTMLElement | null)?.focus();
  }

  private keyMode(e: KeyboardEvent): void {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); this.close(); this.buttons.mode.focus(); return; }
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    e.preventDefault();
    const items = [...(this.menu?.querySelectorAll("button") ?? [])] as HTMLElement[];
    const k = items.indexOf(this.focused() as HTMLElement);
    items[k < 0 ? 0 : (k + (e.key === "ArrowDown" ? 1 : items.length - 1)) % items.length]?.focus();
  }
}
