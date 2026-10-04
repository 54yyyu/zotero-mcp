// The composer's three pickers (model, effort, permission mode) and the menu they open above it.
// Each shows the live value; an empty list hides its button (pi has no modes, some backends no effort).
import type { ModeOption, ModelOption } from "../types.ts";
import { env, errMessage, h, icon, setKids } from "./dom.ts";

export interface Choices {
  models: ModelOption[]; model?: string | undefined;
  efforts: ModeOption[]; effort?: string | undefined;
  modes: ModeOption[]; mode?: string | undefined;
}
export type PickKind = "model" | "effort" | "mode";

interface PickerOpts {
  /** The backend's choices; may start the agent or wait for its catalog. */
  load(): Promise<Choices>;
  pick(kind: PickKind, id: string): void;
  checkSetup(): void;
}

const TITLE: Record<PickKind, string> = { model: "Model", effort: "Effort: how hard the agent thinks", mode: "Permission mode: how much the agent asks before acting" };

export class Pickers {
  readonly buttons: Record<PickKind, HTMLButtonElement>;
  private host: HTMLElement;
  private opts: PickerOpts;
  private choices: Choices = { models: [], efforts: [], modes: [] };
  private menu: HTMLElement | null = null;
  private anchor: HTMLElement | null = null;

  constructor(host: HTMLElement, opts: PickerOpts) {
    this.host = host;
    this.opts = opts;
    const mk = (kind: PickKind) => h(`button.pick.pick--${kind}`, { type: "button", "aria-haspopup": "menu", "aria-expanded": "false", title: TITLE[kind], onclick: () => void this.open(kind) }) as HTMLButtonElement;
    this.buttons = { model: mk("model"), effort: mk("effort"), mode: mk("mode") };
    this.set(this.choices);
  }

  /** Show these values. */
  set(c: Choices): void {
    this.choices = c;
    const name = (list: { id: string; name: string }[], id: string | undefined, fallback: string) => list.find((x) => x.id === id)?.name ?? (id || fallback);
    const model = name(c.models, c.model, "Default model");
    const effort = name(c.efforts, c.effort, "Effort");
    const mode = name(c.modes, c.mode, "Mode");
    setKids(this.buttons.model, h("span.pick__t", null, model), icon("chevDown", "pick__chev"));
    setKids(this.buttons.effort, icon("sparkle"), h("span.pick__t", null, effort));
    setKids(this.buttons.mode, icon("shield"), h("span.pick__t", null, mode));
    this.buttons.effort.hidden = c.efforts.length === 0;
    this.buttons.mode.hidden = c.modes.length === 0;
    this.buttons.mode.className = `pick pick--mode pick--m-${c.mode ?? "default"}`;
    this.buttons.model.setAttribute("aria-label", `Model: ${model}`);
    this.buttons.effort.setAttribute("aria-label", `Effort: ${effort}`);
    this.buttons.mode.setAttribute("aria-label", `Permission mode: ${mode}`);
  }

  get isOpen(): boolean { return this.menu !== null; }

  /** True when the event path is inside the open menu or its button (an outside press closes it). */
  inside(path: EventTarget[]): boolean {
    return !!this.menu && (path.includes(this.menu) || (!!this.anchor && path.includes(this.anchor)));
  }

  close(): void {
    this.menu?.querySelector(".eff")?.dispatchEvent(new env.win.CustomEvent("zmc-close-menu"));
    this.menu?.remove();
    this.menu = null;
    this.anchor = null;
    for (const b of Object.values(this.buttons)) b.setAttribute("aria-expanded", "false");
  }

  private async open(kind: PickKind): Promise<void> {
    const anchor = this.buttons[kind];
    if (this.menu && this.anchor === anchor) { this.close(); anchor.focus(); return; }
    this.close();
    this.anchor = anchor;
    anchor.setAttribute("aria-expanded", "true");
    const menu = h(`div.menu.menu--${kind}`, { role: "menu", "aria-label": TITLE[kind].split(":")[0] }, h("div.menu__note", null, "Loading…"));
    this.menu = menu;
    this.host.appendChild(menu);
    if (kind === "effort") menu.style.left = `${Math.max(0, Math.min(anchor.offsetLeft, this.host.clientWidth - menu.offsetWidth))}px`;
    menu.addEventListener("keydown", (e) => this.key(e as KeyboardEvent, anchor));
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
    const list = kind === "model" ? c.models : kind === "effort" ? c.efforts : c.modes;
    const cur = kind === "model" ? c.model : kind === "effort" ? c.effort : c.mode;
    if (!list.length) { setKids(menu, h("div.menu__note", null, "This agent doesn't offer a choice here.")); return; }
    if (kind === "effort") { setKids(menu, this.slider(list, cur, anchor)); return; }
    setKids(menu, list.map((o) => h("button.menu__item", {
      type: "button", role: "menuitemradio", "aria-checked": String(o.id === cur),
      onclick: () => { this.close(); anchor.focus(); this.opts.pick(kind, o.id); },
    },
      h("span.menu__check", null, o.id === cur ? icon("check") : null),
      h("span.menu__tx", null, h("span.menu__t", null, o.name), o.description ? h("span.menu__d", null, o.description) : null))));
    ((menu.querySelector('[aria-checked="true"]') ?? menu.querySelector("button")) as HTMLElement | null)?.focus();
  }

  /**
   * The effort levels as a stepped slider (after Claude's): drag, click a stop, or use the arrow keys; Faster at one end,
   * Smarter at the other. A change is applied when the pointer is released, or a moment after the last key press.
   */
  private slider(levels: ModeOption[], current: string | undefined, anchor: HTMLElement): HTMLElement {
    const n = levels.length;
    let at = Math.max(0, levels.findIndex((l) => l.id === current));
    let applied = at;
    let timer = 0;
    const cur = h("span.eff__cur");
    const desc = h("div.eff__desc");
    const fill = h("span.eff__fill");
    const stops = levels.map((l) => h("span.eff__stop", { title: l.name }));
    const track = h("div.eff__track", { role: "slider", tabindex: "0", "aria-label": "Effort", "aria-valuemin": "0", "aria-valuemax": String(n - 1), style: `--n:${n}` }, h("span.eff__line"), fill, ...stops);
    const paint = () => {
      cur.textContent = levels[at]?.name ?? "";
      setKids(desc, levels[at]?.description ?? null);
      desc.hidden = !levels[at]?.description;
      track.setAttribute("aria-valuenow", String(at));
      track.setAttribute("aria-valuetext", levels[at]?.name ?? "");
      fill.style.setProperty("--at", String(at));
      stops.forEach((s, i) => s.classList.toggle("eff__stop--on", i === at));
    };
    const apply = () => {
      clearTimeout(timer);
      timer = 0;
      if (at === applied) return;
      applied = at;
      this.opts.pick("effort", levels[at]!.id);
    };
    const move = (i: number) => { at = Math.max(0, Math.min(n - 1, i)); paint(); };
    const nearest = (x: number): number => {
      const r = track.getBoundingClientRect();
      const stop = stops[0]?.getBoundingClientRect().width || 24;
      const span = Math.max(1, r.width - stop);
      return Math.round(((x - r.left - stop / 2) / span) * (n - 1));
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
    track.addEventListener("keydown", (e) => {
      const step = e.key === "ArrowRight" || e.key === "ArrowUp" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowDown" ? -1 : 0;
      if (step || e.key === "Home" || e.key === "End") {
        e.preventDefault();
        e.stopPropagation();
        move(step ? at + step : e.key === "Home" ? 0 : n - 1);
        clearTimeout(timer);
        timer = setTimeout(apply, 250) as unknown as number;
      } else if (e.key === "Enter") {
        e.preventDefault();
        apply();
        this.close();
        anchor.focus();
      }
    });
    const box = h("div.eff", { role: "group", "aria-label": "Effort" },
      h("div.eff__head", null, h("span.eff__t", null, "Effort"), cur),
      h("div.eff__ends", null, h("span", null, "Faster"), h("span", null, "Smarter")), track, desc);
    paint();
    // A level chosen with the keys and closed before the pause still lands.
    box.addEventListener("zmc-close-menu", apply);
    queueMicrotask(() => track.focus({ preventScroll: true }));
    return box;
  }

  private key(e: KeyboardEvent, anchor: HTMLElement): void {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); this.close(); anchor.focus(); return; }
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    e.preventDefault();
    const items = [...(this.menu?.querySelectorAll("button") ?? [])] as HTMLElement[];
    if (!items.length) return;
    const k = items.indexOf((this.host.getRootNode() as ShadowRoot).activeElement as HTMLElement);
    items[k < 0 ? 0 : (k + (e.key === "ArrowDown" ? 1 : items.length - 1)) % items.length]?.focus();
  }
}
