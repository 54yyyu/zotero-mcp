// The composer's three pickers (model, effort, permission mode) and the menu they open above it.
// Each shows the live value; an empty list hides its button (pi has no modes, some backends no effort).
import type { ModeOption, ModelOption } from "../types.ts";
import { errMessage, h, icon, setKids } from "./dom.ts";

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
    setKids(menu, list.map((o) => h("button.menu__item", {
      type: "button", role: "menuitemradio", "aria-checked": String(o.id === cur),
      onclick: () => { this.close(); anchor.focus(); this.opts.pick(kind, o.id); },
    },
      h("span.menu__check", null, o.id === cur ? icon("check") : null),
      h("span.menu__tx", null, h("span.menu__t", null, o.name), o.description ? h("span.menu__d", null, o.description) : null))));
    ((menu.querySelector('[aria-checked="true"]') ?? menu.querySelector("button")) as HTMLElement | null)?.focus();
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
