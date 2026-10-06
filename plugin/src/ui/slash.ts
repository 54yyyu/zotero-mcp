// The `/` menu: typing `/` at the start of the message lists the user's skills, prompts and the few agent commands that
// make sense here, filtered as they type (ui/skills-model.ts rankItems). Same surface as the `@` popup (`.pop`).
import { clear, h, icon, setKids } from "./dom.ts";
import type { IconName } from "./dom.ts";
import { rankItems } from "./skills-model.ts";

export interface SlashItem {
  group: "Skills" | "Prompts" | "Agent";
  id: string;
  label: string;
  detail: string;
  /** Put this in the message box (a skill's `/name `, a prompt's text); otherwise `run` it at once. */
  insert?: string;
  run?: () => void;
}

const GROUPS: SlashItem["group"][] = ["Skills", "Prompts", "Agent"];
const ICON: Record<SlashItem["group"], IconName> = { Skills: "sparkle", Prompts: "note", Agent: "terminal" };

export class SlashMenu {
  readonly el: HTMLElement;
  private list = h("ul.pop__list", { role: "listbox", id: "zmc-slash", "aria-label": "Skills, prompts and commands" });
  private status = h("div.pop__status", { role: "status" });
  private all: SlashItem[] | null = null;
  private shown: SlashItem[] = [];
  private sel = 0;
  private q = "";
  private open_ = false;
  private parent: HTMLElement;
  private load: () => Promise<SlashItem[]>;
  private pick: (it: SlashItem) => void;
  onActive: ((id: string | null) => void) | null = null;

  /** The menu is in `parent` only while open. */
  constructor(parent: HTMLElement, load: () => Promise<SlashItem[]>, pick: (it: SlashItem) => void) {
    this.parent = parent; this.load = load; this.pick = pick;
    this.el = h("div.pop.pop--slash", null, this.list, this.status);
  }

  get isOpen(): boolean { return this.open_; }

  /** Opens (reading the items afresh: a skill added since shows) or refilters for `q`. */
  show(q: string): void {
    this.q = q;
    if (!this.open_) {
      this.open_ = true;
      this.parent.prepend(this.el);
      this.all = null;
      this.status.textContent = "Loading…";
      clear(this.list);
      this.load().then((items) => { if (this.open_) { this.all = items; this.paint(); } }, (e) => {
        if (this.open_) this.status.textContent = `Couldn't read your skills: ${e instanceof Error ? e.message : String(e)}`;
      });
      return;
    }
    if (this.all) this.paint();
  }

  close(): void {
    if (!this.open_) return;
    this.open_ = false;
    this.el.remove();
    clear(this.list);
    this.onActive?.(null);
  }

  private paint(): void {
    const ranked = rankItems(this.all ?? [], this.q);
    this.shown = GROUPS.flatMap((g) => ranked.filter((it) => it.group === g));
    this.sel = 0;
    this.status.textContent = this.shown.length ? "" : this.q ? `Nothing called “${this.q}”. Add skills and prompts in Settings.` : "No skills or prompts yet. Add them in Settings.";
    const kids: HTMLElement[] = [];
    let group = "";
    this.shown.forEach((it, i) => {
      if (it.group !== group) { group = it.group; kids.push(h("li.pop__h", { role: "presentation" }, group)); }
      kids.push(h(`li.pop__i${i === this.sel ? ".pop__i--on" : ""}`, {
        role: "option", id: `zmc-slash-${i}`, "aria-selected": String(i === this.sel), dataset: { group: it.group, id: it.id },
        onmousedown: (e: Event) => { e.preventDefault(); this.pick(it); },
        onmousemove: () => { if (this.sel !== i) { this.sel = i; this.mark(); } },
      },
        h("span.pop__ic", null, icon(ICON[it.group])),
        h("span.pop__tx", null, h(`span.pop__t${it.group === "Skills" ? ".pop__t--cmd" : ""}`, null, it.label), it.detail ? h("span.pop__s", null, it.detail) : null)));
    });
    setKids(this.list, kids);
    this.onActive?.(this.shown.length ? "zmc-slash-0" : null);
  }

  private mark(): void {
    for (const li of this.list.querySelectorAll(".pop__i")) {
      const on = li.id === `zmc-slash-${this.sel}`;
      li.classList.toggle("pop__i--on", on);
      li.setAttribute("aria-selected", String(on));
      if (on) li.scrollIntoView({ block: "nearest" });
    }
    this.onActive?.(`zmc-slash-${this.sel}`);
  }

  /** Arrows move, Enter or Tab picks, Escape closes. True when the key was ours. */
  handleKey(e: KeyboardEvent): boolean {
    if (!this.open_) return false;
    if (e.key === "Escape") { this.close(); return true; }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (this.shown.length) { this.sel = (this.sel + (e.key === "ArrowDown" ? 1 : this.shown.length - 1)) % this.shown.length; this.mark(); }
      return true;
    }
    if (e.key === "Enter" || e.key === "Tab") {
      if (!this.all) return true; // still loading: the draft must not be sent
      const it = this.shown[this.sel];
      if (it) { this.pick(it); return true; }
      return e.key === "Tab";
    }
    return false;
  }
}
