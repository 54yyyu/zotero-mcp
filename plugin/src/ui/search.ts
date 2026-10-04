// The `@` / `+` popup: items, collections and annotations matching a query, from host.search.
import type { ItemHit } from "../types.ts";
import { CHIP_ICON, clear, env, errMessage, h, icon, setKids } from "./dom.ts";

interface SearchSource { search(query: string): Promise<ItemHit[]> }

export class SearchPopup {
  readonly el: HTMLElement;
  private list = h("ul.pop__list", { role: "listbox", id: "zmc-hits", "aria-label": "Matches in your library" });
  private status = h("div.pop__status", { role: "status" });
  private input: HTMLInputElement | null = null;
  private hits: ItemHit[] = [];
  private sel = 0;
  private seq = 0;
  private timer: number | undefined;
  private mode: "at" | "plus" | null = null;
  private lastQuery = "\u0000";
  private pending = false;

  private opts: SearchSource;
  private pick: (hit: ItemHit) => void;
  private closeCb: () => void;

  constructor(opts: SearchSource, pick: (hit: ItemHit) => void, closeCb: () => void) {
    this.opts = opts; this.pick = pick; this.closeCb = closeCb;
    this.el = h("div.pop", { hidden: true }, this.list, this.status);
  }

  get isOpen(): boolean { return this.mode !== null; }
  get activeId(): string | null { return this.mode && this.hits[this.sel] ? `zmc-hit-${this.sel}` : null; }

  open(mode: "at" | "plus"): void {
    if (this.mode === mode) return;
    this.close(false);
    this.mode = mode;
    this.el.hidden = false;
    this.lastQuery = "\u0000";
    if (mode === "plus") {
      this.input = h("input.pop__q", {
        type: "text", placeholder: "Search items, collections, annotations", "aria-label": "Search your library",
        role: "combobox", "aria-expanded": "true", "aria-controls": "zmc-hits",
        oninput: () => this.query((this.input as HTMLInputElement).value),
        onkeydown: (e: KeyboardEvent) => { if (this.handleKey(e)) { e.preventDefault(); e.stopPropagation(); } },
      }) as HTMLInputElement;
      this.el.insertBefore(h("div.pop__search", null, icon("search"), this.input), this.list);
      env.win.setTimeout(() => this.input?.focus(), 0);
    }
    this.query("");
  }

  close(notify = true): void {
    if (!this.mode) return;
    this.mode = null;
    this.pending = false;
    this.seq++;
    env.win.clearTimeout(this.timer);
    this.el.hidden = true;
    this.el.querySelector(".pop__search")?.remove();
    this.input = null;
    this.hits = [];
    clear(this.list);
    if (notify) this.closeCb();
  }

  /** Called as the user types after `@`, or from the `+` popup's own field. */
  query(q: string): void {
    if (!this.mode || q === this.lastQuery) return;
    this.lastQuery = q;
    const seq = ++this.seq;
    this.pending = true;
    this.hits = []; // results of an older query must never be picked by a quick Enter
    env.win.clearTimeout(this.timer);
    this.status.textContent = this.list.children.length ? "" : "Searching…";
    this.timer = env.win.setTimeout(() => {
      this.opts.search(q).then((hits) => {
        if (seq !== this.seq) return; // a newer query is in flight
        this.pending = false;
        this.hits = hits.slice(0, 30);
        this.sel = 0;
        this.paint(q);
      }, (e) => {
        if (seq !== this.seq) return;
        this.pending = false;
        this.hits = [];
        clear(this.list);
        this.status.textContent = `Couldn't search Zotero: ${errMessage(e)}`;
        this.status.className = "pop__status pop__status--bad";
      });
    }, q ? 120 : 0);
  }

  private paint(q: string): void {
    this.status.className = "pop__status";
    this.status.textContent = this.hits.length ? "" : q ? `No matches for "${q}"` : "Nothing to suggest yet. Type to search.";
    setKids(this.list, this.hits.map((hit, i) => h(`li.pop__i${i === this.sel ? ".pop__i--on" : ""}`, {
      role: "option", id: `zmc-hit-${i}`, "aria-selected": String(i === this.sel),
      // mousedown, not click: the textarea must not lose its caret before the pick lands
      onmousedown: (e: Event) => { e.preventDefault(); this.pick(hit); },
      onmousemove: () => { if (this.sel !== i) { this.sel = i; this.mark(); } },
    },
      h("span.pop__ic", null, icon(CHIP_ICON[hit.kind] ?? "item")),
      h("span.pop__tx", null, h("span.pop__t", null, hit.title), hit.subtitle ? h("span.pop__s", null, hit.subtitle) : null))));
    this.syncActive();
  }

  private mark(): void {
    [...this.list.children].forEach((li, i) => {
      li.classList.toggle("pop__i--on", i === this.sel);
      li.setAttribute("aria-selected", String(i === this.sel));
    });
    (this.list.children[this.sel] as HTMLElement | undefined)?.scrollIntoView({ block: "nearest" });
    this.syncActive();
  }
  private syncActive(): void { this.onActive?.(this.activeId); }
  onActive: ((id: string | null) => void) | null = null;

  /** Arrow keys, Enter/Tab to pick, Escape to close. Returns true when the key was ours. */
  handleKey(e: KeyboardEvent): boolean {
    if (!this.mode) return false;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (!this.hits.length) return true;
      this.sel = (this.sel + (e.key === "ArrowDown" ? 1 : this.hits.length - 1)) % this.hits.length;
      this.mark();
      return true;
    }
    if ((e.key === "Enter" || e.key === "Tab") && this.pending) return true; // wait for the results, do not send the draft
    if ((e.key === "Enter" || e.key === "Tab") && this.hits[this.sel]) { this.pick(this.hits[this.sel] as ItemHit); return true; }
    if (e.key === "Escape") { this.close(); return true; }
    return false;
  }
}

