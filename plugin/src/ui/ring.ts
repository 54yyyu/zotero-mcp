// The context ring beside the mode picker (DESIGN.md "Context meter"): how full the agent's context window is. Hover or
// focus shows the number at once in a small tooltip; a click (Enter, Space) opens a popover with what we really know
// about this chat and the ways to make room. All of it is hidden while the backend reports nothing.
import type { Usage } from "../types.ts";
import type { Fill } from "./economy.ts";
import { fmtTokens, h, icon, setKids, svg } from "./dom.ts";

/** The ring turns amber from WARM_AT (and New chat becomes the popover's main action); from FULL_AT it turns red. */
export const WARM_AT = 70;
export const FULL_AT = 85;

const WHY = "Everything the agent holds in mind in this chat: your messages, what it read, its answers. When it fills up, the agent summarises the older parts automatically.";

/** What the popover lists, read when it opens (and again when the fill or the busy state changes while it is open). */
export interface RingInfo {
  stats: { messages: number; compactions: number; last?: Usage };
  /** Cost only when the user turned "Show tokens and cost" on. */
  showCost: boolean;
  /** What the last message's context carried (economy.ts sentLine), or "". */
  sent: string;
  canCompact: boolean;
}

export interface RingOpts {
  info(): RingInfo;
  newChat(): void;
  compact(): void;
  /** The popover is opening: other popups close. */
  opening(): void;
}

export class ContextRing {
  readonly el: HTMLButtonElement;
  private host: HTMLElement;
  private opts: RingOpts;
  private arc = svg("circle", { cx: "8", cy: "8", r: "6", pathLength: "100", class: "cmeter__arc" });
  private tip = h("div.ctip", { hidden: true, "aria-hidden": "true" });
  private pop: HTMLElement | null = null;
  private fill: Fill | null = null;
  private busy = false;
  private sig = "";

  constructor(host: HTMLElement, opts: RingOpts) {
    this.host = host;
    this.opts = opts;
    this.el = h("button.cmeter", {
      type: "button", hidden: true, "aria-haspopup": "dialog", "aria-expanded": "false",
      onclick: () => (this.pop ? this.close(true) : this.open()),
      onpointerenter: () => this.showTip(),
      onpointerleave: () => this.hideTip(),
      onfocus: () => { if (this.el.matches(":focus-visible")) this.showTip(); },
      onblur: () => this.hideTip(),
      onkeydown: (e: KeyboardEvent) => { if (e.key === "Escape" && !this.tip.hidden) { e.preventDefault(); e.stopPropagation(); this.hideTip(); } },
    }, svg("svg", { viewBox: "0 0 16 16", "aria-hidden": "true" }, svg("circle", { cx: "8", cy: "8", r: "6", class: "cmeter__track" }), this.arc)) as HTMLButtonElement;
    host.appendChild(this.tip);
  }

  get isOpen(): boolean { return this.pop !== null; }
  inside(path: EventTarget[]): boolean { return !!this.pop && (path.includes(this.pop) || path.includes(this.el)); }

  /** Called on every render: does nothing unless the fill or the busy state changed. */
  set(fill: Fill | null, busy: boolean): void {
    const sig = fill ? `${fill.used}/${fill.size}/${busy}` : "";
    if (sig === this.sig) return;
    this.sig = sig;
    this.fill = fill;
    this.busy = busy;
    this.el.hidden = !fill;
    if (!fill) { this.close(); this.hideTip(); return; }
    const level = fill.pct >= FULL_AT ? "full" : fill.pct >= WARM_AT ? "warm" : "";
    this.el.dataset["level"] = level;
    this.el.setAttribute("aria-label", `Context: ${fill.pct}% full (${fmtTokens(fill.used)} of ${fmtTokens(fill.size)} tokens). Show details`);
    this.arc.setAttribute("stroke-dasharray", `${fill.pct} 100`);
    if (!this.tip.hidden) this.showTip();
    if (this.pop) this.paint();
  }

  // ───────────────────────────── tooltip ─────────────────────────────

  private showTip(): void {
    const f = this.fill;
    if (!f || this.pop) return;
    setKids(this.tip, h("span.ctip__t", null, `${f.pct}% of context used`), h("span.ctip__sub", null, `${fmtTokens(f.used)} of ${fmtTokens(f.size)} tokens`));
    this.tip.hidden = false;
    // Above the ring, inside the composer: centred on it, but never past either edge.
    const r = this.el, w = this.tip.offsetWidth, cx = r.offsetLeft + r.offsetWidth / 2;
    this.tip.style.bottom = `${this.host.offsetHeight - r.offsetTop + 6}px`;
    this.tip.style.left = `${Math.max(4, Math.min(cx - w / 2, this.host.clientWidth - w - 4))}px`;
  }

  private hideTip(): void { this.tip.hidden = true; }

  // ───────────────────────────── popover ─────────────────────────────

  private open(): void {
    if (!this.fill) return;
    this.opts.opening();
    this.hideTip();
    this.pop = h("div.menu.cpop", { role: "dialog", "aria-label": "Context window", tabindex: "-1", onkeydown: (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      e.stopPropagation();
      this.close(true);
    } });
    this.el.setAttribute("aria-expanded", "true");
    this.host.appendChild(this.pop);
    this.paint();
    this.pop.focus();
  }

  close(focusRing = false): void {
    if (!this.pop) return;
    this.pop.remove();
    this.pop = null;
    this.el.setAttribute("aria-expanded", "false");
    if (focusRing) this.el.focus();
  }

  private paint(): void {
    const { pop, fill: f } = this;
    if (!pop || !f) return;
    const info = this.opts.info();
    const { stats: s, sent } = info;
    const last = s.last;
    const had = (pop.querySelector(":focus") as HTMLElement | null)?.dataset["act"];
    const row = (k: string, v: string | null | undefined) => (v ? [h("dt", null, k), h("dd", null, v)] : []);
    const act = (id: string, label: string, run: () => void, solid: boolean, disabled = false) =>
      h(`button.btn.btn--sm${solid ? ".btn--solid" : ".btn--quiet"}`, { type: "button", dataset: { act: id }, disabled: disabled || null, onclick: () => { this.close(); run(); } }, id === "new" ? icon("plus") : null, label);
    pop.dataset["level"] = this.el.dataset["level"] ?? "";
    setKids(pop,
      h("div.cpop__head", null, h("span.cpop__title", null, "Context window"), h("span.cpop__pct", null, `${f.pct}%`)),
      h("div.cpop__bar", { "aria-hidden": "true" }, h("span.cpop__fill", { style: `width:${f.pct}%` })),
      h("div.cpop__tok", null, `${fmtTokens(f.used)} of ${fmtTokens(f.size)} tokens`),
      h("p.cpop__why", null, WHY),
      h("dl.cpop__facts", null,
        row("Messages", String(s.messages)),
        row("Last turn", last && [last.inputTokens ? `${fmtTokens(last.inputTokens)} in` : "", last.outputTokens ? `${fmtTokens(last.outputTokens)} out` : ""].filter(Boolean).join(" · ")),
        row("Cost", info.showCost && last?.costUsd ? `$${last.costUsd.toFixed(last.costUsd < 0.1 ? 3 : 2)} last turn` : null),
        row("Summarised", s.compactions ? (s.compactions === 1 ? "once" : `${s.compactions} times`) : null),
        sent ? [h("dt.cpop__wide", null, "Sent with your last message"), h("dd.cpop__wide", null, sent)] : []),
      h("div.cpop__acts", null,
        info.canCompact ? act("compact", "Summarise now", () => this.opts.compact(), false, this.busy) : null,
        act("new", "New chat", () => this.opts.newChat(), f.pct >= WARM_AT)));
    this.place();
    if (had) (pop.querySelector(`[data-act="${had}"]`) as HTMLElement | null)?.focus();
  }

  /** Straight above the ring, right-aligned in the composer (as the mode menu is); it never grows past the top. */
  private place(): void {
    const pop = this.pop;
    if (!pop) return;
    pop.style.bottom = `${this.host.offsetHeight - this.el.offsetTop + 4}px`;
    pop.style.maxHeight = `${Math.max(120, Math.min(440, this.el.getBoundingClientRect().top - 12))}px`;
  }
}
