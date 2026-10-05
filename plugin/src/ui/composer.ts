// The composer: context chips, the textarea, `@` and `+` search, the pickers, Send / Stop.
// It owns the draft and the popups; the controller (index.ts) owns what a send does.
import type { BackendId, ContextChip, ItemHit, ZoteroRef } from "../types.ts";
import { CHIP_ICON, append, env, errMessage, h, icon, isMac, setKids, svg } from "./dom.ts";
import { SearchPopup } from "./search.ts";
import { Pickers } from "./pickers.ts";
import type { Choices, PickKind } from "./pickers.ts";

export type { Choices };

export interface ComposerOpts {
  search(query: string): Promise<ItemHit[]>;
  chipFor(hit: ItemHit): Promise<ContextChip>;
  open(target: string | ZoteroRef): void;
  onSend(text: string): void;
  onStop(): void;
  onAddChip(chip: ContextChip): void;
  onRemoveChip(chip: ContextChip): void;
  onTogglePin(chip: ContextChip): void;
  /** Something was dropped on the composer; resolves with how many chips it added. */
  onDrop(data: DataTransfer): Promise<number>;
  onPick(kind: PickKind, id: string): void;
  /** The agent's models and modes. May start the agent, so it is asked only when a picker opens. */
  loadChoices(): Promise<Choices>;
  /** New chats use another agent; a chat with messages is replaced by a new one (the picker asked first). */
  switchAgent(id: BackendId): Promise<void>;
  hasMessages(): boolean;
  /** The composer got focus for the first time: a good moment to warm the agent up. */
  onFirstFocus(): void;
  checkSetup(): void;
  /** The long-chat suggestion's button: a new chat that carries nothing over. */
  onNewChat(): void;
}

/** The meter turns amber from WARM_AT; from FULL_AT it turns red and a new chat is suggested. */
const WARM_AT = 70;
const FULL_AT = 85;
const kTokens = (n: number) => (n >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : `${Math.round(n / 1000)}k`);

const B64 = /^[A-Za-z0-9+/]+={0,2}$/;

// ───────────────────────────── the composer ─────────────────────────────

export class Composer {
  readonly el: HTMLElement;
  readonly ta: HTMLTextAreaElement;
  private chipsEl = h("div.cchips", { hidden: true });
  private sendBtn: HTMLButtonElement;
  private plusBtn: HTMLButtonElement;
  private pop: SearchPopup;
  private pickers: Pickers;
  private enterToSend = true;
  private busy = false;
  private blocked: string | null = null;
  private atStart = -1;
  private warmed = false;
  private meter = h("span.cmeter", { hidden: true, role: "img" });
  private meterArc = svg("circle", { cx: "8", cy: "8", r: "6", pathLength: "100", class: "cmeter__arc" });
  private longNote: HTMLElement;
  private noteDismissed = false;

  private opts: ComposerOpts;

  constructor(opts: ComposerOpts) {
    this.opts = opts;
    this.ta = h("textarea.cin", {
      rows: "1", placeholder: "Ask about your library…", "aria-label": "Message", "aria-autocomplete": "list",
      oninput: () => this.onInput(),
      onkeydown: (e: KeyboardEvent) => this.onKey(e),
      onfocus: () => { if (!this.warmed) { this.warmed = true; this.opts.onFirstFocus(); } },
    }) as HTMLTextAreaElement;
    this.pop = new SearchPopup(opts, (hit) => this.pickHit(hit), () => this.afterPopClose());
    this.pop.onActive = (id) => { if (id) this.ta.setAttribute("aria-activedescendant", id); else this.ta.removeAttribute("aria-activedescendant"); };

    this.plusBtn = h("button.iconbtn.iconbtn--sm", { type: "button", "aria-label": "Add a source", title: "Add a source (@)", "aria-haspopup": "listbox", onclick: () => this.togglePlus() }, icon("plus")) as HTMLButtonElement;
    this.el = h("div.composer");
    this.pickers = new Pickers(this.el, { load: () => opts.loadChoices(), pick: (k, id) => opts.onPick(k, id), switchAgent: (id) => opts.switchAgent(id), hasMessages: () => opts.hasMessages(), checkSetup: () => opts.checkSetup() });
    this.sendBtn = h("button.send", { type: "button", "aria-label": "Send", title: "Send (Enter)", onclick: () => this.send() }, icon("send")) as HTMLButtonElement;

    this.meter.appendChild(svg("svg", { viewBox: "0 0 16 16", "aria-hidden": "true" }, svg("circle", { cx: "8", cy: "8", r: "6", class: "cmeter__track" }), this.meterArc));
    this.longNote = h("div.cnote", { hidden: true, role: "status" },
      h("span.cnote__t", null, "This chat is getting long. A new chat starts fresh and carries nothing over."),
      h("button.lnk", { type: "button", onclick: () => opts.onNewChat() }, icon("plus"), "New chat"),
      h("button.chip__btn", { type: "button", "aria-label": "Dismiss", title: "Dismiss", onclick: () => { this.noteDismissed = true; this.longNote.hidden = true; } }, icon("close")));

    const b = this.pickers.buttons;
    append(this.el, [this.pop.el, this.longNote, this.chipsEl, this.ta,
      h("div.ctools", null, this.plusBtn, b.model, h("span.ctools__fill"), this.meter, b.mode, this.sendBtn)]);
    this.wireDrop();
    env.doc.addEventListener("pointerdown", this.outside, true);
    this.syncSend();
  }

  /** The whole card is the drop target. The overlay is absolute, so nothing shifts while dragging. */
  private wireDrop(): void {
    this.el.appendChild(h("div.dropveil", { "aria-hidden": "true" }, icon("plus"), "Drop to add"));
    let depth = 0;
    const off = () => { depth = 0; this.el.classList.remove("composer--drop"); };
    this.el.addEventListener("dragenter", (e) => { e.preventDefault(); depth++; this.el.classList.add("composer--drop"); });
    this.el.addEventListener("dragover", (e) => { e.preventDefault(); if (e.dataTransfer) e.dataTransfer.dropEffect = "copy"; });
    this.el.addEventListener("dragleave", () => { if (--depth <= 0) off(); });
    this.el.addEventListener("drop", (e) => {
      e.preventDefault();
      off();
      if (e.dataTransfer) void this.opts.onDrop(e.dataTransfer).then((n) => { if (n) this.ta.focus(); });
    });
  }

  dispose(): void { env.doc.removeEventListener("pointerdown", this.outside, true); }

  private outside = (e: Event): void => {
    const path = e.composedPath();
    if (this.pickers.isOpen && !this.pickers.inside(path)) this.pickers.close();
    if (this.pop.isOpen && !path.includes(this.pop.el) && !path.includes(this.plusBtn) && !path.includes(this.ta)) this.pop.close();
  };

  // --- state from the controller

  setBusy(busy: boolean): void { this.busy = busy; this.syncSend(); }
  /** A reason sending is impossible right now (no backend, ...); null when fine. */
  setBlocked(reason: string | null): void { this.blocked = reason; this.syncSend(); }
  setChoices(c: Choices): void { this.pickers.set(c); }
  setEnterToSend(on: boolean): void { this.enterToSend = on; this.syncSend(); }
  focus(): void { this.ta.focus(); }
  setText(t: string): void { this.ta.value = t; this.autosize(); this.syncSend(); }
  clearDraft(): void { this.setText(""); }

  /** How full the agent's context is: a ring whenever the backend says (hidden when it does not), a new-chat suggestion from FULL_AT. */
  setContextFill(fill: { pct: number; used: number; size: number } | null): void {
    const pct = fill?.pct ?? 0;
    this.meter.hidden = !fill;
    if (pct < FULL_AT) this.noteDismissed = false;
    this.longNote.hidden = pct < FULL_AT || this.noteDismissed;
    if (!fill) return;
    const tip = `Context: ${pct}% full (${kTokens(fill.used)} of ${kTokens(fill.size)} tokens). Older parts are summarised automatically.`;
    this.meter.title = tip;
    this.meter.setAttribute("aria-label", tip);
    this.meter.dataset["level"] = pct >= FULL_AT ? "full" : pct >= WARM_AT ? "warm" : "";
    this.meterArc.setAttribute("stroke-dasharray", `${pct} 100`);
  }

  setChips(chips: ContextChip[]): void {
    this.chipsEl.hidden = chips.length === 0;
    setKids(this.chipsEl, chips.map((c) => (c.kind === "area" ? this.areaCard(c) : this.chip(c))));
  }

  private chip(c: ContextChip): HTMLElement {
    const label = c.kind === "selection" ? "Text Selection" : c.label;
    const tip = c.text ? `${label}: ${c.text.slice(0, 400)}` : label;
    const pin = (c.auto || c.pinned) && c.kind !== "selection"
      ? h("button.chip__btn.chip__pin", { type: "button", "aria-pressed": String(c.pinned), "aria-label": c.pinned ? `Unpin ${label}` : `Keep ${label} in this chat`, title: c.pinned ? "Pinned: stays when you switch items" : "Pin: keep this when you switch items", onclick: () => this.opts.onTogglePin(c) }, icon("bookmark"))
      : null;
    const x = h("button.chip__btn.chip__x", { type: "button", "aria-label": `Remove ${label}`, title: "Remove", onclick: () => this.opts.onRemoveChip(c) }, icon("close"));
    return h(`div.chip${c.auto ? ".chip--auto" : ""}${c.pinned ? ".chip--pinned" : ""}`, { title: tip },
      h("button.chip__main", { type: "button", onclick: () => this.opts.open(c.ref), "aria-label": `Open ${label}` },
        h("span.chip__i", null, icon(CHIP_ICON[c.kind] ?? "item")), h("span.chip__t", null, label)),
      pin, x);
  }

  private areaCard(c: ContextChip): HTMLElement {
    const src = c.image && B64.test(c.image.data) ? `data:image/png;base64,${c.image.data}` : null;
    return h("div.area", null,
      h("div.area__head", null, icon("area"), h("span", null, c.label || "Selected Area")),
      src ? h("img.area__img", { src, alt: "The selected area of the page" }) : h("div.area__none", null, "No preview"),
      h("div.area__acts", null,
        h("button.lnk", { type: "button", onclick: () => this.opts.open(c.ref) }, icon("external"), "Go to Annotation"),
        h("button.lnk", { type: "button", onclick: () => this.opts.onRemoveChip(c) }, icon("close"), "Remove")));
  }

  // --- typing

  private autosize(): void {
    const ta = this.ta;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 168)}px`;
  }

  private onInput(): void {
    this.autosize();
    this.syncSend();
    // `@query` right before the caret opens the search.
    const caret = this.ta.selectionStart ?? this.ta.value.length;
    const before = this.ta.value.slice(0, caret);
    const m = /(^|\s)@([^\s@]{0,40})$/.exec(before);
    if (m) {
      this.atStart = (m.index ?? 0) + (m[1] as string).length;
      if (!this.pop.isOpen) this.pop.open("at");
      this.pop.query(m[2] as string);
    } else if (this.pop.isOpen && this.atStart >= 0) {
      this.pop.close();
    }
  }

  private onKey(e: KeyboardEvent): void {
    if (e.isComposing || e.keyCode === 229) return; // an IME is picking a character
    if (this.pop.isOpen && this.atStart >= 0 && this.pop.handleKey(e)) { e.preventDefault(); return; }
    const mod = e.metaKey || e.ctrlKey;
    if (e.key === "Enter" && (this.enterToSend ? !e.shiftKey && !e.altKey && !mod : mod)) {
      e.preventDefault();
      this.send();
    } else if (e.key === "Escape" && this.busy) {
      e.preventDefault();
      this.opts.onStop();
    }
  }

  private syncSend(): void {
    const has = this.ta.value.trim().length > 0;
    const stop = this.busy;
    this.sendBtn.className = stop ? "send send--stop" : "send";
    this.sendBtn.setAttribute("aria-label", stop ? "Stop" : "Send");
    const key = this.enterToSend ? "Enter" : isMac() ? "⌘Enter" : "Ctrl+Enter";
    this.sendBtn.title = stop ? "Stop (Esc)" : this.blocked ?? `Send (${key})`;
    this.ta.setAttribute("aria-keyshortcuts", this.enterToSend ? "Enter" : "Control+Enter Meta+Enter");
    this.sendBtn.disabled = !stop && (!has || !!this.blocked);
    this.sendBtn.replaceChildren(icon(stop ? "stop" : "send"));
  }

  private send(): void {
    if (this.busy) { this.opts.onStop(); return; }
    const t = this.ta.value.trim();
    if (!t || this.blocked) return;
    this.pop.close();
    this.opts.onSend(t);
  }

  // --- search

  private togglePlus(): void {
    if (this.pop.isOpen) { this.pop.close(); return; }
    this.atStart = -1;
    this.pickers.close();
    this.pop.open("plus");
    this.plusBtn.setAttribute("aria-expanded", "true");
  }

  private afterPopClose(): void {
    this.plusBtn.setAttribute("aria-expanded", "false");
    if (this.atStart < 0) this.ta.focus();
    this.atStart = -1;
  }

  private async pickHit(hit: ItemHit): Promise<void> {
    if (this.atStart >= 0) { // remove "@query" from the draft
      const v = this.ta.value;
      this.setText(v.slice(0, this.atStart) + v.slice(this.ta.selectionStart ?? v.length).replace(/^\s/, ""));
      this.ta.setSelectionRange(this.atStart, this.atStart);
    }
    this.pop.close();
    try {
      this.opts.onAddChip(await this.opts.chipFor(hit));
    } catch (e) {
      this.chipsEl.hidden = false;
      const err = h("div.cerr", { role: "alert" }, `Couldn't add ${hit.title}: ${errMessage(e)}`);
      this.chipsEl.appendChild(err);
      env.win.setTimeout(() => err.remove(), 5000);
    }
    this.ta.focus();
  }
}
