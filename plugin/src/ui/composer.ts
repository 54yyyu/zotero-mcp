// The composer: context chips, the textarea, `@` and `+` search, the `/` menu, the pickers, Send / Stop, and the pinned
// skills and prompts as buttons above it on a new chat.
// It owns the draft and the popups; the controller (index.ts) owns what a send does.
import type { BackendId, ContextChip, ItemHit, ZoteroRef } from "../types.ts";
import { CHIP_ICON, append, env, errMessage, h, icon, isMac, setKids, slotLabel } from "./dom.ts";
import type { Fill } from "./economy.ts";
import { SearchPopup } from "./search.ts";
import { Pickers } from "./pickers.ts";
import type { Choices, PickKind } from "./pickers.ts";
import { ContextRing, FULL_AT } from "./ring.ts";
import type { RingInfo } from "./ring.ts";
import { SlashMenu } from "./slash.ts";
import type { SlashItem } from "./slash.ts";

export type { Choices, SlashItem };

/** A pinned skill or prompt: a button above the composer on a new, empty chat, and its shortcut. */
export interface Bubble { label: string; title: string; skill: boolean; slot: number; run(): void }

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
  /** The long-chat suggestion's and the context popover's button: a new chat that carries nothing over. */
  onNewChat(): void;
  /** What the context popover lists about this chat. */
  contextInfo(): RingInfo;
  /** The context popover's Summarise now. */
  onCompact(): void;
  /** What the `/` menu lists, read each time it opens. */
  slashItems(): Promise<SlashItem[]>;
}

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
  private ring: ContextRing;
  private longNote: HTMLElement;
  private noteDismissed = false;
  private slash: SlashMenu;
  /** Above the composer card (the panel puts it there): shown while the chat is new and the draft empty. */
  readonly bubbles = h("div.bubbles", { hidden: true, role: "group", "aria-label": "Pinned skills and prompts" });
  private bubbleCount = 0;

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
    const active = (id: string | null) => { if (id) this.ta.setAttribute("aria-activedescendant", id); else this.ta.removeAttribute("aria-activedescendant"); };
    this.pop.onActive = active;

    this.plusBtn = h("button.iconbtn.iconbtn--sm", { type: "button", "aria-label": "Add a source", title: "Add a source (@)", "aria-haspopup": "listbox", onclick: () => this.togglePlus() }, icon("plus")) as HTMLButtonElement;
    this.el = h("div.composer");
    this.slash = new SlashMenu(this.el, () => opts.slashItems(), (it) => this.pickSlash(it));
    this.slash.onActive = active;
    this.pickers = new Pickers(this.el, { load: () => opts.loadChoices(), pick: (k, id) => opts.onPick(k, id), switchAgent: (id) => opts.switchAgent(id), hasMessages: () => opts.hasMessages(), checkSetup: () => opts.checkSetup() });
    this.sendBtn = h("button.send", { type: "button", "aria-label": "Send", title: "Send (Enter)", onclick: () => this.send() }, icon("send")) as HTMLButtonElement;

    this.ring = new ContextRing(this.el, { info: () => opts.contextInfo(), newChat: () => opts.onNewChat(), compact: () => opts.onCompact(), opening: () => { this.pickers.close(); this.pop.close(); } });
    this.longNote = h("div.cnote", { hidden: true, role: "status" },
      h("span.cnote__t", null, "This chat is getting long. A new chat starts fresh and carries nothing over."),
      h("button.lnk", { type: "button", onclick: () => opts.onNewChat() }, icon("plus"), "New chat"),
      h("button.chip__btn", { type: "button", "aria-label": "Dismiss", title: "Dismiss", onclick: () => { this.noteDismissed = true; this.longNote.hidden = true; } }, icon("close")));

    const b = this.pickers.buttons;
    append(this.el, [this.pop.el, this.longNote, this.chipsEl, this.ta,
      h("div.ctools", null, this.plusBtn, b.model, h("span.ctools__fill"), this.ring.el, b.mode, this.sendBtn)]);
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
    if (this.ring.isOpen && !this.ring.inside(path)) this.ring.close();
    if (this.pop.isOpen && !path.includes(this.pop.el) && !path.includes(this.plusBtn) && !path.includes(this.ta)) this.pop.close();
    if (this.slash.isOpen && !path.includes(this.slash.el) && !path.includes(this.ta)) this.slash.close();
  };

  // --- state from the controller

  setBusy(busy: boolean): void { this.busy = busy; this.syncSend(); }
  /** A reason sending is impossible right now (no backend, ...); null when fine. */
  setBlocked(reason: string | null): void { this.blocked = reason; this.syncSend(); }
  setChoices(c: Choices): void { this.pickers.set(c); }
  setEnterToSend(on: boolean): void { this.enterToSend = on; this.syncSend(); }
  focus(): void { this.ta.focus(); }
  setText(t: string): void { this.ta.value = t; this.autosize(); this.syncSend(); if (!t.startsWith("/")) this.slash.close(); }

  /** The pinned skills and prompts, or none (a chat with messages). */
  setBubbles(items: Bubble[]): void {
    this.bubbleCount = items.length;
    setKids(this.bubbles, items.map((b) => h(`button.bubble${b.skill ? ".bubble--skill" : ""}`, { type: "button", title: `${b.title}\n${slotLabel(b.slot)}`, "aria-keyshortcuts": isMac() ? `Meta+Control+${b.slot}` : `Control+Alt+${b.slot}`, onclick: () => b.run() },
      b.skill ? icon("sparkle") : null, h("span.bubble__t", null, b.label))));
    this.syncSend();
  }
  clearDraft(): void { this.setText(""); }

  /** How full the agent's context is: the ring whenever the backend says (hidden when it does not), a new-chat suggestion from FULL_AT. */
  setContextFill(fill: Fill | null): void {
    const pct = fill?.pct ?? 0;
    this.ring.set(fill, this.busy);
    if (pct < FULL_AT) this.noteDismissed = false;
    this.longNote.hidden = pct < FULL_AT || this.noteDismissed;
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
        h("button.lnk", { type: "button", onclick: () => this.opts.open(c.ref) }, icon("external"), c.ref.annotationKey ? "Go to Annotation" : "Go to Page"),
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
    const caret = this.ta.selectionStart ?? this.ta.value.length;
    const before = this.ta.value.slice(0, caret);
    // `/query` as the whole message so far opens the `/` menu.
    const sl = /^\/([^\s/]{0,40})$/.exec(before);
    if (sl) { this.pop.close(); this.pickers.close(); this.slash.show(sl[1] as string); return; }
    this.slash.close();
    // `@query` right before the caret opens the search.
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
    if (this.slash.handleKey(e)) { e.preventDefault(); return; }
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
    this.bubbles.hidden = has || this.bubbleCount === 0;
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
    this.slash.close();
    this.opts.onSend(t);
  }

  // --- the `/` menu

  /** A skill or prompt fills the message box (Enter again sends, or add a sentence first); a command runs at once. */
  private pickSlash(it: SlashItem): void {
    this.slash.close();
    if (it.insert !== undefined) {
      this.setText(it.insert);
      this.ta.setSelectionRange(it.insert.length, it.insert.length);
      this.ta.focus();
    } else {
      this.setText("");
      it.run?.();
    }
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
