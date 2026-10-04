// The panel: `mountPanel(root, host)` renders into a shadow root and returns its handle. This file wires
// the parts together: header, the feed and composer (the chat), and the other screens. The conversation is
// in chat.ts, setup health in health.ts, and the DOM of each part in messages / composer / views.
import type { BackendId, Catalog, MountPanel, PanelHost, PromptEntry, ZoteroRef } from "../types.ts";
import { STYLES } from "./styles.ts";
import { clear, copyText, env, errMessage, h, icon, initEnv, setKids } from "./dom.ts";
import { Composer } from "./composer.ts";
import type { Choices } from "./composer.ts";
import { Chat } from "./chat.ts";
import { ChipState } from "./context.ts";
import { BLOCKING, Health } from "./health.ts";
import { Feed } from "./messages.ts";
import { settingsView } from "./settings.ts";
import { BACKEND_LABEL, emptyState, historyView, setupCard, statusView } from "./views.ts";
import { welcomeView } from "./welcome.ts";

type View = "chat" | "history" | "settings" | "status" | "welcome";

export const mountPanel: MountPanel = (root, host) => {
  initEnv(root);
  return new Panel(root, host).handle;
};

class Panel {
  readonly handle: ReturnType<MountPanel>;
  private root: ShadowRoot;
  private host: PanelHost;
  private chat: Chat;
  private health: Health;
  private chips: ChipState;
  private feed: Feed;
  private composer: Composer;

  private app: HTMLElement;
  private chatEl: HTMLElement;
  private viewEl: HTMLElement;
  private setupSlot = h("div.setupslot");
  private statusBtn: HTMLButtonElement;
  private accountEl = h("span.stat__t");
  private historyBtn: HTMLButtonElement;

  private view: View = "chat";
  private activeView: { render?: () => void; refresh?: () => void } | null = null;
  private focusPrompts = false;
  private pendingRender = false;
  private emptyShown: boolean | null = null;
  private autoPicked = false;
  private disposers: (() => void)[] = [];
  private catalogs = new Map<BackendId, Catalog>();

  constructor(root: ShadowRoot, host: PanelHost) {
    this.root = root;
    this.host = host;
    this.chips = new ChipState(host);
    this.health = new Health(host, () => this.onHealth());
    this.chat = new Chat({
      host,
      onChange: () => this.scheduleRender(),
      blockReason: () => this.health.blockReason(),
      turnEnded: (stop) => this.feed.announce(stop === "end_turn" ? "Answer finished" : "Turn ended"),
      setupFailed: () => void this.health.refresh(),
      sessionChanged: () => this.syncPickers(),
    });
    this.feed = new Feed({
      open: (t) => void this.open(t),
      copy: (t) => void copyText(t),
      retry: (id) => this.chat.retry(id),
      answer: (turn, pid, oid) => this.chat.answerPermission(turn, pid, oid),
      checkSetup: () => this.show("status"),
    });
    this.composer = new Composer({
      search: (q) => host.search(q),
      chipFor: (hit) => host.chipFor(hit),
      open: (t) => void this.open(t),
      onSend: (t) => this.send(t),
      onStop: () => void this.chat.stop(),
      onAddChip: (c) => { this.chips.add(c); this.refreshChips(); },
      onDrop: async (data) => {
        // Only the host reads a drop: Zotero's drag formats are not the UI's business.
        const dropped = await host.dropChips(data).catch(() => []);
        for (const c of dropped) this.chips.add(c, true);
        if (dropped.length) this.refreshChips();
        return dropped.length;
      },
      onRemoveChip: (c) => { this.chips.remove(c); this.refreshChips(); },
      onTogglePin: (c) => { this.chips.togglePin(c); this.refreshChips(); },
      onPick: (kind, id) => void this.pick(kind, id),
      loadChoices: async () => { await this.loadCatalog(true); return this.choices(); },
      onFirstFocus: () => { if (!this.chat.session && !this.chat.starting && !this.health.blockReason()) void this.chat.ensureSession().catch(() => {}); },
      checkSetup: () => this.show("status"),
    });

    const btn = (label: string, ic: Parameters<typeof icon>[0], onclick: () => void) =>
      h("button.iconbtn", { type: "button", "aria-label": label, title: label, onclick }, icon(ic)) as HTMLButtonElement;
    this.historyBtn = btn("History", "history", () => this.show(this.view === "history" ? "chat" : "history"));
    this.statusBtn = h("button.stat", { type: "button", onclick: () => this.show(this.view === "status" ? "chat" : "status") }, h("span.stat__dot"), this.accountEl) as HTMLButtonElement;
    const hd = h("header.hd", null,
      btn("Close the panel", "close", () => (root.host ?? this.app).dispatchEvent(new env.win.CustomEvent("zmc-close", { bubbles: true, composed: true }))),
      btn("New chat", "plus", () => void this.newChat()), this.historyBtn,
      h("span.hd__fill"), this.statusBtn,
      btn("Settings", "gear", () => this.show(this.view === "settings" ? "chat" : "settings")));
    this.chatEl = h("div.chat", null, this.feed.el, h("div.dock", null, this.setupSlot, this.composer.el));
    this.viewEl = h("div.viewhost", { hidden: true });
    this.app = h("div.zmc", { dataset: { theme: host.theme(), view: "chat" }, onkeydown: (e: KeyboardEvent) => this.onKey(e) }, hd, h("div.body", null, this.chatEl, this.viewEl));
    root.append(h("style", { text: STYLES }), this.app);

    this.disposers.push(host.onContextChange(() => this.refreshChips()), host.onThemeChange(() => { this.app.dataset.theme = host.theme(); }));
    this.refreshChips();
    this.onHealth();
    this.applyBehavior();
    void this.loadCatalog(false);
    this.render();
    void this.health.refresh();
    if (!host.getSettings().welcomed) this.show("welcome");

    this.handle = {
      dispose: () => this.dispose(),
      focusComposer: () => { this.show("chat"); this.composer.focus(); },
      runPrompt: (slot) => { const p = host.getSettings().prompts.find((x) => x.slot === slot); if (p) this.runPrompt(p); },
    };
  }

  // ───────────────────────────── screens ─────────────────────────────

  private show(v: View): void {
    // Until the welcome is finished or skipped it stands in for the chat.
    if (v === "chat" && !this.host.getSettings().welcomed) v = "welcome";
    this.view = v;
    this.app.dataset.view = v;
    this.chatEl.hidden = v !== "chat";
    this.viewEl.hidden = v === "chat";
    this.historyBtn.setAttribute("aria-pressed", String(v === "history"));
    this.statusBtn.setAttribute("aria-pressed", String(v === "status"));
    this.activeView = null;
    clear(this.viewEl);
    if (v === "chat") { this.composer.focus(); return; }
    const back = () => this.show("chat");
    const host = this.host;
    const screen =
      v === "history" ? historyView(host, { current: () => this.chat.saved?.id ?? null, back, resume: (s) => void this.openSaved(s), deleted: (id) => { if (this.chat.saved?.id === id) void this.newChat(); } })
      : v === "welcome" ? welcomeView(host, this.health, { done: () => void this.finishWelcome(), recheck: () => void this.health.refresh(), changed: () => this.onSettings(), openSettings: () => this.show("settings") })
      : v === "status" ? statusView(host, { back, initial: this.health.checks, openSettings: () => this.show("settings"), changed: (c) => this.health.set(c) })
      : settingsView(host, { back, statuses: () => this.health.statuses, refreshStatuses: () => void this.health.refreshStatuses().then(() => this.activeView?.render?.()), changed: () => this.onSettings(), focusPrompts: this.focusPrompts });
    this.focusPrompts = false;
    this.activeView = screen;
    this.viewEl.append(screen.el);
    this.activeView.refresh?.();
    (this.viewEl.querySelector(".vw__head button, .wcard--on") as HTMLElement | null)?.focus();
  }

  private onKey(e: KeyboardEvent): void {
    if (e.key !== "Escape" || this.view === "chat" || this.view === "welcome" || e.defaultPrevented || (e.target as HTMLElement | null)?.tagName === "SELECT") return;
    e.preventDefault();
    this.show("chat");
  }

  /** A notice in the chat that is not saved with it. */
  private warn(message: string): void { this.chat.dispatch({ t: "notice", level: "warn", message }, false); }

  private async finishWelcome(): Promise<void> {
    await this.host.setSettings({ welcomed: true });
    this.show("chat");
    if (!env.win.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
      this.chatEl.animate?.([{ opacity: 0, transform: "translateY(10px)" }, { opacity: 1, transform: "none" }], { duration: 360, easing: "ease-out" });
      this.root.querySelector(".empty .mark")?.classList.add("mark--hello");
    }
  }

  private async open(target: string | ZoteroRef): Promise<void> {
    try { await this.host.open(target); } catch (e) { this.warn(`Couldn't open that in Zotero: ${errMessage(e)}`); }
  }

  private async newChat(): Promise<void> {
    await this.chat.reset();
    this.chips.clearAll();
    this.composer.clearDraft();
    this.feed.clear();
    this.show("chat");
    this.refreshChips();
    this.render();
  }

  private async openSaved(s: Parameters<Chat["load"]>[0]): Promise<void> {
    this.feed.clear();
    await this.chat.load(s);
    this.show("chat");
    this.render();
    this.feed.toBottom(false);
  }

  // ───────────────────────────── health and settings ─────────────────────────────

  /** The doctor or the backend list changed: the dot, the label, the empty state and whether Send works. */
  private onHealth(): void {
    const s = this.host.getSettings();
    const st = this.health.backend();
    this.statusBtn.dataset.state = this.health.state;
    // Say plainly who pays: a subscription's account name, or "API key". pi has neither: it uses the user's own provider keys.
    const label = s.backend === "pi" ? "pi" : s.auth[s.backend] === "api-key" ? `${BACKEND_LABEL[s.backend]} · API key` : st?.account || `${BACKEND_LABEL[s.backend]} · subscription`;
    this.accountEl.textContent = label;
    const bad = this.health.failing;
    const tip = { ok: "Everything is set up", checking: "Checking…", unknown: "Status unknown", warn: "", bad: "" }[this.health.state] || bad.map((c) => c.label).join(", ");
    this.statusBtn.title = `${tip}. Open status`;
    this.statusBtn.setAttribute("aria-label", `${label}. Status: ${tip}`);
    this.composer.setBlocked(this.health.blockReason());
    this.paintEmpty();
    if (this.view === "welcome") {
      this.autoPick();
      this.activeView?.render?.();
    }
  }

  /** On the welcome, once the agents are known: if the chosen one is not usable and another is, start with that one. */
  private autoPick(): void {
    const st = this.health.statuses;
    if (this.autoPicked || !st) return;
    this.autoPicked = true;
    const chosen = this.host.getSettings().backend;
    const alt = st.find((b) => b.available);
    if (st.find((b) => b.id === chosen)?.available || !alt) return;
    void this.host.setSettings({ backend: alt.id }).then(() => { this.onSettings(); void this.health.refresh(); });
  }

  private onSettings(): void {
    this.onHealth();
    this.applyBehavior();
    // An empty chat has no history to protect: let a changed backend take effect on the next send.
    if (!this.chat.tr.messages.length && this.chat.session && !this.chat.sending) void this.chat.closeSession();
    void this.loadCatalog(false);
    void this.health.refreshStatuses();
  }

  /** What the composer's pickers show: the live session's values, else the saved choice over the backend's catalog. */
  private choices(): Choices {
    const s = this.host.getSettings();
    const b = s.backend;
    const sess = this.chat.session;
    const cat = this.catalogs.get(b);
    const saved = (k: "model" | "mode" | "effort") => s[k][b] || cat?.[k];
    return {
      models: sess?.models() ?? cat?.models ?? [], model: sess ? sess.currentModel() : saved("model"),
      efforts: sess?.efforts() ?? cat?.efforts ?? [], effort: sess ? sess.currentEffort() : saved("effort"),
      modes: sess?.modes() ?? cat?.modes ?? [], mode: sess ? sess.currentMode() : saved("mode"),
    };
  }

  private syncPickers(): void { this.composer.setChoices(this.choices()); }

  /** The chosen backend's models, modes and efforts. `strict`: let a failure reach the caller (a picker menu shows it). */
  private async loadCatalog(strict: boolean): Promise<void> {
    const b = this.host.getSettings().backend;
    if (this.catalogs.has(b) || this.chat.session?.backend === b) return this.syncPickers();
    try {
      this.catalogs.set(b, await this.host.runtime.catalog(b));
    } catch (e) {
      if (strict) throw e;
    }
    this.syncPickers();
  }

  private async pick(kind: "model" | "effort" | "mode", id: string): Promise<void> {
    try { await this.chat.pick(kind, id); } catch (e) { this.warn(`Couldn't switch the ${kind}: ${errMessage(e)}`); }
    this.syncPickers();
  }

  /** The chat settings that this layer applies (the host applies the context ones). */
  private applyBehavior(): void {
    const s = this.host.getSettings();
    this.composer.setEnterToSend(s.enterToSend);
    this.feed.setOptions({ showThinking: s.showThinking, expandTools: s.expandTools });
  }

  // ───────────────────────────── the chat ─────────────────────────────

  private refreshChips(): void { this.composer.setChips(this.chips.current()); }

  private paintEmpty(): void {
    const s = this.host.getSettings();
    const first = this.health.first();
    const handlers = { recheck: () => void this.health.refresh(), openStatus: () => this.show("status"), openSettings: () => this.show("settings") };
    const card = () => (first ? setupCard(first, Math.max(0, this.health.failing.length - 1), handlers) : null);
    this.feed.setEmpty(h("div.emptywrap", null, card(),
      emptyState({ prompts: s.prompts, ready: !this.health.blockReason(), run: (p) => this.runPrompt(p), edit: () => { this.focusPrompts = true; this.show("settings"); } })));
    // Mid-conversation only a problem that stops chatting gets a card, above the composer.
    const mid = this.chat.tr.messages.length > 0 && first && (BLOCKING.has(first.id) || first.id === "backend");
    setKids(this.setupSlot, mid ? card() : null);
  }

  private send(text: string, fromDraft = true): void {
    if (this.health.blockReason()) { this.show("status"); return; }
    if (this.chat.busy) return;
    const used = this.chips.current();
    if (fromDraft) this.composer.clearDraft();
    this.chips.clearAdded();
    this.refreshChips();
    void this.chat.send(text, used);
    this.render();
    this.feed.toBottom(false);
  }

  private runPrompt(p: PromptEntry): void {
    this.show("chat");
    if (this.view === "welcome") return;
    if (this.chat.busy || this.health.blockReason()) { this.composer.setText(p.text); this.composer.focus(); return; }
    this.send(p.text, false);
  }

  /** Batch renders: one per frame (a timer too, since a hidden window never gets a frame). */
  private scheduleRender(): void {
    if (this.pendingRender) return;
    this.pendingRender = true;
    const run = () => { if (this.pendingRender) { this.pendingRender = false; this.render(); } };
    env.win.requestAnimationFrame(run);
    env.win.setTimeout(run, 120);
  }

  private render(): void {
    const c = this.chat;
    const empty = c.tr.messages.length === 0;
    if (this.emptyShown !== empty) { this.emptyShown = empty; this.paintEmpty(); }
    this.feed.update(c.tr, !c.busy);
    this.composer.setBusy(c.busy);
    this.feed.setPending(c.sending && c.tr.running === null ? (c.starting ? `Starting ${BACKEND_LABEL[this.host.getSettings().backend]}` : "Sending") : null);
  }

  private dispose(): void {
    this.chat.flush();
    for (const d of this.disposers.splice(0)) { try { d(); } catch { /* ignore */ } }
    this.composer.dispose();
    this.pendingRender = false;
    void this.chat.closeSession();
    clear(this.root);
  }
}
