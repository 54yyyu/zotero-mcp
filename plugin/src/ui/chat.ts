// The conversation: the transcript state, the agent session behind it, and saving. No DOM here; the panel
// renders `tr` whenever `onChange` fires.
import type { AgentSession, BackendId, ChatEvent, ContextChip, PanelHost, PanelSettings, SavedSession } from "../types.ts";
import { cleanTitle, clip, errMessage, nextId } from "./dom.ts";
import { summary } from "./context.ts";
import { chipHash, planContext } from "./economy.ts";
import { Persister } from "./persist.ts";
import { ownProviders } from "./settings-model.ts";
import { applyEvent, emptyTranscript, promptFor, replay } from "./transcript.ts";
import type { TranscriptState } from "./transcript.ts";
import { t } from "../i18n.ts";

export interface ChatDeps {
  host: PanelHost;
  /** The transcript or the busy state changed: render (the panel batches). */
  onChange(): void;
  /** Why sending is impossible right now, or null. */
  blockReason(): string | null;
  /** A turn ended (for the screen-reader announcement). */
  turnEnded(stop: string): void;
  /** The session died or failed to start: look at the setup again. */
  setupFailed(): void;
  /** The session or its choices changed (the pickers show them). */
  sessionChanged(): void;
  /** What the agent reads for what the user typed (`/skill rest` becomes a plain instruction), in the agent's folder `cwd`. */
  expand(text: string, cwd: string): Promise<string>;
  /** Which chips count as already shown changed (a send, a new chat, a compaction): repaint the composer's chips. */
  shownChanged(): void;
}

export class Chat {
  tr: TranscriptState = emptyTranscript();
  session: AgentSession | null = null;
  starting: Promise<AgentSession> | null = null;
  saved: SavedSession | null = null;
  sending = false;
  /** A `/compact` the user asked for is running (sending is set too). */
  compacting = false;
  /** The chips the last message carried, marked `repeat` when only named (the context popover lists them). */
  lastContext: ContextChip[] = [];
  private resume: SavedSession | null = null;
  private unsub: (() => void) | null = null;
  private gen = 0;
  /** The send or /compact under way. A new chat (reset) drops it: what that call still does when it comes back is no longer this chat's. */
  private run: object | null = null;
  /** Sessions being closed: the next one starts once they are gone (a resumed chat's own old process, above all). Never awaited by the UI. */
  private closing: Promise<unknown> = Promise.resolve();
  private cwd = "";
  /** Chips this agent session has already read (id -> fingerprint): unchanged ones are not sent again. */
  private sent = new Map<string, string>();
  /** Chips this chat has shown in a sent message (id -> fingerprint): the composer stops showing them (isShown). */
  private shown = new Map<string, string>();
  private persister: Persister;
  private d: ChatDeps;

  constructor(deps: ChatDeps) {
    this.d = deps;
    this.persister = new Persister((ev) => this.write(ev));
  }

  /**
   * Already shown in a message of this chat. Where the user is (the open paper, a selected item or collection) counts
   * once per item, whatever page they are on; what they point at (a selection, an area, a picture) until it changes.
   */
  isShown(c: ContextChip): boolean {
    const h = this.shown.get(c.id);
    return h !== undefined && (c.kind === "reader" || c.kind === "item" || c.kind === "collection" || h === chipHash(c));
  }

  get busy(): boolean { return this.sending || this.tr.running !== null; }

  /** Fold an event into the transcript, ask for a render, and save it. */
  dispatch(ev: ChatEvent, persist = true): void {
    if (ev.t === "notice" && ev.compacted) { this.sent.clear(); this.shown.clear(); this.d.shownChanged(); } // the summary may have dropped what was sent
    this.tr = applyEvent(this.tr, ev);
    this.d.onChange();
    if (persist && (this.saved || ev.t === "user")) this.persister.push(ev);
  }

  private write(ev: ChatEvent): void {
    if (!this.saved) return;
    this.saved = { ...this.saved, updatedAt: Date.now(), ...(this.session ? { agentSessionId: this.session.sessionId } : {}) };
    this.d.host.appendEvent(this.saved, ev).catch(() => { /* a failed save must not stop the chat */ });
  }

  /** The agent a start would use now: a saved chat is resumed on the agent it was made with, a new one uses the setting. */
  get backend(): BackendId {
    return this.session?.backend ?? (this.resume?.agentSessionId ? this.resume.backend : this.d.host.getSettings().backend);
  }

  async ensureSession(): Promise<AgentSession> {
    if (this.session) return this.session;
    if (this.starting) return this.starting;
    const gen = this.gen;
    const host = this.d.host;
    /** A new chat or a changed agent replaced this start while it ran: its session, if it got one, is closed. */
    const current = (sess?: AgentSession) => {
      if (gen === this.gen) return;
      void sess?.close().catch(() => {});
      throw new Error("cancelled");
    };
    const p = (async () => {
      await this.closing;
      current();
      const resume = this.resume?.agentSessionId ? this.resume : null;
      const folder = this.resume?.cwd;
      const start = async (set: PanelSettings, b: BackendId, resumeSessionId?: string) => {
        const prep = await host.prepareSession(folder, b);
        current();
        this.cwd = prep.cwd;
        return host.runtime.start({
          backend: b, cwd: prep.cwd, brief: prep.brief, env: prep.env,
          ...(set.model[b] ? { model: set.model[b] } : {}),
          ...(set.mode[b] ? { mode: set.mode[b] } : {}),
          ...(set.effort[b] ? { effort: set.effort[b] } : {}),
          ...(!ownProviders(b) ? { auth: set.auth[b] } : {}),
          ...(resumeSessionId ? { resumeSessionId } : {}),
        });
      };
      let set = host.getSettings();
      let sess: AgentSession;
      let resumed = !!resume;
      try {
        sess = await start(set, resume ? resume.backend : set.backend, resume?.agentSessionId);
      } catch (e) {
        current();
        // The agent no longer has this conversation (it prunes old ones), cannot load one, or the chat was made with an
        // agent that does not start now: go on in a fresh session, once, and say so. Any other failure is reported as it is,
        // and the chat can still be resumed when it is fixed.
        if (!resume || !((e as Error | null)?.name === "ResumeError" || resume.backend !== set.backend)) throw e;
        resumed = false;
        sess = await start(set, set.backend);
        current(sess);
        this.resume = null;
        this.dispatch({ t: "notice", level: "info", message: t("chat.resumeFresh") });
      }
      current(sess);
      // The settings may have changed while this started (the panel starts a session as soon as it opens).
      const now = host.getSettings();
      if (!resumed && now.backend !== sess.backend) {
        await sess.close().catch(() => {});
        current();
        sess = await start((set = now), now.backend);
        current(sess);
      } else {
        const b = sess.backend;
        if (now.model[b] !== set.model[b]) { const m = now.model[b] || sess.defaultModel(); if (m && m !== sess.currentModel()) await sess.setModel(m).catch(() => {}); }
        if (now.mode[b] && now.mode[b] !== set.mode[b] && now.mode[b] !== sess.currentMode()) await sess.setMode(now.mode[b]).catch(() => {});
        if (now.effort[b] && now.effort[b] !== set.effort[b] && now.effort[b] !== sess.currentEffort()) await sess.setEffort(now.effort[b]).catch(() => {});
        current(sess);
      }
      this.session = sess;
      this.sent.clear(); // a new or resumed session: everything goes in full once (DESIGN.md "Context budget")
      this.resume = null;
      this.unsub = sess.on((ev) => {
        if (ev.t === "session_title") this.takeAgentTitle(ev.title); // before dispatch: the persist writes the new title to the index
        this.dispatch(ev);
        if (ev.t === "turn_end") { this.d.turnEnded(ev.stop); this.persister.flush(); }
      });
      return sess;
    })();
    this.starting = p;
    p.then(() => {}, () => {}).finally(() => { if (this.starting === p) this.starting = null; this.d.sessionChanged(); this.d.onChange(); });
    return p;
  }

  /** Lets go of the session at once; the process is stopped behind (resolves when it is gone). Nothing in the UI waits for that. */
  closeSession(): Promise<void> {
    const s = this.session;
    this.unsub?.();
    this.unsub = null;
    this.session = null;
    this.starting = null;
    this.gen++;
    const before = this.closing;
    const done = (s ? s.close().catch(() => {}) : Promise.resolve()).then(() => { this.d.sessionChanged(); });
    this.closing = done.then(() => before);
    return done;
  }

  async send(text: string, used: ContextChip[]): Promise<void> {
    if (this.busy || this.d.blockReason()) return;
    const host = this.d.host;
    if (!this.saved) {
      const set = host.getSettings();
      // The chat's name starts on the first filled rung of its source's chain: agent summary can never be ready
      // here (it arrives during the turn), so agent falls to the focused paper, then the question. The source is
      // snapshotted onto the chat: changing the setting later never re-rules an existing conversation.
      const titleSource = set.titleSource;
      const paper = titleSource === "question" ? null : host.paperTitle(used);
      this.saved = { id: nextId("s") + Date.now().toString(36), title: paper ? clip(paper, 80) : clip(text, 60), titleSource, backend: set.backend, cwd: this.resume?.cwd ?? "", agentSessionId: this.resume?.agentSessionId ?? "", updatedAt: Date.now() };
    }
    this.sending = true;
    const run = (this.run = {});
    const mine = () => this.run === run;
    // The bubble shows what the composer showed: context already in this chat is only named to the agent, not shown again.
    const fresh = used.filter((c) => !this.isShown(c));
    for (const c of used) this.shown.set(c.id, chipHash(c));
    this.d.shownChanged();
    this.dispatch({ t: "user", id: nextId("u") + Date.now().toString(36), text, chips: fresh.map(summary) });
    try {
      const starting = this.ensureSession();
      // The focused paper's metadata and full-text file, prepared while the session starts (never longer than 1.5 s after).
      const paper = host.paperContext(used, starting).catch(() => [] as ContextChip[]);
      const sess = await starting;
      if (!mine() || !this.saved) return;
      this.saved = { ...this.saved, backend: sess.backend, agentSessionId: sess.sessionId, cwd: this.cwd || this.saved.cwd };
      let ctx = { text: "", images: [] as { mime: string; data: string }[] };
      const extra = await paper;
      if (!mine()) return;
      try {
        const next = new Map(this.sent);
        const plan = planContext([...used, ...extra], next);
        ctx = host.describeContext(plan);
        this.sent = next;
        this.lastContext = plan;
      } catch { /* the question still goes */ }
      if (ctx.images.length && !sess.supportsImages) this.dispatch({ t: "notice", level: "warn", message: t("chat.noImages") });
      const said = await this.d.expand(text, this.cwd);
      if (!mine()) return;
      await sess.prompt({
        text: [ctx.text, said].filter(Boolean).join("\n\n"),
        ...(sess.supportsImages && ctx.images.length ? { images: ctx.images } : {}),
      });
    } catch (e) {
      // A new chat was started (or the panel closed) while this one was answering: closing its session is what rejected
      // the prompt. That is not a failure of the chat left behind, and nothing of it belongs to the new one.
      if (mine()) await this.fail(e);
    } finally {
      if (mine()) {
        this.run = null;
        this.sending = false;
        this.persister.flush();
        this.d.onChange();
      }
    }
  }

  /** The agent named the conversation (ACP `session_info_update`): it replaces the fallback name whenever it arrives — unless the chat runs under another naming source, or the user has named it themselves. */
  private takeAgentTitle(title: string): void {
    const s = this.saved;
    const name = cleanTitle(title); // agent text: control characters and length are ours to bound (it renders as text, never markup)
    if (!s || !name || s.titleLocked || (s.titleSource === "paper" || s.titleSource === "question")) return;
    this.saved = { ...s, title: name };
  }

  /** A user rename, from the chat header or the history list: it wins over every automatic source, permanently. Empty changes nothing. */
  async rename(title: string): Promise<void> {
    const s = this.saved;
    const name = cleanTitle(title);
    if (!s || !name || (s.title === name && s.titleLocked)) return;
    // Take it in memory first: the transcript saver rewrites this chat's index entry from `this.saved` while the
    // answer streams, and a write queued behind the rename must not put the old title (and no lock) back.
    this.saved = { ...s, title: name, titleLocked: true };
    this.d.onChange();
    try { await this.d.host.renameSession(s.id, name); }
    catch (e) { if (this.saved?.id === s.id) this.saved = s; this.d.onChange(); throw e; }
  }

  /** A rename done elsewhere (the history list): when it is this open chat's, mirror it, so the header stays in step. */
  adoptRename(id: string, title: string): void {
    const s = this.saved;
    if (!s || s.id !== id) return;
    this.saved = { ...s, title, titleLocked: true };
    this.d.onChange();
  }

  /** Summarise the conversation now (the bridge's own /compact, when it has a verified one); a notice tells the result. */
  async compact(): Promise<void> {
    const s = this.session;
    if (!s?.canCompact || this.busy) return;
    this.sending = this.compacting = true;
    const run = (this.run = {});
    this.d.onChange();
    try {
      await s.compact();
    } catch (e) {
      if (this.run === run) await this.fail(e);
    } finally {
      if (this.run === run) {
        this.run = null;
        this.sending = this.compacting = false;
        this.persister.flush();
        this.d.onChange();
      }
    }
  }

  private async fail(e: unknown): Promise<void> {
    const msg = errMessage(e);
    if (msg === "cancelled") return;
    const running = this.tr.running;
    if (running) this.dispatch({ t: "turn_end", turn: running, stop: "error" });
    this.dispatch({ t: "notice", level: "error", message: this.session ? t("chat.stopped", { error: msg }) : t("chat.startFailed", { error: msg }), hint: t("chat.checkStatus") });
    // A rejected prompt means the bridge is gone: the next send starts a fresh one and resumes this chat.
    if (this.session) this.resume = this.saved;
    void this.closeSession();
    this.d.setupFailed();
  }

  async stop(): Promise<void> {
    const s = this.session;
    if (!s) return;
    // A question nobody answered is cancelled with the turn.
    const live = this.tr.messages.find((m) => m.role === "assistant" && m.id === this.tr.running);
    if (live && live.role === "assistant") for (const b of live.blocks) if (b.type === "permission" && b.resolved === undefined) s.respondPermission(b.id, null);
    try { await s.cancel(); } catch { /* the turn_end tells the story */ }
  }

  /** Ask again: the same text and chips as the message an answer replies to. */
  retry(assistantId: string): void {
    const u = promptFor(this.tr, assistantId);
    if (!u || this.busy) return;
    void this.send(u.text, u.chips.map((c) => ({ ...c, auto: false, pinned: false })));
  }

  answerPermission(turn: string, permId: string, optionId: string): void {
    const m = this.tr.messages.find((x) => x.role === "assistant" && x.id === turn);
    const b = m && m.role === "assistant" ? m.blocks.find((x) => x.type === "permission" && x.id === permId) : undefined;
    if (!b || b.type !== "permission") return;
    this.session?.respondPermission(permId, optionId);
    this.dispatch({ t: "permission", turn, id: b.id, title: b.title, kind: b.kind, ...(b.input !== undefined ? { input: b.input } : {}), options: b.options, resolved: optionId, ...(b.name ? { name: b.name } : {}) });
  }

  /** Change the model, effort or permission mode: save `id` as that backend's choice ("" = the agent's default) and apply `apply` to the live session, unless it is already on it. */
  async pick(kind: "model" | "effort" | "mode", id: string, apply: string | undefined = id): Promise<void> {
    const s = this.d.host.getSettings();
    const backend = this.session?.backend ?? s.backend;
    const live = this.session;
    const now = kind === "model" ? live?.currentModel() : kind === "effort" ? live?.currentEffort() : live?.currentMode();
    if (live && apply && apply !== now) await (kind === "model" ? live.setModel(apply) : kind === "effort" ? live.setEffort(apply) : live.setMode(apply));
    await this.d.host.setSettings({ [kind]: { ...s[kind], [backend]: id } } as Partial<PanelSettings>);
  }

  /**
   * A new, empty chat, at once. The chat left behind is saved as it stood (an answer still streaming ends "stopped"), and
   * its agent is stopped behind: nothing here waits for the process to die, and what a send still under way does when it
   * comes back is dropped (`run`).
   */
  async reset(): Promise<void> {
    this.leave();
    this.tr = emptyTranscript();
    this.saved = null;
    this.resume = null;
    this.lastContext = [];
    this.shown.clear();
    this.d.shownChanged();
    this.d.onChange();
  }

  /** Open a saved chat; the agent is started again (and resumes it) on the next send. */
  async load(s: SavedSession): Promise<void> {
    await this.reset();
    try {
      this.tr = replay(await this.d.host.loadEvents(s.id));
      // A save that ends mid-turn (a crash) must not look like it is still streaming.
      if (this.tr.running) this.tr = applyEvent(this.tr, { t: "turn_end", turn: this.tr.running, stop: "cancelled" });
    } catch (e) {
      this.tr = replay([{ t: "notice", level: "error", message: t("chat.loadFailed", { error: errMessage(e) }) }]);
    }
    this.saved = s;
    this.resume = s;
    this.d.onChange();
  }

  /** The saved chats were all deleted: this one is no longer one of them, and nothing of it still waiting is written. */
  forget(): void {
    this.persister.drop();
    this.saved = null;
    this.resume = null;
  }

  /** The chat is being left (a new chat, the panel going away): save it as it stands, and let its session go. */
  private leave(): void {
    this.run = null;
    this.sending = this.compacting = false;
    const running = this.tr.running;
    if (running && this.saved) this.dispatch({ t: "turn_end", turn: running, stop: "cancelled" });
    this.persister.flush();
    void this.closeSession();
  }

  /** The panel is going away: the chat is saved as it stands and its agent stopped; resolves when the process is gone. */
  dispose(): Promise<void> {
    this.leave();
    return this.closing.then(() => {});
  }
}
