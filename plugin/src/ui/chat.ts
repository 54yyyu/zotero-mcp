// The conversation: the transcript state, the agent session behind it, and saving. No DOM here; the panel
// renders `tr` whenever `onChange` fires.
import type { AgentSession, ChatEvent, ContextChip, PanelHost, PanelSettings, SavedSession } from "../types.ts";
import { clip, errMessage, nextId } from "./dom.ts";
import { summary } from "./context.ts";
import { planContext } from "./economy.ts";
import { Persister } from "./persist.ts";
import { applyEvent, emptyTranscript, promptFor, replay } from "./transcript.ts";
import type { TranscriptState } from "./transcript.ts";

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
}

export class Chat {
  tr: TranscriptState = emptyTranscript();
  session: AgentSession | null = null;
  starting: Promise<AgentSession> | null = null;
  saved: SavedSession | null = null;
  sending = false;
  private resume: SavedSession | null = null;
  private unsub: (() => void) | null = null;
  private gen = 0;
  private cwd = "";
  /** Chips this agent session has already read (id -> fingerprint): unchanged ones are not sent again. */
  private sent = new Map<string, string>();
  private persister: Persister;
  private d: ChatDeps;

  constructor(deps: ChatDeps) {
    this.d = deps;
    this.persister = new Persister((ev) => this.write(ev));
  }

  get busy(): boolean { return this.sending || this.tr.running !== null; }

  /** Fold an event into the transcript, ask for a render, and save it. */
  dispatch(ev: ChatEvent, persist = true): void {
    if (ev.t === "notice" && ev.compacted) this.sent.clear(); // the summary may have dropped what was sent
    this.tr = applyEvent(this.tr, ev);
    this.d.onChange();
    if (persist && (this.saved || ev.t === "user")) this.persister.push(ev);
  }

  private write(ev: ChatEvent): void {
    if (!this.saved) return;
    this.saved = { ...this.saved, updatedAt: Date.now(), ...(this.session ? { agentSessionId: this.session.sessionId } : {}) };
    this.d.host.appendEvent(this.saved, ev).catch(() => { /* a failed save must not stop the chat */ });
  }

  async ensureSession(): Promise<AgentSession> {
    if (this.session) return this.session;
    if (this.starting) return this.starting;
    const gen = this.gen;
    const host = this.d.host;
    const p = (async () => {
      const set = host.getSettings();
      const prep = await host.prepareSession(this.resume?.cwd);
      this.cwd = prep.cwd;
      const sess = await host.runtime.start({
        backend: set.backend, cwd: prep.cwd, brief: prep.brief, env: prep.env,
        ...(set.model[set.backend] ? { model: set.model[set.backend] } : {}),
        ...(set.mode[set.backend] ? { mode: set.mode[set.backend] } : {}),
        ...(set.effort[set.backend] ? { effort: set.effort[set.backend] } : {}),
        ...(set.backend !== "pi" ? { auth: set.auth[set.backend] } : {}),
        ...(this.resume?.agentSessionId ? { resumeSessionId: this.resume.agentSessionId } : {}),
      });
      if (gen !== this.gen) { void sess.close().catch(() => {}); throw new Error("cancelled"); }
      this.session = sess;
      this.sent.clear(); // a new or resumed session: everything goes in full once (DESIGN.md "Context budget")
      this.resume = null;
      this.unsub = sess.on((ev) => {
        this.dispatch(ev);
        if (ev.t === "turn_end") { this.d.turnEnded(ev.stop); this.persister.flush(); }
      });
      return sess;
    })();
    this.starting = p;
    p.then(() => {}, () => {}).finally(() => { if (this.starting === p) this.starting = null; this.d.sessionChanged(); this.d.onChange(); });
    return p;
  }

  async closeSession(): Promise<void> {
    const s = this.session;
    this.unsub?.();
    this.unsub = null;
    this.session = null;
    this.starting = null;
    this.gen++;
    if (s) await s.close().catch(() => {});
    this.d.sessionChanged();
  }

  async send(text: string, used: ContextChip[]): Promise<void> {
    if (this.busy || this.d.blockReason()) return;
    const host = this.d.host;
    if (!this.saved) {
      this.saved = { id: nextId("s") + Date.now().toString(36), title: clip(text, 60), backend: host.getSettings().backend, cwd: this.resume?.cwd ?? "", agentSessionId: this.resume?.agentSessionId ?? "", updatedAt: Date.now() };
    }
    this.sending = true;
    this.dispatch({ t: "user", id: nextId("u") + Date.now().toString(36), text, chips: used.map(summary) });
    try {
      const sess = await this.ensureSession();
      this.saved = { ...this.saved, agentSessionId: sess.sessionId, cwd: this.cwd || this.saved.cwd };
      let ctx = { text: "", images: [] as { mime: string; data: string }[] };
      try {
        const next = new Map(this.sent);
        ctx = host.describeContext(planContext(used, next));
        this.sent = next;
      } catch { /* the question still goes */ }
      await sess.prompt({
        text: [ctx.text, text].filter(Boolean).join("\n\n"),
        ...(sess.supportsImages && ctx.images.length ? { images: ctx.images } : {}),
      });
    } catch (e) {
      const msg = errMessage(e);
      if (msg === "cancelled") return;
      const running = this.tr.running;
      if (running) this.dispatch({ t: "turn_end", turn: running, stop: "error" });
      this.dispatch({ t: "notice", level: "error", message: this.session ? `The agent stopped: ${msg}` : `Couldn't start the agent: ${msg}`, hint: "Check Status for what is missing, then send again." });
      // A rejected prompt means the bridge is gone: the next send starts a fresh one and resumes this chat.
      if (this.session) this.resume = this.saved;
      await this.closeSession();
      this.d.setupFailed();
    } finally {
      this.sending = false;
      this.persister.flush();
      this.d.onChange();
    }
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

  /** Change the model, effort or permission mode: on the live session, and as that backend's saved choice. */
  async pick(kind: "model" | "effort" | "mode", id: string): Promise<void> {
    const s = this.d.host.getSettings();
    const backend = this.session?.backend ?? s.backend;
    const live = this.session;
    if (live) await (kind === "model" ? live.setModel(id) : kind === "effort" ? live.setEffort(id) : live.setMode(id));
    await this.d.host.setSettings({ [kind]: { ...s[kind], [backend]: id } } as Partial<PanelSettings>);
  }

  /** A new, empty chat. */
  async reset(): Promise<void> {
    if (this.busy) await this.stop();
    this.persister.flush();
    await this.closeSession();
    this.tr = emptyTranscript();
    this.saved = null;
    this.resume = null;
    this.sending = false;
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
      this.tr = replay([{ t: "notice", level: "error", message: `Couldn't open that chat: ${errMessage(e)}` }]);
    }
    this.saved = s;
    this.resume = s;
    this.d.onChange();
  }

  flush(): void { this.persister.flush(); }
}
