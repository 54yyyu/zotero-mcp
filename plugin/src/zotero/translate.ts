// The translator behind the reader's Translate button (DESIGN.md "Translate"). Loaded with panel.js on the first press.
// One warm, locked session of the user's current backend (StartOpts.locked: no tools, no settings, the translator prompt
// as its whole system prompt), in a folder of its own: never the chat's folder, history or brief. It is started on the first
// request, reused, and closed after IDLE_MS, after MAX_TURNS (its history only grows), or when the backend, model, sign-in
// or language changes. Every permission request is refused.
import type { AgentRuntime, AgentSession, BackendId, Catalog, ChatEvent, PanelSettings } from "../types.ts";
import { pickLowEffort, pickTranslateModel, translatorPrompt } from "../agent/index.ts";
import { language } from "../ui/settings-model.ts";

const IDLE_MS = 5 * 60_000;
const MAX_TURNS = 20;

export interface TranslateRequest {
  text: string;
  /** A language id (LANGUAGES); the setting when omitted. */
  to?: string;
  onText(delta: string): void;
}

export interface Translator {
  /** Streams the translation through onText; rejects with a message fit for the popup. A newer request cancels this one. */
  translate(req: TranslateRequest): Promise<void>;
  /**
   * Start the session now, if Translate is on and none is ready or starting (a selection popup appeared: the press is likely a
   * second away). Costs no tokens; a failed start is silent and not retried for a minute. Never touches a session in use.
   */
  preload(): void;
  /** Stop the translation running, if any (its popup closed). */
  cancel(): void;
  /** Close the session if what it was started for (backend, model, sign-in, language) no longer holds. */
  settingsChanged(): void;
  /** For tests: sessions started, permission requests refused, the live session. */
  stats(): { started: number; refused: number; session: AgentSession | null };
  dispose(): Promise<void>;
}

export function createTranslator(o: {
  runtime: Pick<AgentRuntime, "start">;
  settings(): PanelSettings;
  /** The session's folder (created by the caller): under the plugin's data dir, never the chat folder. */
  cwd(): Promise<string>;
  /** Env for API-key mode (the key from the keychain), {} otherwise. */
  env(s: PanelSettings): Promise<Record<string, string>>;
  /** A catalog already read (the open panel's): lets the first start name its model up front. Never read just for this. */
  known?(b: BackendId): Catalog | undefined;
}): Translator {
  /** `to`: the language it translates into; `setting`: the language setting when it started (a change closes it). */
  type Warm = { key: string; to: string; setting: string; session: AgentSession; turns: number };
  let warm: Warm | null = null;
  let opening: { key: string; p: Promise<Warm> } | null = null;
  let running: AgentSession | null = null;
  let chain: Promise<unknown> = Promise.resolve();
  let idle: ReturnType<typeof setTimeout> | undefined;
  let started = 0, refused = 0, failedAt = 0, disposed = false;
  /** The model picked last time, per backend: a later start names it up front instead of switching after session/new. */
  const picked = new Map<BackendId, string>();

  const keyOf = (s: PanelSettings, to: string) => [s.backend, s.translateModel[s.backend] ?? "", s.model[s.backend] ?? "", s.auth[s.backend], to].join("|");
  const arm = () => { clearTimeout(idle); idle = setTimeout(() => void close(), IDLE_MS); };

  const close = async (): Promise<void> => {
    clearTimeout(idle);
    const w = warm;
    warm = null;
    await w?.session.close().catch(() => undefined);
  };

  async function open(s: PanelSettings, to: string): Promise<AgentSession> {
    const b = s.backend;
    const known = o.known?.(b);
    const model = s.translateModel[b] || picked.get(b) || (known ? pickTranslateModel(known.models, s.model[b]) : undefined);
    const session = await o.runtime.start({
      backend: b, cwd: await o.cwd(), brief: translatorPrompt(language(to).name), locked: true, ephemeral: true,
      auth: s.auth[b], env: await o.env(s), ...(model ? { model } : {}),
    });
    started++;
    // Only what could not be named up front: the model (first start, no catalog known), then the lightest reasoning.
    // No permission mode: a locked session has no tools (Claude, pi) or a read-only sandbox (Codex).
    const pick = model ?? pickTranslateModel(session.models(), s.model[b]);
    if (pick && pick !== session.currentModel()) await session.setModel(pick).catch(() => undefined);
    if (pick && !s.translateModel[b]) picked.set(b, pick);
    const effort = pickLowEffort(session.efforts());
    if (effort && effort !== session.currentEffort()) await session.setEffort(effort).catch(() => undefined);
    session.on((ev) => {
      if (ev.t !== "permission" || ev.resolved) return;
      refused++;
      session.respondPermission(ev.id, ev.options.find((x) => x.kind.startsWith("reject"))?.id ?? null);
    });
    return session;
  }

  /** The ready session for these settings and language; one start at a time, shared by a preload and the press. */
  async function session(s: PanelSettings, to: string): Promise<Warm> {
    const key = keyOf(s, to);
    if (opening && opening.key !== key) await opening.p.catch(() => undefined);
    if (warm && (warm.key !== key || warm.turns >= MAX_TURNS)) void close();
    if (warm) return warm;
    if (!opening) {
      const p = open(s, to).then(async (session) => {
        if (disposed) { await session.close(); throw new Error("closed"); }
        arm();
        return (warm = { key, to, setting: s.translateTo, session, turns: 0 });
      });
      opening = { key, p };
      void p.finally(() => { if (opening?.p === p) opening = null; }).catch(() => undefined);
    }
    return opening.p;
  }

  async function run(req: TranslateRequest): Promise<void> {
    const s = o.settings();
    const to = req.to ?? s.translateTo;
    clearTimeout(idle);
    let w: Warm;
    try {
      w = await session(s, to);
    } catch (e) {
      throw new Error(`The agent could not start: ${e instanceof Error ? e.message : String(e)}`); // node missing, signed out, no answer
    }
    const turn = w.session;
    let text = "", notice = "", stop = "";
    const off = turn.on((ev: ChatEvent) => {
      if (ev.t === "text") { text += ev.delta; req.onText(ev.delta); }
      else if (ev.t === "notice" && ev.level !== "info") notice ||= [ev.message, ev.hint?.replace(/,? then start a new chat\.?$/, ".")].filter(Boolean).join(" ");
      else if (ev.t === "turn_end") stop = ev.stop;
    });
    running = turn;
    try {
      await turn.prompt({ text: req.text });
      w.turns++;
    } catch (e) {
      if (warm?.session === turn) void close(); // the bridge died: the next press starts a fresh one
      throw new Error(notice || `The agent stopped: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      off();
      running = null;
      if (warm) arm();
    }
    if (stop === "cancelled") return;
    if (stop === "error") throw new Error(notice || "The agent could not translate this.");
    if (stop === "refusal") throw new Error("The model declined to translate this.");
    if (!text.trim()) throw new Error("No translation came back. Try again, or choose another model in Settings.");
  }

  return {
    translate(req) {
      // One at a time: a newer selection's request stops the one running (its popup is gone).
      if (running) void running.cancel();
      const p = chain.then(() => run(req));
      chain = p.catch(() => undefined);
      return p;
    },
    preload() {
      const s = o.settings();
      if (disposed || !s.translate || warm || opening || running || Date.now() - failedAt < 60_000) return;
      session(s, s.translateTo).catch(() => { failedAt = Date.now(); });
    },
    cancel() {
      if (running) void running.cancel();
    },
    settingsChanged() {
      const s = o.settings();
      if (warm && (warm.key !== keyOf(s, warm.to) || warm.setting !== s.translateTo)) void close();
    },
    stats: () => ({ started, refused, session: warm?.session ?? null }),
    dispose() { disposed = true; return close(); },
  };
}
