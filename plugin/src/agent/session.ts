// One conversation with an agent: the ACP session on top of an AcpClient, translated into ChatEvents.
//
// Rules, each from a measured bridge or a bug in the reference implementation:
// - Only updates that arrive while a turn is running become live events. History replayed by
//   session/load (and the startup banners some bridges send after session/new) arrive with no turn
//   running and are dropped.
// - Tool names come from `_meta.claudeCode.toolName`, joined on toolCallId; a permission frame may
//   not carry one. Titles are never parsed for a name.
// - An agent->client request that goes unanswered stalls the agent: permission requests are answered
//   through respondPermission(), cancel() answers all pending ones "cancelled", close() too.
// - The model is sent explicitly after session/new / session/load (a bridge's default may be old).
import type { AgentSession, BackendId, ChatEvent, ModeOption, ModelOption, PermissionOption, PromptInput, StartOpts, ToolKind, ToolStatus, Usage } from "../types.ts";
import type { AcpClient, InitializeResult } from "./acp.ts";
import type { BackendSpec } from "./backends.ts";
import { withBrief } from "./brief.ts";
import { JsonRpcError, MethodNotFound } from "./jsonrpc.ts";

type Obj = Record<string, unknown>;
type ToolEvent = Extract<ChatEvent, { t: "tool" }> & { name?: string };
type PermissionEvent = Extract<ChatEvent, { t: "permission" }> & { name?: string };

/** Models, modes and efforts a bridge offers, as session/new (or a later update) last said. */
interface Offers {
  models: ModelOption[];
  currentModel?: string;
  /** The config option that carries the model (claude: "model"), when the bridge has one. */
  modelConfigId?: string;
  modes: ModeOption[];
  currentMode?: string;
  efforts: ModeOption[];
  currentEffort?: string;
  /** The config option that carries the effort (claude "effort", pi "thought_level"); absent for suffix-style (codex). */
  effortConfigId?: string;
  /** Suffix-style backends (codex): the levels each base model comes in, in the bridge's order. */
  levels?: Map<string, string[]>;
}

const TOOL_KINDS: ToolKind[] = ["read", "edit", "delete", "move", "search", "execute", "think", "fetch", "switch_mode", "other"];
const OUTPUT_CAP = 20_000;

function asObj(v: unknown): Obj | undefined {
  return v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Obj) : undefined;
}
function str(v: unknown): string | undefined {
  return typeof v === "string" && v !== "" ? v : undefined;
}
/** The objects in a JSON array (anything else is skipped). */
function objs(v: unknown): Obj[] {
  return Array.isArray(v) ? v.map(asObj).filter((o): o is Obj => !!o) : [];
}
/** The text of a `{type:"text", text}` content block. */
function textOf(content: unknown): string | undefined {
  const c = asObj(content);
  return c?.["type"] === "text" ? str(c["text"]) : undefined;
}

function toolKindOf(v: unknown): ToolKind | undefined {
  return typeof v === "string" && (TOOL_KINDS as string[]).includes(v) ? (v as ToolKind) : undefined;
}

function toolStatusOf(v: unknown): ToolStatus | undefined {
  switch (v) {
    case "pending":
      return "pending";
    case "in_progress":
      return "running";
    case "completed":
      return "done";
    case "failed":
      return "failed";
    default:
      return undefined;
  }
}

function stopOf(reason: unknown): Extract<ChatEvent, { t: "turn_end" }>["stop"] {
  switch (reason) {
    case "cancelled":
      return "cancelled";
    case "refusal":
      return "refusal";
    case "max_tokens":
    case "max_turn_requests":
      return "max_tokens";
    default:
      return "end_turn";
  }
}

/** `_meta.claudeCode.toolName`, or a plain `name` some bridges send. */
function toolNameOf(update: Obj): string | undefined {
  const vendor = asObj(asObj(update["_meta"])?.["claudeCode"])?.["toolName"];
  return str(vendor) ?? str(update["name"]);
}

function cap(s: string): string {
  return s.length > OUTPUT_CAP ? s.slice(0, OUTPUT_CAP) + "\n…(truncated)" : s;
}

/** The text a tool call's `content` (ACP ToolCallContent[]) amounts to. */
function outputOf(update: Obj): string | undefined {
  const parts: string[] = [];
  for (const o of objs(update["content"])) {
    if (o["type"] === "content") {
      const t = textOf(o["content"]);
      if (t) parts.push(t);
    } else if (o["type"] === "diff" && typeof o["path"] === "string") {
      parts.push(`${o["path"]} (edited)`);
    }
  }
  if (parts.length === 0) {
    // Claude's PostToolUse result: Bash {stdout, stderr}, or a bare string.
    const resp = asObj(asObj(update["_meta"])?.["claudeCode"])?.["toolResponse"];
    if (typeof resp === "string") parts.push(resp);
    else if (asObj(resp)) {
      const out = [str(asObj(resp)!["stdout"]), str(asObj(resp)!["stderr"])].filter((x): x is string => !!x).join("\n");
      if (out) parts.push(out);
    }
  }
  const text = parts.join("\n");
  return text ? cap(text) : undefined;
}

function isEmptyInput(v: unknown): boolean {
  if (v === undefined || v === null) return true;
  const o = asObj(v);
  return o !== undefined && Object.keys(o).length === 0;
}

// ───────────────────────────── catalog (models, modes, efforts) ─────────────────────────────

/** `skipDefault`: the model list's "Default (recommended)" is an alias of another entry, not a choice; a mode called "default" is real. */
function readOptions(list: unknown, skipDefault: boolean): ModelOption[] {
  const out: ModelOption[] = [];
  for (const o of objs(list)) {
    const id = str(o["value"]) ?? str(o["modelId"]) ?? str(o["id"]);
    if (!id || (skipDefault && id === "default")) continue;
    out.push({ id, name: str(o["name"]) ?? id, ...(str(o["description"]) ? { description: str(o["description"])! } : {}) });
  }
  return out;
}

function configOption(res: Obj, category: string): Obj | undefined {
  return objs(res["configOptions"]).find((o) => o["category"] === category);
}

/** `gpt-6-luna[high]` -> ["gpt-6-luna", "high"]. */
function splitLevel(id: string): [string, string] | null {
  const m = /^(.+)\[([^\]]+)\]$/.exec(id);
  return m ? [m[1]!, m[2]!] : null;
}

function readCatalog(res: Obj, backend: BackendSpec): Offers {
  const cat: Offers = { models: [], modes: [], efforts: [] };
  const modelOpt = configOption(res, "model");
  const modelsBlock = asObj(res["models"]);
  const fromConfig = () => {
    if (!modelOpt) return;
    cat.models = readOptions(modelOpt["options"], true);
    cat.currentModel = str(modelOpt["currentValue"]);
    cat.modelConfigId = str(modelOpt["id"]);
  };
  const fromModels = () => {
    if (!modelsBlock) return;
    cat.models = readOptions(modelsBlock["availableModels"], true);
    cat.currentModel = str(modelsBlock["currentModelId"]);
  };
  if (backend.modelVia === "set_model") {
    fromModels();
    if (!cat.models.length) fromConfig();
    else cat.modelConfigId = str(modelOpt?.["id"]);
  } else {
    fromConfig();
    if (!cat.models.length) fromModels();
  }
  if (backend.effortVia === "model-suffix") {
    // Codex has no effort option: its model ids carry the level ("gpt-6-luna[high]"). Models are the
    // unique bases, efforts the levels found, and set_model gets them recombined.
    const levels = new Map<string, string[]>();
    const bases: ModelOption[] = [];
    const all: string[] = [];
    for (const m of cat.models) {
      const x = splitLevel(m.id);
      const base = x ? x[0] : m.id;
      if (!levels.has(base)) {
        levels.set(base, []);
        const gist = m.description?.split(". ")[0];
        bases.push({ id: base, name: m.name.replace(/\s*\([^)]*\)\s*$/, ""), ...(gist ? { description: gist } : {}) });
      }
      if (x) {
        levels.get(base)!.push(x[1]);
        if (!all.includes(x[1])) all.push(x[1]);
      }
    }
    const cur = cat.currentModel ? splitLevel(cat.currentModel) : null;
    if (cur) {
      cat.currentModel = cur[0];
      cat.currentEffort = cur[1];
    }
    // The bridge's default model may be missing from its own list (measured on codex-acp 2.1.1: the account's
    // gpt-6-sol is current and in the model option, but has no entries in `availableModels` and cannot be
    // set_model'd). Show it anyway, with the one level it is on.
    if (cur && !levels.has(cur[0])) {
      levels.set(cur[0], [cur[1]]);
      const named = readOptions(modelOpt?.["options"], true).find((o) => o.id === cur[0]);
      bases.unshift({ id: cur[0], name: named?.name ?? cur[0] });
      if (!all.includes(cur[1])) all.push(cur[1]);
    }
    cat.models = bases;
    cat.efforts = all.map((id) => ({ id, name: id }));
    cat.levels = levels;
  } else {
    const effortOpt = configOption(res, "thought_level");
    if (effortOpt) {
      cat.efforts = readOptions(effortOpt["options"], true);
      cat.currentEffort = str(effortOpt["currentValue"]);
      cat.effortConfigId = str(effortOpt["id"]);
    }
  }
  if (backend.permissionModes) {
    const modes = asObj(res["modes"]);
    if (modes) {
      cat.modes = readOptions(modes["availableModes"], false);
      cat.currentMode = str(modes["currentModeId"]);
    }
    if (!cat.modes.length) {
      const modeOpt = configOption(res, "mode");
      if (modeOpt) {
        cat.modes = readOptions(modeOpt["options"], false);
        cat.currentMode = str(modeOpt["currentValue"]);
      }
    }
  }
  return cat;
}

// ───────────────────────────── the session ─────────────────────────────

interface ToolState {
  title: string;
  kind: ToolKind;
  status: ToolStatus;
  input?: unknown;
  output?: string;
  name?: string;
}

interface PendingPermission {
  event: PermissionEvent;
  resolve: (response: Obj) => void;
}

export class AcpAgentSession implements AgentSession {
  readonly sessionId: string;
  readonly backend: BackendId;
  readonly supportsImages: boolean;
  #client: AcpClient;
  #spec: BackendSpec;
  /** The brief to put ahead of the first prompt (bridges with no system-prompt hook), until it is sent. */
  #firstPromptBrief: string | undefined;
  #catalog: Offers;
  #listeners = new Set<(ev: ChatEvent) => void>();
  #turn: { id: string; tools: Map<string, ToolState>; costBase: number; produced: boolean } | undefined;
  #turnCount = 0;
  #permCount = 0;
  #pending = new Map<string, PendingPermission>();
  #cost = 0;
  /** The context window's fill from the last usage_update (claude-agent-acp, codex-acp and pi-acp all send used/size). */
  #ctx: { used: number; size: number } | undefined;
  #compactions = new Set<string>();
  #lastUpdateAt = Date.now();
  /** pi-acp's startup banner (from session/new `_meta.piAcp.startupInfo`): it may be echoed as a message chunk. */
  #banner: string | undefined;
  #closed = false;
  #cancelled = false;

  private constructor(client: AcpClient, spec: BackendSpec, sessionId: string, init: InitializeResult, catalog: Offers, firstPromptBrief: string | undefined) {
    this.#client = client;
    this.#spec = spec;
    this.backend = spec.id;
    this.sessionId = sessionId;
    this.supportsImages = init.agentCapabilities?.promptCapabilities?.image === true;
    this.#catalog = catalog;
    this.#firstPromptBrief = firstPromptBrief;
    client.onUpdate((update, sid) => {
      if (sid === undefined || sid === this.sessionId) this.#onUpdate(update);
    });
    client.handleRequests((method, params) => this.#onRequest(method, params));
  }

  /**
   * Handshake, then session/new (or session/load), then the explicit model and mode. The caller owns
   * `client`: on a rejection it is closed here.
   */
  static async open(client: AcpClient, spec: BackendSpec, opts: StartOpts): Promise<AcpAgentSession> {
    try {
      const init = await client.initialize();
      // A probe (catalog) must not litter the user's own session list: Claude's SDK takes persistSession: false.
      const meta = spec.briefVia === "system-prompt"
        ? { _meta: { systemPrompt: { append: opts.brief }, ...(opts.ephemeral ? { claudeCode: { options: { persistSession: false } } } : {}) } }
        : {};
      let sessionId: string;
      let res: Obj;
      if (opts.resumeSessionId) {
        // Replayed history arrives as session/update notifications before this resolves; no turn is
        // running, so none of it is emitted.
        res = asObj(await client.request("session/load", { sessionId: opts.resumeSessionId, cwd: opts.cwd, mcpServers: [], ...meta })) ?? {};
        sessionId = opts.resumeSessionId;
      } else {
        res = asObj(await client.request("session/new", { cwd: opts.cwd, mcpServers: [], ...meta })) ?? {};
        const id = str(res["sessionId"]);
        if (!id) throw new Error("the agent did not return a sessionId from session/new");
        sessionId = id;
      }
      const session = new AcpAgentSession(client, spec, sessionId, init, readCatalog(res, spec), spec.briefVia === "first-prompt" && !opts.resumeSessionId ? opts.brief : undefined);
      session.#banner = str(asObj(asObj(res["_meta"])?.["piAcp"])?.["startupInfo"]);
      if (opts.model && opts.model !== session.currentModel()) {
        // A model this bridge does not know must not make the chat unusable: keep its default.
        await session.setModel(opts.model).catch(() => undefined);
      }
      if (opts.mode && opts.mode !== session.currentMode() && session.modes().some((m) => m.id === opts.mode)) {
        await session.setMode(opts.mode);
      }
      if (opts.effort && opts.effort !== session.currentEffort() && session.efforts().some((e) => e.id === opts.effort)) {
        await session.setEffort(opts.effort).catch(() => undefined);
      }
      await session.#settle();
      return session;
    } catch (e) {
      await client.close();
      throw e;
    }
  }

  /**
   * Bridges announce things right after session/new / session/load (commands, a banner, replayed
   * history) and nothing orders those against our first prompt. Wait for a short quiet spell so they
   * land while no turn runs and are dropped, rather than showing up inside the first answer.
   */
  async #settle(quietMs = 60, maxMs = 500): Promise<void> {
    const start = Date.now();
    while (Date.now() - this.#lastUpdateAt < quietMs && Date.now() - start < maxMs) await new Promise((r) => setTimeout(r, 15));
  }

  get account(): string | undefined {
    return this.#client.account;
  }

  models(): ModelOption[] {
    return this.#catalog.models;
  }
  currentModel(): string | undefined {
    return this.#catalog.currentModel;
  }
  modes(): ModeOption[] {
    return this.#catalog.modes;
  }
  currentMode(): string | undefined {
    return this.#catalog.currentMode;
  }

  efforts(): ModeOption[] {
    return this.#catalog.efforts;
  }
  currentEffort(): string | undefined {
    return this.#catalog.currentEffort;
  }

  /** What a bridge's fresh `configOptions` / `modes` change in our view. `models`: take the model list and current model too. */
  #take(fresh: Offers, models: boolean): void {
    const c = this.#catalog;
    if (models && fresh.models.length) {
      c.models = fresh.models;
      if (fresh.currentModel) c.currentModel = fresh.currentModel;
    }
    if (fresh.modes.length) {
      c.modes = fresh.modes;
      if (fresh.currentMode) c.currentMode = fresh.currentMode;
    }
    // configOptions are always the full list: no effort option there means this model has none.
    if (this.#spec.effortVia === "config_option") {
      c.efforts = fresh.efforts;
      c.effortConfigId = fresh.effortConfigId;
      c.currentEffort = fresh.currentEffort;
    }
  }

  /** The id set_model wants. Suffix-style (codex): base + level, keeping `want` when this model has it, else its default (medium, else the first). */
  #wire(id: string, want: string | undefined): { base: string; level?: string; wire: string } {
    if (this.#spec.effortVia !== "model-suffix") return { base: id, wire: id };
    const x = splitLevel(id);
    const base = x ? x[0] : id;
    const have = this.#catalog.levels?.get(base);
    if (!have?.length) return { base, ...(x ? { level: x[1] } : {}), wire: id };
    const asked = x ? x[1] : want;
    const level = asked && have.includes(asked) ? asked : have.includes("medium") ? "medium" : have[0]!;
    return { base, level, wire: `${base}[${level}]` };
  }

  async setModel(id: string): Promise<void> {
    const prevEffort = this.#catalog.currentEffort;
    const target = this.#wire(id, prevEffort);
    const order: ("config_option" | "set_model")[] = this.#spec.modelVia === "set_model" ? ["set_model", "config_option"] : ["config_option", "set_model"];
    let last: unknown;
    for (const via of order) {
      if (via === "config_option" && !this.#catalog.modelConfigId && this.#spec.modelVia !== "config_option") continue;
      try {
        const res =
          via === "set_model"
            ? await this.#client.request("session/set_model", { sessionId: this.sessionId, modelId: target.wire })
            : await this.#client.request("session/set_config_option", { sessionId: this.sessionId, configId: this.#catalog.modelConfigId ?? "model", value: target.base });
        const o = asObj(res);
        const fresh = o && Array.isArray(o["configOptions"]) ? readCatalog(o, this.#spec) : undefined;
        if (via === "config_option" && fresh) {
          // The bridge says which option the id landed on (claude: "claude-sonnet-5-5" -> "sonnet"), and
          // which efforts that model has.
          this.#take(fresh, true);
        } else {
          this.#catalog.currentModel = target.base;
        }
        if (this.#spec.effortVia === "model-suffix") {
          if (target.level) this.#catalog.currentEffort = target.level;
        } else if (prevEffort && this.#catalog.currentEffort !== prevEffort && this.#catalog.efforts.some((e) => e.id === prevEffort)) {
          // A bridge may reset the effort on a model change; keep the user's when the new model has it.
          await this.setEffort(prevEffort).catch(() => undefined);
        }
        return;
      } catch (e) {
        last = e;
        const missing = e instanceof MethodNotFound || (e instanceof JsonRpcError && e.rpc.code === -32601);
        if (!missing) break;
      }
    }
    throw last instanceof Error ? last : new Error(String(last));
  }

  async setEffort(id: string): Promise<void> {
    const c = this.#catalog;
    if (this.#spec.effortVia === "model-suffix") {
      const base = c.currentModel;
      if (!base || !c.levels?.get(base)?.includes(id)) throw new Error(`${base ?? "this model"} has no effort level "${id}"`);
      await this.#client.request("session/set_model", { sessionId: this.sessionId, modelId: `${base}[${id}]` });
      c.currentEffort = id;
      return;
    }
    if (!c.effortConfigId) throw new Error("this agent has no effort setting");
    const res = asObj(await this.#client.request("session/set_config_option", { sessionId: this.sessionId, configId: c.effortConfigId, value: id }));
    if (res && Array.isArray(res["configOptions"])) this.#take(readCatalog(res, this.#spec), false);
    c.currentEffort = id;
  }

  async setMode(id: string): Promise<void> {
    await this.#client.request("session/set_mode", { sessionId: this.sessionId, modeId: id });
    this.#catalog.currentMode = id;
  }

  on(listener: (ev: ChatEvent) => void): () => void {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  #emit(ev: ChatEvent): void {
    if (this.#turn && (ev.t === "text" || ev.t === "thought" || ev.t === "tool" || ev.t === "plan" || ev.t === "permission")) this.#turn.produced = true;
    for (const l of [...this.#listeners]) {
      try {
        l(ev);
      } catch {
        // a UI bug must not break the protocol loop
      }
    }
  }

  async prompt(input: PromptInput): Promise<void> {
    if (this.#closed) throw new Error("session is closed");
    if (this.#turn) throw new Error("a turn is already running");
    const turn = { id: `t${++this.#turnCount}-${Date.now().toString(36)}`, tools: new Map<string, ToolState>(), costBase: this.#cost, produced: false };
    this.#turn = turn;
    this.#cancelled = false;
    this.#emit({ t: "turn_start", turn: turn.id });

    const text = this.#firstPromptBrief === undefined ? input.text : withBrief(this.#firstPromptBrief, input.text);
    this.#firstPromptBrief = undefined;
    const blocks: Obj[] = [{ type: "text", text }];
    for (const img of input.images ?? []) blocks.push({ type: "image", data: img.data, mimeType: img.mime });

    let result: Obj | undefined;
    try {
      result = asObj(await this.#client.requestUnbounded("session/prompt", { sessionId: this.sessionId, prompt: blocks }));
    } catch (e) {
      this.#cancelPending();
      const rpc = e instanceof JsonRpcError ? e : undefined;
      if (rpc) {
        // The bridge is alive and said no (auth, bad request, internal error): a notice, the turn ends.
        this.#emit({ t: "notice", level: "error", message: rpc.rpc.message || rpc.message, ...hintFor(rpc, this.account) });
        this.#endTurn(turn.id, "error");
        return;
      }
      const why = (e as Error).message;
      const tail = this.#client.stderrTail();
      this.#emit({
        t: "notice",
        level: "error",
        message: this.#cancelled ? `The agent stopped: ${why}` : `The agent process ended while answering: ${why}`,
        hint: tail ? `Last output from the agent:\n${tail}` : "Start a new chat to continue.",
      });
      this.#endTurn(turn.id, "error");
      throw e;
    }
    this.#cancelPending();
    const usage = usageOf(result, this.#cost - turn.costBase, this.#ctx);
    const stop = stopOf(result?.["stopReason"]);
    // pi, for one, reports a rejected request (a 400 for a reasoning level the model refuses) as a clean, empty end_turn.
    if (stop === "end_turn" && !turn.produced) {
      this.#emit({ t: "notice", level: "warn", message: "The agent finished without answering.", hint: "The model or its reasoning effort may have been rejected. Try another model or a lower effort; the agent's own settings or log can say why." });
    }
    this.#endTurn(turn.id, stop, usage);
  }

  #endTurn(turnId: string, stop: Extract<ChatEvent, { t: "turn_end" }>["stop"], usage?: Usage): void {
    this.#turn = undefined;
    this.#emit({ t: "turn_end", turn: turnId, stop, ...(usage ? { usage } : {}) });
  }

  async cancel(): Promise<void> {
    if (!this.#turn || this.#closed) return;
    this.#cancelled = true;
    // ACP: the cancel notification goes first, then pending permission requests are answered
    // "cancelled", and the turn ends "cancelled".
    this.#client.notify("session/cancel", { sessionId: this.sessionId });
    this.#cancelPending();
  }

  respondPermission(id: string, optionId: string | null): void {
    const p = this.#pending.get(id);
    if (!p) return;
    this.#pending.delete(id);
    // Only an option the bridge offered counts; anything else is a refusal, never an invented "allow".
    const offered = optionId !== null && p.event.options.some((o) => o.id === optionId);
    p.resolve({ outcome: offered ? { outcome: "selected", optionId } : { outcome: "cancelled" } });
    this.#emit({ ...p.event, resolved: offered ? optionId! : "cancelled" });
  }

  #cancelPending(): void {
    for (const id of [...this.#pending.keys()]) this.respondPermission(id, null);
  }

  async close(): Promise<void> {
    this.#closed = true;
    this.#cancelPending();
    await this.#client.close();
  }

  // ───────────── inbound ─────────────

  async #onRequest(method: string, params: unknown): Promise<unknown> {
    if (method !== "session/request_permission") throw new MethodNotFound(`${method} is not served by this client`);
    const p = asObj(params) ?? {};
    const turn = this.#turn;
    // Outside a turn nobody can answer: refuse, never invent an allow.
    if (!turn) return { outcome: { outcome: "cancelled" } };
    const toolCall = asObj(p["toolCall"]) ?? {};
    const callId = str(toolCall["toolCallId"]);
    const joined = callId ? turn.tools.get(callId) : undefined;
    const name = toolNameOf(toolCall) ?? joined?.name;
    const options: PermissionOption[] = objs(p["options"]).map((o) => ({
      id: String(o["optionId"] ?? ""),
      name: String(o["name"] ?? ""),
      kind: String(o["kind"] ?? ""),
    }));
    const input = isEmptyInput(toolCall["rawInput"]) ? joined?.input : toolCall["rawInput"];
    const event: PermissionEvent = {
      t: "permission",
      turn: turn.id,
      id: `perm${++this.#permCount}`,
      title: str(toolCall["title"]) ?? joined?.title ?? name ?? "Permission needed",
      kind: toolKindOf(toolCall["kind"]) ?? joined?.kind ?? "other",
      ...(input !== undefined ? { input } : {}),
      options,
      ...(name ? { name } : {}),
    };
    return new Promise<Obj>((resolve) => {
      this.#pending.set(event.id, { event, resolve });
      this.#emit(event);
    });
  }

  #onUpdate(update: Obj): void {
    this.#lastUpdateAt = Date.now();
    const kind = update["sessionUpdate"];
    // Session-level facts count whenever they arrive (also right after session/new / session/load).
    if (kind === "config_option_update") {
      // Suffix-style models (codex) come from the `models` block, which an update does not carry.
      this.#take(readCatalog(update, this.#spec), this.#spec.modelVia === "config_option");
      return;
    }
    if (kind === "current_mode_update") {
      const id = str(update["currentModeId"]);
      if (id && this.#spec.permissionModes) this.#catalog.currentMode = id;
      return;
    }
    if (kind === "usage_update") {
      const cost = asObj(update["cost"]);
      if (cost && typeof cost["amount"] === "number" && (cost["currency"] === undefined || cost["currency"] === "USD")) this.#cost = cost["amount"];
      const used = update["used"], size = update["size"];
      if (typeof used === "number" && typeof size === "number" && size > 0) this.#ctx = { used, size };
      return;
    }

    const turn = this.#turn;
    if (!turn) return; // replayed history, startup banners: never live events
    switch (kind) {
      case "agent_message_chunk": {
        const notify = asObj(asObj(update["_meta"])?.["piAcp"])?.["notify"];
        const text = textOf(update["content"]);
        if (!text) return;
        if (this.#banner !== undefined && text === this.#banner) return; // pi-acp's startup banner, late
        // pi-acp speaks for itself in message chunks (retries, compaction): a notice, not the model.
        if (notify !== undefined) this.#emit({ t: "notice", level: "info", message: text.trim(), ...(/compaction finished/i.test(text) ? { compacted: true } : {}) });
        else this.#emit({ t: "text", turn: turn.id, delta: text });
        return;
      }
      case "agent_thought_chunk": {
        const text = textOf(update["content"]);
        if (text) this.#emit({ t: "thought", turn: turn.id, delta: text });
        return;
      }
      case "plan": {
        const entries = objs(update["entries"]).map((e) => ({
          content: String(e["content"] ?? ""),
          status: e["status"] === "completed" || e["status"] === "in_progress" ? (e["status"] as "completed" | "in_progress") : ("pending" as const),
        }));
        this.#emit({ t: "plan", turn: turn.id, entries });
        return;
      }
      case "tool_call":
      case "tool_call_update":
        this.#onTool(turn, update, kind);
        return;
      case "compaction_update": // claude and codex, because initialize says we take them (acp.ts)
        if (update["status"] !== "completed" || this.#compactions.has(String(update["compactionId"]))) return; // claude repeats the terminal frame with token counts
        this.#compactions.add(String(update["compactionId"]));
        this.#emit({ t: "notice", level: "info", message: "Older parts of this chat were summarised to make room.", compacted: true });
        return;
      default:
        return; // user_message_chunk, available_commands_update, session_info_update, ...
    }
  }

  #onTool(turn: { id: string; tools: Map<string, ToolState> }, update: Obj, kind: "tool_call" | "tool_call_update"): void {
    const id = str(update["toolCallId"]);
    if (!id) return;
    const prev = turn.tools.get(id);
    const name = toolNameOf(update) ?? prev?.name;
    const status = toolStatusOf(update["status"]) ?? prev?.status ?? (kind === "tool_call" ? "pending" : "running");
    const next: ToolState = {
      title: str(update["title"]) ?? prev?.title ?? name ?? "Tool",
      kind: toolKindOf(update["kind"]) ?? prev?.kind ?? "other",
      status,
      ...(isEmptyInput(update["rawInput"]) ? (prev?.input !== undefined ? { input: prev.input } : {}) : { input: update["rawInput"] }),
      ...(outputOf(update) !== undefined ? { output: outputOf(update)! } : prev?.output !== undefined ? { output: prev.output } : {}),
      ...(name ? { name } : {}),
    };
    turn.tools.set(id, next);
    // Always the merged state: an upsert by id must never lose the title or input a later update omits.
    const ev: ToolEvent = { t: "tool", turn: turn.id, id, ...next };
    this.#emit(ev);
  }
}

function usageOf(result: Obj | undefined, cost: number, ctx?: { used: number; size: number }): Usage | undefined {
  const u = asObj(result?.["usage"]);
  const usage: Usage = {};
  if (typeof u?.["inputTokens"] === "number") usage.inputTokens = u["inputTokens"];
  if (typeof u?.["outputTokens"] === "number") usage.outputTokens = u["outputTokens"];
  if (cost > 0) usage.costUsd = cost;
  if (ctx) { usage.contextUsed = ctx.used; usage.contextSize = ctx.size; }
  return Object.keys(usage).length ? usage : undefined;
}

/** A hint for the user when the bridge refused: signed out is the usual one. */
function hintFor(e: JsonRpcError, account: string | undefined): { hint?: string } {
  const text = `${e.rpc.message} ${JSON.stringify(e.rpc.data ?? "")}`;
  if (/auth|log ?in|sign ?in|credential|api key|unauthor|401/i.test(text)) {
    return { hint: account ? "Sign in again in the agent's own app, then start a new chat." : "Sign in in the agent's own app (for Claude Code: run `claude` once), or use an API key, then start a new chat." };
  }
  return {};
}
