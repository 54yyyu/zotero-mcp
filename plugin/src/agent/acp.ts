// One bridge process speaking ACP: spawn, bounded handshake, request/notify, shutdown.
// Ported (simplified) from meeting-buddy src/agent/acp.ts; each of these was learned the hard way there:
//
// - close() asks the Proc to die: SIGTERM, grace, SIGKILL on the whole process group, and
//   waits (the Spawner implements the group part; a bridge's grandchild otherwise holds the pipes).
// - a spawn failure rejects; it never throws uncaught.
// - the handshake is bounded; a request into a dead pipe rejects at once.
// - the bridge's last stderr lines are kept, to explain a death.
import type { Proc, Spawner } from "../types.ts";
import { JsonRpcPeer, MethodNotFound, type IncomingHandler } from "./jsonrpc.ts";

const ACP_PROTOCOL_VERSION = 1;

export interface InitializeResult {
  protocolVersion: number;
  agentCapabilities?: {
    loadSession?: boolean;
    promptCapabilities?: { image?: boolean; audio?: boolean; embeddedContext?: boolean };
    [k: string]: unknown;
  };
  agentInfo?: { name?: string; title?: string; version?: string };
  authMethods?: unknown[];
}

export type UpdateListener = (update: Record<string, unknown>, sessionId: string | undefined) => void;

export interface AcpClientOpts {
  spawner: Spawner;
  command: string;
  args: string[];
  cwd: string;
  env: Record<string, string>;
  /** Bound on initialize / session/new / session/load / config requests. Default 30 s. */
  initTimeoutMs?: number;
}

export class AcpClient {
  /** The signed-in account the bridge reported (`_auth/status_update`), e.g. "Claude Max". */
  account: string | undefined;
  #proc: Proc;
  #peer: JsonRpcPeer;
  #stderr: string[] = [];
  #initTimeoutMs: number;
  #updateListeners = new Set<UpdateListener>();
  #onRequest: IncomingHandler = async (method) => {
    throw new MethodNotFound(`${method} is not served by this client`);
  };
  #closing: Promise<void> | undefined;

  private constructor(proc: Proc, opts: AcpClientOpts) {
    this.#proc = proc;
    this.#initTimeoutMs = opts.initTimeoutMs ?? 30_000;
    proc.onStderr((chunk) => {
      for (const line of chunk.split("\n")) if (line.trim()) this.#stderr.push(line);
      if (this.#stderr.length > 200) this.#stderr.splice(0, this.#stderr.length - 200);
    });
    this.#peer = new JsonRpcPeer(proc);
    this.#peer.handleNotifications((method, params) => this.#onNotification(method, params));
    this.#peer.handleRequests((method, params) => this.#onRequest(method, params));
  }

  /** Spawn the bridge. Rejects (never throws) when the process cannot start. */
  static async spawn(opts: AcpClientOpts): Promise<AcpClient> {
    let proc: Proc;
    try {
      proc = await opts.spawner.spawn(opts.command, opts.args, { cwd: opts.cwd, env: opts.env });
    } catch (e) {
      throw new Error(`could not start the agent bridge: ${(e as Error)?.message ?? e}`);
    }
    return new AcpClient(proc, opts);
  }

  /** The last few stderr lines, for an error message. */
  stderrTail(lines = 6): string {
    return this.#stderr.slice(-lines).join("\n");
  }

  onUpdate(listener: UpdateListener): () => void {
    this.#updateListeners.add(listener);
    return () => this.#updateListeners.delete(listener);
  }

  handleRequests(handler: IncomingHandler): void {
    this.#onRequest = handler;
  }

  /**
   * initialize: protocolVersion 1, no fs, no terminal; compaction as `compaction_update` (claude and codex otherwise show it as
   * a tool call); bridge notices as `notice` updates (claude otherwise writes them into the answer as bold text, e.g. "Auto mode
   * unavailable" ahead of a translation).
   */
  async initialize(): Promise<InitializeResult> {
    const init = await this.request<InitializeResult>("initialize", {
      protocolVersion: ACP_PROTOCOL_VERSION,
      clientCapabilities: { fs: { readTextFile: false, writeTextFile: false }, terminal: false, session: { compaction: {}, notices: {} } },
    });
    if (init?.protocolVersion !== ACP_PROTOCOL_VERSION) {
      throw new Error(`agent negotiated ACP protocolVersion ${init?.protocolVersion}; we speak ${ACP_PROTOCOL_VERSION}`);
    }
    return init;
  }

  /** A request bounded by `timeoutMs` (default: the handshake bound). A dead pipe rejects at once. */
  request<T = unknown>(method: string, params?: unknown, timeoutMs: number = this.#initTimeoutMs): Promise<T> {
    const answer = this.#peer.request<T>(method, params);
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`agent did not answer ${method} within ${timeoutMs} ms`)), timeoutMs);
      answer.then(
        (v) => {
          clearTimeout(timer);
          resolve(v);
        },
        (e: unknown) => {
          clearTimeout(timer);
          reject(e instanceof Error ? e : new Error(String(e)));
        },
      );
    });
  }

  /** A request with no time bound (session/prompt: a turn takes as long as it takes). */
  requestUnbounded<T = unknown>(method: string, params?: unknown): Promise<T> {
    return this.#peer.request<T>(method, params);
  }

  notify(method: string, params?: unknown): void {
    this.#peer.notify(method, params);
  }

  /** Stop the bridge and wait until it has actually stopped. Safe to call twice. */
  close(): Promise<void> {
    this.#closing ??= (async () => {
      this.#peer.close("session closed");
      await this.#proc.kill().catch(() => undefined); // already gone is fine
    })();
    return this.#closing;
  }

  #onNotification(method: string, params: unknown): void {
    if (method === "_auth/status_update") {
      const label = (params as { authStatus?: { label?: unknown } } | undefined)?.authStatus?.label;
      if (typeof label === "string" && label) this.account = label;
      return;
    }
    if (method !== "session/update") return;
    const p = params as { sessionId?: unknown; update?: Record<string, unknown> } | undefined;
    if (!p?.update) return;
    const sid = typeof p.sessionId === "string" ? p.sessionId : undefined;
    for (const l of [...this.#updateListeners]) {
      try {
        l(p.update, sid);
      } catch {
        // a listener bug must not take the reader down
      }
    }
  }
}
