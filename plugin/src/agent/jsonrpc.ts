// Newline-delimited JSON-RPC 2.0 over a child's stdio, on top of the injected `Proc`.
// Ported from meeting-buddy src/agent/jsonrpc.ts (driven against claude-code-acp and the
// @agentclientprotocol bridges). No node:* here: the same file runs in Gecko.
//
// Wire facts carried over:
// - Claude Code numbers its requests from 0; ids are compared as strings.
// - An agent->client request that goes unanswered stalls the agent, so every inbound
//   request gets a reply, including "method not found".
// - Once the stream is gone it stays gone: a request issued after the pipe died rejects at
//   once instead of waiting for ever.
import type { Proc } from "../types.ts";

interface Frame {
  jsonrpc?: "2.0";
  id?: string | number;
  method?: string;
  params?: unknown;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
}

export type IncomingHandler = (method: string, params: unknown) => Promise<unknown>;
export type NotificationHandler = (method: string, params: unknown) => void;

export class MethodNotFound extends Error {}

export class JsonRpcError extends Error {
  readonly rpc: { code: number; message: string; data?: unknown };
  constructor(rpc: { code: number; message: string; data?: unknown }) {
    super(`${rpc.message} (${rpc.code})${rpc.data ? " " + JSON.stringify(rpc.data) : ""}`);
    this.name = "JsonRpcError";
    this.rpc = rpc;
  }
}

export class JsonRpcPeer {
  #nextId = 1;
  #pending = new Map<string, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
  #onRequest: IncomingHandler = async () => {
    throw new MethodNotFound("no request handler installed");
  };
  #onNotification: NotificationHandler = () => {};
  #proc: Proc;
  /** Why the peer is gone, once it is. The first cause is kept. */
  #dead: Error | undefined;

  constructor(proc: Proc) {
    this.#proc = proc;
    proc.onStdoutLine((line) => {
      if (line.trim().length > 0) void this.#dispatch(line);
    });
    // The process is what matters: a grandchild may hold the pipe open after the bridge is gone.
    void proc.exited.then((code) => this.#die(new Error(`agent exited${code === null ? " (killed)" : ` with code ${code}`}`)));
  }

  get dead(): Error | undefined {
    return this.#dead;
  }

  handleRequests(handler: IncomingHandler): void {
    this.#onRequest = handler;
  }

  handleNotifications(handler: NotificationHandler): void {
    this.#onNotification = handler;
  }

  request<T = unknown>(method: string, params?: unknown): Promise<T> {
    if (this.#dead !== undefined) return Promise.reject(this.#dead);
    const id = this.#nextId++;
    const promise = new Promise<unknown>((resolve, reject) => {
      this.#pending.set(String(id), { resolve, reject });
    });
    this.#write({ jsonrpc: "2.0", id, method, ...(params === undefined ? {} : { params }) });
    return promise as Promise<T>;
  }

  notify(method: string, params?: unknown): void {
    this.#write({ jsonrpc: "2.0", method, ...(params === undefined ? {} : { params }) });
  }

  close(reason = "connection closed"): void {
    this.#die(new Error(reason));
  }

  #write(frame: Frame): void {
    if (this.#dead !== undefined) return;
    try {
      this.#proc.write(JSON.stringify(frame) + "\n");
    } catch (e) {
      this.#die(new Error(`agent pipe write failed: ${(e as Error).message}`));
    }
  }

  async #dispatch(line: string): Promise<void> {
    let frame: Frame;
    try {
      frame = JSON.parse(line) as Frame;
    } catch {
      return;
    }

    if (frame.method !== undefined && frame.id !== undefined) {
      const id = frame.id;
      try {
        const result = await this.#onRequest(frame.method, frame.params);
        this.#write({ jsonrpc: "2.0", id, result });
      } catch (err) {
        const code = err instanceof MethodNotFound ? -32601 : -32603;
        this.#write({ jsonrpc: "2.0", id, error: { code, message: String((err as Error)?.message ?? err) } });
      }
      return;
    }

    if (frame.method !== undefined) {
      try {
        this.#onNotification(frame.method, frame.params);
      } catch {
        // a listener bug must not take the reader down
      }
      return;
    }

    if (frame.id !== undefined) {
      const waiter = this.#pending.get(String(frame.id));
      if (!waiter) return;
      this.#pending.delete(String(frame.id));
      if (frame.error) waiter.reject(new JsonRpcError(frame.error));
      else waiter.resolve(frame.result);
    }
  }

  #die(error: Error): void {
    if (this.#dead === undefined) this.#dead = error;
    const cause = this.#dead;
    for (const [, waiter] of this.#pending) waiter.reject(cause);
    this.#pending.clear();
  }
}
