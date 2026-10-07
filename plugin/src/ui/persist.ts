// Saving a chat: every event the transcript folds is also written to the host, but consecutive text or
// thought deltas of one turn are merged into one line first (a replay of the merged list is identical).
import type { ChatEvent } from "../types.ts";
import { env } from "./dom.ts";

export class Persister {
  private buf: { turn: string; t: "text" | "thought"; delta: string } | null = null;
  private timer: number | undefined;

  private write: (ev: ChatEvent) => void;

  constructor(write: (ev: ChatEvent) => void) { this.write = write; }

  push(ev: ChatEvent): void {
    if (ev.t === "text" || ev.t === "thought") {
      const b = this.buf;
      if (b && b.turn === ev.turn && b.t === ev.t) b.delta += ev.delta;
      else { this.flush(); this.buf = { turn: ev.turn, t: ev.t, delta: ev.delta }; }
      env.win.clearTimeout(this.timer);
      this.timer = env.win.setTimeout(() => this.flush(), 600);
      return;
    }
    this.flush();
    this.write(ev);
  }

  flush(): void {
    const b = this.buf;
    this.buf = null;
    env.win.clearTimeout(this.timer);
    if (b) this.write(b.t === "text" ? { t: "text", turn: b.turn, delta: b.delta } : { t: "thought", turn: b.turn, delta: b.delta });
  }
}
