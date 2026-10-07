// The context chips on the next message: what the host says the user is looking at (auto), chips the user
// pinned so they survive focus changes, chips the user dismissed until that item leaves focus, and chips
// added by hand with `@` or `+`.
import type { ChipSummary, ContextChip, PanelHost } from "../types.ts";

export const summary = (c: ContextChip): ChipSummary => ({
  id: c.id, kind: c.kind, label: c.kind === "selection" ? "Text Selection" : c.label, ref: c.ref, ...(c.text ? { text: c.text.slice(0, 2000) } : {}),
});

export class ChipState {
  private added: ContextChip[] = [];
  private pinned = new Map<string, ContextChip>();
  private dismissed = new Set<string>();

  private host: Pick<PanelHost, "currentContext">;

  constructor(host: Pick<PanelHost, "currentContext">) { this.host = host; }

  /** The chips to show and send now. */
  current(): ContextChip[] {
    let cur: ContextChip[] = [];
    try { cur = this.host.currentContext(); } catch { /* the glue is mid-switch */ }
    const ids = new Set(cur.map((c) => c.id));
    for (const id of this.dismissed) if (!ids.has(id)) this.dismissed.delete(id); // back in focus later: show it again
    const out = new Map<string, ContextChip>();
    for (const p of this.pinned.values()) out.set(p.id, { ...p, auto: true, pinned: true });
    for (const c of cur) if (!this.dismissed.has(c.id)) out.set(c.id, { ...c, auto: true, pinned: this.pinned.has(c.id) });
    for (const a of this.added) out.set(a.id, { ...a, auto: false });
    return [...out.values()];
  }

  /** Add a chip by hand; `pinned` (a drop) keeps it across sends until the user unpins or removes it. */
  add(c: ContextChip, pinned = false): void {
    this.added = [...this.added.filter((x) => x.id !== c.id), { ...c, auto: false, pinned }];
    this.dismissed.delete(c.id);
  }

  remove(c: ContextChip): void {
    if (c.auto) { this.dismissed.add(c.id); this.pinned.delete(c.id); }
    else this.added = this.added.filter((x) => x.id !== c.id);
  }

  togglePin(c: ContextChip): void {
    if (!c.auto) { this.added = this.added.map((x) => (x.id === c.id ? { ...x, pinned: !x.pinned } : x)); return; }
    if (this.pinned.has(c.id)) this.pinned.delete(c.id);
    else this.pinned.set(c.id, c);
  }

  /** After a send: hand-added chips were for that message only. */
  clearAdded(): void { this.added = this.added.filter((x) => x.pinned); }

  /** A new chat starts clean, pinned or not. */
  clearAll(): void { this.added = []; }
}
