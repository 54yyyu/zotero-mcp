// What the doctor and the runtime say about whether chatting can work: the checks, which backends are
// ready, and the one problem worth showing first.
import type { BackendStatus, DoctorCheck, PanelHost } from "../types.ts";
import { errMessage } from "./dom.ts";

/** Which failing check matters most, in order; the first two stop the chat from working at all. */
const PRIORITY: DoctorCheck["id"][] = ["node", "backend", "zotero-api", "cli", "write-access"];
export const BLOCKING = new Set<DoctorCheck["id"]>(["node", "backend"]);

export type HealthState = "unknown" | "checking" | "ok" | "warn" | "bad";

export class Health {
  checks: DoctorCheck[] | null = null;
  statuses: BackendStatus[] | null = null;
  private checking = false;
  private host: PanelHost;
  private changed: () => void;

  constructor(host: PanelHost, changed: () => void) { this.host = host; this.changed = changed; }

  get failing(): DoctorCheck[] {
    return (this.checks ?? []).filter((c) => !c.ok).sort((a, b) => PRIORITY.indexOf(a.id) - PRIORITY.indexOf(b.id));
  }

  get state(): HealthState {
    if (!this.checks) return this.checking ? "checking" : "unknown";
    const bad = this.failing;
    return !bad.length ? "ok" : bad.some((c) => BLOCKING.has(c.id)) ? "bad" : "warn";
  }

  /** The chosen backend's status, if the runtime has reported. */
  backend(): BackendStatus | undefined { return this.statuses?.find((x) => x.id === this.host.getSettings().backend); }

  /** The problem to show first: a failing check, or the backend saying it is not ready before the doctor has noticed. */
  first(): DoctorCheck | undefined {
    const st = this.backend();
    return this.failing[0] ?? (st && !st.available ? { id: "backend", ok: false, label: st.label, ...(st.reason ? { detail: st.reason } : {}) } : undefined);
  }

  /** Why sending is impossible right now, or null. */
  blockReason(): string | null {
    const st = this.backend();
    if (st && !st.available) return `${st.label} isn't ready: ${st.reason ?? "not found"}`;
    const bad = this.checks?.find((c) => !c.ok && BLOCKING.has(c.id));
    return bad ? `${bad.label}${bad.detail ? `: ${bad.detail}` : ""}` : null;
  }

  set(checks: DoctorCheck[]): void { this.checks = checks; this.changed(); }

  async refreshStatuses(): Promise<void> {
    try { this.statuses = await this.host.runtime.detect(); } catch { /* the doctor reports it */ }
    this.changed();
  }

  async refresh(): Promise<void> {
    if (this.checking) return;
    this.checking = true;
    this.changed();
    try {
      const [checks] = await Promise.all([this.host.doctor(), this.refreshStatuses()]);
      this.checks = checks;
    } catch (e) {
      this.checks = [{ id: "backend", ok: false, label: "Status check", detail: errMessage(e) }];
    } finally {
      this.checking = false;
      this.changed();
    }
  }
}
