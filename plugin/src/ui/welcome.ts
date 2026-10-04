// The first-run welcome: pick the agent, watch the setup check run (fixing what can be fixed in one click), then
// start chatting. It is a screen like History, shown in place of the chat until the user has finished or skipped it.
import type { BackendId, BackendStatus, DoctorCheck, PanelHost } from "../types.ts";
import { env, h, icon, setKids, svg } from "./dom.ts";
import { PRIORITY } from "./health.ts";
import type { Health } from "./health.ts";
import { BACKENDS, BACKEND_LABEL, PROBLEM, fixButton, mark } from "./views.ts";

const ABOUT: Record<BackendId, { sub: string; signIn: string }> = {
  "claude-code": { sub: "Anthropic. Uses your Claude subscription.", signIn: "Run `claude` in a terminal and sign in with /login." },
  codex: { sub: "OpenAI. Uses your ChatGPT plan.", signIn: "Run `codex login` in a terminal." },
  pi: { sub: "Open source. Local models or your own provider keys.", signIn: "Run `pi` in a terminal and add a provider with /login." },
};

/** A tick that draws itself (the stroke grows from nothing). */
const tick = () => svg("svg", { class: "tick", viewBox: "0 0 16 16", fill: "none", stroke: "currentColor", "stroke-width": "2", "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" },
  svg("path", { d: "m3.5 8.5 3 3 6-7", pathLength: "1" }));

interface Handlers { done(): void; recheck(): void; changed(): void; openSettings(): void }

export function welcomeView(host: PanelHost, health: Health, o: Handlers): { el: HTMLElement; render(): void } {
  const hero = mark(true);
  const cards = h("div.wcards", { role: "radiogroup", "aria-label": "Agent" });
  const hint = h("p.wel__hint", { "aria-live": "polite" });
  const count = h("span.wel__count");
  const fill = h("div.wbar__fill");
  const rows = h("ol.wrows");
  const foot = h("div.wfoot");
  const settled = new Set<string>(); // checks whose result has been shown (each reveals in turn, so the check reads as work)
  const scheduled = new Set<string>();
  const logs = new Map<string, string>();
  const gap = env.win.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? 0 : 240;
  let lastReady = false;

  const pick = async (id: BackendId) => {
    if (host.getSettings().backend === id) return;
    await host.setSettings({ backend: id });
    o.changed();
    o.recheck();
  };

  const paintCards = () => {
    const chosen = host.getSettings().backend;
    const statuses = health.statuses;
    if (!statuses) { setKids(cards, BACKENDS.map(() => h("div.sk.sk--card"))); setKids(hint, null); return; }
    const status = (id: BackendId): BackendStatus | undefined => statuses.find((s) => s.id === id);
    setKids(cards, BACKENDS.map((id) => {
      const s = status(id);
      const on = id === chosen;
      return h(`button.wcard${on ? ".wcard--on" : ""}`, { type: "button", role: "radio", "aria-checked": String(on), onclick: () => void pick(id) },
        h("span.wcard__dot", { "aria-hidden": "true" }),
        h("span.wcard__tx", null, h("span.wcard__t", null, BACKEND_LABEL[id]), h("span.wcard__d", null, ABOUT[id].sub)),
        h(`span.wcard__st${s?.available ? ".wcard__st--ok" : ""}`, { title: s?.reason ?? "" }, h("span.wcard__led"), s?.available ? s.account || "Ready" : s?.reason ?? "Not set up"));
    }));
    const cur = status(chosen);
    setKids(hint, cur && !cur.available ? `${ABOUT[chosen].signIn} Then press Recheck.` : null);
  };

  const rowFor = (c: DoctorCheck, pending: boolean): HTMLElement => {
    if (pending) {
      return h("li.wrow.wrow--pending", null, h("span.wrow__i", null, h("span.spin")),
        h("div.wrow__tx", null, h("div.wrow__t", null, c.label), h("div.wrow__d", null, "Checking…")));
    }
    const log = h("pre.fixlog", { hidden: !logs.has(c.id), "aria-live": "polite" }, logs.get(c.id) ?? "");
    const fix = c.ok ? null : fixButton(c, ".btn--solid", log, () => { logs.set(c.id, log.textContent ?? ""); o.recheck(); });
    const help = c.id === "backend" ? "Pick another agent above, or sign in and press Recheck." : PROBLEM[c.id].help;
    return h(`li.wrow.wrow--${c.ok ? "ok" : "bad"}`, null,
      h("span.wrow__i", { "aria-hidden": "true" }, c.ok ? tick() : icon("warn")),
      h("div.wrow__tx", null,
        h("div.wrow__t", null, c.label, h("span.sr", null, c.ok ? " (ok)" : " (needs attention)")),
        c.detail ? h("div.wrow__d", null, c.detail) : null,
        c.ok ? null : h("p.wrow__help", null, help),
        c.ok ? null : fix ? h("div.wrow__acts", null, fix) : null,
        log));
  };

  const paintRows = (): boolean => {
    const checks = health.checks ? [...health.checks].sort((a, b) => PRIORITY.indexOf(a.id) - PRIORITY.indexOf(b.id)) : null;
    if (!checks) { setKids(rows, [0, 1, 2, 3, 4].map(() => h("li.sk.sk--row"))); setKids(count, null); fill.style.width = "0"; return false; }
    // Reveal results one after another; a check that is being re-run spins again until it answers.
    let k = 0;
    for (const c of checks) {
      if (settled.has(c.id) || scheduled.has(c.id)) continue;
      scheduled.add(c.id);
      env.win.setTimeout(() => { settled.add(c.id); if (rows.isConnected) render(); }, gap * k++);
    }
    const shown = checks.filter((c) => settled.has(c.id));
    setKids(rows, checks.map((c) => rowFor(c, !settled.has(c.id) || (health.busy && !c.ok))));
    const passed = shown.filter((c) => c.ok).length;
    setKids(count, `${passed} of ${checks.length}`);
    fill.style.width = `${(passed / checks.length) * 100}%`;
    return shown.length === checks.length && !health.busy;
  };

  const paintFoot = (settledAll: boolean) => {
    const bad = health.failing;
    const blocked = health.blockReason();
    const ready = settledAll && !bad.length && !blocked;
    const s = host.getSettings();
    const account = health.backend()?.account;
    const who = account ? `${BACKEND_LABEL[s.backend]} (${account})` : BACKEND_LABEL[s.backend];
    const go = h(`button.btn.${ready ? "btn--solid" : "btn--quiet"}`, { type: "button", disabled: blocked || !settledAll ? true : null, onclick: o.done },
      ready ? "Start chatting" : blocked ? "Start chatting" : "Continue anyway");
    setKids(foot,
      ready ? h("p.wready", { role: "status" }, h("span.wready__i", null, tick()), `All set. You'll chat with ${who}.`) : null,
      !ready && settledAll && !blocked ? h("p.wel__hint", null, "You can finish this later: the dot at the top shows what's missing.") : null,
      h("div.wfoot__acts", null, go,
        settledAll && !ready ? h("button.btn.btn--quiet", { type: "button", onclick: () => o.recheck() }, icon("retry"), "Recheck") : null,
        h("button.lnk", { type: "button", onclick: o.openSettings }, "Settings")),
      !ready ? h("button.lnk.wfoot__skip", { type: "button", onclick: o.done }, "Skip setup") : null);
    hero.classList.toggle("mark--busy", !settledAll);
    if (ready && !lastReady) hero.classList.add("mark--ready");
    if (!ready) hero.classList.remove("mark--ready");
    lastReady = ready;
  };

  const render = () => {
    paintCards();
    paintFoot(paintRows());
  };

  const step = (n: number, title: string, ...tail: HTMLElement[]) =>
    h("div.wel__head", null, h("span.wel__n", null, String(n)), h("h3.wel__h", null, title), ...tail);

  const el = h("div.wel", null, h("div.wel__in", null,
    h("div.wel__hero", null, hero,
      h("h2", null, "Chat with your library"),
      h("p.wel__lead", null, "Ask about the paper you're reading, or search and organize your collection, right from this panel.")),
    h("section.wel__sec", { "aria-label": "Choose your agent" }, step(1, "Choose your agent"), cards, hint),
    h("section.wel__sec", { "aria-label": "Getting ready" }, step(2, "Getting ready", count), h("div.wbar", { "aria-hidden": "true" }, fill), rows),
    foot));
  render();
  return { el, render };
}
