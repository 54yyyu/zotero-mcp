// The panel's other screens: the empty state, history, settings, and the doctor/status view.
// Each is a plain function returning an element and a `refresh` where it has data to reload.
import type { BackendId, DoctorCheck, PanelHost, PromptEntry, SavedSession } from "../types.ts";
import { copyText, dayGroup, env, errMessage, h, icon, relTime, setKids, slotLabel, svg } from "./dom.ts";
import { shortPath } from "./settings-model.ts";

export const BACKEND_LABEL: Record<BackendId, string> = { "claude-code": "Claude Code", codex: "Codex", pi: "pi" };
export const BACKENDS: BackendId[] = ["claude-code", "codex", "pi"];

/**
 * The logo (assets/logo.svg): a chat bubble with typing dots and the agent's red spark; it follows the theme. The parts carry
 * classes so the welcome can animate them (styles-welcome.ts); the little dots around the spark only show in that burst.
 */
export const mark = (hero = false): HTMLElement => h(`div.mark${hero ? ".mark--hero" : ""}`, null, svg("svg", { viewBox: "-16 -16 160 160", "aria-hidden": "true" },
  svg("path", { class: "mk-b", d: "M24 20H104a14 14 0 0 1 14 14V78a14 14 0 0 1-14 14H64L40 112V92H24a14 14 0 0 1-14-14V34a14 14 0 0 1 14-14Z", fill: "var(--ink)" }),
  ...[40, 64, 88].map((cx, i) => svg("g", { class: "mk-w", style: `--i:${i}` }, svg("circle", { class: "mk-d", cx: String(cx), cy: "56", r: "7.5", fill: "var(--paper-raised)" }))),
  ...[[-26, -14], [-6, -30], [22, -26], [28, 4], [14, 22], [-18, 16]].map(([dx, dy]) => svg("circle", { class: "mk-p", cx: "100", cy: "22", r: "2.4", fill: "var(--agent)", style: `--dx:${dx}px;--dy:${dy}px` })),
  svg("g", { class: "mk-t" }, svg("path", { class: "mk-s", d: "M100 0L106.6 15.4L122 22L106.6 28.6L100 44L93.4 28.6L78 22L93.4 15.4Z", fill: "var(--agent)" }))));

// ───────────────────────────── problems (the unavailable states) ─────────────────────────────

export const PROBLEM: Record<DoctorCheck["id"], { title: string; help: string }> = {
  "zotero-api": {
    title: "Restart Zotero: its local API isn't answering",
    help: "A long-running Zotero can stop serving its local API. Restart Zotero (quit it and open it again), then press Recheck. If it still fails, turn on “Allow other applications on this computer to communicate with Zotero” in Zotero's Settings > Advanced.",
  },
  "write-access": {
    title: "The agent can't change your library yet",
    help: "Reading works, but writing needs one authorization. Approve the prompt Zotero shows, or allow write access in Zotero's Settings > Advanced, then press Recheck.",
  },
  cli: {
    title: "zotero-cli isn't installed",
    help: "The agent works your library through zotero-cli. Install it with one click below, or run `uv tool install zotero-mcp-server` in a terminal.",
  },
  node: {
    title: "Node.js wasn't found",
    help: "The agent bridge runs on Node.js 20 or newer. Install it from nodejs.org, then restart Zotero so it sees the new PATH.",
  },
  backend: {
    title: "No agent is ready",
    help: "Sign in to Claude Code in a terminal (run `claude`, then /login), or choose another agent or an API key in Settings.",
  },
};

/** The button that runs a check's fix, streaming its output into `log`; `after` runs when it ends. */
export function fixButton(check: DoctorCheck, cls: string, log: HTMLElement, after: () => void): HTMLButtonElement | null {
  const fix = check.fix;
  if (!fix) return null;
  const btn = h(`button.btn.btn--sm${cls}`, { type: "button" }, fix.label) as HTMLButtonElement;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Working…";
    log.hidden = false;
    log.textContent = "";
    try {
      for await (const chunk of fix.run()) { log.textContent += chunk; log.scrollTop = log.scrollHeight; }
    } catch (e) {
      log.textContent += `\n${errMessage(e)}`;
    }
    btn.disabled = false;
    btn.textContent = fix.label;
    after();
  });
  return btn;
}

/** A card for the check that blocks the most (the empty state shows one; the rest are in Status). */
export function setupCard(check: DoctorCheck, more: number, handlers: { recheck(): void; openStatus(): void; openSettings(): void }): HTMLElement {
  const p = PROBLEM[check.id];
  const log = h("pre.fixlog", { hidden: true, "aria-live": "polite" });
  const fixBtn = fixButton(check, ".btn--solid", log, handlers.recheck);
  return h("div.setup", { role: "region", "aria-label": "Setup needed" },
    h("div.setup__head", null, h("span.setup__i", null, icon("warn")), h("div.setup__tx", null,
      h("div.setup__t", null, p.title),
      check.detail ? h("div.setup__d", null, check.detail) : null)),
    h("p.setup__help", null, p.help),
    log,
    h("div.setup__acts", null,
      fixBtn,
      check.id === "backend" ? h("button.btn.btn--sm", { type: "button", onclick: handlers.openSettings }, "Open settings") : null,
      h("button.btn.btn--sm.btn--quiet", { type: "button", onclick: handlers.recheck }, "Recheck"),
      more > 0 ? h("button.btn.btn--sm.btn--quiet", { type: "button", onclick: handlers.openStatus }, `${more} more`) : null));
}

// ───────────────────────────── empty state ─────────────────────────────

export function emptyState(opts: { prompts: PromptEntry[]; run(p: PromptEntry): void; edit(): void; ready: boolean }): HTMLElement {
  const prompts = [...opts.prompts].sort((a, b) => (a.slot ?? 9) - (b.slot ?? 9));
  return h("div.empty", null,
    mark(),
    h("h2", null, "Ask your library"),
    h("p.empty__lead", null, "Your agent sees what you have open in Zotero and works through zotero-cli. Select text or a figure to ask about it, or type @ to add a source."),
    h("section.prompts", { "aria-label": "Custom prompts" },
      h("div.prompts__head", null, h("h3.eyebrow", null, "Custom prompts"), h("button.lnk", { type: "button", onclick: opts.edit }, icon("pencil"), "Edit")),
      prompts.length
        ? h("ul.prompts__list", null, prompts.map((p) => h("li", null,
          h("button.prompt", { type: "button", disabled: opts.ready ? null : true, title: p.text, onclick: () => opts.run(p) },
            h("span.prompt__t", null, p.title || p.text),
            p.slot ? h("kbd.prompt__k", { title: "Keyboard shortcut" }, slotLabel(p.slot)) : null))))
        : h("p.prompts__none", null, "No prompts yet. Add the questions you ask most, and give four of them a shortcut.")));
}

// ───────────────────────────── a view's frame ─────────────────────────────

export function frame(title: string, back: () => void, ...body: (HTMLElement | null)[]): HTMLElement {
  return h("div.vw", null,
    h("div.vw__head", null,
      h("button.iconbtn", { type: "button", "aria-label": "Back to the chat", title: "Back (Esc)", onclick: back }, icon("back")),
      h("h2.vw__t", null, title)),
    h("div.vw__body", null, ...body));
}

// ───────────────────────────── history ─────────────────────────────

export function historyView(host: PanelHost, o: { current(): string | null; back(): void; resume(s: SavedSession): void; deleted(id: string): void }): { el: HTMLElement; refresh(): void } {
  const list = h("div.hist__list");
  const q = h("input.input", { type: "search", placeholder: "Search chats", "aria-label": "Search chats" }) as HTMLInputElement;
  let all: SavedSession[] | null = null;
  let confirming: string | null = null;
  let copied: string | null = null;

  const copyCmd = async (s: SavedSession, cmd: string) => {
    if (!(await copyText(cmd))) return;
    copied = s.id;
    paint();
    env.win.setTimeout(() => { if (copied === s.id) { copied = null; paint(); } }, 1800);
  };

  const skeleton = () => setKids(list, [0, 1, 2, 3].map(() => h("div.sk.sk--row")));
  const paint = () => {
    if (!all) return;
    const needle = q.value.trim().toLowerCase();
    const rows = all.filter((s) => !needle || s.title.toLowerCase().includes(needle)).sort((a, b) => b.updatedAt - a.updatedAt);
    if (!rows.length) {
      setKids(list, h("div.vempty", null, icon("history"),
        h("div.vempty__t", null, needle ? "No chats match" : "No saved chats yet"),
        h("div.vempty__d", null, needle ? "Try a different word." : "Conversations are saved here as you go, and you can pick one up where you left off.")));
      return;
    }
    const out: HTMLElement[] = [];
    if (rows.some((s) => host.resumeCommand(s))) out.push(h("p.hist__note", null, icon("terminal"), "Paste a copied command in a terminal to continue that chat there. Don't run it in both places at once."));
    let group = "";
    for (const s of rows) {
      const g = dayGroup(s.updatedAt);
      if (g !== group) { group = g; out.push(h("div.hist__group", null, g)); }
      const isCur = o.current() === s.id;
      if (confirming === s.id) {
        out.push(h("div.hrow.hrow--confirm", { role: "group", "aria-label": `Delete ${s.title}?` },
          h("span.hrow__q", null, "Delete this chat?"),
          h("button.btn.btn--sm.btn--danger", { type: "button", onclick: () => void del(s) }, "Delete"),
          h("button.btn.btn--sm", { type: "button", onclick: () => { confirming = null; paint(); } }, "Cancel")));
        continue;
      }
      const cmd = host.resumeCommand(s);
      const done = copied === s.id;
      out.push(h(`div.hrow${isCur ? ".hrow--on" : ""}`, null,
        h("button.hrow__main", { type: "button", onclick: () => o.resume(s), ...(isCur ? { "aria-current": "true" } : {}) },
          h("span.hrow__t", null, s.title || "Untitled chat"),
          h("span.hrow__m", null, `${BACKEND_LABEL[s.backend] ?? s.backend} · ${relTime(s.updatedAt)}`),
          s.cwd ? h("span.hrow__f", { title: s.cwd }, icon("folder"), h("span", null, shortPath(s.cwd, 30))) : null),
        cmd ? h(`button.${done ? "lnk.hrow__copied" : "iconbtn.iconbtn.iconbtn--sm.hrow__cmd"}`, { type: "button", title: "Copy terminal command", "aria-label": done ? "Copied" : `Copy terminal command for ${s.title || "chat"}`, onclick: () => void copyCmd(s, cmd) }, icon(done ? "check" : "terminal"), done ? "Copied" : null) : null,
        h("button.iconbtn.iconbtn--sm.hrow__del", { type: "button", "aria-label": `Delete ${s.title || "chat"}`, title: "Delete", onclick: () => { confirming = s.id; paint(); } }, icon("trash"))));
    }
    setKids(list, out);
  };
  const del = async (s: SavedSession) => {
    try { await host.deleteSession(s.id); } catch (e) { confirming = null; all = (all ?? []); setKids(list, h("div.vwerr", { role: "alert" }, `Couldn't delete: ${errMessage(e)}`)); env.win.setTimeout(paint, 2500); return; }
    all = (all ?? []).filter((x) => x.id !== s.id);
    confirming = null;
    o.deleted(s.id);
    paint();
  };
  const refresh = () => {
    skeleton();
    host.sessions().then((s) => { all = s; confirming = null; paint(); }, (e) => {
      setKids(list, h("div.vempty", { role: "alert" }, icon("warn"), h("div.vempty__t", null, "Couldn't load your chats"), h("div.vempty__d", null, errMessage(e)),
        h("button.btn.btn--sm", { type: "button", onclick: refresh }, "Try again")));
    });
  };
  q.addEventListener("input", paint);
  const el = frame("History", o.back, h("div.hist__search", null, q), list);
  return { el, refresh };
}

// ───────────────────────────── status (the doctor) ─────────────────────────────

export function statusView(host: PanelHost, o: { back(): void; changed(checks: DoctorCheck[]): void; openSettings(): void; initial: DoctorCheck[] | null }): { el: HTMLElement; refresh(): void } {
  const list = h("div.checks");
  const sum = h("div.status__sum", { role: "status" });
  let busy = false;
  const logs = new Map<string, string>();

  const paint = (checks: DoctorCheck[]) => {
    const bad = checks.filter((c) => !c.ok);
    setKids(sum, h("span.status__dot", { "data-ok": String(bad.length === 0) }), bad.length ? `${bad.length} ${bad.length === 1 ? "thing needs" : "things need"} attention` : "Everything is set up");
    setKids(list, checks.map((c) => {
      const log = h("pre.fixlog", { hidden: !logs.has(c.id), "aria-live": "polite" }, logs.get(c.id) ?? "");
      const p = PROBLEM[c.id];
      const fixBtn = c.ok ? null : fixButton(c, ".btn--solid", log, () => { logs.set(c.id, log.textContent ?? ""); refresh(); });
      const settingsBtn = c.id === "backend" && !c.ok ? h("button.btn.btn--sm", { type: "button", onclick: o.openSettings }, "Open settings") : null;
      return h(`div.check.check--${c.ok ? "ok" : "bad"}`, null,
        h("span.check__i", { "aria-hidden": "true" }, icon(c.ok ? "check" : "warn")),
        h("div.check__tx", null,
          h("div.check__t", null, c.label, h("span.sr", null, c.ok ? " (ok)" : " (needs attention)")),
          c.detail ? h("div.check__d", null, c.detail) : null,
          !c.ok && p ? h("p.check__help", null, p.help) : null,
          fixBtn || settingsBtn ? h("div.check__acts", null, fixBtn, settingsBtn) : null,
          log));
    }));
  };
  const refresh = () => {
    if (busy) return;
    busy = true;
    sum.textContent = "Checking…";
    host.doctor().then((checks) => { paint(checks); o.changed(checks); }, (e) => {
      setKids(list, h("div.vempty", { role: "alert" }, icon("warn"), h("div.vempty__t", null, "The check itself failed"), h("div.vempty__d", null, errMessage(e))));
      sum.textContent = "";
    }).finally(() => { busy = false; });
  };
  if (o.initial) paint(o.initial);
  const recheck = h("button.btn.btn--sm", { type: "button", onclick: () => refresh() }, icon("retry"), "Recheck");
  const el = frame("Status", o.back, h("div.status__top", null, sum, recheck), list);
  return { el, refresh };
}

