// The panel's other screens: the empty state, history, settings, and the doctor/status view.
// Each is a plain function returning an element and a `refresh` where it has data to reload.
import type { BackendId, DoctorCheck, PanelHost, SavedSession } from "../types.ts";
import { copyText, dayGroup, env, errMessage, h, icon, isMac, relTime, setKids, slotLabel, svg } from "./dom.ts";
import { shortPath } from "./settings-model.ts";
import { skillLabel } from "./skills-model.ts";
import { t } from "../i18n.ts";
import type { Key } from "../i18n.ts";
import type { Pinned } from "./skills-model.ts";

// Product names stay as they are; "Custom" is a word, so it is looked up each time it is drawn.
export const BACKEND_LABEL: Record<BackendId, string> = { "claude-code": "Claude Code", codex: "Codex", pi: "pi", get custom() { return t("agent.custom.tab"); } };
export const BACKENDS: BackendId[] = ["claude-code", "codex", "pi", "custom"];

/**
 * The logo (assets/logo.svg): a hollow square chat box with a small tail, and a red Z in it (--brand: the accent setting never recolours it). It follows the theme. The parts
 * carry classes so the welcome can animate them (styles-welcome.ts: the lines draw in, the Z breathes while the check runs and
 * bursts when it is done); the small dots only show in the burst.
 */
export const mark = (hero = false): HTMLElement => h(`div.mark${hero ? ".mark--hero" : ""}`, null, svg("svg", { viewBox: "-16 -22 160 160", "aria-hidden": "true" },
  svg("path", { class: "mk-b", d: "M40 12H88a26 26 0 0 1 26 26V78a26 26 0 0 1-26 26H62L38 118V104H40a26 26 0 0 1-26-26V38a26 26 0 0 1 26-26Z", fill: "none", stroke: "var(--ink)", "stroke-width": "5.5", "stroke-linejoin": "round", "stroke-linecap": "round", pathLength: "1" }),
  ...[[67, 28], [28, 67], [-28, 67], [-67, 28], [-67, -28], [-28, -67], [28, -67], [67, -28]].map(([dx, dy]) => svg("circle", { class: "mk-p", cx: "64", cy: "58", r: "3", fill: "var(--brand)", style: `--dx:${dx}px;--dy:${dy}px` })),
  svg("g", { class: "mk-t" }, svg("path", { class: "mk-s", d: "M47 39H81L47 77H81", fill: "none", stroke: "var(--brand)", "stroke-width": "8", "stroke-linejoin": "round", "stroke-linecap": "round", pathLength: "1" }))));

// ───────────────────────────── problems (the unavailable states) ─────────────────────────────

const PROBLEM: Record<DoctorCheck["id"], [Key, Key]> = {
  "zotero-api": ["problem.zoteroApi.title", "problem.zoteroApi.help"],
  "write-access": ["problem.writeAccess.title", "problem.writeAccess.help"],
  cli: ["problem.cli.title", "problem.cli.help"],
  node: ["problem.node.title", "problem.node.help"],
  backend: ["problem.backend.title", "problem.backend.help"],
};
export const problem = (id: DoctorCheck["id"]): { title: string; help: string } => ({ title: t(PROBLEM[id][0]), help: t(PROBLEM[id][1]) });

/** The button that runs a check's fix, streaming its output into `log`; `after` runs when it ends. */
export function fixButton(check: DoctorCheck, cls: string, log: HTMLElement, after: () => void): HTMLButtonElement | null {
  const fix = check.fix;
  if (!fix) return null;
  const btn = h(`button.btn.btn--sm${cls}`, { type: "button" }, fix.label) as HTMLButtonElement;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = t("common.working");
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
  const p = problem(check.id);
  const log = h("pre.fixlog", { hidden: true, "aria-live": "polite" });
  const fixBtn = fixButton(check, ".btn--solid", log, handlers.recheck);
  return h("div.setup", { role: "region", "aria-label": t("setup.region") },
    h("div.setup__head", null, h("span.setup__i", null, icon("warn")), h("div.setup__tx", null,
      h("div.setup__t", null, p.title),
      check.detail ? h("div.setup__d", null, check.detail) : null)),
    h("p.setup__help", null, p.help),
    log,
    h("div.setup__acts", null,
      fixBtn,
      check.id === "backend" ? h("button.btn.btn--sm", { type: "button", onclick: handlers.openSettings }, t("common.openSettings")) : null,
      h("button.btn.btn--sm.btn--quiet", { type: "button", onclick: handlers.recheck }, t("common.recheck")),
      more > 0 ? h("button.btn.btn--sm.btn--quiet", { type: "button", onclick: handlers.openStatus }, t("setup.more", { n: more })) : null));
}

// ───────────────────────────── empty state ─────────────────────────────

/** The intro, then the pinned skills and prompts (slot order) as one list; it goes with the empty state when the chat has messages. */
export function emptyState(o: { pins: Pinned[]; ready: boolean; run(slot: number): void; edit(): void }): HTMLElement {
  const row = (p: Pinned) => {
    const label = p.kind === "prompt" ? p.prompt.title || p.prompt.text : skillLabel(p.name);
    return h("li", null, h("button.pin", {
      type: "button", disabled: o.ready ? null : true, dataset: { slot: String(p.slot) },
      title: p.kind === "prompt" ? (p.prompt.title ? `${p.prompt.title}\n${p.prompt.text}` : p.prompt.text) : t("empty.pinSkillTip", { label, name: p.name }),
      "aria-keyshortcuts": isMac() ? `Meta+Control+${p.slot}` : `Control+Alt+${p.slot}`,
      onclick: () => o.run(p.slot),
    }, h("span.pin__t", null, label), p.kind === "skill" ? h("span.pin__skill", null, icon("sparkle"), h("span.sr", null, t("empty.skill"))) : null, h("kbd.pin__k", null, slotLabel(p.slot))));
  };
  return h("div.empty", null,
    mark(),
    h("h2", null, t("empty.title")),
    h("p.empty__lead", null, t("empty.lead")),
    h("section.pins", { "aria-label": t("empty.startWith") },
      h("div.pins__head", null, h("h3.eyebrow", null, t("empty.startWith")), h("button.lnk", { type: "button", onclick: o.edit }, icon("pencil"), t("common.edit"))),
      o.pins.length
        ? h("ul.pins__list", null, o.pins.map(row))
        : h("p.pins__none", null, t("empty.noPins"))));
}

// ───────────────────────────── a view's frame ─────────────────────────────

export function frame(title: string, back: () => void, ...body: (HTMLElement | null)[]): HTMLElement {
  return h("div.vw", null,
    h("div.vw__head", null,
      h("button.iconbtn", { type: "button", "aria-label": t("frame.back"), title: t("frame.backTip"), onclick: back }, icon("back")),
      h("h2.vw__t", null, title)),
    h("div.vw__body", null, ...body));
}

// ───────────────────────────── history ─────────────────────────────

export function historyView(host: PanelHost, o: { current(): string | null; back(): void; resume(s: SavedSession): void; deleted(id: string): void }): { el: HTMLElement; refresh(): void } {
  const list = h("div.hist__list");
  const q = h("input.input", { type: "search", placeholder: t("history.search"), "aria-label": t("history.search") }) as HTMLInputElement;
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
        h("div.vempty__t", null, needle ? t("history.noMatch") : t("history.none")),
        h("div.vempty__d", null, needle ? t("history.noMatchHint") : t("history.noneHint"))));
      return;
    }
    const out: HTMLElement[] = [];
    if (rows.some((s) => host.resumeCommand(s))) out.push(h("p.hist__note", null, icon("terminal"), t("history.terminalNote")));
    let group = "";
    for (const s of rows) {
      const g = dayGroup(s.updatedAt);
      if (g !== group) { group = g; out.push(h("div.hist__group", null, g)); }
      const isCur = o.current() === s.id;
      if (confirming === s.id) {
        out.push(h("div.hrow.hrow--confirm", { role: "group", "aria-label": t("history.deleteAsk", { title: s.title }) },
          h("span.hrow__q", null, t("history.deleteThis")),
          h("button.btn.btn--sm.btn--danger", { type: "button", onclick: () => void del(s) }, t("common.delete")),
          h("button.btn.btn--sm", { type: "button", onclick: () => { confirming = null; paint(); } }, t("common.cancel"))));
        continue;
      }
      const cmd = host.resumeCommand(s);
      const done = copied === s.id;
      out.push(h(`div.hrow${isCur ? ".hrow--on" : ""}`, null,
        h("button.hrow__main", { type: "button", onclick: () => o.resume(s), ...(isCur ? { "aria-current": "true" } : {}) },
          h("span.hrow__t", null, s.title || t("history.untitled")),
          h("span.hrow__m", null, `${BACKEND_LABEL[s.backend] ?? s.backend} · ${relTime(s.updatedAt)}`),
          s.cwd ? h("span.hrow__f", { title: s.cwd }, icon("folder"), h("span", null, shortPath(s.cwd, 30))) : null),
        cmd ? h(`button.${done ? "lnk.hrow__copied" : "iconbtn.iconbtn.iconbtn--sm.hrow__cmd"}`, { type: "button", title: t("history.copyCmd"), "aria-label": done ? t("common.copied") : t("history.copyCmdFor", { title: s.title || t("history.chat") }), onclick: () => void copyCmd(s, cmd) }, icon(done ? "check" : "terminal"), done ? t("common.copied") : null) : null,
        h("button.iconbtn.iconbtn--sm.hrow__del", { type: "button", "aria-label": t("history.deleteFor", { title: s.title || t("history.chat") }), title: t("common.delete"), onclick: () => { confirming = s.id; paint(); } }, icon("trash"))));
    }
    setKids(list, out);
  };
  const del = async (s: SavedSession) => {
    try { await host.deleteSession(s.id); } catch (e) { confirming = null; all = (all ?? []); setKids(list, h("div.vwerr", { role: "alert" }, t("history.deleteFailed", { error: errMessage(e) }))); env.win.setTimeout(paint, 2500); return; }
    all = (all ?? []).filter((x) => x.id !== s.id);
    confirming = null;
    o.deleted(s.id);
    paint();
  };
  const refresh = () => {
    skeleton();
    host.sessions().then((s) => { all = s; confirming = null; paint(); }, (e) => {
      setKids(list, h("div.vempty", { role: "alert" }, icon("warn"), h("div.vempty__t", null, t("history.loadFailed")), h("div.vempty__d", null, errMessage(e)),
        h("button.btn.btn--sm", { type: "button", onclick: refresh }, t("common.tryAgain"))));
    });
  };
  q.addEventListener("input", paint);
  const el = frame(t("history.title"), o.back, h("div.hist__search", null, q), list);
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
    setKids(sum, h("span.status__dot", { "data-ok": String(bad.length === 0) }), bad.length ? t("status.attention", { n: bad.length }) : t("status.allSet"));
    setKids(list, checks.map((c) => {
      const log = h("pre.fixlog", { hidden: !logs.has(c.id), "aria-live": "polite" }, logs.get(c.id) ?? "");
      const p = problem(c.id);
      const fixBtn = c.ok ? null : fixButton(c, ".btn--solid", log, () => { logs.set(c.id, log.textContent ?? ""); refresh(); });
      const settingsBtn = c.id === "backend" && !c.ok ? h("button.btn.btn--sm", { type: "button", onclick: o.openSettings }, t("common.openSettings")) : null;
      return h(`div.check.check--${c.ok ? "ok" : "bad"}`, null,
        h("span.check__i", { "aria-hidden": "true" }, icon(c.ok ? "check" : "warn")),
        h("div.check__tx", null,
          h("div.check__t", null, c.label, h("span.sr", null, c.ok ? t("status.okSr") : t("status.badSr"))),
          c.detail ? h("div.check__d", null, c.detail) : null,
          !c.ok ? h("p.check__help", null, p.help) : null,
          fixBtn || settingsBtn ? h("div.check__acts", null, fixBtn, settingsBtn) : null,
          log));
    }));
  };
  const refresh = () => {
    if (busy) return;
    busy = true;
    sum.textContent = t("common.checking");
    host.doctor().then((checks) => { paint(checks); o.changed(checks); }, (e) => {
      setKids(list, h("div.vempty", { role: "alert" }, icon("warn"), h("div.vempty__t", null, t("status.checkFailed")), h("div.vempty__d", null, errMessage(e))));
      sum.textContent = "";
    }).finally(() => { busy = false; });
  };
  if (o.initial) paint(o.initial);
  const recheck = h("button.btn.btn--sm", { type: "button", onclick: () => refresh() }, icon("retry"), t("common.recheck"));
  const el = frame(t("status.title"), o.back, h("div.status__top", null, sum, recheck), list);
  return { el, refresh };
}

