// The pieces inside one assistant message: text, thinking, tool steps, plan, permission. Each is a small view
// with `update(seg, ctx)`, patched in place as events arrive.
import type { PermissionOption, ToolKind, ZoteroRef } from "../types.ts";
import type { AssistantMessage, Block, PlanEntry } from "./transcript.ts";
import { MdView, pretty } from "./mdview.ts";
import { h, icon, setKids, unwrapTicks } from "./dom.ts";
import { stepTitle } from "./steptitle.ts";
import type { IconName } from "./dom.ts";

export interface MsgActions {
  open(target: string | ZoteroRef): void;
  copy(text: string): void;
  retry(assistantId: string): void;
  answer(turn: string, permissionId: string, optionId: string): void;
  checkSetup(): void;
}

// ───────────────────────────── segments ─────────────────────────────

type Of<T extends Block["type"]> = Extract<Block, { type: T }>;
type Seg =
  | { kind: "text"; block: Of<"text"> }
  | { kind: "thought"; block: Of<"thought"> }
  | { kind: "steps"; blocks: Of<"tool">[] }
  | { kind: "plan"; block: Of<"plan"> }
  | { kind: "perm"; block: Of<"permission"> };

/** Blocks to segments: consecutive tool calls become one group. */
export function toSegs(blocks: readonly Block[], showThinking = true): Seg[] {
  const out: Seg[] = [];
  for (const b of blocks) {
    switch (b.type) {
      case "text": out.push({ kind: "text", block: b }); break;
      case "thought": if (showThinking) out.push({ kind: "thought", block: b }); break;
      case "plan": out.push({ kind: "plan", block: b }); break;
      case "permission": out.push({ kind: "perm", block: b }); break;
      case "tool": {
        const last = out[out.length - 1];
        if (last && last.kind === "steps") last.blocks.push(b);
        else out.push({ kind: "steps", blocks: [b] });
      }
    }
  }
  return out;
}

export interface Ctx { msg: AssistantMessage; live: boolean; expandTools: boolean; actions: MsgActions; announce(s: string): void }

export interface SegView { kind: Seg["kind"]; el: HTMLElement; update(seg: Seg, ctx: Ctx): void }

const TOOL_ICON: Record<ToolKind, IconName> = {
  read: "file", edit: "pencil", delete: "trash", move: "move", search: "search", execute: "terminal",
  think: "sparkle", fetch: "globe", switch_mode: "shield", other: "dot",
};

/** This block is the one still being written. */
const streaming = (c: Ctx, b: Block) => c.live && c.msg.blocks[c.msg.blocks.length - 1] === b;

export function textSeg(ctx: Ctx): SegView {
  const md = new MdView({ open: (href) => ctx.actions.open(href) });
  return { kind: "text", el: md.el, update(s, c) { if (s.kind === "text") md.set(s.block.text, streaming(c, s.block)); } };
}

// ───────────────────────────── thinking ─────────────────────────────

export function thoughtSeg(): SegView {
  let open = false;
  let block: Of<"thought"> | null = null;
  const label = h("span.thought__t");
  const body = h("div.thought__body", { hidden: true });
  const head = h("button.thought__head", { type: "button", "aria-expanded": "false", onclick: () => { open = !open; sync(); } },
    h("span.thought__dot"), label, icon("chevDown", "thought__chev"));
  const el = h("div.thought", null, head, body);
  let active = false;
  function sync() {
    head.setAttribute("aria-expanded", String(open));
    body.hidden = !open;
    el.classList.toggle("thought--open", open);
    if (open && block) body.textContent = block.text;
    label.textContent = active ? "Thinking" : "Thought";
    el.classList.toggle("thought--active", active);
  }
  return {
    kind: "thought", el,
    update(s, c) {
      if (s.kind !== "thought") return;
      block = s.block;
      active = streaming(c, s.block);
      sync();
    },
  };
}

// ───────────────────────────── tool steps ─────────────────────────────

function statusMark(status: string, stopped: boolean): HTMLElement {
  // A step cut off by the user's Stop is not a failure: a quiet dash, not a red cross.
  const mark = (cls: string, label: string, kid: Node) => h(`span.step__st${cls}`, { "aria-label": label.toLowerCase(), title: label }, kid);
  if (stopped && status === "failed") return mark("", "Stopped", icon("stop"));
  if (status === "done") return mark("", "Done", icon("check"));
  if (status === "failed") return mark(".step__st--failed", "Failed", icon("close"));
  return mark("", status === "pending" ? "Waiting" : "Running", h("span.pulse"));
}

function detailBox(label: string, value: unknown): HTMLElement {
  const { text, trimmed } = pretty(value);
  return h("div.sd", null,
    h("div.sd__head", null, h("span.eyebrow", null, label), trimmed ? h("span.sd__trim", null, "trimmed") : null),
    h("pre.sd__pre", null, text));
}

export function stepsSeg(): SegView {
  const toggled = new Map<string, boolean>(); // what the user opened or closed survives patches
  let expand = false; // the "expand tool steps" setting is the default for the rest
  const isOpen = (id: string) => toggled.get(id) ?? expand;
  let folded = true; // a finished run of more than four steps starts folded
  let foldedTouched = false;
  const head = h("button.steps__head", { type: "button", "aria-expanded": "false", onclick: () => { folded = !folded; foldedTouched = true; paint(); } });
  const list = h("div.steps__list");
  const el = h("div.steps", null, head, list);
  let blocks: readonly Of<"tool">[] = [];
  let done = false;
  let stopped = false;
  // a row is rebuilt only when its block, its open state or the stopped state changed
  const rows = new Map<string, { block: Of<"tool">; open: boolean; stopped: boolean; el: HTMLElement }>();

  function row(b: Of<"tool">): HTMLElement {
    const opened = isOpen(b.id);
    const hasDetail = b.input !== undefined || !!b.output;
    const raw = unwrapTicks(b.title);
    const shown = stepTitle(raw);
    const btn = h("button.step__row", {
      type: "button", "aria-expanded": hasDetail ? String(opened) : null, disabled: hasDetail ? null : true,
      onclick: () => { toggled.set(b.id, !isOpen(b.id)); paint(); },
    },
      h("span.step__icon", null, icon(TOOL_ICON[b.kind] ?? "dot")),
      b.name ? h("span.step__name", null, b.name) : null,
      h("span.step__title", { title: raw && shown !== raw ? raw : null }, shown || "Working"),
      statusMark(b.status, stopped),
      hasDetail ? icon("chevDown", "step__chev") : null);
    const kids: HTMLElement[] = [btn];
    if (opened && hasDetail) {
      kids.push(h("div.step__detail", null,
        b.input !== undefined ? detailBox("Input", b.input) : null,
        b.output ? detailBox("Output", b.output) : null));
    }
    return h(`div.step${opened ? ".step--open" : ""}`, null, ...kids);
  }

  function paint() {
    const fold = done && blocks.length > 4;
    const failed = blocks.filter((b) => b.status === "failed").length;
    head.hidden = !fold;
    el.classList.toggle("steps--folded", fold && folded);
    if (fold) {
      head.setAttribute("aria-expanded", String(!folded));
      setKids(head, icon("list"), h("span.steps__sum", null, `${blocks.length} steps`, failed ? h("span.steps__bad", null, ` · ${failed} failed`) : null), icon("chevDown", "steps__chev"));
    }
    list.hidden = fold && folded;
    if (list.hidden) return;
    // Rebuild only the rows whose block changed or whose open state flipped.
    const seen = new Set<string>();
    let prev: Node | null = null;
    for (const b of blocks) {
      seen.add(b.id);
      const cur = rows.get(b.id);
      let rowEl: HTMLElement;
      if (cur && cur.block === b && cur.open === isOpen(b.id) && cur.stopped === stopped) rowEl = cur.el;
      else {
        rowEl = row(b);
        if (cur) cur.el.replaceWith(rowEl);
        rows.set(b.id, { block: b, open: isOpen(b.id), stopped, el: rowEl });
      }
      if (rowEl.parentNode !== list) list.insertBefore(rowEl, prev ? prev.nextSibling : list.firstChild);
      prev = rowEl;
    }
    for (const [id, r] of rows) if (!seen.has(id)) { r.el.remove(); rows.delete(id); }
  }
  return {
    kind: "steps", el,
    update(s, c) {
      if (s.kind !== "steps") return;
      blocks = s.blocks;
      done = c.msg.done;
      expand = c.expandTools;
      stopped = c.msg.stop === "cancelled";
      if (!foldedTouched) folded = !expand;
      paint();
    },
  };
}

// ───────────────────────────── plan ─────────────────────────────

const PLAN_LABEL: Record<PlanEntry["status"], string> = { pending: "to do", in_progress: "in progress", completed: "done" };

export function planSeg(): SegView {
  const list = h("ul.plan__list");
  const el = h("div.plan", null, h("div.plan__head", null, icon("list"), h("span", null, "Plan")), list);
  let last: PlanEntry[] | null = null;
  return {
    kind: "plan", el,
    update(s) {
      if (s.kind !== "plan" || s.block.entries === last) return;
      last = s.block.entries;
      setKids(list, s.block.entries.map((e) => h(`li.plan__i.plan__i--${e.status}`, null,
        h("span.plan__box", { "aria-hidden": "true" }, e.status === "completed" ? icon("check") : null),
        h("span.plan__t", null, e.content),
        h("span.sr", null, ` (${PLAN_LABEL[e.status]})`))));
    },
  };
}

// ───────────────────────────── permission ─────────────────────────────

const RANK: Record<string, number> = { allow_once: 0, allow_always: 1, reject_once: 2, reject_always: 3 };
const optionClass = (o: PermissionOption) =>
  o.kind === "allow_once" ? "btn.btn--solid" : o.kind.startsWith("allow") ? "btn" : "btn.btn--quiet-danger";

export function permSeg(): SegView {
  const el = h("div.perm", { role: "group" });
  let last: Of<"permission"> | null = null;
  let lastDone: boolean | null = null;
  let showInput = false;
  return {
    kind: "perm", el,
    update(s, c) {
      if (s.kind !== "perm") return;
      const b = s.block;
      if (b === last && lastDone === c.msg.done) return;
      const wasPending = last ? last.resolved === undefined : false;
      last = b; lastDone = c.msg.done;
      el.setAttribute("aria-label", `Permission needed: ${b.title}`);
      const chosen = b.options.find((o) => o.id === b.resolved);
      const hasInput = b.input !== undefined && b.input !== null;
      if (b.resolved !== undefined || c.msg.done) {
        const allowed = chosen ? chosen.kind.startsWith("allow") : false;
        el.className = "perm perm--done";
        setKids(el,
          icon(chosen ? (allowed ? "check" : "close") : "info"),
          h("span.perm__what", null, unwrapTicks(b.title)),
          h("span.perm__res", null, chosen ? chosen.name : "No answer"));
        return;
      }
      el.className = "perm";
      const opts = [...b.options].sort((x, y) => (RANK[x.kind] ?? 9) - (RANK[y.kind] ?? 9));
      const inputBox = h("div.perm__raw", { hidden: !showInput }, hasInput ? detailBox("Input", b.input) : null);
      const toggle = h("button.perm__more", { type: "button", "aria-expanded": String(showInput), onclick: () => {
        showInput = !showInput; inputBox.hidden = !showInput; toggle.setAttribute("aria-expanded", String(showInput));
        toggle.textContent = showInput ? "Hide input" : "Show input";
      } }, showInput ? "Hide input" : "Show input");
      setKids(el,
        h("div.perm__q", null, icon("shield"), "Needs your OK", b.name ? h("span.step__name", null, b.name) : null),
        h("div.perm__what", null, unwrapTicks(b.title)),
        hasInput ? toggle : null, hasInput ? inputBox : null,
        h("div.perm__opts", null, opts.map((o) => h(`button.${optionClass(o)}.btn--sm`, { type: "button", onclick: () => c.actions.answer(c.msg.id, b.id, o.id) }, o.name))));
      if (!wasPending) c.announce(`Permission needed: ${b.title}`);
    },
  };
}

