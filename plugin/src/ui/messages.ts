// The transcript's DOM. `Feed.update(state)` walks the messages, and only a message whose object changed
// (applyEvent keeps untouched ones identical) is patched; inside it only the blocks that changed.
// A streamed token therefore costs one MdView.set on the last block, whatever the length of the chat.
import type { ChipSummary } from "../types.ts";
import type { AssistantMessage, Message, NoticeMessage, TranscriptState, UserMessage } from "./transcript.ts";
import { answerText, textsOf } from "./transcript.ts";
import { collectSources } from "./markdown.ts";
import type { CitedSource } from "./markdown.ts";
import { permSeg, planSeg, stepsSeg, textSeg, thoughtSeg, toSegs } from "./parts.ts";
import type { Ctx, MsgActions, SegView } from "./parts.ts";
import { CHIP_ICON, clear, env, flashCheck, fmtTokens, h, icon, setKids } from "./dom.ts";

export type { MsgActions };

interface PatchFlags { live: boolean; canRetry: boolean; showThinking: boolean; expandTools: boolean; showUsage: boolean }

// ───────────────────────────── assistant message ─────────────────────────────

const STOP_NOTE: Record<string, string> = {
  cancelled: "Stopped",
  max_tokens: "Stopped at the length limit",
  refusal: "The model declined to answer",
};

class AssistantView {
  readonly el: HTMLElement;
  private parts = h("div.parts");
  private working = h("div.working", { hidden: true, role: "status" }, h("span.pulse"), h("span.working__t", null, "Thinking"));
  private foot = h("div.foot", { hidden: true });
  private segs: SegView[] = [];
  private sourcesOpen = false;
  private sourcesFor: AssistantMessage | null = null;
  private sources: CitedSource[] = [];
  private msg!: AssistantMessage;
  private flags: PatchFlags = { live: false, canRetry: false, showThinking: true, expandTools: false, showUsage: false };

  private actions: MsgActions;
  private announce: (s: string) => void;

  constructor(actions: MsgActions, announce: (s: string) => void) {
    this.actions = actions;
    this.announce = announce;
    this.el = h("div.msg.msg--assistant", { role: "article" }, this.parts, this.working, this.foot);
  }

  patch(msg: AssistantMessage, flags: PatchFlags): void {
    const sameMsg = this.msg === msg;
    const f = this.flags;
    const sameOpts = f.showThinking === flags.showThinking && f.expandTools === flags.expandTools && f.showUsage === flags.showUsage;
    if (sameMsg && sameOpts && f.live === flags.live && f.canRetry === flags.canRetry) return;
    this.msg = msg; this.flags = flags;
    this.el.dataset.state = msg.done ? msg.stop ?? "end_turn" : "running";
    const ctx: Ctx = { msg, live: !msg.done, expandTools: flags.expandTools, actions: this.actions, announce: this.announce };
    if (!sameMsg || !sameOpts) this.patchParts(msg, ctx);
    this.paintWorking(msg);
    this.paintFoot(msg, flags);
  }

  private patchParts(msg: AssistantMessage, ctx: Ctx): void {
    const segs = toSegs(msg.blocks, this.flags.showThinking);
    segs.forEach((seg, i) => {
      let v = this.segs[i];
      if (v && v.kind !== seg.kind) { // a different kind at this position: replace it and everything after
        for (const old of this.segs.splice(i)) old.el.remove();
        v = undefined;
      }
      if (!v) {
        v = seg.kind === "text" ? textSeg(ctx) : seg.kind === "thought" ? thoughtSeg() : seg.kind === "steps" ? stepsSeg() : seg.kind === "plan" ? planSeg() : permSeg();
        this.segs[i] = v;
        this.parts.appendChild(v.el);
      }
      v.update(seg, ctx);
    });
    for (const old of this.segs.splice(segs.length)) old.el.remove();
  }

  private paintWorking(msg: AssistantMessage): void {
    const last = msg.blocks[msg.blocks.length - 1];
    const waiting = msg.blocks.some((b) => b.type === "permission" && b.resolved === undefined);
    const between = !last || (last.type === "tool" && last.status !== "running" && last.status !== "pending") || last.type === "plan"
      || (last.type === "permission" && last.resolved !== undefined) || (last.type === "thought" && !this.flags.showThinking);
    const show = !msg.done && !waiting && between;
    this.working.hidden = !show;
    const t = this.working.querySelector(".working__t");
    if (t && show) t.textContent = last ? "Working" : "Thinking";
  }

  private paintFoot(msg: AssistantMessage, f: PatchFlags): void {
    if (!msg.done) { this.foot.hidden = true; return; }
    this.foot.hidden = false;
    if (this.sourcesFor !== msg) {
      this.sourcesFor = msg;
      this.sources = collectSources(textsOf(msg));
    }
    const text = answerText(msg);
    const copyBtn = h("button.iconbtn.iconbtn--sm", { type: "button", "aria-label": "Copy answer", title: "Copy", disabled: text ? null : true }, icon("copy"));
    copyBtn.addEventListener("click", () => { this.actions.copy(text); flashCheck(copyBtn); });
    const retryBtn = f.canRetry
      ? h("button.iconbtn.iconbtn--sm", { type: "button", "aria-label": "Try again", title: "Try again", onclick: () => this.actions.retry(msg.id) }, icon("retry"))
      : null;
    const u = msg.usage;
    // Tokens and cost are off by default: a running price under every answer makes people hesitate to ask.
    const usage = f.showUsage && u && (u.inputTokens || u.outputTokens || u.costUsd)
      ? [u.inputTokens ? `${fmtTokens(u.inputTokens)} in` : "", u.outputTokens ? `${fmtTokens(u.outputTokens)} out` : "", u.costUsd ? `$${u.costUsd.toFixed(u.costUsd < 0.1 ? 3 : 2)}` : ""].filter(Boolean).join(" · ")
      : "";
    const n = this.sources.length;
    const srcBtn = n
      ? h("button.foot__src", { type: "button", "aria-expanded": String(this.sourcesOpen), onclick: () => { this.sourcesOpen = !this.sourcesOpen; this.paintFoot(msg, f); } },
        icon("chev", "foot__chev"), `${n} ${n === 1 ? "source" : "sources"}`)
      : null;
    const list = n && this.sourcesOpen
      ? h("ul.sources", null, this.sources.map((s) => h("li", null,
        h("button.source", { type: "button", onclick: () => this.actions.open(s.href), title: s.href.replace(/^zotero:\/\//, "") },
          icon("item"), h("span.source__t", null, s.label),
          s.pages.length ? h("span.source__p", null, `${s.pages.length === 1 ? "p." : "pp."} ${s.pages.join(", ")}`) : null))))
      : null;
    const note = STOP_NOTE[msg.stop ?? "end_turn"];
    this.foot.className = `foot${this.sourcesOpen ? " foot--open" : ""}`;
    setKids(this.foot,
      note ? h(`div.stopnote.stopnote--${msg.stop}`, null, icon(msg.stop === "cancelled" ? "stop" : "warn"), note) : null,
      h("div.foot__row", null, srcBtn, h("span.foot__fill"), usage ? h("span.foot__usage", null, usage) : null, h("span.foot__acts", null, copyBtn, retryBtn)),
      list);
  }
}

// ───────────────────────────── user and notice ─────────────────────────────

function chipRow(chips: ChipSummary[], actions: MsgActions): HTMLElement {
  return h("div.ubub__chips", null, chips.map((c) =>
    h("button.chip.chip--sm", { type: "button", title: c.text ? `${c.label}: ${c.text.slice(0, 300)}` : c.label, onclick: () => actions.open(c.ref) },
      h("span.chip__i", null, icon(CHIP_ICON[c.kind] ?? "item")), h("span.chip__t", null, c.label))));
}

function userView(m: UserMessage, actions: MsgActions): HTMLElement {
  return h("div.msg.msg--user", null,
    h("div.ubub", null, m.chips.length ? chipRow(m.chips, actions) : null, h("p.ubub__text", null, m.text)));
}

function noticeView(m: NoticeMessage, actions: MsgActions): HTMLElement {
  const bad = m.level !== "info";
  return h(`div.msg.notice.notice--${m.level}`, { role: m.level === "error" ? "alert" : "status" },
    h("span.notice__i", null, icon(m.level === "info" ? "info" : "warn")),
    h("div.notice__body", null,
      h("div.notice__msg", null, m.message),
      m.hint ? h("div.notice__hint", null, m.hint) : null,
      bad ? h("div.notice__acts", null, h("button.btn.btn--sm", { type: "button", onclick: () => actions.checkSetup() }, "Check setup")) : null));
}

// ───────────────────────────── the feed ─────────────────────────────

export class Feed {
  readonly el: HTMLElement;
  private scroller: HTMLElement;
  private inner: HTMLElement;
  private pendingEl = h("div.working.working--pending", { hidden: true, role: "status" }, h("span.pulse"), h("span.working__t"));
  private jump: HTMLElement;
  private live: HTMLElement;
  private views = new Map<string, { msg: Message; el: HTMLElement; asst?: AssistantView }>();
  private stick = true;
  private lastTop = 0;
  private empty: HTMLElement | null = null;
  private lastState: TranscriptState | null = null;
  private lastRetry = false;
  private opts = { showThinking: true, expandTools: false, showUsage: false };

  private actions: MsgActions;

  constructor(actions: MsgActions) {
    this.actions = actions;
    this.inner = h("div.feed__inner", null, this.pendingEl);
    this.scroller = h("div.feed", { tabindex: "0", role: "log", "aria-label": "Conversation", "aria-live": "off" }, this.inner);
    this.jump = h("button.jump", { type: "button", hidden: true, "aria-label": "Jump to the latest message", title: "Jump to the latest", onclick: () => this.toBottom(true) }, icon("down"));
    this.live = h("div.sr", { role: "status", "aria-live": "polite" });
    this.el = h("div.feedwrap", null, this.scroller, this.jump, this.live);
    this.scroller.addEventListener("scroll", () => {
      const s = this.scroller;
      const dist = s.scrollHeight - s.scrollTop - s.clientHeight;
      if (dist < 48) this.stick = true;
      else if (s.scrollTop < this.lastTop - 1) this.stick = false;
      this.lastTop = s.scrollTop;
      this.jump.hidden = this.stick;
    });
    // The scroll event comes a frame late, and a token rendered in that gap would snap a reader back down. Intent to read
    // upward (wheel up, PageUp, Home, arrow up) lets go of the bottom at once; reaching the bottom again takes it back.
    const letGo = () => { if (this.scroller.scrollTop > 0) { this.stick = false; this.jump.hidden = false; } };
    this.scroller.addEventListener("wheel", (e) => { if (e.deltaY < 0) letGo(); }, { passive: true });
    this.scroller.addEventListener("keydown", (e) => { if (e.key === "PageUp" || e.key === "Home" || e.key === "ArrowUp") letGo(); });
    const RO = env.win.ResizeObserver;
    if (RO) new RO(() => { if (this.stick) this.toBottom(false); }).observe(this.inner);
  }

  /** Content shown when the transcript is empty (the empty state); removed as soon as a message exists. */
  setEmpty(node: HTMLElement | null): void {
    if (this.empty && this.empty !== node) this.empty.remove();
    this.empty = node;
    this.syncEmpty();
  }
  private syncEmpty(): void {
    const has = this.views.size > 0;
    if (this.empty) {
      if (!has && this.empty.parentNode !== this.inner) this.inner.insertBefore(this.empty, this.inner.firstChild);
      if (has && this.empty.parentNode === this.inner) this.empty.remove();
    }
    this.inner.classList.toggle("feed__inner--empty", !has && !!this.empty);
  }

  announce(s: string): void { this.live.textContent = s; }

  /** A line under the last message while the agent is starting ("Starting Claude Code"); null hides it. */
  setPending(label: string | null): void {
    this.pendingEl.hidden = label === null;
    const t = this.pendingEl.querySelector(".working__t");
    if (t && label !== null) t.textContent = label;
    if (label !== null && this.stick) this.toBottom(false);
  }

  toBottom(smooth: boolean): void {
    const s = this.scroller;
    this.stick = true;
    this.jump.hidden = true;
    const reduce = env.win.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (smooth && !reduce) s.scrollTo({ top: s.scrollHeight, behavior: "smooth" });
    else s.scrollTop = s.scrollHeight;
    this.lastTop = s.scrollTop;
  }

  /** Re-render after the state changed. `canRetry`: the last answer may offer "Try again". */
  update(state: TranscriptState, canRetry: boolean): void {
    this.lastState = state;
    this.lastRetry = canRetry;
    const msgs = state.messages;
    // A different conversation (new chat, resumed one): rebuild.
    if (this.views.size && (msgs[0]?.id !== this.views.keys().next().value || msgs.length < this.views.size)) this.reset();
    const lastAsst = msgs.findLastIndex((m) => m.role === "assistant");

    msgs.forEach((m, i) => {
      let v = this.views.get(m.id);
      if (!v) {
        v = this.create(m);
        this.views.set(m.id, v);
        this.inner.insertBefore(v.el, this.pendingEl);
      }
      if (v.asst && m.role === "assistant") v.asst.patch(m, { live: state.running === m.id, canRetry: canRetry && i === lastAsst && m.done, ...this.opts });
      else if (v.msg !== m) {
        const fresh = this.create(m).el;
        v.el.replaceWith(fresh);
        v.el = fresh;
      }
      v.msg = m;
    });
    this.syncEmpty();
    if (this.stick) this.toBottom(false);
  }

  private create(m: Message): { msg: Message; el: HTMLElement; asst?: AssistantView } {
    if (m.role === "assistant") {
      const asst = new AssistantView(this.actions, (s) => this.announce(s));
      return { msg: m, el: asst.el, asst };
    }
    return { msg: m, el: m.role === "user" ? userView(m, this.actions) : noticeView(m, this.actions) };
  }

  private reset(): void {
    this.views.clear();
    for (const k of [...this.inner.children]) if (k !== this.empty && k !== this.pendingEl) k.remove();
    this.stick = true;
  }

  /** The chat settings that change how messages look; applied to what is already shown too. */
  setOptions(o: { showThinking: boolean; expandTools: boolean; showUsage: boolean }): void {
    if (o.showThinking === this.opts.showThinking && o.expandTools === this.opts.expandTools && o.showUsage === this.opts.showUsage) return;
    this.opts = o;
    if (this.lastState) this.update(this.lastState, this.lastRetry);
  }

  clear(): void { this.reset(); clear(this.live); this.syncEmpty(); }
}
