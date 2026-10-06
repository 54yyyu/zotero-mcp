import { test } from "node:test";
import assert from "node:assert/strict";
import { Pacer, thinkOf } from "../../src/ui/think.ts";
import type { Think } from "../../src/ui/think.ts";
import type { Block } from "../../src/ui/transcript.ts";
import { THINK_STYLES } from "../../src/ui/styles-think.ts";
import { STYLES } from "../../src/ui/styles.ts";

const tool = (title: string, kind: Extract<Block, { type: "tool" }>["kind"], status: "running" | "pending" | "done" | "failed" = "running", name?: string): Block =>
  ({ type: "tool", id: title, title, kind, status, ...(name ? { name } : {}) });
const said = (blocks: Block[]) => { const t = thinkOf(blocks); return `${t.state}: ${t.label}`; };

test("the turn's activity comes from its newest block", () => {
  assert.equal(said([]), "thinking: Thinking");
  assert.equal(said([{ type: "thought", text: "Let me see" }]), "thinking: Thinking");
  assert.equal(said([{ type: "thought", text: "x" }, { type: "text", text: "The callback" }]), "writing: Writing the answer");
  assert.equal(said([{ type: "text", text: "Done. " }, tool("zotero-cli search callback", "execute", "done")]), "thinking: Thinking", "all steps finished: deciding what next");
  assert.equal(said([{ type: "plan", entries: [] }]), "thinking: Thinking");
  const perm: Block = { type: "permission", id: "p", title: "Run rm", kind: "execute", options: [] };
  assert.equal(said([tool("rm -rf x", "execute", "pending"), perm]), "waiting: Waiting for your OK");
  assert.equal(said([tool("rm -rf x", "execute"), { ...perm, resolved: "allow" }]), "working: Working", "answered: back to the step");
  assert.equal(said([{ type: "text", text: "a" }, perm, { type: "text", text: "b" }]), "waiting: Waiting for your OK", "an open card wins wherever it is");
});

test("a running step reads as what it does, never as a raw command", () => {
  const cases: [Block, string][] = [
    [tool('cd ~/Documents && zotero-cli --json search "hiring discrimination audit" --limit 10', "execute", "running", "Bash"), "searching: Searching your library"],
    [tool("zotero-cli search callback | head -5", "execute"), "searching: Searching your library"],
    [tool("zotero-cli --json search --mode semantic 'callback gap'", "search"), "searching: Searching your library"],
    [tool("hiring audit meta-analysis", "fetch", "running", "WebSearch"), "searching: Searching the web"],
    [tool("Fetch https://osf.io/x", "fetch"), "searching: Searching the web"],
    [tool("grep -r callback .", "search"), "searching: Searching files"],
    [tool("zotero-cli --json read HGEXED7Q --start-page 7", "execute"), "reading: Reading page 7"],
    [tool("cd /tmp && zotero-cli read HGEXED7Q --start-page 7 --end-page 8", "execute", "pending"), "reading: Reading pages 7–8"],
    [tool("zotero-cli get fulltext HGEXED7Q", "execute"), "reading: Reading the full text"],
    [tool("zotero-cli outline HGEXED7Q", "execute"), "reading: Reading the outline"],
    [tool("zotero-cli annotations list HGEXED7Q", "execute"), "reading: Reading the annotations"],
    [tool("Read /Users/you/Zotero Chat/.agents/skills/zotero-cli/SKILL.md", "read", "running", "Read"), "reading: Reading SKILL.md"],
    [tool("Read paper.pdf", "read"), "reading: Reading paper.pdf"],
    [tool("Read Bertrand and Mullainathan 2004 · pp. 6-9", "read"), "reading: Reading"],
    [tool("Edit /Users/you/notes/summary.md", "edit"), "working: Editing summary.md"],
    [tool("python3 -c 'print(1)'", "execute"), "working: Working"],
    [tool("Terminal", "other"), "working: Working"],
  ];
  for (const [b, want] of cases) assert.equal(said([b]), want, (b as { title: string }).title);
  // parallel steps: the most recent running one
  assert.equal(said([tool("zotero-cli search a", "execute"), tool("zotero-cli read K --start-page 2", "execute"), tool("Read x.md", "read", "done")]), "reading: Reading page 2");
});

/** A fake clock: `tick(ms)` runs what came due. */
function fakeClock() {
  let now = 0;
  let jobs: { at: number; fn: () => void; id: number }[] = [];
  let ids = 0;
  return {
    now: () => now,
    set: (fn: () => void, ms: number) => { const id = ++ids; jobs.push({ at: now + ms, fn, id }); return id; },
    clear: (id: unknown) => { jobs = jobs.filter((j) => j.id !== id); },
    tick(ms: number) {
      const end = now + ms;
      for (;;) {
        const due = jobs.filter((j) => j.at <= end).sort((a, b) => a.at - b.at)[0];
        if (!due) break;
        now = due.at; jobs = jobs.filter((j) => j !== due); due.fn();
      }
      now = end;
    },
    get pending() { return jobs.length; },
  };
}

const T = (state: Think["state"], label: string = state): Think => ({ state, label });

test("the label changes at most every 400 ms and only ever to the newest state", () => {
  const clock = fakeClock();
  const shown: string[] = [];
  const p = new Pacer((t) => shown.push(t.label), clock);
  p.push(T("thinking"));
  assert.deepEqual(shown, ["thinking"], "the first state shows at once");
  clock.tick(100); p.push(T("searching"));
  clock.tick(100); p.push(T("reading"));
  clock.tick(100); p.push(T("writing"));
  assert.deepEqual(shown, ["thinking"], "nothing new inside the gap");
  clock.tick(99);
  assert.deepEqual(shown, ["thinking"]);
  clock.tick(1);
  assert.deepEqual(shown, ["thinking", "writing"], "at 400 ms the newest; searching and reading were never shown");
  clock.tick(1000); p.push(T("waiting"));
  assert.deepEqual(shown.at(-1), "waiting", "after a quiet spell a change shows at once");
  clock.tick(50); p.push(T("working")); clock.tick(50); p.push(T("waiting"));
  clock.tick(1000);
  assert.deepEqual(shown, ["thinking", "writing", "waiting"], "back to what is shown before the gap ended: no change at all");
  p.push(T("reading", "Reading page 7")); clock.tick(10); p.push(T("reading", "Reading page 8"));
  clock.tick(400);
  assert.deepEqual(shown.slice(-2), ["Reading page 7", "Reading page 8"], "a new label in the same state is a change too");
  p.push(T("thinking")); clock.tick(10); p.push(T("writing"));
  assert.equal(clock.pending, 1, "one timer, however many pushes");
  p.drop();
  assert.equal(clock.pending, 0, "drop cancels it (the line went away)");
  p.push(T("thinking")); p.jump(T("working"));
  assert.equal(shown.at(-1), "working", "jump shows at once (the turn started), inside the gap");
  assert.equal(clock.pending, 0);
  clock.tick(1000);
  assert.equal(shown.at(-1), "working", "nothing that was waiting shows later");
});

// ───────────────────────────── the stylesheet stays cheap ─────────────────────────────

/** @keyframes name -> the properties its frames set. */
function keyframes(css: string): Map<string, Set<string>> {
  const out = new Map<string, Set<string>>();
  const re = /@keyframes\s+([\w-]+)\s*\{/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(css))) {
    let depth = 1, i = re.lastIndex;
    while (depth && i < css.length) { if (css[i] === "{") depth++; else if (css[i] === "}") depth--; i++; }
    const body = css.slice(re.lastIndex, i - 1);
    out.set(m[1]!, new Set([...body.matchAll(/([a-z-]+)\s*:/g)].map((x) => x[1]!)));
  }
  return out;
}

test("the indicator animates only opacity and transform", () => {
  const all = keyframes(STYLES);
  const used = new Set([...THINK_STYLES.matchAll(/animation:\s*([\w-]+)/g)].map((x) => x[1]!).filter((n) => n !== "none"));
  assert.deepEqual([...used].sort(), ["zmc-dot", "zmc-glint", "zmc-pulse"]);
  for (const name of used) {
    const props = all.get(name);
    assert.ok(props, `${name} is defined`);
    for (const p of props!) assert.ok(p === "opacity" || p === "transform", `@keyframes ${name} animates ${p}`);
  }
  const css = THINK_STYLES.replace(/\/\*[\s\S]*?\*\//g, "");
  // the glint runs only on the running line's label (not while waiting) or the streaming thought's
  const glints = [...css.matchAll(/([^{}]+)\{[^}]*animation:\s*zmc-glint/g)].map((x) => x[1]!.trim());
  assert.deepEqual(glints, ['.zmc .working:not([data-s="waiting"]) .glint > span, .zmc .thought--active .glint > span']);
  // reduced motion: every animation off, the dots still visible
  const reduced = css.slice(css.indexOf("prefers-reduced-motion"));
  assert.match(reduced, /\.dm i \{ animation: none !important; opacity: 0\.6 !important/);
  assert.match(reduced, /\.glint > span \{ animation: none !important/);
  assert.ok(STYLES.includes(THINK_STYLES), "styles.ts includes it");
  assert.ok(!/\.pulse\b|thought__dot/.test(STYLES), "the old pulse dot is gone");
});
