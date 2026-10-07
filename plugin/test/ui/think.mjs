// The working line in a real browser: node test/ui/think.mjs [--browser webkit] (build first: node scripts/preview.mjs).
// A scripted turn drives each state through the fake host's session; checks the state attribute, the label, the pacing,
// that nothing shifts when the label changes, reduced motion, the accent, 300 px, light/dark/glass, and that every
// animation is gone when the turn ends. Then measures what the running line costs per frame against an idle panel.
// Close-ups go to preview/shots/think-*.png (look at them).
import assert from "node:assert/strict";
import { launch, openPanel, shotsDir, sleep } from "./lib.mjs";

const browserName = process.argv.includes("--browser") ? process.argv[process.argv.indexOf("--browser") + 1] : "chromium";
const browser = await launch(browserName);
let n = 0;
const test = async (name, fn, opts = {}) => {
  const params = Object.fromEntries(Object.entries({ nohistory: 1, ...opts.params }).filter(([, v]) => v !== null)); // null drops a parameter
  const page = await openPanel(browser, { ...opts, params });
  try { await fn(page); console.log("PASS", name); n++; }
  catch (e) { console.error("FAIL", name, "\n", e); await page.screenshot({ path: `${shotsDir}/FAIL-think-${name.replace(/\W+/g, "-")}.png` }); process.exitCode = 1; }
  assert.deepEqual(page.errors, [], `page errors in ${name}`);
  await page.context().close();
};

/** Take over the fake agent: a send starts a turn that only moves when the test emits events. */
async function scripted(p) {
  await p.evaluate(() => {
    const rt = window.__zmc.host.runtime;
    const start = rt.start.bind(rt);
    rt.start = async (o) => {
      const s = await start(o);
      s.prompt = () => new Promise((resolve) => { window.__turn = { s, resolve }; s.emit({ t: "turn_start", turn: "T1" }); });
      return s;
    };
    window.__emit = (ev) => window.__turn.s.emit({ turn: "T1", ...ev });
    window.__end = () => { window.__emit({ t: "turn_end", stop: "end_turn" }); window.__turn.resolve(); };
  });
  await p.locator(".cin").fill("What do audit studies find?");
  await p.locator(".cin").press("Enter");
  await p.waitForFunction(() => !!window.__turn && !!window.__zmc.shadow.querySelector(".msg--assistant .working"));
}
const emit = (p, ev) => p.evaluate((e) => window.__emit(e), ev);
/** The visible indicator, wherever it is (the working line, or the streaming thought's row), or null. */
const line = (p) => p.evaluate(() => {
  const S = window.__zmc.shadow;
  const vis = (e) => e && e.checkVisibility();
  const row = S.querySelector(".thought--active");
  const w = S.querySelector(".working");
  const el = vis(w) ? w : vis(row) ? row : null;
  if (!el) return null;
  const t = el.querySelector(".glint"), dot = el.querySelector(".dm i"), r = el.getBoundingClientRect();
  return { where: el === w ? "line" : "thought", s: el === w ? w.dataset.s : "thinking", label: t.textContent, dots: el.querySelectorAll(".dm i").length, h: Math.round(r.height), top: Math.round(r.top),
    dotAnim: getComputedStyle(dot).animationName, labelAnim: getComputedStyle(t.firstElementChild).animationName, live: w.getAttribute("role") };
});
/** Indicators on screen: dot matrices, plus anything left of the old pulse dot or a second working line. */
const COUNT = () => {
  const S = window.__zmc.shadow;
  const vis = [...S.querySelectorAll(".dm, .pulse, .thought__dot, .working")].filter((e) => e.checkVisibility());
  const dms = vis.filter((e) => e.matches(".dm")).length, other = vis.filter((e) => e.matches(".pulse, .thought__dot")).length;
  const lines = vis.filter((e) => e.matches(".working")).length;
  const thinking = [...S.querySelectorAll(".glint")].filter((e) => e.checkVisibility() && e.textContent === "Thinking").length;
  const stale = [...S.querySelectorAll(".msg--assistant .working")].filter((e) => e.checkVisibility() && /^(Starting|Sending)/.test(e.textContent)).length;
  return { n: dms + other, lines, thinking, stale };
};
/** Past the 400 ms pacing. */
const settle = () => sleep(460);
const STEPS = [
  [{ t: "thought", delta: "The user asks about audit studies." }, "thinking", "Thinking"],
  [{ t: "tool", id: "a", title: 'cd ~/Documents && zotero-cli --json search "hiring audit" --limit 10', kind: "execute", status: "running", name: "Bash" }, "searching", "Searching your library"],
  [{ t: "tool", id: "b", title: "zotero-cli --json read BM2004AB --start-page 7", kind: "execute", status: "running", name: "Bash" }, "reading", "Reading page 7"],
  [{ t: "tool", id: "c", title: "python3 tally.py", kind: "execute", status: "running", name: "Bash" }, "working", "Working"],
  [{ t: "permission", id: "p", title: "Create a note", kind: "edit", options: [{ id: "ok", name: "Allow once", kind: "allow_once" }] }, "waiting", "Waiting for your OK"],
  [{ t: "text", delta: "Callbacks for White-sounding names were 50% higher." }, "writing", "Writing the answer"],
];

async function closeup(p, name) {
  const box = await p.evaluate(() => { const S = window.__zmc.shadow; const r = [...S.querySelectorAll(".thought--active .thought__head, .working")].find((e) => e.checkVisibility()).getBoundingClientRect(); return { x: 0, y: Math.max(0, r.top - 10), width: Math.min(innerWidth, 300), height: r.height + 20 }; });
  await p.screenshot({ path: `${shotsDir}/think-${name}.png`, clip: box });
}

await test("each state from a scripted turn: data-s, label, dots, one line, pacing", async (p) => {
  await scripted(p);
  await settle();
  let l = await line(p);
  assert.deepEqual([l.where, l.s, l.label, l.dots, l.live], ["line", "thinking", "Thinking", 9, "status"], "no output yet: the line thinks");
  assert.equal(l.dotAnim, "zmc-dot"); assert.equal(l.labelAnim, "zmc-glint");
  for (const [ev, s, label] of STEPS) {
    await emit(p, ev);
    await settle();
    l = await line(p);
    assert.deepEqual([l.s, l.label], [s, label], `after ${ev.t} ${ev.title ?? ""}`);
    assert.equal(l.where, ev.t === "thought" ? "thought" : "line", "a streaming thought carries the indicator in its row");
    assert.deepEqual(await p.evaluate(COUNT), { n: 1, lines: ev.t === "thought" ? 0 : 1, thinking: s === "thinking" ? 1 : 0, stale: 0 }, "one indicator");
    if (s === "waiting") {
      assert.equal(l.dotAnim, "zmc-pulse", "waiting: all dots blink together");
      assert.equal(l.labelAnim, "none", "waiting: no glint");
      await emit(p, { ...ev, resolved: "ok" });
    } else assert.equal(l.labelAnim, "zmc-glint");
    if (s === "searching") assert.equal(await p.evaluate(() => getComputedStyle(window.__zmc.shadow.querySelector(".working .dm i:nth-child(5)")).animationName), "none", "the ring's centre stays dim");
  }
  // a burst inside the gap shows only its newest state, at most one change per 400 ms
  const seen = [];
  await p.evaluate(() => {
    window.__seen = [];
    new MutationObserver(() => window.__seen.push([performance.now(), window.__zmc.shadow.querySelector(".msg--assistant .working").dataset.s])).observe(window.__zmc.shadow.querySelector(".msg--assistant .working"), { attributes: true, attributeFilter: ["data-s"] });
  });
  await emit(p, { t: "tool", id: "d", title: "zotero-cli search more", kind: "execute", status: "running" });
  await sleep(60);
  await emit(p, { t: "tool", id: "d", title: "zotero-cli search more", kind: "execute", status: "done" });
  await emit(p, { t: "tool", id: "e", title: "zotero-cli read K --start-page 3", kind: "execute", status: "running" });
  await sleep(60);
  await emit(p, { t: "tool", id: "e", title: "zotero-cli read K --start-page 3", kind: "execute", status: "done" });
  await emit(p, { t: "text", delta: " More text." });
  await sleep(900);
  seen.push(...await p.evaluate(() => window.__seen));
  assert.ok(seen.length <= 2, `at most two changes in ~1 s: ${JSON.stringify(seen)}`);
  for (let i = 1; i < seen.length; i++) assert.ok(seen[i][0] - seen[i - 1][0] >= 380, "changes are 400 ms apart");
  assert.equal((await line(p)).s, "writing", "ends on the newest state");
  for (const id of ["a", "b", "c"]) await emit(p, { t: "tool", id, title: "x", kind: "execute", status: "done" });
  await p.evaluate(() => window.__end());
  await p.waitForSelector('.msg--assistant[data-state="end_turn"]');
  assert.equal(await line(p), null, "the line is gone when the turn ends");
  const running = await p.evaluate(() => window.__zmc.shadow.getAnimations().filter((a) => a.playState === "running" && a.effect?.getComputedTiming().iterations === Infinity).map((a) => `${a.animationName} on ${a.effect.target.className}`));
  assert.deepEqual(running, [], "no endless animation left anywhere in the panel");
});

await test("the pending line (Starting / Sending) has the dots too", async (p) => {
  await p.evaluate(() => { window.__zmc.sim.speed = 400; });
  await p.locator(".cin").fill("hi");
  await p.locator(".cin").press("Enter");
  const pend = await p.waitForFunction(() => { const el = window.__zmc.shadow.querySelector(".working--pending:not([hidden])"); return el && { s: el.dataset.s, t: el.textContent, dots: el.querySelectorAll(".dm i").length }; });
  const v = await pend.jsonValue();
  assert.equal(v.s, "thinking"); assert.equal(v.dots, 9); assert.match(v.t, /^(Starting Claude Code|Sending)$/);
  await closeup(p, "pending");
});

await test("no layout shift when only the label changes", async (p) => {
  await scripted(p);
  await settle();
  await emit(p, { t: "tool", id: "a", title: "zotero-cli search a", kind: "execute", status: "running" }); // shown at once (a quiet spell before)
  await sleep(60);
  await emit(p, { t: "tool", id: "a", title: "zotero-cli search a", kind: "execute", status: "done" });
  await emit(p, { t: "tool", id: "b", title: "zotero-cli read K --start-page 2", kind: "execute", status: "running" }); // waits out the gap
  await sleep(120); // the rows are drawn; the label is still the old one
  const before = await p.evaluate(() => { const s = window.__zmc.shadow; const w = s.querySelector(".msg--assistant .working"); return { s: w.dataset.s, top: w.getBoundingClientRect().top, h: w.getBoundingClientRect().height, scroll: s.querySelector(".feed").scrollHeight, comp: s.querySelector(".composer").getBoundingClientRect().top }; });
  await settle();
  const after = await p.evaluate(() => { const s = window.__zmc.shadow; const w = s.querySelector(".msg--assistant .working"); return { s: w.dataset.s, top: w.getBoundingClientRect().top, h: w.getBoundingClientRect().height, scroll: s.querySelector(".feed").scrollHeight, comp: s.querySelector(".composer").getBoundingClientRect().top }; });
  assert.notEqual(before.s, after.s, "the label did change");
  assert.deepEqual({ ...before, s: 0 }, { ...after, s: 0 }, "same place, height, feed height and composer");
  assert.equal(after.h, 28);
  // a label longer than the panel stays one line
  await emit(p, { t: "tool", id: "c", title: "Read /Users/you/" + "very-long-folder-name/".repeat(4) + "a-file-with-a-rather-long-name-indeed.md", kind: "read", status: "running" });
  await settle();
  const l = await line(p);
  assert.equal(l.h, 28, "still one line");
}, { width: 300 });

await test("reduced motion: static dots at 60%, a plain muted label", async (p) => {
  await p.emulateMedia({ reducedMotion: "reduce" });
  await scripted(p);
  await emit(p, STEPS[1][0]);
  await settle();
  const st = await p.evaluate(() => {
    const s = window.__zmc.shadow, dot = s.querySelector(".working .dm i"), t = s.querySelector(".working .working__t");
    const letter = t.querySelector("span");
    return { anim: getComputedStyle(dot).animationName, op: getComputedStyle(dot).opacity, glint: getComputedStyle(letter).animationName, color: getComputedStyle(letter).color, muted: getComputedStyle(s.querySelector(".working")).color };
  });
  assert.deepEqual(st, { anim: "none", op: "0.6", glint: "none", color: st.muted, muted: st.muted });
  await closeup(p, "reduced");
});

await test("the dots take a custom accent", async (p) => {
  await scripted(p);
  const bg = await p.evaluate(() => getComputedStyle(window.__zmc.shadow.querySelector(".working .dm i")).backgroundColor);
  assert.equal(bg, "rgb(204, 41, 54)");
  await emit(p, STEPS[3][0]);
  await settle();
  await closeup(p, "accent-red");
}, { params: { look: JSON.stringify({ accent: "#cc2936" }) } });

await test("never more than one indicator: polled every 50 ms through a whole turn", async (p) => {
  await p.evaluate((COUNT) => {
    const count = new Function(`return (${COUNT})()`);
    window.__polls = [];
    window.__poll = setInterval(() => window.__polls.push(count()), 50);
  }, COUNT.toString());
  await p.evaluate(() => { window.__zmc.sim.speed = 300; }); // a slow start: "Starting Claude Code" / "Sending" are seen
  await scripted(p);
  await p.evaluate(() => { window.__zmc.sim.speed = 0; });
  const think = "Weighing which audit studies matter here. ".split(" ");
  for (const w of think) { await emit(p, { t: "thought", delta: w + " " }); await sleep(40); }
  await sleep(500);
  await emit(p, { t: "tool", id: "a", title: "zotero-cli search audit", kind: "execute", status: "running" });
  await sleep(300);
  await emit(p, { t: "tool", id: "a", title: "zotero-cli search audit", kind: "execute", status: "done" });
  for (const w of "Let me check one more thing. ".split(" ")) { await emit(p, { t: "text", delta: w + " " }); await sleep(40); }
  for (const w of "Now the table.".split(" ")) { await emit(p, { t: "thought", delta: w + " " }); await sleep(40); }
  await emit(p, { t: "permission", id: "p", title: "Create a note", kind: "edit", options: [{ id: "ok", name: "Allow once", kind: "allow_once" }] });
  await sleep(600);
  await emit(p, { t: "permission", id: "p", title: "Create a note", kind: "edit", options: [{ id: "ok", name: "Allow once", kind: "allow_once" }], resolved: "ok" });
  for (const w of "Callbacks were 50% higher for White-sounding names.".split(" ")) { await emit(p, { t: "text", delta: w + " " }); await sleep(40); }
  await sleep(500);
  await p.evaluate(() => window.__end());
  await p.waitForSelector('.msg--assistant[data-state="end_turn"]');
  await sleep(200);
  const polls = await p.evaluate(() => { clearInterval(window.__poll); return window.__polls; });
  assert.ok(polls.length > 60, `polled ${polls.length} times`);
  const worst = polls.reduce((m, x) => Object.fromEntries(Object.keys(m).map((k) => [k, Math.max(m[k], x[k])])), { n: 0, lines: 0, thinking: 0, stale: 0 });
  assert.deepEqual(worst, { n: 1, lines: 1, thinking: 1, stale: 0 }, "at most one indicator, one line, one Thinking at any moment; no Starting… under a running answer");
  assert.ok(polls.some((x) => x.n === 1 && x.lines === 1 && x.thinking === 0), "the pending line was seen");
  assert.ok(polls.filter((x) => x.n === 1).length > polls.length * 0.8, "and there is one for most of the turn");
  assert.deepEqual(polls.at(-1), { n: 0, lines: 0, thinking: 0, stale: 0 }, "none once the turn is over");
});

await test("history: a resumed chat shows no indicator", async (p) => {
  await p.locator('button[aria-label="History"]').click();
  await p.waitForSelector(".hrow__main");
  await p.locator(".hrow__main").first().click();
  await p.waitForSelector(".msg--assistant");
  await sleep(300);
  assert.deepEqual(await p.evaluate(COUNT), { n: 0, lines: 0, thinking: 0, stale: 0 });
}, { params: { nohistory: null } });

for (const [dark, glass] of [[false, true], [true, true], [false, false], [true, false]]) {
  const tag = `${dark ? "dark" : "light"}-${glass ? "glass" : "plain"}`;
  await test(`looks: ${tag}, 300 px`, async (p) => {
    await scripted(p);
    for (const [ev, s] of STEPS) {
      await emit(p, ev);
      await settle();
      if (s === "waiting") { await closeup(p, `${tag}-${s}`); await emit(p, { ...ev, resolved: "ok" }); continue; }
      await closeup(p, `${tag}-${s}`);
    }
    // the label is laid out inside the narrow panel
    const ok = await p.evaluate(() => { const r = window.__zmc.shadow.querySelector(".working .working__t").getBoundingClientRect(); return r.width > 40 && r.right <= innerWidth; });
    assert.ok(ok, "label laid out inside the panel");
    await p.evaluate(() => window.__end());
  }, { dark, width: 300, params: { look: JSON.stringify({ glass }) } });
}

// ───────────────────────────── what it costs ─────────────────────────────

await test("cost per frame: running line vs idle (3 s each)", async (p) => {
  // Frame pacing and long tasks, from requestAnimationFrame deltas (the loop itself makes the main thread draw every frame).
  const pacing = (ms) => p.evaluate((ms) => new Promise((res) => {
    const d = []; let last = performance.now(); const t0 = last; let longs = 0;
    let po; try { po = new PerformanceObserver((l) => { longs += l.getEntries().length; }); po.observe({ type: "longtask" }); } catch {}
    const f = (now) => { d.push(now - last); last = now; if (now - t0 < ms) requestAnimationFrame(f); else { po?.disconnect(); d.sort((a, b) => a - b); res({ median: +d[d.length >> 1].toFixed(1), p95: +d[Math.floor(d.length * 0.95)].toFixed(1), max: +d[d.length - 1].toFixed(1), longs }); } };
    requestAnimationFrame(f);
  }), ms);
  // Main-thread work with no script on the page (Chromium's own counters), in ms per 60 Hz frame: what the animation costs.
  const cdp = browserName === "chromium" ? await p.context().newCDPSession(p) : null;
  if (cdp) await cdp.send("Performance.enable");
  const counters = async () => Object.fromEntries((await cdp.send("Performance.getMetrics")).metrics.map((x) => [x.name, x.value]));
  const work = async (ms) => {
    if (!cdp) return null;
    const a = await counters(); await sleep(ms); const b = await counters();
    const per = (k) => +((b[k] - a[k]) * 1000 / (ms / 1000 * 60)).toFixed(3);
    return { task: per("TaskDuration"), style: per("RecalcStyleDuration"), layout: per("LayoutDuration") };
  };
  const both = async () => ({ work: await work(3000), pacing: await pacing(3000) });
  const idle = await both();
  await scripted(p);
  await emit(p, STEPS[0][0]);
  await settle();
  const running = await both();
  console.log(`  ${browserName} idle    `, JSON.stringify(idle));
  console.log(`  ${browserName} running `, JSON.stringify(running));
  assert.ok(running.pacing.median < 17.5 && running.pacing.longs === 0, "frames keep pace, no long task");
  if (cdp) assert.ok(running.work.task - idle.work.task < 0.3, "under 0.3 ms of main-thread work per frame added");
  await p.evaluate(() => window.__end());
});

await browser.close();
console.log(`${n} passed`);
