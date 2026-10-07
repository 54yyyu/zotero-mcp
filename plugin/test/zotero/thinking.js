// The working line in real Gecko (run with --mock-agent). SCENARIO:states walks thinking, searching, reading, working,
// waiting (a permission card) and writing through the real runtime: each state's data-s, label and computed animations,
// one indicator at any moment (the streaming thought's row carries it, else the working line; polled every 50 ms), and
// nothing animating once the turn is over. Close-ups per state, light and dark: the harness window is in the
// background, where Gecko may not tick CSS animations, so each frame is taken with the line's animations paused at the
// same time (1 s in); the motion itself needs a human look. Then SCENARIO:hold keeps the line running while we measure
// what it costs (rAF frame deltas; the main process's CPU time with no script on the page), against an idle panel and
// against a background-position shimmer (what the glint replaced).
async function main(ctx) {
  const { win } = ctx;
  const out = { states: {} };
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };
  await ctx.resize(1280, 800);
  ctx.plugin.windows.get(win).show();
  const root = await ctx.waitFor(() => win.document.getElementById("zmc-root")?.shadowRoot?.querySelector("textarea"), "panel rendered").then(() => win.document.getElementById("zmc-root").shadowRoot);
  const $ = (s) => root.querySelector(s);
  const running = () => $("button.send")?.classList.contains("send--stop");
  async function say(msg) {
    const ta = $("textarea");
    ta.value = msg; ta.dispatchEvent(new win.Event("input", { bubbles: true }));
    await ctx.waitFor(() => !$("button.send").disabled, "send enabled");
    $("button.send").click();
    await ctx.waitFor(() => running(), "turn started");
  }
  const vis = (e) => !!e && e.checkVisibility();
  /** The visible indicator: the working line, or the streaming thought's row. */
  const line = () => { const w = $(".working"), row = $(".thought--active"); return vis(w) ? w : vis(row) ? row : null; };
  const css = (el, p) => win.getComputedStyle(el)[p];
  const describe = () => {
    const w = line();
    if (!w) return null;
    const g = w.querySelector(".glint");
    return {
      where: w.matches(".working") ? "line" : "thought", s: w.dataset.s ?? "thinking", label: g.textContent, h: Math.round(w.getBoundingClientRect().height),
      dot: css(w.querySelector(".dm i"), "animationName"), glint: css(g.firstElementChild, "animationName"),
    };
  };
  const COUNT = () => {
    const shown = [...root.querySelectorAll(".dm, .pulse, .thought__dot")].filter(vis).length;
    const lines = [...root.querySelectorAll(".working")].filter(vis).length;
    const thinking = [...root.querySelectorAll(".glint")].filter((e) => vis(e) && e.textContent === "Thinking").length;
    return Math.max(shown, lines, thinking);
  };
  const closeup = async (name) => {
    const w = line();
    const anims = w.getAnimations({ subtree: true });
    for (const a of anims) { a.pause(); a.currentTime = 1000; }
    const r = w.getBoundingClientRect();
    const rect = new win.DOMRect(r.left - 12, r.top - 8, 300, r.height + 16);
    const bitmap = await win.browsingContext.currentWindowGlobal.drawSnapshot(rect, 2, "rgb(255,255,255)");
    const canvas = win.document.createElementNS("http://www.w3.org/1999/xhtml", "canvas");
    canvas.width = bitmap.width; canvas.height = bitmap.height;
    canvas.getContext("2d").drawImage(bitmap, 0, 0);
    const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
    await IOUtils.write(PathUtils.join(ctx.env("ZMC_SNAPSHOT_DIR"), `${name}.png`), new Uint8Array(await blob.arrayBuffer()));
    for (const a of anims) a.play();
    return anims.length;
  };
  const theme = async (name) => {
    Services.prefs.setIntPref("browser.theme.toolbar-theme", name === "light" ? 1 : 0);
    await ctx.waitFor(() => $(".zmc").dataset.theme === name, "theme " + name, 5000).catch(() => { $(".zmc").dataset.theme = name; });
  };

  // 1. every state from real events
  await theme("light");
  const polls = [];
  const poll = win.setInterval(() => polls.push(COUNT()), 50);
  await say("SCENARIO:states");
  const want = [
    ["thinking", "Thinking"], ["searching", "Searching your library"], ["reading", "Reading page 7"], ["working", "Working"],
    ["waiting", "Waiting for your OK"], ["writing", "Writing the answer"],
  ];
  for (const [s, label] of want) {
    const d = await ctx.waitFor(() => { const x = describe(); return x && x.s === s && (s !== "thinking" || x.where === "thought") ? x : null; }, `state ${s}`, 8000);
    out.states[s] = d;
    check(d.label === label, `${s}: label ${d.label}`);
    check(d.where === (s === "thinking" ? "thought" : "line") && (d.where === "thought" || d.h === 28), `${s}: where, 28 px: ${JSON.stringify(d)}`);
    check(COUNT() === 1, `${s}: one indicator`);
    check(d.dot === (s === "waiting" ? "zmc-pulse" : "zmc-dot"), `${s}: dot animation ${d.dot}`);
    check(d.glint === (s === "waiting" ? "none" : "zmc-glint"), `${s}: glint ${d.glint}`);
    out.states[s].anims = await closeup(`thinking-${s}-light`);
    if (s === "waiting") {
      await ctx.snapshot("thinking-waiting-window");
      const allow = [...root.querySelectorAll(".perm button")].find((b) => /^Allow/.test(b.textContent));
      check(allow, "the permission card's Allow");
      allow.click();
    }
  }
  await ctx.waitFor(() => !running(), "turn finished", 20000);
  await ctx.sleep(200);
  win.clearInterval(poll);
  out.polls = { n: polls.length, max: Math.max(...polls), last: polls.at(-1) };
  check(polls.length > 100 && out.polls.max === 1 && out.polls.last === 0, "never more than one indicator, none after: " + JSON.stringify(out.polls));
  check(!line(), "no indicator after the turn");
  const endless = root.getAnimations().filter((a) => a.playState === "running" && a.effect.getComputedTiming().iterations === Infinity);
  out.endlessAfter = endless.map((a) => `${a.animationName} on ${a.effect.target.className}`);
  check(endless.length === 0, "nothing animating after the turn: " + out.endlessAfter.join(", "));
  await ctx.snapshot("thinking-after");

  // 2. dark close-ups and the cost, on a turn that keeps thinking until Stop
  const CU = win.ChromeUtils ?? ChromeUtils;
  const proc = async () => { const p = await CU.requestProcInfo(); return { cpu: p.cpuTime, threads: new Map(p.threads.map((t) => [`${t.name || "(main)"}#${t.tid}`, t.cpuTime])) }; };
  /** CPU ms per second of the whole process and its busiest threads, over `ms` with no script running. */
  const cpuFor = async (ms) => {
    const a = await proc(); await ctx.sleep(ms); const b = await proc();
    const per = (ns) => +(ns / 1e6 / (ms / 1000)).toFixed(2);
    const threads = [...b.threads].map(([k, v]) => [k.replace(/#\d+$/, ""), per(v - (a.threads.get(k) ?? v))]).filter(([, v]) => v >= 0.05).sort((x, y) => y[1] - x[1]).slice(0, 5);
    return { processMsPerS: per(b.cpu - a.cpu), threads };
  };
  const frames = (ms) => new Promise((res) => {
    const d = []; let last = win.performance.now(); const t0 = last;
    const f = (now) => { d.push(now - last); last = now; if (now - t0 < ms) win.requestAnimationFrame(f); else { d.sort((x, y) => x - y); res({ frames: d.length, median: +d[d.length >> 1].toFixed(1), p95: +d[Math.floor(d.length * 0.95)].toFixed(1), max: +d[d.length - 1].toFixed(1) }); } };
    win.requestAnimationFrame(f);
  });
  const measure = async () => ({ cpu: await cpuFor(3000), raf: await frames(3000) });
  out.idle = await measure();
  await say("SCENARIO:hold");
  await ctx.waitFor(() => describe()?.where === "thought", "holding: the thought row thinks");
  const dot = line().querySelector(".dm i");
  const t0 = dot.getAnimations()[0]?.currentTime;
  await ctx.sleep(500);
  const t1 = dot.getAnimations()[0]?.currentTime;
  out.animationsTick = { advancedMs: Math.round((t1 ?? 0) - (t0 ?? 0)), focused: win.document.hasFocus() };
  out.running = await measure();
  // a background-position shimmer on the label (what the glint replaced), for comparison only
  const old = win.document.createElementNS("http://www.w3.org/1999/xhtml", "style");
  old.textContent = `.zmc .glint > span { animation: none !important; opacity: 1 !important; color: inherit !important; }
.zmc .glint { background: linear-gradient(100deg, var(--ink-muted) 0 38%, var(--ink) 50%, var(--ink-muted) 62% 100%) 130% 0 / 260% 100%; -webkit-background-clip: text; background-clip: text; color: transparent; animation: zmc-old-shimmer 2.1s linear infinite; }
@keyframes zmc-old-shimmer { from { background-position: 130% 0; } to { background-position: -30% 0; } }`;
  root.appendChild(old);
  await ctx.sleep(300);
  out.oldShimmer = await measure();
  old.remove();
  await theme("dark");
  await ctx.sleep(300);
  await closeup("thinking-thinking-dark");
  await theme("light");
  $("button.send").click(); // Stop
  await ctx.waitFor(() => !running(), "stopped", 10000);
  await ctx.sleep(200);
  check(!line(), "no indicator after Stop");
  ctx.log(JSON.stringify({ idle: out.idle, running: out.running, oldShimmer: out.oldShimmer, ticks: out.animationsTick }));
  return out;
}
