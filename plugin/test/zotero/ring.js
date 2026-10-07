// The context ring in real Gecko (run with --mock-agent): after a turn that reports a fill, hovering the ring shows our
// tooltip and a click opens the popover, both inside the composer's shadow DOM and inside the window, never clipped.
// Summarise now runs the real session's /compact against the mock bridge. Snapshots in light and dark (the window, and a 2x close-up of the composer).
async function main(ctx) {
  const { win } = ctx;
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };
  await ctx.resize(1280, 800);
  ctx.plugin.windows.get(win).show();
  const root = await ctx.waitFor(() => win.document.getElementById("zmc-root")?.shadowRoot?.querySelector("textarea"), "panel rendered").then(() => win.document.getElementById("zmc-root").shadowRoot);
  const $ = (s) => root.querySelector(s);
  const ta = $("textarea");
  ta.value = "hello"; ta.dispatchEvent(new win.Event("input", { bubbles: true }));
  await ctx.waitFor(() => !$("button.send").disabled, "send enabled");
  $("button.send").click();
  const ring = await ctx.waitFor(() => { const r = $(".cmeter"); return r && !r.hidden && $('.msg--assistant[data-state="end_turn"]') ? r : null; }, "the ring after a turn", 30000);
  check(!ring.title && /^Context: \d+% full \(.+ of 200k tokens\)\. Show details$/.test(ring.getAttribute("aria-label")), "aria-label, no native title: " + ring.getAttribute("aria-label"));

  const rect = (el) => el.getBoundingClientRect();
  const panel = rect(win.document.getElementById("zmc-root"));
  const within = (r, c, what) => check(r.left >= c.left - 1 && r.right <= c.right + 1 && r.top >= c.top - 1 && r.bottom <= c.bottom + 1, `${what} inside: ${JSON.stringify([r.left, r.top, r.right, r.bottom])} vs ${JSON.stringify([c.left, c.top, c.right, c.bottom])}`);
  // A window in the background does not tick CSS animations: settle the fade-in before a snapshot (and say how far it got).
  const settle = (el) => el.getAnimations().map((a) => { const at = `${a.playState}@${Math.round(a.currentTime ?? -1)}ms`; a.finish(); return at; }).join(",");
  const closeup = async (name) => {
    const c = rect($(".composer"));
    const r = new win.DOMRect(c.left - 16, c.top - 300, c.width + 32, c.height + 316);
    const bitmap = await win.browsingContext.currentWindowGlobal.drawSnapshot(r, 2, "rgb(255,255,255)");
    const canvas = win.document.createElementNS("http://www.w3.org/1999/xhtml", "canvas");
    canvas.width = bitmap.width; canvas.height = bitmap.height;
    canvas.getContext("2d").drawImage(bitmap, 0, 0);
    const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
    await IOUtils.write(PathUtils.join(ctx.env("ZMC_SNAPSHOT_DIR"), `${name}.png`), new Uint8Array(await blob.arrayBuffer()));
  };

  for (const [t, name] of [[1, "light"], [0, "dark"]]) {
    Services.prefs.setIntPref("browser.theme.toolbar-theme", t);
    await ctx.waitFor(() => $(".zmc").dataset.theme === name, "theme " + name, 5000).catch(() => { $(".zmc").dataset.theme = name; });
    await ctx.sleep(300);
    // Hover: the tooltip at once, above the ring, in the composer
    ring.dispatchEvent(new win.PointerEvent("pointerenter"));
    const tip = $(".ctip");
    check(!tip.hidden && /^\d+% of context used$/.test(tip.firstChild.textContent) && /^.+ of 200k tokens$/.test(tip.lastChild.textContent), "the tooltip: " + tip.textContent);
    const tr = rect(tip), cr = rect($(".composer"));
    check(tr.left >= cr.left && tr.right <= cr.right, "tooltip within the composer's width");
    within(tr, panel, "tooltip");
    check(rect(tip).bottom <= rect(ring).top, "tooltip above the ring");
    out[`tipAnim-${name}`] = settle(tip);
    await closeup(`ring-tip-${name}`);
    ring.dispatchEvent(new win.PointerEvent("pointerleave"));
    check(tip.hidden, "leaving hides the tooltip");
    // Click: the popover, glass on, inside the panel and the window
    // The agent advertises /compact a moment after the session starts; open the popover until that has arrived (a slow machine is not a failure).
    let pop;
    for (let i = 0; i < 20; i++) {
      ring.click();
      pop = await ctx.waitFor(() => $(".cpop"), "the popover");
      if ($('.cpop [data-act="compact"]')) break;
      ring.click();
      await ctx.waitFor(() => !$(".cpop"), "the popover closes");
      await ctx.sleep(300);
    }
    check(ring.getAttribute("aria-expanded") === "true" && $(".cpop__title").textContent === "Context window", "popover open");
    check(/^Messages2Last turn/.test($(".cpop__facts").textContent), "facts: " + $(".cpop__facts").textContent);
    check($('.cpop [data-act="new"]') && $('.cpop [data-act="compact"]'), "New chat, and Summarise now (the mock advertises /compact as claude-agent-acp does)");
    const pr = rect(pop);
    within(pr, panel, "popover");
    check(pr.left >= rect($(".composer")).left - 1 && pr.bottom <= rect(ring).top, "popover above the ring, in the composer's width");
    check([...pop.querySelectorAll("*")].every((e) => e.scrollWidth <= e.clientWidth + 1 || win.getComputedStyle(e).overflowX === "visible"), "nothing in the popover scrolls sideways");
    out[name] = { glass: $(".zmc").dataset.glass, backdrop: win.getComputedStyle(pop).backdropFilter, pop: [Math.round(pr.width), Math.round(pr.height)] };
    out[`popAnim-${name}`] = settle(pop);
    check(win.getComputedStyle(pop).opacity === "1", "the popover is fully opaque");
    await ctx.snapshot(`ring-pop-${name}`);
    await closeup(`ring-pop-${name}-closeup`);
    pop.dispatchEvent(new win.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    check(!$(".cpop") && ring.getAttribute("aria-expanded") === "false", "Esc closes");
  }
  // Summarise now: the real session sends /compact; one notice, no new answer, the ring keeps the new fill
  ring.click();
  await ctx.waitFor(() => $(".cpop"), "the popover");
  $('.cpop [data-act="compact"]').click();
  await ctx.waitFor(() => /summarised to make room/.test($(".notice")?.textContent ?? ""), "the compaction notice", 15000);
  check(root.querySelectorAll(".msg--assistant").length === 1 && !$(".cmeter").hidden, "no answer for it; the ring stays");
  // New chat from the popover: the chat resets and the ring hides
  ring.click();
  await ctx.waitFor(() => $(".cpop"), "the popover again");
  $('.cpop [data-act="new"]').click();
  await ctx.waitFor(() => !$(".msg--assistant") && $(".cmeter").hidden && !$(".cpop"), "a new chat, no ring");
  return out;
}
