// Startup budget: with the panel closed, Zotero's own start pays for the container and button only.
async function main(ctx) {
  const { plugin, win } = ctx;
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };
  const injected = plugin.windows.get(win);

  check(injected.loaded() === null, "panel.js is not loaded while the panel is closed");
  check(plugin.timing.panelLoadMs === null, "panel.js was not even read");
  // Everything Zotero waits for from this plugin before it moves on: reading plugin.js plus startup().
  out.loadMs = plugin.timing.loadMs;
  out.startupMs = plugin.timing.startupMs;
  out.zoteroWaitsMs = out.loadMs + out.startupMs;
  check(out.zoteroWaitsMs < 50, `the plugin holds Zotero up for ${out.zoteroWaitsMs} ms at startup (budget 50)`);
  check(!!win.document.getElementById("zmc-toggle"), "the toolbar button exists");

  // The first open pays for loading panel.js once.
  const t0 = Date.now();
  injected.show();
  out.firstOpenMs = Date.now() - t0;
  out.panelLoadMs = plugin.timing.panelLoadMs;
  check(injected.loaded(), "panel loaded on first open");
  check(out.panelLoadMs < 150, `panel.js load took ${out.panelLoadMs} ms (budget 150)`);
  // Later opens are free.
  const t1 = Date.now();
  injected.hide(); injected.show();
  out.reopenMs = Date.now() - t1;
  check(out.reopenMs < 30, `reopen took ${out.reopenMs} ms (budget 30)`);
  return out;
}
