// Zotero's Settings window gets a "Zotero Agent" pane (run with --mock-agent): registered at startup for almost nothing,
// it loads only when opened, shows the panel's own settings cards, starts no agent, and a change made in it reaches an
// open panel at once (and the other way round). Snapshots in light and dark are for a human to look at.
async function main(ctx) {
  const { Zotero, Services, plugin, win } = ctx;
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };
  const injected = plugin.windows.get(win);

  // 1. startup: the pane is one list entry, and nothing of it (or of the panel) has been read
  out.zoteroWaitsMs = plugin.timing.loadMs + plugin.timing.startupMs;
  out.paneRegisterMs = Math.round(plugin.timing.paneRegisterMs * 1000) / 1000;
  check(out.zoteroWaitsMs < 50, `the plugin holds Zotero up for ${out.zoteroWaitsMs} ms at startup (budget 50)`);
  check(plugin.timing.paneRegisterMs < 5, `registering the pane took ${plugin.timing.paneRegisterMs} ms`);
  await ctx.waitFor(() => Zotero.PreferencePanes.pluginPanes.some((p) => p.id === "zotero-chat-pane"), "the pane is registered");
  const entry = Zotero.PreferencePanes.pluginPanes.find((p) => p.id === "zotero-chat-pane");
  out.label = entry.rawLabel;
  check(entry.rawLabel === "Zotero Agent", "labelled Zotero Agent: " + entry.rawLabel);
  check(plugin.timing.paneLoadMs === null && plugin.timing.panelLoadMs === null && injected.loaded() === null, "neither the pane nor the panel is loaded at startup");

  // 2. open the pane before any panel exists: the cards, and no agent started (the Agent card offers a button instead)
  const openPane = async () => {
    const pw = Zotero.Utilities.Internal.openPreferences("zotero-chat-pane");
    await ctx.waitFor(() => pw.document?.readyState === "complete", "the Settings window");
    pw.resizeTo(900, 860);
    const root = await ctx.waitFor(() => pw.document.getElementById("zmc-prefs")?.shadowRoot, "the pane mounted", 20000);
    await ctx.waitFor(() => root.querySelectorAll("section.sec").length >= 8, "the settings cards in the pane");
    await ctx.sleep(400);
    return { pw, root, $: (s) => root.querySelector(s), $$: (s) => [...root.querySelectorAll(s)] };
  };
  let p = await openPane();
  out.paneLoadMs = plugin.timing.paneLoadMs;
  check(out.paneLoadMs !== null && out.paneLoadMs < 400, `the pane's panel.js read in ${out.paneLoadMs} ms`);
  const cards = p.$$("section.sec").map((s) => s.getAttribute("aria-label"));
  out.cards = cards;
  for (const want of [/agent/i, /appearance/i, /context/i, /chat$/i, /prompt/i, /folder/i, /data/i, /about/i]) check(cards.some((t) => want.test(t)), `a "${want}" card in the pane: ${cards}`);
  check(!p.$(".vw__head") && !p.$('button[aria-label="Back to the chat"]'), "no panel header in the pane: Zotero titles it");
  check(injected.loaded() === null, "opening the pane did not load the panel");
  const agentCard = () => p.$$("section.sec").find((s) => /agent/i.test(s.getAttribute("aria-label")));
  const showBtn = [...agentCard().querySelectorAll("button")].find((b) => /^Show Claude Code's options$/.test(b.textContent));
  check(showBtn && !agentCard().querySelector(".sk-group, select"), "no catalog read on open: a button offers it");
  // our styles stay inside the shadow root, and Zotero's stay out of it
  const cs = p.pw.getComputedStyle(p.$(".sec"));
  check(cs.borderTopLeftRadius === "12px", "the cards keep their own shape: " + cs.borderTopLeftRadius);
  check(p.pw.getComputedStyle(p.$(".btn")).borderTopLeftRadius === "999px", "buttons keep their pill shape (Zotero's button CSS does not reach in)");
  check(!p.pw.document.querySelector(".sec, .zmc"), "nothing of ours in the Settings document itself");
  await ctx.snapshot("prefpane-1-top", p.pw);
  p.pw.close();
  await ctx.sleep(500);

  // 3. with a panel open (it read the catalog), the pane shows the agent's options at once
  await ctx.resize(1280, 900);
  injected.show();
  const panelRoot = await ctx.waitFor(() => win.document.getElementById("zmc-root")?.shadowRoot?.querySelector("textarea") && win.document.getElementById("zmc-root").shadowRoot, "panel rendered");
  const bundle = plugin.panel().bundle;
  await ctx.waitFor(() => bundle.knownCatalog("claude-code"), "the panel read its agent's catalog", 30000);
  p = await openPane();
  const agentCard2 = () => p.$$("section.sec").find((s) => /agent/i.test(s.getAttribute("aria-label")));
  await ctx.waitFor(() => agentCard2().querySelectorAll("select").length >= 1 && agentCard2().querySelectorAll(".radio").length > 3, "the model and permissions from the known catalog");
  check(![...agentCard2().querySelectorAll("button")].some((b) => /options$/.test(b.textContent)), "no button once the catalog is known");

  // 4. a change in the pane reaches the open panel at once, and the other way round
  const panelZmc = () => panelRoot.querySelector(".zmc");
  const glass0 = panelZmc().dataset.glass;
  const glassSwitch = p.$$('input[role="switch"]').find((i) => i.closest(".swrow")?.textContent.includes("Glass"));
  glassSwitch.click();
  await ctx.waitFor(() => panelZmc().dataset.glass !== glass0, "the panel's glass follows the pane");
  check(p.$(".zmc").dataset.glass === panelZmc().dataset.glass, "pane and panel agree");
  const large = p.$$(".seg__opt").find((b) => b.textContent === "Large");
  large.click();
  await ctx.waitFor(() => panelZmc().dataset.size === "large", "text size from the pane reaches the panel");
  const followSwitch = p.$$('input[role="switch"]').find((i) => i.closest(".swrow")?.textContent.includes("Follow what I'm reading"));
  followSwitch.click();
  await ctx.waitFor(() => plugin.panel().host.getSettings().followFocus === false, "the panel's host re-reads the settings (its cache was dropped)");
  out.paneToPanel = "ok";
  await plugin.panel().host.setSettings({ appearance: { ...plugin.panel().host.getSettings().appearance, textSize: "small", glass: true }, followFocus: true });
  await ctx.waitFor(() => p.$(".zmc").dataset.size === "small" && p.$(".zmc").dataset.glass === "on", "a change in the panel reaches the pane");
  await ctx.waitFor(() => p.$$('input[role="switch"]').find((i) => i.closest(".swrow")?.textContent.includes("Follow what I'm reading"))?.checked === true, "the pane's switches re-render");
  out.panelToPane = "ok";
  await plugin.panel().host.setSettings({ appearance: { ...plugin.panel().host.getSettings().appearance, textSize: "default" } });

  // 5. light and dark, glass on (Zotero follows browser.theme.toolbar-theme: 0 dark, 1 light)
  for (const [theme, name] of [[1, "light"], [0, "dark"]]) {
    Services.prefs.setIntPref("browser.theme.toolbar-theme", theme);
    await ctx.waitFor(() => p.$(".zmc").dataset.theme === name, `the pane follows the ${name} theme`);
    await ctx.sleep(400);
    await ctx.snapshot(`prefpane-2-${name}`, p.pw);
    const scroller = (() => { for (let e = p.pw.document.getElementById("zmc-prefs"); e; e = e.parentElement) if (e.scrollHeight > e.clientHeight + 40) return e; return null; })();
    if (scroller) { scroller.scrollTop = scroller.scrollHeight / 2; await ctx.sleep(300); await ctx.snapshot(`prefpane-3-${name}-middle`, p.pw); scroller.scrollTop = 0; }
  }
  Services.prefs.clearUserPref("browser.theme.toolbar-theme");

  // 6. closing the window disposes the pane (its pref observer goes with it); the panel keeps working
  p.pw.close();
  await ctx.sleep(600);
  check(p.pw.closed, "the Settings window closed");
  await plugin.panel().host.setSettings({ appearance: { ...plugin.panel().host.getSettings().appearance, glass: false } });
  check(plugin.panel().host.getSettings().appearance.glass === false && panelZmc(), "the panel's own saves still work with the pane gone");
  out.closed = "ok";
  return out;
}
