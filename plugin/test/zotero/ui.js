// The panel's screens and controls, driven for real in Gecko (run with --mock-agent): settings, chat folder, history with
// the terminal command, drag and drop, the close button, links. Snapshots are for a human to look at.
async function main(ctx) {
  const { Zotero, win } = ctx;
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };
  const injected = ctx.plugin.windows.get(win);

  const att = await Zotero.Attachments.importFromFile({ file: Zotero.File.pathToFile(ctx.env("ZMC_TEST_PDF")), libraryID: Zotero.Libraries.userLibraryID });
  const parent = new Zotero.Item("journalArticle");
  parent.setField("title", "Are Emily and Greg more employable than Lakisha and Jamal?");
  parent.setCreators([{ creatorType: "author", lastName: "Bertrand", firstName: "Marianne" }, { creatorType: "author", lastName: "Mullainathan", firstName: "Sendhil" }]);
  parent.setField("date", "2004");
  await parent.saveTx();
  att.parentID = parent.id; await att.saveTx();

  await ctx.resize(1280, 900);
  injected.show();
  const root = await ctx.waitFor(() => win.document.getElementById("zmc-root")?.shadowRoot?.querySelector("textarea") && win.document.getElementById("zmc-root").shadowRoot, "panel rendered");
  const host = ctx.plugin.panel().host;
  const $ = (sel) => root.querySelector(sel);
  const $$ = (sel) => [...root.querySelectorAll(sel)];
  const text = () => root.textContent.replace(/\s+/g, " ");
  const byLabel = (label) => $$("button").find((b) => b.getAttribute("aria-label") === label);
  const click = (el, what) => { if (!el) throw new Error("no " + what); el.click(); };
  const running = () => $("button.send")?.classList.contains("send--stop");
  async function say(msg) {
    const ta = $("textarea"); ta.value = msg; ta.dispatchEvent(new win.Event("input", { bubbles: true }));
    await ctx.waitFor(() => !$("button.send").disabled, "send enabled"); $("button.send").click();
    await ctx.waitFor(() => running() || text().includes(msg.slice(0, 12)), "turn started");
    await ctx.waitFor(() => !running(), "turn finished", 30000);
  }
  const scrollBox = () => { for (let e = $(".set"); e; e = e.parentElement) if (e.scrollHeight > e.clientHeight + 4) return e; return null; };

  // 1. a chat, so there is history and an answer with links to click
  ctx.win.ZoteroPane.selectItem(parent.id);
  await ctx.sleep(500);
  await say(`SCENARIO:rich ATT=${att.key}`);

  // 2. external links: web links go to the system browser (stubbed), the javascript: one is not even a link
  const launched = []; const realLaunch = Zotero.launchURL; Zotero.launchURL = (u) => launched.push(u);
  const here = win.location.href;
  const webLink = $$("a").find((a) => /zotero\.org/.test(a.getAttribute("href") || a.dataset?.href || a.textContent) || a.textContent.includes("Zotero site"));
  click(webLink, "web link"); await ctx.sleep(300);
  Zotero.launchURL = realLaunch;
  check(launched.length === 1 && /^https:\/\/www\.zotero\.org\/?$/.test(launched[0]) && win.location.href === here, "web link opened externally and the window did not navigate: " + JSON.stringify(launched));
  check(!$$("a").some((a) => /javascript:/i.test(a.getAttribute("href") || "")) && text().includes("a bad link"), "javascript: link is plain text");
  out.links = "ok";
  await ctx.snapshot("ui-1-answer");

  // 2b. the model dropdown opens above its button (it needs nothing the sandbox lacks): the agents with their status,
  // the models, and the effort slider, which steps with the keys, applies and closes
  const modelBtn = $(".pick--model");
  check(modelBtn && !$(".pick--effort"), "one button for the model and effort");
  modelBtn.click();
  await ctx.waitFor(() => $(".mdd .eff__track"), "the dropdown rendered its effort slider (not stuck on Loading)");
  check($$(".mdd__agent").length === 3 && $(".mdd__agent--on")?.dataset.id === "claude-code", "the three agents, Claude Code chosen");
  check($$('.mdd [role="menuitemradio"]').length >= 1, "the models are listed");
  const md = $(".mdd").getBoundingClientRect(), mb = modelBtn.getBoundingClientRect();
  check(md.bottom <= mb.top + 1 && md.left >= $(".composer").getBoundingClientRect().left - 1, "the dropdown sits above its button");
  modelBtn.click(); // a second press closes it
  check(!$(".mdd"), "the button toggles the dropdown");
  for (const [theme, shot] of [[1, "ui-1b-model-dropdown-light"], [0, "ui-1c-model-dropdown-dark"]]) {
    Services.prefs.setIntPref("browser.theme.toolbar-theme", theme);
    await ctx.sleep(500);
    modelBtn.click();
    await ctx.waitFor(() => $(".mdd .eff__track"), "the dropdown in " + shot);
    await ctx.snapshot(shot);
    modelBtn.click();
  }
  Services.prefs.clearUserPref("browser.theme.toolbar-theme");
  await ctx.sleep(300);
  modelBtn.click();
  await ctx.waitFor(() => $(".mdd .eff__track"), "the dropdown again");
  const stops = $$(".eff__stop").length;
  check(stops >= 2, "one stop per level: " + stops);
  const level0 = $(".eff__cur").textContent;
  const key = (k) => $(".eff__track").dispatchEvent(new win.KeyboardEvent("keydown", { key: k, bubbles: true, cancelable: true }));
  key(level0 === $$(".eff__stop")[0]?.title.replace(/ \(recommended\)$/, "") ? "ArrowRight" : "ArrowLeft");
  check($(".eff__cur").textContent !== level0, "an arrow key moves the level");
  key("Enter");
  await ctx.waitFor(() => !$(".mdd"), "Enter applies and closes");
  out.effortSlider = "ok";

  // 3. settings: every section, the catalog-driven pickers, and a save that reaches the host
  click(byLabel("Settings"), "settings button");
  await ctx.waitFor(() => $$("section.sec").length >= 5, "settings sections");
  await ctx.waitFor(() => $$("select, [role=radio]").length > 4, "pickers populated from the catalog");
  const sections = $$("section.sec").map((s) => s.getAttribute("aria-label"));
  out.sections = sections;
  for (const want of [/agent/i, /appearance/i, /context/i, /chat/i, /prompt/i, /folder/i, /data/i, /about/i]) check(sections.some((t) => want.test(t)), `a "${want}" section exists in: ${sections}`);
  const modeRadios = $$("[role=radio]").map((r) => r.textContent.trim()).filter(Boolean);
  out.radios = modeRadios.slice(0, 12);
  check(modeRadios.some((t) => /plan/i.test(t)), "permission modes come from the agent: " + modeRadios);
  await ctx.snapshot("ui-2-settings-top");
  const box = scrollBox();
  if (box) { box.scrollTop = box.scrollHeight / 2; await ctx.sleep(300); await ctx.snapshot("ui-3-settings-middle"); box.scrollTop = box.scrollHeight; await ctx.sleep(300); await ctx.snapshot("ui-4-settings-bottom"); box.scrollTop = 0; }
  const planBtn = $$("[role=radio]").find((r) => /^\s*Plan/i.test(r.textContent));
  click(planBtn, "Plan mode"); 
  await ctx.waitFor(() => host.getSettings().mode["claude-code"] === "plan", "choosing a mode saves it for this backend only");
  check(host.getSettings().mode.codex === "" && host.getSettings().mode.pi === "", "other backends untouched");
  out.settingsSave = "ok";

  // 4. the chat folder row: choose (picker stubbed), shown, and stored
  const picked = PathUtils.join(Zotero.getTempDirectory().path, "picked-chat-folder");
  host.chooseFolder = async () => picked;
  click($$("button").find((b) => b.textContent.trim() === "Choose\u2026"), "Choose… button");
  await ctx.waitFor(() => host.getSettings().chatFolder === picked, "the chosen folder is saved");
  await ctx.waitFor(() => text().includes("picked-chat-folder"), "the row shows the folder");
  click($$("button").find((b) => /Use default/.test(b.textContent)), "Use default");
  await ctx.waitFor(() => host.getSettings().chatFolder === "", "Use default clears it");
  out.chatFolder = "ok";

  // 5. history: the chat is there with its folder and a copyable terminal command
  click(byLabel("History"), "history button");
  await ctx.waitFor(() => $$(".hrow, [class*=hrow]").length, "a history row");
  const saved = (await host.sessions())[0];
  check(saved && saved.cwd && saved.cwd === host.about().workspace, "the saved chat remembers its folder: " + JSON.stringify(saved && saved.cwd));
  const copyBtn = $$("button").find((b) => /terminal command/i.test(b.getAttribute("aria-label") || b.title || b.textContent));
  check(copyBtn, "history offers Copy terminal command");
  let copied = null; const realCopy = Zotero.Utilities.Internal.copyTextToClipboard; Zotero.Utilities.Internal.copyTextToClipboard = (t) => { copied = t; };
  const navClip = win.navigator.clipboard; 
  try { copyBtn.click(); } catch (e) { out.copyError = String(e); }
  await ctx.sleep(500);
  Zotero.Utilities.Internal.copyTextToClipboard = realCopy;
  out.copyAction = copied || "(clipboard write handled by the browser API)";
  check(/Copied/i.test(text()), "the button confirms with Copied");
  out.resumeCommand = host.resumeCommand(saved);
  await ctx.snapshot("ui-5-history");

  // 6. back to the chat, drop an item on the composer
  click(byLabel("History"), "history toggle");
  await ctx.waitFor(() => $("textarea") && !$(".hrow"), "back in the chat");
  const card = $(".composer, .cwrap, form, .cbox") || $("textarea").closest("div[class]");
  const dt = new win.DataTransfer(); dt.setData("zotero/item", String(parent.id));
  const fire = (type) => (($(".dropzone, .composer, .cbox") || card).dispatchEvent(new win.DragEvent(type, { dataTransfer: dt, bubbles: true, cancelable: true })));
  fire("dragenter"); fire("dragover");
  await ctx.sleep(200);
  out.dragoverShown = /Drop to add/.test(text());
  await ctx.snapshot("ui-6-dragover");
  fire("drop");
  await ctx.waitFor(() => $$("[class*=chip]").some((c) => /Bertrand and Mullainathan 2004/.test(c.textContent)), "dropped item became a chip");
  out.drop = "ok";

  // 6b. a long catalog in Gecko: pi as it reports itself with OpenRouter configured (416 + the user's own 2), its default in
  // the middle. The dropdown has a search field, pi's default first and marked Default, the user's providers next; typing in
  // the list searches. (The catalog is stubbed on the host's runtime; the mock bridge speaks Claude for every backend.)
  const big = [];
  for (let i = 0; i < 415; i++) big.push({ id: `openrouter/vendor-${i % 26}/model-${String(i).padStart(3, "0")}`, name: `openrouter/Vendor ${i % 26}: Model ${i}` });
  big.push({ id: "openrouter/moonshotai/kimi-k2.6", name: "openrouter/MoonshotAI: Kimi K2.6" });
  for (const n of ["4", "4.1", "4.5"]) big.push({ id: `openrouter/anthropic/claude-opus-${n}`, name: `openrouter/Anthropic: Claude Opus ${n}` });
  big.sort((a, b) => (a.id < b.id ? -1 : 1));
  big.push({ id: "my-cluster/deepseek-v4-flash", name: "my-cluster/DeepSeek V4 Flash (4xH100)" }, { id: "ollama/qwen3-8b", name: "ollama/Qwen3 8B" });
  const realCatalog = host.runtime.catalog;
  const before = host.getSettings();
  host.runtime.catalog = async (b) => (b === "pi" ? { models: big, modes: [], efforts: [{ id: "low", name: "Low" }, { id: "medium", name: "Medium" }, { id: "high", name: "High" }], model: "openrouter/moonshotai/kimi-k2.6", effort: "medium" } : realCatalog(b));
  click(byLabel("New chat"), "new chat");
  await host.setSettings({ backend: "pi", appearance: { ...before.appearance, glass: true } });
  await ctx.sleep(300);
  for (const [theme, shot] of [[1, "ui-6b-pi-models-light"], [0, "ui-6c-pi-models-dark"]]) {
    Services.prefs.setIntPref("browser.theme.toolbar-theme", theme);
    await ctx.sleep(500);
    $(".pick--model").click();
    await ctx.waitFor(() => $(".mdd .mdd__qi") && $(".mdd .eff__track"), "the long list's search field " + shot);
    const first = $$('.mdd [role="menuitemradio"]').slice(0, 2).map((r) => r.querySelector(".menu__t").textContent + (r.querySelector(".mdd__tag") ? " [Default]" : ""));
    check(first[0] === "MoonshotAI: Kimi K2.6 [Default]" && first[1] === "DeepSeek V4 Flash (4xH100)", "pi's default first, then the user's providers: " + first);
    check($$('.mdd [role="menuitemradio"]').length === 80 && $$(".mdd__h").length === 3, "80 rows drawn under three provider headings");
    await ctx.snapshot(shot);
    const key = (k) => root.activeElement.dispatchEvent(new win.KeyboardEvent("keydown", { key: k, bubbles: true, cancelable: true }));
    key("c"); // on the current model: the search field takes it
    check(root.activeElement === $(".mdd__qi") && $(".mdd__qi").value === "c", "typing in the list goes to the search field");
    const times = [];
    for (const q of ["cl", "cla", "claude", "claude o", "claude opus"]) {
      const t0 = win.performance.now();
      $(".mdd__qi").value = q;
      $(".mdd__qi").dispatchEvent(new win.Event("input", { bubbles: true }));
      $(".mdd__models").getBoundingClientRect();
      times.push(win.performance.now() - t0);
    }
    check($$('.mdd [role="menuitemradio"]').length === 3 && $(".mdd__qn").textContent === "3 models", "every word matches: " + $(".mdd__qn").textContent);
    out.piKeystrokeMs = times.map((t) => Math.round(t * 10) / 10);
    await ctx.snapshot(shot + "-search");
    key("Escape");
    check($(".mdd") && $(".mdd__qi").value === "", "Esc clears the search first");
    key("Escape");
    await ctx.waitFor(() => !$(".mdd"), "a second Esc closes the dropdown");
  }
  Services.prefs.clearUserPref("browser.theme.toolbar-theme");
  host.runtime.catalog = realCatalog;
  await host.setSettings({ backend: before.backend, appearance: before.appearance });
  await ctx.sleep(300);
  out.longModelList = "ok";

  // 7. the close button
  click(byLabel("Close the panel"), "close button");
  await ctx.waitFor(() => !injected.isOpen(), "close button closes the panel");
  check(win.document.getElementById("zmc-panel").hidden, "panel hidden");
  injected.show();
  out.close = "ok";
  return out;
}
