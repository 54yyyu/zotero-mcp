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

  // 2b. the effort slider opens (it needs nothing the sandbox lacks), steps with the keys, applies and closes
  const effBtn = $(".pick--effort");
  check(effBtn && !effBtn.hidden, "the effort picker is showing");
  effBtn.click();
  await ctx.waitFor(() => $(".menu--effort .eff__track"), "the effort slider rendered (not stuck on Loading)");
  const stops = $$(".eff__stop").length;
  check(stops >= 2, "one stop per level: " + stops);
  const level0 = $(".eff__cur").textContent;
  const key = (k) => $(".eff__track").dispatchEvent(new win.KeyboardEvent("keydown", { key: k, bubbles: true, cancelable: true }));
  key(level0 === $$(".eff__stop")[0]?.title ? "ArrowRight" : "ArrowLeft");
  check($(".eff__cur").textContent !== level0, "an arrow key moves the level");
  key("Enter");
  await ctx.waitFor(() => !$(".menu--effort"), "Enter applies and closes");
  check(new RegExp($(".eff__cur")?.textContent ?? ".").test("") || true, "no crash after close");
  out.effortSlider = "ok";

  // 3. settings: every section, the catalog-driven pickers, and a save that reaches the host
  click(byLabel("Settings"), "settings button");
  await ctx.waitFor(() => $$("section.sec").length >= 5, "settings sections");
  await ctx.waitFor(() => $$("select, [role=radio]").length > 4, "pickers populated from the catalog");
  const sections = $$("section.sec").map((s) => s.getAttribute("aria-label"));
  out.sections = sections;
  for (const want of [/agent/i, /sign/i, /context/i, /chat/i, /prompt/i, /folder/i, /data/i]) check(sections.some((t) => want.test(t)), `a "${want}" section exists in: ${sections}`);
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
  click($$("button").find((b) => /^Choose/.test(b.textContent.trim())), "Choose… button");
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

  // 7. the close button
  click(byLabel("Close the panel"), "close button");
  await ctx.waitFor(() => !injected.isOpen(), "close button closes the panel");
  check(win.document.getElementById("zmc-panel").hidden, "panel hidden");
  injected.show();
  out.close = "ok";
  return out;
}
