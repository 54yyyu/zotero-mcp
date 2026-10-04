// Everything PanelHost does that is not the agent: settings, keychain, history, search, open, describe, doctor, spawner.
async function main(ctx) {
  const { Zotero } = ctx;
  const { bundle } = ctx.plugin.panel();
  const host = bundle.host;
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };

  // --- fixtures: a paper with a 4-page PDF, a collection ---------------------------------------------------
  const att = await Zotero.Attachments.importFromFile({ file: Zotero.File.pathToFile(ctx.env("ZMC_TEST_PDF")), libraryID: Zotero.Libraries.userLibraryID });
  const parent = new Zotero.Item("journalArticle");
  parent.setField("title", "Are Emily and Greg more employable than Lakisha and Jamal?");
  parent.setCreators([{ creatorType: "author", lastName: "Bertrand", firstName: "Marianne" }, { creatorType: "author", lastName: "Mullainathan", firstName: "Sendhil" }]);
  parent.setField("date", "2004");
  await parent.saveTx();
  att.parentID = parent.id; await att.saveTx();
  const col = new Zotero.Collection(); col.name = "Labor market discrimination"; col.libraryID = Zotero.Libraries.userLibraryID; await col.saveTx();

  // --- settings --------------------------------------------------------------------------------------------
  const s0 = host.getSettings();
  check(s0.backend === "claude-code" && s0.prompts.length === 4 && s0.prompts[0].slot === 1, "default settings");
  check(s0.model.codex === "" && s0.effort.pi === "" && s0.followFocus && s0.enterToSend && !s0.expandTools && !s0.openAtStart, "defaults for the per-backend maps and switches");
  // per-backend maps never leak across backends
  await host.setSettings({ backend: "codex", mode: { ...s0.mode, codex: "read-only" }, model: { ...s0.model, "claude-code": "haiku" }, effort: { ...s0.effort, codex: "high" } });
  const s1 = host.getSettings();
  check(s1.backend === "codex" && s1.mode.codex === "read-only" && s1.mode["claude-code"] === "" && s1.model["claude-code"] === "haiku" && s1.model.codex === "" && s1.effort.codex === "high", "per-backend values stay per backend: " + JSON.stringify({ backend: s1.backend, mode: s1.mode, model: s1.model, effort: s1.effort }));
  check(s1.auth["claude-code"] === "subscription" && s1.auth.pi === "api-key", "auth defaults survive a partial save");
  // a settings blob saved by an older shape (a missing backend key) is still complete
  Zotero.Prefs.set("extensions.zotero-chat.settings", JSON.stringify({ model: { codex: "gpt-6-luna" } }), true);
  const s2 = host.getSettings();
  check(s2.model.codex === "" || s2.model.codex === "gpt-6-luna", "reads");
  await host.resetSettings();
  const s3 = host.getSettings();
  check(s3.backend === "claude-code" && s3.model.codex === "" && s3.mode["claude-code"] === "" && s3.prompts.length === 4, "reset returns the defaults");
  out.settings = "ok";

  // --- keychain: set, has, replace, clear ------------------------------------------------------------------
  check(!(await host.hasApiKey("claude-code")), "no key at first");
  await host.setApiKey("claude-code", "sk-ant-test-1");
  check(await host.hasApiKey("claude-code"), "key stored");
  await host.setApiKey("claude-code", "sk-ant-test-2");
  const logins = await Services.logins.searchLoginsAsync({ origin: "chrome://zotero-chat", httpRealm: "claude-code API key" });
  check(logins.length === 1 && logins[0].password === "sk-ant-test-2", "replace keeps exactly one login");
  await host.setApiKey("claude-code", null);
  check(!(await host.hasApiKey("claude-code")), "key cleared");
  out.keychain = "ok";

  // --- the context switches change what the agent is told (applied by the host, not the UI) ---------------------------
  const fixtureAtt = Zotero.Items.getByLibraryAndKey(Zotero.Libraries.userLibraryID, att.key);
  await Zotero.Reader.open(fixtureAtt.id);
  await ctx.waitFor(() => host.currentContext()[0]?.kind === "reader", "reader chip for the switch test");
  const trackerCtx = ctx.plugin.panel().bundle.context;
  const rdr = Zotero.Reader._readers.find((r) => r.itemID === fixtureAtt.id);
  await ctx.waitFor(() => rdr._internalReader?._state?.primaryViewStats?.pagesCount, "pdf loaded");
  const stats = rdr._internalReader._state.primaryViewStats;
  rdr._internalReader._state.primaryViewStats = { ...stats, canCopy: 1 };
  trackerCtx.onPopup({ reader: rdr, params: { annotation: { text: "selected words", pageLabel: "1", position: { pageIndex: 0 } } } });
  const kindsOf = () => host.currentContext().map((c) => c.kind).join(",");
  check(kindsOf() === "reader,selection", "selection attached by default: " + kindsOf());
  await host.setSettings({ attachSelection: false });
  check(kindsOf() === "reader", "attachSelection off hides the selection chip: " + kindsOf());
  await host.setSettings({ attachSelection: true, followFocus: false });
  check(kindsOf() === "", "followFocus off attaches nothing automatically: " + kindsOf());
  await host.setSettings({ followFocus: true });
  check(kindsOf() === "reader,selection", "and back on");
  rdr._internalReader._state.primaryViewStats = { ...stats, canCopy: 0 };
  await host.resetSettings();
  out.contextSwitches = "ok";

  // --- history ----------------------------------------------------------------------------------------------
  const sess = { id: "s-test-1", title: "First chat", backend: "claude-code", agentSessionId: "agent-1", updatedAt: 0 };
  await host.appendEvent(sess, { t: "user", id: "u1", text: "hello", chips: [] });
  await host.appendEvent(sess, { t: "turn_start", turn: "t1" });
  await host.appendEvent(sess, { t: "text", turn: "t1", delta: "hi" });
  const list = await host.sessions();
  check(list.length === 1 && list[0].id === "s-test-1" && list[0].updatedAt > 0, "session listed");
  const evs = await host.loadEvents("s-test-1");
  check(evs.length === 3 && evs[2].delta === "hi", "events replay in order");
  await host.deleteSession("s-test-1");
  check((await host.sessions()).length === 0 && (await host.loadEvents("s-test-1")).length === 0, "session deleted");
  await host.appendEvent({ ...sess, id: "s-a" }, { t: "user", id: "u", text: "a", chips: [] });
  await host.appendEvent({ ...sess, id: "s-b" }, { t: "user", id: "u", text: "b", chips: [] });
  check((await host.sessions()).length === 2, "two sessions saved");
  await host.clearHistory();
  check((await host.sessions()).length === 0 && (await host.loadEvents("s-a")).length === 0, "clearHistory removes every chat");
  await host.appendEvent({ ...sess, id: "s-c" }, { t: "user", id: "u", text: "c", chips: [] });
  check((await host.sessions()).length === 1, "history works again after clearing");
  await host.clearHistory();
  out.history = "ok";

  // --- search -----------------------------------------------------------------------------------------------
  const hits = await host.search("Bertrand");
  check(hits.some((h) => h.kind === "item" && h.ref.itemKey === parent.key && h.title === "Bertrand and Mullainathan 2004"), "item found by author, labelled like a citation: " + JSON.stringify(hits.map((h) => h.title)));
  const chits = await host.search("discrimination");
  check(chits.some((h) => h.kind === "collection"), "collection found");
  const chip = await host.chipFor(hits.find((h) => h.kind === "item"));
  check(chip.auto === false && chip.pinned === true && chip.kind === "item", "mentioned chip is user-added and pinned");
  out.search = hits.map((h) => h.title);

  // --- the chat folder: visible by default, configurable, remembered per chat ------------------------------------------
  // In the harness the default is redirected into .dev; with the override off, it is the real, visible location
  // (only read here: nothing may be created there).
  const defaultFolder = host.about().workspace;
  check(defaultFolder === ctx.env("ZMC_DEFAULT_CHAT_FOLDER"), "the harness redirects the default folder: " + defaultFolder);
  Services.env.set("ZMC_DEFAULT_CHAT_FOLDER", "");
  const realDefault = host.about().workspace;
  Services.env.set("ZMC_DEFAULT_CHAT_FOLDER", defaultFolder);
  check(/\/Zotero-Chat$/.test(realDefault) && !realDefault.includes("zotero-chat/workspace") && !realDefault.includes("/Profiles/"), "the real default is visible and predictable, not buried in the profile: " + realDefault);
  const customFolder = PathUtils.join(Zotero.getTempDirectory().path, "my-chat-folder");
  await host.setSettings({ chatFolder: customFolder });
  check(host.about().workspace === customFolder, "setting changes where chats run");
  const prepCustom = await host.prepareSession();
  check(prepCustom.cwd === customFolder && await IOUtils.exists(PathUtils.join(customFolder, ".claude", "skills", "zotero-cli", "SKILL.md")), "new chats run in the chosen folder, with the skill installed in it");
  const oldFolder = PathUtils.join(Zotero.getTempDirectory().path, "old-chat-folder");
  const prepResume = await host.prepareSession(oldFolder);
  check(prepResume.cwd === oldFolder, "a resumed chat runs in the folder it started in, whatever the setting is now");
  await host.setSettings({ chatFolder: "" });
  check(host.about().workspace === defaultFolder, "empty setting is the default folder again");

  // --- terminal resume commands, one per backend, quoted for any path --------------------------------------------------
  const sv = (backend, cwd) => ({ id: "x", title: "t", backend, cwd, agentSessionId: "abc-123", updatedAt: 0 });
  check(host.resumeCommand(sv("claude-code", "/Users/me/Zotero Chat")) === "cd '/Users/me/Zotero Chat' && claude --resume abc-123", "claude: " + host.resumeCommand(sv("claude-code", "/Users/me/Zotero Chat")));
  check(host.resumeCommand(sv("codex", "/a")) === "cd '/a' && codex resume abc-123" && host.resumeCommand(sv("pi", "/a")) === "cd '/a' && pi --session abc-123", "codex and pi");
  check(host.resumeCommand(sv("claude-code", "/it's here")).startsWith("cd '/it'\\''s here' &&"), "a quote in the path is escaped: " + host.resumeCommand(sv("claude-code", "/it's here")));

  // --- open: zotero:// links the agent writes ----------------------------------------------------------------
  await host.open(`zotero://open-pdf/library/items/${att.key}?page=3`);
  const reader = await ctx.waitFor(() => Zotero.Reader._readers.find((r) => r.itemID === att.id && r._internalReader?._state?.primaryViewStats?.pagesCount), "reader opened by link");
  await ctx.waitFor(() => reader._internalReader._state.primaryViewStats.pageIndex === 2, "open-pdf link lands on page 3");
  out.openPdf = "page " + reader._internalReader._state.primaryViewStats.pageLabel;
  await host.open(`zotero://open-pdf/library/items/${att.key}?page=1`);
  await ctx.waitFor(() => reader._internalReader._state.primaryViewStats.pageIndex === 0, "second link moves the same reader to page 1");
  await host.open(`zotero://select/library/items/${parent.key}`);
  await ctx.waitFor(() => ctx.win.Zotero_Tabs.selectedType === "library" && ctx.win.ZoteroPane.getSelectedItems()[0]?.key === parent.key, "select link shows the item");
  await host.open({ libraryID: Zotero.Libraries.userLibraryID, collectionKey: col.key });
  await ctx.waitFor(() => ctx.win.ZoteroPane.getSelectedCollections()[0]?.key === col.key, "collection ref selects it");
  // web links go to the system browser (stubbed here: no browser may open in a test); other schemes are refused
  const launched = []; const realLaunch = Zotero.launchURL; Zotero.launchURL = (u) => launched.push(u);
  await host.open("https://example.com/paper"); await host.open("http://example.org");
  Zotero.launchURL = realLaunch;
  check(launched.join(" ") === "https://example.com/paper http://example.org", "http(s) links open externally: " + launched);
  for (const bad of ["javascript:alert(1)", "file:///etc/passwd", "data:text/html,x", "ftp://x", "chrome://browser/content/"]) {
    let err = null; try { await host.open(bad); } catch (e) { err = String(e); }
    check(err && err.includes("Not a Zotero link"), "refused: " + bad);
  }
  out.open = "ok";

  // --- drops on the composer: the two formats Zotero writes ------------------------------------------------------------
  const dt = (map) => { const d = new win0.DataTransfer(); for (const [k, v] of Object.entries(map)) d.setData(k, v); return d; };
  const win0 = ctx.win;
  const dItems = await host.dropChips(dt({ "zotero/item": [parent.id, att.id, 999999].join(",") }));
  check(dItems.length === 2 && dItems.every((c) => c.kind === "item" && !c.auto && c.pinned && c.label === "Bertrand and Mullainathan 2004"), "dropped items (an attachment shows its parent, unknown ids are skipped): " + JSON.stringify(dItems.map((c) => c.label)));
  const savedAnn = await Zotero.Annotations.saveFromJSON(att, { key: Zotero.DataObjectUtilities.generateKey(), type: "highlight", pageLabel: "2", color: "#ffd400", sortIndex: "00001|000100|00100", position: { pageIndex: 1, rects: [[72, 700, 300, 712]] }, comment: "key claim", text: "the gap is almost unchanged", tags: [] });
  const imgAnn = await Zotero.Annotations.saveFromJSON(att, { key: Zotero.DataObjectUtilities.generateKey(), type: "image", pageLabel: "1", color: "#ffd400", sortIndex: "00000|000500|00100", position: { pageIndex: 0, rects: [[72, 640, 400, 725]] }, comment: "", text: "", tags: [] });
  const dAnn = await host.dropChips(dt({ "zotero/annotation": JSON.stringify([{ id: savedAnn.key, type: "highlight", attachmentItemID: att.id }, { id: imgAnn.key, type: "image", attachmentItemID: att.id }, { id: "UNSAVED1", type: "highlight", text: "dragged selection", pageLabel: "3", attachmentItemID: att.id }]) }));
  check(dAnn.length === 3, "three annotation chips: " + JSON.stringify(dAnn.map((c) => c.kind)));
  check(dAnn[0].kind === "annotation" && dAnn[0].text === "the gap is almost unchanged" && dAnn[0].ref.annotationKey === savedAnn.key && dAnn[0].pinned && !dAnn[0].auto, "saved highlight chip carries its text and key");
  check(dAnn[1].kind === "area" && dAnn[1].image && dAnn[1].image.data.length > 200 && dAnn[1].pinned, "dropped area carries its rendered PNG");
  check(dAnn[2].kind === "annotation" && dAnn[2].text === "dragged selection" && dAnn[2].ref.pageLabel === "3", "an unsaved selection still becomes a chip from the drag data");
  check((await host.dropChips(dt({ "text/plain": "hello" }))).length === 0, "anything else drops nothing");
  check((await host.dropChips(dt({ "zotero/annotation": "{not json" }))).length === 0, "garbled drag data is ignored");
  out.drops = "ok";

  // --- describeContext ---------------------------------------------------------------------------------------
  await host.open(`zotero://open-pdf/library/items/${att.key}?page=2`);
  await ctx.waitFor(() => host.currentContext()[0]?.kind === "reader" && host.currentContext()[0].ref.pageIndex === 1, "context follows to page 2");
  const d = host.describeContext(host.currentContext());
  check(d.text.startsWith("<zotero-context>") && d.text.endsWith("</zotero-context>"), "wrapped in the context tag");
  check(d.text.includes(att.key) && d.text.includes(parent.key) && d.text.includes("on p.2") && d.text.includes("Are Emily and Greg"), "names the keys, title and page: " + d.text);
  const libText = (await (async () => { ctx.win.Zotero_Tabs.select("zotero-pane"); ctx.win.ZoteroPane.selectItem(parent.id); await ctx.sleep(500); return host.describeContext(host.currentContext()).text; })());
  check(libText.includes(`item ${parent.key}`) && libText.includes(`PDF attachment ${att.key}`), "library item lists its PDF key so the agent can cite it: " + libText);
  check(host.describeContext([]).text === "", "no context, no block");
  out.describe = d.text;

  // --- doctor ------------------------------------------------------------------------------------------------
  const checks = await host.doctor();
  out.doctor = checks.map((c) => `${c.ok ? "ok  " : "FAIL"} ${c.id}: ${c.label}${c.detail ? " (" + String(c.detail).slice(0, 70) + ")" : ""}${c.fix ? " [fix: " + c.fix.label + "]" : ""}`);
  const api = checks.find((c) => c.id === "zotero-api");
  // The harness Zotero listens on 23200, and zotero-cli only talks to 23119: the doctor must say so.
  check(api && !api.ok && api.label.includes("23200") && api.label.includes("23119"), "zotero-api flags a non-default port: " + JSON.stringify(api));
  check(checks.find((c) => c.id === "node")?.ok, "node found through the login shell");

  // --- spawner: run, spawn with stdin, kill the tree -----------------------------------------------------------
  const sp = bundle.spawner;
  const env = await sp.baseEnv();
  check(env.PATH && env.PATH.split(":").length > 4 && env.HOME, "baseEnv has a login PATH");
  const echo = await sp.run("/bin/echo", ["hello", "gecko"], { env });
  check(echo.code === 0 && echo.stdout.trim() === "hello gecko", "run captures stdout");
  const cat = await sp.spawn("/bin/cat", [], { env });
  const lines = []; cat.onStdoutLine((l) => lines.push(l));
  cat.write("one\n"); cat.write("two\n");
  await ctx.waitFor(() => lines.length === 2, "cat echoes both lines");
  const sh = await sp.spawn("/bin/sh", ["-c", "sleep 300 & sleep 300"], { env });
  await ctx.sleep(500);
  const before = (await sp.run("/usr/bin/pgrep", ["-P", String(sh.pid)], { env })).stdout.trim().split("\n").filter(Boolean).length;
  await sh.kill(); await cat.kill();
  await ctx.sleep(300);
  const after = (await sp.run("/usr/bin/pgrep", ["-f", "sleep 300"], { env })).stdout.trim();
  check(before >= 1 && after === "", `kill() takes the whole tree (children before=${before}, sleeps left='${after}')`);
  out.spawner = "ok";

  // --- prepareSession: workspace with the skill in it ----------------------------------------------------------
  const prep = await host.prepareSession();
  check(prep.brief.includes("zotero-cli") && prep.cwd === defaultFolder, "brief, and the default chat folder: " + prep.cwd);
  out.workspace = prep.cwd;
  const about = host.about();
  check(about.version === ctx.plugin.init.version && about.workspace === prep.cwd, "about() reports the version and the workspace: " + JSON.stringify(about));
  const skill = PathUtils.join(prep.cwd, ".claude", "skills", "zotero-cli", "SKILL.md");
  out.skillInstalled = await IOUtils.exists(skill);
  return out;
}
