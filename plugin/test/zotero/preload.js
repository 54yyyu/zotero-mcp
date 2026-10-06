// The open paper ahead of time, in a real Zotero (run with --mock-agent): focus in the composer extracts the PDF's text
// (Zotero's PDF worker) into the default chat folder's papers/, the first message carries the paper's metadata and
// the file's path (once), changed metadata goes again, a changed PDF is extracted again, a chat folder of the user's own
// gets the profile's cache instead, and the cache sheds old files.
async function main(ctx) {
  const { Zotero, win, IOUtils, PathUtils } = ctx;
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };
  const bundle = () => ctx.plugin.panel().bundle;
  const defaultPapers = PathUtils.join(ctx.env("ZMC_DEFAULT_CHAT_FOLDER"), "papers");
  const profilePapers = PathUtils.join(Services.dirsvc.get("ProfD", Ci.nsIFile).path, "zotero-chat", "papers");

  async function paper(title, last, collection) {
    const att = await Zotero.Attachments.importFromFile({ file: Zotero.File.pathToFile(ctx.env("ZMC_TEST_PDF")), libraryID: Zotero.Libraries.userLibraryID });
    const item = new Zotero.Item("journalArticle");
    item.setField("title", title);
    item.setCreators([{ creatorType: "author", lastName: last, firstName: "Ann" }, { creatorType: "author", lastName: "Smith", firstName: "Bo" }]);
    item.setField("date", "2017");
    item.setField("publicationTitle", "Journal of Tests");
    item.setField("DOI", "10.1000/test.2017");
    item.setField("abstractNote", "We test whether a professional destination raises pay. ".repeat(40));
    item.addTag("labor");
    item.addTag("to-read");
    if (collection) item.setCollections([collection.id]);
    await item.saveTx();
    att.parentID = item.id;
    await att.saveTx();
    return { item, att };
  }
  const col = new Zotero.Collection({ name: "Pay gaps", libraryID: Zotero.Libraries.userLibraryID });
  await col.saveTx();
  const a = await paper("Test paper for the chat plugin", "Bell", col);

  // The profile's cache outlives a harness run; this run's attachment is new, so no file may name it yet.
  const named = async (dir, key) => (await IOUtils.exists(dir)) && (await IOUtils.getChildren(dir)).some((f) => PathUtils.filename(f).startsWith(key));
  check(!(await IOUtils.exists(defaultPapers)) && !(await named(profilePapers, a.att.key)), "nothing is extracted before the panel is used");
  await ctx.resize(1280, 800);
  await Zotero.Reader.open(a.att.id);
  const reader = await ctx.waitFor(() => Zotero.Reader._readers.find((r) => r.itemID === a.att.id && r._internalReader), "reader");
  await reader._initPromise;
  await ctx.waitFor(() => reader._internalReader._state?.primaryViewStats?.pagesCount, "pdf loaded");
  ctx.plugin.windows.get(win).show();
  const root = await ctx.waitFor(() => win.document.getElementById("zmc-root")?.shadowRoot?.querySelector("textarea.cin"), "panel rendered").then(() => win.document.getElementById("zmc-root").shadowRoot);
  const host = ctx.plugin.panel().host;
  await ctx.waitFor(() => host.currentContext().some((c) => c.kind === "reader"), "reader chip");

  // 1. Focus in the composer starts the extraction; the file is page-marked, named by key and author.
  const file = PathUtils.join(defaultPapers, `${a.att.key}-bell-2017.txt`);
  const t0 = Date.now();
  const ta = root.querySelector("textarea.cin");
  // The harness window is never the OS's active one, so focus() fires no focus event there: the event a click makes is sent.
  ta.dispatchEvent(new win.FocusEvent("focusin", { bubbles: true, composed: true }));
  await ctx.waitFor(() => IOUtils.exists(file), "the full text file", 30000);
  out.extractMs = Date.now() - t0;
  const text = await IOUtils.readUTF8(file);
  out.fileHead = text.split("\n", 1)[0];
  check(new RegExp(`^Full text of "Test paper for the chat plugin" \\(item ${a.item.key}, PDF ${a.att.key}\\), 4 pages; a line \\[p\\.N\\] starts page N, as zotero-cli read and citations count pages\\. Source: \\d+ bytes, modified \\d+\\.$`).test(out.fileHead), "the head line: " + out.fileHead);
  check([1, 2, 3, 4].every((n) => text.includes(`\n[p.${n}]\n`)) && !text.includes("[p.5]"), "page markers 1-4 and nothing else: " + text.slice(0, 400));
  check(/\[p\.3\]\n\n[^[]*Test paper for the chat plugin, page 3/.test(text), "page 3's text under its marker");

  // 2. The first message carries the metadata and the file's path; the second neither.
  const running = () => root.querySelector("button.send")?.classList.contains("send--stop");
  async function say(msg, k) {
    ta.value = msg; ta.dispatchEvent(new win.Event("input", { bubbles: true }));
    await ctx.waitFor(() => !root.querySelector("button.send").disabled, "send enabled");
    root.querySelector("button.send").click();
    await ctx.waitFor(() => root.querySelectorAll('.msg--assistant[data-state="end_turn"]').length === k && !running(), `turn ${k} finished`, 30000);
  }
  async function echoes(n) {
    for (let i = 0; i < 60; i++) {
      const [sv] = await host.sessions();
      const evs = sv ? await host.loadEvents(sv.id) : [];
      const byTurn = new Map();
      for (const e of evs) if (e.t === "text") byTurn.set(e.turn, (byTurn.get(e.turn) ?? "") + e.delta);
      const got = [...byTurn.values()].filter((t) => t.startsWith("{")).map((t) => JSON.parse(t));
      if (got.length >= n) return got;
      await ctx.sleep(250);
    }
    throw new Error("FAILED: echoes not saved");
  }
  await say("SCENARIO:echo first", 1);
  await say("SCENARIO:echo second", 2);
  a.item.addTag("seminar");
  await a.item.saveTx();
  await say("SCENARIO:echo third", 3);
  const [e1, e2, e3] = await echoes(3);
  const block = (e) => e.prompt.slice(0, e.prompt.indexOf("SCENARIO:echo"));
  out.firstBlock = block(e1);
  check(new RegExp(`About item ${a.item.key}: Ann Bell, Bo Smith · 2017 · Journal of Tests · DOI 10\\.1000/test\\.2017 · 4 pages\\nTags: labor, to-read\\nIn collections: Pay gaps\\nAbstract: We test whether`).test(e1.prompt), "turn 1 has the metadata: " + out.firstBlock);
  check(/Abstract: [^\n]{1500}\n/.test(e1.prompt) && /…\n/.test(e1.prompt), "the abstract is capped at 1,500 characters");
  check(e1.prompt.includes(`Full text is at ${file} (4 pages, page markers like [p.7]); read it with grep/sed or zotero-cli read for specific pages.`), "turn 1 names the file");
  check(e1.prompt.indexOf("Reading in the Zotero reader") < e1.prompt.indexOf("About item"), "after the reader line");
  check(!e2.prompt.includes("About item") && !e2.prompt.includes("Full text is at"), "turn 2 repeats neither: " + block(e2));
  check(/About item [^\n]*\nTags: labor, seminar, to-read/.test(e3.prompt) && !e3.prompt.includes("Full text is at"), "changed metadata goes again, the file line does not: " + block(e3));
  out.tokens = { firstBlock: Math.ceil(out.firstBlock.length / 4), secondBlock: Math.ceil(block(e2).length / 4) };
  check(/zotero-cli commands \(KEY/.test(e1.brief ?? "") && !/skill here first/.test(e1.brief), "the brief carries the tool sheet");

  // 3. Asked again for the same PDF: no extraction (in memory), the same file.
  let t = Date.now();
  const again = await host.paperContext(host.currentContext(), Promise.resolve());
  out.cachedMs = Date.now() - t;
  check(again.some((c) => c.id === `fulltext:${a.att.key}` && c.text.includes(file)) && out.cachedMs < 200, `a cache hit in ${out.cachedMs} ms`);

  // 4. A changed PDF (its mtime) is extracted again: the head's source changes.
  const pdf = await a.att.getFilePathAsync();
  await IOUtils.setModificationTime(pdf, Date.now() + 60_000);
  t = Date.now();
  await host.paperContext(host.currentContext(), Promise.resolve());
  await ctx.waitFor(async () => (await IOUtils.readUTF8(file)).split("\n", 1)[0] !== out.fileHead, "re-extracted after the PDF changed", 30000);
  out.reextractMs = Date.now() - t;

  // 5. A chat folder of the user's own: the file goes to the profile's cache, not into their folder; the cache sheds old files.
  const custom = PathUtils.join(PathUtils.parent(ctx.env("ZMC_DEFAULT_CHAT_FOLDER")), "custom-project");
  await IOUtils.makeDirectory(custom, { createAncestors: true, ignoreExisting: true });
  await host.setSettings({ chatFolder: custom });
  await IOUtils.makeDirectory(profilePapers, { createAncestors: true, ignoreExisting: true });
  const stale = PathUtils.join(profilePapers, "AAAAAAAA-old-paper.txt"), mine = PathUtils.join(profilePapers, "notes.md");
  await IOUtils.writeUTF8(stale, "old");
  await IOUtils.setModificationTime(stale, Date.now() - 70 * 86_400_000);
  await IOUtils.writeUTF8(mine, "not ours");
  const b = await paper("Another paper", "Cruz");
  win.Zotero_Tabs.select("zotero-pane");
  await win.ZoteroPane.selectItem(b.item.id);
  await ctx.waitFor(() => host.currentContext().length === 1 && host.currentContext()[0].ref.itemKey === b.item.key, "the one selected item");
  bundle().prefetchPaper();
  const bfile = PathUtils.join(profilePapers, `${b.att.key}-cruz-2017.txt`);
  await ctx.waitFor(() => IOUtils.exists(bfile), "the profile cache file", 30000);
  check(!(await IOUtils.exists(PathUtils.join(custom, "papers"))), "nothing written into the user's folder");
  const bctx = await host.paperContext(host.currentContext(), Promise.resolve());
  check(bctx.some((c) => c.text?.includes(`Full text is at ${bfile} (4 pages`)), "the absolute path in the profile: " + JSON.stringify(bctx.map((c) => c.text)));
  await ctx.waitFor(async () => !(await IOUtils.exists(stale)), "the 70-day-old file is gone");
  check(await IOUtils.exists(mine), "a file that is not ours stays");
  await host.setSettings({ chatFolder: "" });
  out.files = { defaultFolder: await IOUtils.getChildren(defaultPapers), profile: await IOUtils.getChildren(profilePapers) };
  return out;
}
