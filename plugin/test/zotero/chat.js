// The whole chat in a real Zotero, with the mock agent standing in for the bridge (run with --mock-agent):
// real panel, real runtime, real Gecko rendering. Streams, tool rows, permission, citations that open the page, history.
async function main(ctx) {
  const { Zotero, win } = ctx;
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };

  const att = await Zotero.Attachments.importFromFile({ file: Zotero.File.pathToFile(ctx.env("ZMC_TEST_PDF")), libraryID: Zotero.Libraries.userLibraryID });
  const parent = new Zotero.Item("journalArticle");
  parent.setField("title", "Are Emily and Greg more employable than Lakisha and Jamal?");
  parent.setCreators([{ creatorType: "author", lastName: "Bertrand", firstName: "Marianne" }, { creatorType: "author", lastName: "Mullainathan", firstName: "Sendhil" }]);
  parent.setField("date", "2004");
  await parent.saveTx();
  att.parentID = parent.id; await att.saveTx();

  await ctx.resize(1280, 800);
  ctx.plugin.windows.get(win).show();
  const root = await ctx.waitFor(() => win.document.getElementById("zmc-root")?.shadowRoot?.querySelector("textarea"), "panel rendered").then(() => win.document.getElementById("zmc-root").shadowRoot);
  const $ = (sel) => root.querySelector(sel);
  const $$ = (sel) => [...root.querySelectorAll(sel)];
  const text = () => root.textContent.replace(/\s+/g, " ");
  const running = () => $("button.send")?.classList.contains("send--stop");
  async function say(msg) {
    const ta = $("textarea");
    ta.value = msg; ta.dispatchEvent(new win.Event("input", { bubbles: true }));
    await ctx.waitFor(() => !$("button.send").disabled, "send enabled");
    $("button.send").click();
    await ctx.waitFor(() => running() || text().includes(msg.slice(0, 12)), "turn started");
  }
  const idle = () => ctx.waitFor(() => !running() && $$("button.send").length, "turn finished", 30000);
  const clickText = (re) => { const b = $$("button").find((x) => re.test(x.textContent) || re.test(x.getAttribute("aria-label") || "")); if (!b) throw new Error("no button " + re); b.click(); };

  // 1. empty state in the library view
  ctx.win.ZoteroPane.selectItem(parent.id);
  await ctx.sleep(600);
  await ctx.snapshot("chat-1-empty-library");
  out.emptyHasPrompts = ["Detailed summary", "Short summary", "hypotheses"].every((t) => text().includes(t));
  check(out.emptyHasPrompts, "the pinned prompts are listed on the empty chat: " + text().slice(0, 300));

  // 2. a turn with tool rows
  await say("SCENARIO:tool find papers");
  await idle();
  check(text().includes("done"), "assistant text rendered");
  check(/zotero-cli search foo|Terminal|Bash/.test(text()), "tool step shown: " + text().slice(0, 400));
  await ctx.snapshot("chat-2-tool");

  // 3. permission card, answered
  await say("SCENARIO:permit please");
  await ctx.waitFor(() => /Allow/.test(text()) && /Reject|Deny/.test(text()), "permission card");
  await ctx.snapshot("chat-3-permission");
  clickText(/^Allow( once)?$/);
  await idle();
  check(text().includes("permission:allow"), "agent got the answer: " + text().slice(-200));

  // 4. a rich answer: markdown, math, code, table, citations
  await say(`SCENARIO:rich ATT=${att.key}`);
  await idle();
  await ctx.snapshot("chat-4-rich-library");
  const links = $$("a[href^='zotero:'], [data-href^='zotero:'], .cite");
  out.citationChips = links.length;
  check(links.length >= 2, "two citation chips rendered, found " + links.length);
  check(!$$("script, iframe, [onerror]").length, "no script/iframe/handler elements");
  check(/50% more callbacks/.test(text()) && /Sales/.test(text()), "markdown body and table rendered");
  await ctx.waitFor(() => $$("math").length >= 3, "formulas rendered as MathML");

  // 5. a citation chip opens the page in the reader
  const page3 = links.find((l) => /p\.3/.test(l.textContent));
  check(page3, "a p.3 chip exists: " + links.map((l) => l.textContent).join(" | "));
  page3.click();
  const reader = await ctx.waitFor(() => Zotero.Reader._readers.find((r) => r.itemID === att.id && r._internalReader?._state?.primaryViewStats?.pageIndex === 2), "chip click opened the reader on page 3");
  out.chipOpened = "page " + reader._internalReader._state.primaryViewStats.pageLabel;
  await ctx.sleep(800);
  await ctx.snapshot("chat-5-reader-after-cite");
  out.contextInReader = text().includes("Bertrand and Mullainathan 2004");

  // 6. history reached the disk and replays (the UI batches its writes, so wait for them)
  const host = ctx.plugin.panel().host;
  const count = async () => { const [sv] = await host.sessions(); const evs = sv ? await host.loadEvents(sv.id) : []; return { sessions: sv ? 1 : 0, users: evs.filter((e) => e.t === "user").length, ends: evs.filter((e) => e.t === "turn_end").length, total: evs.length, texts: evs.filter((e) => e.t === "user").map((e) => e.text.slice(0, 18)) }; };
  let c = await count();
  for (let i = 0; i < 40 && !(c.users === 3 && c.ends === 3); i++) { await ctx.sleep(250); c = await count(); }
  out.saved = c;
  check(c.sessions === 1 && c.users === 3 && c.ends === 3, "three turns saved: " + JSON.stringify(c));
  return out;
}
