// Context economy in a real Zotero (run with --mock-agent): two turns with the same selection and area send them once,
// a page turn re-sends the where-am-I line, and the real <zotero-context> blocks are measured (chars/4, images by size).
async function main(ctx) {
  const { Zotero, win } = ctx;
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };
  const tokens = (s) => Math.ceil(s.length / 4);
  const imageTokens = (b64) => { const b = atob(b64.slice(0, 32)); const u = (o) => ((b.charCodeAt(o) << 24) | (b.charCodeAt(o + 1) << 16) | (b.charCodeAt(o + 2) << 8) | b.charCodeAt(o + 3)) >>> 0; const w = u(16), h = u(20), s = Math.min(1, 1568 / Math.max(w, h)); return { w, h, tokens: Math.ceil((w * s * h * s) / 750) }; };

  const att = await Zotero.Attachments.importFromFile({ file: Zotero.File.pathToFile(ctx.env("ZMC_TEST_PDF")), libraryID: Zotero.Libraries.userLibraryID });
  const parent = new Zotero.Item("journalArticle");
  parent.setField("title", "Test paper for the chat plugin");
  parent.setCreators([{ creatorType: "author", lastName: "Bell", firstName: "A" }]);
  parent.setField("date", "2017");
  await parent.saveTx();
  att.parentID = parent.id; await att.saveTx();

  await ctx.resize(1280, 800);
  ctx.plugin.windows.get(win).show();
  const root = await ctx.waitFor(() => win.document.getElementById("zmc-root")?.shadowRoot?.querySelector("textarea"), "panel rendered").then(() => win.document.getElementById("zmc-root").shadowRoot);
  const host = ctx.plugin.panel().host;
  const tracker = ctx.plugin.panel().bundle.context;

  await Zotero.Reader.open(att.id);
  const reader = await ctx.waitFor(() => Zotero.Reader._readers.find((r) => r.itemID === att.id && r._internalReader), "reader");
  await reader._initPromise;
  await ctx.waitFor(() => reader._internalReader._state?.primaryViewStats?.pagesCount, "pdf loaded");
  await ctx.sleep(800);

  // A selection of the size describe.ts caps at (1500 chars), fed the way the reader does (see context.js).
  const selection = "Assigning a professional destination would increase pay for workers in the treated firms. ".repeat(20).slice(0, 1500);
  const setSelection = () => {
    reader._internalReader._state.primaryViewStats = { ...reader._internalReader._state.primaryViewStats, canCopy: 1 };
    tracker.onPopup({ reader, params: { annotation: { text: selection, pageLabel: "1", position: { pageIndex: 0 } } } });
  };
  setSelection();
  await ctx.waitFor(() => host.currentContext().some((c) => c.kind === "selection"), "selection chip");

  // Measure the real blocks: reader only, reader + selection, and an area image.
  const readerOnly = host.describeContext(host.currentContext().filter((c) => c.kind === "reader")).text;
  const withSel = host.describeContext(host.currentContext()).text;
  const ann = await Zotero.Annotations.saveFromJSON(att, {
    key: Zotero.DataObjectUtilities.generateKey(), type: "image", pageLabel: "1", color: "#ffd400",
    sortIndex: "00000|000500|00100", position: { pageIndex: 0, rects: [[72, 500, 540, 725]] }, comment: "", text: "", tags: [],
  });
  await ctx.sleep(1200);
  reader.navigate({ annotationID: ann.key });
  const area = await ctx.waitFor(() => { const c = host.currentContext().find((x) => x.kind === "area"); return c && c.image ? c : null; }, "area image rendered", 20000);
  const img = imageTokens(area.image.data);
  setSelection(); // selecting the annotation may have cleared the text selection
  await ctx.waitFor(() => host.currentContext().some((c) => c.kind === "selection"), "selection chip again");
  out.measure = {
    readerLine: { chars: readerOnly.length, tokens: tokens(readerOnly) },
    readerPlusSelection1500: { chars: withSel.length, tokens: tokens(withSel) },
    areaImage: { px: `${img.w}x${img.h}`, base64Chars: area.image.data.length, tokens: img.tokens },
  };

  // Two turns, same focus: the selection text and the image go once.
  const ta = root.querySelector("textarea");
  const running = () => root.querySelector("button.send")?.classList.contains("send--stop");
  async function say(msg, k) {
    ta.value = msg; ta.dispatchEvent(new win.Event("input", { bubbles: true }));
    await ctx.waitFor(() => !root.querySelector("button.send").disabled, "send enabled");
    root.querySelector("button.send").click();
    await ctx.waitFor(() => root.querySelectorAll('.msg--assistant[data-state="end_turn"]').length === k && !running(), `turn ${k} finished`, 30000);
  }
  const echoes = async (n) => {
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
  };
  await say("SCENARIO:echo first", 1);
  await say("SCENARIO:echo second", 2);
  reader.navigate({ pageIndex: 2 });
  await ctx.waitFor(() => host.currentContext()[0]?.ref.pageIndex === 2, "page turn");
  setSelection();
  await ctx.waitFor(() => host.currentContext().some((c) => c.kind === "selection"), "selection after page turn");
  await say("SCENARIO:echo third", 3);
  // A new highlight changes the PDF's annotation counts: the index line goes again (the notifier rebuilds it).
  await Zotero.Annotations.saveFromJSON(att, {
    key: Zotero.DataObjectUtilities.generateKey(), type: "highlight", pageLabel: "3", color: "#ffd400", text: "the gap is almost unchanged",
    sortIndex: "00002|000500|00100", position: { pageIndex: 2, rects: [[72, 600, 300, 612]] }, comment: "", tags: [],
  });
  await ctx.waitFor(() => /2 annotations/.test(host.currentContext()[0]?.text ?? ""), "index counts the new highlight");
  setSelection();
  await say("SCENARIO:echo fourth", 4);
  const [a, b, c, d] = await echoes(4);
  check(a.prompt.includes(selection.slice(0, 200)) && a.images === 1, "turn 1 carries the selection and the image: " + a.prompt.slice(0, 300) + " images=" + a.images);
  check(!b.prompt.includes(selection.slice(0, 200)) && b.images === 0, "turn 2 does not resend them: " + b.prompt.slice(0, 400) + " images=" + b.images);
  check(/Still pointing at, unchanged[^\n]*selected text p\.1 "Assigning a professional destination/.test(b.prompt), "turn 2 names them: " + b.prompt);
  check(/You are pointing at:\n- selected text \(p\.1\)/.test(a.prompt), "turn 1 labels the focus: " + a.prompt.slice(0, 600));
  check(/In this PDF: 1 annotation \(1 area\); on this page: 1 area\. Read them with `zotero-cli annotations list --item-key \w+`/.test(a.prompt), "turn 1 has the annotation index: " + a.prompt.slice(0, 600));
  check(!b.prompt.includes("In this PDF"), "turn 2 does not repeat the index");
  check(/In this PDF: 1 annotation \(1 area\)\. Read/.test(c.prompt), "page 3 has none of them: the index without a page part: " + c.prompt);
  check(/In this PDF: 2 annotations \(1 area, 1 highlight\); on this page: 1 highlight\./.test(d.prompt), "new counts go again: " + d.prompt);
  check(/Reading in the Zotero reader: "Test paper for the chat plugin"[^\n]*on p\.3/.test(c.prompt) && c.images === 0, "a page turn re-sends the reader line, still no image: " + c.prompt);
  const firstBlock = a.prompt.slice(0, a.prompt.indexOf("SCENARIO:echo"));
  const secondBlock = b.prompt.slice(0, b.prompt.indexOf("SCENARIO:echo"));
  out.measure.turn1 = { tokens: tokens(firstBlock) + a.images * img.tokens };
  out.measure.turn2 = { tokens: tokens(secondBlock), text: secondBlock.trim() };
  out.measure.brief = a.brief ? { words: a.brief.split(/\s+/).length, chars: a.brief.length, tokens: tokens(a.brief) } : null;
  await ctx.snapshot("context-economy");

  // "This page" in the + popup: first while a PDF is open, the page as an image chip, described as the whole page.
  const first = (await host.search(""))[0];
  check(first?.kind === "page" && /^This page \(p\. \d+\)$/.test(first.title), "the page hit comes first: " + JSON.stringify(first));
  check(!(await host.search("Bell")).some((x) => x.kind === "page") && (await host.search("page"))[0]?.kind === "page", "only for an empty query or 'page'");
  root.querySelector('button[aria-label="Add a source"]').click();
  const row = await ctx.waitFor(() => [...root.querySelectorAll(".pop__i")].find((li) => li.textContent.startsWith(first.title)), "the page row");
  row.dispatchEvent(new win.MouseEvent("mousedown", { bubbles: true, cancelable: true }));
  const card = await ctx.waitFor(() => [...root.querySelectorAll(".area")].find((el) => /Page \d+/.test(el.textContent) && el.querySelector("img.area__img")), "the page chip");
  check(/Go to Page/.test(card.textContent), "its action says Page");
  const pageImg = card.querySelector("img.area__img");
  await ctx.waitFor(() => pageImg.complete && pageImg.naturalWidth > 0, "the page image decodes");
  out.pageImage = [pageImg.naturalWidth, pageImg.naturalHeight];
  check(Math.max(pageImg.naturalWidth, pageImg.naturalHeight) <= 1568 && pageImg.naturalWidth > 200, "a real page image, at most 1568 px: " + out.pageImage);
  const described = host.describeContext([await host.chipFor(first)]);
  check(/the whole page p\.\d+: the image is attached/.test(described.text) && described.images.length === 1 && described.images[0].mime === "image/png", "described as the whole page: " + described.text);
  return out;
}
