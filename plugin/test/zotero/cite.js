// Sentence-precise citation jumps: a zotero://open-pdf link with &quote= opens the page and the reader flashes the
// quoted words for about 2 s (its own highlightedPosition, no annotation). A quote that is not there is a page jump.
async function main(ctx) {
  const { Zotero } = ctx;
  const host = ctx.plugin.panel().host;
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };

  const att = await Zotero.Attachments.importFromFile({ file: Zotero.File.pathToFile(ctx.env("ZMC_TEST_PDF")), libraryID: Zotero.Libraries.userLibraryID });
  const link = (page, quote) => `zotero://open-pdf/library/items/${att.key}?page=${page}&quote=${encodeURIComponent(quote)}`;
  const readers = () => Zotero.Reader._readers.filter((r) => r.itemID === att.id);
  await ctx.resize(1280, 900);

  // 1. reader not open yet: a quote that runs across a line break, cited on page 1
  const QUOTE = "would increase pay in both groups, but the causal effects are about the same size";
  await host.open(link(1, QUOTE));
  const reader = await ctx.waitFor(() => readers()[0]?._internalReader && readers()[0], "reader");
  const view = () => reader._internalReader._lastView;
  const hl = await ctx.waitFor(() => view()?._highlightedPosition, "highlighted position", 10000);
  const t0 = Date.now();
  const rects = Array.from(hl.rects, (r) => Array.from(r));
  out.firstHighlight = { pageIndex: hl.pageIndex, rects };
  check(hl.pageIndex === 0, "highlight on page 1, got " + hl.pageIndex);
  check(rects.length === 2, "one rect per line (2 lines), got " + rects.length);

  // the rects cover exactly the quoted characters (found here independently, from the raw glyphs)
  const chars = Array.from(view()._pdfPages[0].chars);
  out.charShape = Object.keys(chars[0]).sort();
  const glyphs = chars.map((c) => c.c).join("");
  const want = QUOTE.replace(/[^A-Za-z]/g, "");
  const letters = []; chars.forEach((c, i) => { if (/[A-Za-z]/.test(c.c)) letters.push(i); });
  const at = letters.map((i) => chars[i].c).join("").indexOf(want);
  check(at >= 0, "the quote is on page 1: " + glyphs.slice(0, 200));
  const inQuote = letters.slice(at, at + want.length);
  const centre = (c) => [(c.rect[0] + c.rect[2]) / 2, (c.rect[1] + c.rect[3]) / 2];
  const covered = (c) => { const [x, y] = centre(c); return rects.some((r) => x >= r[0] && x <= r[2] && y >= r[1] && y <= r[3]); };
  check(inQuote.every((i) => covered(chars[i])), "every quoted character is inside a rect");
  const before = letters[at - 1], after = letters[at + want.length];
  check(!covered(chars[before]) && !covered(chars[after]), `the words around it are not: "${chars[before].c}" and "${chars[after].c}"`);
  check(Zotero.Items.get(att.id).getAnnotations().length === 0, "no annotation was made");
  await ctx.sleep(250);
  await ctx.snapshot("cite-1-highlighted");

  // it fades by itself after about 2 s
  await ctx.waitFor(() => !view()._highlightedPosition, "highlight cleared", 5000);
  out.clearedAfterMs = Date.now() - t0;
  check(out.clearedAfterMs > 1200 && out.clearedAfterMs < 3500, "cleared about 2 s later: " + out.clearedAfterMs);
  await ctx.snapshot("cite-2-faded");

  // 2. reader already open: no new tab; a quote cited on the wrong page is found on the next one
  await host.open(link(1, "Test paper for the chat plugin, page 2"));
  const hl2 = await ctx.waitFor(() => view()._highlightedPosition, "second highlight", 10000);
  check(readers().length === 1, "the open reader was reused");
  check(hl2.pageIndex === 1 && hl2.rects.length === 1, "found on the neighbouring page 2: " + JSON.stringify({ p: hl2.pageIndex, n: hl2.rects.length }));
  out.neighbour = { pageIndex: hl2.pageIndex };
  await ctx.waitFor(() => !view()._highlightedPosition, "second highlight cleared", 5000);

  // 3. a quote that is not in the paper: a plain page jump, no highlight, no error
  await host.open(link(3, "a sentence the agent paraphrased that appears nowhere in this paper"));
  await ctx.waitFor(() => reader._internalReader._state.primaryViewStats.pageIndex === 2, "jumped to page 3");
  await ctx.sleep(600);
  check(!view()._highlightedPosition, "no highlight for a quote that is not there");
  out.missing = "page jump";

  // 4. the same miss on a fresh reader still lands on the page
  ctx.win.Zotero_Tabs.close(reader.tabID);
  await ctx.waitFor(() => !readers().length, "reader closed");
  await host.open(link(4, "nothing like this is in the paper, not even a little bit"));
  const r2 = await ctx.waitFor(() => readers()[0]?._internalReader && readers()[0], "reader reopened");
  await ctx.waitFor(() => r2._internalReader._state?.primaryViewStats?.pageIndex === 3, "fresh reader at page 4");
  out.missingFresh = "page jump";
  return out;
}
