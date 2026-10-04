// Context chips in a real reader: page, text selection (and clearing it), selected area with image.
async function main(ctx) {
  const { Zotero } = ctx;
  const host = ctx.plugin.panel().host;
  const chips = () => host.currentContext();
  const kinds = () => chips().map((c) => c.kind).join(",");
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg + " | chips=" + JSON.stringify(chips().map((c) => [c.kind, c.label, c.ref.pageIndex]))); };

  const att = await Zotero.Attachments.importFromFile({ file: Zotero.File.pathToFile(ctx.env("ZMC_TEST_PDF")), libraryID: Zotero.Libraries.userLibraryID });
  const parent = new Zotero.Item("journalArticle");
  parent.setField("title", "Test paper for the chat plugin");
  parent.setCreators([{ creatorType: "author", lastName: "Bell", firstName: "A" }]);
  parent.setField("date", "2017");
  await parent.saveTx();
  att.parentID = parent.id;
  await att.saveTx();

  await Zotero.Reader.open(att.id);
  const reader = await ctx.waitFor(() => Zotero.Reader._readers.find((r) => r.itemID === att.id && r._internalReader), "reader");
  await reader._initPromise;
  await ctx.waitFor(() => reader._internalReader._state?.primaryViewStats?.pagesCount, "pdf loaded");
  await ctx.sleep(1000);

  // 1. the reader chip, labelled like Zotero's own citations
  check(chips()[0]?.kind === "reader", "reader chip first");
  out.label = chips()[0].label;
  check(out.label === "Bell 2017", "label is 'Bell 2017', got " + out.label);
  check(chips()[0].ref.pageIndex === 0, "page 0 at start");

  // 2. turning the page changes the chip's page
  reader.navigate({ pageIndex: 2 });
  await ctx.waitFor(() => chips()[0].ref.pageIndex === 2, "chip follows page turn");
  out.page = chips()[0].ref.pageLabel;
  reader.navigate({ pageIndex: 0 });
  await ctx.waitFor(() => chips()[0].ref.pageIndex === 0, "chip returns to page 0");

  // 3. a text selection appears, and disappears when cleared. Synthetic pointer/DOM events cannot start a
  //    selection in Zotero's own PDF view, so this feeds the reader event and the view stat the way the reader does;
  //    that a real drag fires renderTextSelectionPopup is the part only a human check covers.
  const tracker = ctx.plugin.panel().bundle.context;
  const stats = reader._internalReader._state.primaryViewStats;
  reader._internalReader._state.primaryViewStats = { ...stats, canCopy: 1 };
  tracker.onPopup({ reader, params: { annotation: { text: "Assigning a professional destination would increase pay", pageLabel: "1", position: { pageIndex: 0 } } } });
  await ctx.waitFor(() => kinds().includes("selection"), "selection chip");
  const selChip = chips().find((c) => c.kind === "selection");
  check(selChip.text.startsWith("Assigning a professional"), "selection text captured");
  out.selection = selChip.text;
  await ctx.snapshot("context-selection");
  reader._internalReader._state.primaryViewStats = { ...stats, canCopy: 0 };
  await ctx.waitFor(() => !kinds().includes("selection"), "selection chip clears");

  // 4. a selected area: an image chip with the rendered PNG
  const ann = await Zotero.Annotations.saveFromJSON(att, {
    key: Zotero.DataObjectUtilities.generateKey(), type: "image", pageLabel: "1", color: "#ffd400",
    sortIndex: "00000|000500|00100", position: { pageIndex: 0, rects: [[72, 640, 400, 725]] }, comment: "", text: "", tags: [],
  });
  await ctx.sleep(1500);
  reader.navigate({ annotationID: ann.key });
  try { await ctx.waitFor(() => kinds().includes("area"), "area chip"); }
  catch (e) { throw new Error(`${e.message} | chips=${JSON.stringify(chips().map((c) => ({ ...c, image: c.image ? "(png)" : undefined })))} selectedAnnotationIDs=${JSON.stringify(reader._internalReader._state.selectedAnnotationIDs)} annKey=${ann.key} exists=${!!Zotero.Items.getByLibraryAndKey(att.libraryID, ann.key)}`); }
  const area = await ctx.waitFor(() => { const c = chips().find((x) => x.kind === "area"); return c && c.image ? c : null; }, "area image rendered", 20000);
  check(area.image.mime === "image/png" && area.image.data.length > 200, "area has PNG bytes");
  out.area = { label: area.label, bytes: area.image.data.length };
  await ctx.snapshot("context-area");

  // 5. back in the library, the selected item becomes the chip
  ctx.win.Zotero_Tabs.select("zotero-pane");
  await ctx.sleep(500);
  ctx.win.ZoteroPane.selectItem(parent.id);
  await ctx.waitFor(() => chips()[0]?.kind === "item", "library item chip");
  out.libraryChip = chips()[0].label;
  return out;
}
