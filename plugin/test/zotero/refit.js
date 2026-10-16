// Opening the panel narrows the reader: a fit-mode refits by itself, and a fixed zoom that would now spill out of the
// view becomes page-width (a fixed zoom that still fits is left alone).
async function main(ctx) {
  const { Zotero } = ctx;
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };
  const att = await Zotero.Attachments.importFromFile({ file: Zotero.File.pathToFile(ctx.env("ZMC_TEST_PDF")), libraryID: Zotero.Libraries.userLibraryID });
  await ctx.resize(1280, 900);
  await Zotero.Reader.open(att.id);
  const reader = await ctx.waitFor(() => Zotero.Reader._readers.find((r) => r.itemID === att.id && r._internalReader?._primaryView?._iframeWindow?.PDFViewerApplication?.pdfViewer?.currentScaleValue) , "reader", 15000);
  const viewer = () => reader._internalReader._primaryView._iframeWindow.PDFViewerApplication.pdfViewer;
  const inj = ctx.plugin.windows.get(ctx.win);
  const overflows = () => viewer().container.scrollWidth > viewer().container.clientWidth + 1;
  await ctx.sleep(1000);

  // 1. a fixed zoom wider than the view once the panel is open
  viewer().currentScaleValue = "1.2"; await ctx.sleep(500);
  check(!overflows(), "1.2 fits the full-width view");
  inj.show();
  await ctx.waitFor(() => viewer().currentScaleValue === "page-width", "refit to page-width", 5000);
  await ctx.sleep(300);
  check(!overflows(), "no horizontal overflow after the refit");
  out.refit = { scale: viewer().currentScale };

  // 2. a fixed zoom that still fits stays as the user set it
  inj.hide(); await ctx.sleep(800);
  viewer().currentScaleValue = "0.8"; await ctx.sleep(500);
  inj.show(); await ctx.sleep(1000);
  check(viewer().currentScaleValue === "0.8", "a zoom that fits is left alone, got " + viewer().currentScaleValue);
  out.kept = true;
  return out;
}
