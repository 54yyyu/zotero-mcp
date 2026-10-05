// "Ask in chat" in the reader's text selection popup. The listener is registered at startup, so this checks that
// rendering the button loads nothing (panel.js stays unread until it is pressed), and that pressing it opens the panel
// with the selection as a chip and the composer focused. A real mouse selection cannot be synthesized in Zotero's PDF
// view, so the popup event is dispatched the way the reader dispatches it (Zotero.Reader._dispatchEvent).
async function main(ctx) {
  const { Zotero, win } = ctx;
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };
  const injected = ctx.plugin.windows.get(win);
  check(injected.loaded() === null && ctx.plugin.timing.panelLoadMs === null, "the panel is not loaded at start");

  const att = await Zotero.Attachments.importFromFile({ file: Zotero.File.pathToFile(ctx.env("ZMC_TEST_PDF")), libraryID: Zotero.Libraries.userLibraryID });
  const parent = new Zotero.Item("journalArticle");
  parent.setField("title", "Ask test");
  await parent.saveTx();
  att.parentID = parent.id;
  await att.saveTx();
  await ctx.resize(1280, 900);
  await Zotero.Reader.open(att.id);
  const reader = await ctx.waitFor(() => Zotero.Reader._readers.find((r) => r.itemID === att.id && r._internalReader?._state?.primaryViewStats?.pagesCount), "reader");

  const popup = (text) => {
    const doc = reader._iframeWindow.document;
    const box = doc.createElement("div");
    Zotero.Reader._dispatchEvent({ type: "renderTextSelectionPopup", reader, doc, params: { annotation: { text, pageLabel: "2", position: { pageIndex: 1 } } }, append: (...els) => box.append(...els) });
    return box;
  };

  // 1. no text, no button; with text, one quiet button in the reader's own style, and still nothing loaded
  check(popup("   ").children.length === 0, "no button without a selection");
  const t0 = Date.now();
  const box = popup("Assigning a professional destination would increase pay");
  out.renderMs = Date.now() - t0;
  const btn = box.querySelector("button");
  check(box.children.length === 1 && btn?.textContent === "Ask in chat" && btn.classList.contains("toolbar-button"), "the button: " + box.innerHTML);
  check(injected.loaded() === null && ctx.plugin.timing.panelLoadMs === null, "showing the button loads nothing");

  // 2. pressed: the panel opens with the selection as a chip, the composer has the focus
  btn.click();
  const root = await ctx.waitFor(() => win.document.getElementById("zmc-root")?.shadowRoot?.querySelector("textarea") && win.document.getElementById("zmc-root").shadowRoot, "panel opened");
  check(injected.isOpen(), "the panel is open");
  const selChips = () => [...root.querySelectorAll(".cchips .chip")].filter((c) => /^Text Selection/.test(c.getAttribute("title") || ""));
  const chip = await ctx.waitFor(() => selChips()[0], "the selection chip");
  out.panelLoadMs = ctx.plugin.timing.panelLoadMs;
  out.chipTitle = chip.getAttribute("title");
  check(out.chipTitle === "Text Selection: Assigning a professional destination would increase pay", "the chip carries the text: " + out.chipTitle);
  await ctx.waitFor(() => root.activeElement === root.querySelector("textarea"), "the composer has the focus");
  check(selChips().length === 1, "one selection chip");
  await ctx.sleep(300);
  await ctx.snapshot("ask-1-chip-in-composer");

  // 3. with the panel loaded, a new selection's button replaces the chip's text: still one chip
  popup("A second selection about callbacks").querySelector("button").click();
  await ctx.waitFor(() => selChips()[0]?.getAttribute("title") === "Text Selection: A second selection about callbacks", "second selection");
  check(selChips().length === 1, "still one selection chip");

  // 4. "This page" in the + popup: first while a PDF is open, the page as an image chip, described as the whole page
  const host = ctx.plugin.panel().host;
  const first = (await host.search(""))[0];
  check(first?.kind === "page" && first.title === "This page (p. 1)", "the page hit comes first: " + JSON.stringify(first));
  check(!(await host.search("Ask")).some((x) => x.kind === "page") && (await host.search("page"))[0]?.kind === "page", "only for an empty query or 'page'");
  root.querySelector('button[aria-label="Add a source"]').click();
  const row = await ctx.waitFor(() => [...root.querySelectorAll(".pop__i")].find((li) => li.textContent.startsWith("This page (p. 1)")), "the page row");
  row.dispatchEvent(new win.MouseEvent("mousedown", { bubbles: true, cancelable: true }));
  const card = await ctx.waitFor(() => [...root.querySelectorAll(".area")].find((c) => /Page 1/.test(c.textContent) && c.querySelector("img.area__img")), "the page chip");
  check(/Go to Page/.test(card.textContent), "its action says Page");
  const img = card.querySelector("img.area__img");
  await ctx.waitFor(() => img.complete && img.naturalWidth > 0, "the page image decodes");
  out.pageImage = [img.naturalWidth, img.naturalHeight];
  check(Math.max(img.naturalWidth, img.naturalHeight) <= 1568 && img.naturalWidth > 200, "a real page image, at most 1568 px: " + out.pageImage);
  const chipObj = await host.chipFor(first);
  const d = host.describeContext([chipObj]);
  out.pageLine = d.text.split("\n").find((l) => l.includes("whole page"));
  check(/the whole page p\.1: the image is attached/.test(d.text) && d.images.length === 1 && d.images[0].mime === "image/png", "described as the whole page: " + d.text);
  await ctx.sleep(300);
  await ctx.snapshot("ask-2-page-chip");
  return out;
}
