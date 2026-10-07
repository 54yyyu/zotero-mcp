// Translate in the reader's text selection popup (run with --mock-agent). The popup is Zotero's own: the test hands the PDF
// view a selection range over real glyphs (what a drag ends in), so the view draws the selection, its React code renders the
// popup at the selection and dispatches renderTextSelectionPopup through its own customEvent path. Only the drag itself is
// skipped: synthetic pointer events do not select in Zotero's PDF view, so a real mouse drag stays a human check.
// Checks: the button loads nothing until pressed and obeys the setting; the translation streams into the same popup,
// which widens and stays inside the view; Copy and the language chip; a permission request is refused; the chat and its
// history and folder are untouched; closing the popup stops it. Snapshots in light and dark, plus 2x close-ups.
async function main(ctx) {
  const { Zotero, Services, IOUtils, PathUtils, win } = ctx;
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };
  const injected = ctx.plugin.windows.get(win);
  const SETTINGS = "extensions.zotero-chat.settings";
  const setPref = (patch) => Zotero.Prefs.set(SETTINGS, JSON.stringify({ ...JSON.parse(Zotero.Prefs.get(SETTINGS, true) || "{}"), ...patch }), true);
  check(injected.loaded() === null && ctx.plugin.timing.panelLoadMs === null, "nothing loaded at start");

  const att = await Zotero.Attachments.importFromFile({ file: Zotero.File.pathToFile(ctx.env("ZMC_TEST_PDF")), libraryID: Zotero.Libraries.userLibraryID });
  const parent = new Zotero.Item("journalArticle");
  parent.setField("title", "Translate test");
  await parent.saveTx();
  att.parentID = parent.id;
  await att.saveTx();
  await ctx.resize(1280, 860);
  await Zotero.Reader.open(att.id);
  const reader = await ctx.waitFor(() => Zotero.Reader._readers.find((r) => r.itemID === att.id && r._internalReader?._state?.primaryViewStats?.pagesCount), "reader");
  await ctx.sleep(600);
  const ir = reader._internalReader;
  const doc = reader._iframeWindow.document;
  const view = ir._primaryView;
  await view._ensureBasicPageData(0);
  // The glyphs of the second line of page 1: the rect the selection covers (the text the popup gets is the test's own).
  const chars = view._pdfPages[0].chars;
  const breaks = chars.map((c, i) => (c.lineBreakAfter ? i : -1)).filter((i) => i >= 0);
  const line = chars.slice(breaks[0] + 1, breaks[1] + 1);
  const lineRect = [Math.min(...line.map((c) => c.rect[0])), Math.min(...line.map((c) => c.rect[1])), Math.max(...line.map((c) => c.rect[2])), Math.max(...line.map((c) => c.rect[3]))];
  // Built with the reader window's own JSON, so the view (another compartment) can read the ranges.
  const select = (text) => { view._setSelectionRanges(reader._iframeWindow.wrappedJSObject.JSON.parse(JSON.stringify([{ collapsed: false, anchor: true, head: true, text, sortIndex: "00000|000050|00100", position: { pageIndex: 0, rects: [lineRect] } }]))); view._render(); };
  const deselect = () => { view._setSelectionRanges(); view._render(); }; // what a click elsewhere does
  // React renders the popup a moment later: until then the old popup (and its button) is still there. Wait for this one.
  let rendered = null;
  Zotero.Reader.registerEventListener("renderTextSelectionPopup", (e) => { rendered = String(e.params.annotation.text); }, "translate-test@zotero-chat");
  const fresh = async (text) => { deselect(); rendered = null; select(text); await ctx.waitFor(() => rendered === text && popup(), "the popup for " + text.slice(0, 30)); };
  const popup = () => doc.querySelector(".selection-popup");
  const button = () => [...(popup()?.querySelectorAll("button") ?? [])].find((b) => b.textContent === "Translate");
  const result = () => popup()?.querySelector(".zmc-tr");
  // A timeout says what the popup showed.
  const waitFor = ctx.waitFor;
  ctx.waitFor = (fn, what, ms) => waitFor(fn, what, ms).catch((e) => { throw new Error(`${e.message}; popup: ${popup() ? popup().textContent.slice(0, 300) : "none"}`); });

  // 1. the handler is cheap and synchronous: it builds the button and loads nothing (the preload waits for idle)
  const h0 = win.performance.now();
  const parts = [];
  ctx.plugin.onSelectionPopup({ reader, doc, params: { annotation: { text: "probe" } }, append: (...els) => parts.push(...els) });
  out.handlerMs = win.performance.now() - h0;
  check(parts.length === 1 && ctx.plugin.timing.panelLoadMs === null, "the handler returns having loaded nothing");
  // the button, in the reader's own style; none for a blank selection
  select("   ");
  await ctx.sleep(300);
  check(!button(), "no button for a blank selection");
  deselect();
  const t0 = Date.now();
  select("SCENARIO:translate-sample Les travailleurs affectés à une destination professionnelle gagnaient 8 % de plus.");
  const btn = await ctx.waitFor(button, "the Translate button");
  out.renderMs = Date.now() - t0;
  check(btn.classList.contains("toolbar-button") && btn.classList.contains("wide-button"), "styled like the reader's own: " + btn.className);
  // Once the popup has painted (idle), the translator's session is started: exactly one, however many popups follow.
  await ctx.waitFor(() => ctx.plugin.translate?.translator.stats().started === 1 && ctx.plugin.translate.translator.stats().session, "the preload started one session", 15000);
  out.preloadPanelJsMs = ctx.plugin.timing.panelLoadMs;
  check(injected.loaded() === null, "the panel itself is not loaded by it");

  // 2. the switch, read live as the popup renders
  setPref({ translate: false });
  await fresh("Some text");
  await ctx.waitFor(() => popup(), "popup");
  await ctx.sleep(200);
  check(!button(), "no button when the setting is off");
  setPref({ translate: true });
  await fresh("Some text");
  await ctx.waitFor(button, "the button is back when the setting is on");
  await ctx.sleep(500);
  check(ctx.plugin.translate.translator.stats().started === 1, "more popups start no second session");

  // 3. a chat exists before any translation: it must be untouched afterwards
  injected.show();
  const root = await ctx.waitFor(() => win.document.getElementById("zmc-root")?.shadowRoot?.querySelector("textarea"), "panel rendered").then(() => win.document.getElementById("zmc-root").shadowRoot);
  const ta = root.querySelector("textarea");
  ta.value = "hello from the chat"; ta.dispatchEvent(new win.Event("input", { bubbles: true }));
  await ctx.waitFor(() => !root.querySelector("button.send").disabled, "send enabled");
  root.querySelector("button.send").click();
  await ctx.waitFor(() => /Hello from mock/.test(root.textContent), "the chat answered");
  await ctx.sleep(500);
  const host = ctx.plugin.panel().host;
  const chats = await host.sessions();
  check(chats.length === 1, "one saved chat: " + chats.length);
  const events = JSON.stringify(await host.loadEvents(chats[0].id));
  const folder = ctx.env("ZMC_DEFAULT_CHAT_FOLDER");
  const listing = async () => (await IOUtils.getChildren(folder)).sort().join("\n");
  const folderBefore = await listing();
  const transcript = root.querySelector(".feed")?.textContent ?? root.textContent;

  // 4. pressed: the result streams into the same popup, under the button; the popup widens and stays in view
  await fresh("SCENARIO:translate-sample Les travailleurs affectés à une destination professionnelle gagnaient 8 % de plus.");
  const narrow = (await ctx.waitFor(() => popup(), "popup")).offsetWidth;
  let renders = 0;
  const countRenders = () => renders++;
  Zotero.Reader.registerEventListener("renderTextSelectionPopup", countRenders, "translate-test@zotero-chat");
  const tPress = Date.now();
  (await ctx.waitFor(button, "button")).click();
  check(button().getAttribute("aria-expanded") === "true", "the button says it is open");
  await ctx.waitFor(() => result()?.querySelector(".zmc-tr__sk") || result()?.querySelector(".zmc-tr__text span"), "the shimmer, or the first words");
  const streamed = await ctx.waitFor(() => result()?.querySelector(".zmc-tr__text span"), "streamed pieces arrive", 20000);
  out.warmPressToFirstTextMs = Date.now() - tPress;
  check(out.warmPressToFirstTextMs < 500, `a preloaded press shows text at once (mock agent): ${out.warmPressToFirstTextMs} ms`);
  check(!!streamed, "streamed as it arrives");
  await ctx.waitFor(() => result() && !result().hasAttribute("aria-busy"), "the translation finished", 20000);
  const text = result().querySelector(".zmc-tr__text").textContent;
  out.popupRendersWhileStreaming = renders; // the reader rebuilding its popup mid-answer: the result moves along (revive)
  Zotero.Reader.unregisterEventListener("renderTextSelectionPopup", countRenders);
  out.translation = text;
  check(/^Workers assigned to a professional destination earned 8% more/.test(text), "the translation: " + text);
  check(result().querySelector(".zmc-tr__text").childNodes.length === 1, "finished text is one plain text node");
  const p = popup(), pr = p.getBoundingClientRect(), vb = p.parentElement.getBoundingClientRect();
  out.popup = { narrow, wide: p.offsetWidth, height: p.offsetHeight, left: Math.round(pr.left - vb.left), top: Math.round(pr.top - vb.top), side: /page-popup-(\w+)-center/.exec(p.className)?.[1] };
  check(p.offsetWidth >= 300 && narrow < 260, "the popup widens for the result: " + JSON.stringify(out.popup));
  check(pr.left >= vb.left && pr.right <= vb.right && pr.top >= vb.top && pr.bottom <= vb.bottom, "the popup stays inside the view: " + JSON.stringify(out.popup));
  const sel = ir._state.primaryViewSelectionPopup.rect, anchorX = (sel[0] + sel[2]) / 2, mid = pr.left - vb.left + pr.width / 2;
  out.popup.anchorX = Math.round(anchorX);
  check(Math.abs(mid - anchorX) < 2 || pr.left - vb.left <= 21 || vb.right - pr.right <= 21, `still centred on the selection (or held inside the view): ${mid} vs ${anchorX}`);
  const tr = ctx.plugin.translate.translator;
  check(tr.stats().started === 1 && tr.stats().session.currentModel() === "haiku", "a locked translator session on the fastest model");

  // 5. Copy, and the language chip translating into another language for this popup only
  result().querySelector(".zmc-tr__act").click();
  await ctx.waitFor(() => /Copied/.test(result().querySelector(".zmc-tr__act").textContent), "Copy says Copied");
  const clip = Zotero.Utilities.Internal.getClipboard("text/plain") ?? "";
  check(clip.startsWith("Workers assigned"), "the clipboard has the translation: " + clip.slice(0, 40));
  await shots("1");
  const chip = result().querySelector("select");
  check(chip.value === "en" && chip.options.length === 18, "the chip shows the target, with the 18 languages");
  chip.value = "ja"; chip.dispatchEvent(new reader._iframeWindow.Event("change", { bubbles: true }));
  await ctx.waitFor(() => result() && !result().hasAttribute("aria-busy") && result().querySelector(".zmc-tr__text")?.textContent, "re-translated", 20000);
  out.chipTranslation = result().querySelector(".zmc-tr__text").textContent;
  check(JSON.parse(Zotero.Prefs.get(SETTINGS, true)).translateTo === undefined, "the chip leaves the setting alone");
  check(tr.stats().started === 2, "Japanese is a session of its own");

  // 6. pressed again: the result goes, the popup is its own size again
  button().click();
  await ctx.waitFor(() => !result(), "collapsed");
  check(popup().offsetWidth < 260 && button().getAttribute("aria-expanded") === "false", "back to the reader's own width");

  // 7. a permission request (the agent asking for a tool) is refused; the answer still comes
  await fresh("SCENARIO:translate-permit Bonjour");
  (await ctx.waitFor(button, "button")).click();
  await ctx.waitFor(() => result() && !result().hasAttribute("aria-busy"), "answered", 20000);
  out.permitAnswer = result().querySelector(".zmc-tr__text").textContent;
  check(/^\(permission:reject\) \[Japanese\]|^\(permission:reject\) \[English\]/.test(out.permitAnswer), "the request was refused: " + out.permitAnswer);
  check(tr.stats().refused === 1, "refused once");

  // 8. the selection goes away mid-answer (Esc, a click elsewhere, a new selection all clear the reader's popup): it stops
  await fresh("SCENARIO:translate slowly a long passage that takes a while to stream into the popup, word by word");
  (await ctx.waitFor(button, "button")).click();
  await ctx.waitFor(() => result()?.querySelector(".zmc-tr__text span"), "streaming");
  // Esc with the focus in the popup (after pressing Translate the reader does not take it there): the selection and popup go.
  result().querySelector(".zmc-tr__text").focus();
  result().dispatchEvent(new doc.defaultView.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  await ctx.waitFor(() => !popup() && !view._selectionRanges.length, "Esc cleared the selection and its popup");
  const t1 = Date.now();
  await fresh("SCENARIO:translate after");
  (await ctx.waitFor(button, "button")).click();
  await ctx.waitFor(() => result() && !result().hasAttribute("aria-busy"), "the next one answers at once", 5000);
  out.afterCancelMs = Date.now() - t1;
  check(out.afterCancelMs < 3000, "the closed one was stopped, not waited for: " + out.afterCancelMs);

  // A click elsewhere in the PDF clears the selection (the view's pointerdown does just this): the answer stops with it.
  await fresh("SCENARIO:translate slowly another long passage that takes a while to stream into the popup, word by word");
  button().click();
  await ctx.waitFor(() => result()?.querySelector(".zmc-tr__text span"), "streaming again");
  deselect();
  await ctx.waitFor(() => !popup(), "the popup is gone with the selection");

  // 9. errors are said plainly in the popup
  await fresh("SCENARIO:autherror");
  (await ctx.waitFor(button, "button")).click();
  const err = await ctx.waitFor(() => result()?.querySelector(".zmc-tr__err"), "the error");
  out.error = err.textContent;
  check(/Couldn't translate/.test(out.error) && /Authentication required/.test(out.error) && /Try again/.test(result().textContent), "the error and Try again: " + out.error);

  // 10. the chat, its history and its folder are untouched; the translator ran in its own folder
  const after = await host.sessions();
  check(after.length === 1 && after[0].id === chats[0].id, "still one chat");
  check(JSON.stringify(await host.loadEvents(chats[0].id)) === events, "its history is unchanged");
  check((await listing()) === folderBefore, "the chat folder is unchanged");
  check((root.querySelector(".feed")?.textContent ?? root.textContent) === transcript, "the open chat shows nothing new");
  check(await IOUtils.exists(PathUtils.join(Zotero.Profile.dir, "zotero-chat", "translate")), "the translator's own folder, under the plugin's data dir");

  // 11. a settings change (another model) closes the warm session at once
  setPref({ translateModel: { "claude-code": "sonnet" } });
  await ctx.waitFor(() => tr.stats().session === null, "closed on a settings change", 5000);
  setPref({ translateModel: { "claude-code": "" } });

  // 12. snapshots: light and dark, the window and a 2x close-up of the popup
  async function shots(tag) {
    for (const [theme, name] of [[1, "light"], [0, "dark"]]) {
      Services.prefs.setIntPref("browser.theme.toolbar-theme", theme);
      await ctx.sleep(500);
      await ctx.snapshot(`translate-${tag}-${name}`);
      const r = popup().getBoundingClientRect(), f = reader._iframe.getBoundingClientRect();
      const rect = new win.DOMRect(f.left + r.left - 24, f.top + r.top - 24, r.width + 48, r.height + 48);
      const bitmap = await win.browsingContext.currentWindowGlobal.drawSnapshot(rect, 2, "rgb(255,255,255)");
      const canvas = win.document.createElementNS("http://www.w3.org/1999/xhtml", "canvas");
      canvas.width = bitmap.width; canvas.height = bitmap.height;
      canvas.getContext("2d").drawImage(bitmap, 0, 0);
      const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
      await IOUtils.write(PathUtils.join(ctx.env("ZMC_SNAPSHOT_DIR"), `translate-${tag}-${name}-closeup.png`), new Uint8Array(await blob.arrayBuffer()));
    }
    Services.prefs.clearUserPref("browser.theme.toolbar-theme");
  }
  // A long one, scrolled inside its max height with the fade at the edge, and the shimmer while waiting.
  await fresh("SCENARIO:translate " + "The gap persisted across cities, occupations and firm sizes, and it was not explained by differences in the applicants' stated experience. ".repeat(6));
  (await ctx.waitFor(button, "button")).click();
  await ctx.waitFor(() => result() && !result().hasAttribute("aria-busy"), "long answer", 20000);
  check(result().querySelector(".zmc-tr__scroll").hasAttribute("data-more"), "the fade shows while more is below");
  await shots("2-long");
  await fresh("SCENARIO:translate-wait waiting");
  (await ctx.waitFor(button, "button")).click();
  await shots("3-loading");
  deselect();
  return out;
}
