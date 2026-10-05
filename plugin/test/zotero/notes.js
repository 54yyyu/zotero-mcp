// Save as note, for real (run with --mock-agent): an answer and a diagram become Zotero notes in the throwaway library.
// Checks where the note goes (child of the open reader's parent, of a selected item's parent, or standalone in the
// selected collection), what HTML Zotero stores (math as the editor's own nodes, zotero:// links, a table, escaped
// hostile text), the embedded image (an "embedded image" attachment of the note, a real PNG file), that Open selects
// the note, and that Zotero's own note editor renders the image and the math. Snapshots are for a human to look at.
async function main(ctx) {
  const { Zotero, win, IOUtils } = ctx;
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };
  const injected = ctx.plugin.windows.get(win);
  const host = ctx.plugin.panel().host;

  // a paper with its PDF, open in the reader
  const att = await Zotero.Attachments.importFromFile({ file: Zotero.File.pathToFile(ctx.env("ZMC_TEST_PDF")), libraryID: Zotero.Libraries.userLibraryID });
  const parent = new Zotero.Item("journalArticle");
  parent.setField("title", "Test paper for notes");
  await parent.saveTx();
  att.parentID = parent.id;
  await att.saveTx();
  await ctx.resize(1280, 900);
  await Zotero.Reader.open(att.id);
  await ctx.waitFor(() => Zotero.Reader._readers.find((r) => r.itemID === att.id && r._internalReader?._state?.primaryViewStats?.pagesCount), "reader");

  injected.show();
  const root = await ctx.waitFor(() => win.document.getElementById("zmc-root")?.shadowRoot?.querySelector("textarea") && win.document.getElementById("zmc-root").shadowRoot, "panel rendered");
  const $ = (sel) => root.querySelector(sel);
  const $$ = (sel) => [...root.querySelectorAll(sel)];
  async function say(msg) {
    const n = $$(".msg--assistant").length;
    const ta = $("textarea"); ta.value = msg; ta.dispatchEvent(new win.Event("input", { bubbles: true }));
    await ctx.waitFor(() => !$("button.send").disabled, "send enabled"); $("button.send").click();
    await ctx.waitFor(() => $$(".msg--assistant")[n]?.dataset.state === "end_turn" && !$("button.send").classList.contains("send--stop"), "turn finished", 30000);
    return $$(".msg--assistant")[n];
  }
  const notesOf = (item) => item.getNotes().map((id) => Zotero.Items.get(id));
  const embedded = (note) => note.getAttachments().map((id) => Zotero.Items.get(id));

  // 1. an answer with citations, math, a table and code: the Save button in its footer
  const rich = await say(`SCENARIO:rich ATT=${att.key}`);
  rich.querySelector('button[aria-label="Save as a Zotero note"]').click();
  const note1 = await ctx.waitFor(() => notesOf(parent)[0], "a child note of the reader's parent");
  await ctx.waitFor(() => rich.querySelector(".noteline"), "the footer confirms");
  const html1 = note1.getNote();
  out.answerNote = { key: note1.key, title: note1.getNoteTitle(), bytes: html1.length };
  check(note1.parentID === parent.id, "the note is a child of the paper");
  check(note1.getNoteTitle() === `SCENARIO:rich ATT=${att.key}`, "the question is the title: " + note1.getNoteTitle());
  check(/^<div data-schema-version="9">/.test(html1), "math schema: " + html1.slice(0, 60));
  check(html1.includes('<span class="math">$1.50$</span>'), "inline math as the editor's math node");
  check(html1.includes('<span class="math">$\\displaystyle \\text{callback}'), "display math inside a sentence stays inline");
  check(html1.includes(`<a href="zotero://open-pdf/library/items/${att.key}?page=3">Bertrand and Mullainathan 2004, p.3</a>`), "citation as a zotero:// link");
  check(html1.includes('<a href="https://www.zotero.org">the Zotero site</a>') && !/javascript:/.test(html1), "web link kept, javascript: dropped");
  check(/<table>.*<th>Sample<\/th>.*<td>1\.22<\/td>.*<\/table>/.test(html1) && html1.includes("<pre>ratio = 9.65 / 6.45  # 1.50</pre>"), "table and code");
  check(embedded(note1).length === 0, "no drawing, no image");

  // 2. Open selects the note in the library, and Zotero's note editor renders it
  rich.querySelector(".noteline .lnk").click();
  await ctx.waitFor(() => win.Zotero_Tabs.selectedType === "library" && win.ZoteroPane.getSelectedItems()[0]?.id === note1.id, "Open selects the note");
  const editorDoc = () => win.document.getElementById("zotero-note-editor")?._iframe?.contentDocument;
  await ctx.waitFor(() => editorDoc()?.querySelector(".ProseMirror table"), "the note editor shows the note");
  // the editor's math nodes, typeset by its KaTeX: <math-inline class="math-node"><span class="math-render"><span class="katex">
  const math = await ctx.waitFor(() => { const n = [...editorDoc().querySelectorAll(".ProseMirror .math-node")]; return n.length && n.every((e) => e.querySelector(".katex")) ? n.length : 0; }, "the editor typesets the math");
  out.editorMath = math;
  await ctx.snapshot("notes-1-answer-in-editor");

  // 3. a diagram's Add to a note, with the reader in front again: the PNG becomes an embedded image
  win.Zotero_Tabs.select(win.Zotero_Tabs._tabs.find((t) => t.type === "reader").id);
  await ctx.waitFor(() => win.Zotero_Tabs.selectedType === "reader", "reader in front");
  const dg = await say("SCENARIO:diagrams");
  await ctx.waitFor(() => dg.querySelectorAll(".dg .dg__fig svg").length === 3, "three diagrams");
  dg.querySelectorAll('.dg button[aria-label="Add to a note"]')[0].click();
  // images are imported first (they belong to the saved note), the HTML that points at them is written last
  const note2 = await ctx.waitFor(() => notesOf(parent).find((n) => n.id !== note1.id && n.getNote().includes("data-attachment-key")), "the diagram note with its image", 20000);
  await ctx.waitFor(() => dg.querySelector(".dg .noteline"), "the card confirms");
  const html2 = note2.getNote();
  const img = embedded(note2)[0];
  const imgPath = await img.getFilePathAsync();
  const bytes = await IOUtils.read(imgPath);
  out.diagramNote = { title: note2.getNoteTitle(), image: { key: img.key, linkMode: img.attachmentLinkMode, type: img.attachmentContentType, bytes: bytes.length } };
  check(note2.getNoteTitle() === "From question to cited answer", "the drawing's title is the note's: " + note2.getNoteTitle());
  check(img.attachmentLinkMode === Zotero.Attachments.LINK_MODE_EMBEDDED_IMAGE && img.parentID === note2.id, "an embedded-image attachment of the note");
  check(img.attachmentContentType === "image/png" && bytes[0] === 0x89 && bytes[1] === 0x50 && bytes.length > 20000, "a real PNG file: " + bytes.length);
  check(new RegExp(`<img data-attachment-key="${img.key}" width="360" height="132">`).test(html2), "the note points at it: " + html2.slice(0, 200));

  // the whole answer: three drawings, three images, in order
  dg.querySelector('.foot button[aria-label="Save as a Zotero note"]').click();
  const note3 = await ctx.waitFor(() => notesOf(parent).find((n) => n.getNote().split("data-attachment-key").length === 4), "the answer note with three images", 20000);
  const keys = [...note3.getNote().matchAll(/data-attachment-key="([A-Z0-9]{8})"/g)].map((m) => m[1]);
  check(keys.length === 3 && keys.every((k) => embedded(note3).some((a) => a.key === k)), "three images, each referenced: " + keys);

  // the editor shows the embedded image (it resolves data-attachment-key itself)
  dg.querySelector(".foot .noteline .lnk").click();
  await ctx.waitFor(() => win.ZoteroPane.getSelectedItems()[0]?.id === note3.id, "Open selects the diagram note");
  const shown = await ctx.waitFor(() => { const im = editorDoc()?.querySelector(".ProseMirror img"); return im?.complete && im.naturalWidth > 0 ? im.naturalWidth : 0; }, "the note editor renders the image", 20000);
  out.editorImageWidth = shown;
  await ctx.sleep(400);
  await ctx.snapshot("notes-2-diagram-in-editor");

  // 4. the library: a selected child (the PDF) puts the note on its parent; nothing selected, a standalone note in the collection
  win.ZoteroPane.selectItem(att.id);
  await ctx.waitFor(() => win.ZoteroPane.getSelectedItems()[0]?.id === att.id, "the PDF selected");
  const r4 = await host.saveNote({ markdown: "On the parent" });
  check(r4.itemKey === parent.key && Zotero.Items.getByLibraryAndKey(1, r4.noteKey).parentID === parent.id, "a selected attachment's parent gets the note");

  const col = new Zotero.Collection({ name: "Notes test", libraryID: Zotero.Libraries.userLibraryID });
  await col.saveTx();
  await win.ZoteroPane.collectionsView.selectCollection(col.id);
  await ctx.waitFor(() => win.ZoteroPane.getSelectedCollections()[0]?.id === col.id, "collection selected");
  win.ZoteroPane.itemsView.selection.clearSelection();
  await ctx.sleep(300);
  const r5 = await host.saveNote({ title: "Standalone", markdown: "$$\\int_0^1 x\\,dx = \\tfrac12$$\n\n<img src=x onerror=\"window.__pwned=1\"> [x](javascript:alert(1))" });
  const note5 = Zotero.Items.getByLibraryAndKey(1, r5.noteKey);
  const html5 = note5.getNote();
  out.standalone = { uri: r5.uri, html: html5 };
  check(!r5.itemKey && !note5.parentID && note5.getCollections().includes(col.id), "standalone, in the selected collection");
  check(r5.uri === `zotero://select/library/items/${note5.key}`, "uri selects it");
  check(html5.includes('<pre class="math">$$\\int_0^1 x\\,dx = \\tfrac12$$</pre>'), "a display formula is a math block");
  check(html5.includes("&lt;img src=x onerror=") && !/<img|<a /.test(html5), "hostile text stays text: " + html5);
  check(win.__pwned === undefined, "nothing ran");
  return out;
}
