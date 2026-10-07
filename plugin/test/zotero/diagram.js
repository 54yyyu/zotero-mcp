// Diagrams in the real Gecko panel (run with --mock-agent): ```svg answers render as themed figures inside the shadow DOM,
// a hostile one runs nothing, Copy puts a PNG on the clipboard (stubbed: the user's real clipboard is never touched) and
// Save goes through PanelHost.saveFile with Zotero's file picker (a mock picker registered for the test writes to the
// throwaway data dir). Snapshots in light and dark are for a human to look at. ZMC_DG_SHOTS=<dir> copies them out.
async function main(ctx) {
  const { Zotero, win, Services, IOUtils, PathUtils } = ctx;
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };
  const injected = ctx.plugin.windows.get(win);

  await ctx.resize(1280, 900);
  injected.show();
  const root = await ctx.waitFor(() => win.document.getElementById("zmc-root")?.shadowRoot?.querySelector("textarea") && win.document.getElementById("zmc-root").shadowRoot, "panel rendered");
  const $ = (sel) => root.querySelector(sel);
  const $$ = (sel) => [...root.querySelectorAll(sel)];
  const running = () => $("button.send")?.classList.contains("send--stop");
  async function say(msg) {
    const n = $$(".msg--assistant").length;
    const ta = $("textarea"); ta.value = msg; ta.dispatchEvent(new win.Event("input", { bubbles: true }));
    await ctx.waitFor(() => !$("button.send").disabled, "send enabled"); $("button.send").click();
    await ctx.waitFor(() => $$(".msg--assistant").length > n, "answer started");
    await ctx.waitFor(() => !running() && $$(".msg--assistant")[n]?.dataset.state === "end_turn", "turn finished", 30000);
  }
  const shots = Services.env.get("ZMC_DG_SHOTS");
  const snap = async (name) => {
    await ctx.snapshot(name);
    if (shots) await IOUtils.copy(PathUtils.join(Services.env.get("ZMC_SNAPSHOT_DIR"), `${name}.png`), PathUtils.join(shots, `${name}.png`)).catch(() => {});
  };

  // 0. before any diagram, the diagram code has not run: no stylesheet for it in the shadow root
  check(!$$("style").some((s) => s.textContent.includes(".dg__bar")), "no diagram styles before the first diagram");

  // 1. three diagrams, in the shadow DOM, themed
  await say("SCENARIO:diagrams");
  const figs = await ctx.waitFor(() => $$(".dg .dg__fig svg").length === 3 && $$(".dg .dg__fig svg"), "three diagrams");
  check(!win.document.querySelector(".dg"), "diagrams live in the shadow DOM, not in Zotero's document");
  check($$("style").filter((s) => s.textContent.includes(".dg__bar")).length === 1, "the diagram stylesheet was added once");
  const app = $(".zmc");
  const theme0 = app.dataset.theme;
  app.dataset.theme = "light"; // whatever the machine's theme is
  await ctx.sleep(200);
  const accentRect = figs[0].querySelectorAll("rect")[1];
  const cs = win.getComputedStyle(accentRect);
  out.light = { stroke: cs.stroke, fill: cs.fill };
  check(cs.stroke !== "rgb(0, 0, 0)" && cs.stroke !== "none", "a palette name became a colour: " + cs.stroke);
  check(figs.every((f) => f.getAttribute("role") === "img" && f.getAttribute("aria-label")), "each figure is labelled");
  $(".feed").scrollTop = 0;
  await ctx.sleep(300);
  await snap("diagram-1-light");

  // 2. dark: the same drawing recolours from the panel's variables
  app.dataset.theme = "dark";
  await ctx.sleep(300);
  out.dark = { stroke: win.getComputedStyle(accentRect).stroke };
  check(out.dark.stroke !== out.light.stroke, "dark theme recolours the accent: " + JSON.stringify(out));
  await snap("diagram-2-dark");
  app.dataset.theme = theme0;

  // 3. Copy: a PNG on the clipboard (navigator.clipboard.write is stubbed, so the user's clipboard is untouched)
  const clip = win.navigator.clipboard;
  const realWrite = clip.write;
  let copied = null;
  Object.defineProperty(clip, "write", { configurable: true, value: async (items) => { copied = items; } });
  try {
    $$('.dg button[aria-label="Copy as an image"]')[0].click();
    await ctx.waitFor(() => copied, "clipboard write");
    const item = copied[0];
    check(item.types.includes("image/png"), "a PNG item: " + item.types);
    const bytes = new Uint8Array(await (await item.getType("image/png")).arrayBuffer());
    check(bytes[0] === 0x89 && bytes[1] === 0x50 && bytes.length > 20000, "real PNG bytes: " + bytes.length);
    out.copiedPng = bytes.length;
  } finally {
    if (realWrite) Object.defineProperty(clip, "write", { configurable: true, value: realWrite }); else delete clip.write;
  }

  // 4. Save as PNG / SVG: PanelHost.saveFile with a mock nsIFilePicker that answers a path in the throwaway data dir
  const dir = PathUtils.join(Zotero.DataDirectory.dir, "dg-saves");
  await IOUtils.makeDirectory(dir, { ignoreExisting: true });
  const reg = Components.manager.QueryInterface(Ci.nsIComponentRegistrar);
  const CONTRACT = "@mozilla.org/filepicker;1";
  const realCID = reg.contractIDToCID(CONTRACT);
  const mockCID = Components.ID("{6f1d3b9e-2c4a-4f7e-9d1b-7a0c5e8f2d41}");
  const picked = [];
  const factory = {
    createInstance(iid) {
      const fp = {
        QueryInterface: ChromeUtils.generateQI(["nsIFilePicker"]),
        defaultString: "", defaultExtension: "", filters: [],
        init(_bc, title, mode) { this.title = title; this.mode = mode; },
        appendFilter(t, f) { this.filters.push([t, f]); },
        get file() { return Zotero.File.pathToFile(PathUtils.join(dir, this.defaultString.replace(/\.\w+$/, ""))); }, // no extension: saveFile adds it
        open(cb) { picked.push({ name: this.defaultString, mode: this.mode, filters: this.filters }); cb.done ? cb.done(Ci.nsIFilePicker.returnOK) : cb(Ci.nsIFilePicker.returnOK); },
      };
      return fp.QueryInterface(iid);
    },
  };
  reg.registerFactory(mockCID, "mock file picker", CONTRACT, factory);
  try {
    $$('.dg button[aria-label="Save as PNG"]')[0].click();
    await ctx.waitFor(() => IOUtils.exists(PathUtils.join(dir, "from-question-to-cited-answer.png")), "PNG written");
    $$('.dg button[aria-label="Save as SVG"]')[1].click();
    await ctx.waitFor(() => IOUtils.exists(PathUtils.join(dir, "designs-by-control-and-external-validity.svg")), "SVG written");
  } finally {
    reg.unregisterFactory(mockCID, factory);
    reg.registerFactory(realCID, "", CONTRACT, null);
  }
  check(picked.length === 2 && picked.every((p) => p.mode === Ci.nsIFilePicker.modeSave), "the save dialog was asked twice: " + JSON.stringify(picked));
  const png = await IOUtils.read(PathUtils.join(dir, "from-question-to-cited-answer.png"));
  check(png[0] === 0x89 && png[1] === 0x50 && png[2] === 0x4e && png[3] === 0x47, "the PNG file is a PNG");
  const svgText = await IOUtils.readUTF8(PathUtils.join(dir, "designs-by-control-and-external-validity.svg"));
  check(/^<\?xml[\s\S]*<svg [^>]*xmlns="http:\/\/www.w3.org\/2000\/svg"[^>]*width="360"/.test(svgText) && !svgText.includes("var("), "a standalone SVG with baked colours");
  out.saved = { png: png.length, svg: svgText.length };
  // the exported SVG is a valid XML document Gecko itself can parse
  const parsed = new win.DOMParser().parseFromString(svgText, "image/svg+xml");
  check(!parsed.getElementsByTagName("parsererror").length, "the exported SVG parses as XML");

  // 5. a hostile drawing: nothing runs, nothing external, only the shapes
  win.__pwned = undefined;
  await say("SCENARIO:hostile-svg");
  const evil = await ctx.waitFor(() => $$(".dg .dg__fig svg")[3], "the hostile drawing rendered");
  evil.querySelector("circle")?.dispatchEvent(new win.MouseEvent("click", { bubbles: true }));
  await ctx.sleep(300);
  check(win.__pwned === undefined, "no script ran: " + win.__pwned);
  const tags = [...new Set([evil, ...evil.querySelectorAll("*")].map((e) => e.localName))].sort();
  check(JSON.stringify(tags) === JSON.stringify(["circle", "rect", "svg", "text"]), "only drawing elements: " + tags);
  const attrs = [evil, ...evil.querySelectorAll("*")].flatMap((e) => [...e.attributes].map((a) => `${a.name}=${a.value}`));
  check(!attrs.some((a) => /^on|^class=|javascript:|https?:|evil/i.test(a)), "no handlers or external references: " + attrs.join(" "));
  out.hostile = tags;
  return out;
}
