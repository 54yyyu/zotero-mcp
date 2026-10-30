// Visual contract: the chat should render like system-ui, sans-serif on THIS machine.
// npm run test:fonts — actual Gecko pixels, no stored OS-specific golden images or model calls.
// Product CSS is never changed by this test. A temporary --font override supplies the reference.
async function main(ctx) {
  const { win, Zotero } = ctx;
  const check = (ok, message) => { if (!ok) throw new Error("FAILED: " + message); };
  await ctx.resize(1280, 960);
  ctx.plugin.windows.get(win).show();
  const root = await ctx.waitFor(() => win.document.getElementById("zmc-root")?.shadowRoot?.querySelector("textarea") && win.document.getElementById("zmc-root").shadowRoot, "panel rendered");
  const $ = (s) => root.querySelector(s);
  const app = $(".zmc");
  const host = ctx.plugin.panel().host;
  const originalDoctor = host.doctor;
  const originalDetect = host.runtime.detect;
  const originalFont = app.style.getPropertyValue("--font");
  const originalLang = app.getAttribute("lang");
  const originalTheme = app.dataset.theme;
  const panel = win.document.getElementById("zmc-panel");
  const originalWidth = panel.style.width;
  panel.style.width = "480px";
  const referenceFont = "system-ui, sans-serif";
  const samples = "日本語：漢字とかな\n简体中文：汉字字体\n繁體中文：漢字字體";
  const style = win.document.createElementNS("http://www.w3.org/1999/xhtml", "style");
  // No blinking caret, fading scrollbars, focus decoration, background blur or transitions in pixel comparisons.
  style.textContent = `.zmc *, .zmc *::before, .zmc *::after { animation: none !important; transition: none !important; caret-color: transparent !important; }
    .zmc :focus-visible { outline: none !important; }
    .zmc .feed { scrollbar-width: none !important; }`;
  root.append(style);
  const report = { platform: Zotero.isMac ? "mac" : Zotero.isWin ? "windows" : "linux", zotero: Zotero.version,
    locale: Zotero.locale, referenceFont, actualFont: win.getComputedStyle(app).fontFamily, cases: [] };
  const failures = [];
  const dir = ctx.env("ZMC_SNAPSHOT_DIR");
  const restoreFont = () => originalFont ? app.style.setProperty("--font", originalFont) : app.style.removeProperty("--font");
  const restoreLang = () => originalLang === null ? app.removeAttribute("lang") : app.setAttribute("lang", originalLang);

  async function save(canvas, name) {
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    await IOUtils.write(PathUtils.join(dir, `${name}.png`), new Uint8Array(await blob.arrayBuffer()));
  }
  async function capture(name) {
    await win.document.fonts.ready;
    // drawSnapshot flushes Gecko's painting; avoid rAF, which may pause in a background test window.
    await ctx.sleep(100);
    const r = app.getBoundingClientRect();
    const rect = new win.DOMRect(Math.ceil(r.left), Math.ceil(r.top), Math.floor(r.width) - 1, Math.floor(r.height) - 1);
    check(rect.width > 400 && rect.height > 600, "a visible, unclipped chat panel");
    const bitmap = await win.browsingContext.currentWindowGlobal.drawSnapshot(rect, 2, "rgb(255,255,255)");
    const canvas = win.document.createElementNS("http://www.w3.org/1999/xhtml", "canvas");
    canvas.width = bitmap.width; canvas.height = bitmap.height;
    const g = canvas.getContext("2d");
    g.drawImage(bitmap, 0, 0); bitmap.close();
    await save(canvas, name);
    return { canvas, pixels: g.getImageData(0, 0, canvas.width, canvas.height).data };
  }
  async function diff(actual, expected, name) {
    check(actual.canvas.width === expected.canvas.width && actual.canvas.height === expected.canvas.height, "snapshot dimensions match");
    const canvas = win.document.createElementNS("http://www.w3.org/1999/xhtml", "canvas");
    canvas.width = actual.canvas.width; canvas.height = actual.canvas.height;
    const g = canvas.getContext("2d");
    const image = g.createImageData(canvas.width, canvas.height);
    let changed = 0;
    for (let i = 0; i < actual.pixels.length; i += 4) {
      const different = [0, 1, 2, 3].some((c) => actual.pixels[i + c] !== expected.pixels[i + c]);
      if (different) changed++;
      image.data.set(different ? [220, 0, 100, 255] : [255, 255, 255, 255], i);
    }
    g.putImageData(image, 0, 0);
    await save(canvas, name);
    return changed;
  }

  try {
    // Only the mock bridge is needed for this visual test, regardless of installed agents or CLI setup.
    await ctx.waitFor(() => !["unknown", "checking"].includes($(".stat").dataset.state), "initial status check", 30000);
    host.doctor = async () => [];
    host.runtime.detect = async () => [{ id: host.getSettings().backend, label: "Mock agent", available: true }];
    // Write as an outside settings change so the panel refreshes both its look and backend status.
    Zotero.Prefs.set("extensions.zotero-chat.settings", JSON.stringify({ ...host.getSettings(),
      appearance: { ...host.getSettings().appearance, glass: false, background: "none" } }), true);
    $(".stat").click();
    await ctx.waitFor(() => $(".status__sum")?.textContent === "Everything is set up" && $(".stat").dataset.state === "ok", "mock agent ready");
    $('button[aria-label="Back to the chat"]').click();
    const input = $("textarea");
    input.value = `SCENARIO:fonts\n${samples}`;
    input.dispatchEvent(new win.Event("input", { bubbles: true }));
    await ctx.waitFor(() => !$("button.send").disabled, "send enabled");
    $("button.send").click();
    await ctx.waitFor(() => $('.msg--assistant[data-state="end_turn"] .md strong'), "multilingual mock reply", 30000);
    input.value = samples;
    input.dispatchEvent(new win.Event("input", { bubbles: true }));
    input.blur();
    $(".feed").scrollTop = 0;
    for (const selector of [".ubub__text", ".md p", ".md strong", ".cin"]) {
      const el = $(selector);
      check(el && el.checkVisibility(), selector + " is visible");
      const r = el.getBoundingClientRect(), frame = app.getBoundingClientRect();
      check(r.top >= frame.top && r.bottom <= frame.bottom, selector + " is inside the snapshot");
    }

    for (const theme of ["light", "dark"]) {
      app.dataset.theme = theme;
      for (const lang of [null, "ja", "zh-Hans", "zh-Hant"]) {
        if (lang === null) restoreLang(); else app.setAttribute("lang", lang);
        const name = `fonts-${lang ?? "inherited"}-${theme}`;
        // Render the same real chat twice at the same coordinates, changing only the font selection.
        app.style.setProperty("--font", referenceFont);
        const reference = await capture(name + "-system");
        const repeat = await capture(name + "-repeat");
        const noise = await diff(repeat, reference, name + "-stability-diff");
        check(noise === 0, name + ": reference captures must be stable, changed pixels: " + noise);
        restoreFont();
        const actual = await capture(name + "-actual");
        const changedPixels = await diff(actual, reference, name + "-diff");
        report.cases.push({ lang: lang ?? "inherited", theme, changedPixels, totalPixels: reference.pixels.length / 4 });
        ctx.log(`${name}: ${changedPixels} pixels differ from system fonts`);
        if (changedPixels) failures.push(name);
      }
    }
    // Negative control: the comparison must notice a visibly different font, even when the product already passes.
    app.style.setProperty("--font", referenceFont);
    const control = await capture("fonts-control-system");
    app.style.setProperty("--font", "serif");
    const serif = await capture("fonts-control-serif");
    report.controlChangedPixels = await diff(serif, control, "fonts-control-diff");
    check(report.controlChangedPixels > 0, "the pixel comparison detects a different font");
    await IOUtils.writeUTF8(PathUtils.join(dir, "fonts-report.json"), JSON.stringify(report, null, 2));
    check(failures.length === 0, "chat differs from system-ui, sans-serif: " + failures.join(", ") + "; see fonts-report.json and fonts-*-diff.png");
    return report;
  } finally {
    host.doctor = originalDoctor; host.runtime.detect = originalDetect;
    restoreFont(); restoreLang(); app.dataset.theme = originalTheme;
    panel.style.width = originalWidth;
    style.remove();
  }
}
