// The look in real Gecko (run with --mock-agent): glass is on by default and the floating surfaces really blur (computed
// backdrop-filter), the accent and a background preset apply live from the settings card, a picture goes through the real
// host (read, downscaled to 1600 px, saved as a file beside the prefs, not in them), everything survives the panel being
// mounted again from scratch, and Remove deletes the file. Snapshots, light and dark, are for a human to look at.
async function main(ctx) {
  const { Zotero, win } = ctx;
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };

  await ctx.resize(1280, 900);
  const mount = async () => {
    ctx.plugin.windows.get(win).show();
    return ctx.waitFor(() => win.document.getElementById("zmc-root")?.shadowRoot?.querySelector("textarea") && win.document.getElementById("zmc-root").shadowRoot, "panel rendered");
  };
  let root = await mount();
  const host = ctx.plugin.panel().host;
  const bundle = ctx.plugin.panel().bundle;
  const $ = (sel) => root.querySelector(sel);
  const $$ = (sel) => [...root.querySelectorAll(sel)];
  const zmc = () => $(".zmc");
  const cssVar = (name) => win.getComputedStyle(zmc()).getPropertyValue(name).trim();
  const byLabel = (label) => $$("button, input").find((b) => b.getAttribute("aria-label") === label);
  const click = (el, what) => { if (!el) throw new Error("no " + what); el.click(); };
  const pref = () => JSON.parse(Zotero.Prefs.get("extensions.zotero-chat.settings", true) || "{}");
  const say = async (msg) => {
    const ta = $("textarea"); ta.value = msg; ta.dispatchEvent(new win.Event("input", { bubbles: true }));
    await ctx.waitFor(() => !$("button.send").disabled, "send enabled"); $("button.send").click();
    await ctx.waitFor(() => $("button.send")?.classList.contains("send--stop"), "turn started", 10000).catch(() => {});
    await ctx.waitFor(() => !$("button.send").classList.contains("send--stop"), "turn finished", 30000);
    await ctx.sleep(400); // the Stop button's colour eases back
  };

  // Zotero's own theme pref flips prefers-color-scheme for the window and the panel follows; forced only if it does not.
  const theme = async (t) => {
    Services.prefs.setIntPref("browser.theme.toolbar-theme", t === "dark" ? 0 : 1);
    await ctx.waitFor(() => zmc().dataset.theme === t, "the panel follows Zotero to " + t, 5000).catch(() => { out.themeForced = true; zmc().dataset.theme = t; });
    await ctx.sleep(300);
  };
  await theme("light");

  // 1. the default: glass on, mono, and Gecko really computes the blur on the composer's frost layer
  check(zmc().dataset.glass === "on" && zmc().dataset.bg === "none", "glass on, no background by default: " + JSON.stringify(zmc().dataset));
  check(cssVar("--accent") === "#16181d" && cssVar("--link") === "#2563c9", "mono ink is the default accent, links stay blue: " + cssVar("--accent") + " " + cssVar("--link"));
  const frost = win.getComputedStyle($(".composer"), "::before").backdropFilter;
  out.backdropFilter = frost;
  check(/blur\(18px\)/.test(frost), "Gecko computes backdrop-filter on the composer: " + frost);
  check(win.CSS.supports("backdrop-filter", "blur(1px)"), "backdrop-filter supported (else the fallback applies)");
  check(win.getComputedStyle(zmc(), "::before").display === "none" && win.getComputedStyle(zmc(), "::after").display === "none", "Plain by default: no glow, no backdrop layer");
  await say("SCENARIO:rich ATT=ABCD1234");
  await ctx.snapshot("look-1-glass-answer-light");

  // 2. the settings card, driven for real: an accent swatch, the custom colour input, a background preset
  click(byLabel("Settings"), "settings button");
  const card = await ctx.waitFor(() => root.querySelector('section.sec[aria-label="Appearance"]'), "the Appearance card");
  out.cards = $$("section.sec").map((s) => s.getAttribute("aria-label"));
  check(out.cards.join() === "Agent,Appearance,Context,Chat,Custom prompts,Chat folder,Data,About", "the cards, in order: " + out.cards);
  click(byLabel("Blue"), "Blue swatch");
  await ctx.waitFor(() => host.getSettings().appearance.accent === "#2563c9", "Blue is saved");
  check(cssVar("--accent") === "#2563c9", "the accent variable changed at once: " + cssVar("--accent"));
  check(zmc().dataset.accent === "custom", "a chosen accent marks the root data-accent=custom");
  const color = byLabel("Custom colour");
  check(color && color.type === "color", "Gecko keeps <input type=color> (not a text fallback): " + color?.type);
  color.value = "#8a2be2";
  color.dispatchEvent(new win.Event("input", { bubbles: true }));
  check(cssVar("--accent") === "#8a2be2", "the picker previews while it is open");
  color.dispatchEvent(new win.Event("change", { bubbles: true }));
  await ctx.waitFor(() => host.getSettings().appearance.accent === "#8a2be2", "the custom colour is saved");
  await ctx.waitFor(() => byLabel("Accent colour as hex"), "the hex field shows for a custom colour");
  click(byLabel("Red"), "Red swatch");
  await ctx.waitFor(() => host.getSettings().appearance.accent === "#cc2936" && zmc().dataset.accent === "custom", "our red is a chosen colour");
  click(byLabel("Mono"), "Mono swatch");
  await ctx.waitFor(() => host.getSettings().appearance.accent === "", "back to the default");
  check(!zmc().hasAttribute("data-accent"), "mono has no data-accent");
  click(byLabel("Dawn"), "Dawn tile");
  await ctx.waitFor(() => zmc().dataset.bg === "dawn", "a preset paints the backdrop");
  check(/gradient/.test(win.getComputedStyle(zmc(), "::before").backgroundImage), "the preset is a gradient on ::before");
  out.accentAndPreset = "ok";

  // 3. a picture through the real host: a 2400x1500 PNG, picked (the picker stubbed), downscaled, stored as a file
  const c = win.document.createElementNS("http://www.w3.org/1999/xhtml", "canvas");
  c.width = 2400; c.height = 1500;
  const g = c.getContext("2d");
  const grad = g.createLinearGradient(0, 0, 0, 1500);
  grad.addColorStop(0, "#2b3a67"); grad.addColorStop(0.6, "#e07a5f"); grad.addColorStop(1, "#f2cc8f");
  g.fillStyle = grad; g.fillRect(0, 0, 2400, 1500);
  g.fillStyle = "#fbe7c6"; g.beginPath(); g.arc(1700, 820, 160, 0, 7); g.fill();
  g.fillStyle = "#3d405b"; g.beginPath(); g.moveTo(0, 1100); g.quadraticCurveTo(700, 850, 1300, 1080); g.quadraticCurveTo(1900, 1250, 2400, 1000); g.lineTo(2400, 1500); g.lineTo(0, 1500); g.fill();
  const png = await new Promise((r) => c.toBlob(r, "image/png"));
  const src = PathUtils.join(Zotero.getTempDirectory().path, "evening-hills.png");
  await IOUtils.write(src, new Uint8Array(await png.arrayBuffer()));
  host.chooseImage = () => bundle.importImage(src);
  click($$("button").find((b) => b.textContent.trim() === "Choose an image…"), "Choose an image…");
  await ctx.waitFor(() => zmc().dataset.bg === "image", "the picture is the background");
  const file = PathUtils.join(Zotero.Profile.dir, "zotero-chat", "background.jpg");
  check(await IOUtils.exists(file), "the picture is a file in the plugin's data dir");
  const bmp = await win.createImageBitmap(new win.Blob([await IOUtils.read(file)]));
  out.image = { width: bmp.width, height: bmp.height, bytes: (await IOUtils.stat(file)).size, prefChars: Zotero.Prefs.get("extensions.zotero-chat.settings", true).length };
  check(bmp.width === 1600 && bmp.height === 1000, "downscaled to 1600 px on the long edge: " + JSON.stringify(out.image));
  check(pref().appearance.image === "evening-hills.png" && out.image.prefChars < 2000, "the prefs hold the name only: " + out.image.prefChars + " chars");
  check(/^url\("data:image\/jpeg;base64,/.test(win.getComputedStyle(zmc(), "::before").backgroundImage), "Gecko loads the data: picture from the stylesheet");
  root.querySelector('section.sec[aria-label="Appearance"]').scrollIntoView({ block: "start" });
  await ctx.sleep(300);
  await ctx.snapshot("look-2-settings-picture-light");

  // 4. mounted again from scratch: a new host reads the prefs and the file
  ctx.plugin.onMainWindowUnload(win);
  ctx.plugin.onMainWindowLoad(win);
  root = await mount();
  await theme("light");
  await ctx.waitFor(() => zmc()?.dataset.bg === "image" && cssVar("--bg-img").startsWith('url("data:image/jpeg'), "the picture is back after a remount");
  check(zmc().dataset.glass === "on" && ctx.plugin.panel().host.getSettings().appearance.image === "evening-hills.png", "the look persisted");
  out.remount = "ok";
  await say("SCENARIO:rich ATT=ABCD1234");
  await ctx.snapshot("look-3-picture-answer-light");

  // 5. dark
  await theme("dark");
  check(cssVar("--accent") === "#e6e8eb", "mono in dark is near-white: " + cssVar("--accent"));
  await ctx.snapshot("look-4-picture-answer-dark");
  const host2 = ctx.plugin.panel().host;
  click($$("button").find((b) => b.getAttribute("aria-label") === "Settings"), "settings button");
  await ctx.waitFor(() => root.querySelector('section.sec[aria-label="Appearance"]'), "the Appearance card");
  click($$("button").find((b) => b.getAttribute("aria-label") === "Plain"), "Plain tile");
  await ctx.waitFor(() => zmc().dataset.bg === "none" && host2.getSettings().appearance.image === "evening-hills.png", "plain glass; the picture is kept for later");
  root.querySelector('section.sec[aria-label="Appearance"]').scrollIntoView({ block: "start" });
  await ctx.sleep(300);
  await ctx.snapshot("look-5-settings-glass-dark");

  // 6. Remove: the file goes, the prefs forget it, the backdrop is plain again
  click($$("button").find((b) => b.getAttribute("aria-label") === "Remove the background picture"), "Remove");
  await ctx.waitFor(() => host2.getSettings().appearance.image === "", "the picture is forgotten");
  check(!(await IOUtils.exists(file)), "the file is deleted");
  check(zmc().dataset.bg === "none" && !cssVar("--bg-img"), "no picture on the panel");
  Services.prefs.clearUserPref("browser.theme.toolbar-theme");
  out.remove = "ok";
  return out;
}
