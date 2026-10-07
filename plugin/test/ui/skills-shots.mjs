// Screenshots of skills and prompts (the empty state's pinned list, the / menu, the settings card, the add-skill preview), light
// and dark, glass on (or plain).
//   node scripts/preview.mjs && node test/ui/skills-shots.mjs [--widths 300,420] [--looks glass,plain] [--only pins,pins-none] [--browser webkit]
import { launch, openPanel, shotsDir } from "./lib.mjs";
import { mkdirSync } from "node:fs";

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const widths = arg("widths", "300,420").split(",").map(Number);
const looks = arg("looks", "glass").split(",");
const only = arg("only", "").split(",").filter(Boolean);
const browser = await launch(arg("browser", "chromium"));
mkdirSync(shotsDir, { recursive: true });
const pinSkill = (p) => p.evaluate(async () => {
  const h = window.__zmc.host;
  const s = h.getSettings();
  await h.setSettings({ prompts: s.prompts.map((x) => (x.id === "p4" ? { ...x, slot: undefined } : x)), skills: { "annotate-paper": { slot: 4 } } });
  window.__zmc.sim.setSettingsElsewhere({});
});
const toCard = (p) => p.evaluate(() => { const s = window.__zmc.shadow.querySelector("#skills"); window.__zmc.shadow.querySelector(".vw__body").scrollTop = s.offsetTop - 8; });
const STATES = {
  pins: async (p) => { await p.waitForSelector(".pin"); await p.locator(".pin").nth(1).hover(); },
  "pins-skill": async (p) => { await pinSkill(p); await p.waitForSelector(".pin .pin__skill"); },
  "pins-none": async (p) => { await p.evaluate(() => window.__zmc.sim.setSettingsElsewhere({ prompts: window.__zmc.host.getSettings().prompts.map((x) => ({ ...x, slot: undefined })) })); await p.waitForSelector(".pins__none"); },
  slash: async (p) => { await p.locator(".cin").click(); await p.keyboard.type("/"); await p.waitForSelector(".pop--slash .pop__i"); },
  "slash-filter": async (p) => { await p.locator(".cin").click(); await p.keyboard.type("/ann"); await p.waitForSelector(".pop--slash .pop__i"); },
  card: async (p) => { await pinSkill(p); await p.locator('button[aria-label="Settings"]').click(); await p.waitForSelector(".sp"); await toCard(p); await p.waitForTimeout(150); },
  "card-confirm": async (p) => { await p.locator('button[aria-label="Settings"]').click(); await p.waitForSelector(".sp"); await p.locator('button[aria-label="Edit Short summary"]').click(); await p.getByRole("button", { name: "Delete Short summary" }).click(); await p.waitForTimeout(100); await p.evaluate(() => window.__zmc.shadow.querySelector(".sp--confirm").scrollIntoView({ block: "center" })); },
  "card-edit": async (p) => { await p.locator('button[aria-label="Settings"]').click(); await p.waitForSelector(".sp"); await p.locator('button[aria-label="Edit /annotate-paper"]').click(); await p.waitForSelector(".sp__src"); await p.waitForTimeout(100); await p.evaluate(() => window.__zmc.shadow.querySelector(".sp--open").scrollIntoView({ block: "start" })); },
  "preview-files": async (p) => { await p.locator('button[aria-label="Settings"]').click(); await p.waitForSelector(".sp"); await p.getByRole("button", { name: "Add skill…" }).click(); await p.waitForSelector(".imp"); await p.evaluate(() => window.__zmc.shadow.querySelector(".imp__files").scrollIntoView({ block: "center" })); },
  preview: async (p) => { await p.locator('button[aria-label="Settings"]').click(); await p.waitForSelector(".sp"); await p.getByRole("button", { name: "Add skill…" }).click(); await p.waitForSelector(".imp"); await p.evaluate(() => { const s = window.__zmc.shadow.querySelector(".imp"); window.__zmc.shadow.querySelector(".vw__body").scrollTop = s.offsetTop - 60; }); },
};
for (const [name, run] of Object.entries(STATES)) for (const look of looks) for (const width of widths) for (const dark of [false, true]) {
  if (only.length && !only.includes(name)) continue;
  const p = await openPanel(browser, { width, height: 760, dark, params: { look: JSON.stringify({ glass: look === "glass" }) } });
  await run(p);
  await p.waitForTimeout(200);
  await p.screenshot({ path: `${shotsDir}/skills-${name}-${look}-${width}-${dark ? "dark" : "light"}.png` });
  if (p.errors.length) console.log(name, p.errors);
  await p.context().close();
}
await browser.close();
console.log("shots in", shotsDir);
