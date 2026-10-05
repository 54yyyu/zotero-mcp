// Screenshot every state of the panel at 320 / 420 / 700 px, light and dark, into preview/shots/.
//   node scripts/preview.mjs && node test/ui/shots.mjs [--only name,name] [--widths 420] [--themes light] [--browser chromium|webkit]
import { launch, openPanel, shotsDir } from "./lib.mjs";
import { mkdirSync } from "node:fs";

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const only = arg("only", "")?.split(",").filter(Boolean);
const widths = arg("widths", "320,420,700").split(",").map(Number);
const themes = arg("themes", "light,dark").split(",");
const browserName = arg("browser", "chromium");

const send = async (p, text) => { await p.locator(".cin").fill(text); await p.locator(".cin").press("Enter"); };
const done = (p, state = "end_turn") => p.waitForSelector(`.msg--assistant[data-state="${state}"]`, { timeout: 8000 });
const settle = (p, ms = 250) => p.waitForTimeout(ms);
const toTop = (p) => p.evaluate(() => { const f = window.__zmc.shadow.querySelector(".feed"); f.scrollTop = 0; });
const openSettings = async (p) => { await p.locator('button[aria-label="Settings"]').click(); await p.waitForSelector(".field"); await settle(p, 200); };
const pickBackend = (p, name) => p.locator(".seg__opt", { hasText: new RegExp(`^${name}$`) }).first().click();
const toSection = async (p, title) => { await p.evaluate((t) => { const s = [...window.__zmc.shadow.querySelectorAll("section.sec")].find((x) => x.getAttribute("aria-label") === t); const b = window.__zmc.shadow.querySelector(".vw__body"); b.scrollTop = s.offsetTop - 8; }, title); await settle(p, 150); };
const scrollBy = (p, y) => p.evaluate((y) => { const f = window.__zmc.shadow.querySelector(".feed"); f.scrollTop = y; }, y);

/** name -> { params?, run(page) }. The screenshot is taken after run(). */
const STATES = {
  "empty": { run: async () => {} },
  "empty-no-context": { params: { ctx: "none", nohistory: 1 }, run: async () => {} },
  "chip-selection": { params: { ctx: "selection" }, run: async (p) => { await p.locator(".cin").fill("Has anyone looked at how this differs by r"); } },
  "chip-area": { params: { ctx: "area" }, run: async (p) => { await p.locator(".cin").fill("Explain equation 7"); } },
  "chip-pinned-hover": { run: async (p) => { await p.locator(".chip__pin").click(); await p.locator(".chip__main").hover(); } },
  "at-popup": { run: async (p) => { await p.locator(".cin").click(); await p.keyboard.type("Compare with @pag"); await settle(p); } },
  "at-popup-empty-query": { run: async (p) => { await p.locator(".cin").click(); await p.keyboard.type("@"); await settle(p); } },
  "at-no-matches": { run: async (p) => { await p.locator(".cin").click(); await p.keyboard.type("@zzzz"); await settle(p); } },
  "plus-popup": { run: async (p) => { await p.locator('button[aria-label="Add a source"]').click(); await settle(p); await p.keyboard.type("lund"); await settle(p); } },
  "search-error": { run: async (p) => { await p.evaluate(() => { window.__zmc.sim.searchFails = true; }); await p.locator('button[aria-label="Add a source"]').click(); await settle(p, 400); } },
  "chips-added": { run: async (p) => { await p.locator(".cin").click(); await p.keyboard.type("@pager"); await settle(p); await p.keyboard.press("Enter"); await settle(p); await p.keyboard.type("@05_"); await settle(p); await p.keyboard.press("Enter"); await settle(p); await p.keyboard.type("What do these say about stigma?"); } },
  "dragover": { run: async (p) => { await p.evaluate(() => { const d = new DataTransfer(); d.setData("text/plain", "zmc-item:PAGER003"); window.__zmc.shadow.querySelector(".composer").dispatchEvent(new DragEvent("dragenter", { dataTransfer: d, bubbles: true, cancelable: true })); }); await settle(p, 150); } },
  "dropped": { run: async (p) => { await p.evaluate(() => { const d = new DataTransfer(); d.setData("text/plain", "zmc-item:PAGER003\nzmc-item:LUND2021"); window.__zmc.shadow.querySelector(".composer").dispatchEvent(new DragEvent("drop", { dataTransfer: d, bubbles: true, cancelable: true })); }); await settle(p, 250); } },
  "cite-wrap": { run: async (p) => { await p.evaluate(() => { window.__zmc.sim.nextAnswer = "Prose that goes on for a while so the line is nearly full before it reaches the chip [Pager et al. 2009, p.8](zotero://open-pdf/library/items/PAGER009?page=8). Next [Quillian 2017](zotero://select/library/items/QUIL2017), then more words [A 2020](zotero://select/library/items/AAAAAAAA); done."; }); await send(p, "x"); await done(p); await settle(p, 300); } },
  "model-menu": { run: async (p) => { await p.locator(".pick--model").click(); await settle(p, 600); } },
  "mode-menu": { run: async (p) => { await p.locator(".pick--mode").click(); await settle(p, 600); } },
  "thinking": { params: { speed: 30 }, run: async (p) => { await send(p, "How does this compare?"); await settle(p, 700); } },
  "streaming": { params: { speed: 6 }, run: async (p) => { await send(p, "How does this compare?"); await p.waitForSelector(".md--streaming .cite", { timeout: 8000 }); await settle(p, 200); } },
  "answer": { run: async (p) => { await send(p, "How does this compare?"); await done(p); await settle(p, 500); } },
  "answer-top": { run: async (p) => { await send(p, "How does this compare?"); await done(p); await settle(p, 500); await toTop(p); await settle(p, 100); } },
  "answer-sources": { run: async (p) => { await send(p, "How does this compare?"); await done(p); await settle(p, 400); await p.locator(".foot__src").click(); await settle(p, 150); } },
  "answer-step-open": { run: async (p) => { await send(p, "How does this compare?"); await done(p); await settle(p, 300); await toTop(p); await p.locator(".step__row").first().click(); await settle(p, 150); } },
  "math": { run: async (p) => { await send(p, "show math"); await done(p); await settle(p, 700); } },
  "diagram": { run: async (p) => { await send(p, "draw it"); await done(p); await settle(p, 300); await p.locator(".dg").first().hover(); } },
  "plan": { params: { speed: 25 }, run: async (p) => { await send(p, "plan the comparison"); await p.waitForSelector(".plan__i--in_progress", { timeout: 6000 }); await settle(p, 120); } },
  "permission": { params: { speed: 20 }, run: async (p) => { await send(p, "add a note to this paper"); await p.waitForSelector(".perm .perm__opts"); await settle(p, 300); } },
  "permission-allowed": { params: { speed: 5 }, run: async (p) => { await send(p, "add a note to this paper"); await p.waitForSelector(".perm .perm__opts"); await p.getByRole("button", { name: "Allow once" }).click(); await done(p); await settle(p, 400); } },
  "permission-denied": { params: { speed: 5 }, run: async (p) => { await send(p, "add a note to this paper"); await p.waitForSelector(".perm .perm__opts"); await p.getByRole("button", { name: "Deny" }).click(); await done(p); await settle(p, 400); } },
  "error-notice": { run: async (p) => { await send(p, "this will error"); await done(p, "error"); await settle(p, 300); } },
  "stopped": { params: { speed: 15 }, run: async (p) => { await send(p, "slow answer"); await p.waitForSelector(".md--streaming"); await settle(p, 300); await p.locator(".cin").press("Escape"); await done(p, "cancelled"); await settle(p, 300); } },
  "start-failed": { run: async (p) => { await p.evaluate(() => { window.__zmc.sim.startFails = "prompt"; }); await send(p, "hello"); await p.waitForSelector(".notice--error"); await settle(p, 300); } },
  "history": { run: async (p) => { await p.locator('button[aria-label="History"]').click(); await settle(p, 400); } },
  "history-confirm": { run: async (p) => { await p.locator('button[aria-label="History"]').click(); await settle(p, 300); await p.locator(".hrow__del").first().click(); await settle(p, 150); } },
  "history-empty": { params: { nohistory: 1 }, run: async (p) => { await p.locator('button[aria-label="History"]').click(); await settle(p, 400); } },
  "history-resumed": { run: async (p) => { await p.locator('button[aria-label="History"]').click(); await settle(p, 300); await p.locator(".hrow__main").first().click(); await settle(p, 400); } },
  "settings": { run: async (p) => { await openSettings(p); } },
  "settings-loading": { params: { catalogDelay: 4000 }, run: async (p) => { await p.locator('button[aria-label="Settings"]').click(); await settle(p, 300); } },
  "settings-error": { run: async (p) => { await openSettings(p); await pickBackend(p, "Codex"); await settle(p, 400); } },
  "settings-codex": { run: async (p) => { await p.evaluate(() => { window.__zmc.sim.statuses[1].available = true; }); await openSettings(p); await pickBackend(p, "Codex"); await settle(p, 400); } },
  "settings-pi": { run: async (p) => { await openSettings(p); await pickBackend(p, "pi"); await settle(p, 400); } },
  "settings-apikey": { run: async (p) => { await openSettings(p); await p.getByRole("radio", { name: "API key" }).click(); await settle(p, 300); await toSection(p, "Sign-in"); } },
  "settings-context": { run: async (p) => { await openSettings(p); await toSection(p, "Context"); } },
  "settings-chat": { run: async (p) => { await openSettings(p); await toSection(p, "Chat"); } },
  "settings-prompts": { run: async (p) => { await openSettings(p); await toSection(p, "Custom prompts"); } },
  "settings-folder": { run: async (p) => { await openSettings(p); await p.getByRole("button", { name: "Choose…" }).click(); await settle(p, 250); await toSection(p, "Chat folder"); } },
  "history-copied": { run: async (p) => { await p.evaluate(() => Object.defineProperty(navigator, "clipboard", { value: { writeText: async () => {} }, configurable: true })); await p.locator('button[aria-label="History"]').click(); await settle(p, 300); await p.locator(".hrow", { hasText: "Which papers cite" }).hover(); await p.locator(".hrow", { hasText: "Which papers cite" }).getByRole("button", { name: /Copy terminal/ }).click(); await settle(p, 200); } },
  "settings-data": { run: async (p) => { await openSettings(p); await toSection(p, "Data"); await p.getByRole("button", { name: "Clear all history" }).click(); await settle(p, 150); } },
  "settings-shortcuts": { run: async (p) => { await openSettings(p); await toSection(p, "Shortcuts"); } },
  "status-ok": { run: async (p) => { await p.locator(".stat").click(); await settle(p, 500); } },
  "status-problems": { params: { doctor: "many" }, run: async (p) => { await p.locator(".stat").click(); await settle(p, 600); } },
  "status-fix-running": { params: { doctor: "cli", speed: 200 }, run: async (p) => { await p.locator(".stat").click(); await settle(p, 700); await p.getByRole("button", { name: "Install zotero-cli" }).click(); await settle(p, 330); } },
  "setup-zotero-api": { params: { doctor: "zotero-api" }, run: async (p) => { await settle(p, 400); } },
  "setup-cli": { params: { doctor: "cli" }, run: async (p) => { await settle(p, 400); } },
  "setup-backend": { params: { doctor: "backend" }, run: async (p) => { await settle(p, 400); } },
  "setup-node": { params: { doctor: "node" }, run: async (p) => { await settle(p, 400); } },
  "stress-jump": { params: { stress: 1 }, run: async (p) => { await p.locator('button[aria-label="History"]').click(); await settle(p, 300); await p.locator(".hrow__main", { hasText: "Long literature" }).click(); await settle(p, 1200); await scrollBy(p, 3000); await settle(p, 300); } },
};

mkdirSync(shotsDir, { recursive: true });
const browser = await launch(browserName);
const names = Object.keys(STATES).filter((n) => !only.length || only.includes(n));
let bad = 0;
for (const name of names) {
  for (const theme of themes) {
    for (const w of widths) {
      const st = STATES[name];
      const page = await openPanel(browser, { width: w, height: w === 700 ? 760 : 740, dark: theme === "dark", params: st.params });
      try {
        await st.run(page);
        await page.waitForTimeout(100);
        await page.screenshot({ path: `${shotsDir}/${browserName === "chromium" ? "" : browserName + "-"}${name}-${w}-${theme}.png` });
      } catch (e) {
        bad++;
        console.error(`FAILED ${name} ${w} ${theme}: ${String(e).split("\n")[0]}`);
        await page.screenshot({ path: `${shotsDir}/FAIL-${name}-${w}-${theme}.png` }).catch(() => {});
      }
      if (page.errors.length) { bad++; console.error(`page errors in ${name} ${w} ${theme}:`, page.errors); }
      await page.context().close();
    }
  }
}
await browser.close();
console.log(`${names.length} states, ${bad} problems`);
process.exit(bad ? 1 : 0);
