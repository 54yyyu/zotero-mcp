// Shared by the Playwright scripts: launch a browser, open the preview with a scenario, helpers.
import { pathToFileURL } from "node:url";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

export const PW = "/Users/yiyu/Documents/projects/autorev-video/node_modules/playwright/index.mjs";
export const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
export const shotsDir = join(root, "preview", "shots");
export const pageUrl = (params = {}) => {
  const u = pathToFileURL(join(root, "preview", "dist", "index.html"));
  u.search = new URLSearchParams(params).toString();
  return u.href;
};

export async function launch(name = "chromium") {
  const pw = await import(PW);
  return pw[name].launch();
}

/** A page at `width` x `height` showing the preview; collects console errors in page.errors. */
export async function openPanel(browser, { width = 420, height = 760, dark = false, params = {} } = {}) {
  const ctx = await browser.newContext({ viewport: { width, height }, colorScheme: dark ? "dark" : "light", deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  page.errors = [];
  page.on("pageerror", (e) => page.errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") page.errors.push(m.text()); });
  await page.goto(pageUrl({ theme: dark ? "dark" : "light", speed: 0, ...params }));
  await page.waitForFunction(() => !!window.__zmc);
  await page.waitForTimeout(80);
  return page;
}

/** Locators inside the shadow root (Playwright pierces open shadow DOM with plain css). */
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
