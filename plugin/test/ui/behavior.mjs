// Behaviour checks in a real browser against the fake host: node test/ui/behavior.mjs [--browser webkit]
// (build first: node scripts/preview.mjs). Exits non-zero on the first failure.
import assert from "node:assert/strict";
import { launch, openPanel } from "./lib.mjs";
import { XSS_CORPUS } from "./corpus.ts";

const browserName = process.argv.includes("--browser") ? process.argv[process.argv.indexOf("--browser") + 1] : "chromium";
const browser = await launch(browserName);
let n = 0;
const test = async (name, fn, opts) => {
  const page = await openPanel(browser, opts);
  try { await fn(page); console.log("PASS", name); n++; }
  catch (e) { console.error("FAIL", name, "\n", e); await page.screenshot({ path: `preview/shots/FAIL-behavior-${name.replace(/\W+/g, "-")}.png` }); process.exitCode = 1; }
  assert.deepEqual(page.errors, [], `page errors in ${name}`);
  await page.context().close();
};
const send = async (p, text) => { await p.locator(".cin").fill(text); await p.locator(".cin").press("Enter"); };
const done = (p, state = "end_turn") => p.waitForSelector(`.msg--assistant[data-state="${state}"]`, { timeout: 8000 });
const sim = (p, fn, arg) => p.evaluate(fn, arg);

await test("enter sends, shift+enter adds a line, the draft clears", async (p) => {
  await p.locator(".cin").fill("line one");
  await p.locator(".cin").press("Shift+Enter");
  await p.keyboard.type("line two");
  assert.equal(await p.locator(".cin").inputValue(), "line one\nline two");
  await p.locator(".cin").press("Enter");
  await done(p);
  assert.equal(await p.locator(".cin").inputValue(), "");
  assert.equal(await p.locator(".ubub__text").first().innerText(), "line one\nline two");
  const prompt = await sim(p, () => window.__zmc.sim.prompts[0].text);
  assert.match(prompt, /<zotero-context>[\s\S]*Bertrand and Mullainathan 2004[\s\S]*<\/zotero-context>\n\nline one\nline two/);
});

await test("escape stops a running answer; the stopped turn says so", async (p) => {
  await sim(p, () => { window.__zmc.sim.speed = 20; });
  await send(p, "slow answer");
  await p.waitForSelector(".md--streaming");
  await p.locator(".cin").press("Escape");
  await done(p, "cancelled");
  assert.match(await p.locator(".stopnote").innerText(), /Stopped/);
  assert.equal(await p.locator(".send--stop").count(), 0);
});

await test("citation chips open the page; sources list each item once", async (p) => {
  await send(p, "compare");
  await done(p);
  assert.equal(await p.locator(".cite").count(), 4);
  await p.locator(".cite").nth(1).click();
  const opened = await sim(p, () => window.__zmc.sim.opened);
  assert.deepEqual(opened, ["zotero://open-pdf/library/items/PAGER009?page=9"]);
  await p.locator(".foot__src").click();
  assert.equal(await p.locator(".source").count(), 3);
  assert.match(await p.locator(".source").nth(1).innerText(), /Pager et al\. 2009\s*pp\. 9, 12/);
  await p.locator("a", { hasText: "the OSF page" }).click();
  assert.equal((await sim(p, () => window.__zmc.sim.opened)).at(-1), "https://osf.io/example", "http links go through host.open, the window never navigates");
});

await test("@ search adds a chip and removes the @text; keyboard only", async (p) => {
  await p.locator(".cin").click();
  await p.keyboard.type("see @lund");
  await p.waitForSelector('.pop__i:has-text("Lundberg")');
  assert.equal(await p.locator(".pop__i").count(), 1);
  await p.keyboard.press("Enter");
  await p.waitForSelector(".cchips .chip:not(.chip--auto)");
  assert.equal(await p.locator(".cin").inputValue(), "see ");
  assert.equal(await p.locator(".pop").isVisible(), false);
  assert.match(await p.locator(".cchips .chip:not(.chip--auto)").innerText(), /Lundberg 2021/);
  await p.locator(".cchips .chip:not(.chip--auto) .chip__x").click();
  assert.equal(await p.locator(".cchips .chip:not(.chip--auto)").count(), 0);
});

await test("escape closes the @ popup without stopping or sending", async (p) => {
  await p.locator(".cin").click();
  await p.keyboard.type("@pa");
  await p.waitForSelector(".pop__i");
  await p.keyboard.press("Escape");
  assert.equal(await p.locator(".pop").isVisible(), false);
  assert.equal(await p.locator(".cin").inputValue(), "@pa");
});

await test("a failing search shows the error inside the popup", async (p) => {
  await sim(p, () => { window.__zmc.sim.searchFails = true; });
  await p.locator('button[aria-label="Add a source"]').click();
  await p.waitForSelector(".pop__status--bad");
  assert.match(await p.locator(".pop__status").innerText(), /local API/);
});

await test("pinning keeps a chip when the focus moves; dismissing hides it until it returns", async (p) => {
  await p.locator(".chip__pin").click();
  await sim(p, () => window.__zmc.sim.setContext("none"));
  assert.match(await p.locator(".cchips").innerText(), /Bertrand/);
  await p.locator(".chip__pin").click(); // unpin
  assert.equal(await p.locator(".cchips .chip").count(), 0);
  await sim(p, () => window.__zmc.sim.setContext("item"));
  await p.locator(".chip").hover();
  await p.locator(".chip__x").click();
  assert.equal(await p.locator(".cchips .chip").count(), 0);
  await sim(p, () => window.__zmc.sim.setContext("none"));
  await sim(p, () => window.__zmc.sim.setContext("item"));
  assert.equal(await p.locator(".cchips .chip").count(), 1, "back in focus: shown again");
});

await test("selected area card: thumbnail, go to annotation, remove", async (p) => {
  await sim(p, () => window.__zmc.sim.setContext("area"));
  assert.ok(await p.locator(".area__img").isVisible());
  await p.getByRole("button", { name: "Go to Annotation" }).click();
  assert.equal((await sim(p, () => window.__zmc.sim.opened)).at(-1).annotationKey, "ANNAREA1");
  await send(p, "explain");
  await done(p);
  const images = await sim(p, () => window.__zmc.sim.prompts[0].images?.length);
  assert.equal(images, 1, "the area image goes with the question");
  await p.getByRole("button", { name: "Remove" }).click();
  assert.equal(await p.locator(".area").count(), 0);
});

await test("permission: answering resolves the card and the saved log replays the same", async (p) => {
  await sim(p, () => { window.__zmc.sim.speed = 5; });
  await send(p, "add a note");
  await p.waitForSelector(".perm__opts");
  await p.getByRole("button", { name: "Always allow" }).click();
  await done(p);
  assert.match(await p.locator(".perm--done").innerText(), /Always allow/);
  const saved = await sim(p, () => window.__zmc.sim.writes.filter((w) => w.ev.t === "permission").map((w) => w.ev.resolved));
  assert.deepEqual(saved, [undefined, "always"]);
});

await test("saving merges text deltas: far fewer writes than events", async (p) => {
  await send(p, "compare");
  await done(p);
  await p.waitForTimeout(800);
  const w = await sim(p, () => window.__zmc.sim.writes.map((x) => x.ev.t));
  assert.ok(w.filter((t) => t === "text").length <= 3, `text writes: ${w.filter((t) => t === "text").length}`);
  assert.ok(w.includes("user") && w.includes("turn_end"));
});

await test("history: resume, continue, delete; new chat clears", async (p) => {
  await p.locator('button[aria-label="History"]').click();
  await p.waitForSelector(".hrow__main");
  await p.locator(".hrow__main").first().click();
  await p.waitForSelector(".msg--assistant");
  assert.match(await p.locator(".ubub__text").first().innerText(), /Compare callback ratios/);
  await send(p, "and again");
  await done(p);
  assert.equal(await p.locator(".msg--user").count(), 2);
  await p.locator('button[aria-label="History"]').click();
  await p.locator(".hrow__del").first().click();
  await p.getByRole("button", { name: "Delete", exact: true }).click();
  await p.locator('button[aria-label="New chat"]').click();
  assert.equal(await p.locator(".msg").count(), 0);
  assert.ok(await p.locator(".empty").isVisible());
});

await test("settings: api key saved, removed; prompts edited and shortcut runs it", async (p) => {
  await p.locator('button[aria-label="Settings"]').click();
  await p.getByRole("radio", { name: "API key" }).click();
  await p.locator('input[type="password"]').fill("sk-test-123");
  await p.getByRole("button", { name: "Save", exact: true }).click();
  await p.waitForSelector(".keyrow__ok");
  assert.equal(await sim(p, () => window.__zmc.sim.keys["claude-code"]), "sk-test-123");
  assert.match(await p.locator(".stat__t").innerText(), /API key/);
  await p.getByRole("button", { name: "Remove" }).click();
  await p.waitForFunction(() => !window.__zmc.sim.keys["claude-code"]);
  await p.locator('input[aria-label="Prompt 1 title"]').fill("Brief summary");
  await p.locator('input[aria-label="Prompt 1 title"]').press("Tab");
  await p.waitForFunction(() => window.__zmc.host.getSettings().prompts[0].title === "Brief summary");
  await p.keyboard.press("Escape");
  await sim(p, () => window.__zmc.panel.runPrompt(2));
  await done(p);
  assert.match(await p.locator(".ubub__text").first().innerText(), /five sentences/);
});

await test("status: fix streams its output, then the check passes", async (p) => {
  await p.locator(".stat").click();
  await p.getByRole("button", { name: "Install zotero-cli" }).click();
  await p.waitForFunction(() => [...window.__zmc.shadow.querySelectorAll(".check .fixlog")].some((l) => !l.hidden && l.textContent.includes("Installed 1 executable")));
  await p.waitForFunction(() => window.__zmc.shadow.querySelector(".status__sum")?.textContent.includes("Everything is set up"));
}, { params: { doctor: "cli", speed: 10 } });

await test("zotero API down: designed card with restart guidance, chat still allowed", async (p) => {
  const card = p.locator(".setup");
  await card.waitFor();
  assert.match(await card.innerText(), /Restart Zotero/);
  assert.equal(await p.locator(".send").isDisabled(), true);
  await p.locator(".cin").fill("hi");
  assert.equal(await p.locator(".send").isDisabled(), false);
}, { params: { doctor: "zotero-api" } });

await test("no backend: Send is blocked and says why", async (p) => {
  await p.locator(".setup").waitFor();
  await p.locator(".cin").fill("hi");
  assert.equal(await p.locator(".send").isDisabled(), true);
  assert.match(await p.locator(".setup").innerText(), /not logged in/);
}, { params: { doctor: "backend" } });

await test("a dead bridge shows an error notice with a fix route, then a retry starts a fresh one", async (p) => {
  await sim(p, () => { window.__zmc.sim.startFails = "prompt"; });
  await send(p, "hello");
  await p.waitForSelector(".notice--error");
  await p.getByRole("button", { name: "Check setup" }).click();
  assert.ok(await p.locator(".status__top").isVisible());
  await p.keyboard.press("Escape");
  await send(p, "hello again");
  await done(p);
}, {});

await test("mode and model pickers apply and persist", async (p) => {
  await p.locator(".pick--mode").click();
  await p.getByRole("menuitemradio", { name: /Plan/ }).click();
  await p.waitForFunction(() => /Plan/.test(window.__zmc.shadow.querySelector(".pick--mode").textContent));
  assert.equal(await sim(p, () => window.__zmc.host.getSettings().mode["claude-code"]), "plan");
  await p.locator(".pick--model").click();
  await p.getByRole("menuitemradio", { name: /Haiku/ }).click();
  await p.waitForFunction(() => /Haiku/.test(window.__zmc.shadow.querySelector(".pick--model").textContent));
});

const openSettings = async (p) => { await p.locator('button[aria-label="Settings"]').click(); await p.waitForSelector(".field"); };
const toggle = (p, label) => p.getByRole("switch", { name: label }).click();
const settings = (p) => p.evaluate(() => JSON.parse(JSON.stringify(window.__zmc.host.getSettings())));

await test("settings: switches save at once; Enter-to-send changes the composer's keys", async (p) => {
  await openSettings(p);
  await toggle(p, /Follow what I'm reading/);
  assert.equal((await settings(p)).followFocus, false);
  await toggle(p, /Press Enter to send/);
  assert.equal((await settings(p)).enterToSend, false);
  await p.getByRole("button", { name: "Back to the chat" }).click();
  await p.locator(".cin").fill("first");
  await p.locator(".cin").press("Enter");
  assert.equal(await p.locator(".msg").count(), 0, "Enter no longer sends");
  assert.equal(await p.locator(".cin").inputValue(), "first\n");
  assert.match(await p.locator(".send").getAttribute("title"), /Enter/);
  await p.keyboard.type("second");
  await p.locator(".cin").press("Control+Enter");
  await done(p);
  assert.equal(await p.locator(".ubub__text").first().innerText(), "first\nsecond");
});

await test("settings: hiding the thinking and expanding tool steps apply to what is already shown", async (p) => {
  await send(p, "compare");
  await done(p);
  assert.equal(await p.locator(".thought").count(), 1);
  assert.equal(await p.locator(".step--open").count(), 0);
  await openSettings(p);
  await toggle(p, /Show the agent's thinking/);
  await toggle(p, /Expand tool steps/);
  await p.getByRole("button", { name: "Back to the chat" }).click();
  assert.equal(await p.locator(".thought").count(), 0);
  assert.equal(await p.locator(".step--open").count(), 2);
  await p.locator(".step__detail").first().scrollIntoViewIfNeeded();
  assert.ok(await p.locator(".step__detail .sd__pre").first().isVisible());
  await p.locator(".step__row").first().click();
  assert.equal(await p.locator(".step--open").count(), 1, "a step the user closes stays closed");
});

await test("pickers: effort, model and mode act on the live session and save into that backend only", async (p) => {
  await p.locator(".cin").click();
  await p.waitForFunction(() => window.__zmc.sim.closed === 0 && /Sonnet/.test(window.__zmc.shadow.querySelector(".pick--model").textContent));
  await p.locator(".pick--effort").click();
  await p.getByRole("menuitemradio", { name: "High", exact: true }).click();
  await p.waitForFunction(() => window.__zmc.host.getSettings().effort["claude-code"] === "high");
  assert.match(await p.locator(".pick--effort").innerText(), /High/);
  await p.locator(".pick--mode").click();
  await p.getByRole("menuitemradio", { name: /Plan/ }).click();
  await p.waitForFunction(() => window.__zmc.host.getSettings().mode["claude-code"] === "plan");
  const s = await settings(p);
  assert.deepEqual([s.effort.codex, s.effort.pi, s.mode.codex, s.model["claude-code"]], ["", "", "", ""]);
  await openSettings(p);
  await p.locator(".seg__opt", { hasText: /^pi$/ }).first().click();
  await p.waitForFunction(() => window.__zmc.shadow.querySelectorAll(".radio").length === 0 && window.__zmc.shadow.querySelectorAll(".field").length === 2);
  await p.keyboard.press("Escape");
  assert.equal(await p.locator(".pick--mode").isVisible(), false, "pi has no permission modes");
  assert.match(await p.locator(".pick--effort").innerText(), /Off/, "pi's own effort default");
});

await test("settings: the agent section shows loading, then the catalog; a failing backend shows an error with Try again", async (p) => {
  await p.locator('button[aria-label="Settings"]').click();
  await p.waitForSelector(".sk-group");
  await p.waitForSelector(".field", { timeout: 6000 });
  assert.equal(await p.locator(".sk-group").count(), 0);
  assert.equal(await p.locator(".radio").count(), 5);
  await p.locator(".seg__opt", { hasText: /^Codex$/ }).first().click();
  await p.waitForSelector(".inlineerr");
  assert.match(await p.locator(".inlineerr").innerText(), /not installed/);
  await sim(p, () => { window.__zmc.sim.statuses[1].available = true; });
  await p.getByRole("button", { name: "Try again" }).click();
  await p.waitForSelector(".radio");
  assert.deepEqual(await p.locator(".radio__t").allInnerTexts(), ["Read only", "Workspace write", "Agent", "Full access"]);
  await p.locator(".radio", { hasText: "Read only" }).click();
  await p.waitForFunction(() => window.__zmc.host.getSettings().mode.codex === "read-only");
  assert.equal((await settings(p)).mode["claude-code"], "");
  await p.locator(".field", { hasText: "Model" }).locator("select").selectOption("gpt-5");
  await p.waitForFunction(() => window.__zmc.host.getSettings().model.codex === "gpt-5");
}, { params: { catalogDelay: 1200 } });

await test("settings: clear history and reset settings ask first, then reach the host", async (p) => {
  await openSettings(p);
  await toggle(p, /Expand tool steps/);
  await p.getByRole("button", { name: "Clear all history" }).click();
  await p.getByRole("button", { name: "Cancel" }).click();
  assert.equal(await sim(p, () => window.__zmc.sim.data.cleared), 0, "cancel does nothing");
  await p.getByRole("button", { name: "Clear all history" }).click();
  await p.getByRole("button", { name: "Delete all" }).click();
  await p.waitForFunction(() => window.__zmc.sim.data.cleared === 1);
  await p.getByRole("button", { name: "Open", exact: true }).click();
  assert.equal(await sim(p, () => window.__zmc.sim.data.revealed), 1);
  await p.getByRole("button", { name: "Reset settings" }).click();
  await p.getByRole("button", { name: "Reset", exact: true }).click();
  await p.waitForFunction(() => window.__zmc.sim.data.resets === 1);
  assert.equal((await settings(p)).expandTools, false);
  assert.match(await p.locator(".about").innerText(), /Zotero chat 0\.1\.0/);
  await p.keyboard.press("Escape");
  await p.locator('button[aria-label="History"]').click();
  await p.waitForSelector(".vempty");
});

const dragData = (p, text) => p.evaluateHandle((t) => { const d = new DataTransfer(); d.setData("text/plain", t); return d; }, text);
const fire = (p, type, dt) => p.evaluate(([type, dt]) => window.__zmc.shadow.querySelector(".composer").dispatchEvent(new DragEvent(type, { dataTransfer: dt, bubbles: true, cancelable: true })), [type, dt]);

await test("drop on the composer: dragover state without layout shift, chips added pinned via host.dropChips, junk ignored", async (p) => {
  const box = () => p.locator(".composer").boundingBox();
  const before = await box();
  const dt = await dragData(p, "zmc-item:PAGER003\nzmc-item:LUND2021");
  await fire(p, "dragenter", dt);
  await fire(p, "dragover", dt);
  assert.ok(await p.locator(".composer--drop .dropveil").isVisible());
  assert.match(await p.locator(".dropveil").innerText(), /Drop to add/);
  assert.deepEqual(await box(), before, "no layout shift while dragging");
  await fire(p, "dragleave", dt);
  assert.equal(await p.locator(".composer--drop").count(), 0, "dragleave cleans up");
  await fire(p, "dragenter", dt);
  await fire(p, "drop", dt);
  await p.waitForSelector(".chip--pinned:not(.chip--auto)");
  assert.equal(await p.locator(".composer--drop").count(), 0);
  assert.equal(await p.locator(".cchips .chip:not(.chip--auto)").count(), 2);
  assert.equal(await p.locator(".chip--pinned:not(.chip--auto)").count(), 2);
  const junk = await dragData(p, "just some text");
  await fire(p, "drop", junk);
  await p.waitForTimeout(100);
  assert.equal(await p.locator(".cchips .chip:not(.chip--auto)").count(), 2, "an empty result adds nothing");
  await send(p, "what do these say?");
  await done(p);
  assert.equal(await p.locator(".cchips .chip:not(.chip--auto)").count(), 2, "pinned drops stay after a send");
  assert.match(await sim(p, () => window.__zmc.sim.prompts[0].text), /Pager 2003[\s\S]*Lundberg 2021/);
  await p.locator(".cchips .chip:not(.chip--auto) .chip__pin").first().click();
  await p.locator('button[aria-label="New chat"]').click();
  assert.equal(await p.locator(".cchips .chip:not(.chip--auto)").count(), 0, "a new chat starts clean");
});

await test("a citation chip keeps its trailing punctuation: no line starts with a lone . , ; :", async (p) => {
  await sim(p, () => { window.__zmc.sim.nextAnswer = "Prose that goes on for a while so the line is nearly full before it reaches the chip [Pager et al. 2009, p.8](zotero://open-pdf/library/items/PAGER009?page=8). Next [Quillian 2017](zotero://select/library/items/QUIL2017), then more words [A 2020](zotero://select/library/items/AAAAAAAA); done."; });
  for (const w of [300, 320, 360, 420, 480]) {
    await p.setViewportSize({ width: w, height: 760 });
    await send(p, "x");
    await done(p);
    const split = await p.evaluate(() => [...window.__zmc.shadow.querySelectorAll(".md-nobr")].filter((n) => new Set([...n.getClientRects()].map((r) => Math.round(r.top))).size > 1).length);
    assert.equal(split, 0, `a chip and its punctuation split across lines at ${w}px`);
    await sim(p, () => { window.__zmc.sim.nextAnswer = "Prose that goes on for a while so the line is nearly full before it reaches the chip [Pager et al. 2009, p.8](zotero://open-pdf/library/items/PAGER009?page=8). Next [Quillian 2017](zotero://select/library/items/QUIL2017), then more words [A 2020](zotero://select/library/items/AAAAAAAA); done."; });
  }
  assert.equal(await p.locator(".md-nobr").count() > 0, true);
});

await test("settings: chat folder choose, cancel and use default; the section explains itself", async (p) => {
  await openSettings(p);
  const folder = () => p.locator(".folder__p").innerText();
  assert.equal(await folder(), "/Users/you/Documents/Zotero Chat");
  await p.getByRole("button", { name: "Use default" }).isDisabled().then((d) => assert.ok(d, "nothing to reset yet"));
  await p.getByRole("button", { name: "Choose…" }).click();
  await p.waitForFunction(() => window.__zmc.host.getSettings().chatFolder.endsWith("paper-notes"));
  assert.match(await folder(), /^\/Users\/you\/…\/paper-notes$/);
  assert.equal(await p.locator(".folder").getAttribute("title"), "/Users/you/Documents/Projects/hiring-audits/paper-notes");
  await sim(p, () => { window.__zmc.sim.pickFolder = null; });
  await p.getByRole("button", { name: "Choose…" }).click();
  await p.waitForTimeout(150);
  assert.match(await sim(p, () => window.__zmc.host.getSettings().chatFolder), /paper-notes$/, "cancel keeps it");
  await p.getByRole("button", { name: "Open", exact: true }).click();
  assert.equal(await sim(p, () => window.__zmc.sim.data.revealed), 1);
  assert.match(await p.locator("section.sec", { hasText: "Chat folder" }).innerText(), /remembers its own folder[\s\S]*\.claude\/skills and AGENTS\.md/);
  await p.getByRole("button", { name: "Use default" }).click();
  await p.waitForFunction(() => window.__zmc.host.getSettings().chatFolder === "");
  assert.equal(await folder(), "/Users/you/Documents/Zotero Chat");
});

await test("history: each chat shows its folder; Copy terminal command copies the backend's command and says Copied; a resumed chat starts in its own folder", async (p) => {
  await p.evaluate(() => { window.__copied = []; Object.defineProperty(navigator, "clipboard", { value: { writeText: async (t) => { window.__copied.push(t); } }, configurable: true }); });
  await p.locator('button[aria-label="History"]').click();
  await p.waitForSelector(".hrow__main");
  assert.match(await p.locator(".hist__note").innerText(), /Don't run it in both places at once/);
  const row = p.locator(".hrow", { hasText: "Which papers cite Pager" });
  assert.equal(await row.locator(".hrow__f").getAttribute("title"), "/Users/you/Documents/Projects/hiring-audits/paper-notes");
  assert.match(await row.locator(".hrow__f").innerText(), /…\/paper-notes|paper-notes/);
  await row.hover();
  await row.getByRole("button", { name: /Copy terminal command/ }).click();
  await p.waitForSelector(".hrow__copied");
  assert.match(await p.locator(".hrow__copied").innerText(), /Copied/);
  assert.deepEqual(await p.evaluate(() => window.__copied), [`cd '/Users/you/Documents/Projects/hiring-audits/paper-notes' && codex resume agent-s2`]);
  await p.waitForFunction(() => !window.__zmc.shadow.querySelector(".hrow__copied"), null, { timeout: 4000 });
  await row.locator(".hrow__main").click();
  await p.waitForSelector(".msg--assistant");
  await send(p, "and one more");
  await p.waitForFunction(() => window.__zmc.sim.preparedCwd.length > 0);
  assert.deepEqual(await sim(p, () => window.__zmc.sim.preparedCwd), ["/Users/you/Documents/Projects/hiring-audits/paper-notes"]);
  await done(p);
  await p.locator('button[aria-label="New chat"]').click();
  await send(p, "fresh");
  await p.waitForFunction(() => window.__zmc.sim.preparedCwd.length === 2);
  assert.equal((await sim(p, () => window.__zmc.sim.preparedCwd))[1], undefined, "a new chat uses the setting");
});

await test("welcome: first run shows it instead of the chat; the checks settle one by one; a failing check is fixed in place; Start chatting hands over", async (p) => {
  assert.equal(await p.locator(".cin").isVisible(), false, "the chat is not showing yet");
  assert.match(await p.locator(".wel h2").innerText(), /Chat with your library/);
  assert.equal(await p.locator(".wcard").count(), 3);
  assert.equal(await p.locator(".wcard[aria-checked=true]").innerText().then((t) => t.split("\n")[0]), "Claude Code");
  await p.waitForSelector(".wrow--pending");
  await p.waitForSelector(".wrow--bad", { timeout: 4000 });
  await p.waitForFunction(() => !window.__zmc.shadow.querySelector(".wrow--pending"));
  assert.equal(await p.locator(".wrow--bad").count(), 1);
  assert.match(await p.locator(".wrow--bad").innerText(), /zotero-cli/);
  assert.equal(await p.locator(".wfoot .btn--solid").count(), 0, "not ready: no primary button yet");
  await p.locator(".wrow--bad .btn").click();
  await p.waitForSelector(".wready", { timeout: 6000 });
  assert.match(await p.locator(".wready").innerText(), /All set. You'll chat with Claude Code \(Claude Max\)/);
  assert.ok(await p.locator(".mark--ready").count(), "the logo celebrates");
  await p.locator(".wfoot .btn--solid").click();
  await p.locator(".cin").waitFor({ state: "visible" });
  assert.equal(await p.locator(".wel").count(), 0);
  assert.equal(await p.evaluate(() => window.__zmc.host.getSettings().welcomed), true);
  await send(p, "hello");
  await done(p);
}, { params: { welcome: "1", doctor: "cli" } });

await test("welcome: choosing an agent changes the backend; an agent that is not ready says what to do; Esc does not skip it; Skip setup does", async (p) => {
  await p.locator(".wcard", { has: p.getByText("pi", { exact: true }) }).click();
  await p.waitForFunction(() => window.__zmc.host.getSettings().backend === "pi");
  assert.equal(await p.locator(".wcard[aria-checked=true]").count(), 1);
  await p.locator(".wcard", { hasText: "Codex" }).click();
  await p.waitForFunction(() => window.__zmc.host.getSettings().backend === "codex");
  assert.match(await p.locator(".wel__hint").first().innerText(), /codex login/);
  assert.match(await p.locator(".wcard--on").innerText(), /codex is not installed/);
  await p.keyboard.press("Escape");
  assert.equal(await p.locator(".wel").count(), 1);
  await p.getByRole("button", { name: "Settings" }).first().click();
  await p.waitForSelector("section.sec");
  await p.keyboard.press("Escape");
  await p.waitForSelector(".wel");
  await p.locator(".wfoot__skip").click();
  await p.locator(".cin").waitFor({ state: "visible" });
  assert.equal(await p.evaluate(() => window.__zmc.host.getSettings().welcomed), true);
}, { params: { welcome: "1" } });

await test("welcome: if the chosen agent is unusable and another works, it starts on the one that works", async (p) => {
  await p.waitForFunction(() => window.__zmc.host.getSettings().backend === "pi");
  await p.waitForSelector(".wcard--on");
  assert.match(await p.locator(".wcard--on").innerText(), /pi/);
}, { params: { welcome: "1", doctor: "backend" } });

await test("theme follows the host; close is an event for the glue", async (p) => {
  assert.equal(await p.locator(".zmc").getAttribute("data-theme"), "light");
  await sim(p, () => window.__zmc.sim.setTheme("dark"));
  assert.equal(await p.locator(".zmc").getAttribute("data-theme"), "dark");
  await p.locator('button[aria-label="Close the panel"]').click();
  assert.equal(await sim(p, () => document.getElementById("host").dataset.closed), "1");
});

await test("hostile markdown in a real answer: no script, no handlers, no foreign links or images, nothing runs", async (p) => {
  await sim(p, () => { window.__zmc.sim.speed = 0; });
  for (const src of XSS_CORPUS) {
    await sim(p, (s) => { window.__zmc.sim.nextAnswer = s; }, src);
    await send(p, "x");
    await p.waitForFunction((k) => window.__zmc.shadow.querySelectorAll('.msg--assistant[data-state="end_turn"]').length === k, XSS_CORPUS.indexOf(src) + 1);
  }
  await p.waitForTimeout(500); // lazy KaTeX
  const bad = await p.evaluate(() => {
    const out = [];
    const root = window.__zmc.shadow;
    for (const el of root.querySelectorAll(".md *")) {
      const tag = el.tagName.toLowerCase();
      if (el.closest(".cite, button.iconbtn")) continue; // our own icons
      if (["script", "iframe", "object", "embed", "style", "link", "meta", "base", "form", "input", "svg", "audio", "video"].includes(tag)) out.push(`<${tag}>`);
      for (const a of el.attributes) if (/^on/i.test(a.name)) out.push(`${tag}@${a.name}`);
      const href = el.getAttribute("href");
      if (href && !/^(https?|zotero):\/\//i.test(href)) out.push(`href=${href}`);
      if (tag === "img" && !/^data:image\/(png|jpeg);base64,/.test(el.getAttribute("src") || "")) out.push(`img src`);
      if (tag === "a" && el.target) out.push("target");
    }
    return [...new Set(out)].concat(window.__pwned ? ["__pwned ran"] : []);
  });
  assert.deepEqual(bad, []);
  // a click on every rendered link or citation must not navigate or run anything
  await p.evaluate(() => { for (const el of window.__zmc.shadow.querySelectorAll(".md a, .md .cite")) el.click(); });
  assert.equal(await p.evaluate(() => window.__pwned), undefined);
  assert.ok((await p.evaluate(() => location.href)).includes("index.html"));
});

await test("streaming patches only the last message; earlier ones are never touched", async (p) => {
  await send(p, "compare");
  await done(p);
  await sim(p, () => { window.__zmc.sim.speed = 3; });
  await send(p, "compare again");
  await p.waitForSelector(".msg--assistant[data-state='running']");
  await p.evaluate(() => {
    const inner = window.__zmc.shadow.querySelector(".feed__inner");
    window.__muts = [];
    new MutationObserver((l) => { for (const m of l) window.__muts.push(m.target.closest?.(".msg") ? [...inner.children].indexOf(m.target.closest(".msg")) : -1); }).observe(inner, { subtree: true, childList: true, characterData: true, attributes: true });
  });
  await done(p);
  const idx = await p.evaluate(() => [...new Set(window.__muts)]);
  const kids = await p.evaluate(() => window.__zmc.shadow.querySelector(".feed__inner").children.length);
  const last = kids - 2; // the pending line follows the last message
  assert.ok(idx.every((i) => i === last || i === -1 || i === last - 0), `mutated message indexes: ${idx}`);
});

await test("a 400-turn chat opens fast and a streamed token stays cheap; the jump pill appears and works", async (p) => {
  await p.locator('button[aria-label="History"]').click();
  const t0 = Date.now();
  await p.locator(".hrow__main", { hasText: "Long literature" }).click();
  await p.waitForSelector(".msg--assistant");
  const open = Date.now() - t0;
  console.log(`  opened 400 turns in ${open} ms`);
  assert.ok(open < 4000, `open ${open} ms`);
  await p.waitForTimeout(300);
  assert.equal(await p.locator(".jump").isVisible(), false, "at the bottom after opening");
  await sim(p, () => { window.__zmc.sim.speed = 2; });
  await send(p, "compare");
  await p.waitForSelector(".md--streaming");
  await p.mouse.move(200, 300);
  for (let i = 0; i < 6; i++) await p.mouse.wheel(0, -20000); // a reader scrolls up, as a person does
  await p.waitForSelector(".jump");
  const stats = await p.evaluate(async () => {
    const feed = window.__zmc.shadow.querySelector(".feed");
    const frames = []; let last = performance.now(); const end = last + 800;
    await new Promise((res) => { const f = (t) => { frames.push(t - last); last = t; if (t < end) requestAnimationFrame(f); else res(); }; requestAnimationFrame(f); });
    frames.sort((a, b) => a - b);
    return { median: frames[Math.floor(frames.length / 2)], worst: frames.at(-1), away: feed.scrollHeight - feed.scrollTop - feed.clientHeight };
  });
  console.log(`  frame median ${stats.median.toFixed(1)} ms, worst ${stats.worst.toFixed(1)} ms while streaming into 400 turns`);
  assert.ok(stats.median < 34);
  assert.ok(stats.away > 2000, `a reader scrolled up is not yanked down (${stats.away}px from the bottom)`);
  await p.locator(".jump").click();
  await p.waitForFunction(() => { const f = window.__zmc.shadow.querySelector(".feed"); return f.scrollHeight - f.scrollTop - f.clientHeight < 60; });
}, { params: { stress: 1, speed: 0 }, height: 760 });

// no horizontal overflow at 300 px in any state: nothing may stick out of the panel except inside scroll containers
const OVERFLOW = () => {
  const root = window.__zmc.shadow;
  const app = root.querySelector(".zmc").getBoundingClientRect();
  const out = [];
  for (const el of root.querySelectorAll(".zmc *")) {
    if (el.closest(".md-table, .code pre, .math--display, .sd__pre, .fixlog, .thought__body, svg, [hidden]") || el.tagName === "svg") continue;
    const r = el.getBoundingClientRect();
    if (r.width && r.right > app.right + 1) out.push(`${el.tagName.toLowerCase()}.${el.className}`);
    if (r.width && r.left < app.left - 1) out.push(`left:${el.tagName.toLowerCase()}.${el.className}`);
  }
  for (const s of root.querySelectorAll(".feed, .vw__body, .composer, .hd, .dock")) if (s.scrollWidth > s.clientWidth + 1) out.push(`scrollX:${s.className}`);
  return [...new Set(out)];
};
for (const [name, params, run] of [
  ["empty", {}, async () => {}],
  ["answer", {}, async (p) => { await send(p, "compare"); await done(p); await p.locator(".foot__src").click(); await p.locator(".step__row").first().click(); }],
  ["chips", { ctx: "area" }, async (p) => { await p.locator(".cin").click(); await p.keyboard.type("@Discrim"); await p.waitForSelector(".pop__i"); await p.keyboard.press("Enter"); await p.waitForTimeout(200); }],
  ["popup", {}, async (p) => { await p.locator('button[aria-label="Add a source"]').click(); await p.waitForSelector(".pop__i"); }],
  ["permission", { speed: 5 }, async (p) => { await send(p, "add a note"); await p.waitForSelector(".perm__opts"); }],
  ["error", {}, async (p) => { await send(p, "error"); await done(p, "error"); }],
  ["history", {}, async (p) => { await p.locator('button[aria-label="History"]').click(); await p.waitForSelector(".hrow__main"); }],
  ["settings", {}, async (p) => { await p.locator('button[aria-label="Settings"]').click(); await p.waitForTimeout(300); }],
  ["status", { doctor: "many" }, async (p) => { await p.locator(".stat").click(); await p.waitForSelector(".check"); }],
  ["setup", { doctor: "many" }, async (p) => { await p.locator(".setup").waitFor(); }],
  ["welcome", { welcome: "1", doctor: "many" }, async (p) => { await p.waitForSelector(".wrow--bad"); await p.waitForFunction(() => !window.__zmc.shadow.querySelector(".wrow--pending")); await p.locator(".wrow--bad .btn").first().click(); await p.waitForTimeout(500); }],
]) {
  for (const width of [300, 320]) for (const dark of [false, true]) {
    await test(`no overflow at ${width}px ${dark ? "dark" : "light"}: ${name}`, async (p) => { await run(p); await p.waitForTimeout(150); assert.deepEqual(await p.evaluate(OVERFLOW), []); }, { width, dark, params });
  }
}

await browser.close();
console.log(`${n} passed`);
