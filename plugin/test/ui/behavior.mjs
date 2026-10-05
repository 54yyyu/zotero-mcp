// Behaviour checks in a real browser against the fake host: node test/ui/behavior.mjs [--browser webkit]
// (build first: node scripts/preview.mjs). Exits non-zero on the first failure.
import assert from "node:assert/strict";
import { launch, openPanel } from "./lib.mjs";
import { XSS_CORPUS } from "./corpus.ts";
import { HOSTILE } from "../../src/ui/fake-diagrams.ts";

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
  // a link with &quote=: the chip's label stays "Author year, p.N", the tooltip shows the quote, a click passes it on
  const quoted = p.locator(".cite").first();
  assert.equal(await quoted.innerText(), "Bertrand and Mullainathan 2004, p.8");
  assert.equal(await quoted.getAttribute("title"), "“applicants with White names receive 50 percent more callbacks for interviews”");
  await quoted.click();
  assert.match((await sim(p, () => window.__zmc.sim.opened)).at(-1), /^zotero:\/\/open-pdf\/library\/items\/BM2004AB\?page=8&quote=applicants%20with%20White/);
  await p.locator(".source").first().click();
  assert.equal((await sim(p, () => window.__zmc.sim.opened)).at(-1), "zotero://open-pdf/library/items/BM2004AB?page=8", "the sources fold opens the page");
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

await test("an unchanged area and selection go once; later turns only name them", async (p) => {
  await sim(p, () => window.__zmc.sim.setContext("area"));
  for (const q of ["first", "second"]) { await send(p, q); await done(p); await p.waitForFunction((k) => window.__zmc.sim.prompts.length === k, q === "first" ? 1 : 2); }
  const [a, b] = await sim(p, () => window.__zmc.sim.prompts.map((x) => ({ text: x.text, images: x.images?.length ?? 0 })));
  assert.equal(a.images, 1);
  assert.equal(b.images, 0, "the same picture is not sent twice");
  assert.match(b.text, /Still pointing at, unchanged[^\n]*selected area p\.19 \(annotation ANNAREA1\)/);
});

await test("context meter: hidden without numbers, a ring whenever the backend says, amber at 70%, red and a new-chat nudge at 85%", async (p) => {
  const turn = async (q, k) => { await send(p, q); await p.waitForFunction((n) => window.__zmc.shadow.querySelectorAll('.msg--assistant[data-state="end_turn"]').length === n, k); };
  const level = () => p.locator(".cmeter").getAttribute("data-level");
  await turn("hello", 1);
  assert.equal(await p.locator(".cmeter").isVisible(), false, "the backend reported nothing: no meter");
  await sim(p, () => { window.__zmc.sim.contextUsage = { used: 30000, size: 200000 }; });
  await turn("again", 2);
  assert.ok(await p.locator(".cmeter").isVisible(), "15%: the ring shows, quietly");
  assert.equal(await level(), "");
  await sim(p, () => { window.__zmc.sim.contextUsage = { used: 124000, size: 200000 }; });
  await turn("more", 3);
  assert.equal(await p.locator(".cmeter").getAttribute("title"), "Context: 62% full (124k of 200k tokens). Older parts are summarised automatically.");
  assert.equal(await p.locator(".cmeter").getAttribute("aria-label"), await p.locator(".cmeter").getAttribute("title"));
  assert.equal(await p.locator(".cmeter__arc").getAttribute("stroke-dasharray"), "62 100");
  assert.equal(await level(), "");
  await sim(p, () => { window.__zmc.sim.contextUsage = { used: 150000, size: 200000 }; });
  await turn("warmer", 4);
  assert.equal(await level(), "warm");
  assert.equal(await p.locator(".cnote").isVisible(), false);
  await sim(p, () => { window.__zmc.sim.contextUsage = { used: 176000, size: 1000000 }; });
  await turn("a bigger window", 5);
  assert.match(await p.locator(".cmeter").getAttribute("title"), /^Context: 18% full \(176k of 1M tokens\)/);
  await sim(p, () => { window.__zmc.sim.contextUsage = { used: 176000, size: 200000 }; });
  await turn("even more", 6);
  assert.equal(await level(), "full");
  assert.ok(await p.locator(".cnote").isVisible(), "88%: the suggestion shows");
  await p.locator(".cnote").getByRole("button", { name: "Dismiss" }).click();
  assert.equal(await p.locator(".cnote").isVisible(), false, "dismissed");
  await p.locator(".cnote").evaluate((e) => { e.hidden = false; }); // bring it back to test its button
  await p.locator(".cnote").getByRole("button", { name: "New chat" }).click();
  await p.waitForFunction(() => !window.__zmc.shadow.querySelector(".msg--assistant"));
  assert.equal(await p.locator(".cmeter").isVisible(), false, "a new chat starts empty");
  assert.equal(await p.locator(".cnote").isVisible(), false);
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

await test("a tool step reads as what it does; the raw command stays in the tooltip", async (p) => {
  await send(p, "compare");
  await done(p);
  const title = p.locator(".step__title").first();
  assert.equal(await title.innerText(), "Search library for “hiring discrimination audit”");
  assert.equal(await title.getAttribute("title"), 'cd /Users/you/Documents && zotero-cli --json search "hiring discrimination audit" --limit 10');
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
  await p.locator(".pick--model").click();
  await p.locator('.mdd .eff__stop[title="High"]').click();
  await p.waitForFunction(() => window.__zmc.host.getSettings().effort["claude-code"] === "high");
  assert.match(await p.locator(".pick--model .pick__e").innerText(), /High/);
  assert.equal(await p.locator(".eff__cur").innerText(), "High");
  assert.equal(await p.locator(".eff__rec").count(), 0, "Recommended marks only the agent's default level");
  // dragging along the track moves through the levels and applies the one it is released on
  const t = await p.locator(".eff__track").boundingBox();
  await p.mouse.move(t.x + t.width - 14, t.y + t.height / 2); await p.mouse.down();
  await p.mouse.move(t.x + 14, t.y + t.height / 2, { steps: 8 }); await p.mouse.up();
  await p.waitForFunction(() => window.__zmc.host.getSettings().effort["claude-code"] === "low");
  // the keys: arrows step, Enter applies and closes
  await p.locator(".eff__track").focus();
  await p.keyboard.press("ArrowRight"); await p.keyboard.press("ArrowRight");
  await p.keyboard.press("Enter");
  await p.waitForFunction(() => window.__zmc.host.getSettings().effort["claude-code"] === "high");
  assert.equal(await p.locator(".mdd").count(), 0, "Enter on the slider applies and closes");
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
  assert.match(await p.locator(".pick--model .pick__e").innerText(), /Off/, "pi's own effort default");
});

const agentTab = (p, name) => p.locator(".mdd__agent", { hasText: name });
await test("agent switch: an empty chat switches at once; a chat with messages asks first and is never cut silently", async (p) => {
  await p.locator(".pick--model").click();
  await p.waitForSelector(".mdd .menu__item");
  assert.deepEqual(await p.locator(".mdd__agent").evaluateAll((els) => els.map((e) => [e.textContent, e.dataset.state, e.getAttribute("aria-checked")])),
    [["Claude Code", "ok", "true"], ["Codex", "bad", "false"], ["pi", "ok", "false"]], "every agent, with the status detect() gave");
  // an agent that is not ready says why and stays unchosen
  await agentTab(p, "Codex").click();
  assert.match(await p.locator(".mdd__note").innerText(), /Codex isn't ready: codex is not installed/);
  assert.equal(await sim(p, () => window.__zmc.host.getSettings().backend), "claude-code");
  // empty chat: at once, and the dropdown now lists pi's models
  await agentTab(p, "pi").click();
  await p.waitForFunction(() => window.__zmc.host.getSettings().backend === "pi");
  await p.waitForSelector('.mdd [role="menuitemradio"]:has-text("Small and fast")');
  assert.equal(await p.locator(".mdd__note").isVisible(), false);
  assert.match(await p.locator(".pick--model").innerText(), /Provider default/);
  await p.keyboard.press("Escape");
  // with messages: an inline question; Cancel keeps everything
  await send(p, "hello");
  await done(p);
  await p.locator(".pick--model").click();
  await agentTab(p, "Claude Code").click();
  assert.equal(await p.locator(".mdd__note").innerText().then((t) => t.replace(/\s+/g, " ").trim()), "Switching starts a new chat with Claude Code. Start new chat Cancel");
  await p.getByRole("button", { name: "Cancel" }).click();
  assert.equal(await p.locator(".mdd__note").isVisible(), false);
  assert.deepEqual([await sim(p, () => window.__zmc.host.getSettings().backend), await p.locator(".msg").count() > 0], ["pi", true]);
  // Start new chat: the new agent, an empty chat, the dropdown stays open on the new agent's models
  await agentTab(p, "Claude Code").click();
  await p.getByRole("button", { name: "Start new chat" }).click();
  await p.waitForFunction(() => window.__zmc.host.getSettings().backend === "claude-code");
  await p.waitForSelector('.mdd [role="menuitemradio"]:has-text("Claude Sonnet")');
  assert.equal(await p.locator(".msg").count(), 0, "a new chat");
  assert.equal(await agentTab(p, "Claude Code").getAttribute("aria-checked"), "true");
});

await test("models: the first four in the catalog's order, the rest under More models; the current one always shows", async (p) => {
  await p.locator(".pick--model").click();
  await p.waitForSelector(".mdd .menu__item");
  const names = () => p.locator('.mdd [role="menuitemradio"]').evaluateAll((els) => els.filter((e) => e.getClientRects().length).map((e) => e.querySelector(".menu__t").textContent));
  assert.deepEqual(await names(), ["Claude Opus", "Claude Sonnet", "Claude Haiku", "Claude Opus 4.1"]);
  assert.equal(await p.locator(".mdd__more").innerText().then((t) => t.replace(/\s+/g, " ").trim()), "More models 2");
  await p.locator(".mdd__more").click();
  assert.deepEqual((await names()).slice(4), ["Claude Sonnet 4", "Claude Haiku 3.5"], "expanded inline, in the same dropdown");
  await p.getByRole("menuitemradio", { name: /Haiku 3\.5/ }).click();
  await p.waitForFunction(() => window.__zmc.host.getSettings().model["claude-code"] === "haiku-3-5");
  assert.match(await p.locator(".pick--model .pick__t").innerText(), /Haiku 3\.5/);
  await p.locator(".pick--model").click();
  await p.waitForSelector(".mdd .menu__item");
  assert.deepEqual(await names(), ["Claude Opus", "Claude Sonnet", "Claude Haiku", "Claude Opus 4.1", "Claude Haiku 3.5"], "the current model is shown without expanding");
  assert.equal(await p.locator('.mdd [aria-checked="true"] .menu__t').innerText(), "Claude Haiku 3.5");
  assert.match(await p.locator(".mdd__more").innerText(), /1/);
});

await test("model dropdown: the keyboard opens it, moves through agents, models and effort, and Esc gives the focus back", async (p) => {
  const focused = () => p.evaluate(() => { const a = window.__zmc.shadow.activeElement; return a ? (a.querySelector?.(".menu__t")?.textContent ?? a.dataset.id ?? a.getAttribute("role") ?? a.className) : null; });
  await p.waitForFunction(() => /Sonnet/.test(window.__zmc.shadow.querySelector(".pick--model").textContent));
  await p.locator(".pick--model").focus();
  await p.keyboard.press("Enter");
  await p.waitForSelector(".mdd .menu__item");
  await p.waitForFunction(() => window.__zmc.shadow.activeElement?.getAttribute("aria-checked") === "true");
  assert.equal(await focused(), "Claude Sonnet", "the current model has the focus");
  await p.keyboard.press("ArrowDown");
  assert.equal(await focused(), "Claude Haiku");
  await p.keyboard.press("ArrowUp"); await p.keyboard.press("ArrowUp"); await p.keyboard.press("ArrowUp");
  assert.equal(await focused(), "claude-code", "up from the first model is the agent row");
  await p.keyboard.press("ArrowRight");
  assert.equal(await focused(), "codex", "left and right move between agents without choosing one");
  assert.equal(await sim(p, () => window.__zmc.host.getSettings().backend), "claude-code");
  await p.keyboard.press("ArrowUp");
  assert.equal(await focused(), "slider", "up from the top wraps to the effort slider");
  await p.keyboard.press("ArrowRight");
  await p.waitForFunction(() => window.__zmc.host.getSettings().effort["claude-code"] === "high");
  assert.equal(await focused(), "slider", "left and right move the slider, the focus stays");
  await p.keyboard.press("Escape");
  assert.equal(await p.locator(".mdd").count(), 0);
  assert.equal(await p.evaluate(() => window.__zmc.shadow.activeElement?.classList.contains("pick--model")), true, "Esc returns the focus to the button");
  await p.keyboard.press(" ");
  await p.waitForSelector(".mdd");
});

await test("model dropdown at 300px: one button beside the ring and the mode; the model name truncates first", async (p) => {
  await sim(p, () => { window.__zmc.sim.contextUsage = { used: 62000, size: 200000 }; });
  await send(p, "hello");
  await done(p);
  await p.waitForSelector(".cmeter:not([hidden])");
  const g = await p.evaluate(() => {
    const q = (s) => window.__zmc.shadow.querySelector(s), r = (s) => q(s).getBoundingClientRect();
    const parts = [".pick--model", ".cmeter", ".pick--mode", ".send"].map((s) => [s, r(s)]);
    const overlap = parts.some(([, a], i) => parts.slice(i + 1).some(([, b]) => a.right > b.left + 0.5));
    const t = q(".pick--model .pick__t"), e = q(".pick--model .pick__e");
    return { overlap, inside: parts.every(([, x]) => x.right <= r(".composer").right && x.width > 0), modelCut: t.scrollWidth > t.clientWidth, effortWhole: e.scrollWidth <= e.clientWidth, effort: e.textContent };
  });
  assert.deepEqual(g, { overlap: false, inside: true, modelCut: true, effortWhole: true, effort: "Medium" });
  await p.locator(".pick--model").click();
  await p.waitForSelector(".mdd .eff__track");
  const m = await p.evaluate(() => { const r = (s) => window.__zmc.shadow.querySelector(s).getBoundingClientRect(); const d = r(".mdd"), c = r(".composer"), b = r(".pick--model"); return { inside: d.left >= c.left - 1 && d.right <= c.right + 1, above: d.bottom <= b.top + 1 }; });
  assert.deepEqual(m, { inside: true, above: true }, "the dropdown sits above its button, inside the composer");
}, { width: 300 });

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
  assert.equal(await folder(), "/Users/you/Documents/Zotero-Chat");
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
  assert.equal(await folder(), "/Users/you/Documents/Zotero-Chat");
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

const look = (p) => p.evaluate(() => { const z = window.__zmc.shadow.querySelector(".zmc"); const cs = getComputedStyle(z); return { ...z.dataset, accent: cs.getPropertyValue("--accent").trim(), veil: cs.getPropertyValue("--bg-veil").trim(), img: cs.getPropertyValue("--bg-img").trim().slice(0, 30), fs4: cs.getPropertyValue("--fs-4").trim(), before: getComputedStyle(z, "::before").backgroundImage.slice(0, 40) }; });
const appearance = (p) => p.evaluate(() => JSON.parse(JSON.stringify(window.__zmc.host.getSettings().appearance)));

await test("appearance: glass, accent, background, size and density apply live and are saved", async (p) => {
  assert.deepEqual([(await look(p)).glass, (await look(p)).bg, (await look(p)).accent], ["on", "none", "#16181d"], "glass on, no background, mono by default");
  assert.equal(await p.evaluate(() => getComputedStyle(window.__zmc.shadow.querySelector(".zmc")).getPropertyValue("--link").trim()), "#2563c9", "links stay the calm blue by default");
  assert.equal(await p.evaluate(() => window.__zmc.shadow.querySelector(".zmc").hasAttribute("data-accent")), false);
  await openSettings(p);
  const card = p.locator('section.sec[aria-label="Appearance"]');
  await card.getByRole("switch", { name: /Glass/ }).click();
  assert.equal((await look(p)).glass, "off");
  assert.equal((await appearance(p)).glass, false);
  await card.getByRole("radio", { name: "Blue" }).click();
  assert.equal((await look(p)).accent, "#2563c9", "the accent variable changes at once");
  assert.equal(await p.evaluate(() => window.__zmc.shadow.querySelector(".zmc").getAttribute("data-accent")), "custom", "a chosen accent marks the root");
  assert.equal((await appearance(p)).accent, "#2563c9");
  // the custom colour: the native input, then the hex field it reveals
  await card.locator('input[type="color"]').fill("#8a2be2");
  await p.waitForFunction(() => window.__zmc.host.getSettings().appearance.accent === "#8a2be2");
  assert.equal((await look(p)).accent, "#8a2be2");
  await card.getByRole("textbox", { name: "Accent colour as hex" }).fill("12a150");
  await card.getByRole("textbox", { name: "Accent colour as hex" }).press("Enter");
  await p.waitForFunction(() => window.__zmc.host.getSettings().appearance.accent === "#12a150");
  await card.getByRole("textbox", { name: "Accent colour as hex" }).fill("nope");
  await card.getByRole("textbox", { name: "Accent colour as hex" }).press("Enter");
  await card.getByRole("alert").waitFor();
  assert.equal((await appearance(p)).accent, "#12a150", "a bad hex is refused, not saved");
  // the send button wears the accent, in readable text
  await card.getByRole("radio", { name: "Red" }).click();
  assert.equal((await appearance(p)).accent, "#cc2936", "our red is a chosen colour");
  assert.equal(await p.evaluate(() => window.__zmc.shadow.querySelector(".zmc").getAttribute("data-accent")), "custom");
  await card.getByRole("radio", { name: "Mono" }).click();
  assert.equal((await appearance(p)).accent, "", "mono is the default");
  assert.equal(await p.evaluate(() => window.__zmc.shadow.querySelector(".zmc").hasAttribute("data-accent")), false, "mono: no data-accent");
  const layer = () => p.evaluate(() => getComputedStyle(window.__zmc.shadow.querySelector(".zmc"), "::before").display);
  assert.deepEqual([(await look(p)).bg, await layer()], ["none", "none"], "Plain by default: no glow, no backdrop");
  await card.getByRole("radio", { name: "Glow" }).click();
  assert.deepEqual([(await look(p)).bg, /radial-gradient/.test((await look(p)).before)], ["glow", true], "Glow is a choice");
  await card.getByRole("radio", { name: "Dawn" }).click();
  const l = await look(p);
  assert.deepEqual([l.bg, /gradient/.test(l.before)], ["dawn", true], "a preset paints the backdrop");
  await card.getByRole("radio", { name: "Large" }).click();
  assert.equal((await look(p)).fs4, "15.5px");
  await card.getByRole("radio", { name: "Compact" }).click();
  assert.equal((await look(p)).density, "compact");
  await p.getByRole("button", { name: "Back to the chat" }).click();
  await p.locator(".cin").fill("x");
  const send = await p.evaluate(() => getComputedStyle(window.__zmc.shadow.querySelector(".send")).backgroundColor);
  assert.equal(send, "rgb(22, 24, 29)", "Send is the accent, mono ink (flat, so no gem)");
  assert.deepEqual(await appearance(p), { glass: false, accent: "", background: "dawn", image: "", imageVisibility: 50, imageBlur: 0, textSize: "large", density: "compact" });
});

await test("appearance: a picture is chosen, the sliders preview while dragged and save on release, Remove clears it; cancel does nothing", async (p) => {
  await openSettings(p);
  const card = p.locator('section.sec[aria-label="Appearance"]');
  await sim(p, () => { window.__zmc.sim.pickImage = null; });
  await card.getByRole("button", { name: "Choose an image…" }).click();
  await p.waitForTimeout(100);
  assert.deepEqual([(await appearance(p)).image, (await look(p)).bg], ["", "none"], "cancelled: nothing changes");
  await sim(p, () => { window.__zmc.sim.pickImage = { name: "kyoto.jpg", dataUrl: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'/%3E" }; });
  await card.getByRole("button", { name: "Choose an image…" }).click();
  await p.waitForFunction(() => window.__zmc.host.getSettings().appearance.background === "image");
  let l = await look(p);
  assert.deepEqual([l.bg, l.img.startsWith('url("data:image/svg'), l.veil], ["image", true, "65%"]);
  assert.equal((await appearance(p)).image, "kyoto.jpg");
  assert.ok(await card.getByRole("radio", { name: /Your picture \(kyoto\.jpg\)/ }).isVisible(), "the picture is a tile");
  // a drag: input events preview, nothing is saved until the change event
  await card.getByRole("slider", { name: "Picture visibility" }).evaluate((el) => { el.value = "100"; el.dispatchEvent(new Event("input", { bubbles: true })); });
  assert.equal((await look(p)).veil, "30%", "live while dragging");
  assert.equal((await appearance(p)).imageVisibility, 50, "not saved mid-drag");
  await card.getByRole("slider", { name: "Picture visibility" }).evaluate((el) => el.dispatchEvent(new Event("change", { bubbles: true })));
  await p.waitForFunction(() => window.__zmc.host.getSettings().appearance.imageVisibility === 100);
  await card.getByRole("slider", { name: "Picture blur" }).fill("8");
  await p.waitForFunction(() => window.__zmc.host.getSettings().appearance.imageBlur === 8);
  // a preset, then back to the picture without choosing it again
  await card.getByRole("radio", { name: "Mist" }).click();
  assert.equal((await look(p)).bg, "mist");
  await card.getByRole("radio", { name: /Your picture/ }).click();
  assert.equal((await look(p)).bg, "image");
  await card.getByRole("button", { name: "Remove the background picture" }).click();
  await p.waitForFunction(() => window.__zmc.host.getSettings().appearance.image === "");
  l = await look(p);
  assert.deepEqual([l.bg, l.img, await sim(p, () => window.__zmc.sim.image)], ["none", "", null]);
  assert.equal(await card.getByRole("slider").count(), 0, "the picture's sliders go with it");
});

await test("appearance: a saved picture is read back when the panel opens", async (p) => {
  await p.waitForFunction(() => window.__zmc.shadow.querySelector(".zmc").dataset.bg === "image");
  assert.match((await look(p)).img, /^url\("data:image\/svg/);
}, { params: { look: JSON.stringify({ background: "image", image: "kyoto.jpg" }), image: "1" } });

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

// ───────────── diagrams (```svg blocks) ─────────────

const dgFill = (sel, prop) => (p) => p.evaluate(([s, pr]) => getComputedStyle(window.__zmc.shadow.querySelector(s))[pr], [sel, prop]);
const barOpacity = (p, i = 0) => p.evaluate((k) => getComputedStyle(window.__zmc.shadow.querySelectorAll(".dg__bar")[k]).opacity, i);

await test("diagrams render as themed figures; the toolbar shows on hover and focus; Source toggles; Save calls the host", async (p) => {
  await send(p, "draw it");
  await done(p);
  assert.equal(await p.locator(".dg .dg__fig svg").count(), 3);
  assert.equal(await p.locator(".dg--pending").count(), 0);
  assert.deepEqual(await p.locator(".dg__fig svg").evaluateAll((a) => a.map((s) => [s.getAttribute("role"), s.getAttribute("aria-label")])), [
    ["img", "From question to cited answer"], ["img", "Designs by control and external validity"], ["img", "Instrument, treatment, outcome"]]);
  // palette names became the panel's colours: the accent box is drawn in the accent, nothing fell back to black
  const accent = await p.evaluate(() => getComputedStyle(window.__zmc.shadow.querySelector(".dg")).getPropertyValue("--dg-accent").trim());
  const rect = await p.evaluate(() => { const r = window.__zmc.shadow.querySelectorAll(".dg__fig svg")[0].querySelectorAll("rect")[1]; const c = getComputedStyle(r); return [c.stroke, c.fill]; });
  const asRgb = await p.evaluate((c) => { const d = document.createElement("i"); d.style.color = c; document.body.append(d); const v = getComputedStyle(d).color; d.remove(); return v; }, accent);
  assert.equal(rect[0], asRgb);
  assert.notEqual(rect[1], "rgb(0, 0, 0)");
  // the toolbar: hidden at rest, shown on hover, and on keyboard focus
  await p.mouse.move(5, 5);
  await p.waitForTimeout(250);
  assert.equal(await barOpacity(p), "0");
  await p.locator(".dg").first().hover();
  await p.waitForTimeout(250);
  assert.equal(await barOpacity(p), "1");
  await p.mouse.move(5, 5);
  await p.locator(".dg").nth(1).locator("button").first().focus();
  await p.waitForTimeout(250);
  assert.equal(await barOpacity(p, 1), "1");
  // Source shows the raw SVG as a code block, and hides it again
  const src = p.locator(".dg").first().locator('button[aria-label="Show the SVG source"]');
  await p.locator(".dg").first().hover();
  await src.click();
  assert.match(await p.locator(".dg").first().locator(".dg__src .code").innerText(), /<svg viewBox="0 0 360 132"/);
  assert.equal(await src.getAttribute("aria-pressed"), "true");
  await src.click();
  assert.equal(await p.locator(".dg__src").count(), 0);
  // Save as PNG / SVG hand the host a real file
  await p.locator(".dg").first().locator('button[aria-label="Save as PNG"]').click();
  await p.locator(".dg").first().locator('button[aria-label="Save as SVG"]').click();
  await p.waitForFunction(() => window.__zmc.sim.saved.length === 2);
  const saved = await sim(p, () => window.__zmc.sim.saved);
  saved.sort((a, b) => a.name.localeCompare(b.name)); // the PNG takes longer to make
  assert.deepEqual(saved.map((x) => [x.name, x.mime]), [["from-question-to-cited-answer.png", "image/png"], ["from-question-to-cited-answer.svg", "image/svg+xml"]]);
  assert.equal(saved[0].head, "89504e470d0a1a0a", "a PNG signature");
  assert.ok(saved[0].size > 20000, `a large PNG (${saved[0].size} bytes)`);
  assert.match(saved[1].head, /^<\?xml[^]*<svg [^>]*xmlns="http:\/\/www.w3.org\/2000\/svg"/);
  assert.ok(!saved[1].head.includes("var("));
  await p.locator(".dg").first().locator('button[aria-label="Copy as an image"]').click(); // must not throw, whatever the clipboard allows
  await p.waitForTimeout(300);
});

await test("a diagram streams as a quiet placeholder, never half drawn", async (p) => {
  await sim(p, () => { window.__zmc.sim.speed = 4; });
  await send(p, "draw it");
  await p.waitForSelector(".dg--pending", { timeout: 8000 });
  assert.match(await p.locator(".dg--pending").first().innerText(), /Drawing/);
  // whatever is drawn while streaming is a whole diagram (its title is there)
  for (const label of await p.locator(".dg__fig svg").evaluateAll((a) => a.map((s) => s.getAttribute("aria-label")))) assert.ok(label && label !== "Diagram", label);
  await done(p);
  assert.equal(await p.locator(".dg--pending").count(), 0);
  assert.equal(await p.locator(".dg__fig svg").count(), 3);
});

await test("a hostile diagram is neutralised: no script runs, nothing external, only the drawing", async (p) => {
  await sim(p, (t) => { window.__zmc.sim.nextAnswer = "```svg\n" + t + "\n```"; }, HOSTILE);
  await send(p, "anything");
  await done(p);
  await p.locator(".dg__fig svg circle").click({ force: true });
  await p.waitForTimeout(200);
  assert.equal(await p.evaluate(() => window.__pwned), undefined);
  const r = await p.evaluate(() => {
    const svg = window.__zmc.shadow.querySelector(".dg__fig svg");
    const els = [svg, ...svg.querySelectorAll("*")];
    return { tags: [...new Set(els.map((e) => e.localName))].sort(), attrs: els.flatMap((e) => [...e.attributes].map((a) => `${a.name}=${a.value}`)) };
  });
  assert.deepEqual(r.tags, ["circle", "rect", "svg", "text"]);
  for (const a of r.attrs) assert.ok(!/^on|^class=|javascript:|https?:|evil/i.test(a), a);
});

await test("an invalid diagram is shown as its code; dark theme recolours a drawing", async (p) => {
  await sim(p, () => { window.__zmc.sim.nextAnswer = "```svg\n<div>not a drawing</div>\n```"; });
  await send(p, "one");
  await done(p);
  assert.equal(await p.locator(".dg").count(), 0);
  assert.equal(await p.locator(".code .code__lang").innerText(), "svg");
  await send(p, "draw it");
  await p.waitForFunction(() => window.__zmc.shadow.querySelectorAll(".dg__fig svg").length === 3, null, { timeout: 8000 });
  const stroke = dgFill(".dg__fig svg rect:nth-of-type(2)", "stroke");
  const light = await stroke(p);
  await sim(p, () => window.__zmc.sim.setTheme("dark"));
  await p.waitForTimeout(100);
  const dark = await stroke(p);
  assert.notEqual(light, dark);
  const want = await p.evaluate(() => { const e = window.__zmc.shadow.querySelector(".dg"); const d = document.createElement("i"); d.style.color = getComputedStyle(e).getPropertyValue("--dg-accent"); e.append(d); const c = getComputedStyle(d).color; d.remove(); return c; });
  assert.equal(dark, want, "the dark theme's accent");
});

await test("diagram accent is blue unless the user chose an accent; the toolbar sits above the drawing", async (p) => {
  await send(p, "draw it");
  await p.waitForFunction(() => window.__zmc.shadow.querySelectorAll(".dg__fig svg").length === 3, null, { timeout: 8000 });
  const stroke = () => p.evaluate(() => getComputedStyle(window.__zmc.shadow.querySelectorAll(".dg__fig svg")[0].querySelectorAll("rect")[1]).stroke);
  assert.equal(await stroke(), "rgb(59, 91, 219)", "the calm default blue");
  // the appearance contract: data-accent="custom" and --accent on the root when (and only when) the user picked one
  await p.evaluate(() => { const z = window.__zmc.shadow.querySelector(".zmc"); z.dataset.accent = "custom"; z.style.setProperty("--accent", "#c2410c"); z.style.setProperty("--accent-l", "#c2410c"); });
  assert.equal(await stroke(), "rgb(194, 65, 12)");
  await p.locator(".dg").first().hover();
  await p.locator(".dg").first().locator('button[aria-label="Save as SVG"]').click();
  await p.waitForFunction(() => window.__zmc.sim.saved.length === 1);
  const svgFile = await sim(p, () => window.__zmc.sim.saved[0].head);
  assert.ok(!svgFile.includes("#3b5bdb"), "the export follows the custom accent too");
  await p.evaluate(() => { const z = window.__zmc.shadow.querySelector(".zmc"); delete z.dataset.accent; });
  assert.equal(await stroke(), "rgb(59, 91, 219)");
  // the toolbar never covers the drawing, and showing it moves nothing
  const box = () => p.evaluate(() => { const c = window.__zmc.shadow.querySelectorAll(".dg")[1]; const top = c.getBoundingClientRect().top; const r = (s) => c.querySelector(s).getBoundingClientRect(); return { bar: r(".dg__bar").bottom - top, fig: r(".dg__fig svg").top - top }; });
  await p.mouse.move(5, 5); await p.waitForTimeout(200);
  const rest = await box();
  await p.locator(".dg").nth(1).hover(); await p.waitForTimeout(200);
  const hover = await box();
  assert.ok(hover.bar <= hover.fig, `toolbar above the drawing: ${JSON.stringify(hover)}`);
  assert.equal(rest.fig, hover.fig, "nothing jumps when the toolbar shows");
});

await test("at 300px a drawing keeps its labels readable and scrolls sideways, with a soft edge", async (p) => {
  await send(p, "draw it");
  await p.waitForFunction(() => window.__zmc.shadow.querySelectorAll(".dg__fig svg").length === 3, null, { timeout: 8000 });
  await p.waitForTimeout(200);
  const g = await p.evaluate(() => { const w = window.__zmc.shadow.querySelector(".dg__fig"); const s = w.querySelector("svg"); return { scale: s.getBoundingClientRect().width / 360, scrolls: w.scrollWidth > w.clientWidth, fade: w.dataset.fade, bar: getComputedStyle(window.__zmc.shadow.querySelector(".dg__bar")).opacity }; });
  assert.ok(g.scale >= 0.89, `scale ${g.scale}`);
  assert.deepEqual([g.scrolls, g.fade, g.bar], [true, "r", "1"], "scrolls, fades on the right, toolbar always shown when narrow");
  await p.evaluate(() => { const w = window.__zmc.shadow.querySelector(".dg__fig"); w.scrollLeft = w.scrollWidth; });
  await p.waitForFunction(() => window.__zmc.shadow.querySelector(".dg__fig").dataset.fade === "l");
}, { width: 300 });

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
  ["long model name", {}, async (p) => {
    await p.waitForFunction(() => /Sonnet/.test(window.__zmc.shadow.querySelector(".pick--model").textContent) && window.__zmc.shadow.querySelector(".pick--model .pick__e"));
    await p.evaluate(() => { window.__zmc.shadow.querySelector(".pick--model .pick__t").textContent = "Claude Opus 5.5 with a very long name and a million tokens of context"; });
    const t = await p.evaluate(() => { const e = window.__zmc.shadow.querySelector(".pick--model .pick__e"); return [e.scrollWidth, e.clientWidth, e.textContent]; });
    assert.deepEqual([t[0] <= t[1], t[2]], [true, "Medium"], "the effort level is never the thing that truncates");
    await p.locator(".pick--model").click();
    await p.waitForSelector(".mdd .eff__track");
    const g = await p.evaluate(() => { const r = (s) => window.__zmc.shadow.querySelector(s).getBoundingClientRect(); const m = r(".mdd"), b = r(".pick--model"), c = r(".composer"); return { off: Math.abs(m.left - Math.max(c.left, Math.min(b.left - c.left, c.width - m.width) + c.left)), inside: m.right <= c.right + 1 }; });
    assert.ok(g.off < 3 && g.inside, "the dropdown opens above its own button, inside the composer");
  }],
  ["diagram", {}, async (p) => { await send(p, "draw it"); await done(p); await p.locator(".dg").first().hover(); await p.locator(".dg").first().locator('button[aria-label="Show the SVG source"]').click(); }],
  ["meter", {}, async (p) => { await p.evaluate(() => { window.__zmc.sim.contextUsage = { used: 176000, size: 200000 }; }); await send(p, "hello"); await done(p); await p.waitForSelector(".cnote:not([hidden])"); }],
  ["popup", {}, async (p) => { await p.locator('button[aria-label="Add a source"]').click(); await p.waitForSelector(".pop__i"); }],
  ["permission", { speed: 5 }, async (p) => { await send(p, "add a note"); await p.waitForSelector(".perm__opts"); }],
  ["error", {}, async (p) => { await send(p, "error"); await done(p, "error"); }],
  ["history", {}, async (p) => { await p.locator('button[aria-label="History"]').click(); await p.waitForSelector(".hrow__main"); }],
  ["settings", {}, async (p) => { await p.locator('button[aria-label="Settings"]').click(); await p.waitForTimeout(300); }],
  ["appearance", {}, async (p) => { await p.locator('button[aria-label="Settings"]').click(); await p.getByRole("button", { name: "Choose an image…" }).click(); await p.waitForSelector(".bgsw--image"); await p.locator('input[type="color"]').fill("#8a2be2"); await p.waitForFunction(() => window.__zmc.shadow.querySelector(".hex")); }],
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
await test("save as note: the answer goes to the host with its question as the title and a PNG per drawing; the line confirms, Open selects it, a failure says why", async (p) => {
  await send(p, "draw it");
  await done(p);
  const answer = await p.evaluate(() => [...window.__zmc.shadow.querySelectorAll(".msg--assistant .md")].map((m) => m.textContent).join(""));
  assert.ok(answer.length > 0);
  await p.locator('button[aria-label="Save as a Zotero note"]').click();
  await p.waitForSelector(".foot .noteline");
  const notes = await sim(p, () => window.__zmc.sim.notes);
  assert.equal(notes.length, 1);
  assert.equal(notes[0].title, "draw it");
  assert.match(notes[0].markdown, /^I'll look for related research[\s\S]*```svg\n<svg/);
  assert.equal(notes[0].images.length, 3, "one image per drawing");
  assert.ok(notes[0].images.every((i) => i && i.png && i.size > 2000 && i.width > 100 && i.width <= 640 && i.height > 20), JSON.stringify(notes[0].images));
  assert.equal(await p.locator(".foot .noteline").getAttribute("role"), "status");
  assert.match(await p.locator(".foot .noteline").innerText(), /Saved to note\s*Open/);
  await p.locator(".foot .noteline .lnk").click();
  assert.equal((await sim(p, () => window.__zmc.sim.opened)).at(-1), "zotero://select/library/items/NOTE0001");
  await sim(p, () => { window.__zmc.sim.noteFails = "this library is read-only"; });
  await p.locator('button[aria-label="Save as a Zotero note"]').click();
  await p.waitForSelector(".foot .noteline--err");
  assert.equal(await p.locator(".foot .noteline").count(), 1, "the error replaces the confirmation");
  assert.equal(await p.locator(".foot .noteline--err").innerText(), "Couldn't save the note: this library is read-only");
  // an answer without drawings: no images, and diagram.ts is not needed for it
  await sim(p, () => { window.__zmc.sim.noteFails = null; window.__zmc.sim.nextAnswer = "Plain **answer** with $x^2$."; });
  await send(p, "plain");
  await p.waitForFunction(() => window.__zmc.shadow.querySelectorAll('.msg--assistant[data-state="end_turn"]').length === 2);
  await p.locator('button[aria-label="Save as a Zotero note"]').last().click();
const WIDE_MATH = String.raw`$$\hat{\tau} = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{T_i Y_i}{e(X_i)} - \frac{(1-T_i)Y_i}{1-e(X_i)}\right)$$

$$\mathcal{L}(\theta) = \sum_{i=1}^{n} \log p_\theta(y_i \mid x_i) + \lambda_1 \lVert \theta \rVert_1 + \lambda_2 \lVert \theta \rVert_2^2 + \sum_{j=1}^{m} \mu_j \, g_j(\theta) + \sum_{k=1}^{K} \nu_k h_k(\theta) + \gamma \, \mathrm{KL}\left(q_\phi(z \mid x) \,\Vert\, p(z)\right)$$

$$E = mc^2$$`;

await test("display math: Copy TeX copies the source; a wide formula shrinks to 72% at most, then scrolls with a soft edge, never clipped", async (p) => {
  await p.evaluate(() => { window.__copied = []; Object.defineProperty(navigator, "clipboard", { value: { writeText: async (t) => { window.__copied.push(t); } }, configurable: true }); });
  await sim(p, (t) => { window.__zmc.sim.nextAnswer = t; }, WIDE_MATH);
  await send(p, "x");
  await done(p);
  await p.waitForFunction(() => !window.__zmc.shadow.querySelector(".math--raw"));
  await p.waitForTimeout(150);
  const g = await p.evaluate(() => [...window.__zmc.shadow.querySelectorAll(".math--display")].map((m) => ({ fs: m.style.fontSize, over: m.scrollWidth > m.clientWidth + 1, fade: m.dataset.fade ?? "", left: Math.round(m.firstElementChild.getBoundingClientRect().left - m.getBoundingClientRect().left) })));
  assert.deepEqual(g.map((x) => [x.over, x.fade, x.left]), [[false, "", 0], [true, "r", 0], [false, "", 0]], JSON.stringify(g));
  assert.equal(g[1].fs, "72%", "shrunk to the floor before scrolling");
  assert.ok(g[0].fs === "" || parseInt(g[0].fs) >= 72, "a formula that nearly fits is only made a little smaller");
  assert.equal(g[2].fs, "", "a short formula keeps its size");
  await p.evaluate(() => { const m = window.__zmc.shadow.querySelectorAll(".math--display")[1]; m.scrollLeft = m.scrollWidth; });
  await p.waitForFunction(() => window.__zmc.shadow.querySelectorAll(".math--display")[1].dataset.fade === "l");
  // Copy TeX: hidden until hover, then the original source
  const btn = p.locator('.mathblock button[aria-label="Copy TeX"]').nth(2);
  assert.equal(await btn.evaluate((b) => getComputedStyle(b).opacity), "0");
  await p.locator(".mathblock").nth(2).hover();
  await p.waitForFunction(() => getComputedStyle(window.__zmc.shadow.querySelectorAll(".math__copy")[2]).opacity === "1");
  await btn.click();
  assert.deepEqual(await p.evaluate(() => window.__copied), ["E = mc^2"]);
  // wider panel: the same formula needs less shrinking, and is re-fitted on resize
  await p.setViewportSize({ width: 700, height: 760 });
  await p.waitForFunction(() => { const m = window.__zmc.shadow.querySelectorAll(".math--display")[1]; return m.style.fontSize !== "72%" || m.scrollWidth <= m.clientWidth + 1; });
}, { width: 320 });

await test("streaming: never a half-written formula, math fence or table header as raw text; finished blocks are never rebuilt", async (p) => {
  const fence = "```";
  const ans = String.raw`Intro paragraph that stays put.

$$\hat{\tau} = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{T_i Y_i}{e(X_i)}\right)$$

The gap is $\Delta = \ln(1.5)$ and the cost is $5 or $10 today.

| Study | Ratio |
|---|---:|
| Pager | 1.44 |

Inline display \[x^2 + y^2\] here and \(a+b\) there.

${fence}math
\int_0^1 x\,dx
${fence}

Done.`;
  await p.evaluate(() => {
    window.__seen = new Set(); window.__first = null; window.__stop = false;
    const tick = () => {
      const md = window.__zmc.shadow.querySelector(".msg--assistant .md");
      if (!md) return;
      if (md.children.length > 2) window.__first ??= md.querySelector("p");
      const raws = [...md.querySelectorAll(".math--raw")];
      const clone = md.cloneNode(true);
      [...clone.querySelectorAll(".math--raw")].forEach((e, k) => { if (getComputedStyle(raws[k]).opacity === "0") e.remove(); });
      clone.querySelectorAll(".math:not(.math--raw), .code").forEach((e) => e.remove());
      for (const bad of ["$$", "\\frac", "\\[", "\\(", "\\Delta", "\\int", "| Study", "|---"]) if (clone.textContent.includes(bad)) window.__seen.add(bad);
      if (md.querySelector(".math--error")) window.__seen.add("a formula error");
    };
    new MutationObserver(tick).observe(window.__zmc.shadow.querySelector(".feed__inner"), { subtree: true, childList: true, characterData: true });
  });
  await sim(p, (t) => { window.__zmc.sim.nextAnswer = t; window.__zmc.sim.speed = 4; }, ans);
  await send(p, "x");
  await done(p);
  const r = await p.evaluate(() => ({ seen: [...window.__seen], kept: !!window.__first && window.__first === window.__zmc.shadow.querySelector(".msg--assistant .md p") }));
  assert.deepEqual(r, { seen: [], kept: true });
  assert.equal(await p.locator(".msg--assistant .math--display").count(), 3, "$$, \\[ \\] and the math fence");
  assert.equal(await p.locator(".msg--assistant table").count(), 1);
  assert.match(await p.locator(".msg--assistant .md").innerText(), /cost is \$5 or \$10 today/, "money stays text");
});

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
  ["long model name", {}, async (p) => {
    await p.waitForFunction(() => /Sonnet/.test(window.__zmc.shadow.querySelector(".pick--model").textContent) && window.__zmc.shadow.querySelector(".pick--model .pick__e"));
    await p.evaluate(() => { window.__zmc.shadow.querySelector(".pick--model .pick__t").textContent = "Claude Opus 5.5 with a very long name and a million tokens of context"; });
    const t = await p.evaluate(() => { const e = window.__zmc.shadow.querySelector(".pick--model .pick__e"); return [e.scrollWidth, e.clientWidth, e.textContent]; });
    assert.deepEqual([t[0] <= t[1], t[2]], [true, "Medium"], "the effort level is never the thing that truncates");
    await p.locator(".pick--model").click();
    await p.waitForSelector(".mdd .eff__track");
    const g = await p.evaluate(() => { const r = (s) => window.__zmc.shadow.querySelector(s).getBoundingClientRect(); const m = r(".mdd"), b = r(".pick--model"), c = r(".composer"); return { off: Math.abs(m.left - Math.max(c.left, Math.min(b.left - c.left, c.width - m.width) + c.left)), inside: m.right <= c.right + 1 }; });
    assert.ok(g.off < 3 && g.inside, "the dropdown opens above its own button, inside the composer");
  }],
  ["diagram", {}, async (p) => { await send(p, "draw it"); await done(p); await p.locator(".dg").first().hover(); await p.locator(".dg").first().locator('button[aria-label="Show the SVG source"]').click(); }],
  ["meter", {}, async (p) => { await p.evaluate(() => { window.__zmc.sim.contextUsage = { used: 176000, size: 200000 }; }); await send(p, "hello"); await done(p); await p.waitForSelector(".cnote:not([hidden])"); }],
  ["popup", {}, async (p) => { await p.locator('button[aria-label="Add a source"]').click(); await p.waitForSelector(".pop__i"); }],
  ["permission", { speed: 5 }, async (p) => { await send(p, "add a note"); await p.waitForSelector(".perm__opts"); }],
  ["error", {}, async (p) => { await send(p, "error"); await done(p, "error"); }],
  ["history", {}, async (p) => { await p.locator('button[aria-label="History"]').click(); await p.waitForSelector(".hrow__main"); }],
  ["settings", {}, async (p) => { await p.locator('button[aria-label="Settings"]').click(); await p.waitForTimeout(300); }],
  ["appearance", {}, async (p) => { await p.locator('button[aria-label="Settings"]').click(); await p.getByRole("button", { name: "Choose an image…" }).click(); await p.waitForSelector(".bgsw--image"); await p.locator('input[type="color"]').fill("#8a2be2"); await p.waitForFunction(() => window.__zmc.shadow.querySelector(".hex")); }],
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
  await p.waitForFunction(() => window.__zmc.sim.notes.length === 2);
  assert.deepEqual(await sim(p, () => window.__zmc.sim.notes[1]), { title: "plain", markdown: "Plain **answer** with $x^2$.", images: [] });
});

await test("a diagram's Add to a note saves that drawing alone, titled, with its PNG; the card says so", async (p) => {
  await send(p, "draw it");
  await done(p);
  const card = p.locator(".dg").nth(1);
  await card.hover();
  await card.locator('button[aria-label="Add to a note"]').click();
  await card.locator(".noteline").waitFor();
  const n = (await sim(p, () => window.__zmc.sim.notes))[0];
  assert.equal(n.title, "Designs by control and external validity");
  assert.match(n.markdown, /^```svg\n<svg[\s\S]*<\/svg>\n```$/);
  assert.equal(n.images.length, 1);
  assert.ok(n.images[0].png && n.images[0].width === 360, JSON.stringify(n.images));
  assert.equal(await p.locator(".dg .noteline").count(), 1, "only that card shows the line");
  await card.locator(".noteline .lnk").click();
  assert.equal((await sim(p, () => window.__zmc.sim.opened)).at(-1), "zotero://select/library/items/NOTE0001");
});

await test("explain better: only on the last finished answer; it asks again, intuition first, with the same context", async (p) => {
  await send(p, "compare");
  await done(p);
  assert.equal(await p.locator('button[aria-label="Explain it better"]').count(), 1);
  await p.locator('button[aria-label="Explain it better"]').click();
  await p.waitForFunction(() => window.__zmc.shadow.querySelectorAll('.msg--assistant[data-state="end_turn"]').length === 2);
  const last = (await sim(p, () => window.__zmc.sim.prompts)).at(-1).text;
  assert.match(last, /<zotero-context>[\s\S]*<\/zotero-context>\n\nExplain that again from the intuition first, with a tiny example, then the details\.$/);
  assert.equal(await p.locator(".ubub__text").last().innerText(), "Explain that again from the intuition first, with a tiny example, then the details.");
  assert.equal(await p.locator('button[aria-label="Explain it better"]').count(), 1, "the earlier answer lost it");
  assert.equal(await p.locator('.msg--assistant').last().locator('button[aria-label="Explain it better"]').count(), 1);
});

