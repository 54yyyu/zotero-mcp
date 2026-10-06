import { test } from "node:test";
import assert from "node:assert/strict";
import { FakeHost } from "../../src/ui/fake-host.ts";
import { CATALOGS, defaultSettings } from "../../src/ui/fake-catalog.ts";
import { addPrompt, editPrompt, effective, modeHelp, removePrompt, setAuth, setFlag, setFolder, setPerBackend, shortPath, shortcuts } from "../../src/ui/settings-model.ts";

const host = () => new FakeHost({ speed: 0, noHistory: true });

test("model, mode and effort are saved per backend and never leak across backends", async () => {
  const h = host();
  await h.setSettings(setPerBackend(h.getSettings(), "model", "claude-code", "haiku"));
  await h.setSettings(setPerBackend(h.getSettings(), "mode", "claude-code", "plan"));
  await h.setSettings(setPerBackend(h.getSettings(), "effort", "codex", "high"));
  const s = h.getSettings();
  assert.deepEqual(s.model, { "claude-code": "haiku", codex: "", pi: "" });
  assert.deepEqual(s.mode, { "claude-code": "plan", codex: "", pi: "" });
  assert.deepEqual(s.effort, { "claude-code": "", codex: "high", pi: "" });
});

test("effective() is the saved choice, else the catalog default, else empty", () => {
  const s = defaultSettings();
  assert.equal(effective(s, CATALOGS["claude-code"], "mode", "claude-code"), "default");
  assert.equal(effective(s, CATALOGS.codex, "mode", "codex"), "agent");
  assert.equal(effective(s, undefined, "mode", "pi"), "");
  assert.equal(effective({ ...s, mode: { ...s.mode, codex: "read-only" } }, CATALOGS.codex, "mode", "codex"), "read-only");
});

test("the catalogs differ per backend: Codex's own modes, pi has none and its own efforts", async () => {
  const h = host();
  const claude = await h.runtime.catalog("claude-code");
  assert.deepEqual(claude.modes.map((m) => m.id), ["default", "acceptEdits", "plan", "auto", "bypassPermissions"]);
  assert.deepEqual(claude.efforts.map((m) => m.id), ["low", "medium", "high", "xhigh", "max"]);
  assert.deepEqual(CATALOGS.codex.modes.map((m) => m.id), ["read-only", "workspace-write", "agent", "agent-full-access"]);
  assert.deepEqual(CATALOGS.pi.modes, []);
  assert.deepEqual(CATALOGS.pi.efforts.map((m) => m.id), ["off", "minimal", "low", "medium", "high", "xhigh"]);
  await assert.rejects(h.runtime.catalog("codex"), /not installed/, "an unavailable backend has no catalog, and says why");
  h.sim.catalogFails.push("pi");
  await assert.rejects(h.runtime.catalog("pi"), /did not answer/);
});

test("mode help: friendly text for known ids, the catalog's words otherwise", () => {
  assert.match(modeHelp({ id: "plan", name: "Plan" }), /changes nothing/i);
  assert.equal(modeHelp({ id: "weird", name: "Weird", description: "its own words" }), "its own words");
  assert.equal(modeHelp({ id: "weird", name: "Weird" }), "");
  for (const c of Object.values(CATALOGS)) for (const m of c.modes) assert.ok(modeHelp(m), `${m.id} has help`);
});

test("flags and sign-in mode patch only what they name", async () => {
  const h = host();
  await h.setSettings(setFlag("enterToSend", false));
  await h.setSettings(setFlag("showThinking", false));
  await h.setSettings(setFlag("expandTools", true));
  await h.setSettings(setFlag("openAtStart", true));
  await h.setSettings(setFlag("attachAreas", false));
  await h.setSettings(setAuth(h.getSettings(), "codex", "api-key"));
  const s = h.getSettings();
  assert.deepEqual([s.enterToSend, s.showThinking, s.expandTools, s.openAtStart, s.attachAreas, s.followFocus, s.attachSelection], [false, false, true, true, false, true, true]);
  assert.deepEqual(s.auth, { "claude-code": "subscription", codex: "api-key", pi: "subscription" });
});

test("prompt edits: add, edit and delete return new lists and keep the pins", () => {
  const p0 = defaultSettings().prompts;
  const edited = editPrompt(p0, 2, { title: "Hypotheses" });
  assert.equal(edited[2]?.title, "Hypotheses");
  assert.equal(edited[2]?.text, p0[2]?.text);
  assert.equal(edited[2]?.slot, 3, "an edit keeps the pin");
  assert.equal(p0[2]?.title, "Propose testable hypotheses", "the input list is untouched");
  const added = addPrompt(p0, "pX");
  assert.equal(added.length, 5);
  assert.deepEqual(added[4], { id: "pX", title: "", text: "" });
  assert.deepEqual(removePrompt(added, 0).map((p) => p.id), ["p2", "p3", "p4", "pX"]);
});

test("data actions reach the host: clear history, reveal the workspace, reset settings (keeps keys and chats gone)", async () => {
  const h = new FakeHost({ speed: 0 });
  assert.ok((await h.sessions()).length > 0);
  await h.setApiKey("claude-code", "sk-x");
  await h.setSettings({ ...setFlag("showThinking", false), ...setPerBackend(h.getSettings(), "model", "claude-code", "opus") });
  await h.clearHistory();
  await h.revealWorkspace();
  await h.resetSettings();
  assert.deepEqual((await h.sessions()), []);
  assert.deepEqual(h.sim.data, { cleared: 1, revealed: 1, resets: 1 });
  assert.deepEqual(h.getSettings(), defaultSettings());
  assert.equal(await h.hasApiKey("claude-code"), true, "reset keeps API keys");
  assert.match(h.about().version, /^\d+\.\d+\.\d+/);
});

test("the shortcut reference follows the platform and the send key", () => {
  const mac = shortcuts(true, true);
  assert.deepEqual(mac.find(([w]) => w === "Send"), ["Send", "Enter"]);
  assert.ok(mac.some(([, k]) => k.includes("⌘")));
  const win = shortcuts(false, false);
  assert.deepEqual(win.find(([w]) => w === "Send"), ["Send", "Ctrl+Enter"]);
  assert.deepEqual(win.find(([w]) => w === "New line"), ["New line", "Enter"]);
  assert.ok(win.every(([, k]) => !k.includes("⌘")));
});

test("chat folder: choose saves it, use default clears it, a cancelled picker changes nothing; about() reports the folder in use", async () => {
  const h = host();
  assert.equal(h.about().workspace, "/Users/you/Documents/Zotero-Chat");
  const picked = await h.chooseFolder(h.about().workspace);
  assert.ok(picked);
  await h.setSettings(setFolder(picked));
  assert.equal(h.getSettings().chatFolder, "/Users/you/Documents/Projects/hiring-audits/paper-notes");
  assert.equal(h.about().workspace, picked);
  h.sim.pickFolder = null;
  assert.equal(await h.chooseFolder(), null);
  assert.equal(h.getSettings().chatFolder, picked);
  await h.setSettings(setFolder(""));
  assert.equal(h.about().workspace, "/Users/you/Documents/Zotero-Chat");
});

test("a path is shortened in the middle, keeping the start and the last folder", () => {
  assert.equal(shortPath("/Users/you/Zotero Chat"), "/Users/you/Zotero Chat");
  assert.equal(shortPath("/Users/you/Documents/Projects/hiring-audits/paper-notes", 34), "/Users/you/\u2026/paper-notes");
  assert.ok(shortPath("/a/b/c/" + "x".repeat(80), 30).length <= 30);
  assert.ok(shortPath("/Users/you/Documents/Projects/hiring-audits/paper-notes", 20).length <= 20);
});

test("resume commands per backend, with the folder quoted for the shell", async () => {
  const h = host();
  const base = { id: "s", title: "t", cwd: "/Users/you/Documents/Zotero Chat", agentSessionId: "abc123", updatedAt: 1 };
  assert.equal(h.resumeCommand({ ...base, backend: "claude-code" }), `cd '/Users/you/Documents/Zotero Chat' && claude --resume abc123`);
  assert.equal(h.resumeCommand({ ...base, backend: "codex" }), `cd '/Users/you/Documents/Zotero Chat' && codex resume abc123`);
  assert.equal(h.resumeCommand({ ...base, backend: "pi" }), `cd '/Users/you/Documents/Zotero Chat' && pi --session abc123`);
  assert.equal(h.resumeCommand({ ...base, cwd: "/Users/o'neil/x", backend: "pi" }), `cd '/Users/o'\\''neil/x' && pi --session abc123`);
  assert.equal(h.resumeCommand({ ...base, agentSessionId: "", backend: "pi" }), null, "no agent session yet: nothing to resume");
});

test("prepareSession takes the saved chat's folder, else the setting", async () => {
  const h = host();
  assert.equal((await h.prepareSession()).cwd, "/Users/you/Documents/Zotero-Chat");
  assert.equal((await h.prepareSession("/some/old/folder")).cwd, "/some/old/folder");
  await h.setSettings(setFolder("/new/folder"));
  assert.equal((await h.prepareSession()).cwd, "/new/folder");
  assert.deepEqual(h.sim.preparedCwd, [undefined, "/some/old/folder", undefined]);
});
