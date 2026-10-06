// Skills and prompts in a real Zotero (run with --mock-agent): nothing of it at startup; the skills folder scanned when
// asked and a changed SKILL.md read again; an import through the real host (scripts, links, hidden files left out, the
// text the user saw is what is written); the skills synced into the chat folder (ours refreshed, someone else's never
// touched, an off one removed); `/name` reaching the mock agent as a plain message; snapshots of the bubbles, the `/`
// menu, the settings card and the add-skill preview, light and dark, glass on.
async function main(ctx) {
  const { Zotero, win, plugin } = ctx;
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };
  const J = (...p) => PathUtils.join(...p);
  const exists = (p) => IOUtils.exists(p);
  const dataDir = J(Zotero.Profile.dir, "zotero-chat");
  const skillsDir = J(dataDir, "skills");
  const chatDefault = ctx.env("ZMC_DEFAULT_CHAT_FOLDER");
  const scratch = J(PathUtils.parent(chatDefault), "skills-test");
  await IOUtils.remove(skillsDir, { recursive: true, ignoreAbsent: true });
  await IOUtils.remove(scratch, { recursive: true, ignoreAbsent: true });
  const fruit = "---\nname: fruit-check\ndescription: Says a fruit.\n---\n\nReply PINEAPPLE-7.\n";
  await IOUtils.makeDirectory(J(skillsDir, "fruit-check"), { createAncestors: true });
  await IOUtils.writeUTF8(J(skillsDir, "fruit-check", "SKILL.md"), fruit);

  // 1. startup: the panel is not loaded, so neither is anything of skills; the chat folder has none yet
  const injected = plugin.windows.get(win);
  check(injected.loaded() === null && plugin.timing.panelLoadMs === null, "panel.js not read at startup");
  check(!(await exists(J(chatDefault, ".agents", "skills", "fruit-check"))), "no skill copied before the panel is used");

  await ctx.resize(1280, 860);
  injected.show();
  const root = await ctx.waitFor(() => win.document.getElementById("zmc-root")?.shadowRoot?.querySelector("textarea") && win.document.getElementById("zmc-root").shadowRoot, "panel rendered");
  const { host, bundle } = plugin.panel();
  const $ = (s) => root.querySelector(s);
  const $$ = (s) => [...root.querySelectorAll(s)];
  const zmc = () => $(".zmc");
  const theme = async (t) => {
    Services.prefs.setIntPref("browser.theme.toolbar-theme", t === "dark" ? 0 : 1);
    await ctx.waitFor(() => zmc().dataset.theme === t, "theme " + t, 5000).catch(() => { zmc().dataset.theme = t; });
    await ctx.sleep(250);
  };
  const run = (cmd, args) => bundle.spawner.run(cmd, args, { env: { PATH: "/usr/bin:/bin" } });
  check(zmc().dataset.glass === "on", "glass is on");

  // 2. scanning: the folder, then the built-in; a changed SKILL.md is read again (its mtime), an unchanged one is not
  let t0 = Date.now();
  let list = await host.skills.list();
  out.firstScanMs = Date.now() - t0;
  check(list.map((k) => k.name).join() === "fruit-check,create-skill" && list[0].description === "Says a fruit." && list[1].builtin, "the list: " + JSON.stringify(list));
  await IOUtils.writeUTF8(J(skillsDir, "fruit-check", "SKILL.md"), fruit.replace("Says a fruit.", "Says a fruit, changed."));
  await IOUtils.setModificationTime(J(skillsDir, "fruit-check", "SKILL.md"), Date.now() + 2000);
  t0 = Date.now();
  list = await host.skills.list();
  out.rescanMs = Date.now() - t0;
  check(list[0].description === "Says a fruit, changed.", "a changed SKILL.md is read again: " + list[0].description);
  await IOUtils.makeDirectory(J(skillsDir, "Not A Name"));
  await IOUtils.makeDirectory(J(skillsDir, "no-skill-md"));
  check((await host.skills.list()).length === 2, "folders that are not skills are ignored");

  // 3. import through the real host
  const src = J(scratch, "picked skill");
  await IOUtils.makeDirectory(J(src, "sub"), { createAncestors: true });
  const picked = "---\nname: import-test\ndescription: From disk.\nlicense: MIT\n---\n\n# Import test\n\nUse notes.md.\n";
  await IOUtils.writeUTF8(J(src, "SKILL.md"), picked);
  await IOUtils.writeUTF8(J(src, "notes.md"), "notes");
  await IOUtils.writeUTF8(J(src, "sub", "data.csv"), "a,b\n1,2\n");
  await IOUtils.writeUTF8(J(src, "run.sh"), "#!/bin/sh\necho hi\n");
  await IOUtils.writeUTF8(J(src, "tool"), "#!/usr/bin/env python3\nprint(1)\n");
  await IOUtils.writeUTF8(J(src, "plain-exec.txt"), "text but executable");
  await IOUtils.writeUTF8(J(src, "helper.py"), "print(1)\n");
  await IOUtils.writeUTF8(J(src, ".hidden.md"), "secret");
  await run("/bin/chmod", ["+x", J(src, "plain-exec.txt")]);
  await run("/bin/ln", ["-s", "/etc/hosts", J(src, "evil-link.md")]);
  await run("/bin/ln", ["-s", "/etc", J(src, "linked-dir")]);
  const plan = await host.skills.inspect(J(src, "SKILL.md"));
  out.plan = { copy: plan.copy, skip: plan.skip.map((s) => `${s.path}: ${s.reason}${s.keepable ? " (keepable)" : ""}`) };
  check(plan.text === picked && plan.name === "import-test" && plan.description === "From disk." && !plan.loose, "the plan reads the file: " + JSON.stringify(plan).slice(0, 300));
  check(plan.copy.join() === "notes.md,sub/data.csv", "copies only the plain files: " + plan.copy);
  const why = Object.fromEntries(plan.skip.map((s) => [s.path, s.reason]));
  check(why["run.sh"] === "a script or program" && why["tool"] === "a script or program" && why["plain-exec.txt"] === "a script or program" && why["helper.py"] === "a script or program", "scripts (by extension, shebang, exec bit): " + JSON.stringify(why));
  check(/link/.test(why["evil-link.md"]) && /link/.test(why["linked-dir"]) && why[".hidden.md"] === "hidden", "links and hidden files: " + JSON.stringify(why));
  await host.skills.add(plan, { name: "import-test", description: "Imported: from a folder.", keepSkipped: false });
  const dst = J(skillsDir, "import-test");
  const written = await IOUtils.readUTF8(J(dst, "SKILL.md"));
  check(written.startsWith('---\nname: import-test\ndescription: "Imported: from a folder."\nlicense: MIT\n---\n') && written.endsWith("Use notes.md.\n"), "SKILL.md with the chosen name and description: " + written);
  check(await exists(J(dst, "notes.md")) && await exists(J(dst, "sub", "data.csv")), "the plain files copied");
  for (const f of ["run.sh", "tool", "plain-exec.txt", "helper.py", ".hidden.md", "evil-link.md", "linked-dir"]) check(!(await exists(J(dst, f))), f + " left out");
  let refused = "";
  await host.skills.add(plan, { name: "import-test", description: "x", keepSkipped: false }).catch((e) => { refused = String(e); });
  check(/already have/.test(refused), "a taken name is refused: " + refused);
  await host.skills.add(plan, { name: "import-keep", description: "Kept scripts.", keepSkipped: true });
  check(await exists(J(skillsDir, "import-keep", "run.sh")) && await exists(J(skillsDir, "import-keep", "helper.py")) && !(await exists(J(skillsDir, "import-keep", "evil-link.md"))), "keepSkipped copies scripts, never links");
  await IOUtils.writeUTF8(J(scratch, "loose notes.md"), "# Loose Notes\n\nTag every paper I read with its week. Then stop.\n");
  const loose = await host.skills.inspect(J(scratch, "loose notes.md"));
  check(loose.loose && loose.name === "loose-notes" && loose.description === "Tag every paper I read with its week." && loose.copy.length === 0, "a loose .md proposes a name and a line: " + JSON.stringify(loose));
  await host.skills.add(loose, { name: "loose-notes", description: loose.description, keepSkipped: false });
  check((await IOUtils.readUTF8(J(skillsDir, "loose-notes", "SKILL.md"))).startsWith("---\nname: loose-notes\ndescription: Tag every paper I read with its week.\n---\n\n# Loose Notes"), "wrapped into a skill folder");
  await run("/bin/ln", ["-s", J(src, "SKILL.md"), J(scratch, "link.md")]);
  refused = "";
  await host.skills.inspect(J(scratch, "link.md")).catch((e) => { refused = String(e); });
  check(/not a link/.test(refused), "a picked link is refused: " + refused);

  // 4. sync into the chat folder at session start: on skills and the built-in, with the marker; an off one removed
  await host.setSettings({ skills: { "import-keep": { off: true } } });
  let prep = await host.prepareSession();
  check(prep.cwd === chatDefault, "the default folder: " + prep.cwd);
  for (const sub of [[".agents", "skills"], [".claude", "skills"]]) {
    const base = J(prep.cwd, ...sub);
    check(await exists(J(base, "fruit-check", "SKILL.md")) && await exists(J(base, "fruit-check", ".zotero-chat")), "fruit-check in " + sub.join("/"));
    check(await exists(J(base, "import-test", "sub", "data.csv")), "the whole folder copied");
    check(!(await exists(J(base, "import-keep"))), "an off skill is not copied");
    const cs = await IOUtils.readUTF8(J(base, "create-skill", "SKILL.md"));
    check(cs.includes(skillsDir) && !cs.includes("{{"), "create-skill knows where skills live");
  }
  await host.setSettings({ skills: { "import-test": { off: true } } });
  await host.prepareSession();
  check(!(await exists(J(prep.cwd, ".agents", "skills", "import-test"))) && !(await exists(J(prep.cwd, ".claude", "skills", "import-test"))), "turned off: our copies removed");
  await host.setSettings({ skills: {} });
  // someone else's project: a skill of the same name there is never touched, only ours are refreshed or removed
  const proj = J(scratch, "their project");
  await IOUtils.makeDirectory(J(proj, ".agents", "skills", "fruit-check"), { createAncestors: true });
  await IOUtils.writeUTF8(J(proj, ".agents", "skills", "fruit-check", "SKILL.md"), "MINE");
  await IOUtils.makeDirectory(J(proj, ".agents", "skills", "stale-ours"), { createAncestors: true });
  await IOUtils.writeUTF8(J(proj, ".agents", "skills", "stale-ours", ".zotero-chat"), "x");
  await IOUtils.makeDirectory(J(proj, ".agents", "skills", "their-own"), { createAncestors: true });
  await host.prepareSession(proj);
  check((await IOUtils.readUTF8(J(proj, ".agents", "skills", "fruit-check", "SKILL.md"))) === "MINE", "their fruit-check is left alone");
  check(await exists(J(proj, ".claude", "skills", "fruit-check", ".zotero-chat")), "where there was none, ours goes in");
  check(!(await exists(J(proj, ".agents", "skills", "stale-ours"))) && await exists(J(proj, ".agents", "skills", "their-own")), "a copy of ours that is gone is removed; theirs stays");
  // an edited skill is refreshed by use() (what /name does before sending)
  await IOUtils.writeUTF8(J(skillsDir, "fruit-check", "SKILL.md"), fruit.replace("PINEAPPLE-7", "PINEAPPLE-8"));
  check((await host.skills.use("fruit-check", chatDefault)) === ".agents/skills/fruit-check/SKILL.md", "use() returns the path the agent reads");
  check((await IOUtils.readUTF8(J(chatDefault, ".agents", "skills", "fruit-check", "SKILL.md"))).includes("PINEAPPLE-8"), "an edit reaches the agent's copy");

  // 5. the / menu in the panel: scanned when it opens; pick with the keyboard
  await theme("light");
  const ta = $("textarea");
  ta.focus();
  t0 = Date.now();
  ta.value = "/"; ta.dispatchEvent(new win.Event("input", { bubbles: true }));
  await ctx.waitFor(() => $$(".pop--slash .pop__i").length, "the / menu lists items");
  out.slashOpenMs = Date.now() - t0;
  out.slashItems = $$(".pop--slash .pop__i .pop__t").map((e) => e.textContent);
  check(out.slashItems.includes("/fruit-check") && out.slashItems.includes("/create-skill") && out.slashItems.includes("Detailed summary"), "skills and prompts listed: " + out.slashItems);
  const pop = $(".pop--slash");
  out.slashBackdrop = win.getComputedStyle(pop).backdropFilter;
  check(/blur/.test(out.slashBackdrop), "the menu is frosted glass: " + out.slashBackdrop);
  const pr = pop.getBoundingClientRect(), cr = $(".composer").getBoundingClientRect();
  check(pr.left >= cr.left - 1 && pr.right <= cr.right + 1 && pr.bottom <= cr.top, "the menu sits above the composer, within its width");
  await ctx.snapshot("skills-1-slash-light");
  await theme("dark");
  await ctx.snapshot("skills-2-slash-dark");
  ta.dispatchEvent(new win.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  check(!$(".pop--slash"), "Esc closes the menu");

  // 6. /name reaches the agent as a plain message, the chips first; the transcript shows what was typed
  ta.value = "/fruit-check SCENARIO:echo"; ta.dispatchEvent(new win.Event("input", { bubbles: true }));
  await ctx.waitFor(() => !$("button.send").disabled, "send enabled");
  $("button.send").click();
  await ctx.waitFor(() => $('.msg--assistant[data-state="end_turn"]'), "the echo", 30000);
  check($(".ubub__text").textContent === "/fruit-check SCENARIO:echo", "the user bubble: " + $(".ubub__text").textContent);
  let echo = null;
  for (let i = 0; i < 40 && !echo; i++) {
    const [sv] = await host.sessions();
    const text = sv ? (await host.loadEvents(sv.id)).filter((e) => e.t === "text").map((e) => e.delta).join("") : "";
    try { echo = JSON.parse(text); } catch { await ctx.sleep(250); }
  }
  check(echo, "the mock echoed what it got");
  out.agentGot = echo.prompt;
  check(/^Use my "fruit-check" skill\. Before you answer, read \.agents\/skills\/fruit-check\/SKILL\.md in your working folder, then do what it says for this request: SCENARIO:echo$/m.test(echo.prompt) && !echo.prompt.startsWith("/"), "a plain message, not a slash command: " + echo.prompt);

  // 7. the bubbles on a new chat, a pinned skill among them
  await host.setSettings({ prompts: host.getSettings().prompts.map((p) => (p.slot === 4 ? { ...p, slot: undefined } : p)), skills: { "fruit-check": { slot: 4 } } });
  root.querySelector('button[aria-label="New chat"]').click();
  await ctx.waitFor(() => $$(".bubble").length === 4, "four bubbles on a new chat");
  out.bubbles = $$(".bubble").map((b) => b.textContent);
  check(out.bubbles[3] === "Fruit check" && $(".bubble--skill"), "the pinned skill is the fourth: " + out.bubbles);
  for (const t of ["light", "dark"]) { await theme(t); await ctx.snapshot(`skills-3-bubbles-${t}`); }

  // 8. the settings card and the add-skill preview, through the real host (the picker answered by inspect)
  host.skills.pick = () => host.skills.inspect(J(src, "SKILL.md"));
  root.querySelector('button[aria-label="Settings"]').click();
  const card = await ctx.waitFor(() => $$(".sp").length >= 5 && $("#skills"), "the Skills and prompts card");
  out.rows = $$(".sp .sp__t").map((e) => e.textContent);
  check(out.rows.slice(0, 5).join() === "/fruit-check,/import-keep,/import-test,/loose-notes,/create-skill", "the rows: " + out.rows);
  check(card.querySelector(".sp__dir").textContent.endsWith("skills") && card.querySelector(".folder").title === skillsDir, "where the skills live: " + card.querySelector(".folder").title);
  card.scrollIntoView({ block: "start" });
  for (const t of ["light", "dark"]) { await theme(t); await ctx.snapshot(`skills-4-card-${t}`); }
  [...card.querySelectorAll("button")].find((b) => /Add skill/.test(b.textContent)).click();
  const imp = await ctx.waitFor(() => $(".imp"), "the add-skill preview");
  check($(".imp__pre").textContent === picked && /trust or wrote/.test($(".imp__warn").textContent), "the full text and the safety line");
  check($$(".imp__list--skip .imp__f").length === 7, "seven files left out, each listed: " + $$(".imp__list--skip .imp__f").map((e) => e.textContent));
  imp.scrollIntoView({ block: "start" });
  for (const t of ["light", "dark"]) { await theme(t); await ctx.snapshot(`skills-5-preview-${t}`); }
  await theme("light");
  return out;
}
