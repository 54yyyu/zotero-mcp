import { test } from "node:test";
import assert from "node:assert/strict";
import { defaultSettings } from "../../src/ui/fake-catalog.ts";
import { withDefaults } from "../../src/zotero/defaults.ts";
import { CREATE_SKILL_MD, createSkillText } from "../../src/ui/create-skill.ts";
import {
  enabledSkills, fuzzyScore, invocationText, nameProblem, parseInvocation, parseSkill, pinned, planImport, proposeSkill, rankItems,
  setOn, skillLabel, skillPath, slotOf, togglePin, usedSlots, withFrontmatter,
} from "../../src/ui/skills-model.ts";
import type { FoundFile } from "../../src/ui/skills-model.ts";
import type { PanelSettings } from "../../src/types.ts";

test("frontmatter: name and description, quoted and folded values, CRLF, a BOM; no frontmatter is all body", () => {
  assert.deepEqual(parseSkill("---\nname: annotate-paper\ndescription: Read and annotate a paper.\n---\n\n# Body\n"), { name: "annotate-paper", description: "Read and annotate a paper.", body: "\n# Body\n" });
  assert.equal(parseSkill('---\nname: x\ndescription: "Has: a colon"\n---\n').description, "Has: a colon");
  assert.equal(parseSkill("---\nname: x\ndescription: 'It''s quoted'\n---\n").description, "It's quoted");
  assert.equal(parseSkill("---\nname: x\ndescription: >\n  Folded over\n  two lines.\nother: 1\n---\nbody").description, "Folded over two lines.");
  assert.equal(parseSkill("﻿---\r\nname: crlf\r\ndescription: Windows file.\r\n---\r\nbody").name, "crlf");
  assert.deepEqual(parseSkill("# Just markdown\n\nNo frontmatter."), { body: "# Just markdown\n\nNo frontmatter." });
  assert.deepEqual(parseSkill("---\nnot closed\n"), { body: "---\nnot closed\n" });
  // the user's real skill parses: folder name and description
  const real = "---\nname: annotate-paper\ndescription: Read a paper in Steven's Zotero library and write study annotations into the PDF with zotero-cli - an orange context box. Use when asked to \"annotate\".\n---\n";
  assert.match(parseSkill(real).description!, /^Read a paper .* Use when asked to "annotate"\.$/);
});

test("withFrontmatter: sets name and description, keeps other keys and the body, adds a frontmatter when there is none", () => {
  const out = withFrontmatter("---\nname: old\nlicense: MIT\ndescription: >\n  old\n  folded\n---\n# Body\n", "new-name", "Does: things");
  assert.equal(out, '---\nname: new-name\ndescription: "Does: things"\nlicense: MIT\n---\n# Body\n');
  assert.deepEqual(parseSkill(out), { name: "new-name", description: "Does: things", body: "# Body\n" });
  const loose = withFrontmatter("# Notes\n\nPlain text.", "notes", "Write notes  across\nlines.");
  assert.equal(loose, "---\nname: notes\ndescription: Write notes across lines.\n---\n\n# Notes\n\nPlain text.");
});

test("names: lowercase-hyphen, at most 64, not the panel's own, not taken", () => {
  assert.equal(nameProblem("annotate-paper"), null);
  for (const bad of ["", "Annotate", "a_b", "-a", "a--b", "a/b", "../x", "x".repeat(65)]) assert.ok(nameProblem(bad), bad);
  assert.match(nameProblem("zotero-cli")!, /panel's own/);
  assert.match(nameProblem("create-skill")!, /panel's own/);
  assert.match(nameProblem("mine", ["mine"])!, /already/);
});

test("a picked file proposes a name and a one-line description", () => {
  assert.deepEqual(proposeSkill("---\nname: From Front\ndescription: Given.\n---\nx", "whatever.md"), { name: "from-front", description: "Given." });
  assert.deepEqual(proposeSkill("# Weekly Tagging\n\nTag every paper I read this week with its course week. Then list them.\n", "notes.md"),
    { name: "notes", description: "Tag every paper I read this week with its course week." });
  assert.equal(proposeSkill("# Weekly Tagging\n\nx", "SKILL.md").name, "weekly-tagging", "a SKILL.md without a name: its heading");
  assert.equal(proposeSkill("text", "Résumé Helper.md").name, "resume-helper");
});

test("import: plain reference files copy; scripts, executables and unknown files are skipped but keepable; links, hidden, traversal and big files never", () => {
  const f = (path: string, o: Partial<FoundFile> = {}): FoundFile => ({ path, kind: "file", size: 100, exec: false, shebang: false, ...o });
  const plan = planImport([
    f("SKILL.md"), f("reference.md"), f("colours.csv"), f("img/example.png"), f("data.json"),
    f("run.sh"), f("helper.py"), f("tool", { exec: true }), f("noext", { shebang: true }), f("archive.zip"),
    f("evil", { kind: "link" }), f(".DS_Store"), f(".git", { kind: "dir" }), f("sub/.hidden.md"), f("../outside.md"), f("/etc/passwd"), f("a/../../b.md"),
    f("big.md", { size: 3 * 1024 * 1024 }), f("img", { kind: "dir" }), f("fifo", { kind: "other" }),
  ]);
  assert.deepEqual(plan.copy, ["colours.csv", "data.json", "img/example.png", "reference.md"]);
  const why = Object.fromEntries(plan.skip.map((s) => [s.path, `${s.keepable ? "keep?" : "never"}: ${s.reason}`]));
  assert.deepEqual(why, {
    "run.sh": "keep?: a script or program", "helper.py": "keep?: a script or program", tool: "keep?: a script or program", noext: "keep?: a script or program",
    "archive.zip": "keep?: not a plain text or image file",
    evil: "never: a link (only real files are copied)", ".DS_Store": "never: hidden", ".git/": "never: hidden", "sub/.hidden.md": "never: hidden",
    "../outside.md": "never: outside the skill's folder", "/etc/passwd": "never: outside the skill's folder", "a/../../b.md": "never: outside the skill's folder",
    "big.md": "never: larger than 2 MB", fifo: "never: not a regular file",
  });
  assert.ok(!plan.copy.includes("SKILL.md") && !plan.skip.some((s) => s.path === "SKILL.md"), "SKILL.md is written by the panel, not copied");
});

test("import: at most 200 files", () => {
  const many = Array.from({ length: 250 }, (_, i): FoundFile => ({ path: `n${String(i).padStart(3, "0")}.md`, kind: "file", size: 1, exec: false, shebang: false }));
  const plan = planImport(many);
  assert.equal(plan.copy.length, 200);
  assert.equal(plan.skip.length, 50);
  assert.ok(plan.skip.every((s) => /more than 200/.test(s.reason) && !s.keepable));
});

test("invocation: /name rest becomes a plain message naming the file; anything else is not a skill", () => {
  assert.deepEqual(parseInvocation("/annotate-paper"), { name: "annotate-paper", rest: "" });
  assert.deepEqual(parseInvocation("  /annotate-paper  focus on figure 2\nand 3 "), { name: "annotate-paper", rest: "focus on figure 2\nand 3" });
  for (const no of ["annotate-paper", "/Annotate", "/../etc/passwd", "/a/b", "hi /x", "/", "/-x"]) assert.equal(parseInvocation(no), null, no);
  assert.equal(skillPath("x"), ".agents/skills/x/SKILL.md");
  assert.equal(invocationText("annotate-paper", ""), 'Use my "annotate-paper" skill. Before you answer, read .agents/skills/annotate-paper/SKILL.md in your working folder, then do what it says for this request: what I have open.');
  assert.match(invocationText("annotate-paper", "only the figures"), /for this request: only the figures$/);
});

test("the / menu ranks a prefix over a word start over a substring over letters in order; the detail counts less", () => {
  assert.ok(fuzzyScore("ann", "/annotate-paper".slice(1)) > fuzzyScore("pap", "annotate-paper"));
  assert.ok(fuzzyScore("pap", "annotate-paper") > fuzzyScore("tat", "annotate-paper"));
  assert.ok(fuzzyScore("tat", "annotate-paper") > fuzzyScore("anp", "annotate-paper"));
  assert.equal(fuzzyScore("xyz", "annotate-paper"), 0);
  const items = [
    { label: "Short summary", detail: "Summarize this paper in five sentences." },
    { label: "annotate-paper", detail: "Annotate the PDF" },
    { label: "Summarise now", detail: "free up context" },
    { label: "reading-note", detail: "a summary note" },
  ];
  assert.deepEqual(rankItems(items, "sum").map((x) => x.label), ["Summarise now", "Short summary", "reading-note"], "prefix, word start, then a detail match");
  assert.deepEqual(rankItems(items, "an").map((x) => x.label)[0], "annotate-paper");
  assert.deepEqual(rankItems(items, "").map((x) => x.label), items.map((x) => x.label), "no query: everything, in order");
  assert.deepEqual(rankItems(items, "zzz"), []);
});

test("pins: four slots shared by prompts and skills; full refuses; unpin frees; turning off unpins", () => {
  let s: PanelSettings = { ...defaultSettings(), skills: {} };
  const apply = (p: Partial<PanelSettings> | null) => { assert.ok(p); s = { ...s, ...p }; };
  assert.deepEqual([...usedSlots(s)].sort(), [1, 2, 3, 4], "the default prompts hold all four");
  assert.equal(togglePin(s, { kind: "skill", name: "annotate-paper" }), null, "full");
  apply(togglePin(s, { kind: "prompt", id: "p2" }));
  assert.equal(slotOf(s, { kind: "prompt", id: "p2" }), undefined);
  apply(togglePin(s, { kind: "skill", name: "annotate-paper" }));
  assert.equal(slotOf(s, { kind: "skill", name: "annotate-paper" }), 2, "the lowest free slot");
  assert.deepEqual(pinned(s).map((p) => (p.kind === "prompt" ? p.prompt.id : p.name)), ["p1", "annotate-paper", "p3", "p4"], "slot order");
  apply(setOn(s, { kind: "prompt", id: "p3" }, false));
  assert.equal(slotOf(s, { kind: "prompt", id: "p3" }), undefined, "off unpins");
  assert.equal(s.prompts.find((p) => p.id === "p3")?.off, true);
  apply(setOn(s, { kind: "prompt", id: "p3" }, true));
  assert.equal("off" in s.prompts.find((p) => p.id === "p3")!, false, "on again leaves no stray key");
  apply(togglePin(s, { kind: "skill", name: "annotate-paper" }));
  assert.deepEqual(s.skills, {}, "an unpinned, on skill leaves no entry behind");
  apply(setOn(s, { kind: "skill", name: "reading-note" }, false));
  assert.deepEqual(enabledSkills(s, [{ name: "reading-note", description: "" }, { name: "annotate-paper", description: "" }]).map((k) => k.name), ["annotate-paper"]);
  assert.equal(skillLabel("annotate-paper"), "Annotate paper");
});

test("migration: saved prompts from before skills come through unchanged, with an empty skills map", () => {
  const old = [{ id: "a", title: "Mine", text: "My own prompt", slot: 2 }, { id: "b", title: "", text: "No title" }];
  const s = withDefaults({ prompts: old });
  assert.deepEqual(s.prompts, old);
  assert.deepEqual(s.skills, {});
  assert.deepEqual(pinned(s).map((p) => p.slot), [2]);
  assert.deepEqual(withDefaults({ skills: "junk" as never }).skills, {});
});

test("the built-in create-skill: valid frontmatter, under 60 lines, and the skills folder filled in everywhere", () => {
  const p = parseSkill(CREATE_SKILL_MD);
  assert.equal(p.name, "create-skill");
  assert.ok(p.description && p.description.length < 300);
  assert.ok(CREATE_SKILL_MD.split("\n").length < 60);
  const t = createSkillText("/x/zotero-chat/skills");
  assert.ok(!t.includes("{{"), "no placeholder left");
  assert.equal(t.split("/x/zotero-chat/skills").length - 1, 2, "where the skills live, and where to save the new one");
  assert.match(t, /zotero-cli skill/);
  assert.match(t, /prompt/i);
});
