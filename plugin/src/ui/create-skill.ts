// The built-in create-skill skill. The host writes it into the agent's folder with SKILLS_DIR filled in, so the agent
// knows where the user's skills live even when it picks the skill up by itself (Claude and Codex list workspace skills).

export const SKILLS_DIR_MARK = "{{SKILLS_DIR}}";

export const CREATE_SKILL_MD = `---
name: create-skill
description: Write a new skill for Zotero Agent (the chat panel in Zotero) with the user - a SKILL.md of plain, numbered instructions the agent follows with zotero-cli. Use when the user asks to create, make or write a skill, or types /create-skill.
---

# Create a skill

A skill is a folder with one \`SKILL.md\`: a frontmatter with \`name\` and \`description\`, then plain instructions.
No scripts: \`zotero-cli\` and your shell are the tools. The user's skills live in

    ${SKILLS_DIR_MARK}

The panel lists every folder there. Save new skills only there.

## 1. Interview, briefly

Ask in one message, and skip what the user already said:

1. What task, on what (the open paper, a selection, a collection, every item with a tag)?
2. What should the result look like (annotations, a note, tags, a table in the chat), and where does it go?
3. Rules: colours, format, length, tone.
4. Which tags or collections should it read or write?
5. One example of a good result, if they have one (an item key, or pasted text).

## 2. Check what zotero-cli can do

Read the zotero-cli skill in your working folder (\`.agents/skills/zotero-cli/SKILL.md\`) and its reference before you write
any command. Use only commands and flags that exist there; if a step needs something zotero-cli cannot do, say so.

## 3. Skill or prompt?

If it fits in one or two sentences with no steps, rules or checks ("Summarise this paper in five bullets"), it is better
as a prompt. Say so, give the text in a code block, and tell the user to add it under Settings > Skills and prompts >
New prompt. Write no file.

## 4. Write SKILL.md

- \`name\`: lowercase words joined by hyphens, at most 64 characters, not already a folder above.
- \`description\`: one sentence on what it does and when to use it, with the words a user would say ("annotate",
  "tag by week"). The description decides when the skill is picked, so make it specific.
- Body: a one-line goal; the user's rules (colours, formats) as a small table near the top, easy to change; then numbered
  steps. Each step says what to run (zotero-cli commands in code blocks) and how to check it worked before going on (a
  dry run, a count, reading back what was written). End with Gotchas if there are any.
- Plain words, no filler, under about 150 lines.

Save it as \`${SKILLS_DIR_MARK}/<name>/SKILL.md\` (make the folder). Never overwrite an existing skill without asking.

## 5. Suggest a test

Suggest trying it on one paper: open it, type \`/<name>\` in the panel. Offer to adjust the skill after that first run.
`;

export const createSkillText = (skillsDir: string): string => CREATE_SKILL_MD.split(SKILLS_DIR_MARK).join(skillsDir);
