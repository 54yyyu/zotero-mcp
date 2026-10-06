// The preview's skills folder, in memory (FakeHost.skills). Playwright drives it through `window.__zmc.host.skills`:
// `nextPick` is what the file picker "returns", `used` records each skill put into an agent's folder.
import type { SkillEntry, SkillHost, SkillImport } from "../types.ts";
import { createSkillText } from "./create-skill.ts";
import { CREATE_SKILL, nameProblem, parseSkill, skillPath, withFrontmatter } from "./skills-model.ts";

const ANNOTATE = `---
name: annotate-paper
description: Read the open paper and annotate the PDF with zotero-cli - a context box on the title, a four-part summary on the abstract, one box per figure panel and colour-coded highlights. Use when asked to annotate or help me understand a paper.
---

# Annotate a paper for understanding

| Annotation | Colour |
|---|---|
| Context box on the title | orange |
| Four-part summary on the abstract | magenta |
| One box per figure panel | purple |

1. Read the whole paper with \`zotero-cli read KEY\`.
2. Plan every annotation, dry-run it, read every match.
3. Write them in one batch and count them back.
`;

const NOTE = `---
name: reading-note
description: Write a one-page reading note for the open paper (question, approach, findings, discussion points) and save it as a child note.
---

1. Read the abstract and the conclusion first.
2. Save with \`zotero-cli notes create --item-key KEY --text -\`.
`;

export const SAMPLE_PICK: SkillImport = {
  source: "/Users/you/Downloads/figure-explainer/SKILL.md",
  loose: false,
  name: "figure-explainer",
  description: "Explain every figure of the open paper panel by panel, with the numbers that matter.",
  text: "---\nname: figure-explainer\ndescription: Explain every figure of the open paper panel by panel, with the numbers that matter.\n---\n\n# Explain the figures\n\n1. List the figures with `zotero-cli outline KEY`.\n2. Render each figure page and look at it before you describe it.\n3. For each panel: what is plotted, what to notice (with numbers), why it matters.\n\nUse the palette in palette.csv for any drawing.\n",
  copy: ["examples.md", "palette.csv"],
  skip: [
    { path: ".DS_Store", reason: "hidden", keepable: false },
    { path: "helpers/render.py", reason: "a script or program", keepable: true },
    { path: "install.sh", reason: "a script or program", keepable: true },
    { path: "refs", reason: "a link (only real files are copied)", keepable: false },
  ],
};

export class FakeSkills implements SkillHost {
  readonly dir = "/Users/you/Library/Application Support/Zotero/Profiles/abc.default/zotero-chat/skills";
  files = new Map<string, string>([["annotate-paper", ANNOTATE], ["reading-note", NOTE]]);
  /** What pick() hands back next (null: the picker was cancelled). */
  nextPick: SkillImport | null = SAMPLE_PICK;
  used: { name: string; cwd: string }[] = [];
  added: { name: string; keepSkipped: boolean }[] = [];
  revealed = 0;

  async list(): Promise<SkillEntry[]> {
    const mine = [...this.files.entries()].sort(([a], [b]) => (a < b ? -1 : 1)).map(([name, text]) => ({ name, description: parseSkill(text).description ?? "" }));
    return [...mine, { name: CREATE_SKILL, description: "Make a new skill with your agent: it asks a few questions, then writes it.", builtin: true }];
  }
  async read(name: string) { return name === CREATE_SKILL ? createSkillText(this.dir) : this.files.get(name) ?? ""; }
  async write(name: string, text: string) { this.files.set(name, text); }
  async remove(name: string) { this.files.delete(name); }
  async reveal() { this.revealed++; }
  async pick() { return this.nextPick ? structuredClone(this.nextPick) : null; }
  async inspect(): Promise<SkillImport> { return structuredClone(SAMPLE_PICK); }
  async add(plan: SkillImport, o: { name: string; description: string; keepSkipped: boolean }) {
    const problem = nameProblem(o.name, [...this.files.keys()]);
    if (problem) throw new Error(problem);
    this.files.set(o.name, withFrontmatter(plan.text, o.name, o.description));
    this.added.push({ name: o.name, keepSkipped: o.keepSkipped });
  }
  async use(name: string, cwd: string) { this.used.push({ name, cwd }); return skillPath(name); }
}
