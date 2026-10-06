// The user's skills folder, <profile>/zotero-chat/skills/<name>/SKILL.md, and getting a skill into the agent's folder.
// Nothing here runs at Zotero's startup: the panel scans the folder the first time the `/` menu or the settings need it,
// and a session copies the skills in when it starts. The rules (what an import copies, names, the message) are in
// ui/skills-model.ts; this file only touches disk.
//
// Copies in an agent's folder carry a marker file. The panel refreshes and removes only those (and anything in its own
// default folder); a skill folder of the same name that someone else put in their project is never touched.
import type { SkillEntry, SkillHost, SkillImport } from "../types.ts";
import { AGENT_SKILL_DIRS, CREATE_SKILL, NAME_RE, nameProblem, parseSkill, planImport, proposeSkill, skillPath, withFrontmatter } from "../ui/skills-model.ts";
import type { FoundFile } from "../ui/skills-model.ts";
import { createSkillText } from "../ui/create-skill.ts";

const MARK = ".zotero-chat";
const MARK_TEXT = "Copied here by Zotero Chat from its skills folder; changes here are replaced. Edit the skill in Zotero Chat's settings.\n";
const MAX_TEXT = 256 * 1024;
const BUILTIN: SkillEntry = { name: CREATE_SKILL, description: "Make a new skill with your agent: it asks a few questions, then writes it.", builtin: true };

const file = (p: string) => Zotero.File.pathToFile(p);
const join = (...p: string[]) => PathUtils.join(...p);

export function createSkills(o: { dataDir: string; win: any; defaultFolder(): string }) {
  const dir = join(o.dataDir, "skills");
  const cache = new Map<string, { mtime: number; entry: SkillEntry }>();
  const checked = (name: string) => { if (!NAME_RE.test(name) || name.length > 64) throw new Error(`"${name}" is not a skill name`); return join(dir, name); };

  async function list(): Promise<SkillEntry[]> {
    let kids: string[] = [];
    try { kids = await IOUtils.getChildren(dir); } catch { /* no folder yet: no skills */ }
    const out: SkillEntry[] = [];
    for (const k of kids.sort()) {
      const name = PathUtils.filename(k);
      if (!NAME_RE.test(name) || name === CREATE_SKILL) continue;
      const md = join(k, "SKILL.md");
      let mtime: number;
      try { mtime = (await IOUtils.stat(md)).lastModified ?? 0; } catch { continue; }
      let hit = cache.get(name);
      if (!hit || hit.mtime !== mtime) {
        const p = parseSkill(await IOUtils.readUTF8(md).catch(() => ""));
        hit = { mtime, entry: { name, description: p.description ?? "" } };
        cache.set(name, hit);
      }
      out.push(hit.entry);
    }
    return [...out, BUILTIN];
  }

  /** Every file under `root` (relative paths), as planImport needs them; hidden folders are listed, not entered. */
  async function walk(root: string): Promise<FoundFile[]> {
    const out: FoundFile[] = [];
    const queue: [string, number][] = [["", 0]];
    while (queue.length && out.length < 500) {
      const [rel, depth] = queue.shift()!;
      for (const child of await IOUtils.getChildren(rel ? join(root, ...rel.split("/")) : root)) {
        const name = PathUtils.filename(child);
        const path = rel ? `${rel}/${name}` : name;
        const f = file(child);
        if (f.isSymlink()) { out.push({ path, kind: "link", size: 0, exec: false, shebang: false }); continue; }
        if (f.isDirectory()) { out.push({ path, kind: "dir", size: 0, exec: false, shebang: false }); if (!name.startsWith(".") && depth < 6) queue.push([path, depth + 1]); continue; }
        if (!f.isFile()) { out.push({ path, kind: "other", size: 0, exec: false, shebang: false }); continue; }
        const head = f.fileSize >= 2 ? await IOUtils.read(child, { maxBytes: 2 }) : new Uint8Array(0);
        out.push({ path, kind: "file", size: f.fileSize, exec: (f.permissions & 0o111) !== 0, shebang: head[0] === 0x23 && head[1] === 0x21 });
      }
    }
    return out;
  }

  async function inspect(path: string): Promise<SkillImport> {
    const f = file(path);
    if (f.isSymlink() || !f.isFile()) throw new Error("pick a regular .md file, not a link or a folder");
    if (f.fileSize > MAX_TEXT) throw new Error("that file is over 256 KB, too large for a skill");
    const text = await IOUtils.readUTF8(path);
    const loose = !/^skill\.md$/i.test(PathUtils.filename(path));
    const folder = PathUtils.parent(path)!;
    const plan = loose ? { copy: [], skip: [] } : planImport(await walk(folder));
    return { source: path, loose, text, ...proposeSkill(text, loose ? PathUtils.filename(path) : PathUtils.filename(folder)), ...plan };
  }

  /** `rel` under `root` is a plain file reached through no link: checked again when copying, not only when listing. */
  function safeFile(root: string, rel: string): string | null {
    const parts = rel.split("/");
    if (parts.some((s) => !s || s === "." || s === ".." || s.startsWith("."))) return null;
    for (let i = 1; i <= parts.length; i++) {
      const f = file(join(root, ...parts.slice(0, i)));
      if (!f.exists() || f.isSymlink()) return null;
      if (i === parts.length && !f.isFile()) return null;
    }
    return join(root, ...parts);
  }

  const host: SkillHost = {
    dir,
    list,
    read: (name) => (name === CREATE_SKILL ? Promise.resolve(createSkillText(dir)) : IOUtils.readUTF8(join(checked(name), "SKILL.md"))),
    async write(name, text) {
      if (name === CREATE_SKILL) throw new Error("the built-in skill can't be changed");
      await IOUtils.writeUTF8(join(checked(name), "SKILL.md"), text);
    },
    async remove(name) {
      if (name === CREATE_SKILL) throw new Error("the built-in skill can't be deleted");
      await IOUtils.remove(checked(name), { recursive: true, ignoreAbsent: true });
      cache.delete(name);
    },
    async reveal() {
      await IOUtils.makeDirectory(dir, { createAncestors: true, ignoreExisting: true });
      Zotero.File.reveal(dir);
    },
    async pick() {
      const fp = Cc["@mozilla.org/filepicker;1"].createInstance(Ci.nsIFilePicker);
      fp.init(o.win.browsingContext, "Add a skill: its SKILL.md, or any .md file", Ci.nsIFilePicker.modeOpen);
      fp.appendFilter("Skill or Markdown", "*.md; *.markdown; *.txt");
      const result: number = await new Promise((r) => fp.open(r));
      return result === Ci.nsIFilePicker.returnOK ? inspect(fp.file.path) : null;
    },
    inspect,
    async add(plan, a) {
      const problem = nameProblem(a.name, (await list()).map((s) => s.name));
      if (problem) throw new Error(problem);
      const desc = a.description.trim();
      if (!desc) throw new Error("Give it a one-line description: the agent reads it to know when to use the skill.");
      const dst = checked(a.name);
      if (await IOUtils.exists(dst)) throw new Error(`a folder called ${a.name} is already in the skills folder`);
      await IOUtils.makeDirectory(dst, { createAncestors: true });
      try {
        // The text the user read in the preview, not the file as it is now.
        await IOUtils.writeUTF8(join(dst, "SKILL.md"), withFrontmatter(plan.text, a.name, desc));
        if (!plan.loose) {
          const root = PathUtils.parent(plan.source)!;
          for (const rel of [...plan.copy, ...(a.keepSkipped ? plan.skip.filter((s) => s.keepable).map((s) => s.path) : [])]) {
            const src = safeFile(root, rel);
            if (!src) continue;
            const to = join(dst, ...rel.split("/"));
            await IOUtils.makeDirectory(PathUtils.parent(to)!, { createAncestors: true, ignoreExisting: true });
            await IOUtils.copy(src, to);
          }
        }
      } catch (e) {
        await IOUtils.remove(dst, { recursive: true, ignoreAbsent: true });
        throw e;
      }
    },
    async use(name, cwd) {
      await syncOne(name, cwd, cwd === o.defaultFolder());
      return skillPath(name);
    },
  };

  /** Files under a skill folder with their sizes, the marker left out: enough to tell a stale copy. */
  async function signature(root: string): Promise<string> {
    const parts: string[] = [];
    const visit = async (d: string, rel: string) => {
      for (const c of (await IOUtils.getChildren(d)).sort()) {
        const n = PathUtils.filename(c);
        if (!rel && n === MARK) continue;
        const st = await IOUtils.stat(c);
        if (st.type === "directory") await visit(c, `${rel}${n}/`);
        else parts.push(`${rel}${n}:${st.size}`);
      }
    };
    await visit(root, "");
    return parts.join("\n");
  }

  /** One skill into `cwd`'s .agents/skills and .claude/skills; `refresh` (the panel's own folder) may replace any copy, elsewhere only ours. */
  async function syncOne(name: string, cwd: string, refresh: boolean): Promise<void> {
    const builtin = name === CREATE_SKILL;
    const src = builtin ? null : checked(name);
    const text = builtin ? createSkillText(dir) : await IOUtils.readUTF8(join(src!, "SKILL.md"));
    const sig = builtin ? null : await signature(src!);
    for (const sub of AGENT_SKILL_DIRS) {
      const dst = join(cwd, ...sub.split("/"), name);
      if (await IOUtils.exists(dst)) {
        if (!refresh && !(await IOUtils.exists(join(dst, MARK)))) continue; // someone else's skill of that name
        const same = (await IOUtils.readUTF8(join(dst, "SKILL.md")).catch(() => null)) === text && (builtin || (await signature(dst)) === sig);
        if (same) continue;
        await IOUtils.remove(dst, { recursive: true });
      }
      await IOUtils.makeDirectory(PathUtils.parent(dst)!, { createAncestors: true, ignoreExisting: true });
      if (builtin) {
        await IOUtils.makeDirectory(dst);
        await IOUtils.writeUTF8(join(dst, "SKILL.md"), text);
      } else {
        await IOUtils.copy(src!, dst, { recursive: true });
      }
      await IOUtils.writeUTF8(join(dst, MARK), MARK_TEXT);
    }
  }

  /** At session start: the skills that are on, copied in; copies of ours that are off or deleted, removed. */
  async function sync(cwd: string, wanted: string[]): Promise<void> {
    const refresh = cwd === o.defaultFolder();
    for (const name of wanted) await syncOne(name, cwd, refresh).catch((e) => Zotero.logError(e));
    for (const sub of AGENT_SKILL_DIRS) {
      let kids: string[] = [];
      try { kids = await IOUtils.getChildren(join(cwd, ...sub.split("/"))); } catch { continue; }
      for (const k of kids) {
        if (!wanted.includes(PathUtils.filename(k)) && (await IOUtils.exists(join(k, MARK)))) await IOUtils.remove(k, { recursive: true, ignoreAbsent: true });
      }
    }
  }

  return { host, sync };
}
