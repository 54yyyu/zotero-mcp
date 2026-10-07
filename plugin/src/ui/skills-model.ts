// Skills and prompts as pure functions: SKILL.md frontmatter, what an import copies or skips, the `/` menu's ranking,
// the message a `/name` turns into, and the four pin slots. No DOM, no Zotero: zotero/skills.ts does the files.
import type { PanelSettings, PromptEntry, SkillEntry } from "../types.ts";

// ───────────────────────────── SKILL.md ─────────────────────────────

const FRONT = /^﻿?---[ \t]*\r?\n([\s\S]*?)\r?\n---[ \t]*(?:\r?\n|$)/;

/** `name` and `description` from the frontmatter (a `>` or `|` block joined into one line), and the body after it. */
export function parseSkill(text: string): { name?: string; description?: string; body: string } {
  const m = FRONT.exec(text);
  if (!m) return { body: text };
  const meta: Record<string, string> = {};
  const lines = (m[1] as string).split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const kv = /^([A-Za-z_][\w-]*):[ \t]*(.*)$/.exec(lines[i] as string);
    if (!kv) continue;
    let v = (kv[2] as string).trim();
    if (/^[>|][+-]?$/.test(v) || v === "") {
      const block: string[] = [];
      while (i + 1 < lines.length && /^[ \t]+\S/.test(lines[i + 1] as string)) block.push((lines[++i] as string).trim());
      v = block.join(" ");
    }
    meta[kv[1] as string] = unquote(v);
  }
  return { ...(meta["name"] ? { name: meta["name"] } : {}), ...(meta["description"] ? { description: meta["description"] } : {}), body: text.slice(m[0].length) };
}

function unquote(v: string): string {
  if (v.length >= 2 && v.startsWith('"') && v.endsWith('"')) { try { return JSON.parse(v) as string; } catch { return v.slice(1, -1); } }
  if (v.length >= 2 && v.startsWith("'") && v.endsWith("'")) return v.slice(1, -1).replace(/''/g, "'");
  return v;
}

/** A YAML scalar that stays one line and means what it says. */
const yaml = (v: string): string => (/^[\w(][^:#\n]*$/.test(v) && !/\s$/.test(v) ? v : JSON.stringify(v));

/** `text` with this name and description in its frontmatter (other keys kept; a frontmatter added if it had none). */
export function withFrontmatter(text: string, name: string, description: string): string {
  const desc = description.replace(/\s+/g, " ").trim();
  const m = FRONT.exec(text);
  if (!m) return `---\nname: ${name}\ndescription: ${yaml(desc)}\n---\n\n${text.replace(/^﻿/, "")}`;
  const kept: string[] = [];
  const lines = (m[1] as string).split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const key = /^([A-Za-z_][\w-]*):/.exec(lines[i] as string)?.[1];
    if (key === "name" || key === "description") { while (i + 1 < lines.length && /^[ \t]+\S/.test(lines[i + 1] as string)) i++; continue; }
    kept.push(lines[i] as string);
  }
  return `---\nname: ${name}\ndescription: ${yaml(desc)}\n${kept.length ? kept.join("\n") + "\n" : ""}---\n${text.slice(m[0].length)}`;
}

/** A skill's name: lowercase words joined by hyphens, as Claude, Codex and pi all accept. */
export const NAME_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
/** Taken by the panel itself: the zotero-cli skill and the built-in one. */
export const RESERVED = new Set(["zotero-cli", "create-skill"]);
export const CREATE_SKILL = "create-skill";

export function nameProblem(name: string, taken: string[] = []): string | null {
  if (!name) return "Give it a name.";
  if (name.length > 64 || !NAME_RE.test(name)) return "Use lowercase letters, digits and single hyphens (at most 64), like annotate-paper.";
  if (RESERVED.has(name)) return `“${name}” is the panel's own; choose another name.`;
  if (taken.includes(name)) return `You already have a skill called “${name}”.`;
  return null;
}

export const slug = (s: string): string => s.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 64).replace(/-+$/, "");

/** A name and a one-line description proposed for a picked file: its frontmatter, else its first heading / paragraph, else the file name. */
export function proposeSkill(text: string, fileName: string): { name: string; description: string } {
  const p = parseSkill(text);
  const heading = /^#{1,3}\s+(.+?)\s*#*\s*$/m.exec(p.body)?.[1];
  const base = fileName.replace(/\.(md|markdown|txt)$/i, "");
  const name = slug(p.name ?? "") || slug(/^skill$/i.test(base) ? heading ?? "" : base) || slug(heading ?? "") || "my-skill";
  let description = p.description ?? "";
  if (!description) {
    const para = p.body.split(/\n\s*\n/).map((b) => b.trim()).find((b) => b && !/^(#|```|[-*|>]|\d+\.)/.test(b)) ?? "";
    const flat = para.replace(/\s+/g, " ").replace(/[*_`]/g, "");
    const sentence = /^.{20,180}?[.!?](?=\s|$)/.exec(flat)?.[0] ?? flat.slice(0, 160);
    description = sentence.trim();
  }
  return { name, description };
}

// ───────────────────────────── import: what is copied ─────────────────────────────

/** One thing found under a picked skill's folder (`path` relative, "/"-separated), as the host saw it on disk. */
export interface FoundFile { path: string; kind: "file" | "dir" | "link" | "other"; size: number; exec: boolean; shebang: boolean }

/** Reference files a skill may point to. Anything else is skipped; scripts and unknown files can be kept after a warning. */
const PLAIN = /\.(md|markdown|txt|csv|tsv|json|ya?ml|png|jpe?g|gif|webp|pdf)$/i;
const SCRIPT = /\.(sh|bash|zsh|fish|command|py|pyc|js|mjs|cjs|ts|rb|pl|php|lua|ps1|bat|cmd|exe|app|jar|scpt|applescript|bin|so|dylib|dll|wasm)$/i;
export const MAX_FILE = 2 * 1024 * 1024;
export const MAX_FILES = 200;

/** What goes in and what stays out, with a reason for each file left out. SKILL.md itself is written by the panel. */
export function planImport(found: FoundFile[]): { copy: string[]; skip: { path: string; reason: string; keepable: boolean }[] } {
  const copy: string[] = [];
  const skip: { path: string; reason: string; keepable: boolean }[] = [];
  const out = (path: string, reason: string, keepable = false) => skip.push({ path, reason, keepable });
  for (const f of [...found].sort((a, b) => (a.path < b.path ? -1 : 1))) {
    const parts = f.path.split("/");
    if (!f.path || f.path.startsWith("/") || f.path.includes("\\") || parts.some((s) => s === ".." || s === "" || s === ".")) { out(f.path, "outside the skill's folder"); continue; }
    if (f.path === "SKILL.md") continue;
    if (parts.some((s) => s.startsWith("."))) { out(f.kind === "dir" ? `${f.path}/` : f.path, "hidden"); continue; }
    if (f.kind === "dir") continue;
    if (f.kind === "link") { out(f.path, "a link (only real files are copied)"); continue; }
    if (f.kind !== "file") { out(f.path, "not a regular file"); continue; }
    if (copy.length + skip.filter((s) => s.keepable).length >= MAX_FILES) { out(f.path, `more than ${MAX_FILES} files`); continue; }
    if (f.size > MAX_FILE) { out(f.path, "larger than 2 MB"); continue; }
    if (f.exec || f.shebang || SCRIPT.test(f.path)) { out(f.path, "a script or program", true); continue; }
    if (!PLAIN.test(f.path)) { out(f.path, "not a plain text or image file", true); continue; }
    copy.push(f.path);
  }
  return { copy, skip };
}

// ───────────────────────────── invoking a skill ─────────────────────────────

/** Where the panel puts a skill in the agent's folder: the shared place Codex reads, and Claude's own. */
export const AGENT_SKILL_DIRS = [".agents/skills", ".claude/skills"] as const;
export const skillPath = (name: string): string => `${AGENT_SKILL_DIRS[0]}/${name}/SKILL.md`;

/** `/name rest`: the skill's name and what follows it. Only names, never paths. */
export function parseInvocation(text: string): { name: string; rest: string } | null {
  const m = /^\/([a-z0-9]+(?:-[a-z0-9]+)*)(?:\s+([\s\S]*))?$/.exec(text.trim());
  return m && (m[1] as string).length <= 64 ? { name: m[1] as string, rest: (m[2] ?? "").trim() } : null;
}

/**
 * What the agent reads for `/name rest`: a plain message, the same on every agent (no bridge's own skill command), naming
 * the file to read first. The context chips go before it as usual.
 */
export function invocationText(name: string, rest: string, path = skillPath(name)): string {
  return `Use my "${name}" skill. Before you answer, read ${path} in your working folder, then do what it says for this request: ${rest || "what I have open."}`;
}

// ───────────────────────────── the `/` menu ─────────────────────────────

/** How well `q` matches `text`: a prefix beats a word start, which beats anywhere, which beats letters in order; 0 = no match. */
export function fuzzyScore(q: string, text: string): number {
  const t = text.toLowerCase();
  const n = q.toLowerCase().trim();
  if (!n) return 1;
  if (t.startsWith(n)) return 100 - Math.min(20, t.length - n.length) * 0.1;
  const at = t.indexOf(n);
  if (at > 0) return /[^a-z0-9]/.test(t[at - 1] as string) ? 80 : 60;
  // Letters in order, the first one starting a word ("anp" finds annotate-paper; "ann" does not find reading-note).
  for (let s = 0; s < t.length; s++) {
    if (t[s] !== n[0] || (s > 0 && /[a-z0-9]/.test(t[s - 1] as string))) continue;
    let i = 0, run = 0, best = 0;
    for (let k = s; k < t.length && i < n.length; k++) {
      if (t[k] === n[i]) { i++; run++; best = Math.max(best, run); } else run = 0;
    }
    if (i === n.length) return 20 + Math.min(best, 10);
  }
  return 0;
}

/** Items that match, best first (a label match counts fully, words in the detail half); the original order breaks ties. */
export function rankItems<T extends { label: string; detail?: string }>(items: T[], q: string): T[] {
  if (!q.trim()) return items;
  return items.map((it, i) => ({ it, i, s: Math.max(fuzzyScore(q, it.label), it.detail && it.detail.toLowerCase().includes(q.toLowerCase().trim()) ? 30 : 0) }))
    .filter((x) => x.s > 0).sort((a, b) => b.s - a.s || a.i - b.i).map((x) => x.it);
}

/** "annotate-paper" → "Annotate paper": a skill's name as a button label. */
export const skillLabel = (name: string): string => { const t = name.replace(/-/g, " "); return t.charAt(0).toUpperCase() + t.slice(1); };

// ───────────────────────────── pins and on/off ─────────────────────────────

export type ItemRef = { kind: "prompt"; id: string } | { kind: "skill"; name: string };
export const PIN_SLOTS = 4;

const skillState = (s: PanelSettings, name: string) => s.skills?.[name] ?? {};
export const slotOf = (s: PanelSettings, r: ItemRef): number | undefined => (r.kind === "prompt" ? s.prompts.find((p) => p.id === r.id)?.slot : skillState(s, r.name).slot);
export const isOn = (s: PanelSettings, r: ItemRef): boolean => (r.kind === "prompt" ? !s.prompts.find((p) => p.id === r.id)?.off : !skillState(s, r.name).off);

/** The slots in use: prompts' and skills' share 1-4 (a skill no longer in the folder still holds its slot until unpinned). */
export function usedSlots(s: PanelSettings): Set<number> {
  const used = new Set<number>();
  for (const p of s.prompts) if (p.slot) used.add(p.slot);
  for (const v of Object.values(s.skills ?? {})) if (v.slot) used.add(v.slot);
  return used;
}

function withSlot(s: PanelSettings, r: ItemRef, patch: { slot?: number; off?: boolean }): Partial<PanelSettings> {
  const clean = <T extends { slot?: number; off?: boolean }>(o: T): T => {
    const x = { ...o, ...patch };
    if (!x.slot) delete x.slot;
    if (!x.off) delete x.off;
    return x;
  };
  if (r.kind === "prompt") return { prompts: s.prompts.map((p) => (p.id === r.id ? clean(p) : p)) };
  const next = clean(skillState(s, r.name));
  const skills = { ...(s.skills ?? {}) };
  if (Object.keys(next).length) skills[r.name] = next; else delete skills[r.name];
  return { skills };
}

/** Pin to the lowest free slot, or unpin. null: all four slots are taken. */
export function togglePin(s: PanelSettings, r: ItemRef): Partial<PanelSettings> | null {
  if (slotOf(s, r)) return withSlot(s, r, { slot: 0 });
  const used = usedSlots(s);
  const free = [1, 2, 3, 4].find((n) => !used.has(n));
  return free ? withSlot(s, r, { slot: free, off: false }) : null;
}

/** Turning an item off also unpins it (an off item has no button). */
export const setOn = (s: PanelSettings, r: ItemRef, on: boolean): Partial<PanelSettings> => withSlot(s, r, on ? { off: false } : { off: true, slot: 0 });

/** A skill's settings go with it: deleting forgets its pin. */
export function forgetSkill(s: PanelSettings, name: string): Partial<PanelSettings> {
  const skills = { ...(s.skills ?? {}) };
  delete skills[name];
  return { skills };
}

export type Pinned = { slot: number } & ({ kind: "prompt"; prompt: PromptEntry } | { kind: "skill"; name: string });

/** The pinned items in slot order: the buttons on a new chat and the shortcuts. A pinned skill needs no scan to show (its label is its name). */
export function pinned(s: PanelSettings): Pinned[] {
  const out: Pinned[] = [];
  for (const p of s.prompts) if (p.slot && !p.off && (p.title || p.text)) out.push({ slot: p.slot, kind: "prompt", prompt: p });
  for (const [name, v] of Object.entries(s.skills ?? {})) if (v.slot && !v.off) out.push({ slot: v.slot, kind: "skill", name });
  return out.sort((a, b) => a.slot - b.slot).slice(0, PIN_SLOTS);
}

/** The skills that are on, in the folder's order. */
export const enabledSkills = (s: PanelSettings, all: SkillEntry[]): SkillEntry[] => all.filter((k) => !skillState(s, k.name).off);
