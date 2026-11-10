// A tool step's title as a person reads it. The agent's raw title is a shell line
// (`cd /tmp && zotero-cli --json read KEY --start-page 7 --end-page 8`); the row shows "Read pages 7–8 of KEY"
// and keeps the raw line in its tooltip and its expanded input. PURE.
import { t } from "../i18n.ts";
import type { Key, Vars } from "../i18n.ts";

/** Flags of zotero-cli that take no value (everything else after `--x` takes the next word). */
const BOOL = new Set(["--json", "--all-libraries", "--no-abstract", "--allow-note", "--force"]);

/** Shell words of a simple command, or null if it has pipes, redirects, `;`, `&&`, substitutions. */
function words(line: string): string[] | null {
  const cmd = line.trim();
  const out: string[] = [];
  const re = /\s*(?:"((?:\\.|[^"\\])*)"|'([^']*)'|([^\s"'`$;&|<>()]+))/y;
  let m: RegExpExecArray | null;
  while (re.lastIndex < cmd.length && (m = re.exec(cmd))) out.push(m[1] ?? m[2] ?? m[3] ?? "");
  return re.lastIndex === cmd.length ? out : null;
}

/** A zotero-cli call whose meaning is clear, as a message key and its values ("step.readPages" {from 7, to 8, key}), else null. */
export type CliStep = [Key, Vars];

function cliStep(cmd: string): CliStep | null {
  const w = words(cmd.replace(/\s+2>&1\s*$/, ""));
  if (!w || w[0] !== "zotero-cli") return null;
  const pos: string[] = [];
  const opt: Record<string, string> = {};
  for (let i = 1; i < w.length; i++) {
    const x = w[i]!;
    if (!x.startsWith("--")) { pos.push(x); continue; }
    const eq = x.indexOf("=");
    if (eq > 0) opt[x.slice(0, eq)] = x.slice(eq + 1);
    else if (!BOOL.has(x) && i + 1 < w.length && !w[i + 1]!.startsWith("--")) opt[x] = w[++i]!;
    else opt[x] = "";
  }
  const [cmdName, a, b] = pos;
  const q = (s: string) => (s.length > 60 ? s.slice(0, 59) + "…" : s);
  switch (cmdName) {
    case "read": {
      const from = opt["--start-page"], to = opt["--end-page"];
      if (!a || !from) return null;
      const img = opt["--format"] === "image";
      return to && to !== from ? [img ? "step.readPagesImages" : "step.readPages", { from, to, key: a }] : [img ? "step.readPageImages" : "step.readPage", { from, key: a }];
    }
    case "search": case "s":
      return a ? [opt["--mode"] === "semantic" ? "step.semantic" : "step.search", { q: q(a) }] : null;
    case "get": case "g": {
      const what: Record<string, Key> = { metadata: "step.getMetadata", fulltext: "step.getFulltext", bibtex: "step.getBibtex", children: "step.getChildren" };
      if (b && what[a!]) return [what[a!]!, { key: b }];
      const list: Record<string, Key> = { collections: "step.listCollections", tags: "step.listTags", recent: "step.listRecent", libraries: "step.listLibraries" };
      if (list[a!]) return [list[a!]!, {}];
      return a === "collection-items" && b ? ["step.listCollectionItems", { key: b }] : null;
    }
    case "annotations": case "ann": case "notes": case "n": {
      const ann = cmdName.startsWith("a");
      if (a === "list") return b ? [ann ? "step.listAnnotationsOf" : "step.listNotesOf", { key: b }] : [ann ? "step.listAnnotations" : "step.listNotes", {}];
      return a === "create" && b ? [ann ? "step.addAnnotation" : "step.addNote", { key: b }] : null;
    }
    case "outline": return a ? ["step.outline", { key: a }] : null;
    case "layout": return a ? ["step.layout", { key: a }] : null;
    case "open": {
      if (opt["--annotation"]) return ["step.openAnnotation", { key: opt["--annotation"] }];
      return a ? (opt["--page"] ? ["step.openPage", { key: a, page: opt["--page"] }] : ["step.open", { key: a }]) : null;
    }
    default: return null;
  }
}

/** The step's command tidied (`cd DIR &&` dropped, the home folder as ~, skill paths relative) and, for zotero-cli, what it does. */
export function stepOf(raw: string): { cmd: string; step: CliStep | null } {
  let s = raw.trim().replace(/^`([^`]*)`$/, "$1");
  while (/^cd\s/.test(s)) {
    const m = /^cd\s+(?:"[^"]*"|'[^']*'|\S+)\s*&&\s*/.exec(s);
    if (!m) break;
    s = s.slice(m[0].length);
  }
  s = s
    .replace(/(^|[\s"'=:(])\/(?:Users|home)\/[^/\s"']+(?=[/\s"']|$)/g, "$1~")
    .replace(/(["'])[^"'\n]*?\/(\.(?:agents|claude)\/sk)/g, "$1$2")
    // unquoted, a path may hold a space when the next word is not another path or a flag ("Read ~/Zotero Chat/...")
    .replace(/(^|\s)[~/](?:[^\s"';&|<>]| (?![-/~]))*?\/(\.(?:agents|claude)\/sk)/g, "$1$2");
  return { cmd: s, step: cliStep(s) };
}

/** The step row's title: the zotero-cli summary in the user's language, else the tidied command. */
export function stepTitle(raw: string): string {
  const { cmd, step } = stepOf(raw);
  return step ? t(...step) : cmd;
}
