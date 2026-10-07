// A tool step's title as a person reads it. The agent's raw title is a shell line
// (`cd /tmp && zotero-cli --json read KEY --start-page 7 --end-page 8`); the row shows "Read pages 7–8 of KEY"
// and keeps the raw line in its tooltip and its expanded input. PURE.

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

/** "Read pages 7–8 of KEY" for the zotero-cli calls whose meaning is clear, else null. */
function cliSummary(cmd: string): string | null {
  const w = words(cmd.replace(/\s+2>&1\s*$/, ""));
  if (!w || w[0] !== "zotero-cli") return null;
  const pos: string[] = [];
  const opt: Record<string, string> = {};
  for (let i = 1; i < w.length; i++) {
    const t = w[i]!;
    if (!t.startsWith("--")) { pos.push(t); continue; }
    const eq = t.indexOf("=");
    if (eq > 0) opt[t.slice(0, eq)] = t.slice(eq + 1);
    else if (!BOOL.has(t) && i + 1 < w.length && !w[i + 1]!.startsWith("--")) opt[t] = w[++i]!;
    else opt[t] = "";
  }
  const [cmdName, a, b] = pos;
  const q = (s: string | undefined) => (s ? `“${s.length > 60 ? s.slice(0, 59) + "…" : s}”` : "");
  switch (cmdName) {
    case "read": {
      const from = opt["--start-page"], to = opt["--end-page"];
      if (!a || !from) return null;
      const pages = to && to !== from ? `pages ${from}–${to}` : `page ${from}`;
      return `Read ${pages} of ${a}${opt["--format"] === "image" ? " as images" : ""}`;
    }
    case "search": case "s":
      return a ? `${opt["--mode"] === "semantic" ? "Semantic search" : "Search library"} for ${q(a)}` : null;
    case "get": case "g": {
      const what: Record<string, string> = { metadata: "metadata", fulltext: "full text", bibtex: "BibTeX", children: "attachments and notes" };
      if (b && what[a!]) return `Get ${what[a!]} of ${b}`;
      const list: Record<string, string> = { collections: "collections", tags: "tags", recent: "recent items", libraries: "libraries" };
      if (list[a!]) return `List ${list[a!]}`;
      return a === "collection-items" && b ? `List items in collection ${b}` : null;
    }
    case "annotations": case "ann": case "notes": case "n": {
      const kind = cmdName.startsWith("a") ? "annotations" : "notes";
      if (a === "list") return b ? `List ${kind} of ${b}` : `List ${kind}`;
      return a === "create" && b ? `Add ${kind === "notes" ? "a note" : "an annotation"} to ${b}` : null;
    }
    case "outline": return a ? `Get the outline of ${a}` : null;
    case "layout": return a ? `Find figures and tables in ${a}` : null;
    case "open": {
      if (opt["--annotation"]) return `Open annotation ${opt["--annotation"]}`;
      return a ? `Open ${a}${opt["--page"] ? ` at page ${opt["--page"]}` : ""}` : null;
    }
    default: return null;
  }
}

/** The step row's title: `cd DIR &&` dropped, the home folder as ~, skill paths relative, zotero-cli summarized. */
export function stepTitle(raw: string): string {
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
  return cliSummary(s) ?? s;
}
