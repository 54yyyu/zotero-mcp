// The paper the user is on, ready before the agent asks (DESIGN.md "The open paper, ahead of time"): its metadata goes
// in the <zotero-context> block (read in-process, no CLI), and its full text is extracted in the background (Zotero's
// PDF worker) into a page-marked file the agent greps, never into the conversation. Both are chips of kind "paper", so
// the delta rules send each once per paper per session, and again when it changes. Nothing here runs at Zotero's startup.
import type { ContextChip } from "../types.ts";
import { itemLabel, viewStats } from "./context.ts";

/** The longest the first message waits for the file once the session is up; later turns name it when it is ready. */
export const WAIT_MS = 1500;
/** The cache: files unused for 60 days go, then the least recently used beyond 200 MB. */
const MAX_BYTES = 200 * 1024 * 1024;
const MAX_AGE_MS = 60 * 86_400_000;
/** A failed extraction is tried again after a minute, not on every turn. */
const RETRY_MS = 60_000;
const ABSTRACT_MAX = 1500, TAGS_MAX = 20, AUTHORS_MAX = 8, COLLECTIONS_MAX = 10, PAPERS_MAX = 3;
/** Our files in a papers folder: `<attachment key>-<slug>.txt`. Nothing else there is ever touched. */
const PAPER_FILE = /^([A-Z0-9]{8})(-[a-z0-9-]+)?\.txt$/;

// ───────────── pure (test/zotero/paper.test.ts) ─────────────

export interface PaperFacts {
  itemKey: string; creators: string[]; year: string; venue: string[]; doi: string;
  abstract: string; tags: string[]; collections: string[]; pages?: number;
}

const list = (xs: string[], max: number) => (xs.length > max ? `${xs.slice(0, max).join(", ")} and ${xs.length - max} more` : xs.join(", "));
const squash = (s: string) => s.replace(/\s+/g, " ").trim();

/** What the agent would otherwise fetch with `zotero-cli get metadata`, in a few lines. */
export function aboutText(f: PaperFacts): string {
  const head = [list(f.creators, AUTHORS_MAX), f.year, ...f.venue, f.doi && `DOI ${f.doi}`, f.pages ? `${f.pages} pages` : ""].filter(Boolean);
  const abs = squash(f.abstract);
  return [
    `About item ${f.itemKey}: ${head.join(" · ") || "no metadata"}`,
    ...(f.tags.length ? [`Tags: ${list(f.tags, TAGS_MAX)}`] : []),
    ...(f.collections.length ? [`In collections: ${list(f.collections, COLLECTIONS_MAX)}`] : []),
    ...(abs ? [`Abstract: ${abs.length > ABSTRACT_MAX ? `${abs.slice(0, ABSTRACT_MAX - 1)}…` : abs}`] : []),
  ].join("\n");
}

export const fullTextLine = (path: string, pages: number) =>
  `Full text is at ${path} (${pages} page${pages === 1 ? "" : "s"}, page markers like [p.7]); read it with grep/sed or zotero-cli read for specific pages.`;

/** `3QW3D95Y-callaway-2021.txt`: the attachment key, then the first author and year (or the title's first words). */
export function fileName(attKey: string, creator: string, year: string, title: string): string {
  const ascii = (s: string) => s.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const slug = (creator ? ascii(`${creator} ${year}`) : ascii(title.split(/\s+/).slice(0, 3).join(" "))).slice(0, 40).replace(/-$/, "");
  return `${attKey}${slug ? `-${slug}` : ""}.txt`;
}

/** The file's first line: what it is, how pages are marked, and the PDF it was made from (size and mtime: the cache key). */
export const headLine = (o: { title: string; itemKey: string; attKey: string; pages: number; source: string }) =>
  `Full text of "${squash(o.title)}" (item ${o.itemKey}, PDF ${o.attKey}), ${o.pages} pages; a line [p.N] starts page N, as zotero-cli read and citations count pages. Source: ${o.source}.`;

export function parseHead(text: string): { pages: number; source: string } | null {
  const m = /^Full text of .*?, (\d+) pages; a line \[p\.N\] starts page N, .*Source: (.+)\.$/.exec(text.split("\n", 1)[0] ?? "");
  return m ? { pages: Number(m[1]), source: m[2]! } : null;
}

/** Zotero's text of a PDF (pages end with a form feed) as page-marked text: `[p.N]` above page N, 1-based like the reader's page positions. */
export function markPages(text: string): { body: string; pages: number } {
  const pages = text.split("\f");
  if (pages.length > 1 && !pages.at(-1)!.trim()) pages.pop(); // a form feed after the last page, not another page
  if (!pages.some((p) => p.trim())) return { body: "", pages: 0 };
  return { body: pages.map((p, i) => `[p.${i + 1}]\n\n${p.trim() || "(no text on this page: a scan or a picture)"}`).join("\n\n"), pages: pages.length };
}

/** What a papers folder sheds: older copies of one attachment, files unused for `maxAge`, then the oldest beyond `maxBytes`. */
export function evictions(files: { path: string; size: number; mtime: number }[], now: number, maxBytes = MAX_BYTES, maxAge = MAX_AGE_MS): string[] {
  const out: string[] = [];
  const key = (path: string) => PAPER_FILE.exec(path.split("/").pop() ?? "")?.[1] ?? path;
  const newest = new Map<string, number>();
  for (const f of files) newest.set(key(f.path), Math.max(newest.get(key(f.path)) ?? 0, f.mtime));
  const kept = files.filter((f) => {
    const gone = now - f.mtime > maxAge || f.mtime < newest.get(key(f.path))!;
    if (gone) out.push(f.path);
    return !gone;
  }).sort((a, b) => a.mtime - b.mtime);
  let bytes = kept.reduce((n, f) => n + f.size, 0);
  while (bytes > maxBytes && kept.length) { const f = kept.shift()!; bytes -= f.size; out.push(f.path); }
  return out;
}

/** `p`, or null once `ms` have passed: the first message never waits longer for the file. */
export function within<T>(p: Promise<T | null>, ms: number): Promise<T | null> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  return Promise.race([p.catch(() => null), new Promise<null>((r) => { timer = setTimeout(() => r(null), ms); })]).finally(() => clearTimeout(timer));
}

// ───────────── Zotero ─────────────

interface Target { att: any; item: any }
type Done = { path: string; pages: number };

const field = (item: any, f: string): string => { try { return String(item.getField(f) ?? "").trim(); } catch { return ""; } };

function facts(item: any, pages?: number): PaperFacts {
  const creators = (item.getCreators?.() ?? []).map((c: any) => squash(c.fieldMode === 1 ? c.lastName : `${c.firstName ?? ""} ${c.lastName ?? ""}`)).filter(Boolean);
  const venue = ["publicationTitle", "proceedingsTitle", "conferenceName", "bookTitle", "websiteTitle", "repository", "publisher", "university", "institution"].map((f) => field(item, f)).find(Boolean);
  // The user's own tags before automatic ones, each by name.
  const tags = [...(item.getTags?.() ?? [])].sort((a: any, b: any) => (a.type ?? 0) - (b.type ?? 0) || String(a.tag).localeCompare(b.tag)).map((t: any) => t.tag);
  return {
    itemKey: item.key, creators, year: (item.getField("date", true, true) || "").slice(0, 4).replace(/^0000$/, ""),
    venue: [venue, field(item, "archiveID")].filter((v): v is string => !!v), doi: field(item, "DOI"), abstract: field(item, "abstractNote"),
    tags, collections: (item.getCollections?.() ?? []).map((id: number) => Zotero.Collections.get(id)?.name).filter(Boolean), ...(pages ? { pages } : {}),
  };
}

const firstPdf = (item: any) => (item.isRegularItem?.() ? item.getAttachments() : []).map((id: number) => Zotero.Items.get(id)).find((a: any) => a?.attachmentContentType === "application/pdf") ?? null;
const pagesOf = (att: any): number | undefined => viewStats(Zotero.Reader._readers?.find((r: any) => r.itemID === att.id))?.pagesCount || undefined;

/** The regular items the chips are about (the open PDF's parent, attached or selected items), at most three. */
function items(chips: ContextChip[]): any[] {
  const out = new Map<string, any>();
  for (const c of chips) {
    if (c.kind !== "reader" && c.kind !== "item") continue;
    const it = c.ref.itemKey ? Zotero.Items.getByLibraryAndKey(c.ref.libraryID, c.ref.itemKey) : null;
    if (it?.isRegularItem?.()) out.set(`${it.libraryID}/${it.key}`, it);
  }
  return [...out.values()].slice(0, PAPERS_MAX);
}

/** The PDF whose text to prepare: the one open in the reader, else the PDF of the one item in focus. */
function target(chips: ContextChip[]): Target | null {
  const reader = chips.find((c) => c.kind === "reader" && c.ref.attachmentKey);
  if (reader) {
    const att = Zotero.Items.getByLibraryAndKey(reader.ref.libraryID, reader.ref.attachmentKey!);
    return att?.attachmentContentType === "application/pdf" ? { att, item: att.parentItem ?? att } : null;
  }
  const its = chips.filter((c) => c.kind === "item");
  if (its.length !== 1) return null;
  const it = Zotero.Items.getByLibraryAndKey(its[0]!.ref.libraryID, its[0]!.ref.itemKey ?? "");
  const att = it?.isRegularItem?.() ? firstPdf(it) : it?.attachmentContentType === "application/pdf" ? it : null;
  return att ? { att, item: att.parentItem ?? att } : null;
}

export interface Papers {
  /** The user is about to write: start preparing the focused PDF's text. */
  prefetch(chips: ContextChip[]): void;
  /** The chips the agent gets beside `chips`: metadata now, and the full text's file if it is ready within WAIT_MS of `ready`. */
  context(chips: ContextChip[], ready: Promise<unknown>): Promise<ContextChip[]>;
}

export function createPapers(o: { dataDir: string; defaultFolder(): string; chatFolder(): string }): Papers {
  /** `<attachment key>|<source>` -> the file being made or made: a PDF that changes gets a new key, hence a new file. */
  const jobs = new Map<string, Promise<Done | null>>();

  /** The panel's own folder keeps a visible papers/ (documented); someone's project folder is left alone: the profile's cache. */
  const dirFor = () => PathUtils.join(o.chatFolder() === o.defaultFolder() ? o.defaultFolder() : o.dataDir, "papers");

  async function job(t: Target): Promise<Done | null> {
    const pdf: string | false = await t.att.getFilePathAsync();
    if (!pdf) return null;
    const st = await IOUtils.stat(pdf);
    const source = `${st.size} bytes, modified ${Math.round(st.lastModified ?? 0)}`;
    const key = `${t.att.key}|${source}`;
    let p = jobs.get(key);
    if (!p) {
      p = make(t, source).catch((e) => { Zotero.debug(`zotero-chat: full text of ${t.att.key} failed: ${e}`); return null; });
      jobs.set(key, p);
      void p.then((d) => { if (!d) setTimeout(() => jobs.delete(key), RETRY_MS); });
    }
    return p;
  }

  async function make(t: Target, source: string): Promise<Done | null> {
    const dir = dirFor();
    const year = (t.item.getField?.("date", true, true) || "").slice(0, 4).replace(/^0000$/, "");
    const first = t.item.getCreators?.()[0];
    const path = PathUtils.join(dir, fileName(t.att.key, first?.lastName ?? "", year, t.item.getDisplayTitle?.() ?? ""));
    // Made before, from this very PDF: use it (and mark it used, for the LRU).
    try {
      const head = parseHead(new TextDecoder().decode(await IOUtils.read(path, { maxBytes: 1024 })));
      if (head?.source === source) { await IOUtils.setModificationTime(path).catch(() => {}); return { path, pages: head.pages }; }
    } catch { /* not made yet */ }
    // Zotero's own extractor (its PDF worker, off the main thread; 0.3 s for 39 pages), measured against `zotero-cli read`
    // (1.5-1.8 s: Python start-up, and a copy of zotero.sqlite while Zotero holds it): DESIGN.md has the numbers.
    const r = await Zotero.PDFWorker.getFullText(t.att.id, null, true);
    const { body, pages } = markPages(String(r?.text ?? ""));
    if (!pages) return null;
    await IOUtils.makeDirectory(dir, { createAncestors: true, ignoreExisting: true });
    const head = headLine({ title: t.item.getDisplayTitle?.() ?? "", itemKey: t.item.key, attKey: t.att.key, pages, source });
    await IOUtils.writeUTF8(path, `${head}\n\n${body}\n`, { tmpPath: `${path}.tmp` });
    void prune(dir);
    return { path, pages };
  }

  async function prune(dir: string): Promise<void> {
    try {
      const files = [];
      for (const p of await IOUtils.getChildren(dir)) {
        if (!PAPER_FILE.test(PathUtils.filename(p))) continue;
        const s = await IOUtils.stat(p);
        files.push({ path: p, size: s.size, mtime: s.lastModified ?? 0 });
      }
      for (const p of evictions(files, Date.now())) await IOUtils.remove(p).catch(() => {});
    } catch (e) { Zotero.debug(`zotero-chat: papers cleanup: ${e}`); }
  }

  return {
    prefetch(chips) {
      try { const t = target(chips); if (t) void job(t); } catch { /* the send tries again */ }
    },
    async context(chips, ready) {
      let t: Target | null = null, meta: ContextChip[] = [];
      try {
        t = target(chips);
        meta = items(chips).map((it): ContextChip => ({
          id: `paper:${it.key}`, kind: "paper", label: `${itemLabel(it)} metadata`, auto: true, pinned: false,
          ref: { libraryID: it.libraryID, itemKey: it.key }, text: aboutText(facts(it, t && t.item === it ? pagesOf(t.att) : undefined)),
        }));
      } catch (e) { Zotero.debug(`zotero-chat: paper metadata: ${e}`); }
      const pending = t ? job(t) : null; // started before the session is up: it overlaps with that
      await ready.catch(() => {});
      const done = pending ? await within(pending, WAIT_MS) : null;
      if (!t || !done) return meta;
      return [...meta, {
        id: `fulltext:${t.att.key}`, kind: "paper", label: `${itemLabel(t.item)} full text`, auto: true, pinned: false,
        ref: { libraryID: t.att.libraryID, itemKey: t.item.key, attachmentKey: t.att.key }, text: fullTextLine(done.path, done.pages),
      }];
    },
  };
}
