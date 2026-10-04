// A PanelHost with a scripted agent, so every state of the panel can be screenshotted and tested without
// Zotero. The preview page and the Playwright scripts drive it through `host.sim`.
import { CATALOGS, defaultSettings } from "./fake-catalog.ts";
import type {
  AgentRuntime, AgentSession, BackendId, BackendStatus, ChatEvent, ContextChip, DoctorCheck, ItemHit, PanelHost,
  PanelSettings, PermissionOption, PromptInput, SavedSession, StartOpts, ZoteroRef,
} from "../types.ts";

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

// A tiny valid PNG (a light page with a grey figure), as base64, for the Selected Area card.
const AREA_PNG = "iVBORw0KGgoAAAANSUhEUgAAAUAAAAB4CAIAAAAMrLyJAAAI/klEQVR4nO3dbUhT/R/H8d9uFEkTelI9MbiwqJghSmBSRi4rH4hhRHdmRRClbSKB6UaGNVGTgqLQRwUXOgu7+UskmQpSGjofdEOoRDdGCC4LIWxleOb58+9wnWv/M503be18f+fzogfHee42z9vfb9OZ7m3vJgYANOnDfQIAsHAIGIAwo7wUn9IV1jMBgLl650qTFjACAxCGgAEIQ8AAhCFgAMIQMABhCFjVenp6BEFQ3CgIQm9vb5jOCPgNOCMjw2q1FhYW5ufnP3z4cO4bZmZmznrLgnc1q4aGhvmuPDQ01NzcHNIDMcbGxsba29uNRmN5ebn1l/z8/KysLKPR2NbWNjY2Nt8TAJ5/DhyEfRmNV69eZYxNTEyUlJRERUWlp6cz1WtoaDh48OC8Vv7rl5AeiDHW0tKSmprKGCsvL5duefDgwadPnxhjGzZsaGlpycvLm+85AGdCMoWOiooqKCi4ffv22NhYcXGxxWIpLi6WRgzfEdJ3uba29uTJkxaLZWRkRL7x6NGjw8PDjDGPx7N//35RFOUNKysr79y5479//135H/Hu3btHf+nr67t+/fqPHz9OnTolrTA+Pu5wOIqKiiwWy+DgYICV5d1mZmZWVVXt3bu3ubnZ4XDs2bOnqalJGqULCgoOHTokfShvO+1jIt0dxcPocrkSEhLkD0VRvHfv3q5duxhjJpPJ5XIF9YsGNL3t3ST9E3/bjh075OXJycns7Oxz5861traKotja2nr+/HnFOvKy2Wzu6OgQRfHRo0c2m03+lNPpbGxsFEWxo6OjtrZW3tBsNrtcLlEU/fc/0658j5iVleXxeD58+OBwOBQrVFdX9/f3i6LodruPHDkSYGV5IT09vb+/3+12p6WlDQwMjIyM7Ny5UxTFixcvvnjx4uvXr9KH8ibTnrN0dxSys7O9Xq/8YVdX14ULF6RlQRCys7N/+ysGVMnZhupFLK/XazAYnj9/bjabGWNms/nZs2eKdeQRVafTbd68mTEmxSCvkJGR8fTpU8ZYd3d3RkaGfLter1+/fj1jzH//M+3K94ipqakVFRWjo6NnzpxRrOByuerq6qxWa0VFxcTExNTUVICV5ZNZs2bNsmXLjEbj6tWrly9fPjExwRjLz8//+PGj0+n0eDy+6/ufs3x3FARB0Ov//QLdunVr37590rLBYPB/cQs0KJjPgX0NDg7Gx8e/efNmpmi/ffs2OTkpLet0OoPBIC1HRETIKy9dulSn033+/Nntdq9atUq+3WAwSFe2vDeZ/678j2i321++fNnU1NTe3m63230393q9ly5dioyMnJqaevXqlV6vD7CyxGg0SicTGRnp21tZWdmWLVt2796teLnL/5zlu6MQGxv7/fv3RYsWMcYGBgYWL168YsUK6VMejyc2NtZ/E9CakIzA4+PjdXV1Bw4cSE5O7uzsZIx1dnYmJSUxxqKjo4eGhhhjbW1tOp1OWt/r9fb09EirJScn++5q69at165dS0lJmfZA/vv335XiiB6Px2KxmEymsrIy6YcxoihOTU1JO1y3bt2TJ0+kobi+vj7wyoG9fv3abDb//PlT/j4lbet/zjMxmUzv37+XlhsbG+XhV3qCbTKZ5nIawLdgjsCCIFitVp1OJwhCbm5uUlJSXFxcdXX1/fv3o6KibDYbY6yoqOjs2bNLlixZu3atPNhGRkY+fvz45s2bMTExpaWljLG4uLj6+vq8vLz09PQrV64cO3Zs2iMWFBQo9u+/K8URo6OjN27cePz4cVEUDx8+zBhLTEwsLS2tqalhjBUWFtbU1DQ3NxsMhpKSksArB5aTk3PixImVK1fGxMRMTk5GRERI254+fVpxzjPZtm1bX19fQkLC8PDwly9fEhMT5U+5XK7t27fP/0sEvNHJf5FDnW8nHB0draysvHz5MtMeURQdDofdbjca/+/7rCAIVVVVZWVl4Ts1UMvbCVUdcHd3940bN2w2m+8TYAB490/AoXoRKyg2/RLuswBQL/wuNABhCBiAMAQMQBgCBtVJy/k73KdAhqpfxALNpuvbcNd//vcTeJgWAga1D7mIOQAEDOpKV5GoYgXErKDqX+QAjinKnEuNAQbqLo3FTOM3sYBLC0h31p1oLWYEDKqbLQdxz9zHjICBh3Q1GzMCBjKz5WAdnaeYETBwMuRqM2YEDFpJl8uYETBwNVvWWswIGLSeLumYETBoYrbMa8wIGBaO73RJxKzRgKWvBN9XW0hxM1umHrMWA9bauBFEeOjUFrPmAg78hjVcjjNBuuqMWUMBz3QJTvuI4+qUYbas5pi1EvBcBhCUrIB01R+zJgKe74Wo8ZIxWyYUM/8B/84worWSkS65aQ7nAQdrEsh9yZgtq0Fazt8IOLSDCWclY8iljs+A/8B1Sb1kpMsHDgP+w7NBciVjtswT3gIO49Wp/pKRLn+4ClglF6jaSsZsmWOcBKzOazTsJavzYYEg4iFg9V+mf75klUxGINTIB0zrSg11yer/XgbBRTtgWvWGtGSkq02EA6Zbb3BL5uNxAA0FzOVos4CSkS68Ixcwl/XOq2TuHwHgNmBNjTmB/3iIFh4BmHvABP6Db03V63sHMeTCrNQesNbqnalkTd1x4CFgjD8yzd5xmJWeqRLqBaA6Amt52gxAewRGvQBUA0a9ACSn0HjSC0B1BEa9AFRHYEybAaiOwKgXgGrAqBeA5BQaT3oBqI7AqBeA6giMaTMA1REY9QJQDRj1ApCcQuNJLwDVERj1AlAdgTFtBqA6AqNeAKoBo14AklNoPOkFoDoCo14AqiMwps0AVEdg1AtANWDUC0ByCo0nvQBUR2DUC0B1BMa0GYDwCOxbLOoFoDeFlrpFvQBUX4VGvQBa/8PuALAwCBiAMAQMQBgCBiAMAQMQhoABCEPAAFp9M0Nubm7wzgRAu5xOZ3jejbTgA/+O3NxcHBfH5em4C94WU2gAwhAwAGEIGIAwBAxAGAIGIEz3tneTtBSf0hXukwGAOXnnSpMWMAIDEIaAAQhDwACEIWAAwhAwAGEIGIAwBAxAGAIGIAwBAxBm9P/dDgCgAiMwAGEIGIDR9V97FyF6K2mQxgAAAABJRU5ErkJggg==";

export type DoctorMode = "ok" | "zotero-api" | "write-access" | "cli" | "node" | "backend" | "many";

interface FakeOptions {
  /** Delay between streamed chunks, ms. 0 = as fast as the event loop allows. */
  speed?: number;
  theme?: "light" | "dark";
  doctor?: DoctorMode;
  context?: "item" | "selection" | "area" | "none";
  /** Adds a 400-turn saved chat to History, for scrolling and performance checks. */
  stress?: boolean;
  /** Start with no saved chats (the History empty state). */
  noHistory?: boolean;
  catalogDelay?: number;
}

interface Sim {
  opened: (string | ZoteroRef)[];
  prompts: PromptInput[];
  doctorMode: DoctorMode;
  searchFails: boolean;
  startFails: string | null;
  /** Backends whose catalog() fails (besides the ones that are not installed), and how long it takes. */
  catalogFails: BackendId[];
  /** What chooseFolder() answers next (null = the user cancelled), and every folder prepareSession() was asked for. */
  pickFolder: string | null;
  preparedCwd: (string | undefined)[];
  catalogDelay: number;
  /** What the Data section did. */
  data: { cleared: number; revealed: number; resets: number };
  /** Say exactly this as the next answer (no tools): for the XSS corpus and layout tests. */
  nextAnswer: string | null;
  speed: number;
  setContext(kind: "item" | "selection" | "area" | "none"): void;
  setTheme(t: "light" | "dark"): void;
  statuses: BackendStatus[];
  writes: { session: string; ev: ChatEvent }[];
  keys: Partial<Record<BackendId, string>>;
  closed: number;
}

const DEFAULT_FOLDER = "/Users/you/Documents/Zotero Chat";
const CITE = (k: string, p: number) => `zotero://open-pdf/library/items/${k}?page=${p}`;

const SAMPLE_ANSWER = `Based on **Bertrand and Mullainathan (2004)** and the follow-ups in your library, the callback gap largely persists.

- **Field experiments.** Resumes with white-sounding names received about 50% more callbacks [Bertrand and Mullainathan 2004, p.8](${CITE("BM2004AB", 8)}).
- **Audit studies.** The same pattern shows up in person: a clear racial hierarchy among equally qualified applicants [Pager et al. 2009, p.9](${CITE("PAGER009", 9)}), including for applicants with no criminal record [Pager et al. 2009, p.12](${CITE("PAGER009", 12)}).
- **Meta-analysis.** No decline in discrimination against Black applicants in 25 years [Quillian et al. 2017, p.4](${CITE("QUIL2017", 4)}).

The callback ratio is $R = c_w / c_b \\approx 1.50$, so in log points the gap is

$$\\Delta = \\ln\\left(\\frac{c_w}{c_b}\\right) = \\ln(1.5) \\approx 0.405.$$

| Study | Ratio | Applications |
|---|---:|---:|
| Bertrand and Mullainathan 2004 | 1.50 | 4,870 |
| Pager et al. 2009 | 1.44 | 1,500 |
| Quillian et al. 2017 (pooled) | 1.36 | 55,842 |

To pull the numbers yourself, run \`zotero-cli search "callback" --json\` or:

\`\`\`bash
zotero-cli search "audit study" --limit 5 | zotero-cli get --fields title,date
\`\`\`

See [the OSF page](https://osf.io/example) for the replication data.`;

const LONG_PARAS = Array.from({ length: 9 }, (_, i) =>
  `### Section ${i + 1}\n\nThe literature on field experiments of hiring is large, and the findings differ in size more than in sign [Quillian et al. 2017, p.${4 + i}](${CITE("QUIL2017", 4 + i)}). ` +
  `Resume audits vary the name, the school, the gap in employment history and the neighbourhood, and each variation answers a slightly different question about what employers infer from a document. ` +
  `A careful reading separates the *level* of discrimination from its *trend*, which is what the meta-analysis does.\n\n- first point about identification\n- second point about external validity\n- third point about measurement`).join("\n\n");

const LIBRARY: ItemHit[] = [
  { kind: "item", title: "Are Emily and Greg More Employable than Lakisha and Jamal?", subtitle: "Bertrand and Mullainathan 2004", ref: { libraryID: 1, itemKey: "BM2004AB" } },
  { kind: "item", title: "The Mark of a Criminal Record", subtitle: "Pager 2003", ref: { libraryID: 1, itemKey: "PAGER003" } },
  { kind: "item", title: "Discrimination in a Low-Wage Labor Market: A Field Experiment", subtitle: "Pager, Western and Bonikowski 2009", ref: { libraryID: 1, itemKey: "PAGER009" } },
  { kind: "item", title: "Meta-analysis of field experiments shows no change in hiring discrimination against Black Americans", subtitle: "Quillian et al. 2017", ref: { libraryID: 1, itemKey: "QUIL2017" } },
  { kind: "item", title: "Doubly robust estimation of causal effects", subtitle: "Funk et al. 2011", ref: { libraryID: 1, itemKey: "FUNK2011" } },
  { kind: "item", title: "The Inheritance of Race: Generational Transfers", subtitle: "Lundberg 2021", ref: { libraryID: 1, itemKey: "LUND2021" } },
  { kind: "collection", title: "05_Projects / Hiring audits", subtitle: "Collection · 23 items", ref: { libraryID: 1, collectionKey: "COLL0001" } },
  { kind: "collection", title: "00_READING LIST", subtitle: "Collection · 41 items", ref: { libraryID: 1, collectionKey: "COLL0002" } },
  { kind: "annotation", title: "“the stigma of being a minority in the labor market is equivalent to…”", subtitle: "Pager et al. 2009 · p.9", ref: { libraryID: 1, itemKey: "PAGER009", attachmentKey: "ATT00009", annotationKey: "ANN00001", pageIndex: 8, pageLabel: "9" } },
];

function chipFromHit(hit: ItemHit): ContextChip {
  const kind = hit.kind === "annotation" ? "annotation" : hit.kind === "collection" ? "collection" : "item";
  return { id: `${kind}:${hit.ref.itemKey ?? hit.ref.collectionKey}:${hit.ref.annotationKey ?? ""}`, kind, label: hit.kind === "collection" ? hit.title : hit.subtitle.split(" · ")[0] || hit.title, auto: false, pinned: false, ref: hit.ref, text: hit.title };
}

const ITEM_CHIP: ContextChip = { id: "reader:BM2004AB", kind: "reader", label: "Bertrand and Mullainathan 2004", auto: true, pinned: false, ref: { libraryID: 1, itemKey: "BM2004AB", attachmentKey: "ATT00001", pageIndex: 6, pageLabel: "997" } };
const SELECTION_CHIP: ContextChip = { id: "selection:1", kind: "selection", label: "Text Selection", auto: true, pinned: false, ref: { libraryID: 1, itemKey: "BM2004AB", pageIndex: 6 }, text: "The table reports, for the entire sample and different subsamples of sent resumes, the callback rates for applicants with a White-sounding name." };
const AREA_CHIP: ContextChip = { id: "area:1", kind: "area", label: "Selected Area · p.19", auto: true, pinned: false, ref: { libraryID: 1, itemKey: "LUND2021", pageIndex: 18, pageLabel: "19", annotationKey: "ANNAREA1" }, image: { mime: "image/png", data: AREA_PNG } };

// ───────────────────────────── the scripted agent ─────────────────────────────

class FakeSession implements AgentSession {
  readonly backend: BackendId;
  readonly account: string | undefined;
  readonly supportsImages = true;
  private listeners = new Set<(ev: ChatEvent) => void>();
  private model: string;
  private mode: string;
  private effort: string;
  private cancelled = false;
  private waiting = new Map<string, (opt: string | null) => void>();
  private turnNo = 0;
  readonly sessionId: string;

  private sim: Sim;

  constructor(sim: Sim, opts: StartOpts, account?: string) {
    this.sim = sim;
    this.backend = opts.backend;
    this.account = account;
    const cat = CATALOGS[opts.backend];
    this.model = opts.model || cat.model || "";
    this.mode = opts.mode || cat.mode || "";
    this.effort = opts.effort || cat.effort || "";
    this.sessionId = opts.resumeSessionId ?? `agent-${Math.random().toString(36).slice(2, 8)}`;
  }
  models() { return CATALOGS[this.backend].models; }
  currentModel() { return this.model; }
  modes() { return CATALOGS[this.backend].modes; }
  currentMode() { return this.mode || undefined; }
  efforts() { return CATALOGS[this.backend].efforts; }
  currentEffort() { return this.effort || undefined; }
  async setModel(id: string) { await sleep(20); this.model = id; }
  async setMode(id: string) { await sleep(20); this.mode = id; }
  async setEffort(id: string) { await sleep(20); this.effort = id; }
  on(l: (ev: ChatEvent) => void) { this.listeners.add(l); return () => { this.listeners.delete(l); }; }
  private emit(ev: ChatEvent) { for (const l of [...this.listeners]) l(ev); }
  async cancel() { this.cancelled = true; for (const [, r] of this.waiting) r(null); }
  respondPermission(id: string, optionId: string | null) { this.waiting.get(id)?.(optionId); this.waiting.delete(id); }
  async close() { this.sim.closed++; this.listeners.clear(); }

  private async stream(turn: string, kind: "text" | "thought", text: string): Promise<void> {
    const parts = text.match(/[\s\S]{1,9}/g) ?? [];
    for (const p of parts) {
      if (this.cancelled) return;
      this.emit({ t: kind, turn, delta: p });
      if (this.sim.speed > 0) await sleep(this.sim.speed);
      else await Promise.resolve();
    }
  }

  async prompt(input: PromptInput): Promise<void> {
    this.sim.prompts.push(input);
    this.cancelled = false;
    const turn = `turn-${++this.turnNo}-${this.sessionId}`;
    const q = input.text.toLowerCase();
    this.emit({ t: "turn_start", turn });
    if (this.sim.startFails === "prompt") { this.sim.startFails = null; throw new Error("the bridge exited (code 1)"); }
    await sleep(this.sim.speed ? 120 : 0);

    if (q.includes("fail") || q.includes("error")) {
      await this.stream(turn, "text", "Let me check your library. ");
      this.emit({ t: "tool", turn, id: "e1", title: "Searched library · “callback”", kind: "search", status: "failed", input: { query: "callback" }, output: "connect ECONNREFUSED 127.0.0.1:23119" });
      this.emit({ t: "notice", level: "error", message: "Zotero's local API isn't answering", hint: "Restart Zotero, then try again." });
      this.emit({ t: "turn_end", turn, stop: "error" });
      return;
    }

    if (this.sim.nextAnswer !== null) {
      const text = this.sim.nextAnswer;
      this.sim.nextAnswer = null;
      await this.stream(turn, "text", text);
      this.emit({ t: "turn_end", turn, stop: this.cancelled ? "cancelled" : "end_turn" });
      return;
    }
    const long = q.includes("long");
    const slow = q.includes("slow");
    if (slow) this.sim.speed = Math.max(this.sim.speed, 40);

    await this.stream(turn, "thought", "The user asks about hiring discrimination. I should search the library for audit studies, then read the callback table in the open paper before answering.");
    await this.stream(turn, "text", "I'll look for related research in your library first. ");
    const search = { t: "tool", turn, id: "t1", name: "Bash", title: "Searched library · “hiring discrimination audit”", kind: "search" } as const;
    const read = { t: "tool", turn, id: "t2", name: "Read", title: "Read Bertrand and Mullainathan 2004 · pp. 6-9", kind: "read" } as const;
    this.emit({ ...search, status: "running", input: { query: "hiring discrimination audit", limit: 10 } });
    await sleep(this.sim.speed ? 350 : 0);
    if (this.cancelled) return this.end(turn, "cancelled");
    this.emit({ ...search, status: "done", output: "10 items\n  BM2004AB  Bertrand and Mullainathan 2004\n  PAGER009  Pager et al. 2009\n  QUIL2017  Quillian et al. 2017" });
    this.emit({ ...read, status: "running", input: { item: "BM2004AB", pages: "6-9" } });
    await sleep(this.sim.speed ? 300 : 0);
    this.emit({ ...read, status: "done", output: "(4 pages, 5,120 characters)" });

    if (q.includes("plan")) {
      this.emit({ t: "plan", turn, entries: [{ content: "Find audit studies in the library", status: "completed" }, { content: "Read the callback table", status: "in_progress" }, { content: "Compare effect sizes", status: "pending" }] });
      await sleep(this.sim.speed ? 250 : 0);
      this.emit({ t: "plan", turn, entries: [{ content: "Find audit studies in the library", status: "completed" }, { content: "Read the callback table", status: "completed" }, { content: "Compare effect sizes", status: "in_progress" }] });
    }

    if (q.includes("permission") || q.includes("edit") || q.includes("note")) {
      const options: PermissionOption[] = [
        { id: "allow", name: "Allow once", kind: "allow_once" },
        { id: "always", name: "Always allow", kind: "allow_always" },
        { id: "deny", name: "Deny", kind: "reject_once" },
      ];
      const id = "perm-1";
      this.emit({ t: "permission", turn, id, title: "Create a note on Bertrand and Mullainathan 2004", kind: "edit", input: { command: "zotero-cli note add BM2004AB --text \"Callback ratio 1.50\"" }, options });
      const choice = await new Promise<string | null>((res) => { this.waiting.set(id, res); });
      if (this.cancelled) return this.end(turn, "cancelled");
      if (choice === "deny" || choice === null) {
        await this.stream(turn, "text", "Okay, I won't add the note. Here is the summary instead, so you can paste it yourself.");
      } else {
        this.emit({ t: "tool", turn, id: "t3", title: "Added a note · zotero-cli note add", kind: "edit", status: "done", input: { item: "BM2004AB" }, output: "created note NOTE0001" });
        await this.stream(turn, "text", "Done: I added the note. ");
      }
    }

    const body = long ? `${SAMPLE_ANSWER}\n\n${LONG_PARAS}` : q.includes("math") ? "The estimator is\n\n$$\\hat{\\tau} = \\frac{1}{n}\\sum_{i=1}^{n}\\left(\\frac{T_i Y_i}{e(X_i)} - \\frac{(1-T_i)Y_i}{1-e(X_i)}\\right)$$\n\nwith propensity $e(X_i) = P(T_i = 1 \\mid X_i)$. An unclosed $$ stays text while it streams, and \\$5 is just five dollars." : SAMPLE_ANSWER;
    await this.stream(turn, "text", (this.cancelled ? "" : body));
    if (this.cancelled) return this.end(turn, "cancelled");
    this.emit({ t: "turn_end", turn, stop: "end_turn", usage: { inputTokens: 12840, outputTokens: 912, costUsd: 0.031 } });
  }

  private end(turn: string, stop: "cancelled" | "error") { this.emit({ t: "turn_end", turn, stop }); }
}

// ───────────────────────────── the host ─────────────────────────────

export class FakeHost implements PanelHost {
  readonly sim: Sim;
  readonly runtime: AgentRuntime;
  private settings: PanelSettings = defaultSettings();
  private ctx: ContextChip[] = [];
  private ctxCbs = new Set<() => void>();
  private themeCbs = new Set<() => void>();
  private _theme: "light" | "dark";
  private store = new Map<string, { meta: SavedSession; events: ChatEvent[] }>();
  private fixed = new Set<string>();

  constructor(o: FakeOptions = {}) {
    this._theme = o.theme ?? "light";
    const sim: Sim = {
      opened: [], prompts: [], doctorMode: o.doctor ?? "ok", searchFails: false, startFails: null, catalogFails: [], pickFolder: "/Users/you/Documents/Projects/hiring-audits/paper-notes", preparedCwd: [], catalogDelay: o.catalogDelay ?? 120, data: { cleared: 0, revealed: 0, resets: 0 }, nextAnswer: null, speed: o.speed ?? 18, writes: [], keys: {}, closed: 0,
      statuses: [
        { id: "claude-code", label: "Claude Code", available: true, account: "Claude Max" },
        { id: "codex", label: "Codex", available: false, reason: "codex is not installed" },
        { id: "pi", label: "pi", available: true },
      ],
      setContext: (k) => { this.ctx = k === "none" ? [] : k === "selection" ? [ITEM_CHIP, SELECTION_CHIP] : k === "area" ? [{ ...ITEM_CHIP, id: "reader:LUND2021", label: "Lundberg 2021 - The G…", ref: { libraryID: 1, itemKey: "LUND2021" } }, AREA_CHIP] : [ITEM_CHIP]; for (const cb of [...this.ctxCbs]) cb(); },
      setTheme: (t) => { this._theme = t; for (const cb of [...this.themeCbs]) cb(); },
    };
    this.sim = sim;
    this.ctx = [];
    sim.setContext(o.context ?? "item");
    if (!o.noHistory) this.seedHistory(!!o.stress);
    this.runtime = {
      detect: async () => { await sleep(10); return this.statuses(); },
      catalog: async (b) => {
        await sleep(sim.catalogDelay);
        const st = this.statuses().find((x) => x.id === b);
        if (st && !st.available) throw new Error(st.reason ?? `${st.label} is not available`);
        if (sim.catalogFails.includes(b)) throw new Error("the bridge did not answer within 20 s");
        return CATALOGS[b];
      },
      start: async (opts) => {
        await sleep(sim.speed ? 250 : 0);
        if (sim.startFails === "start") { sim.startFails = null; throw new Error("claude-agent-acp could not be started: spawn node ENOENT"); }
        return new FakeSession(sim, opts, this.statuses().find((s) => s.id === opts.backend)?.account);
      },
    };
  }

  private statuses(): BackendStatus[] {
    const m = this.sim.doctorMode;
    return this.sim.statuses.map((s) => (s.id === "claude-code" && (m === "backend" || m === "many") && !this.fixed.has("backend")
      ? { ...s, available: false, reason: "claude is not logged in", account: undefined } : s));
  }

  // --- context
  currentContext() { return this.ctx; }
  onContextChange(cb: () => void) { this.ctxCbs.add(cb); return () => { this.ctxCbs.delete(cb); }; }
  async search(query: string): Promise<ItemHit[]> {
    await sleep(this.sim.speed ? 150 : 0);
    if (this.sim.searchFails) throw new Error("Zotero's local API isn't answering");
    const q = query.trim().toLowerCase();
    return q ? LIBRARY.filter((h) => `${h.title} ${h.subtitle}`.toLowerCase().includes(q)) : LIBRARY.slice(0, 5);
  }
  async chipFor(hit: ItemHit) { return chipFromHit(hit); }
  /** The preview's drag format: a `text/plain` marker "zmc-item:KEY" per line (the real host reads Zotero's own formats). */
  async dropChips(data: DataTransfer): Promise<ContextChip[]> {
    const keys = [...data.getData("text/plain").matchAll(/^zmc-item:(\w+)$/gm)].map((m) => m[1]);
    return LIBRARY.filter((h) => h.kind !== "collection" && keys.includes(h.ref.itemKey ?? "")).map(chipFromHit);
  }
  async open(target: string | ZoteroRef) { this.sim.opened.push(target); }
  describeContext(chips: ContextChip[]) {
    const lines = chips.map((c) => `- ${c.kind}: ${c.label}${c.ref.itemKey ? ` (key ${c.ref.itemKey})` : ""}${c.text ? `\n  text: ${c.text}` : ""}`);
    return {
      text: chips.length ? `<zotero-context>\n${lines.join("\n")}\n</zotero-context>` : "",
      images: chips.flatMap((c) => (c.image ? [{ mime: c.image.mime, data: c.image.data }] : [])),
    };
  }

  // --- settings
  getSettings() { return this.settings; }
  async setSettings(patch: Partial<PanelSettings>) { this.settings = { ...this.settings, ...patch }; }
  async clearHistory() { this.store.clear(); this.sim.data.cleared++; }
  async revealWorkspace() { this.sim.data.revealed++; }
  async resetSettings() { this.settings = defaultSettings(); this.sim.data.resets++; }
  private folder() { return this.settings.chatFolder || DEFAULT_FOLDER; }
  about() { return { version: "0.1.0", workspace: this.folder() }; }
  async setApiKey(b: BackendId, key: string | null) { if (key) this.sim.keys[b] = key; else delete this.sim.keys[b]; }
  async hasApiKey(b: BackendId) { return !!this.sim.keys[b]; }

  // --- sessions
  private seedHistory(stress: boolean) {
    const now = Date.now();
    const mk = (id: string, title: string, ago: number, events: ChatEvent[], cwd = DEFAULT_FOLDER, backend: BackendId = "claude-code") =>
      this.store.set(id, { meta: { id, title, backend, cwd, agentSessionId: `agent-${id}`, updatedAt: now - ago }, events });
    const small = (q: string, a: string): ChatEvent[] => [
      { t: "user", id: `u-${q.length}`, text: q, chips: [] },
      { t: "turn_start", turn: `t-${q.length}` },
      { t: "text", turn: `t-${q.length}`, delta: a },
      { t: "turn_end", turn: `t-${q.length}`, stop: "end_turn" },
    ];
    mk("s1", "Compare callback ratios across audit studies", 12 * 60e3, small("Compare callback ratios across audit studies", "Across the three studies the ratio ranges from 1.36 to 1.50 [Quillian et al. 2017, p.4](" + CITE("QUIL2017", 4) + ")."));
    mk("s2", "Which papers cite Pager 2003?", 3 * 3600e3, small("Which papers cite Pager 2003?", "Two items in your library cite it."), "/Users/you/Documents/Projects/hiring-audits/paper-notes", "codex");
    mk("s3", "Summarize the Lundberg doubly robust section", 26 * 3600e3, small("Summarize the Lundberg doubly robust section", "The estimator reweights outcomes by the inverse propensity score."), DEFAULT_FOLDER, "pi");
    mk("s4", "Tag everything in 05_Projects with the course week", 4 * 864e5, small("Tag everything in 05_Projects with the course week", "Done: 23 items tagged."));
    mk("s5", "A very long chat title that should truncate in the history list because it keeps going and going", 40 * 864e5, small("x", "ok"));
    if (stress) {
      const ev: ChatEvent[] = [];
      for (let i = 0; i < 400; i++) {
        ev.push({ t: "user", id: `lu${i}`, text: `Question ${i}: what does section ${i % 12} say?`, chips: [] });
        ev.push({ t: "turn_start", turn: `lt${i}` });
        ev.push({ t: "tool", turn: `lt${i}`, id: `lx${i}`, title: `Read section ${i % 12}`, kind: "read", status: "done", output: "ok" });
        ev.push({ t: "text", turn: `lt${i}`, delta: `Section ${i % 12} argues that audit estimates are robust [Pager et al. 2009, p.${i % 30 + 1}](${CITE("PAGER009", i % 30 + 1)}). It also notes a **limitation** in external validity.\n\n- point one\n- point two` });
        ev.push({ t: "turn_end", turn: `lt${i}`, stop: "end_turn" });
      }
      mk("stress", "Long literature review (stress)", 60e3, ev);
    }
  }
  async sessions() { await sleep(this.sim.speed ? 120 : 0); return [...this.store.values()].map((s) => s.meta); }
  async loadEvents(id: string) { return this.store.get(id)?.events ?? []; }
  async appendEvent(session: SavedSession, ev: ChatEvent) {
    this.sim.writes.push({ session: session.id, ev });
    const cur = this.store.get(session.id) ?? { meta: session, events: [] };
    cur.meta = session;
    cur.events.push(ev);
    this.store.set(session.id, cur);
  }
  async deleteSession(id: string) { this.store.delete(id); }

  // --- doctor
  private async *runFix(id: string, lines: string[]): AsyncGenerator<string> {
    for (const l of lines) { await sleep(this.sim.speed ? 140 : 0); yield l + "\n"; }
    this.fixed.add(id); // the next doctor() finds it fixed
  }
  async doctor(): Promise<DoctorCheck[]> {
    await sleep(this.sim.speed ? 200 : 0);
    const m = this.sim.doctorMode;
    const bad = (id: DoctorCheck["id"]) => (m === id || m === "many") && !this.fixed.has(id);
    const checks: DoctorCheck[] = [
      { id: "zotero-api", ok: !bad("zotero-api"), label: "Zotero local API", detail: bad("zotero-api") ? "No answer on 127.0.0.1:23119" : "Answering on 127.0.0.1:23119" },
      { id: "write-access", ok: !bad("write-access"), label: "Write access", detail: bad("write-access") ? "Not authorized yet" : "Authorized" },
      { id: "cli", ok: !bad("cli"), label: "zotero-cli", detail: bad("cli") ? "Not found on PATH" : "0.13.1 at ~/.local/bin/zotero-cli" },
      { id: "node", ok: !bad("node"), label: "Node.js", detail: bad("node") ? "Not found on PATH" : "v24.1.0" },
      { id: "backend", ok: !bad("backend"), label: "Claude Code", detail: bad("backend") ? "claude is not logged in" : "Signed in as Claude Max" },
    ];
    const cli = checks.find((c) => c.id === "cli");
    if (cli && !cli.ok) {
      cli.fix = { label: "Install zotero-cli", run: () => this.runFix("cli", ["$ uv tool install zotero-mcp-server", "Resolved 14 packages in 412ms", "Installed 1 executable: zotero-cli"]) };
    }
    return checks;
  }
  async chooseFolder(_start?: string) { return this.sim.pickFolder; }
  /** What each backend's own CLI takes to continue a session; the folder is quoted for the shell. */
  resumeCommand(s: SavedSession) {
    const cmd = { "claude-code": `claude --resume ${s.agentSessionId}`, codex: `codex resume ${s.agentSessionId}`, pi: `pi --session ${s.agentSessionId}` }[s.backend];
    return s.agentSessionId ? `cd '${s.cwd.replace(/'/g, "'\\''")}' && ${cmd}` : null;
  }
  async prepareSession(cwd?: string) { this.sim.preparedCwd.push(cwd); return { cwd: cwd ?? this.folder(), brief: "You are in a Zotero side panel.", env: {} }; }

  // --- theme
  theme() { return this._theme; }
  onThemeChange(cb: () => void) { this.themeCbs.add(cb); return () => { this.themeCbs.delete(cb); }; }
}
