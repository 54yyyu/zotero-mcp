# Machine-Native Science — Master Roadmap (internal)

> Working doc for us. Companion to `essay-science-file-format.md` (public-facing narrative)
> and `~/Downloads/zotero-graph-handoff.md` (Library Graph strategy/engineering — now
> reframed as **Stage 1** of this bigger arc). June 2026.

---

## 0. The thesis in one paragraph

Scientific knowledge is born structured, destroyed at publication (PDF = printout for human
eyes), and expensively re-extracted by every reader — increasingly machines. Worse, the
literature records only successes, so humanity has no map of what's been *tried*. Fix = two
coupled changes: (1) a **machine-native format** (claims / evidence / methods / trajectories,
incl. negative branches), (2) an **incentive system** that rewards informative contributions
(incl. failures and evaluations), not success narratives. Every prior attempt (nanopubs 2009→,
Octopus, ORKG) died on adoption because it asked authors to do extra work for community
benefit. Our inversion: **start selfish** — value at n=1 — then ride network effects outward.
zotero-mcp (~700k downloads) is the distribution wedge nobody else has.

---

## 1. Prior art map (what exists, what's missing)

| Effort | What it proved | Why it didn't win / gap |
|---|---|---|
| **OpenEval** (eLife, the user's grounding paper) | Claims extraction at journal scale (1.96M claims / 16k papers); machine review ≈ human review (81% agreement, 0.7% direct disagreement); machines cover 92.8% of claims vs humans' 68.1%; cross-paper result network via embeddings | A pipeline + dataset, not a product; top-down (publisher corpus), no personal-utility loop |
| **Ara — Agent-Native Research Artifacts** (arXiv 2604.24658, Liu et al., UMich+Stanford+MIT+CMU+Meta, 2026) | Names our premises: "Storytelling Tax" (discarded failures) + "Engineering Tax" (omitted tacit details). 4 layers: /logic (claims), /src, **/trace (exploration DAG incl. dead ends)**, /evidence. Ara beats PDF+repo: 93.7% vs 72.4% knowledge extraction, 64.4% vs 57.4% reproduction. Has a PDF→Ara compiler + 3-level review system | Format-first, top-down; no distribution; no network/sharing layer; no credit economy. **Watch closely — candidate format to adopt/extend rather than reinvent** |
| **Nanopublications** (2009→, active 2026: provenance trust networks, contradiction detection) | Atomic claims + provenance as RDF is technically sound and durable | 17 years, near-zero author adoption. The cautionary tale: correct format, no selfish payoff |
| **ORKG** | Curated structured contributions across papers | Manual curation bottleneck |
| **Octopus.ac / ResearchEquals** (Jisc-backed) | Micropublications; explicitly welcomes negative results; removes editorial gatekeepers | Asks authors to change behavior first; niche adoption. Validates demand for the *idea*, shows the *path* fails |
| **DeSci / ResearchHub etc.** | Tokenized peer-review rewards, reputation experiments; 2026 momentum | Crypto-first repels academics; built the credit roof with no knowledge substrate under it |
| **Elicit / SciSpace / Consensus** | Extraction-as-a-service over huge corpora | They monetize the extraction tax; they don't eliminate it. Centralized, no user-owned substrate |
| **DeepXiv-SDK, "Knows" (agent-native representations)** | 2026 trend: intermediate representations richer than metadata, cheaper than full text | Same direction, no personal-library wedge |

**Net read:** both pillars (machine-readable artifacts; failure traces) are now *validated by
others*. The unclaimed territory is the **adoption machine**: bottom-up conversion of personal
libraries → sharing networks → born-structured publishing → credit. That's a distribution +
product problem, and distribution is our asset.

---

## 2. The staircase (each stage valuable even if the next never happens)

```
Stage 0  Validate            graph > flat RAG on real questions (gate for everything)
Stage 1  Selfish utility     convert YOUR library → claims/trajectory graph + graphRAG
Stage 2  Network             share conversions: lab → org → community (dedup = built-in economics)
Stage 3  Born-structured     authoring tools; new work published in-format; failures captured live
Stage 4  Credit layer        signed provenance graph; eval-as-contribution; reputation economy
```

Design rule: **never ship a stage that depends on the next stage existing.**

---

### Stage 0 — Validation gate — ✅ RUN & PASSED (2026-06-10)

Executed: `experiments/library-graph-validation/` (RESULTS.md). 80-paper real library from
Steven's Zotero, 4 cross-paper questions × 4 arms (A current semantic search / B chunked
full-text / C knowledge graph / D full-context oracle) × Claude synthesis × 3 blind
order-randomized Claude judges. Local MiniLM embedder (the shipped default).

**Result — the graph wins, with nuance:**

| arm | composite | takeaway |
|---|---|---|
| A current semantic search | 7.10 | weakest; drops & *fabricates* papers (invented ESM3, Rectified Flow, etc.) |
| B chunked full-text | 7.15 | ≈ A — **"just ship chunking" is refuted**; failure mode is missing *papers*, not missing *passages* |
| C knowledge graph | **8.07** | beats B on **4/4** questions (Δ +0.67…+1.40); 94% of oracle quality at ~20% of context |
| D full-context oracle | 8.62 | ceiling — but 1.7–6.7× the context cost; infeasible once cluster outgrows window |

Pre-registered rule was "C beats B on ≥3/4 → build the graph." **C beat B on 4/4.** Two
findings the rule didn't anticipate:
1. The graph **beat the oracle outright** on the two purely-structural questions (Q2 consensus
   8.53>8.33, Q4 critique 8.80>8.60) — when the task IS cross-paper structure, structuring
   claims helps more than raw text. **⇒ lead the product with consensus/contradiction synthesis.**
2. The graph's real pitch is **"oracle quality at bounded cost,"** not "only the graph can
   answer this." Long-context is a genuine competitor at small corpus scale; the graph wins on
   cost now and on *feasibility* as the relevant set outgrows the window.

**Weaknesses the experiment exposed (these are the build priorities):**
- C's correctness (8.25) < oracle (8.67) from **occasional overclaimed typed edges** ("SI
  strictly generalizes DDPM/DDIM" — a subsumption overstatement). ⇒ precision threshold on
  typed edges (the handoff's named risk, now observed).
- C's Q3 coverage gap (reached 6/9, traversal added 0): OT papers with distinct vocabulary
  weren't reached by semantic edges from seeds. ⇒ **concept canonicalization + structural
  (citation) edges** — the "accuracy linchpin" — is the highest-value next investment.
- Caveats (honest): Claude was synthesizer AND judge (shared blind spots / self-preference);
  extraction used on-topic prompting (favorable); n=4 questions, one ML/comp-bio library.
  Directional, not definitive.

**Still open (Stage 0 → Stage 1 bridge):**
- [ ] **Conversion-pipeline cost/quality memo** — the validation extracted nodes with favorable
      prompting; now measure real $/paper + claim quality on a cold batch (this IS the converter
      everything downstream depends on; steal OpenEval's evidence taxonomy for the schema).
- [ ] Read Ara spec deeply; decision memo: *adopt Ara (or a profile of it) as interchange format
      vs define our own.* Compatibility with an MIT/Stanford/CMU-backed format may beat ownership;
      their PDF→Ara compiler may be reusable.
- [ ] Re-run a slice with a **non-Claude judge** to retire the self-preference caveat before
      betting heavily.

### Stage 1 — Selfish utility: "convert your collection" (the Library Graph, upgraded)

Everything in handoff doc §4–6 stands (tiered lazy pipeline, Tier 0 free structural+semantic,
query-time extraction, canonicalization, provenance, open-core funnel). Reframe + additions:

- The indexer's LLM-extraction tier IS the **PDF→structured-format converter**. Build it as a
  real artifact store, not an internal cache:
  - Per-paper output = a versioned, content-addressed **structured artifact** (claims, evidence
    pointers w/ item+page provenance, methods, entities) — exportable, diffable, shareable later.
    This is the seed of the Stage 2 network; design the schema for exchange from day one.
  - Schema: claims (explicit/implicit), evidence type (data/citation/knowledge/inference/
    speculation — steal OpenEval's taxonomy), results (collated claims), typed edges
    (supports/contradicts/extends/uses-method).
- Headline tools stay: `zotero_graph_synthesize`, `zotero_graph_find_contradictions`,
  compounding memory (`zotero_remember`).
- **Success metric:** weekly active graph-tool calls; % of users who convert >100 items;
  the demo where graph beats flat search, recorded and shareable.
- Monetization posture unchanged: free/local first, hosted convenience later (managed conversion
  is the natural hosted product — conversion at scale is exactly the painful part).

### Stage 2 — The network: shared conversions (the network-effect unlock)

The economic insight that makes this stage *want* to exist: **conversion is expensive and
duplicated.** Two people with the same paper should never both pay to convert it.

- **2a — Content-addressed conversion registry.** Key = DOI/hash of source + converter version.
  Before converting, check registry; after converting, (optionally) contribute. Even purely
  selfish users benefit → the commons grows as a side effect. This is the BitTorrent move:
  sharing isn't altruism, it's the cheaper path.
- **2b — Groups/orgs.** Lab-level shared graphs (Zotero already has group libraries — ride that
  mental model). A lab's merged claim graph over its collective corpus = immediate institutional
  value (onboarding, "what has our lab tried", grant writing).
- **2c — Community/public layer.** Federated or hosted public claim graph over commonly-held
  papers (start: arXiv/OA subset — no copyright fight; artifacts are *derived facts + pointers*,
  not the text itself ⇒ cleaner legal posture, but get this reviewed).
- Trust mechanics needed here (pre-credit-system): converter version + model provenance on every
  artifact; re-derivability (anyone can re-run conversion and diff); flag/dispute on claims.
- **Cold-start:** seed the registry ourselves — batch-convert top-N most-common OA papers
  (detectable from zotero-mcp usage patterns / public citation counts). Day-one hit rate for new
  users = instant value.

### Stage 3 — Born-structured publishing (stop converting the future)

Trigger: when a community on Stage 2 asks "why am I converting my own new paper?"

- Authoring-side tooling: capture structure **during** research, not after — the Ara "Live
  Research Manager" idea; for computational fields this is a Claude-Code/agent-side hook that
  logs hypotheses, runs, pivots, dead ends into the /trace DAG as they happen. Failure capture
  becomes a byproduct of working, not an extra chore (this is the only way negative results ever
  get recorded at scale).
- Output: artifact-first publication; PDF becomes a *rendering* (auto-generated narrative for
  humans) instead of the source of truth. Inversion complete.
- Venue strategy: don't fight journals head-on. Position as *supplement* first ("structured
  artifact attached to your arXiv submission"), the way code releases crept in. Workshops/
  overlay journals that require artifacts come later.
- Trajectory/lineage layer goes public here: research lineages **across** papers/groups — the
  "tree of what's been tried", positive and negative branches, queryable by agents planning
  experiments ("has anyone tried X?" gets a real answer).

### Stage 4 — Credit & evaluation layer (the "blockchain-ish" part, de-crypto'd)

Only on top of a living substrate. Mechanics sketch:

- **Not a blockchain.** Signed, append-only, provenance-carrying DAG (it's git + web-of-trust,
  not coins). Every claim/eval/replication is a signed assertion by an identity (ORCID-anchored).
  Nanopub trust-network research (2025–26) is directly reusable here.
- **Structural credit:** when your claim/negative-result node is load-bearing in later
  trajectories (traversed, built-on, cited-by-claims), credit flows automatically — measurable
  because the graph records use, not just citation. A failure that saves 10 labs a detour
  finally *shows up*.
- **Evaluation as contribution:** assessing claims (support/uncertain/unsupported, importance)
  builds reputation — Stack-Overflow dynamics, not token dynamics. Machine pre-evaluation
  (OpenEval showed 81% agreement) makes human eval *cheap to do well*: humans adjudicate
  disagreements and high-stakes claims instead of reviewing everything.
- Reputation gates nothing initially (no gatekeeping replay); it's a signal layer first. Whether
  it ever becomes career currency depends on institutions — that's a decade game; design so the
  system is useful even if tenure committees never care.

---

## 3. Strategy notes

- **Wedge asset:** zotero-mcp distribution (~700k downloads) + MCP-substrate position (upstream
  of all agent clients) + already-shipped embeddings/relations. No competitor in §1 has a
  bottom-up adoption machine. Keep one brand/identity (per handoff §5 — no premature repo split).
- **Format politics:** if Ara gets traction, being the *adoption layer for Ara* beats owning a
  rival format. Decide at Stage 0 gate; stay schema-agnostic internally (our store, their
  interchange profile) to hedge.
- **What we don't do:** hypothesis generation (DeepMind/FutureHouse territory — unchanged);
  fighting publishers; crypto framing.
- **Relation to the Rust rewrite plan** (memory: engine-first AI-native rewrite): Stage 1's
  artifact store/indexer is the natural first engine component. Don't rewrite first — validate
  in Python where the pieces exist (Chroma, pipeline, tools), port the hot indexer/graph core
  when shape is proven.
- **Sequencing discipline:** the staircase is also a *fundability* ladder — Stage 1 is an
  open-source feature, Stage 2 is a product/company, Stage 3–4 is a movement (grant-fundable:
  Sloan/Astera/Schmidt/Mellon all fund open-science infra; Octopus got Jisc money for far less).

## 4. Top risks

1. ~~**Stage 0 fails** (graph ≯ flat RAG on personal corpora)~~ → **RETIRED 2026-06-10**: graph
   beat chunked search 4/4 and beat the oracle on structural questions. Residual risk = the lift
   was Claude-judged; confirm with a non-Claude judge before scaling spend.
2. **Conversion quality/cost** — hallucinated claims poison trust permanently. Mitigation:
   provenance-mandatory, precision>recall, re-derivability, human spot-check loops.
3. **Copyright** on shared derived artifacts of paywalled papers. Mitigation: OA-first; share
   claims+pointers not text; legal review before 2c.
4. **Ara/big-lab steamroll** — they have the format + papers; we have distribution. Move fast on
   Stage 1–2, stay compatible.
5. **Incentive-layer naivety** — credit systems attract gaming; ship it last, signal-only first.

## 5. Next actions (concrete, this month)

- [ ] Stage 0 validation experiment (handoff §3 protocol) — schedule it.
- [ ] OpenEval-style extraction pilot on 20 of Steven's PDFs; cost + quality memo.
- [ ] Deep-read Ara (arXiv 2604.24658) + check their repo/compiler; adopt-vs-define memo.
- [ ] Draft artifact schema v0 (claims/evidence/provenance/edges) designed for exchange.
- [ ] Essay (`essay-science-file-format.md`) → polish → publish (blog/HN) to flag-plant the
      narrative and attract collaborators while we build Stage 1.
