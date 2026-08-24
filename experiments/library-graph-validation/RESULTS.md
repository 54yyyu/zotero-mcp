# Library Graph — Validation Results

Run: 2026-06-10. Corpus: 80-paper realistic library from Steven's Zotero (77 with
full text). 4 questions × 4 arms × synthesis by Claude subagents × 3 blind,
order-randomized Claude judges (12 judgments). Local MiniLM embedder. See README
for the design and fairness controls.

## Headline numbers

Composite = mean of 5 judged dimensions (coverage, synthesis, correctness,
faithfulness, directness), averaged over 3 judges. Rank points: 1st=4 … 4th=1.

| arm | composite | rank pts | the takeaway |
|-----|-----------|----------|--------------|
| A — current semantic search | 7.10 | 1.67 | weakest; drops/fabricates papers |
| B — chunked full-text | 7.15 | 1.83 | ≈ A; chunking alone is **not** the win |
| C — knowledge graph | **8.07** | **2.92** | consistent lift; beats oracle on 2/4 |
| D — full-context oracle | 8.62 | 3.58 | ceiling — but 1.7–6.7× the context cost |

Per-question composite:

| question | A | B | C | D | Δ(C−B) |
|----------|---|---|---|---|--------|
| Q1 diffusion lineage | 8.27 | 6.80 | 7.67 | 8.53 | +0.87 |
| Q2 protein-ensemble consensus | 7.27 | 7.87 | **8.53** | 8.33 | +0.67 |
| Q3 optimal-transport cross-cut | 5.67 | 5.87 | 7.27 | **9.00** | +1.40 |
| Q4 DNA-LM critique | 7.20 | 8.07 | **8.80** | 8.60 | +0.73 |

Dimension breakdown (mean over all questions/judges):

| arm | coverage | synthesis | correctness | faithfulness | directness |
|-----|----------|-----------|-------------|--------------|------------|
| A | 5.92 | 7.50 | 7.25 | 7.25 | 7.58 |
| B | 6.58 | 7.83 | 7.08 | 7.00 | 7.25 |
| C | 7.92 | 8.42 | 8.25 | 7.58 | 8.17 |
| D | 9.00 | 8.92 | 8.67 | 8.50 | 8.00 |

## Verdict against the pre-registered decision rule

> "C clearly beats B on ≥3/4 → build the graph."

**C beats B on 4/4** (Δ from +0.67 to +1.40). The rule says **build the graph.**

But the rule didn't anticipate the oracle. Two refinements the data forces:

1. **"Just ship chunking" is refuted.** B (7.15) ≈ A (7.10) overall, and B *lost*
   to A on Q1. Finer retrieval granularity did not capture the value, because the
   failure mode is *missing papers*, not *missing passages* — and chunking can't
   retrieve a paper the ranking never surfaced. The wedge is **not** chunking alone.

2. **The graph's pitch is "oracle quality at bounded cost," not "only the graph can
   answer this."** The oracle wins overall (8.62) — at this corpus scale a question's
   relevant cluster (≤16 papers) still fits in one long-context call. But D used
   **1.7×–6.7× the context** of C (Q1: 6.7×). C reaches **94% of oracle composite at
   ~20% of the context**, and — critically — stays bounded when the cluster won't fit.
   The graph beats the oracle outright on the two purely-structural questions
   (Q2 consensus 8.53>8.33, Q4 critique 8.80>8.60): when the task IS cross-paper
   structure, structuring claims helps more than raw text.

## Why each arm wins/loses (from judge notes — consistent across rounds)

- **A & B fail by omission + fabrication.** Repeatedly declared relevant papers
  "absent"/"off-topic" (Distributional Graphormer, GeneTrajectory, NovoSpaRc,
  PromoterAI) and *invented out-of-library papers* to fill gaps (ESM3, Shannon,
  Mean Flows, Rectified Flow, ESMDiff). Coverage is the killer (A 5.92, B 6.58).
- **C wins on coverage + synthesis.** Claim-node compression fit more papers'
  key points in the same budget (coverage 7.92), and cross-paper edges produced
  sharper consensus/lineage structure (synthesis 8.42). Best non-oracle on 3/4.
- **C's weakness = occasional overclaimed edges** (judges flagged "SI strictly
  generalizes DDPM/DDIM", a "subsumption" overstatement). This is exactly the
  *precision-on-typed-edges* risk the handoff named — correctness 8.25, below D's 8.67.
- **C's coverage gap on Q3** (reached 6/9; traversal added 0): the missed
  optimal-transport papers (spatial reconstruction, gene trajectories) use distinct
  vocabulary, so semantic edges from the seeds didn't reach them. This points
  precisely at the fix: **concept canonicalization + structural (citation) edges**
  — the handoff's "accuracy linchpin" — would connect them.
- **D wins where full text is irreducible**: lineage nuance (Q1, reading the actual
  subsumption argument) and raw coverage (Q3, having the papers at all).

## What this says to do

1. **Build the graph** — it passes the rule and delivers a real, consistent lift.
2. **Lead with consensus/contradiction synthesis** (Q2/Q4 shape) — the graph's
   clearest standalone win, beating even brute-force full context.
3. **Frame value as bounded-cost-at-scale**, not magic. Long-context is a genuine
   competitor at small scale; the graph wins on cost now and on feasibility as the
   relevant set outgrows the context window.
4. **Invest first in the accuracy linchpins the experiment stressed**: entity/concept
   canonicalization + structural citation edges (fixes Q3 coverage), and a
   precision threshold on typed edges (fixes C's overclaim correctness gap).
5. **Chunking is still worth doing** (C's nodes are extracted from full text, and
   A/B's whole-doc-truncation hurt them) — but as plumbing under the graph, not as
   the headline.

## Caveats (honesty)

- Claude was synthesizer **and** judge (shared blind spots, possible self-preference).
- Graph claim-nodes were LLM-extracted with on-topic prompting — a favorable setup;
  the upfront extraction is a real cost the lazy/query-time design must bound.
- Graph covered 41 of 80 papers (the retrieved+cluster union); a full Tier-0 graph
  would cover all — uncovered papers were unreachable, which only *hurt* C here.
- n=4 questions, one library, one domain (ML/comp-bio). Directional, not definitive.
