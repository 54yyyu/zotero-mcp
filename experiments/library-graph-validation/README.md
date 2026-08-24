# Library Graph — Validation Experiment

**Question this answers:** On a *personal* corpus (tens–hundreds of papers), does a
persistent knowledge **graph** actually beat the existing semantic search + a good LLM
on real cross-paper synthesis questions? If not, the graph is over-engineering and the
wedge should be expressed as chunking + compounding memory instead.

This is the gate from the strategy handoff (`~/Downloads/zotero-graph-handoff.md`, §3)
that must pass *before* building the `[graph]` extra.

## The four test questions (grounded in Steven's actual library)

| id | mode the graph should win at | cluster |
|----|------------------------------|---------|
| Q1 | lineage / "generalizes" relationships | Diffusion (16 papers) |
| Q2 | consensus/conflict detection (headline) | Protein ensembles / Dynamics (9) |
| Q3 | cross-collection latent concept | Optimal transport (SingleCell + Diffusion + Architecture) |
| Q4 | method comparison + internal critique | DNA LMs / variant effect (4) |

## The four arms (same synthesis LLM, only the context differs)

- **A — current semantic search.** One-doc-per-item embedding (title+abstract+fulltext
  truncated to the model's 256-token limit — the *shipped* default), retrieve top-k items.
- **B — chunked full-text search.** ~256-token chunks embedded, retrieve top-k chunks.
  Controls for the "no chunking" confound — without this, the graph could win merely
  because arm A barely sees full text.
- **C — knowledge graph.** Claim/method/finding nodes extracted per paper, cross-paper
  semantic + typed edges, bounded-subgraph traversal from query seeds, provenance.
- **D — full-context oracle.** All cluster papers' text dumped into one long-context call.
  The ceiling. If D ≈ C and D is cheap at this scale, long-context — not a graph — is the
  right tool, and the graph is deferred until corpus size makes D infeasible.

**Fairness controls**
- Retrieval pool = the whole "library" (~all ML+protein collections), so A/B/C must
  actually *find* the right papers; trivial only if you hand them the cluster.
- A, B, C context budgets are matched (~10k tokens). D is intentionally larger (it's the
  ceiling, not a peer).
- Local embedder (`all-MiniLM-L6-v2`, the repo default) for all retrieval → reproducible,
  free, and faithfully replicates the shipped 256-token behaviour.
- Synthesis + extraction + judging done by Claude subagents (same model across arms).
- Judging is blind to arm identity and order-randomized.

## Decision rule

- C clearly beats B on ≥3 of 4 questions → **build the graph.**
- C ≈ B → the value is **chunking + memory**, not a graph.
- C ≈ D but D is cheap at this scale → graph **deferred** until corpus size makes D infeasible.

## Pipeline

```
01_build_corpus.py   -> data/corpus.json     (metadata + fulltext from Zotero SQLite)
02_retrieve.py       -> data/contexts.json   (arm A / B / D contexts + retrieval recall)
03_graph_*.py        -> data/graph.json      (arm C: extracted nodes, edges, subgraphs)
04_*                 -> results/answers.json  (16 syntheses, by Claude subagents)
05_*                 -> results/scores.json   (blind judging)
```

Everything here is scratch (gitignored) — not part of the shipped package.
