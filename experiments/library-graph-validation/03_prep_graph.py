#!/usr/bin/env python
"""
Stage 3a — prepare graph extraction inputs (arm C).

The graph operates over the realistic retrieved neighborhood (LazyGraphRAG-style:
seeds from retrieval, then claim extraction at query time over the bounded set).
Seed papers per question = cluster_keys UNION arm-A retrieved UNION arm-B retrieved.
We extract claim/method/finding nodes for the union of all seed papers across the
4 questions, then (next stage) link them with cross-paper semantic edges.

Writes:
  data/papers/<key>.txt        -- text for each paper needing extraction
  data/extract_manifest.json   -- {batches: [[key,...]], seeds_per_q: {qid: [keys]}}
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PAPERS = DATA / "papers"
PAPERS.mkdir(exist_ok=True)

MAX_CHARS = 20000      # per paper, fed to extractor subagent
BATCH = 5              # papers per extraction subagent


def main():
    corpus = json.loads((DATA / "corpus.json").read_text())
    contexts = json.loads((DATA / "contexts.json").read_text())
    items = corpus["items"]

    seeds_per_q = {}
    all_seeds = set()
    for qid, ctx in contexts.items():
        r = ctx["retrieved"]
        seeds = set(r["cluster_keys"]) | set(r["A_keys"]) | set(r["B_keys"])
        seeds = {k for k in seeds if items.get(k, {}).get("fulltext_chars", 0) > 1000}
        seeds_per_q[qid] = sorted(seeds)
        all_seeds |= seeds

    all_seeds = sorted(all_seeds)
    for key in all_seeds:
        it = items[key]
        txt = (it.get("fulltext") or it.get("abstract") or "")[:MAX_CHARS]
        body = f"PAPER KEY: {key}\nTITLE: {it['title']}\nAUTHORS: {it.get('authors','')}\n\n{txt}"
        (PAPERS / f"{key}.txt").write_text(body)

    batches = [all_seeds[i:i + BATCH] for i in range(0, len(all_seeds), BATCH)]
    manifest = {"batches": batches, "seeds_per_q": seeds_per_q, "n_papers": len(all_seeds)}
    (DATA / "extract_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Seed papers needing extraction: {len(all_seeds)}")
    for qid, s in seeds_per_q.items():
        print(f"  {qid}: {len(s)} seeds")
    print(f"Batches of {BATCH}: {len(batches)}")
    print(f"Wrote per-paper text to {PAPERS}/ and manifest to {DATA/'extract_manifest.json'}")


if __name__ == "__main__":
    main()
