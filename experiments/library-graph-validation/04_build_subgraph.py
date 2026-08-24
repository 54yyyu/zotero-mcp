#!/usr/bin/env python
"""
Stage 4 — assemble arm C (graph) contexts.

The knowledge graph = all extracted claim/method/finding nodes (the library
graph, Tier-0 built offline) with cross-paper semantic edges between nodes.

Per question, faithfully mimicking the LazyGraphRAG query path:
  1. SEED from retrieval only (arm A keys UNION arm B keys) -- NOT the ground-truth
     cluster. Arm C must FIND the rest by traversal, like A/B must by ranking.
  2. From seed papers' nodes, follow cross-paper semantic edges (cosine >= THRESH)
     to reach neighbour papers (bounded traversal, <= MAX_HOP_PAPERS).
  3. Subgraph = seed papers UNION reached papers.
  4. Assemble a budget-matched context: per paper, its nodes most relevant to the
     query, plus cross-paper link annotations + provenance.

Fairness notes:
  - Graph covers the 41 extracted papers (union of all retrieved + cluster sets),
    a stand-in for a full Tier-0 graph; papers with no extracted nodes are
    unreachable, which only HURTS arm C (conservative).
  - Context budget matched to arms A/B.
  - cluster keys used ONLY to score recall, never injected into context.

Output: data/graph_contexts.json  { qid: {C: str, subgraph_papers, C_recall, ...} }
"""
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
NODES = DATA / "nodes"

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EDGE_THRESH = 0.55       # cosine for a cross-paper semantic edge
MAX_HOP_PAPERS = 6       # bounded traversal
NODES_PER_PAPER = 6      # most query-relevant nodes per paper in context
CTX_CHAR_BUDGET = 42000  # matched to arms A/B


def main():
    corpus = json.loads((DATA / "corpus.json").read_text())
    contexts = json.loads((DATA / "contexts.json").read_text())
    items = corpus["items"]

    # ---- load all nodes ----
    nodes = []  # {gid, key, title, type, text, evidence}
    for f in sorted(NODES.glob("*.json")):
        d = json.loads(f.read_text())
        for nd in d.get("nodes", []):
            nodes.append({
                "gid": len(nodes), "key": d["key"], "title": d.get("title", ""),
                "type": nd.get("type", "claim"), "text": nd.get("text", ""),
                "evidence": nd.get("evidence", ""),
            })
    print(f"Loaded {len(nodes)} nodes across {len(set(n['key'] for n in nodes))} papers")

    model = SentenceTransformer(MODEL)
    nvecs = model.encode([n["text"] for n in nodes], normalize_embeddings=True, show_progress_bar=False)

    by_paper = {}
    for i, n in enumerate(nodes):
        by_paper.setdefault(n["key"], []).append(i)

    graph_ctx = {}
    for qid, ctx in contexts.items():
        r = ctx["retrieved"]
        seeds = set(r["A_keys"]) | set(r["B_keys"])
        cluster = set(r["cluster_keys"])
        qtext = ctx["question"]
        qv = model.encode([qtext], normalize_embeddings=True)[0]

        seed_node_idx = [i for i in range(len(nodes)) if nodes[i]["key"] in seeds]
        if not seed_node_idx:
            graph_ctx[qid] = {"C": "", "subgraph_papers": [], "C_recall": "0/%d" % len(cluster)}
            continue

        # ---- bounded traversal: seed nodes -> cross-paper neighbours ----
        seed_mat = nvecs[seed_node_idx]                  # (S, d)
        sims = nvecs @ seed_mat.T                         # (N, S)
        best_to_seed = sims.max(axis=1)                   # strongest edge of each node to any seed
        hop_strength = {}
        for i in range(len(nodes)):
            k = nodes[i]["key"]
            if k in seeds:
                continue
            if best_to_seed[i] >= EDGE_THRESH:
                hop_strength[k] = hop_strength.get(k, 0.0) + float(best_to_seed[i])
        hop_papers = sorted(hop_strength, key=lambda k: -hop_strength[k])[:MAX_HOP_PAPERS]
        subgraph_papers = list(dict.fromkeys(list(seeds) + hop_papers))

        # ---- assemble context: per paper, top query-relevant nodes ----
        parts, used = [], 0
        # order papers by max node relevance to query
        def paper_rel(k):
            idxs = by_paper.get(k, [])
            return max((float(nvecs[i] @ qv) for i in idxs), default=-1)
        for k in sorted(subgraph_papers, key=lambda k: -paper_rel(k)):
            idxs = by_paper.get(k, [])
            if not idxs:
                continue
            idxs = sorted(idxs, key=lambda i: -float(nvecs[i] @ qv))[:NODES_PER_PAPER]
            title = nodes[idxs[0]]["title"] or items.get(k, {}).get("title", k)
            tag = "SEED" if k in seeds else "via-traversal"
            lines = [f"### {title}  [{tag}]"]
            for i in idxs:
                n = nodes[i]
                ev = f"  (evidence: {n['evidence'][:160]})" if n["evidence"] else ""
                lines.append(f"- [{n['type']}] {n['text']}{ev}")
            # cross-paper links for this paper's nodes
            links = []
            for i in idxs:
                row = nvecs @ nvecs[i]
                for j in np.argsort(-row)[:4]:
                    if nodes[j]["key"] != k and nodes[j]["key"] in subgraph_papers and row[j] >= EDGE_THRESH:
                        links.append(f"   ~ links to «{(nodes[j]['title'] or nodes[j]['key'])[:40]}»: {nodes[j]['text'][:90]}")
            if links:
                lines.append("  cross-paper edges:")
                lines.extend(dict.fromkeys(links[:5]))
            block = "\n".join(lines) + "\n"
            if used + len(block) > CTX_CHAR_BUDGET:
                break
            parts.append(block)
            used += len(block)

        ctx_c = "\n".join(parts)
        reached_cluster = len(set(subgraph_papers) & cluster)
        # how many cluster papers did traversal ADD beyond what retrieval seeded?
        added = len((set(hop_papers) & cluster) - seeds)
        graph_ctx[qid] = {
            "C": ctx_c,
            "subgraph_papers": subgraph_papers,
            "seed_papers": sorted(seeds),
            "hop_papers": hop_papers,
            "C_recall": f"{reached_cluster}/{len(cluster)}",
            "traversal_added_cluster_papers": added,
            "seed_recall": f"{len(seeds & cluster)}/{len(cluster)}",
        }
        print(f"{qid}: seeds {len(seeds)} (cluster {len(seeds & cluster)}/{len(cluster)}) "
              f"-> +{len(hop_papers)} hop papers -> C recall {reached_cluster}/{len(cluster)} "
              f"(traversal added {added} cluster papers)")

    (DATA / "graph_contexts.json").write_text(json.dumps(graph_ctx, indent=2))
    print(f"\nWrote {DATA/'graph_contexts.json'}")


if __name__ == "__main__":
    main()
