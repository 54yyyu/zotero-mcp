#!/usr/bin/env python
"""
Stage 2 — build retrieval indices and assemble per-arm contexts.

Uses a LOCAL embedder (all-MiniLM-L6-v2, the repo's shipped default, 256-token
limit) so the experiment is reproducible and free, and so arm A faithfully
replicates the *shipped* one-doc-per-item-truncated-to-256 behaviour.

Arms produced here:
  A  = current semantic search: whole-doc embedding (title+authors+abstract+
       fulltext truncated to model limit), retrieve top-k ITEMS.
  B  = chunked full-text search: ~256-token chunks embedded, retrieve top-k CHUNKS.
  D  = full-context oracle: ALL cluster papers' text (truncated per paper).
Arm C (graph) is assembled in a later stage (needs LLM-extracted claim nodes).

Fairness: A, B context budgets are matched (~CTX_TOKENS). D is intentionally
larger (it is the ceiling). Retrieval pool = the whole "library" (corpus.pool),
so retrieval must actually FIND the right papers (esp. for the cross-cut Q3).

Output: data/contexts.json  { qid: {A: str, B: str, D: str, retrieved: {...}} }
"""
import json
import os
import re
from pathlib import Path

os.environ.setdefault("USE_TF", "0")          # force torch backend; skip TF/Keras-3 import
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # the repo's shipped default embedder
ARM_A_TRUNC_CHARS = 1100            # ~256 tokens: replicate shipped whole-doc truncation
TOP_K_ITEMS = 8                     # arm A
TOP_K_CHUNKS = 45                   # arm B (budget-capped below; ~42k chars to match A/C)
CHUNK_CHARS = 1100                  # ~256 tokens of prose
CHUNK_OVERLAP = 150
CTX_CHAR_BUDGET = 42000            # ~10.5k tokens, matched for A/B
D_PER_PAPER_CHARS = 18000          # oracle: truncate each cluster paper


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def doc_text_for_item(it: dict) -> str:
    """Replicate the shipped _create_document_text granularity."""
    parts = []
    if it.get("title"):
        parts.append(f"Title: {it['title']}")
    if it.get("authors"):
        parts.append(f"Authors: {it['authors']}")
    if it.get("abstract"):
        parts.append(f"Abstract: {it['abstract']}")
    if it.get("fulltext"):
        parts.append(f"Content: {it['fulltext']}")
    return "\n".join(parts)


def main():
    corpus = json.loads((DATA / "corpus.json").read_text())
    items = corpus["items"]
    pool = corpus["pool"]
    clusters = corpus["clusters"]
    questions = corpus["questions"]

    model = SentenceTransformer(MODEL)

    # ---------- Arm A index: one vector per item ----------
    pool_keys = [k for k in pool if items.get(k)]
    # Truncate to ~256 tokens to faithfully replicate the shipped one-doc-per-item
    # behaviour (whole doc truncated to the default model's 256-token limit).
    doc_texts = [doc_text_for_item(items[k])[:ARM_A_TRUNC_CHARS] for k in pool_keys]
    print(f"Embedding {len(pool_keys)} item-docs (arm A)...")
    item_vecs = model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=False)

    # ---------- Arm B index: chunks ----------
    chunk_recs = []  # (key, chunk_idx, text)
    for k in pool_keys:
        ft = items[k].get("fulltext") or ""
        body = (items[k].get("abstract") or "") + "\n" + ft
        for ci, ch in enumerate(chunk_text(body)):
            chunk_recs.append((k, ci, ch))
    print(f"Embedding {len(chunk_recs)} chunks (arm B)...")
    chunk_vecs = model.encode([c[2] for c in chunk_recs], normalize_embeddings=True, show_progress_bar=False)

    contexts = {}
    for qid, q in questions.items():
        qtext = q["text"]
        qv = model.encode([qtext], normalize_embeddings=True)[0]

        # ---- Arm A: top-k items ----
        sims = item_vecs @ qv
        order = np.argsort(-sims)[:TOP_K_ITEMS]
        a_keys = [pool_keys[i] for i in order]
        a_parts, used = [], 0
        for rank, i in enumerate(order, 1):
            k = pool_keys[i]
            it = items[k]
            excerpt = (it.get("fulltext") or "")[:6000]
            block = (f"[A{rank}] {it['title']} ({it.get('authors','')[:80]})\n"
                     f"Abstract: {it.get('abstract') or '(none)'}\n"
                     f"Excerpt: {excerpt}\n")
            if used + len(block) > CTX_CHAR_BUDGET:
                block = block[: CTX_CHAR_BUDGET - used]
            a_parts.append(block)
            used += len(block)
            if used >= CTX_CHAR_BUDGET:
                break
        ctx_a = "\n".join(a_parts)

        # ---- Arm B: top-k chunks ----
        csims = chunk_vecs @ qv
        corder = np.argsort(-csims)[:TOP_K_CHUNKS]
        b_parts, used, b_keys = [], 0, []
        for rank, i in enumerate(corder, 1):
            k, ci, ch = chunk_recs[i]
            b_keys.append(k)
            title = items[k]["title"]
            block = f"[B{rank}] (from: {title}, chunk {ci}) {ch}\n"
            if used + len(block) > CTX_CHAR_BUDGET:
                block = block[: CTX_CHAR_BUDGET - used]
            b_parts.append(block)
            used += len(block)
            if used >= CTX_CHAR_BUDGET:
                break
        ctx_b = "\n".join(b_parts)

        # ---- Arm D: full cluster (oracle) ----
        d_parts = []
        for k in clusters[qid]:
            it = items.get(k)
            if not it:
                continue
            txt = (it.get("fulltext") or it.get("abstract") or "")[:D_PER_PAPER_CHARS]
            d_parts.append(f"=== {it['title']} ({it.get('authors','')[:80]}) ===\n{txt}\n")
        ctx_d = "\n".join(d_parts)

        # retrieval diagnostics: did A/B actually find the ground-truth cluster?
        cluster_set = set(clusters[qid])
        a_hits = sum(1 for k in a_keys if k in cluster_set)
        b_hits = len(set(b_keys) & cluster_set)
        contexts[qid] = {
            "question": qtext,
            "A": ctx_a, "B": ctx_b, "D": ctx_d,
            "retrieved": {
                "A_keys": a_keys,
                "B_keys": list(dict.fromkeys(b_keys)),
                "cluster_keys": sorted(cluster_set),
                "A_recall": f"{a_hits}/{len(cluster_set)}",
                "B_recall_papers": f"{b_hits}/{len(cluster_set)}",
            },
        }
        print(f"{qid}: A found {a_hits}/{len(cluster_set)} cluster papers; "
              f"B found {b_hits}/{len(cluster_set)} (as chunks)")

    (DATA / "contexts.json").write_text(json.dumps(contexts, indent=2))
    print(f"\nWrote {DATA/'contexts.json'}")


if __name__ == "__main__":
    main()
