#!/usr/bin/env python
"""
Stage 1 — build the corpus for the Library Graph validation experiment.

Pulls metadata + full text directly from Zotero's SQLite (via the repo's
LocalZoteroReader) for:
  - the retrieval POOL (the "realistic personal library" arms A/B must search), and
  - the four QUESTION CLUSTERS (ground-truth relevant papers per question).

Output: data/corpus.json  { items: {key: {...}}, pool: [keys], clusters: {qid: [keys]} }
"""
import json
import sqlite3
import sys
from pathlib import Path

# repo import
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from zotero_mcp.local_db import LocalZoteroReader  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

# ---- Collections (name -> Zotero collection key) -------------------------
COLLECTIONS = {
    "Diffusion": "L9VYBVAJ",
    "Architecture": "G2Z6YCXL",
    "Dynamics": "SYJDTNU8",
    "ProteinDesign": "GHSQWMZN",
    "ProteinFolding": "A9QFLC4G",
    "PLInteraction": "Q4T7L29E",
    "PPInteraction": "4JVCC5G8",
    "SingleCell": "IJP3VPEP",
    "VariantEffect": "WL5NWJL9",
    "DREAM": "CRWISFGV",
    "MechInterp": "8TJUSWF9",
    "TheoryLLM": "Q5DSUNGL",
    "Techniques": "98H55MKY",
    "Robotics": "MQXEDFBY",
}

# ---- Question clusters ----------------------------------------------------
# Defined as (collections to union) + (explicit extra item keys to add).
# The four validation questions:
QUESTIONS = {
    "Q1_diffusion_lineage": {
        "text": ("Across my diffusion papers, which frameworks claim to unify or "
                 "generalize the others (flow matching vs stochastic interpolants vs "
                 "score-based diffusion vs rectified flow), and what is the actual "
                 "subsumption hierarchy?"),
        "collections": ["Diffusion"],
        "extra_keys": [],
    },
    "Q2_protein_ensemble_consensus": {
        "text": ("Do my protein-ensemble-generation methods (AlphaFlow, P2DFlow, "
                 "Distributional Graphormer, BioEmu, ESMAdam, force-guided SE(3) "
                 "diffusion) actually claim to recover the true Boltzmann/equilibrium "
                 "distribution, or just generate diverse conformations? Where do they "
                 "disagree?"),
        "collections": ["Dynamics"],
        "extra_keys": [],
    },
    "Q3_optimal_transport_crosscut": {
        "text": ("Where in my library is optimal transport used, and for what distinct "
                 "purposes -- generative modeling vs data alignment vs trajectory "
                 "inference vs Schrodinger bridges?"),
        # OT is latent across SingleCell + Diffusion + Architecture; defined by
        # explicit keys plus the SingleCell collection.
        "collections": ["SingleCell"],
        "extra_keys": [
            "46F2QSTY",  # Optimal transport mapping via input convex neural networks (Architecture)
            "36CMANBU",  # Diffusion Schrodinger Bridge Matching (Diffusion)
            "C7VCWNCW",  # Stochastic Interpolants (Diffusion)
        ],
    },
    "Q4_dna_lm_variant_critique": {
        "text": ("Do my DNA language model papers agree they predict variant effects "
                 "well -- and how does the benchmarking paper's conclusion compare to "
                 "the method papers' own claims?"),
        "collections": ["VariantEffect"],
        "extra_keys": [],
    },
}

# Pool = realistic library the retrieval arms must search: union of the major
# ML + protein trees (includes all four clusters plus distractors).
POOL_COLLECTIONS = [
    "Diffusion", "Architecture", "Dynamics", "ProteinDesign", "ProteinFolding",
    "PLInteraction", "PPInteraction", "SingleCell", "VariantEffect", "DREAM",
    "MechInterp", "TheoryLLM", "Techniques", "Robotics",
]


def collection_item_keys(conn: sqlite3.Connection, coll_key: str) -> list[str]:
    """Resolve item keys belonging to a collection (top-level items only)."""
    q = """
    SELECT it.key
    FROM collections c
    JOIN collectionItems ci ON ci.collectionID = c.collectionID
    JOIN items it ON it.itemID = ci.itemID
    JOIN itemTypes ty ON ty.itemTypeID = it.itemTypeID
    WHERE c.key = ?
      AND ty.typeName NOT IN ('attachment', 'note', 'annotation')
      AND it.itemID NOT IN (SELECT itemID FROM deletedItems)
    """
    return [r[0] for r in conn.execute(q, (coll_key,))]


def main():
    try:
        reader = LocalZoteroReader()
    except Exception as e:
        print(f"ERROR: could not open local Zotero DB: {e}", file=sys.stderr)
        sys.exit(1)
    conn = reader._get_connection()

    # Resolve membership
    coll_keys = {name: set(collection_item_keys(conn, k)) for name, k in COLLECTIONS.items()}
    for name, ks in coll_keys.items():
        print(f"  {name:16s}: {len(ks)} items")

    clusters = {}
    for qid, q in QUESTIONS.items():
        keys = set(q["extra_keys"])
        for cname in q["collections"]:
            keys |= coll_keys[cname]
        clusters[qid] = sorted(keys)
        print(f"  cluster {qid}: {len(clusters[qid])} items")

    pool = set()
    for cname in POOL_COLLECTIONS:
        pool |= coll_keys[cname]
    # ensure every cluster paper is in the pool
    for ks in clusters.values():
        pool |= set(ks)
    pool = sorted(pool)
    print(f"  POOL: {len(pool)} items")

    # Pull metadata + fulltext for every pool item
    items = {}
    for i, key in enumerate(pool, 1):
        it = reader.get_item_by_key(key)
        if it is None:
            print(f"    [{i}/{len(pool)}] {key}: NOT FOUND", file=sys.stderr)
            continue
        ft, src = None, None
        res = reader.extract_fulltext_for_item(it.item_id)
        if res:
            ft, src = res
        items[key] = {
            "key": key,
            "title": it.title,
            "authors": it.creators,
            "date": (it.date_added or "")[:10],
            "item_type": it.item_type,
            "doi": it.doi,
            "abstract": it.abstract,
            "fulltext": ft or "",
            "fulltext_source": src,
            "fulltext_chars": len(ft or ""),
        }
        print(f"    [{i}/{len(pool)}] {key}: {items[key]['fulltext_chars']:>7d} chars  {(it.title or '')[:60]}")

    out = {"questions": QUESTIONS, "pool": pool, "clusters": clusters, "items": items}
    (DATA / "corpus.json").write_text(json.dumps(out, indent=2))
    n_ft = sum(1 for v in items.values() if v["fulltext_chars"] > 1000)
    print(f"\nWrote {DATA/'corpus.json'}: {len(items)} items, {n_ft} with substantial fulltext")


if __name__ == "__main__":
    main()
