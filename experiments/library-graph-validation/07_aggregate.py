#!/usr/bin/env python
"""
Stage 7 — de-anonymize judge outputs and aggregate into a verdict.

Produces per-question and overall mean scores by arm, plus rank-based wins,
plus the experiment's decision rule (C vs B).
"""
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
JOUT = HERE / "results" / "judge_outputs"

ARMS = ["A", "B", "C", "D"]
ARM_NAME = {"A": "current-semantic", "B": "chunked", "C": "graph", "D": "oracle"}
DIMS = ["coverage", "synthesis", "correctness", "faithfulness", "directness"]


def load_json_loose(text):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def main():
    key = json.loads((DATA / "judge_key.json").read_text())
    corpus = json.loads((DATA / "corpus.json").read_text())
    qids = list(corpus["questions"].keys())

    graph = json.loads((DATA / "graph_contexts.json").read_text())
    contexts = json.loads((DATA / "contexts.json").read_text())

    # per (qid, arm) -> list of dim scores; and rank points
    dim_scores = defaultdict(lambda: defaultdict(list))   # qid -> arm -> [mean-per-judge]
    rank_points = defaultdict(lambda: defaultdict(list))  # qid -> arm -> [points]
    overall_by_arm = defaultdict(list)                    # arm -> [composite across all]

    for f in sorted(JOUT.glob("*.json")):
        stem = f.stem  # qid__rN
        qid, rnd = stem.rsplit("__r", 1)
        kmap = key[f"{qid}__r{rnd}"]   # Option k -> arm
        verdict = load_json_loose(f.read_text())
        # scores
        for opt, sc in verdict["scores"].items():
            arm = kmap[opt]
            mean = sum(float(sc[d]) for d in DIMS) / len(DIMS)
            dim_scores[qid][arm].append({"mean": mean, **{d: float(sc[d]) for d in DIMS}})
        # ranking -> points (1st=4 ... 4th=1)
        for pos, opt in enumerate(verdict["ranking"]):
            arm = kmap[opt]
            rank_points[qid][arm].append(4 - pos)

    def avg(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    print("=" * 78)
    print("PER-QUESTION MEAN COMPOSITE SCORE (avg of 5 dims, avg over 3 judges)")
    print("=" * 78)
    header = f"{'question':<34}" + "".join(f"{ARM_NAME[a]:>17}" for a in ARMS)
    print(header)
    composite = defaultdict(dict)
    for qid in qids:
        row = f"{qid:<34}"
        for a in ARMS:
            means = [x["mean"] for x in dim_scores[qid][a]]
            composite[qid][a] = avg(means)
            row += f"{avg(means):>17.2f}"
        print(row)

    print("\n" + "=" * 78)
    print("PER-QUESTION MEAN RANK POINTS (1st=4, 4th=1; over 3 judges)")
    print("=" * 78)
    print(header)
    for qid in qids:
        row = f"{qid:<34}"
        for a in ARMS:
            row += f"{avg(rank_points[qid][a]):>17.2f}"
        print(row)

    print("\n" + "=" * 78)
    print("OVERALL (mean across questions)")
    print("=" * 78)
    for a in ARMS:
        comp = avg([composite[q][a] for q in qids])
        rp = avg([avg(rank_points[q][a]) for q in qids])
        print(f"  {ARM_NAME[a]:<18} composite={comp:5.2f}   rankpts={rp:4.2f}")

    print("\n" + "=" * 78)
    print("DIMENSION BREAKDOWN (mean across all questions & judges)")
    print("=" * 78)
    print(f"{'arm':<18}" + "".join(f"{d:>14}" for d in DIMS))
    for a in ARMS:
        vals = []
        for d in DIMS:
            xs = [x[d] for q in qids for x in dim_scores[q][a]]
            vals.append(avg(xs))
        print(f"{ARM_NAME[a]:<18}" + "".join(f"{v:>14.2f}" for v in vals))

    print("\n" + "=" * 78)
    print("DECISION RULE: graph (C) vs chunked (B), per question")
    print("=" * 78)
    c_wins = 0
    for qid in qids:
        c, b = composite[qid]["C"], composite[qid]["B"]
        d = composite[qid]["D"]
        delta = c - b
        verdict = "C>B" if delta > 0.3 else ("C~B" if abs(delta) <= 0.3 else "B>C")
        if delta > 0.3:
            c_wins += 1
        rec_c = graph[qid]["C_recall"]
        print(f"  {qid:<34} C={c:.2f} B={b:.2f} D={d:.2f}  Δ(C-B)={delta:+.2f}  {verdict}  (C recall {rec_c})")
    print(f"\n  C clearly beats B on {c_wins}/{len(qids)} questions "
          f"(rule: build graph if >=3/4).")

    out = {"composite": composite, "qids": qids}
    (HERE / "results" / "summary.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
