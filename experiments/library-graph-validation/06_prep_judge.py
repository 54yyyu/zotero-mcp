#!/usr/bin/env python
"""
Stage 6a — prepare BLIND, order-randomized judge inputs.

For each question we create 3 independent judge rounds, each with a different
random ordering of the four answers (anonymized as Option 1..4). The judge never
sees which arm (A/B/C/D) produced an answer. A key file records the mapping for
later de-anonymization.
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ANS = HERE / "results" / "answers"
OUT = HERE / "results" / "judge_inputs"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = ["A", "B", "C", "D"]
ROUNDS = 3

RUBRIC = """You are a meticulous expert reviewer comparing four answers to the SAME research
question, each generated from a different (undisclosed) evidence-retrieval method over
the same personal paper library. Judge ONLY answer quality. Do not try to guess the method.

Score each answer 1-10 on each dimension:
- coverage: how many of the genuinely relevant papers/points are represented.
- synthesis: quality of CROSS-PAPER structure (consensus, disagreement, lineage/subsumption,
  taxonomy) -- NOT just a list of per-paper summaries.
- correctness: are the relational and technical claims ACCURATE? Heavily penalize hallucinated
  or wrong subsumption/contradiction claims (e.g. "X generalizes Y" when false).
- faithfulness: claims attributed to specific papers; no unsupported assertions; no outside facts.
- directness: answers the actual question precisely without padding (length is NOT a virtue).

Then give an overall ranking of the four (best to worst). Ties are NOT allowed in the ranking.

The relevant papers for this question (ground truth, for assessing coverage) are listed below.
Be skeptical: a confident, well-written answer can still be wrong or shallow.

Output STRICT JSON only, no prose:
{"scores": {"Option 1": {"coverage":N,"synthesis":N,"correctness":N,"faithfulness":N,"directness":N},
            "Option 2": {...}, "Option 3": {...}, "Option 4": {...}},
 "ranking": ["Option X","Option Y","Option Z","Option W"],
 "notes": "1-3 sentences on the key differentiators"}
"""


def main():
    corpus = json.loads((DATA / "corpus.json").read_text())
    items = corpus["items"]
    clusters = corpus["clusters"]
    questions = corpus["questions"]

    key = {}
    n = 0
    for qid, q in questions.items():
        answers = {arm: (ANS / f"{qid}__{arm}.md").read_text().strip() for arm in ARMS}
        cluster_titles = []
        for k in clusters[qid]:
            it = items.get(k, {})
            cluster_titles.append(f"- {it.get('title', k)} ({(it.get('authors') or '')[:60]})")
        gt = "\n".join(cluster_titles)

        for rnd in range(ROUNDS):
            rng = random.Random(f"{qid}-{rnd}")
            order = ARMS[:]
            rng.shuffle(order)
            opt_to_arm = {f"Option {i+1}": arm for i, arm in enumerate(order)}
            key[f"{qid}__r{rnd}"] = opt_to_arm

            blocks = []
            for i, arm in enumerate(order):
                blocks.append(f"===== Option {i+1} =====\n{answers[arm]}\n")
            doc = (f"{RUBRIC}\n\n=== QUESTION ===\n{q['text']}\n\n"
                   f"=== RELEVANT PAPERS (ground truth) ===\n{gt}\n\n"
                   f"=== ANSWERS TO JUDGE ===\n" + "\n".join(blocks))
            (OUT / f"{qid}__r{rnd}.md").write_text(doc)
            n += 1

    (DATA / "judge_key.json").write_text(json.dumps(key, indent=2))
    print(f"Wrote {n} judge input files to {OUT}")
    print(f"Wrote key to {DATA/'judge_key.json'}")


if __name__ == "__main__":
    main()
