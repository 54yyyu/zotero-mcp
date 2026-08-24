#!/usr/bin/env python
"""
Stage 5a — write the 16 synthesis input files (4 questions x 4 arms).

Each file has the SAME instruction header and question; only the CONTEXT differs.
Arm labels are written into filenames only (not into the context shown to the
synthesizer would-be-biased -- but synthesis isn't blind; judging is). Synthesis
subagents read one file and write one answer.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results" / "synth_inputs"
OUT.mkdir(parents=True, exist_ok=True)

INSTR = """You are answering a research question using ONLY the provided context from a
researcher's personal paper library. Write a focused synthesis that:
- directly answers the question with cross-paper structure (consensus, disagreement,
  lineage/subsumption, or a taxonomy as the question demands);
- cites specific papers by title for every substantive claim;
- is explicit about genuine disagreements or contested points between papers;
- does NOT pad, does NOT use outside knowledge beyond the context, and flags if the
  context is insufficient to answer part of the question.
Keep it tight: ~400-600 words.

Do not mention which retrieval method produced the context. Just answer.
"""


def main():
    contexts = json.loads((DATA / "contexts.json").read_text())
    graph = json.loads((DATA / "graph_contexts.json").read_text())

    n = 0
    for qid, ctx in contexts.items():
        q = ctx["question"]
        arms = {"A": ctx["A"], "B": ctx["B"], "C": graph[qid]["C"], "D": ctx["D"]}
        for arm, body in arms.items():
            doc = (f"{INSTR}\n\n=== QUESTION ===\n{q}\n\n=== CONTEXT ===\n{body}\n")
            (OUT / f"{qid}__{arm}.md").write_text(doc)
            n += 1
    print(f"Wrote {n} synthesis input files to {OUT}")


if __name__ == "__main__":
    main()
