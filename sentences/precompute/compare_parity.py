"""Milestone 0 gate: do Python and transformers.js embeddings agree?

Passes if per-sentence cosine(py, js) > 0.999 for every parity sentence.
Also reports whether the two runtimes preserve the same neighbor ranking
(the property that actually matters for retrieval).
"""

import json
import math
from pathlib import Path

HERE = Path(__file__).parent


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main():
    py = json.loads((HERE / "parity_py.json").read_text())
    js = json.loads((HERE / "parity_js.json").read_text())
    sentences = json.loads((HERE / "parity_sentences.json").read_text(encoding="utf-8"))

    assert len(py) == len(js) == len(sentences), "vector count mismatch"

    print(f"{'#':>2}  {'cos(py,js)':>10}  sentence")
    worst = 1.0
    for i, (a, b, s) in enumerate(zip(py, js, sentences)):
        c = cosine(a, b)
        worst = min(worst, c)
        flag = "" if c > 0.999 else "  <-- FAIL"
        print(f"{i:>2}  {c:>10.6f}  {s[:56]}{flag}")

    # ranking parity: for sentence 0, do both runtimes pick the same top-3?
    def top3(vectors, idx):
        scored = sorted(
            ((cosine(vectors[idx], vectors[j]), j) for j in range(len(vectors)) if j != idx),
            reverse=True,
        )
        return [j for _, j in scored[:3]]

    rank_agree = sum(top3(py, i) == top3(js, i) for i in range(len(py)))

    print(f"\nworst per-sentence cosine: {worst:.6f}  (gate: > 0.999)")
    print(f"top-3 neighbor ranking agrees on {rank_agree}/{len(py)} sentences")
    print("PARITY PASS" if worst > 0.999 else "PARITY FAIL — investigate pooling/quantization")


if __name__ == "__main__":
    main()
