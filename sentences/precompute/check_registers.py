"""Register-separation check (milestone 2 gate).

Embeds a sample per source and verifies the four registers actually cluster
apart — intra-source similarity should exceed cross-source similarity. Also
embeds the dopamine sanity pair and reports its neighbor source-mix, the
effect the whole app is built to show.
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
SOURCES = ["arxiv", "wikipedia", "eli5", "social"]
SAMPLE = 400


def main():
    by_source = {s: [] for s in SOURCES}
    for line in open(HERE / "corpus.jsonl", encoding="utf-8"):
        row = json.loads(line)
        by_source[row["source"]].append(row["text"])

    random.seed(0)
    sampled = {s: random.sample(texts, min(SAMPLE, len(texts))) for s, texts in by_source.items()}

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = {s: model.encode(t, normalize_embeddings=True, convert_to_numpy=True) for s, t in sampled.items()}
    centroids = {s: v.mean(axis=0) for s, v in emb.items()}

    print("mean cosine to each source centroid (row = passages from source):\n")
    print("           " + "  ".join(f"{s[:8]:>8}" for s in SOURCES))
    for s in SOURCES:
        # vectors are unit-norm; cosine to a centroid = mean dot product
        sims = [float((emb[s] @ centroids[t]).mean() / np.linalg.norm(centroids[t])) for t in SOURCES]
        row = "  ".join(f"{v:>8.3f}" for v in sims)
        own = max(range(4), key=lambda i: sims[i])
        flag = "  self-nearest OK" if SOURCES[own] == s else "  <-- NOT self-nearest"
        print(f"{s[:9]:>9}  {row}{flag}")

    # sanity pair: neighbor source-mix for two phrasings of the same idea
    all_texts, all_src = [], []
    for line in open(HERE / "corpus.jsonl", encoding="utf-8"):
        row = json.loads(line)
        all_texts.append(row["text"])
        all_src.append(row["source"])
    corpus_emb = model.encode(all_texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=256, show_progress_bar=False)

    pair = {
        "A (academic)": "dopamine mediates reward prediction error signaling in mesolimbic pathways",
        "B (bro)": "dopamine is basically the feel-good hit you get when something goes better than you expected",
    }
    print("\nneighbor source-mix for the sanity pair (top-50):")
    for label, text in pair.items():
        q = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        top = np.argsort(corpus_emb @ q)[::-1][:50]
        mix = Counter(all_src[i] for i in top)
        print(f"  {label:>14}: " + "  ".join(f"{s}={mix.get(s,0)}" for s in SOURCES))


if __name__ == "__main__":
    main()
