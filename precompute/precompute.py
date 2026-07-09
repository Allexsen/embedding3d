"""Precompute runtime assets for Embedding3D.

Loads GloVe vectors, runs 3D UMAP, and exports the three static files the
frontend loads at startup:

  embeddings.bin  Float32 (count x dim)  raw word vectors (client normalizes)
  projected.bin   Float32 (count x 3)    UMAP coordinates (client recenters/rescales)
  manifest.json   word list + shapes

Run once:
  pip install -r requirements.txt
  python precompute.py --glove glove.6B.50d.txt --top 50000 --out ..

GloVe download: https://nlp.stanford.edu/projects/glove/ (glove.6B.zip)
UMAP on 50K x 50 takes roughly 5-15 minutes.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_glove(path: Path, top_n: int):
    words = []
    vectors = []
    with open(path, "r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= top_n:
                break
            parts = line.rstrip().split(" ")
            words.append(parts[0])
            vectors.append(np.asarray(parts[1:], dtype=np.float32))
    return words, np.vstack(vectors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glove", default="glove.6B.50d.txt", help="path to GloVe txt file")
    parser.add_argument("--top", type=int, default=50000, help="number of words to keep (file is frequency ordered)")
    parser.add_argument("--out", default="..", help="output directory (repo root so GitHub Pages serves the files)")
    parser.add_argument("--neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading GloVe from {args.glove} (top {args.top}) ...")
    words, matrix = load_glove(Path(args.glove), args.top)
    print(f"Loaded {len(words)} words, dim {matrix.shape[1]}.")

    print("Running UMAP (this takes a few minutes) ...")
    import umap  # imported late: slow import

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=42,
        verbose=True,
    )
    projected = reducer.fit_transform(matrix).astype(np.float32)

    print("Writing embeddings.bin / projected.bin / manifest.json ...")
    (out_dir / "embeddings.bin").write_bytes(matrix.astype(np.float32).tobytes())
    (out_dir / "projected.bin").write_bytes(projected.tobytes())

    manifest = {
        "source": Path(args.glove).stem,
        "words": words,
        "count": len(words),
        "vectorLength": int(matrix.shape[1]),
        "projectionLength": 3,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False)

    emb_mb = matrix.nbytes / 1e6
    proj_mb = projected.nbytes / 1e6
    print(f"Done. embeddings.bin {emb_mb:.1f} MB, projected.bin {proj_mb:.1f} MB.")
    print("Spot check — first 10 words:", words[:10])


if __name__ == "__main__":
    main()
