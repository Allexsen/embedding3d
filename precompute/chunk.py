"""Slice precompute output into size-tiered datasets the frontend can pick from.

Reads embeddings.bin / projected.bin / manifest.json (written by precompute.py)
and writes data/<size>/{embeddings.bin,projected.bin,manifest.json} for every
requested size <= available count, plus data/index.json.

Because GloVe is frequency-ordered, size N is simply the first N rows — so one
UMAP run at the largest size serves every tier, and word positions stay stable
across tiers.

Run after precompute.py:
  python chunk.py --src .. --out ../data
"""

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_SIZES = "500,1000,2000,5000,10000,20000,50000,100000,200000,400000"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="..", help="directory containing precompute output")
    parser.add_argument("--out", default="../data", help="output directory for tiered datasets")
    parser.add_argument("--sizes", default=DEFAULT_SIZES, help="comma-separated word counts")
    parser.add_argument("--default", type=int, default=20000, help="size the frontend loads first")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    count = manifest["count"]
    dim = manifest["vectorLength"]
    words = manifest["words"]
    emb = np.fromfile(src / "embeddings.bin", dtype=np.float32).reshape(count, dim)
    proj = np.fromfile(src / "projected.bin", dtype=np.float32).reshape(count, 3)

    sizes = sorted({int(s) for s in args.sizes.split(",") if int(s) <= count})
    if not sizes:
        raise SystemExit(f"no requested size fits the available {count} words")

    for size in sizes:
        folder = out / str(size)
        folder.mkdir(parents=True, exist_ok=True)
        emb[:size].tofile(folder / "embeddings.bin")
        proj[:size].tofile(folder / "projected.bin")
        tier_manifest = {
            "source": manifest.get("source", ""),
            "words": words[:size],
            "count": size,
            "vectorLength": dim,
            "projectionLength": 3,
        }
        (folder / "manifest.json").write_text(
            json.dumps(tier_manifest, ensure_ascii=False), encoding="utf-8"
        )
        print(f"data/{size}: {(size * dim * 4) / 1e6:.1f} MB embeddings")

    default = args.default if args.default in sizes else sizes[-1]
    index = {"sizes": sizes, "default": default, "vectorLength": dim}
    (out / "index.json").write_text(json.dumps(index), encoding="utf-8")
    print(f"data/index.json: sizes={sizes} default={default}")


if __name__ == "__main__":
    main()
