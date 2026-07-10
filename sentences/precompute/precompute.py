"""Embed the corpus, project to 3D, quantize, and export runtime tiers.

Pipeline:
  1. embed corpus.jsonl with all-MiniLM-L6-v2 (L2-normalized, 384d)
  2. UMAP -> 3D (cosine)
  3. quantize embeddings to int8 / int16 (+ keep f32), per-vector scale
  4. export tiers x precisions + a shared manifest, plus data/index.json

Because corpus.jsonl is round-robin ordered across sources, every tier is a
prefix of the same run and keeps the full register spread.

Run:
  python precompute.py --tiers 10000,40000,100000 --default 40000
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SANITY_QUERIES = 100
TOP_K = 10


def load_corpus():
    texts, sources, registers = [], [], []
    for line in open(HERE / "corpus.jsonl", encoding="utf-8"):
        row = json.loads(line)
        texts.append(row["text"])
        sources.append(row["source"])
        registers.append(row["register"])
    return texts, sources, registers


# model key -> (python id, browser/ONNX id, dims). Both ids must exist and be
# parity-verified (see parity_*.py) so offline and in-browser vectors match.
MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "Xenova/all-MiniLM-L6-v2", 384),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "Xenova/all-mpnet-base-v2", 768),
}


def embed(texts, model_id):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id)
    print(f"embedding {len(texts)} passages with {model_id} …", flush=True)
    return model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True,
        batch_size=256, show_progress_bar=True,
    ).astype(np.float32)


def quantize(emb, bits):
    """Per-vector symmetric quantization. Returns (codes, scales)."""
    qmax = (1 << (bits - 1)) - 1  # 127 for int8, 32767 for int16
    peak = np.abs(emb).max(axis=1, keepdims=True)
    peak[peak == 0] = 1.0
    scales = (peak / qmax).astype(np.float32)
    dtype = np.int8 if bits == 8 else np.int16
    codes = np.round(emb / scales).clip(-qmax, qmax).astype(dtype)
    return codes, scales.reshape(-1)


def verify_quant(emb, bits, n=SANITY_QUERIES, k=TOP_K):
    """Top-k neighbor overlap between f32 and dequantized, on random queries."""
    codes, scales = quantize(emb, bits)
    deq = codes.astype(np.float32) * scales.reshape(-1, 1)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(emb), size=min(n, len(emb)), replace=False)
    overlaps = []
    for i in idx:
        true_top = set(np.argsort(emb @ emb[i])[::-1][1:k + 1])
        deq_top = set(np.argsort(deq @ deq[i])[::-1][1:k + 1])
        overlaps.append(len(true_top & deq_top) / k)
    return float(np.mean(overlaps))


def run_umap(emb):
    import umap
    print("running UMAP -> 3D …", flush=True)
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                        metric="cosine", random_state=42, verbose=True)
    return reducer.fit_transform(emb).astype(np.float32)


SOURCE_PALETTE = {"arxiv": "#78e0ff", "wikipedia": "#8de28f",
                  "eli5": "#ffc66d", "social": "#ff8db3"}
SOURCE_NAMES = ["arxiv", "wikipedia", "eli5", "social"]


def write_precision(folder, emb_slice, bits):
    folder.mkdir(parents=True, exist_ok=True)
    if bits == 32:
        emb_slice.tofile(folder / "embeddings.bin")
    else:
        codes, scales = quantize(emb_slice, bits)
        codes.tofile(folder / "embeddings.bin")
        scales.tofile(folder / "scales.bin")


def write_shared_assets(tier_dir, texts, source_idx, tier):
    """Model-independent per-tier assets: projection is model-specific and
    lives under the model folder; texts/sources are shared across models."""
    source_idx[:tier].tofile(tier_dir / "sources.bin")
    blob = bytearray()
    offsets = [0]
    for t in texts[:tier]:
        blob += t.encode("utf-8")
        offsets.append(len(blob))
    (tier_dir / "texts.bin").write_bytes(bytes(blob))
    np.array(offsets, dtype=np.uint32).tofile(tier_dir / "text_offsets.bin")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="minilm", choices=list(MODELS))
    parser.add_argument("--tiers", default="10000,40000,100000")
    parser.add_argument("--default", type=int, default=40000)
    parser.add_argument("--out", default="../data")
    args = parser.parse_args()

    model_id, browser_id, dim = MODELS[args.model]
    out = HERE / args.out
    out.mkdir(parents=True, exist_ok=True)

    texts, sources, registers = load_corpus()
    count = len(texts)
    tiers = sorted({int(t) for t in args.tiers.split(",") if int(t) <= count})
    source_idx = np.array([SOURCE_NAMES.index(s) for s in sources], dtype=np.uint8)
    source_meta = [{"name": s, "color": SOURCE_PALETTE[s]} for s in SOURCE_NAMES]

    emb = embed(texts, model_id)
    proj = run_umap(emb)

    print("\nquantization accuracy (top-10 overlap vs f32, gate > 0.95):")
    for bits in (8, 16):
        print(f"  int{bits}: {verify_quant(emb, bits):.3f}")

    for tier in tiers:
        tier_dir = out / str(tier)
        tier_dir.mkdir(parents=True, exist_ok=True)
        if not (tier_dir / "texts.bin").exists():
            write_shared_assets(tier_dir, texts, source_idx, tier)

        model_dir = tier_dir / args.model
        model_dir.mkdir(parents=True, exist_ok=True)
        proj[:tier].tofile(model_dir / "projected.bin")

        manifest = {"count": tier, "dim": dim, "projectionLength": 3, "sources": source_meta}
        (model_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        precisions = ["int8"]
        if tier <= 40000:
            precisions += ["int16", "f32"]
        for prec in precisions:
            bits = {"int8": 8, "int16": 16, "f32": 32}[prec]
            write_precision(model_dir / prec, emb[:tier], bits)
        print(f"data/{tier}/{args.model}: {precisions}  (int8 {tier*dim/1e6:.0f}MB)")

    rebuild_index(out, args.default)


def rebuild_index(out, default):
    """Scan the data tree and emit index.json describing every model x tier x
    precision that exists — so adding a model is just another precompute run."""
    tiers = sorted(int(p.name) for p in out.iterdir() if p.is_dir() and p.name.isdigit())
    model_meta = {"minilm": {"label": "MiniLM-L6 (384d, fast)", "browserId": "Xenova/all-MiniLM-L6-v2"},
                  "mpnet": {"label": "mpnet-base (768d, sharper)", "browserId": "Xenova/all-mpnet-base-v2"}}
    tier_entries = []
    models_present = set()
    for tier in tiers:
        tier_dir = out / str(tier)
        models = {}
        for mdir in tier_dir.iterdir():
            if not mdir.is_dir():
                continue
            precisions = [p.name for p in mdir.iterdir() if p.is_dir()]
            if precisions:
                models[mdir.name] = precisions
                models_present.add(mdir.name)
        tier_entries.append({"size": tier, "models": models})
    index = {
        "tiers": tier_entries,
        "models": [{"key": k, **model_meta[k]} for k in ("minilm", "mpnet") if k in models_present],
        "default": default if default in tiers else tiers[0],
        "defaultModel": "mpnet" if "mpnet" in models_present else "minilm",
        "defaultPrecision": "int8",
    }
    (out / "index.json").write_text(json.dumps(index), encoding="utf-8")
    print(f"\nwrote data/index.json  tiers={tiers} models={sorted(models_present)}")


if __name__ == "__main__":
    main()
