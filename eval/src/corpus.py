"""The site's sentence corpus, loaded exactly as the browser loads it (int8 + scales)."""
import json

import numpy as np

from common import CONFIG, REPO_DIR

_state = None


def _load():
    global _state
    if _state is not None:
        return _state
    spec = CONFIG["corpus"]
    tier_dir = REPO_DIR / "sentences" / "data" / str(spec["tier"])
    model_dir = tier_dir / spec["model"]
    prec_dir = model_dir / spec["precision"]

    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    count, dim = manifest["count"], manifest["dim"]

    codes = np.frombuffer((prec_dir / "embeddings.bin").read_bytes(), dtype=np.int8)
    codes = codes.reshape(count, dim).astype(np.float32)
    scales = np.frombuffer((prec_dir / "scales.bin").read_bytes(), dtype=np.float32)
    source_idx = np.frombuffer((tier_dir / "sources.bin").read_bytes(), dtype=np.uint8)
    text_blob = (tier_dir / "texts.bin").read_bytes()
    offsets = np.frombuffer((tier_dir / "text_offsets.bin").read_bytes(), dtype=np.uint32)

    _state = {
        "codes": codes, "scales": scales, "source_idx": source_idx,
        "sources": [s["name"] for s in manifest["sources"]],
        "text_blob": text_blob, "offsets": offsets, "count": count,
    }
    return _state


def top_k(query_vec: np.ndarray, k: int | None = None) -> list[dict]:
    """Same ranking the site's search worker computes: (int8 codes . q) * scale."""
    s = _load()
    k = k or CONFIG["corpus"]["top_k"]
    scores = (s["codes"] @ query_vec.astype(np.float32)) * s["scales"]
    idx = np.argpartition(-scores, k)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [{"index": int(i), "score": float(scores[i]),
             "source": s["sources"][s["source_idx"][i]],
             "text": passage_text(int(i))} for i in idx]


def passage_text(index: int) -> str:
    s = _load()
    return s["text_blob"][s["offsets"][index]:s["offsets"][index + 1]].decode("utf-8")


def source_mix(neighbors: list[dict]) -> dict:
    s = _load()
    mix = {name: 0 for name in s["sources"]}
    for n in neighbors:
        mix[n["source"]] += 1
    total = max(1, len(neighbors))
    return {name: round(v / total, 3) for name, v in mix.items()}


def overlap(a: list[dict], b: list[dict]) -> float:
    ia, ib = {n["index"] for n in a}, {n["index"] for n in b}
    return round(len(ia & ib) / max(1, len(ia)), 3)
