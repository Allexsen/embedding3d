"""Embed the parity sentences with sentence-transformers (reference runtime).

Mean pooling + L2 normalize, matching what transformers.js will do in the
browser. Writes parity_py.json (list of 384-float vectors) for compare_parity.py.
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    sentences = json.loads((HERE / "parity_sentences.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(MODEL)
    # normalize_embeddings=True → L2 normalized; default pooling is mean
    vectors = model.encode(sentences, normalize_embeddings=True, convert_to_numpy=True)
    out = [[round(float(x), 6) for x in vec] for vec in vectors]
    (HERE / "parity_py.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote parity_py.json: {len(out)} vectors x {len(out[0])} dims")


if __name__ == "__main__":
    main()
