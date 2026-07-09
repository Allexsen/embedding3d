# Embedding3D

Interactive 3D word embedding explorer — GloVe vectors projected to 3D with UMAP, rendered as a WebGL point cloud. Fully static, no backend, no build step.

Inspired by the [TensorFlow Embedding Projector](https://projector.tensorflow.org), with word arithmetic on top.

## Features

- **Search & arithmetic** — one query box for words and vector math: `king - man + woman`, `paris - france + italy`, `(coffee + tea) * 0.5`. Autocomplete, typo suggestions, quoted tokens (`"1990"`, `"u.s."`).
- **Traversal path** — expressions render as a hop-by-hop path through the space, color-graded from start to result. A jumper bar lets you scrub through the intermediate stops and treat any partial sum as the destination.
- **Dataset tiers** — 500 to 400,000 words (frequency-ordered slices of one UMAP run), switchable at runtime. Default loads the 20K tier (~4 MB).
- **Tuning** — neighbor count, point size, depth fog, dimming, words-shown limit, points-vs-vectors display, labels, auto-rotate, animations.
- Nearest-neighbor search runs in a Web Worker on pre-normalized vectors; rendering and selection animations are GPU-side, so the 400K tier stays interactive.

## Run

Serve the repo root over HTTP and open it:

```bash
python -m http.server 8000
# → http://localhost:8000
```

Opening `index.html` directly from disk falls back to a small built-in demo dataset.

## Regenerating the data

Requires Python 3.11–3.13 (umap-learn depends on numba).

```bash
cd precompute
pip install -r requirements.txt
# GloVe from https://nlp.stanford.edu/projects/glove/ (glove.6B.zip)
python precompute.py --glove glove.6B.50d.txt --top 400000 --out ..   # UMAP, ~1-2h at 400K
python chunk.py --src .. --out ../data                                # slice into size tiers
```

`chunk.py` writes `data/<size>/{embeddings.bin,projected.bin,manifest.json}` plus `data/index.json`. Because GloVe is frequency-ordered, every tier is a prefix of the same projection, so word positions stay consistent across sizes.

## Stack

Vanilla JS + raw WebGL (points, lines, fog, animations in shaders), a 2D canvas overlay for labels and markers, and typed-array binary assets. No dependencies at runtime.

## License

[MIT](LICENSE)
