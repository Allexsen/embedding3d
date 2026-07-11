# Embedding3D

Interactive 3D explorations of embedding space — the geometry machines use to
arrange meaning, and what that arrangement quietly does to search results and
AI answers. Fully static, no backend, no build step: everything from
nearest-neighbor search to LLM inference runs in your browser.

Two apps, three demos, cross-linked in the top bars:

| Where | What | The question it answers |
|---|---|---|
| **/** (words, this repo's root) | 3D cloud of up to 400K GloVe word vectors, with word arithmetic | How do words arrange themselves by meaning — and does `king − man + woman` really land on `queen`? |
| **[/sentences](sentences/)** | Same idea phrased two ways, retrieved against a 100K-passage corpus | Does *how* you word something change what search finds? |
| **[/sentences](sentences/)** ✦ Generate | Each phrasing sampled against a real LLM, answers placed in the same space | …and does it change what a model *says*? |

**The reasoning — why the sentence & generation demos exist and what they're
useful for — lives in [sentences/README.md](sentences/README.md).**
Engineering details: [sentences/engdoc.md](sentences/engdoc.md).

---

## Words — the root app

Interactive 3D word embedding explorer — GloVe vectors projected to 3D with
UMAP, rendered as a WebGL point cloud. Inspired by the [3b1b content](https://www.youtube.com/@3blue1brown) and the
[TensorFlow Embedding Projector](https://projector.tensorflow.org), with word
arithmetic on top.

- **Search & arithmetic** — one query box for words and vector math: `king - man + woman`, `paris - france + italy`, `(coffee + tea) * 0.5`. Autocomplete, typo suggestions, quoted tokens (`"1990"`, `"u.s."`).
- **Traversal path** — expressions render as a hop-by-hop path through the space, color-graded from start to result. A jumper bar lets you scrub through the intermediate stops and treat any partial sum as the destination.
- **Dataset tiers** — 500 to 400,000 words (frequency-ordered slices of one UMAP run), switchable at runtime. Default loads the 20K tier (~4 MB).
- **Shareable links** — the query lives in the URL hash (`/#q=king - man + woman`; on sentences, `/sentences/#s=first+phrasing&s=second`), so any result is a plain link that replays itself on load.
- **Tuning** — neighbor count, point size, depth fog, dimming, words-shown limit, points-vs-vectors display, labels, auto-rotate, animations.
- Nearest-neighbor search runs in a Web Worker on pre-normalized vectors; rendering and selection animations are GPU-side, so the 400K tier stays interactive.

## Run

Serve the repo root over HTTP and open it:

```bash
python -m http.server 8000
# → http://localhost:8000            (words)
# → http://localhost:8000/sentences  (sentences + generation)
```

Opening `index.html` directly from disk falls back to a small built-in demo dataset.

## Regenerating the data (words)

Requires Python 3.11–3.13 (umap-learn depends on numba).

```bash
cd precompute
pip install -r requirements.txt
# GloVe from https://nlp.stanford.edu/projects/glove/ (glove.6B.zip)
python precompute.py --glove glove.6B.50d.txt --top 400000 --out ..   # UMAP, ~1-2h at 400K
python chunk.py --src .. --out ../data                                # slice into size tiers
```

`chunk.py` writes `data/<size>/{embeddings.bin,projected.bin,manifest.json}` plus `data/index.json`. Because GloVe is frequency-ordered, every tier is a prefix of the same projection, so word positions stay consistent across sizes.

The sentence corpus has its own pipeline — see [sentences/engdoc.md](sentences/engdoc.md).

## Stack

Vanilla JS + raw WebGL (points, lines, fog, animations in shaders), a 2D canvas overlay for labels and markers, and typed-array binary assets. The sentences page adds [transformers.js](https://github.com/huggingface/transformers.js) for in-browser embedding and WebGPU LLM inference. No dependencies at runtime beyond that CDN import.

## License

Code: [MIT](LICENSE)

Word vectors: [GloVe](https://nlp.stanford.edu/projects/glove/) by Stanford NLP,
released under the Public Domain Dedication and License v1.0 (PDDL).

Sentence corpus (`/sentences`, shipped as embeddings + ≤160-char display snippets):

- arXiv abstracts — arXiv metadata, CC0
- Wikipedia lead paragraphs — [Wikimedia](https://dumps.wikimedia.org/), CC BY-SA 4.0
- ELI5 and Reddit-derived passages (r/explainlikeimfive answers; posts via
  [webis/tldr-17](https://huggingface.co/datasets/webis/tldr-17)) — content
  © their authors; used as short research excerpts

Models (downloaded at runtime from Hugging Face, cached by the browser, each
under its own license): [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) /
[all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) (Apache-2.0),
[Llama-3.2-1B-Instruct](https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct)
(Llama 3.2 Community License) — the LLM downloads only after explicit
confirmation in the UI.
