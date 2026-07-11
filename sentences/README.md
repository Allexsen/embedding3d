# Sentences & Generation — why this page exists

## The one-sentence version

**How you word something decides what a machine finds for you — and what it
says back.** This page makes that visible.

## The longer version

Semantic search, RAG pipelines, recommendation feeds, AI assistants that
retrieve documents before answering — under almost all of them sits the same
mechanism: text becomes a point in a high-dimensional space (an *embedding*),
and "relevant" means "nearby". The comforting assumption is that embeddings
capture *meaning* — that two sentences saying the same thing land in the same
place, and everything else about them washes out.

They don't, and it doesn't. Embeddings capture meaning **and** everything that
rides along with it: tone, register, vocabulary, the kind of person who tends
to write that way. Ask about dopamine like a redditor and you land among
Reddit posts; ask like a neuroscientist and you land among paper abstracts.
Same question, different neighborhood, different results. The system didn't
misunderstand either phrasing — it understood both perfectly, *including the
part you didn't mean to say*: what kind of text you sounded like.

That's an invisible property of every embedding-based system you use. This
page exists to make it visible, measurable, and hard to unsee.

## What the demo shows

- **Compare** — type one idea in two phrasings; each lights up its own
  neighborhood in a corpus of ~100K passages drawn from four registers
  (arXiv, Wikipedia, ELI5, social media). The source-mix bars are the
  receipts: the casual phrasing pulls social/ELI5 passages, the academic one
  pulls arXiv — for the *same proposition*.
- **Morph** — slide between the two phrasings and watch the neighborhood
  migrate mid-flight: meaning held constant, register drifting, retrieval
  following the register.
- **✦ Generate** — the same effect one step downstream. Each phrasing is put
  to an actual language model several times; every sampled answer is embedded
  and placed in the same space as a ✦ star. If wording didn't matter, the two
  answer clouds would overlap. They don't: the model answers in the register
  it was asked in — different content, framing, vocabulary — and the
  within/between similarity card puts a number on it.

## Why that's useful to know

- **If you build with RAG or semantic search** — your users' phrasing
  silently selects which slice of your corpus they can reach. Two users with
  the same question get different context, therefore different answers.
  Query rewriting, register normalization, and corpus curation stop looking
  like nice-to-haves once you've watched this happen.
- **If you prompt LLMs** — the register of your question measurably shapes
  the *distribution* of answers you get back, not just their politeness.
  Wording is a control surface, not decoration.
- **If you teach or evaluate this stuff** — it's a concrete, interactive
  counter to two naive positions at once: "embeddings capture pure meaning"
  and "phrasing doesn't matter, the model knows what I mean."

## Honest footnote

The 3D cloud is a UMAP projection of 384/768-dimensional space down to three —
suggestive, not proof. The proof is the numbers, computed in the full space:
cosine similarities, neighbor overlap, source mix, within/between answer
similarity. The picture exists to make the numbers feel like what they mean.

Everything runs client-side: the corpus and models download into your browser
(sizes disclosed in-app, the LLM only after explicit confirmation), and the
optional API mode calls your own provider with your own key. There is no
server behind this page to send anything to.
