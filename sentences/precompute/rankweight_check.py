import json, numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

rows = [json.loads(l) for l in open('corpus.jsonl', encoding='utf-8')][:15000]
texts=[r['text'] for r in rows]; src=[r['source'] for r in rows]
SRC=['arxiv','wikipedia','eli5','social']
pairs = {
  'dopamine': ('dopamine is basically the feel-good hit you get when something goes better than you expected',
               'dopamine mediates reward prediction error signaling in mesolimbic pathways'),
  'stocks':   ('the stock market basically went up because everyone was feeling good about the economy',
               'equity indices advanced amid improved investor sentiment and easing inflation expectations'),
  'immunity': ('vaccines basically teach your body to recognize a bug before you actually get sick',
               'immunization induces adaptive immune memory via antigen presentation'),
}
def mix(emb, q, k=15):
    sims = emb@q; top = np.argsort(sims)[::-1][:k]
    w = np.maximum(sims[top],0)**2  # rank/score weighting
    d = defaultdict(float)
    for i,wi in zip(top,w): d[src[i]] += wi
    tot = sum(d.values()) or 1
    return {s: round(100*d[s]/tot) for s in SRC}

for name, mid in [('mpnet','sentence-transformers/all-mpnet-base-v2')]:
    m = SentenceTransformer(mid)
    emb = m.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=256, show_progress_bar=False)
    print(f'=== {name} (rank-weighted top-15, 15k) ===')
    for pname,(a,b) in pairs.items():
        qs = m.encode([a,b], normalize_embeddings=True, convert_to_numpy=True)
        ma, mb = mix(emb,qs[0]), mix(emb,qs[1])
        print(f'\n{pname}:  cos={float(qs[0]@qs[1]):.2f}')
        print(f'  bro     : {ma}   formal={ma["arxiv"]+ma["wikipedia"]}')
        print(f'  academic: {mb}   formal={mb["arxiv"]+mb["wikipedia"]}')
