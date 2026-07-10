import json, numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

rows = [json.loads(l) for l in open('corpus.jsonl', encoding='utf-8')]
# balanced 4k sample (round-robin already), take first 4000
rows = rows[:4000]
texts = [r['text'] for r in rows]; src = [r['source'] for r in rows]
SRC = ['arxiv','wikipedia','eli5','social']

pair = {
  'A bro': 'dopamine is basically the feel-good hit you get when something goes better than you expected',
  'B academic': 'dopamine mediates reward prediction error signaling in mesolimbic pathways',
}

for name, mid in [('MiniLM','sentence-transformers/all-MiniLM-L6-v2'), ('mpnet','sentence-transformers/all-mpnet-base-v2')]:
    m = SentenceTransformer(mid)
    emb = m.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=256, show_progress_bar=False)
    print(f'\n=== {name} ===')
    qs = m.encode(list(pair.values()), normalize_embeddings=True, convert_to_numpy=True)
    print('  query cosine A-B: %.3f' % float(qs[0]@qs[1]))
    for label, q in zip(pair, qs):
        top = np.argsort(emb@q)[::-1][:30]
        mix = Counter(src[i] for i in top)
        formal = mix.get('arxiv',0)+mix.get('wikipedia',0)
        print(f'  {label:>12}: ' + '  '.join(f'{s}={mix.get(s,0)}' for s in SRC) + f'   formal={formal}')
