import json, math
from sentence_transformers import SentenceTransformer
s = json.load(open('parity_sentences.json', encoding='utf-8'))
m = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
py = m.encode(s, normalize_embeddings=True, convert_to_numpy=True)
js = json.load(open('parity_mpnet_js.json'))
def cos(a,b):
    d=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(y*y for y in b))
    return d/(na*nb) if na and nb else 0
worst=min(cos(list(map(float,py[i])), js[i]) for i in range(len(js)))
print(f'mpnet parity worst cosine: {worst:.6f}  ->', 'PASS' if worst>0.999 else 'FAIL')
