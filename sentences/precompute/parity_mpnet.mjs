import { pipeline } from '@huggingface/transformers';
import { readFileSync, writeFileSync } from 'node:fs';
const s = JSON.parse(readFileSync('parity_sentences.json','utf8'));
const ex = await pipeline('feature-extraction', 'Xenova/all-mpnet-base-v2');
const out = [];
for (const t of s) { const o = await ex(t,{pooling:'mean',normalize:true}); out.push(Array.from(o.data, x=>Math.round(x*1e6)/1e6)); }
writeFileSync('parity_mpnet_js.json', JSON.stringify(out));
console.log(`mpnet js: ${out.length} x ${out[0].length}`);
