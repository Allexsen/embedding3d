// Embed the parity sentences with transformers.js (browser runtime, Node host).
// The feature-extraction pipeline with { pooling: 'mean', normalize: true }
// mirrors sentence-transformers' default. Writes parity_js.json.

import { pipeline } from '@huggingface/transformers';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const MODEL = 'Xenova/all-MiniLM-L6-v2';

const sentences = JSON.parse(readFileSync(join(HERE, 'parity_sentences.json'), 'utf8'));

const extractor = await pipeline('feature-extraction', MODEL);

const vectors = [];
for (const text of sentences) {
  const output = await extractor(text, { pooling: 'mean', normalize: true });
  vectors.push(Array.from(output.data, (x) => Math.round(x * 1e6) / 1e6));
}

writeFileSync(join(HERE, 'parity_js.json'), JSON.stringify(vectors));
console.log(`wrote parity_js.json: ${vectors.length} vectors x ${vectors[0].length} dims`);
