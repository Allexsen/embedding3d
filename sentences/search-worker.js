// Nearest-neighbor search over int8/int16/f32 embeddings.
// Dequantizes to cosine via per-vector scales; integer inner loop for int8.

let codes = null;      // Int8Array | Int16Array | Float32Array
let scales = null;     // Float32Array | null (f32 has none)
let count = 0;
let dim = 0;
let isFloat = false;

onmessage = (event) => {
  const msg = event.data;

  if (msg.type === 'init') {
    dim = msg.dim;
    count = msg.count;
    isFloat = msg.precision === 'f32';
    codes = isFloat
      ? new Float32Array(msg.codes)
      : msg.precision === 'int16'
        ? new Int16Array(msg.codes)
        : new Int8Array(msg.codes);
    scales = msg.scales ? new Float32Array(msg.scales) : null;
    postMessage({ type: 'ready' });
    return;
  }

  if (msg.type === 'query') {
    // query is a normalized f32 vector; ranking by dot product == cosine
    const q = new Float32Array(msg.vector);
    const topN = msg.topN;
    const exclude = new Set(msg.exclude || []);
    const limit = Math.min(count, msg.limit || count);

    const idx = new Int32Array(topN).fill(-1);
    const sc = new Float32Array(topN).fill(-2);

    for (let i = 0; i < limit; i++) {
      if (exclude.has(i)) continue;
      const off = i * dim;
      let dot = 0;
      if (isFloat) {
        for (let j = 0; j < dim; j++) dot += codes[off + j] * q[j];
      } else {
        let acc = 0;
        for (let j = 0; j < dim; j++) acc += codes[off + j] * q[j];
        dot = acc * scales[i];
      }
      if (dot > sc[topN - 1]) {
        let p = topN - 1;
        while (p > 0 && sc[p - 1] < dot) {
          sc[p] = sc[p - 1];
          idx[p] = idx[p - 1];
          p--;
        }
        sc[p] = dot;
        idx[p] = i;
      }
    }

    const neighbors = [];
    for (let i = 0; i < topN; i++) {
      if (idx[i] >= 0) neighbors.push({ index: idx[i], score: sc[i] });
    }
    postMessage({ type: 'result', id: msg.id, neighbors });
  }
};
