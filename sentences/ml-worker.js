// ML worker: transformers.js runs in here — model downloads, tokenization,
// embedding, and local WebGPU sampling — so the main thread only ever paints
// frames and handles input. The page talks to it via small RPC messages;
// vectors travel back as transferred ArrayBuffers.
// 3.8.1: needed for clean WebGPU text-generation (3.0.x degenerates); embedder
// output is unchanged across the bump (same ONNX weights + pooling).
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

// transformers.js: allow remote model download, cache in browser
env.allowLocalModels = false;

const extractors = new Map();       // browserId -> extractor
const extractorLoading = new Map(); // browserId -> promise
let generator = null;
let generatorLoading = null;

function getExtractor(browserId) {
  if (extractors.has(browserId)) return Promise.resolve(extractors.get(browserId));
  if (extractorLoading.has(browserId)) return extractorLoading.get(browserId);
  const p = pipeline('feature-extraction', browserId, {
    progress_callback: (pr) => {
      if (pr.status === 'progress' && pr.total) {
        self.postMessage({ type: 'progress', what: 'embedder', model: browserId, loaded: pr.loaded, total: pr.total });
      }
    },
  }).then((ex) => {
    extractors.set(browserId, ex);
    self.postMessage({ type: 'modelReady', what: 'embedder', model: browserId });
    return ex;
  }).catch((err) => { extractorLoading.delete(browserId); throw err; });
  extractorLoading.set(browserId, p);
  return p;
}

function getGenerator(modelId, dtype) {
  if (generator) return Promise.resolve(generator);
  if (generatorLoading) return generatorLoading;
  generatorLoading = pipeline('text-generation', modelId, {
    dtype,
    device: 'webgpu',
    progress_callback: (pr) => {
      if (pr.status === 'progress' && pr.total) {
        self.postMessage({ type: 'progress', what: 'generator', model: modelId, loaded: pr.loaded, total: pr.total });
      }
    },
  }).then((g) => {
    generator = g;
    self.postMessage({ type: 'modelReady', what: 'generator', model: modelId });
    return g;
  }).catch((err) => { generatorLoading = null; throw err; });
  return generatorLoading;
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'embed') {
      const ex = await getExtractor(m.browserId);
      const out = await ex(m.text, { pooling: 'mean', normalize: true });
      const vec = new Float32Array(out.data);
      self.postMessage({ type: 'result', id: m.id, vec: vec.buffer }, [vec.buffer]);
    } else if (m.type === 'warmEmbedder') {
      await getExtractor(m.browserId);
      self.postMessage({ type: 'result', id: m.id });
    } else if (m.type === 'warmGenerator') {
      await getGenerator(m.modelId, m.dtype);
      self.postMessage({ type: 'result', id: m.id });
    } else if (m.type === 'generate') {
      const gen = await getGenerator(m.modelId, m.dtype);
      const out = await gen(m.messages, m.options);
      const g = out[0].generated_text;
      const text = typeof g === 'string' ? g : (Array.isArray(g) ? g[g.length - 1].content : String(g));
      self.postMessage({ type: 'result', id: m.id, text: text.trim() });
    }
  } catch (err) {
    self.postMessage({ type: 'result', id: m.id, error: String((err && err.message) || err) });
  }
};

// worker-side WebGPU capability — the page gates the local-generation UI on
// this, since the sampling actually happens in here, not on the page
self.postMessage({ type: 'boot', webgpu: typeof navigator !== 'undefined' && 'gpu' in navigator });
