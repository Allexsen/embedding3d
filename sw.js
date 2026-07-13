// Cache-first for the big binary assets (word tiers, sentence corpus,
// embeddings, projections). GitHub Pages serves everything with max-age=600,
// so without this every visit past 10 minutes re-downloads tens of MB.
// Models are not handled here — transformers.js keeps those in its own
// Cache API store. HTML/JS/CSS/JSON are also untouched: they go to the
// network as before, so deploys can never be masked by a stale app shell.
// Bump the version when regenerated data should replace what visitors hold.
const DATA_CACHE = 'e3d-bin-v1';

self.addEventListener('install', () => self.skipWaiting());

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) {
      if (key.startsWith('e3d-bin-') && key !== DATA_CACHE) await caches.delete(key);
    }
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || url.origin !== self.location.origin || !url.pathname.endsWith('.bin')) return;
  event.respondWith((async () => {
    const cache = await caches.open(DATA_CACHE);
    const hit = await cache.match(event.request);
    if (hit) return hit;
    const resp = await fetch(event.request);
    if (resp.ok) cache.put(event.request, resp.clone());
    return resp;
  })());
});
