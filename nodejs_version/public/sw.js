const CACHE_NAME = 'autovideo-ai-v1';
const urlsToCache = [
  '/',
  '/index.html',
  '/js/main.js',
  '/manifest.json',
  'https://cdn.jsdelivr.net/npm/@shadcn/ui@latest/dist/index.min.css',
  'https://cdn.tailwindcss.com'
];

self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(function(cache) {
        return cache.addAll(urlsToCache);
      })
  );
});

self.addEventListener('fetch', function(event) {
  event.respondWith(
    caches.match(event.request)
      .then(function(response) {
        if (response) {
          return response;
        }
        return fetch(event.request);
      }
    )
  );
});
