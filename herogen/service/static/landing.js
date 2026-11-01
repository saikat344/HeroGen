import { createPreview } from '/assets/preview.js';
import { bindOrbitDrag, track } from './analytics.js';

export function landingGallery() {
  const $ = s => document.querySelector(s), viewer = createPreview();
  let items = [], selected = null, revision = 0, visible = true, destroyed = false, thumbViewer = null, thumbCanvas = null, thumbBusy = false;
  const status = $('#landingStatus'), canvas = $('#landingCanvas'), retry = $('#retryLanding');
  function closeThumbs() { thumbViewer?.dispose(); thumbViewer = null; thumbCanvas?.remove(); thumbCanvas = null; }
  function renderable() { return visible && !document.hidden && !destroyed; }
  async function select(item) {
    selected = item; const current = ++revision;
    $('#featuredTitle').textContent = item.title;
    $('#featuredArtwork').src = item.assets.artwork.url;
    $('#featuredArtwork').alt = `Original character artwork for ${item.title}`;
    $('#featuredArtwork').hidden = false;
    $('#featuredDetails').href = `/gallery/${encodeURIComponent(item.id)}`; $('#featuredDetails').hidden = false;
    for (const button of $('#modelPicker').children) button.setAttribute('aria-pressed', String(button.dataset.id === item.id));
    if (!renderable()) return;
    status.textContent = 'Loading the saved 3D model…'; retry.hidden = true;
    canvas.setAttribute('aria-busy', 'true');
    try {
      await viewer.show(canvas, item.assets.model.url);
      if (current !== revision || !renderable()) return;
      status.textContent = 'Drag to rotate · Scroll to zoom';
      thumbnails();
    } catch {
      if (current !== revision || !renderable()) return;
      status.textContent = '3D could not be loaded. Retry or inspect the saved run.'; retry.hidden = false;
    } finally { if (current === revision) canvas.setAttribute('aria-busy', 'false'); }
  }
  async function thumbnails() {
    if (thumbBusy || destroyed) return;
    const pending = items.filter(item => ![...$('#modelPicker').children].find(b => b.dataset.id === item.id)?.querySelector('img'));
    if (!pending.length) return;
    thumbBusy = true;
    thumbViewer = createPreview(); thumbCanvas = document.createElement('canvas');
    thumbCanvas.className = 'thumbnail-render'; thumbCanvas.setAttribute('aria-hidden', 'true');
    document.body.append(thumbCanvas);
    // Reuse one temporary renderer, then release its context and all model data.
    try {
      for (const item of pending) {
        if (destroyed) break;
        try {
          await thumbViewer.show(thumbCanvas, item.assets.model.url);
          if (destroyed) break;
          const url = thumbViewer.snapshot();
          const button = [...$('#modelPicker').children].find(b => b.dataset.id === item.id);
          const image = document.createElement('img'); image.src = url; image.alt = ''; image.width = 64; image.height = 74;
          button?.querySelector('.model-thumb').replaceChildren(image);
        } catch { /* The named picker and primary viewer remain available. */ }
      }
    } finally { closeThumbs(); thumbBusy = false; }
  }
  async function catalog() {
    retry.hidden = true; status.textContent = 'Loading the public gallery…';
    try {
      const response = await fetch('/api/gallery', {credentials: 'omit', cache: 'no-store'});
      if (!response.ok) throw new Error('unavailable');
      const data = await response.json(); if (!Array.isArray(data.items)) throw new Error('unavailable');
      if (destroyed) return;
      items = [...data.items].sort((a, b) => Number(b.id === 'woodland-scout') - Number(a.id === 'woodland-scout')); $('#modelPicker').replaceChildren();
      for (const item of items) {
        const button = document.createElement('button'); button.type = 'button'; button.dataset.id = item.id;
        button.setAttribute('aria-pressed', 'false'); button.setAttribute('aria-label', `View ${item.title} in 3D`);
        const thumb = document.createElement('span'); thumb.className = 'model-thumb'; thumb.textContent = '3D'; thumb.setAttribute('aria-hidden', 'true');
        const label = document.createElement('span'); label.textContent = item.title; button.append(thumb, label);
        button.addEventListener('click', () => { track('picker_select', { id: item.id }); select(item); }); $('#modelPicker').append(button);
      }
      if (!items.length) { status.textContent = 'No saved models have been published yet.'; return; }
      // Prefer the published ranger; preserve catalog order for other results.
      await select(items[0]); if (!destroyed) thumbnails();
    } catch { status.textContent = 'The gallery is unavailable. Retry to load the saved models.'; retry.hidden = false; }
  }
  bindOrbitDrag(canvas, 'landing');
  retry.addEventListener('click', () => selected ? select(selected) : catalog());
  $('#featuredArtwork').addEventListener('error', () => { $('#featuredArtwork').hidden = true; });
  $('#featuredDetails').addEventListener('click', () => track('inspect_saved_run', selected?.id ? { id: selected.id } : null, { flush: true }));
  document.querySelectorAll('[data-shade]').forEach(button => button.addEventListener('click', () => {
    viewer.shading(button.dataset.shade);
    document.querySelectorAll('[data-shade]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    track('shade_toggle', { mode: button.dataset.shade });
  }));
  function visibility() { if (renderable() && selected) select(selected); else { ++revision; viewer.pause(); } }
  document.addEventListener('visibilitychange', visibility);
  window.addEventListener('pagehide', () => { destroyed = true; ++revision; viewer.dispose(); closeThumbs(); });
  window.addEventListener('pageshow', event => { if (event.persisted) { destroyed = false; visibility(); } });
  catalog();
  return {setVisible(value) { visible = value; visibility(); }};
}
