// Landing page: the showcase stage (a saved gallery model with an idle orbit), its hero picker, the How it works
// band and the gallery band. Everything shown comes from the public gallery catalog (/api/gallery).
import { createPreview } from '/assets/preview.js';
import { bindOrbitDrag, track } from './analytics.js';

const SPRITE = '/assets/icons/sprite.svg';
const IDLE_MS = 4000, ORBIT_MS = 40000;   // resume after 4 s idle; one revolution takes 40 s

function icon(name, cls = 'icon icon--s') {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', cls); svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', `${SPRITE}#i-${name}`); svg.append(use);
  return svg;
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}
function image(src, alt, cls) {
  const img = el('img', cls); img.alt = alt; img.loading = 'lazy'; img.decoding = 'async'; img.src = src;
  return img;
}
// Gallery items may carry small WebP thumbnails; the full-size PNG is the fallback.
export function thumbUrl(item, kind) {
  const asset = item?.assets?.[kind];
  return asset?.thumb_url || asset?.url || `/gallery-assets/${encodeURIComponent(item.id)}/${kind}`;
}
export function minutes(item) {
  const s = item?.refinement?.seconds;
  return Number.isFinite(s) && s > 0 ? Math.max(1, Math.round(s / 60)) : null;
}

// Trims a transparent model render to the figure plus a margin, so every card shows the hero at the same scale.
async function crop(url, margin = 0.06) {
  const img = new Image(); img.src = url; await img.decode();
  const w = img.naturalWidth, h = img.naturalHeight, c = document.createElement('canvas'); c.width = w; c.height = h;
  const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(img, 0, 0);
  const data = g.getImageData(0, 0, w, h).data;
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (data[(y * w + x) * 4 + 3] > 8) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  }
  if (x1 < 0) return url;
  const m = Math.round(Math.max(x1 - x0, y1 - y0) * margin);
  x0 = Math.max(0, x0 - m); y0 = Math.max(0, y0 - m); x1 = Math.min(w - 1, x1 + m); y1 = Math.min(h - 1, y1 + m);
  const out = document.createElement('canvas'); out.width = x1 - x0 + 1; out.height = y1 - y0 + 1;
  out.getContext('2d').drawImage(c, x0, y0, out.width, out.height, 0, 0, out.width, out.height);
  return out.toDataURL('image/png');
}
// Frames the showcase a little closer and higher than the default, leaving room for the dock under the feet.
function frameShowcase(viewer, canvas) {
  const v = viewer.view(); if (!v) return;
  // Portrait stages (phones) already frame the full figure tightly, so they only lift it clear of the dock.
  const portrait = canvas.clientWidth < canvas.clientHeight * 0.9;
  const [px, py, pz] = v.position, [tx, ty, tz] = v.target, k = portrait ? 1.08 : 0.92;
  const dist = Math.hypot(px - tx, py - ty, pz - tz), drop = dist * (portrait ? -0.025 : 0.05);
  viewer.setView({ ...v, position: [tx + (px - tx) * k, ty + (py - ty) * k - drop, tz + (pz - tz) * k], target: [tx, ty - drop, tz] });
}

// Fades sections in as they scroll into view. Without IntersectionObserver, or with reduced motion, all show at once.
function revealer(root) {
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduce || typeof IntersectionObserver === 'undefined') return { watch() {} };
  root.classList.add('landing--reveal');
  const io = new IntersectionObserver(entries => {
    for (const entry of entries) if (entry.isIntersecting) { entry.target.classList.add('is-in'); io.unobserve(entry.target); }
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });
  const watch = scope => scope.querySelectorAll('[data-reveal]:not(.is-in)').forEach(node => io.observe(node));
  watch(root);
  return { watch };
}

export function landingGallery() {
  const $ = s => document.querySelector(s), viewer = createPreview({ floor: false });
  let items = [], selected = null, revision = 0, visible = true, destroyed = false, ready = false;
  let thumbViewer = null, thumbCanvas = null, thumbBusy = false;
  const shots = new Map();   // id -> {textured, clay} data URLs, rendered once from the saved model
  const status = $('#landingStatus'), canvas = $('#landingCanvas'), retry = $('#retryLanding'), picker = $('#modelPicker');
  const reveal = revealer($('#entry'));
  function closeThumbs() { thumbViewer?.dispose(); thumbViewer = null; thumbCanvas?.remove(); thumbCanvas = null; }
  function renderable() { return visible && !document.hidden && !destroyed; }

  // ---- idle orbit ------------------------------------------------------------------------------------------
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let idleUntil = performance.now() + IDLE_MS, onscreen = true, frame = 0, last = 0;
  function orbit(now) {
    frame = requestAnimationFrame(orbit);
    const dt = Math.min(64, now - (last || now)); last = now;
    if (reduceMotion.matches || !ready || !onscreen || !renderable() || now < idleUntil) return;
    const v = viewer.view(); if (!v) return;
    const a = 2 * Math.PI * dt / ORBIT_MS, c = Math.cos(a), s = Math.sin(a);
    const [px, py, pz] = v.position, [tx, , tz] = v.target, x = px - tx, z = pz - tz;
    viewer.setView({ ...v, position: [tx + x * c - z * s, py, tz + x * s + z * c] });
  }
  function startOrbit() { if (!frame) { last = 0; frame = requestAnimationFrame(orbit); } }
  function stopOrbit() { cancelAnimationFrame(frame); frame = 0; }
  const hold = () => { idleUntil = performance.now() + IDLE_MS; };
  for (const type of ['pointerdown', 'pointermove', 'wheel', 'keydown', 'touchstart']) canvas.addEventListener(type, event => {
    if (type !== 'pointermove' || event.buttons) hold();
  }, { passive: true });
  if (typeof IntersectionObserver !== 'undefined') new IntersectionObserver(([entry]) => {
    onscreen = entry.isIntersecting; if (onscreen) startOrbit(); else stopOrbit();
  }).observe(canvas);
  startOrbit();

  // ---- showcase --------------------------------------------------------------------------------------------
  async function select(item) {
    const switching = selected?.id !== item.id;
    selected = item; const current = ++revision; ready = false;
    $('#featuredTitle').textContent = item.title;
    $('#featuredDetails').href = `/gallery/${encodeURIComponent(item.id)}`; $('#featuredDetails').hidden = false;
    for (const button of picker.children) button.setAttribute('aria-pressed', String(button.dataset.id === item.id));
    if (!renderable()) return;
    status.textContent = 'Loading the saved 3D model…'; retry.hidden = true;
    canvas.setAttribute('aria-busy', 'true'); if (switching) canvas.classList.remove('is-shown');
    try {
      await viewer.show(canvas, item.assets.model.url);
      if (current !== revision || !renderable()) return;
      frameShowcase(viewer, canvas); ready = true; hold(); canvas.classList.add('is-shown');
      canvas.setAttribute('aria-label', `${item.title}, a saved 3D character. Drag to rotate or scroll to zoom.`);
      status.textContent = 'Drag to rotate · Scroll to zoom';
      renders();
    } catch {
      if (current !== revision || !renderable()) return;
      status.textContent = '3D could not be loaded. Retry or inspect the saved run.'; retry.hidden = false;
    } finally { if (current === revision) canvas.setAttribute('aria-busy', 'false'); }
  }

  // ---- model renders for How it works and the gallery cards ------------------------------------------------
  function paint(item) {
    const shot = shots.get(item.id);
    for (const img of document.querySelectorAll(`[data-snapshot="${CSS.escape(item.id)}"]`)) {
      const card = img.closest('.g-card');
      if (shot?.textured) { img.src = shot.textured; card?.classList.add('has-model'); }
      else card?.classList.add('no-model');
    }
    if (item !== items[0]) return;
    for (const kind of ['clay', 'textured']) {
      const box = document.querySelector(`.step__art[data-step="${kind}"]`); if (!box) continue;
      box.classList.remove('is-loading');
      if (shot?.[kind]) {
        box.replaceChildren(image(shot[kind], kind === 'clay' ? `${item.title} as an untextured 3D shape` : `${item.title}, the finished textured 3D model`, 'reveal'));
      } else box.replaceChildren(icon(kind === 'clay' ? 'box' : 'person-standing', 'icon icon--l step__placeholder'));
    }
  }
  async function renders() {
    if (thumbBusy || destroyed) return;
    const pending = items.filter(item => !shots.has(item.id));
    if (!pending.length) return;
    thumbBusy = true;
    thumbViewer = createPreview({ damping: false, floor: false }); thumbCanvas = document.createElement('canvas');
    thumbCanvas.className = 'thumbnail-render'; thumbCanvas.setAttribute('aria-hidden', 'true');
    document.body.append(thumbCanvas);
    // Reuse one temporary renderer, then release its context and all model data.
    try {
      for (const item of pending) {
        if (destroyed) break;
        try {
          await thumbViewer.show(thumbCanvas, item.assets.model.url);
          if (destroyed) break;
          thumbViewer.shading('textured');
          const shot = { textured: thumbViewer.snapshot(), clay: null };
          if (item === items[0]) { thumbViewer.shading('clay'); shot.clay = thumbViewer.snapshot(); thumbViewer.shading('textured'); }
          shot.textured = await crop(shot.textured);
          if (shot.clay) shot.clay = await crop(shot.clay);
          shots.set(item.id, shot);
        } catch { shots.set(item.id, {}); }   // the card keeps its artwork; the stage still shows the model
        paint(item);
      }
    } finally { closeThumbs(); thumbBusy = false; }
  }

  // ---- catalog -> picker, How it works, gallery band -------------------------------------------------------
  function pickerButton(item) {
    const button = el('button', 'thumb showcase__pick'); button.type = 'button'; button.dataset.id = item.id;
    button.setAttribute('aria-pressed', 'false'); button.setAttribute('aria-label', `View ${item.title} in 3D`);
    const frame = el('span', 'thumb__img'); frame.append(image(thumbUrl(item, 'artwork'), ''));
    button.append(frame, el('span', 'thumb__label', item.title));
    button.addEventListener('click', () => { track('picker_select', { id: item.id }); select(item); });
    return button;
  }
  function journey(item) {
    const sketch = document.querySelector('.step__art[data-step="sketch"]'), art = document.querySelector('.step__art[data-step="artwork"]');
    sketch.replaceChildren(image(thumbUrl(item, 'sketch'), `Pencil sketch of ${item.title}`));
    art.replaceChildren(image(thumbUrl(item, 'artwork'), `Coloured artwork of ${item.title}`));
    for (const kind of ['clay', 'textured']) {
      const box = document.querySelector(`.step__art[data-step="${kind}"]`);
      if (!shots.has(item.id)) { box.classList.add('is-loading'); box.replaceChildren(el('span', 'step__skeleton skeleton')); }
    }
    const credit = $('#journeyCredit'); credit.textContent = `Shown: ${item.title}, from the public gallery.`; credit.hidden = false;
  }
  function card(item) {
    const link = el('a', 'card card--interactive g-card'); link.href = `/gallery/${encodeURIComponent(item.id)}`;
    link.dataset.reveal = '';
    const stage = el('div', 'viewer g-card__stage');
    const model = el('img', 'g-card__model'); model.alt = ''; model.dataset.snapshot = item.id;
    const art = image(thumbUrl(item, 'artwork'), '', 'g-card__art');
    stage.append(el('span', 'g-card__skeleton skeleton'), model, art);
    const body = el('div', 'g-card__body');
    body.append(el('h3', '', item.title));
    if (item.description) body.append(el('p', 'g-card__desc', item.description));
    const meta = el('p', 'g-card__meta'), mins = minutes(item);
    if (mins) { const span = el('span'); span.append(icon('timer'), document.createTextNode(`Built in about ${mins} min`)); meta.append(span); }
    const open = el('span', 'g-card__open'); open.append(document.createTextNode('Open'), icon('arrow-right')); meta.append(open);
    body.append(meta); link.append(stage, body);
    link.addEventListener('click', () => track('inspect_saved_run', { id: item.id }, { flush: true }));
    return link;
  }
  async function catalog() {
    retry.hidden = true; status.textContent = 'Loading the public gallery…';
    try {
      const response = await fetch('/api/gallery', {credentials: 'omit', cache: 'no-store'});
      if (!response.ok) throw new Error('unavailable');
      const data = await response.json(); if (!Array.isArray(data.items)) throw new Error('unavailable');
      if (destroyed) return;
      // Prefer the published ranger; preserve catalog order for other results.
      items = [...data.items].sort((a, b) => Number(b.id === 'woodland-scout') - Number(a.id === 'woodland-scout'));
      picker.replaceChildren(...items.map(pickerButton));
      $('#galleryGrid').replaceChildren(...items.map(card));
      $('#gallery').hidden = !items.length;
      reveal.watch(document);
      if (!items.length) { status.textContent = 'No saved models have been published yet.'; $('#howItWorks').classList.add('is-empty'); return; }
      journey(items[0]);
      await select(items[0]); if (!destroyed && ready) renders();
    } catch { status.textContent = 'The gallery is unavailable. Retry to load the saved models.'; retry.hidden = false; }
  }
  bindOrbitDrag(canvas, 'landing');
  retry.addEventListener('click', () => selected ? select(selected) : catalog());
  $('#featuredDetails').addEventListener('click', () => track('inspect_saved_run', selected?.id ? { id: selected.id } : null, { flush: true }));
  document.querySelectorAll('[data-shade]').forEach(button => button.addEventListener('click', () => {
    viewer.shading(button.dataset.shade);
    document.querySelectorAll('[data-shade]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    track('shade_toggle', { mode: button.dataset.shade });
  }));
  function visibility() { if (renderable() && selected) select(selected); else { ++revision; ready = false; viewer.pause(); } }
  document.addEventListener('visibilitychange', visibility);
  window.addEventListener('pagehide', () => { destroyed = true; ++revision; stopOrbit(); viewer.dispose(); closeThumbs(); });
  window.addEventListener('pageshow', event => { if (event.persisted) { destroyed = false; startOrbit(); visibility(); } });
  catalog();
  return {setVisible(value) { visible = value; visibility(); }};
}
